"""
Extract voice style JSON from a WAV file for Supertonic TTS (v3).

This script mirrors the upstream `optimize_style.py` from the
`kdrkdrkdr/supertonic.Embed` repository but contains adjustments to
accommodate Supertonic‑3 models:

  * When converting the ONNX TTS models to PyTorch via
    `onnx2torch`, the ONNX opset version is upgraded to at least 18.
    Supertonic‑3 distributes its models with opset 18; leaving the
    version at 17 (as in the original script) would result in
    shape inference and conversion errors【387035667251124†L70-L81】.
  * All other logic, including the layout of style vectors (TTL and
    DP), WavLM feature matching and gradient‑based optimisation, is
    preserved unchanged.  The style tensor dimensions remain
    [1, 50, 256] for `style_ttl` and [1, 8, 16] for `style_dp` as these
    are still valid for Supertonic‑3【423589999849011†L48-L74】【423589999849011†L274-L300】.

Usage:
    python optimize_style.py <config_name>

Refer to the original upstream documentation for details on
configuration JSON formats and examples.  Ensure that the ONNX models
from Supertonic‑3 are placed in the `onnx/` subdirectory alongside
`tts.json` and `unicode_indexer.json`.
"""

import json
import os
import sys
import glob
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
import onnxslim
import onnx
from onnx import shape_inference
import onnx2torch
from onnx2torch import convert

from helper import load_text_to_speech, load_voice_style

# SSL certificate workaround: some environments may have invalid CA bundles.
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
import httpx  # noqa: E402

# Disable SSL verification globally for httpx.  This mirrors the upstream
# behaviour; see the original script for rationale.  It is safe
# here because we only download models from trusted endpoints.
_orig_client = httpx.Client
class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== ONNX to PyTorch conversion =====

def _patch_onnx2torch() -> None:
    """
    Bypass onnx2torch's safe_shape_inference which writes temporary files.
    We inline shape inference here to avoid file system side effects.
    """

    def patched(m):
        if isinstance(m, str):
            m = onnx.load(m)
        try:
            return shape_inference.infer_shapes(m)
        except Exception:
            return m
    onnx2torch.converter.safe_shape_inference = patched


def _fix_clip(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove empty Clip inputs that cause onnx2torch conversion errors."""
    for node in model.graph.node:
        if node.op_type == 'Clip':
            inputs: List[str] = list(node.input)
            while inputs and inputs[-1] == '':
                inputs.pop()
            del node.input[:]
            node.input.extend(inputs)
    return model


def load_pt_model(name: str, onnx_dir: str = "onnx") -> torch.nn.Module:
    """
    Load an ONNX model, slim it, fix the opset version and convert
    it into a PyTorch module.  Supertonic‑3 models require opset >= 18;
    to remain compatible with older models we upgrade any version below
    18 to 18.
    """
    # Slim unused weights to reduce memory footprint
    slimmed = onnxslim.slim(os.path.join(onnx_dir, name))
    # Upgrade opset version for the main domain (ai.onnx) to 18 if
    # necessary【387035667251124†L70-L81】.
    for opset in slimmed.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            if opset.version < 18:
                opset.version = 18
    # Remove empty Clip inputs for older exported models
    _fix_clip(slimmed)
    # Perform conversion using patched shape inference
    _patch_onnx2torch()
    model = convert(slimmed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(DEVICE)


# ===== WavLM perceptual loss =====

def load_wavlm() -> torch.nn.Module:
    """
    Load WavLM‑Large from the HuggingFace Transformers library.  According
    to Chiu et al. (2025), layer 3 best encodes speaker identity.  The
    model is moved to the appropriate device and set to eval mode.  All
    parameters have gradients disabled.
    """
    from transformers import WavLMModel  # type: ignore
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def wavlm_feature_loss(
    wavlm: torch.nn.Module,
    gen_wav: torch.Tensor,
    target_features: Tuple[torch.Tensor, torch.Tensor],
    layer: int = 3,
) -> torch.Tensor:
    """
    Compute perceptual loss between generated and target audio using
    statistics of WavLM hidden states.  The time‑averaged mean and
    standard deviation capture speaker identity independent of content.
    """
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)

    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features

    gen_mean = gen_feat.mean(dim=1)
    gen_std = gen_feat.std(dim=1)

    return F.mse_loss(gen_mean, tgt_mean) + F.mse_loss(gen_std, tgt_std)


def extract_wavlm_targets(
    wavlm: torch.nn.Module, target_wav: torch.Tensor, layer: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre‑compute WavLM hidden state statistics for a target audio tensor.
    Returns the mean and standard deviation of the selected layer.
    """
    if target_wav.ndim == 1:
        target_wav = target_wav.unsqueeze(0)
    wav_16k = torchaudio.functional.resample(target_wav, 44100, 16000)
    with torch.no_grad():
        out = wavlm(wav_16k, output_hidden_states=True)
    feat = out.hidden_states[layer]
    mean = feat.mean(dim=1)
    std = feat.std(dim=1)
    return (mean, std)


# ===== Differentiable TTS forward pass =====

def tts_forward(
    text_ids: torch.Tensor,
    text_mask: torch.Tensor,
    style_ttl: torch.Tensor,
    style_dp: torch.Tensor,
    dp_model: torch.nn.Module,
    te_model: torch.nn.Module,
    ve_model: torch.nn.Module,
    voc_model: torch.nn.Module,
    total_step: int,
    speed: float,
    noisy_latent: torch.Tensor,
    latent_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a forward pass through the differentiable TTS pipeline.  This
    consists of running the duration predictor, text encoder, vector
    estimator (iteratively) and vocoder to produce a waveform given a
    pair of style tensors and a sampled latent noise tensor.
    """
    dur = dp_model(text_ids, style_dp, text_mask)
    dur = dur / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur


# ===== Style saving =====

def save_style(
    path: str,
    style_ttl: torch.Tensor,
    style_dp: torch.Tensor,
    source_file: Optional[str] = None,
) -> None:
    """
    Persist style vectors to disk in SupertonicTTS‑compatible JSON format.
    The shapes [1, 50, 256] and [1, 8, 16] are hard‑coded here because
    Supertonic‑3 retains the same style dimensions as previous versions【423589999849011†L48-L74】【423589999849011†L274-L300】.
    """
    style_json = {
        "style_ttl": {
            "data": style_ttl.cpu().numpy().tolist(),
            "dims": [1, 50, 256],
            "type": "float32",
        },
        "style_dp": {
            "data": style_dp.cpu().numpy().tolist(),
            "dims": [1, 8, 16],
            "type": "float32",
        },
        "metadata": {
            "source_file": source_file or "unknown",
            "source_sample_rate": 44100,
            "target_sample_rate": 44100,
            "extracted_at": datetime.now().isoformat(),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(style_json, f)


# ===== Main optimisation loop =====

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python optimize_style.py <config_name>")
        sys.exit(1)
    config_name = sys.argv[1]
    config_path = os.path.join("configs", f"{config_name}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load reference WAV and configuration parameters
    target_wav_path: str = config["target_wav"]
    total_step: int = config.get("total_step", 5)
    speed: float = config.get("speed", 1.05)
    lr: float = config.get("lr", 1e-2)
    num_steps: int = config.get("num_steps", 500)
    threshold: float = config.get("threshold", 0.24)
    save_every: int = config.get("save_every", 100)
    start_step: int = config.get("start_step", 0)
    log_dir: str = config.get("log_dir", "logs")
    reference_style: Optional[str] = config.get("reference_style")

    # Ensure logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    print("=== Loading models ===")
    # Load ONNX models via helper; this uses the CPU provider by default
    tts = load_text_to_speech("onnx")
    # Convert ONNX models to differentiable PyTorch modules
    dp_model = load_pt_model("duration_predictor.onnx")
    te_model = load_pt_model("text_encoder.onnx")
    ve_model = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")

    # Load WavLM for perceptual loss
    wavlm = load_wavlm()

    # Read target WAV file at its native sample rate and resample to 44.1 kHz
    wav, sr = sf.read(target_wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mix down to mono
    # Use torchaudio to resample if necessary
    if sr != 44100:
        wav = librosa.resample(wav.astype(np.float32), sr, 44100)
    target_wav = torch.tensor(wav, dtype=torch.float32).to(DEVICE)

    # Pre‑compute target WavLM features
    target_feats = extract_wavlm_targets(wavlm, target_wav, layer=3)

    # Prepare text inputs for gradient descent; use default prompts if not specified
    texts: List[str] = config.get("texts", ["Testing style extraction."])
    langs: List[str] = config.get("langs", ["en"] * len(texts))
    if len(langs) != len(texts):
        raise ValueError("Length of 'langs' must match length of 'texts'")

    text_inputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for text, lang in zip(texts, langs):
        # Use ONNX TTS text processor to get IDs and masks
        ids, mask = tts.text_processor(text, lang)
        text_inputs.append((torch.tensor(ids, dtype=torch.long).to(DEVICE), torch.tensor(mask, dtype=torch.float32).to(DEVICE)))

    # Sample a fixed latent noise tensor for reproducible optimisation
    duration_dummy = torch.ones(1, dtype=torch.float32)
    noisy_latent_np, latent_mask_np = tts._sample_noisy_latent(duration_dummy.cpu().numpy())
    noisy_latent = torch.tensor(noisy_latent_np, dtype=torch.float32).to(DEVICE)
    latent_mask = torch.tensor(latent_mask_np, dtype=torch.float32).to(DEVICE)

    # Determine initial style vectors
    if reference_style is not None:
        print(f"\nInitializing style from: {reference_style}")
        ref_style = load_voice_style(reference_style)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE).clone()
    else:
        # Choose the best existing style from the voice_styles directory by comparing WavLM features
        print("\nSearching for the closest pre‑existing style in 'voice_styles/'...")
        all_style_paths = sorted(glob.glob("voice_styles/[FM]*.json"))
        best_dist: float = float('inf')
        best_path: Optional[str] = None
        # Reuse fixed latent noise and text inputs for quick comparison
        for sp in all_style_paths:
            s = load_voice_style(sp)
            s_ttl = torch.tensor(s.ttl, dtype=torch.float32).to(DEVICE)
            s_dp = torch.tensor(s.dp, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                test_wav, _ = tts_forward(
                    text_inputs[0][0],
                    text_inputs[0][1],
                    s_ttl,
                    s_dp,
                    dp_model,
                    te_model,
                    ve_model,
                    voc_model,
                    total_step,
                    speed,
                    noisy_latent,
                    latent_mask,
                )
                dist = wavlm_feature_loss(wavlm, test_wav.squeeze(), target_feats).item()
            print(f"  {os.path.basename(sp)}: {dist:.4f}")
            if dist < best_dist:
                best_dist = dist
                best_path = sp
        if best_path is None:
            raise RuntimeError("No pre‑existing styles found in 'voice_styles/'")
        print(f"  >> Best: {os.path.basename(best_path)} (dist={best_dist:.4f})")
        ref_style = load_voice_style(best_path)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE).clone()

    print(f"  style_ttl: {tuple(style_ttl.shape)}, style_dp: {tuple(style_dp.shape)} (dp frozen)")

    # ===== Optimisation setup =====
    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=lr * 0.01
    )

    end_step = num_steps
    if start_step >= end_step:
        print(f"\nAlready reached target step ({end_step}). Nothing to do.")
        return

    print(f"\nStarting optimisation (step {start_step + 1} -> {end_step}, early stop at {threshold})...")
    start_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
    if start_time is not None:
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    best_loss: float = float('inf')
    best_ttl: Optional[torch.Tensor] = None
    best_dp = style_dp.detach().clone()

    # Optimisation loop
    for step in range(start_step, end_step):
        optimizer.zero_grad()
        # Rotate through provided texts to encourage robustness
        text_idx = step % len(text_inputs)
        text_ids, text_mask = text_inputs[text_idx]
        # Forward pass
        wav_out, _ = tts_forward(
            text_ids,
            text_mask,
            style_ttl,
            style_dp,
            dp_model,
            te_model,
            ve_model,
            voc_model,
            total_step,
            speed,
            noisy_latent,
            latent_mask,
        )
        gen_wav = wav_out.squeeze()
        # Compute perceptual loss
        loss = wavlm_feature_loss(wavlm, gen_wav, target_feats)
        # Backpropagate and update style_ttl
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        # Track best loss and style_ttl
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_ttl = style_ttl.detach().clone()
        # Logging
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step + 1}/{end_step} | Loss: {loss.item():.4f} | LR: {current_lr:.4f} | Best: {best_loss:.4f}")
        # Save intermediate checkpoints
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(log_dir, f"{config_name}_{step + 1:04d}.json")
            save_style(ckpt_path, best_ttl, best_dp, target_wav_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")
        # Early stopping
        if best_loss <= threshold:
            print(f"  Early stop at step {step + 1}: best loss {best_loss:.4f} <= {threshold}")
            break

    # ===== Save final result =====
    final_path = os.path.join(log_dir, f"{config_name}_final.json")
    print(f"\nSaving best style to: {final_path}")
    save_style(final_path, best_ttl, best_dp, target_wav_path)
    # Timing information
    if start_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time) / 1000.0  # seconds
        print(f"  Done! Best loss: {best_loss:.4f} | Time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    else:
        print(f"  Done! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
