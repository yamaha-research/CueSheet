"""HTS-AT (Hierarchical Token-Semantic Audio Transformer) wrapper that
mirrors PANN's classify_windows signature and returns (n_seconds, 527)
AudioSet posteriors.

Uses the official AudioSet-fine-tuned checkpoint from RetroCirce's repo
(~32M params, AudioSet mAP 0.471). 32 kHz sample rate, 10-sec window,
1-sec hop -- 10-sec mirrors EfficientAT and AST for grid parity.

Setup (one-time):
    cd vendor && git clone --depth 1 https://github.com/RetroCirce/HTS-Audio-Transformer.git HTS-AT
    uv add gdown torchlibrosa
    mkdir -p vendor/HTS-AT/ckpt
    uv run gdown 1OK8a5XuMVLyeVKF117L8pfxeZYdfSDZv -O vendor/HTS-AT/ckpt/HTSAT_AudioSet_Saved_1.ckpt

Usage:
    from encoder_htsat import classify_windows_htsat
    probs, timestamps = classify_windows_htsat(wav_path, device="cuda")
    # probs.shape == (n_seconds, 527)  -- sigmoid-activated AudioSet posteriors
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HTSAT_ROOT = REPO / "vendor" / "HTS-AT"
HTSAT_CKPT = HTSAT_ROOT / "ckpt" / "HTSAT_AudioSet_Saved_1.ckpt"

SAMPLE_RATE = 32_000
WINDOW_SEC = 10.0
HOP_SEC = 1.0

_MODEL_CACHE: dict[str, object] = {}


def _load(device: str):
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]
    if not HTSAT_CKPT.exists():
        raise SystemExit(
            f"HTS-AT checkpoint missing at {HTSAT_CKPT}. See module docstring "
            f"for one-time setup commands."
        )
    # HTS-AT's config.py uses relative paths internally for its training data;
    # we only need it for inference-side hyperparams (sr, mel bins, etc.),
    # so we chdir into the vendor root temporarily so its imports succeed.
    prev_cwd = os.getcwd()
    os.chdir(HTSAT_ROOT)
    sys.path.insert(0, str(HTSAT_ROOT))
    try:
        from model.htsat import HTSAT_Swin_Transformer  # type: ignore
        import config as htsat_cfg  # type: ignore
        model = HTSAT_Swin_Transformer(
            spec_size=256, patch_size=4, patch_stride=(4, 4),
            num_classes=527, window_size=8, config=htsat_cfg,
            depths=[2, 2, 6, 2], embed_dim=96, num_heads=[4, 8, 16, 32],
        )
        sd = torch.load(HTSAT_CKPT, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("sed_model.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
    finally:
        os.chdir(prev_cwd)
        # Keep HTSAT_ROOT in sys.path so future imports inside the model
        # (e.g., torchlibrosa lookups) still resolve.

    _MODEL_CACHE[device] = model
    return model


def classify_windows_htsat(wav_path: str | Path, device: str = "cuda",
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, timestamps).

    probs has shape (n_seconds, 527) with sigmoid-activated AudioSet
    posteriors. AudioSet 527 ordering matches PANN's, so the 4-group
    mapping in bootstrap_labels.py works without re-indexing.
    """
    model = _load(device)

    wav, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    win_n = int(WINDOW_SEC * SAMPLE_RATE)
    hop_n = int(HOP_SEC * SAMPLE_RATE)
    if len(wav) < win_n:
        wav = np.pad(wav, (0, win_n - len(wav)))

    starts = list(range(0, len(wav) - win_n + 1, hop_n))
    probs = np.zeros((len(starts), 527), dtype=np.float32)
    ts = np.zeros(len(starts), dtype=np.float32)

    with torch.no_grad():
        for i, s in enumerate(starts):
            x = torch.from_numpy(wav[s:s + win_n]).unsqueeze(0).to(device)
            out = model(x, None, infer_mode=False)
            # HTS-AT returns sigmoid-activated clipwise_output already.
            probs[i] = out["clipwise_output"].squeeze(0).cpu().numpy()
            ts[i] = s / SAMPLE_RATE
            if (i + 1) % 200 == 0:
                print(f"  htsat {i + 1}/{len(starts)}", flush=True)

    return probs, ts
