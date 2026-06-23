"""EfficientAT encoder adapter — mirrors PANN's `classify_windows` signature
so the rest of the pipeline (4-group mapping, boosts, HMM, heuristics)
can be reused unchanged.

EfficientAT (Schmid et al. 2023) distills PaSST into a MobileNet-style CNN
that's ~5 M params and ~0.65 GFLOPs (MN10) vs PANN CNN14's ~80 M / ~21 G —
designed for online / edge inference. Same 527-class AudioSet output, so
it drops into our pipeline without changing anything downstream.

Repo is vendored at ``vendor/EfficientAT/`` (gitignored) — clone it with::

    cd vendor && git clone --depth 1 https://github.com/fschmid56/EfficientAT.git

Usage (mirrors bootstrap_labels.classify_windows):

    from scripts.encoder_efficientat import classify_windows_efficientat
    probs, timestamps = classify_windows_efficientat(
        wav_path, device="cuda", model_name="mn10_as",
    )
"""
from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

# Add the vendored EfficientAT to sys.path so its `models.mn.model` and
# `models.preprocess` imports resolve. The repo expects its own root in
# sys.path (not as a package), so we shim it in here lazily.
REPO_ROOT = Path(__file__).resolve().parents[2]
EFFAT_ROOT = REPO_ROOT / "vendor" / "EfficientAT"
if not EFFAT_ROOT.exists():
    raise ImportError(
        f"EfficientAT not vendored at {EFFAT_ROOT}. Clone with:\n"
        f"  cd vendor && git clone --depth 1 "
        f"https://github.com/fschmid56/EfficientAT.git"
    )
if str(EFFAT_ROOT) not in sys.path:
    sys.path.insert(0, str(EFFAT_ROOT))

# EfficientAT's preprocessor expects 32 kHz mel-spec.
SAMPLE_RATE = 32_000   # EfficientAT default
WIN_SAMPLES = 800      # AugmentMelSTFT win_length default
HOP_SAMPLES = 320      # AugmentMelSTFT hopsize default
N_MELS = 128           # AudioSet-tuned

# EfficientAT was trained on 10-second clips. Using a 2-second window (PANN's
# default) saturates the sigmoid (every class fires at 1.0, row sum ~520)
# because the receptive field is too small for the model's learned features.
# We keep HOP_SEC = 1.0 so downstream code still gets one prediction per
# second; the window just looks at a longer context per prediction.
# (Verified empirically: 10-sec window row sums 0.71-5.64, comparable to
# PANN's 1.8-2.4. 5-sec window also works.)
WINDOW_SEC = 10.0
HOP_SEC = 1.0


_MODEL_CACHE: dict = {}


def _load(model_name: str, device: str):
    """Lazy-load the model + mel preprocessor. Cached per (model_name, device).

    EfficientAT's helpers/utils.py reads ``metadata/class_labels_indices.csv``
    with a relative path that assumes you ran the script from the repo root.
    We chdir into the vendored repo just for the import, then restore.
    """
    key = (model_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import os
    saved_cwd = os.getcwd()
    try:
        os.chdir(EFFAT_ROOT)
        from models.mn.model import get_model as get_mobilenet   # type: ignore
        from models.dymn.model import get_model as get_dymn       # type: ignore
        from models.preprocess import AugmentMelSTFT              # type: ignore
        from helpers.utils import NAME_TO_WIDTH                    # type: ignore
    finally:
        os.chdir(saved_cwd)

    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=NAME_TO_WIDTH(model_name),
                          pretrained_name=model_name,
                          strides=[2, 2, 2, 2])
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name),
                                pretrained_name=model_name,
                                strides=[2, 2, 2, 2],
                                head_type="mlp")
    model.to(device); model.eval()
    mel = AugmentMelSTFT(n_mels=N_MELS, sr=SAMPLE_RATE,
                          win_length=WIN_SAMPLES, hopsize=HOP_SAMPLES)
    mel.to(device); mel.eval()
    _MODEL_CACHE[key] = (model, mel)
    return model, mel


def classify_windows_efficientat(
    wav_path: Path,
    device: str = "cuda",
    model_name: str = "mn10_as",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window AudioSet posteriors. Matches `bootstrap_labels.classify_windows`.

    Returns
    -------
    probs : (T, 527) float32 — per-second AudioSet posteriors (sigmoid). T is
        derived from the audio length using the same WINDOW_SEC / HOP_SEC as
        PANN (2-sec windows, 1-sec hop) so downstream class-grouping and HMM
        code see identical-shape inputs from either encoder.
    timestamps : (T,) float32 — window-start seconds.
    """
    import librosa
    from torch import autocast
    model, mel = _load(model_name, device)
    # Resample to EfficientAT's native 32 kHz on load (librosa handles it)
    waveform, _ = librosa.core.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    waveform_t = torch.from_numpy(waveform).to(device)

    win_n = int(WINDOW_SEC * SAMPLE_RATE)
    hop_n = int(HOP_SEC * SAMPLE_RATE)
    if len(waveform_t) < win_n:
        # Pad short clips up to one window
        waveform_t = torch.nn.functional.pad(waveform_t, (0, win_n - len(waveform_t)))
    starts = list(range(0, len(waveform_t) - win_n + 1, hop_n))

    probs_out = np.zeros((len(starts), 527), dtype=np.float32)
    timestamps = np.zeros(len(starts), dtype=np.float32)
    ctx = autocast(device_type=device) if device == "cuda" else nullcontext()
    with torch.no_grad(), ctx:
        for i, s in enumerate(starts):
            chunk = waveform_t[s : s + win_n].unsqueeze(0)   # (1, win_n)
            spec = mel(chunk)                                  # (1, n_mels, T_mel)
            preds, _features = model(spec.unsqueeze(0))        # (1, 527)
            probs_out[i] = torch.sigmoid(preds.float()).squeeze(0).cpu().numpy()
            timestamps[i] = s / SAMPLE_RATE
            if (i + 1) % 200 == 0:
                print(f"  efficientat {i + 1}/{len(starts)}")
    return probs_out, timestamps


def benchmark_rtf(
    wav_path: Path,
    device: str = "cuda",
    model_name: str = "mn10_as",
) -> dict:
    """Run inference once and report Real-Time Factor = wall_time / audio_dur."""
    import librosa
    audio_dur = librosa.get_duration(path=str(wav_path))
    # Warmup pass — first call also loads weights, biases timing
    _ = classify_windows_efficientat(wav_path, device=device, model_name=model_name)
    t0 = time.perf_counter()
    probs, _ = classify_windows_efficientat(wav_path, device=device,
                                              model_name=model_name)
    wall = time.perf_counter() - t0
    return {
        "model": f"efficientat-{model_name}",
        "device": device,
        "audio_duration_sec": float(audio_dur),
        "wall_time_sec": float(wall),
        "rtf": float(wall / audio_dur),
        "frames": int(probs.shape[0]),
    }
