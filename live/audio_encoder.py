"""Streaming-friendly EfficientAT inference.

Wraps the offline per-iteration step from
`cuesheet/scripts/encoder_efficientat.py::classify_windows_efficientat`
so that one 10 second window of 32 kHz mono audio yields the same 527-class
AudioSet posterior as the offline batch would produce on that same window.

Strict requirement: the streaming output for a window must equal the offline
output for the identical window, to within float-32 numerical noise.
"""
from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

# Reuse the offline module's model loader and constants verbatim so the
# pre-trained weights, mel front-end, and sample rate are byte-identical.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cuesheet"))
from scripts.encoder_efficientat import (  # noqa: E402
    HOP_SEC, SAMPLE_RATE, WINDOW_SEC, _load,
)

WIN_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)   # 320_000 at 10 s, 32 kHz
HOP_SAMPLES = int(HOP_SEC * SAMPLE_RATE)      # 32_000 at 1 s, 32 kHz


class AudioEncoder:
    """One-shot inference on a single 10 s window. Stateless w.r.t. the
    waveform; safe to call repeatedly from a streaming loop."""

    def __init__(self, device: str = "cpu", model_name: str = "mn10_as"):
        self.device = device
        self.model_name = model_name
        # `_load` caches by (model_name, device); calling again returns the
        # same model/mel pair, so a streaming run reuses the offline cache.
        self.model, self.mel = _load(model_name, device)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Return a (527,) float32 sigmoid posterior for one 10 s window."""
        if window.dtype != np.float32:
            raise TypeError(f"window dtype must be float32, got {window.dtype}")
        if window.shape != (WIN_SAMPLES,):
            raise ValueError(
                f"window shape must be ({WIN_SAMPLES},), got {window.shape}")
        chunk = torch.from_numpy(window).to(self.device).unsqueeze(0)
        autocast = (torch.autocast(device_type=self.device)
                    if self.device == "cuda" else nullcontext())
        with torch.no_grad(), autocast:
            spec = self.mel(chunk)                       # (1, n_mels, T_mel)
            preds, _ = self.model(spec.unsqueeze(0))     # (1, 527)
            probs = torch.sigmoid(preds.float()).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)
