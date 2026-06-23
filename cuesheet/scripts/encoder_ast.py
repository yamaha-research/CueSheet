"""AST (Audio Spectrogram Transformer) wrapper that mirrors PANN's
classify_windows signature and returns (n_seconds, 527) AudioSet posteriors.

Uses MIT/ast-finetuned-audioset-10-10-0.4593 from Hugging Face --
the canonical AudioSet-fine-tuned AST (~88M params, AudioSet mAP 0.485).

10-sec windows (model's training regime), 1-sec hop -- mirrors the EfficientAT
wrapper so downstream HMM smoothing operates on the same per-second grid.

Usage:
    from encoder_ast import classify_windows_ast
    probs, timestamps = classify_windows_ast(wav_path, device="cuda")
    # probs.shape == (n_seconds, 527)  -- sigmoid-activated AudioSet posteriors
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# AST: 16 kHz sampling, 10-sec training clip, 1-sec hop matches our grid.
SAMPLE_RATE = 16_000
WINDOW_SEC = 10.0
HOP_SEC = 1.0
HF_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _load(device: str):
    key = (HF_MODEL_ID, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    feature_extractor = ASTFeatureExtractor.from_pretrained(HF_MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(HF_MODEL_ID).to(device).eval()
    _MODEL_CACHE[key] = (feature_extractor, model)
    return _MODEL_CACHE[key]


def classify_windows_ast(wav_path: str | Path, device: str = "cuda",
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, timestamps).

    probs has shape (n_seconds, 527) with sigmoid-activated AudioSet
    posteriors aligned to integer seconds (window_start). The label
    ordering matches PANN's AudioSet 527 ordering, so the 4-group
    mapping in bootstrap_labels.py works without re-indexing.
    """
    feature_extractor, model = _load(device)

    wav, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    win_n = int(WINDOW_SEC * SAMPLE_RATE)
    hop_n = int(HOP_SEC * SAMPLE_RATE)
    if len(wav) < win_n:
        wav = np.pad(wav, (0, win_n - len(wav)))

    starts = list(range(0, len(wav) - win_n + 1, hop_n))
    probs = np.zeros((len(starts), 527), dtype=np.float32)
    ts = np.zeros(len(starts), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(starts), 8):
            batch_starts = starts[i:i + 8]
            chunks = np.stack([wav[s:s + win_n] for s in batch_starts])
            inputs = feature_extractor(
                [c for c in chunks],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            for j, s in enumerate(batch_starts):
                probs[i + j] = batch_probs[j]
                ts[i + j] = s / SAMPLE_RATE

    return probs, ts
