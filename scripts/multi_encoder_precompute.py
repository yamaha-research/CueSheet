"""Precompute per-second posteriors for PANN, AST, HTS-AT on the 4 paper shows.

Output is one JSON per (show, encoder) that the HF Space frontend loads when the
user swaps the encoder, so the swap actually changes what the ribbon and the
per-second card show. EfficientAT outputs already exist as {show}.json and are
kept verbatim — this script only fills in PANN, AST, HTS-AT.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cuesheet"))

from live.online_hmm import OnlineCausalHMM
from live.causal_heuristics import CausalIntermission
from cuesheet.scripts.pann_baseline import map_groups_to_classes
from cuesheet.scripts.bootstrap_labels import (
    apply_pre_concert_heuristic, apply_post_performance_applause_heuristic,
    build_group_indices, compute_ambient_blind_onset, load_audioset_index,
)

CLASSES = ["Pre_Concert", "Performance", "MC_Talk", "Applause",
           "Intermission", "Ambient"]
SHOWS = ["tinydesk_seventeen", "f_7ntJHYAmc",
         "boilerroom_fredagain_london", "allofbach_bwv140"]
AUDIO_DIR = REPO / "data/raw"
OUT_DIR = REPO / "demo/data"


def load_audio_at_sr(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def encode_pann(audio_path: Path) -> np.ndarray:
    """Run PANN CNN14 over the full audio, return (T, 527) per-second posterior."""
    import tempfile
    import librosa
    from scripts.bootstrap_labels import classify_windows
    audio32k = load_audio_at_sr(audio_path, 32_000)
    audio16k = librosa.resample(audio32k, orig_sr=32_000, target_sr=16_000)
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "audio16.wav"
        sf.write(str(clip), audio16k, 16_000)
        probs, _ = classify_windows(clip, "cpu")
    return np.asarray(probs, dtype=np.float32)


def encode_ast(audio_path: Path) -> np.ndarray:
    """Run AST over the full audio."""
    import tempfile
    import librosa
    from scripts.encoder_ast import classify_windows_ast
    audio32k = load_audio_at_sr(audio_path, 32_000)
    audio16k = librosa.resample(audio32k, orig_sr=32_000, target_sr=16_000)
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "audio16.wav"
        sf.write(str(clip), audio16k, 16_000)
        probs, _ = classify_windows_ast(str(clip), "cpu")
    return np.asarray(probs, dtype=np.float32)


def encode_htsat(audio_path: Path) -> np.ndarray:
    """Run HTS-AT over the full audio (32 kHz mono)."""
    import tempfile
    from scripts.encoder_htsat import classify_windows_htsat
    audio32k = load_audio_at_sr(audio_path, 32_000)
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "audio32.wav"
        sf.write(str(clip), audio32k, 32_000)
        probs, _ = classify_windows_htsat(str(clip), "cpu")
    return np.asarray(probs, dtype=np.float32)


def full_pipeline(probs527: np.ndarray, group_idx) -> tuple[np.ndarray, np.ndarray]:
    """Apply group map + HMM + Pre-concert + post-applause + Intermission.
    Returns (segment_int, post6_smoothed) — both length T."""
    # Group map 527 -> 6 (Pre-concert and Intermission columns stay 0)
    post6 = np.stack([
        map_groups_to_classes(p[None, :], group_idx, CLASSES, beat_density=None)[0]
        for p in probs527
    ]).astype(np.float32)
    # HMM
    K = len(CLASSES)
    off = (1.0 - 0.95) / (K - 1)
    T = np.full((K, K), off, dtype=np.float64)
    np.fill_diagonal(T, 0.95)
    hmm = OnlineCausalHMM(T)
    smoothed = np.zeros(len(post6), dtype=int)
    for t, p in enumerate(post6):
        smoothed[t] = hmm.step(p.astype(np.float64))
    # Pre-concert rule
    out = apply_pre_concert_heuristic(
        smoothed, CLASSES, active_classes=("Performance",), min_active_run=10,
        onset_labels=compute_ambient_blind_onset(post6, CLASSES))
    out = apply_post_performance_applause_heuristic(
        out, CLASSES,
        applause_posterior=post6[:, CLASSES.index("Applause")],
        min_perf_run=40, window_sec=12, min_post_evidence=0.10)
    # CausalIntermission
    inter = CausalIntermission()
    final = np.zeros_like(out)
    for t in range(len(out)):
        final[t] = inter.update(int(out[t]))
    return final, post6


def load_audioset_names() -> list[str]:
    import csv
    with open(Path.home() / "panns_data/class_labels_indices.csv") as f:
        rows = list(csv.reader(f))[1:]
    return [r[2] for r in rows]


def main():
    print(f"Group map index loading...")
    group_idx = build_group_indices(load_audioset_index())
    audioset_names = load_audioset_names()
    encoders = {
        "pann": encode_pann,
        "ast": encode_ast,
        "htsat": encode_htsat,
    }

    for show in SHOWS:
        audio_path = AUDIO_DIR / f"{show}.mp3"
        if not audio_path.exists():
            print(f"  ⚠ {show}: audio not found at {audio_path}")
            continue
        # Load existing show JSON (EfficientAT result)
        eff_path = OUT_DIR / f"{show}.json"
        with open(eff_path) as f:
            base = json.load(f)
        n_sec = base["n_seconds"]
        print(f"\n=== {show} (n_seconds={n_sec}) ===")

        for enc_name, enc_fn in encoders.items():
            out_path = OUT_DIR / f"{show}_{enc_name}.json"
            if out_path.exists():
                print(f"  {enc_name}: already exists, skipping")
                continue
            print(f"  {enc_name}: encoding...")
            t0 = time.time()
            probs527 = enc_fn(audio_path)
            print(f"    encoded {len(probs527)}s in {time.time()-t0:.0f}s")

            # Trim to base n_seconds for consistency
            probs527 = probs527[:n_sec]

            # Apply full pipeline
            segment_int, post6 = full_pipeline(probs527, group_idx)

            # The demo posterior is the 4-class [Performance, MC_Talk, Applause, Ambient]
            # subset — indices 1, 2, 3, 5 of the 6-class post6.
            posterior_4 = post6[:, [1, 2, 3, 5]]

            # Build audioset_top — top-7 AudioSet classes per second
            audioset_top = []
            for t in range(len(probs527)):
                top_idx = np.argsort(probs527[t])[-7:][::-1]
                audioset_top.append([
                    {"name": audioset_names[i] if i < len(audioset_names) else f"class_{i}",
                     "p": float(probs527[t, i])}
                    for i in top_idx
                ])

            # Write compact JSON keeping the base structure but overwriting
            # segment_int / posterior / audioset_top with the encoder-specific values.
            out_data = dict(base)
            out_data["segment_int"] = segment_int.tolist()
            out_data["posterior"] = posterior_4.tolist()
            out_data["audioset_top"] = audioset_top
            out_data["encoder"] = enc_name
            with open(out_path, "w") as f:
                json.dump(out_data, f)
            print(f"    saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
