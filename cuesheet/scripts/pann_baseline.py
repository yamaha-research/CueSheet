"""Zero-training baseline: PANN AudioSet probabilities → group → HMM smoothing.

Useful sanity-check answer to 'do we even need a trainable network?'.
This script does no learning at all — it just maps PANN's per-window class
probabilities onto our 6-class scheme, then runs the same Viterbi smoother the
trained model uses, and exports the resulting segments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.fusion.temporal import HMMSmoother, enforce_min_segment_per_class
from src.utils.config import load_config
from src.utils.metrics import merge_to_segments

# Tuned for the 6-class taxonomy. Without these biases Applause is eaten by
# the HMM (raw PANN detects it at ~1.4 % of seconds, default HMM keeps only
# 0.3–0.4 %). The empirical sweep confirmed the values below recover
# ~0.8 % Applause on f_7ntJHYAmc while preserving the gt_jazz
# GT span accuracy at 92.6 / 97.0 / 100.0 %.
APPLAUSE_BOOST_FLOOR  = 0.4    # raw Applause group score below this is ignored
APPLAUSE_BOOST_WEIGHT = 2.0    # added to the Applause posterior when score > floor
PERF_TO_APP_PRIOR     = 0.10   # P(Performance → Applause), vs 0.01 default
MC_TO_APP_PRIOR       = 0.10   # P(MC_Talk → Applause), vs 0.01 default
APPLAUSE_MIN_SEG_SEC  = 3      # post-Viterbi min-segment for Applause specifically

# Reuse the bootstrap labeler's group definitions
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling-import the helpers
from bootstrap_labels import (  # noqa: E402
    CLASS_GROUPS, apply_pre_concert_heuristic,
    apply_post_performance_applause_heuristic,
    build_group_indices, classify_windows, extract_wav, load_audioset_index,
)

TARGET_SR = 16_000


def map_groups_to_classes(probs: np.ndarray, group_idx: dict[str, list[int]],
                          classes: list[str], beat_density: np.ndarray | None,
                          ambient_beat_thresh: float = 5.0) -> np.ndarray:
    """Per-window posteriors over our 6 classes (no temperature scaling).

    Maps the 4 AudioSet groups onto our concert taxonomy:

      Performance group   → Performance
      MC_Talk group       → MC_Talk
      Applause group      → Applause (a class in its own right)
      AudioSet Silence / low beat → Ambient (room tone between active segments)

    Pre_Concert and Intermission have no PANN equivalent and stay near 0
    in this zero-training baseline — they need human labels.
    """
    T = len(probs)
    K = len(classes)
    name_to_id = {c: i for i, c in enumerate(classes)}

    # Sum AudioSet class probabilities within each group
    grp = {g: probs[:, idx].sum(axis=-1) for g, idx in group_idx.items()}

    out = np.zeros((T, K), dtype=np.float32)
    out[:, name_to_id["Performance"]] = grp["Performance"]
    out[:, name_to_id["MC_Talk"]]     = grp["MC_Talk"]
    out[:, name_to_id["Applause"]]    = grp["Applause"]
    out[:, name_to_id["Ambient"]]     = grp["Ambient"]
    # Pre_Concert / Intermission stay 0 — PANN has no signal for them.

    # Beat-aware Ambient boost — the boost adds Ambient mass when the
    # per-second beat_density is below ambient_beat_thresh; quiet
    # seconds get more boost. A low threshold (e.g. 0.5) is a near no-op
    # because per-second beat density on real concert recordings sits in
    # the 3-5 range nearly everywhere. The LOSO sweep picks threshold =
    # 5.0 mult = 0.3 as macroF1-optimal. NOTE: this is only active when
    # callers PASS beat_density; the live audio pipeline currently passes
    # None (streaming beat tracker is future work), so the boost is
    # OFFLINE-only.
    if beat_density is not None:
        boost = np.maximum(0.0, ambient_beat_thresh - beat_density) / ambient_beat_thresh
        out[:, name_to_id["Ambient"]] += 0.3 * boost

    # Applause-confidence boost — raw Applause group probability gets a
    # multiplier when PANN is confident (score > floor). This helps Applause
    # win argmax / survive HMM smoothing for the brief post-song burst that
    # otherwise gets absorbed into Performance.
    app_boost = np.maximum(0.0, grp["Applause"] - APPLAUSE_BOOST_FLOOR) * APPLAUSE_BOOST_WEIGHT
    out[:, name_to_id["Applause"]] += app_boost

    # Normalize rows to sum to 1
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
    return out / row_sum


def _check_audio_signal(wav_path: Path) -> None:
    """Warn loudly when the input audio is essentially silent.

    Found while auditing a close-mic studio session recording: the source
    MP3 was 100 % zeros (every one of 51M samples = 0). PANN then
    classified the entire 54-min show as Ambient, which *looked* like
    a model failure but was actually correct. Catching this at ingest
    saves the next investigator from chasing the wrong cause.

    Heuristic: peak amplitude < 1 % of full-scale = no real signal.
    Median absolute amplitude < 0.001 of full-scale = silent throughout.
    Reads the wav as int16, no full decode needed beyond what we already
    do for PANN downstream.
    """
    import soundfile as sf
    samples, _ = sf.read(str(wav_path), dtype="int16")
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)
    full_scale = 32767
    peak = int(np.abs(samples).max())
    med_abs = float(np.median(np.abs(samples)))
    if peak == 0:
        raise RuntimeError(
            f"AUDIO IS COMPLETELY SILENT — every sample in {wav_path.name} "
            f"is exactly zero. Source file is likely corrupt or was recorded "
            f"with the microphone disconnected. Refusing to run PANN; fix "
            f"the input file."
        )
    if peak < 0.01 * full_scale:
        print(f"WARNING: very quiet audio in {wav_path.name} — "
               f"peak amplitude {peak}/32767 ({100*peak/full_scale:.2f}% of "
               f"full-scale). PANN predictions may be unreliable.")
    elif med_abs < 0.001 * full_scale and peak > 0.01 * full_scale:
        print(f"WARNING: audio in {wav_path.name} is mostly silent — "
               f"median |amplitude| {med_abs:.0f}/32767 "
               f"({100*med_abs/full_scale:.3f}%) but peak is {peak}.")


def build_transition_matrix(classes: list[str]) -> np.ndarray:
    """Strong stay everywhere, plus a cheap entry into Applause from
    Performance and MC_Talk (those are the natural moments audience reacts).

    Without this bias, the default uniform leave=0.01 makes any
    Applause→Performance round-trip too costly for the Viterbi DP to take,
    so brief Applause spans get absorbed into the surrounding Performance run.
    """
    K = len(classes)
    T = np.full((K, K), 0.01, dtype=np.float64)
    np.fill_diagonal(T, 0.95)
    if "Applause" in classes and "Performance" in classes:
        T[classes.index("Performance"), classes.index("Applause")] = PERF_TO_APP_PRIOR
    if "Applause" in classes and "MC_Talk" in classes:
        T[classes.index("MC_Talk"), classes.index("Applause")] = MC_TO_APP_PRIOR
    # Renormalize rows so they sum to 1.
    return T / T.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=Path("pann_baseline.json"))
    p.add_argument("--beat-labels", type=Path, default=None)
    args = p.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name_to_idx = load_audioset_index()
    group_idx = build_group_indices(name_to_idx)

    # Run PANN inference
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = extract_wav(args.audio, tmp)
        _check_audio_signal(wav_path)
        print(f"Running PANN inference on {device}...")
        probs, timestamps = classify_windows(wav_path, device)

    # Optional beat labels for silence boost
    beat_density = None
    if args.beat_labels is None:
        guess = Path("data/labels/beats") / f"{args.audio.stem}_beats.npz"
        if guess.exists():
            args.beat_labels = guess
    if args.beat_labels and args.beat_labels.exists():
        beat_density = np.load(args.beat_labels)["beat_density"]
        print(f"Using beat density from {args.beat_labels.name}")

    # Posterior → HMM Viterbi (with Applause-aware transitions, internal
    # min-segment disabled) → per-class min-segment (Applause=3s, rest=10s)
    # → Pre_Concert temporal heuristic.
    posteriors = map_groups_to_classes(probs, group_idx, config.fusion.classes, beat_density)
    hmm = HMMSmoother(
        num_states=config.fusion.num_classes,
        min_segment_sec=1,  # we apply per-class min-segment ourselves below
        transition_matrix=build_transition_matrix(config.fusion.classes),
    )
    labels = hmm.decode_viterbi(posteriors)
    classes = config.fusion.classes
    default_min = int(config.temporal.min_segment_sec)
    per_class_min = {}
    if "Applause" in classes:
        per_class_min[classes.index("Applause")] = APPLAUSE_MIN_SEG_SEC
    labels = enforce_min_segment_per_class(labels, per_class_min, default=default_min)
    # Use the raw posterior argmax for music-onset detection, not the
    # Viterbi-smoothed labels. Otherwise the heuristic gives up on
    # music-dominated shows where Viterbi back-smoothes Performance to t=0
    # (e.g., PANN clearly sees Speech at sec 0-5, but Viterbi
    # absorbs those frames into the dominant Performance run).
    labels = apply_pre_concert_heuristic(
        labels, classes, onset_labels=compute_ambient_blind_onset(posteriors, CLASSES if 'CLASSES' in dir() else classes)
    )
    # Structural post-Performance applause heuristic — for shows where
    # PANN's Applause posterior is too weak to win argmax (e.g., Japanese
    # close-mic stage capture). Uses the smoothed Performance segments to
    # locate applause windows, then requires only a tiny posterior floor
    # so silence after a song doesn't get falsely labeled Applause.
    app_id = classes.index("Applause") if "Applause" in classes else None
    app_posterior = posteriors[:, app_id] if app_id is not None else None
    labels = apply_post_performance_applause_heuristic(
        labels, classes, applause_posterior=app_posterior,
    )

    # Export segments
    segments = merge_to_segments(labels, timestamps, config.fusion.classes)
    out = {
        "show_id": args.audio.stem,
        "duration_sec": int(timestamps[-1]) + 1,
        "segments": [
            {"start": round(s["start"], 1), "end": round(s["end"], 1),
             "label": s["label_name"], "duration": round(s["duration"], 1)}
            for s in segments if s["label"] >= 0
        ],
        "per_second": {
            "labels": labels.tolist(),
            "label_names": [config.fusion.classes[l] if l >= 0 else "unknown"
                            for l in labels],
        },
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Saved → {args.output}  ({len(out['segments'])} segments)")


if __name__ == "__main__":
    main()
