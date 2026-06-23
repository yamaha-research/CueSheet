"""Full 6-class pipeline evaluation on the 4 paper shows.

Pipelines compared:
  1. Naive baseline: argmax(group-mapped post6), no HMM, no rules
  2. HMM smoothing + Pre-concert rule (no Intermission emit)
  3. (2) + CausalIntermission heuristic (full 6 classes emitted)

Computes per-class + weighted + macro F1 per show + pooled.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cuesheet"))

from live.audio_encoder import AudioEncoder, HOP_SAMPLES, WIN_SAMPLES, SAMPLE_RATE
from live.online_hmm import OnlineCausalHMM
from live.causal_heuristics import CausalIntermission
from cuesheet.scripts.pann_baseline import map_groups_to_classes
from cuesheet.scripts.bootstrap_labels import (
    apply_pre_concert_heuristic, apply_post_performance_applause_heuristic,
    build_group_indices, compute_ambient_blind_onset, load_audioset_index,
)

CLASSES = ["Pre_Concert", "Performance", "MC_Talk", "Applause",
           "Intermission", "Ambient"]
SHOWS = [
    ("tinydesk_seventeen", "pop"),
    ("f_7ntJHYAmc", "jazz"),
    ("boilerroom_fredagain_london", "electronic"),
    ("allofbach_bwv140", "classical"),
]
AUDIO_DIR = REPO / "data/raw"  # full audio; fetch with scripts/download_data.sh
# Per-show JSONs carrying the per-second human GT (gt_labels). The public repo
# includes them in demo/data/.
DATA_DIR = REPO / "demo/data"


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def run_encoder_full(enc: AudioEncoder, group_idx, audio: np.ndarray) -> np.ndarray:
    """Return (T, 6) post6 array per second."""
    n_samples = len(audio)
    if n_samples < WIN_SAMPLES:
        pad = np.zeros(WIN_SAMPLES - n_samples, dtype=np.float32)
        audio = np.concatenate([audio, pad])
        n_samples = len(audio)
    n_seconds = (n_samples - WIN_SAMPLES) // HOP_SAMPLES + 1
    out = np.zeros((n_seconds, len(CLASSES)), dtype=np.float32)
    t0 = time.time()
    for t in range(n_seconds):
        start = t * HOP_SAMPLES
        window = audio[start:start + WIN_SAMPLES]
        probs527 = enc.predict(window)
        post6 = map_groups_to_classes(
            probs527[None, :], group_idx, CLASSES, beat_density=None
        )[0]
        out[t] = post6.astype(np.float32)
        if t % 500 == 0 and t > 0:
            print(f"    [{t}/{n_seconds}] {(time.time()-t0):.0f}s elapsed")
    return out


def v1_0_pipeline(post6: np.ndarray) -> np.ndarray:
    """HMM + Pre-concert rule (no Intermission emit)."""
    K = len(CLASSES)
    off = (1.0 - 0.95) / (K - 1)
    T = np.full((K, K), off, dtype=np.float64)
    np.fill_diagonal(T, 0.95)
    hmm = OnlineCausalHMM(T)
    hmm_labels = np.zeros(len(post6), dtype=int)
    for t, p in enumerate(post6):
        hmm_labels[t] = hmm.step(p.astype(np.float64))
    # Pre-concert rule
    out = apply_pre_concert_heuristic(
        hmm_labels, CLASSES, active_classes=("Performance",),
        min_active_run=10,
        onset_labels=compute_ambient_blind_onset(post6, CLASSES))
    # Post-performance applause heuristic (already in eval_on_full_shows.py)
    out = apply_post_performance_applause_heuristic(
        out, CLASSES,
        applause_posterior=post6[:, CLASSES.index("Applause")],
        min_perf_run=40, window_sec=12, min_post_evidence=0.10)
    return out


def v1_1_pipeline(post6: np.ndarray) -> np.ndarray:
    """v1_0_pipeline + CausalIntermission heuristic (HMM + both rules)."""
    base = v1_0_pipeline(post6)
    inter = CausalIntermission()
    final = np.zeros_like(base)
    for t in range(len(base)):
        final[t] = inter.update(int(base[t]))
    return final


def hmm_only_pipeline(post6: np.ndarray) -> np.ndarray:
    """B: HMM smoothing only, NO rules."""
    K = len(CLASSES)
    off = (1.0 - 0.95) / (K - 1)
    T = np.full((K, K), off, dtype=np.float64)
    np.fill_diagonal(T, 0.95)
    hmm = OnlineCausalHMM(T)
    out = np.zeros(len(post6), dtype=int)
    for t, p in enumerate(post6):
        out[t] = hmm.step(p.astype(np.float64))
    return out


def rules_only_pipeline(post6: np.ndarray) -> np.ndarray:
    """C: Pre-concert rule + CausalIntermission rule, NO HMM (operates on raw argmax)."""
    raw = post6.argmax(axis=1)
    # Apply Pre-concert rule to raw labels
    out = apply_pre_concert_heuristic(
        raw, CLASSES, active_classes=("Performance",),
        min_active_run=10,
        onset_labels=compute_ambient_blind_onset(post6, CLASSES))
    out = apply_post_performance_applause_heuristic(
        out, CLASSES,
        applause_posterior=post6[:, CLASSES.index("Applause")],
        min_perf_run=40, window_sec=12, min_post_evidence=0.10)
    # Apply CausalIntermission on top
    inter = CausalIntermission()
    final = np.zeros_like(out)
    for t in range(len(out)):
        final[t] = inter.update(int(out[t]))
    return final


def per_class_f1(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict:
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    per_class = {}
    for c in range(n_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((gt == c).sum())
        per_class[CLASSES[c]] = {"f1": f1, "support": support}
    total_support = sum(p["support"] for p in per_class.values())
    if total_support > 0:
        wf1 = sum(p["f1"] * p["support"] for p in per_class.values()) / total_support
    else:
        wf1 = 0.0
    classes_with_support = [p for p in per_class.values() if p["support"] > 0]
    macro = sum(p["f1"] for p in classes_with_support) / max(len(classes_with_support), 1)
    return {"weighted_f1": wf1, "macro_f1": macro, "per_class": per_class, "n": n}


def main():
    print("Loading encoder + group_map...")
    audioset_idx = load_audioset_index()
    group_idx = build_group_indices(audioset_idx)
    enc = AudioEncoder(device="cpu")

    results = {"per_show": {}, "pooled": {}}
    pooled = {"A_naive": [], "B_hmm_only": [], "C_rules_only": [], "D_full": [], "gt": []}

    for show_id, genre in SHOWS:
        audio_path = AUDIO_DIR / f"{show_id}.mp3"
        data_path = DATA_DIR / f"{show_id}.json"
        if not audio_path.exists() or not data_path.exists():
            print(f"  skip {show_id}: audio not in data/raw/ "
                  "(run: bash scripts/download_data.sh)")
            continue
        print(f"\n=== {show_id} ({genre}) ===")
        audio = load_audio(audio_path)
        print(f"  Loading + encoding {len(audio)/SAMPLE_RATE:.0f}s of audio...")
        t0 = time.time()
        post6 = run_encoder_full(enc, group_idx, audio)
        print(f"  Encoded {len(post6)}s in {time.time()-t0:.0f}s")

        with open(data_path) as f:
            d = json.load(f)
        # CRITICAL: use gt_labels (human annotations), NOT segment_int (prior pipeline output)
        gt = np.asarray(d["gt_labels"], dtype=int)
        n = min(len(post6), len(gt))
        post6 = post6[:n]
        gt = gt[:n]

        # Run all 4 variants (A: naive, B: HMM only, C: rules only, D: HMM+rules)
        naive = post6.argmax(axis=1)
        hmm_only = hmm_only_pipeline(post6)[:n]
        rules_only = rules_only_pipeline(post6)[:n]
        v11 = v1_1_pipeline(post6)[:n]

        # Per-show eval
        m_naive = per_class_f1(naive, gt, len(CLASSES))
        m_hmm = per_class_f1(hmm_only, gt, len(CLASSES))
        m_rules = per_class_f1(rules_only, gt, len(CLASSES))
        m_full = per_class_f1(v11, gt, len(CLASSES))
        print(f"  A. Naive:        wF1={m_naive['weighted_f1']*100:.1f}%  macroF1={m_naive['macro_f1']*100:.1f}%")
        print(f"  B. HMM only:     wF1={m_hmm['weighted_f1']*100:.1f}%  macroF1={m_hmm['macro_f1']*100:.1f}%")
        print(f"  C. Rules only:   wF1={m_rules['weighted_f1']*100:.1f}%  macroF1={m_rules['macro_f1']*100:.1f}%")
        print(f"  D. HMM+Rules:    wF1={m_full['weighted_f1']*100:.1f}%  macroF1={m_full['macro_f1']*100:.1f}%")

        results["per_show"][show_id] = {
            "genre": genre,
            "A_naive": m_naive,
            "B_hmm_only": m_hmm,
            "C_rules_only": m_rules,
            "D_full": m_full,
            "n": n,
        }
        pooled["A_naive"].extend(naive.tolist())
        pooled["B_hmm_only"].extend(hmm_only.tolist())
        pooled["C_rules_only"].extend(rules_only.tolist())
        pooled["D_full"].extend(v11.tolist())
        pooled["gt"].extend(gt.tolist())

    if not pooled["gt"]:
        print("\nNo gallery audio found in data/raw/, so nothing was scored.")
        print("Fetch the audio first:  bash scripts/download_data.sh")
        return

    # Pooled
    print("\n=== POOLED across 4 shows ===")
    for name in ["A_naive", "B_hmm_only", "C_rules_only", "D_full"]:
        m = per_class_f1(np.asarray(pooled[name]), np.asarray(pooled["gt"]), len(CLASSES))
        results["pooled"][name] = m
        print(f"  {name:15s} wF1={m['weighted_f1']*100:.1f}%  macroF1={m['macro_f1']*100:.1f}%")
        for cls, p in m["per_class"].items():
            if p["support"] > 0:
                print(f"    {cls:14s} F1={p['f1']*100:5.1f}% (support={p['support']})")
        print()

    out_path = REPO / "scripts/full_pipeline_6class_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
