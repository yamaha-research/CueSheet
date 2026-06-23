"""Score the pipeline under the live causal window alignment.

The Section 4 numbers assign second t the label of window [t, t+10) (leading,
sees 9 s past t). The deployed causal mode labels t from [t-10, t] (trailing).
This re-scores the same encoder posteriors under the trailing convention —
post6_trail[t] = post6_lead[max(0, t-9)] — so the only change is the window
the label is attributed to, exactly the shift a live listener experiences.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cuesheet"))

import scripts.full_pipeline_6class_4shows as fp

SHIFT = 9  # leading window [t, t+10) re-attributed to its last covered second


def weighted_f1(pred, gt, n_classes=6):
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    num = den = 0.0
    for c in range(n_classes):
        tp = ((pred == c) & (gt == c)).sum()
        fp_ = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        f1 = 2 * tp / max(1, (2 * tp + fp_ + fn))
        sup = (gt == c).sum()
        num += f1 * sup
        den += sup
    return num / max(1, den)


def main():
    enc = fp.AudioEncoder(device="cpu")
    group_idx = fp.build_group_indices(fp.load_audioset_index())
    pooled = {"lead": ([], []), "trail": ([], [])}
    for show_id, genre in fp.SHOWS:
        audio = fp.load_audio(fp.AUDIO_DIR / f"{show_id}.mp3")
        post6 = fp.run_encoder_full(enc, group_idx, audio)
        gt_path = REPO / f"demo/data/{show_id}.json"
        d = json.loads(gt_path.read_text())
        gt = np.asarray(d["gt_labels"], dtype=int)
        post6_trail = np.concatenate(
            [np.repeat(post6[:1], SHIFT, axis=0), post6[:-SHIFT]])
        res = {}
        for tag, p6 in [("lead", post6), ("trail", post6_trail)]:
            pred = fp.v1_1_pipeline(p6)
            res[tag] = weighted_f1(pred, gt)
            n = min(len(pred), len(gt))
            pooled[tag][0].append(pred[:n])
            pooled[tag][1].append(gt[:n])
        print(f"{show_id} ({genre}): lead {res['lead']*100:.1f} "
              f"trail {res['trail']*100:.1f} "
              f"delta {(res['trail']-res['lead'])*100:+.1f}", flush=True)
    for tag in ("lead", "trail"):
        pred = np.concatenate(pooled[tag][0])
        gt = np.concatenate(pooled[tag][1])
        print(f"POOLED {tag}: {weighted_f1(pred, gt)*100:.2f}")


if __name__ == "__main__":
    main()
