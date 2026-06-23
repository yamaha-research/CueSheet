"""Quick eval: compare model predictions against hand-noted ground truth segments.

Edit GT_SEGMENTS and run with predictions json. Reports per-second confusion
matrix on covered ranges only — predictions outside the GT ranges are ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# (start_sec, end_sec, class_name) — extend as more reference annotations come in
GT_SEGMENTS: list[tuple[float, float, str]] = [
    (0.0,    1020.0, "Pre_Concert"),  # 0:00 – 17:00
    (1020.0, 1120.0, "MC_Talk"),      # 17:00 – 18:40
    (1134.0, 1406.0, "Performance"),  # 18:54 – 23:26
]


def predictions_per_second(pred_json: dict) -> tuple[np.ndarray, list[str]]:
    """Reconstruct per-second predicted class names from the segment list."""
    duration = pred_json["duration_sec"]
    classes = pred_json["per_second"].get("label_names")
    if classes:
        return np.array(classes, dtype=object), classes
    # Fallback: rebuild from segments
    out = np.array(["unknown"] * duration, dtype=object)
    for seg in pred_json["segments"]:
        s, e = int(seg["start"]), int(seg["end"])
        out[s:e] = seg["label"]
    return out, list(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    args = p.parse_args()

    with open(args.predictions) as f:
        pj = json.load(f)
    pred = np.array(pj["per_second"]["label_names"], dtype=object)
    duration = len(pred)

    print(f"Predictions cover {duration}s ({duration / 60:.1f} min)\n")

    total_correct = 0
    total_seconds = 0
    print(f"{'GT span':<22} {'GT class':<14} {'pred breakdown'}")
    print("-" * 90)
    for s, e, gt_cls in GT_SEGMENTS:
        s_i, e_i = int(round(s)), min(int(round(e)), duration)
        seg_preds = pred[s_i:e_i]
        counts = Counter(seg_preds)
        total = len(seg_preds)
        correct = counts.get(gt_cls, 0)
        total_correct += correct
        total_seconds += total

        pct_str = "  ".join(
            f"{cls}={c}({100*c/total:.0f}%)"
            for cls, c in counts.most_common()
        )
        print(f"{s:>5.0f}–{e:<5.0f}  ({total}s)   {gt_cls:<14} {pct_str}")
        print(f"{'':>30}            accuracy: {100*correct/total:.1f}%")

    print()
    print(f"Overall per-second accuracy on labeled ranges: "
          f"{total_correct}/{total_seconds} = {100*total_correct/total_seconds:.1f}%")


if __name__ == "__main__":
    main()
