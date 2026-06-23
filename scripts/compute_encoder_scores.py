"""Compute pooled weighted-F1 for each encoder across the 4 gallery shows, then
write the scores into a small JSON the frontend reads so the encoder dropdown
can show 'PANN CNN14 · 81 M · wF1 88%'.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "demo/data"
SHOWS = ["tinydesk_seventeen", "f_7ntJHYAmc",
         "boilerroom_fredagain_london", "allofbach_bwv140"]
ENCODERS = ["efficientat", "pann", "ast", "htsat"]
N_CLASSES = 6


def per_class_f1(pred, gt, n_classes=N_CLASSES):
    n = min(len(pred), len(gt))
    pred, gt = np.asarray(pred[:n]), np.asarray(gt[:n])
    per_class = []
    for c in range(n_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((gt == c).sum())
        per_class.append({"f1": f1, "support": support})
    total = sum(p["support"] for p in per_class)
    if total > 0:
        wf1 = sum(p["f1"] * p["support"] for p in per_class) / total
    else:
        wf1 = 0.0
    return wf1, n


def main():
    out = {}
    for enc in ENCODERS:
        pooled_pred, pooled_gt = [], []
        per_show = {}
        for show in SHOWS:
            suffix = "" if enc == "efficientat" else f"_{enc}"
            path = DATA_DIR / f"{show}{suffix}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            gt = d.get("gt_labels")
            state = d.get("segment_int")
            if gt is None or state is None:
                continue
            wf1, n = per_class_f1(state, gt)
            per_show[show] = round(wf1 * 100, 1)
            pooled_pred.extend(state[:n])
            pooled_gt.extend(gt[:n])
        if pooled_pred:
            pwf1, _ = per_class_f1(pooled_pred, pooled_gt)
            out[enc] = {
                "pooled_wf1_pct": round(pwf1 * 100, 1),
                "per_show_wf1_pct": per_show,
            }

    out_path = DATA_DIR / "encoder_scores.json"
    out_path.write_text(json.dumps(out, indent=2))
    # Also write into demo/static/data so the frontend can fetch it.
    static_dir = REPO / "demo/static/data"
    if static_dir.is_dir():
        (static_dir / "encoder_scores.json").write_text(json.dumps(out, indent=2))

    print(f"Saved {out_path}")
    for enc, info in out.items():
        print(f"  {enc:12s} pooled wF1 {info['pooled_wf1_pct']}%  per-show: {info['per_show_wf1_pct']}")


if __name__ == "__main__":
    main()
