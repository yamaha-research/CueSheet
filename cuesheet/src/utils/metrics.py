"""Evaluation metrics: per-class F1, segment IoU, boundary displacement error."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, f1_score


def merge_to_segments(labels: np.ndarray, timestamps: np.ndarray | None = None, class_names: list[str] | None = None) -> list[dict]:
    """Run-length encode a per-second label array into segment dicts."""
    if len(labels) == 0:
        return []
    segments = []
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start_idx]:
            seg_label = int(labels[start_idx])
            start_sec = float(timestamps[start_idx]) if timestamps is not None else float(start_idx)
            end_sec   = float(timestamps[i - 1])     if timestamps is not None else float(i - 1)
            segments.append({
                "start": start_sec,
                "end":   end_sec + 1.0,
                "label": seg_label,
                "label_name": class_names[seg_label] if class_names and seg_label >= 0 else str(seg_label),
                "duration": end_sec + 1.0 - start_sec,
            })
            start_idx = i
    # final segment
    seg_label = int(labels[start_idx])
    start_sec = float(timestamps[start_idx]) if timestamps is not None else float(start_idx)
    end_sec   = float(timestamps[-1])        if timestamps is not None else float(len(labels) - 1)
    segments.append({
        "start": start_sec,
        "end":   end_sec + 1.0,
        "label": seg_label,
        "label_name": class_names[seg_label] if class_names and seg_label >= 0 else str(seg_label),
        "duration": end_sec + 1.0 - start_sec,
    })
    return segments


def compute_metrics(pred: np.ndarray, true: np.ndarray, class_names: list[str]) -> dict:
    """Compute per-class F1, macro F1, segment IoU, and boundary displacement error."""
    mask = true >= 0
    pred_m, true_m = pred[mask], true[mask]
    if len(pred_m) == 0:
        return {}

    labels = list(range(len(class_names)))
    f1_macro = float(f1_score(true_m, pred_m, labels=labels, average="macro", zero_division=0))
    report = classification_report(true_m, pred_m, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    per_class_f1 = {c: report[c]["f1-score"] for c in class_names if c in report}

    seg_iou = _segment_iou(pred, true)
    bde_mean, bde_p90 = _boundary_displacement_error(pred, true)

    return {
        "f1_macro": f1_macro,
        "f1_per_class": per_class_f1,
        "segment_iou": seg_iou,
        "boundary_error_mean": bde_mean,
        "boundary_error_p90": bde_p90,
    }


def _segment_iou(pred: np.ndarray, true: np.ndarray) -> float:
    pred_segs = merge_to_segments(pred)
    true_segs = merge_to_segments(true)
    if not pred_segs or not true_segs:
        return 0.0
    ious = []
    for ps in pred_segs:
        if ps["label"] < 0:
            continue
        best = 0.0
        for ts in true_segs:
            if ts["label"] != ps["label"]:
                continue
            inter = max(0, min(ps["end"], ts["end"]) - max(ps["start"], ts["start"]))
            union = max(ps["end"], ts["end"]) - min(ps["start"], ts["start"])
            if union > 0:
                best = max(best, inter / union)
        ious.append(best)
    return float(np.mean(ious)) if ious else 0.0


def _boundary_displacement_error(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    def get_boundaries(arr: np.ndarray) -> np.ndarray:
        return np.where(np.diff(arr) != 0)[0].astype(float) + 0.5

    pred_b = get_boundaries(pred)
    true_b = get_boundaries(true)
    if len(pred_b) == 0 or len(true_b) == 0:
        return 0.0, 0.0
    errors = []
    for pb in pred_b:
        errors.append(float(np.min(np.abs(true_b - pb))))
    return float(np.mean(errors)), float(np.percentile(errors, 90))
