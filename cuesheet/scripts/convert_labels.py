"""Convert Label Studio export JSON to second-level label arrays.

Usage:
    uv run python scripts/convert_labels.py data/labels/export.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Order MUST match configs/default.yaml `fusion.classes`
CLASSES = ["Pre_Concert", "Performance", "MC_Talk", "Applause", "Intermission", "Ambient"]


def parse_ls_export(export_path: Path, output_dir: Path) -> None:
    with open(export_path) as f:
        tasks = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        # extract filename from task data (audio or video key)
        file_url = task["data"].get("audio") or task["data"].get("video", "")
        video_name = Path(re.split(r"[?/]", file_url)[-1]).stem

        annotations = task.get("annotations", [])
        if not annotations:
            print(f"  SKIP {video_name} — no annotations")
            continue

        results = annotations[0]["result"]  # take first annotator

        # collect (start_sec, end_sec, label) tuples
        segments = []
        for r in results:
            if r["type"] != "videorectangle" and r.get("from_name") != "label":
                continue
            value = r["value"]
            start = value.get("start", value.get("startTime", 0))
            end   = value.get("end",   value.get("endTime",   0))
            label = value["labels"][0] if value.get("labels") else value.get("label")
            if label in CLASSES:
                segments.append((float(start), float(end), label))

        if not segments:
            print(f"  SKIP {video_name} — no valid segments")
            continue

        # build second-level label array; unlabeled gaps → Ambient (residual)
        max_sec = int(max(e for _, e, _ in segments)) + 1
        ambient_idx = CLASSES.index("Ambient")
        labels = np.full(max_sec, ambient_idx, dtype=np.int8)

        for start, end, label in segments:
            idx = CLASSES.index(label)
            labels[int(start) : int(end) + 1] = idx

        out = output_dir / f"{video_name}_labels.npz"
        np.savez(out, labels=labels, classes=CLASSES)
        print(f"  {video_name}: {max_sec}s → {out.name}")
        for i, cls in enumerate(CLASSES):
            n = (labels == i).sum()
            print(f"    {cls:12s}: {n}s ({100*n/max_sec:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path, help="Label Studio export JSON")
    parser.add_argument("--output", type=Path, default=Path("data/labels"))
    args = parser.parse_args()
    parse_ls_export(args.export, args.output)


if __name__ == "__main__":
    main()
