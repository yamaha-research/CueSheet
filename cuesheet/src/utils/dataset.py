"""PyTorch Dataset and LightningDataModule for concert embeddings + labels."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from src.utils.config import ConcertConfig

Mode = Literal["supervised", "ssl", "pseudo"]


class ConcertWindowDataset(Dataset):
    """Per-second embedding windows with optional labels.

    Loads precomputed .npz embedding files from multiple encoders and
    optional per-second label arrays.

    In "ssl" mode, returns (anchor, positive) pairs for contrastive training.
    """

    def __init__(
        self,
        feature_dir: Path,
        label_dir: Path | None,
        config: ConcertConfig,
        mode: Mode = "supervised",
        pseudo_label_dir: Path | None = None,
        beat_label_dir: Path | None = None,
    ):
        self.config = config
        self.mode = mode
        self.positive_radius = config.ssl.positive_radius

        self._windows: list[dict] = []  # each entry: {emb, label, timestamp, show_id, total_sec}
        self._show_ranges: dict[int, list[int]] = {}  # show_id → sorted list of global indices
        self._load_data(feature_dir, label_dir, pseudo_label_dir, beat_label_dir)

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_data(self, feature_dir: Path, label_dir: Path | None,
                   pseudo_label_dir: Path | None,
                   beat_label_dir: Path | None = None) -> None:
        encoders = self.config.fusion.active_encoders

        # Collect all show stems from first encoder's files
        stem_to_paths: dict[str, dict[str, Path]] = {}
        for enc in encoders:
            for f in sorted(feature_dir.glob(f"*_{enc}.npz")):
                stem = f.name.replace(f"_{enc}.npz", "")
                stem_to_paths.setdefault(stem, {})[enc] = f

        for show_id, (stem, paths) in enumerate(stem_to_paths.items()):
            if len(paths) != len(encoders):
                print(f"  WARNING: {stem} missing encoders {set(encoders) - set(paths)}, skipping")
                continue

            # Load and concatenate embeddings
            embs, timestamps = [], None
            for enc in encoders:
                data = np.load(paths[enc])
                embs.append(data["embeddings"].astype(np.float32))
                if timestamps is None:
                    timestamps = data["timestamps"]
            embeddings = np.concatenate(embs, axis=-1)  # (T, D)
            T = len(embeddings)
            total_sec = float(timestamps[-1]) + 1.0

            # Load labels
            labels = np.full(T, -1, dtype=np.int64)
            label_source = None
            if pseudo_label_dir:
                label_source = pseudo_label_dir / f"{stem}_labels.npz"
            elif label_dir:
                label_source = label_dir / f"{stem}_labels.npz"

            if label_source and label_source.exists():
                ldata = np.load(label_source)
                raw = ldata["labels"]
                n = min(T, len(raw))
                labels[:n] = raw[:n]

            beat = np.full(T, -1.0, dtype=np.float32)
            if beat_label_dir:
                beat_path = beat_label_dir / f"{stem}_beats.npz"
                if beat_path.exists():
                    bdata = np.load(beat_path)
                    raw_beat = bdata["music_active"].astype(np.float32)
                    n = min(T, len(raw_beat))
                    beat[:n] = raw_beat[:n]

            start_idx = len(self._windows)
            for t in range(T):
                self._windows.append({
                    "emb":       embeddings[t],
                    "label":     int(labels[t]),
                    "beat":      float(beat[t]),
                    "timestamp": float(timestamps[t]),
                    "show_id":   show_id,
                    "total_sec": total_sec,
                    "global_idx": len(self._windows),
                })
            self._show_ranges[show_id] = list(range(start_idx, start_idx + T))

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        w = self._windows[idx]
        rel_pos = w["timestamp"] / w["total_sec"]
        emb = torch.from_numpy(w["emb"])

        if self.mode == "ssl":
            pos_idx = self._sample_positive(idx)
            pos_w = self._windows[pos_idx]
            return {
                "anchor":   emb,
                "positive": torch.from_numpy(pos_w["emb"]),
            }

        return {
            "features":    emb,
            "label":       torch.tensor(w["label"], dtype=torch.long),
            "beat_target": torch.tensor(w["beat"], dtype=torch.float32),
            "rel_pos":     torch.tensor(rel_pos, dtype=torch.float32),
            "timestamp":   torch.tensor(w["timestamp"], dtype=torch.float32),
            "show_id":     torch.tensor(w["show_id"], dtype=torch.long),
        }

    def _sample_positive(self, idx: int) -> int:
        show = self._windows[idx]["show_id"]
        show_idxs = self._show_ranges[show]
        # Binary-search to locate idx within show_idxs, then slice ±radius
        pos = show_idxs.index(idx)
        r = self.positive_radius
        lo, hi = max(0, pos - r), min(len(show_idxs), pos + r + 1)
        candidates = [show_idxs[i] for i in range(lo, hi) if show_idxs[i] != idx]
        return candidates[np.random.randint(len(candidates))] if candidates else idx

    # ── split helpers ─────────────────────────────────────────────────────────

    def temporal_split(self, val_fraction: float = 0.2) -> tuple["ConcertWindowDataset", "ConcertWindowDataset"]:
        """Split by time within each show (last val_fraction% → validation)."""
        from torch.utils.data import Subset

        train_idx, val_idx = [], []
        show_ids = sorted({w["show_id"] for w in self._windows})
        for sid in show_ids:
            idxs = [i for i, w in enumerate(self._windows) if w["show_id"] == sid]
            cut = int(len(idxs) * (1 - val_fraction))
            train_idx.extend(idxs[:cut])
            val_idx.extend(idxs[cut:])

        train_ds = Subset(self, train_idx)
        val_ds   = Subset(self, val_idx)
        return train_ds, val_ds

    def all_labels(self) -> np.ndarray:
        """Return all per-window label integers as a 1-D array."""
        return np.array([w["label"] for w in self._windows], dtype=np.int64)


class ConcertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ConcertConfig,
        features_dir: Path,
        labels_dir: Path | None = None,
        pseudo_label_dir: Path | None = None,
        beat_label_dir: Path | None = None,
        mode: Mode = "supervised",
        val_fraction: float = 0.2,
    ):
        super().__init__()
        self.config = config
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.pseudo_label_dir = pseudo_label_dir
        self.beat_label_dir = beat_label_dir
        self.mode = mode
        self.val_fraction = val_fraction
        self._train_ds = None
        self._val_ds = None

    def setup(self, stage: str = "") -> None:
        full = ConcertWindowDataset(
            self.features_dir, self.labels_dir, self.config,
            mode=self.mode, pseudo_label_dir=self.pseudo_label_dir,
            beat_label_dir=self.beat_label_dir,
        )
        self._train_ds, self._val_ds = full.temporal_split(self.val_fraction)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.config.train.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def get_ssl_dataloader(self) -> DataLoader:
        ssl_ds = ConcertWindowDataset(
            self.features_dir, None, self.config, mode="ssl"
        )
        return DataLoader(ssl_ds, batch_size=self.config.train.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)

    def get_full_dataset(self) -> ConcertWindowDataset:
        return ConcertWindowDataset(
            self.features_dir, self.labels_dir, self.config, mode=self.mode
        )
