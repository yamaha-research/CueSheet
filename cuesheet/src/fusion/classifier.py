"""Offline and Online classifiers + PyTorch Lightning training module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.fusion.model import FusionModel
from src.fusion.temporal import HMMSmoother, CRFSmoother
from src.utils.config import ConcertConfig, map_to_stage1
from src.utils.metrics import compute_metrics, merge_to_segments


@dataclass
class ClassifierOutput:
    labels: np.ndarray          # (T,) integer class indices
    probabilities: np.ndarray   # (T, num_classes) softmax probs
    stage1_labels: np.ndarray   # (T,) 3-class
    timestamps: np.ndarray      # (T,) seconds


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_encoder_npz(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate embeddings from multiple encoder .npz files.
    Returns (embeddings: (T, D), timestamps: (T,)).
    """
    embs, ts = [], None
    for enc_name, path in paths.items():
        data = np.load(path)
        embs.append(data["embeddings"].astype(np.float32))
        if ts is None:
            ts = data["timestamps"]
    return np.concatenate(embs, axis=-1), ts


def _rel_pos(timestamps: np.ndarray, total: float | None = None) -> np.ndarray:
    t = total if total else float(timestamps[-1]) + 1.0
    return (timestamps / t).astype(np.float32)


# ── offline classifier ────────────────────────────────────────────────────────

class OfflineClassifier:
    def __init__(self, model: FusionModel, smoother: HMMSmoother | CRFSmoother,
                 config: ConcertConfig, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.smoother = smoother
        self.config = config
        self.device = device

    def classify_array(self, embeddings: np.ndarray, timestamps: np.ndarray,
                       chunk_size: int = 512) -> ClassifierOutput:
        T = len(embeddings)
        rp = _rel_pos(timestamps)
        all_s1, all_s2 = [], []

        with torch.no_grad():
            for start in range(0, T, chunk_size):
                emb_t = torch.from_numpy(embeddings[start:start + chunk_size]).to(self.device)
                rp_t  = torch.from_numpy(rp[start:start + chunk_size]).to(self.device)
                out = self.model(emb_t, rp_t)
                all_s1.append(out["stage1_logits"].cpu())
                all_s2.append(out["stage2_logits"].cpu())

        s1_logits = torch.cat(all_s1, dim=0)
        s2_logits = torch.cat(all_s2, dim=0)
        probs = F.softmax(s2_logits, dim=-1).numpy()

        if isinstance(self.smoother, HMMSmoother):
            labels = self.smoother.decode_viterbi(probs)
        else:
            labels = np.array(self.smoother.decode(s2_logits.unsqueeze(0))[0])

        stage1_labels = F.softmax(s1_logits, dim=-1).argmax(dim=-1).numpy()

        return ClassifierOutput(labels=labels, probabilities=probs,
                                stage1_labels=stage1_labels, timestamps=timestamps)

    def classify_file(self, feature_npz_paths: dict[str, Path],
                      chunk_size: int = 512) -> ClassifierOutput:
        embeddings, timestamps = _load_encoder_npz(feature_npz_paths)
        return self.classify_array(embeddings, timestamps, chunk_size)


# ── online classifier ─────────────────────────────────────────────────────────

class OnlineClassifier:
    def __init__(self, model: FusionModel, hmm: HMMSmoother, config: ConcertConfig,
                 device: str = "cpu", total_steps_estimate: int | None = None):
        assert model.mode == "online", "OnlineClassifier requires FusionModel(mode='online')"
        self.model = model.to(device).eval()
        self.hmm = hmm
        self.config = config
        self.device = device
        self.total_steps_estimate = total_steps_estimate
        self._hmm_alpha: np.ndarray = hmm.init_alpha()
        self._lstm_hidden: Any = None
        self._step_count: int = 0

    def reset(self, total_steps_estimate: int | None = None) -> None:
        self._hmm_alpha = self.hmm.init_alpha()
        self._lstm_hidden = None
        self._step_count = 0
        if total_steps_estimate is not None:
            self.total_steps_estimate = total_steps_estimate

    def step(self, embedding: np.ndarray) -> tuple[int, dict]:
        """Process one 1-second window. embedding: (raw_concat_dim,)"""
        total = self.total_steps_estimate or max(self._step_count + 1, 1)
        rel_pos = self._step_count / total

        emb_t = torch.from_numpy(embedding[np.newaxis].astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits, self._lstm_hidden = self.model.forward_step(emb_t, rel_pos, self._lstm_hidden)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        self._hmm_alpha, label = self.hmm.forward_step(self._hmm_alpha, probs)
        self._step_count += 1

        return label, {
            "probabilities": probs,
            "hmm_alpha": self._hmm_alpha.copy(),
            "step": self._step_count,
            "label_name": self.config.fusion.classes[label],
        }

    def get_state(self) -> dict:
        state: dict[str, Any] = {
            "hmm_alpha": self._hmm_alpha.tolist(),
            "step_count": self._step_count,
            "total_steps_estimate": self.total_steps_estimate,
        }
        if self._lstm_hidden is not None:
            h, c = self._lstm_hidden
            state["lstm_h"] = h.cpu().numpy().tolist()
            state["lstm_c"] = c.cpu().numpy().tolist()
        return state

    def load_state(self, state: dict) -> None:
        self._hmm_alpha = np.array(state["hmm_alpha"])
        self._step_count = state["step_count"]
        self.total_steps_estimate = state.get("total_steps_estimate")
        if "lstm_h" in state:
            h = torch.tensor(state["lstm_h"], device=self.device)
            c = torch.tensor(state["lstm_c"], device=self.device)
            self._lstm_hidden = (h, c)


# ── lightning module ──────────────────────────────────────────────────────────

class ConcertLightningModule(pl.LightningModule):
    def __init__(self, model: FusionModel, config: ConcertConfig,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.config = config
        self.s1_weight = config.train.stage1_loss_weight
        self.s2_weight = config.train.stage2_loss_weight
        self.beat_weight = config.train.beat_loss_weight
        w = class_weights
        self.ce_s1 = nn.CrossEntropyLoss(ignore_index=-1)
        self.ce_s2 = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
        self._val_preds: list[np.ndarray] = []
        self._val_trues: list[np.ndarray] = []

    def forward(self, embeddings, rel_pos):
        return self.model(embeddings, rel_pos)

    def _shared_step(self, batch: dict) -> dict[str, torch.Tensor]:
        emb = batch["features"]
        rp  = batch["rel_pos"]
        labels_s2 = batch["label"]
        labels_s1 = map_to_stage1(labels_s2, self.config.fusion.classes)

        out = self.model(emb, rp)
        loss_s1 = self.ce_s1(out["stage1_logits"], labels_s1)
        loss_s2 = self.ce_s2(out["stage2_logits"], labels_s2)

        loss_beat = torch.zeros((), device=emb.device)
        if "beat_target" in batch and self.beat_weight > 0:
            target = batch["beat_target"]
            mask = target >= 0
            if mask.any():
                loss_beat = F.mse_loss(out["beat_pred"][mask], target[mask])

        loss = (self.s1_weight * loss_s1
                + self.s2_weight * loss_s2
                + self.beat_weight * loss_beat)
        return {"loss": loss, "loss_s1": loss_s1, "loss_s2": loss_s2, "loss_beat": loss_beat,
                "logits": out["stage2_logits"], "labels": labels_s2}

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._shared_step(batch)
        self.log_dict({"train/loss": out["loss"], "train/s1_loss": out["loss_s1"],
                       "train/s2_loss": out["loss_s2"], "train/beat_loss": out["loss_beat"]},
                      prog_bar=True, on_step=True, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self._shared_step(batch)
        self.log("val/loss", out["loss"], prog_bar=True, on_epoch=True)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        trues = out["labels"].cpu().numpy()
        self._val_preds.append(preds)
        self._val_trues.append(trues)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        pred = np.concatenate(self._val_preds)
        true = np.concatenate(self._val_trues)
        metrics = compute_metrics(pred, true, self.config.fusion.classes)
        self.log("val/f1_macro", metrics.get("f1_macro", 0.0), prog_bar=True)
        for cls, f1 in metrics.get("f1_per_class", {}).items():
            self.log(f"val/f1_{cls}", f1)
        self.log("val/seg_iou", metrics.get("segment_iou", 0.0))
        self.log("val/bde_mean", metrics.get("boundary_error_mean", 0.0))
        self._val_preds.clear()
        self._val_trues.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.train.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config.train.epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
