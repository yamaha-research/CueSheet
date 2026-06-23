"""Load and expose the YAML config as typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ENCODER_DIMS: dict[str, int] = {
    "ast":      768,
    "pann":     2048,
    "clap":     512,
    "w2v2":     768,
    "hubert":   768,
    "mert":     768,
}

STAGE1_CLASSES = ["Performance", "MC_Talk", "Other"]
STAGE2_TO_STAGE1 = {
    "Performance":  0,
    "MC_Talk":      1,
    "Pre_Concert":  2,
    "Applause":     2,
    "Intermission": 2,
    "Ambient":      2,
}


@dataclass
class AudioConfig:
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    sample_rate: int = 16000
    window_sec: float = 2.0
    hop_sec: float = 1.0


@dataclass
class FusionConfig:
    hidden_dim: int = 256
    dropout: float = 0.2
    projected_dim: int = 256
    classes: list[str] = field(default_factory=lambda: [
        "Pre_Concert", "Performance", "MC_Talk", "Applause", "Intermission", "Ambient"
    ])
    multilabel_pairs: list[list[str]] = field(default_factory=lambda: [])
    active_encoders: list[str] = field(default_factory=lambda: ["ast", "pann", "hubert", "mert"])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def concat_dim(self) -> int:
        return self.projected_dim * len(self.active_encoders)

    @property
    def stage2_input_dim(self) -> int:
        # fused(concat_dim) + stage1_probs(3) + rel_pos(1)
        return self.concat_dim + 3 + 1


@dataclass
class TemporalConfig:
    type: str = "hmm"
    min_segment_sec: int = 10


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 30
    val_split_by: str = "show"
    stage1_loss_weight: float = 0.3
    stage2_loss_weight: float = 0.7
    beat_loss_weight: float = 0.1


@dataclass
class SSLConfig:
    temperature: float = 0.07
    positive_radius: int = 3
    ssl_epochs: int = 50
    proj_dim: int = 128
    lr: float = 1e-4


@dataclass
class PseudoConfig:
    n_iterations: int = 3
    confidence_thresholds: list[float] = field(default_factory=lambda: [0.95, 0.90, 0.85])
    min_segment_sec: int = 10
    max_pseudo_ratio: float = 0.7
    epochs_per_iter: list[int] = field(default_factory=lambda: [10, 8, 6])


@dataclass
class ConcertConfig:
    project: str = "cuesheet"
    seed: int = 42
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    pseudo: PseudoConfig = field(default_factory=PseudoConfig)


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_config(d: dict) -> ConcertConfig:
    cfg = ConcertConfig()
    if "audio" in d:
        cfg.audio = AudioConfig(**{k: v for k, v in d["audio"].items() if hasattr(AudioConfig, k)})
    if "fusion" in d:
        cfg.fusion = FusionConfig(**{k: v for k, v in d["fusion"].items() if k in FusionConfig.__dataclass_fields__})
    if "temporal" in d:
        cfg.temporal = TemporalConfig(**d["temporal"])
    if "train" in d:
        cfg.train = TrainConfig(**{k: v for k, v in d["train"].items() if k in TrainConfig.__dataclass_fields__})
    if "ssl" in d:
        cfg.ssl = SSLConfig(**d["ssl"])
    if "pseudo" in d:
        cfg.pseudo = PseudoConfig(**{k: v for k, v in d["pseudo"].items() if k in PseudoConfig.__dataclass_fields__})
    for k in ("project", "seed"):
        if k in d:
            setattr(cfg, k, d[k])
    return cfg


def load_config(path: str | Path | None = None) -> ConcertConfig:
    defaults_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    with open(defaults_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    if path is not None:
        with open(path) as f:
            override = yaml.safe_load(f)
        _deep_update(data, override)
    return _dict_to_config(data)


def detect_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path: Path, model: "torch.nn.Module", device: str) -> "torch.nn.Module":
    """Load a Lightning checkpoint into a bare nn.Module.

    Handles two wrapper conventions:
      * ConcertLightningModule saves keys as 'model.<...>'
      * ContrastiveLightningModule saves keys as 'fusion_model.<...>'
    """
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("fusion_model."):
            cleaned[k[len("fusion_model."):]] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if cleaned:
        print(f"  loaded {len(cleaned)} tensors from {path.name} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def get_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels[labels >= 0], minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def map_to_stage1(labels: np.ndarray | torch.Tensor, classes: list[str]) -> np.ndarray | torch.Tensor:
    mapping = [STAGE2_TO_STAGE1.get(c, 2) for c in classes]
    if isinstance(labels, torch.Tensor):
        lut = torch.tensor(mapping, dtype=torch.long, device=labels.device)
        mask = labels >= 0
        out = torch.full_like(labels, -1)
        out[mask] = lut[labels[mask]]
        return out
    lut = np.array(mapping, dtype=np.int64)
    out = np.full_like(labels, -1)
    mask = labels >= 0
    out[mask] = lut[labels[mask]]
    return out
