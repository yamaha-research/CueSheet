"""Fusion model: encoder projections, Stage 1 MLP, Stage 2 LSTM/Transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import ConcertConfig


class EncoderProjection(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiEncoderFusion(nn.Module):
    """Projects each encoder embedding to proj_dim, then concatenates."""

    def __init__(self, encoder_dims: dict[str, int], proj_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder_names = list(encoder_dims.keys())
        self.proj_dim = proj_dim
        self.projections = nn.ModuleDict({
            name: EncoderProjection(dim, proj_dim, dropout)
            for name, dim in encoder_dims.items()
        })
        self.concat_dim = proj_dim * len(encoder_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, sum_of_raw_dims) — concatenated raw embeddings in encoder order."""
        parts = torch.split(x, [self.projections[n].net[0].in_features for n in self.encoder_names], dim=-1)
        projected = [self.projections[name](part) for name, part in zip(self.encoder_names, parts)]
        return torch.cat(projected, dim=-1)  # (B, concat_dim)


class Stage1Classifier(nn.Module):
    """Local per-second classifier: Performance / MC_Talk / Other."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 3)


class Stage2LSTM(nn.Module):
    """Causal LSTM for online temporal classification."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int = 5,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, hidden=None):
        """Batch forward: x (B, T, D) → logits (B, T, n_classes), hidden."""
        out, hidden = self.lstm(x, hidden)
        return self.head(out), hidden

    def forward_step(self, x: torch.Tensor, hidden=None):
        """Single step: x (1, 1, D) → logits (1, n_classes), hidden."""
        out, hidden = self.lstm(x, hidden)
        return self.head(out.squeeze(1)), hidden


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 12000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class Stage2Transformer(nn.Module):
    """Bidirectional Transformer for offline temporal classification."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int = 5,
                 nhead: int = 4, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = _PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D) → logits (B, T, n_classes)."""
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(x)


class FusionModel(nn.Module):
    """Top-level model combining encoder fusion, Stage 1, and Stage 2."""

    def __init__(self, config: ConcertConfig, mode: str = "offline"):
        super().__init__()
        assert mode in ("offline", "online")
        self.mode = mode
        self.config = config

        from src.utils.config import ENCODER_DIMS
        encoder_dims = {enc: ENCODER_DIMS[enc] for enc in config.fusion.active_encoders}
        proj_dim = config.fusion.projected_dim
        hidden_dim = config.fusion.hidden_dim
        dropout = config.fusion.dropout
        n5 = config.fusion.num_classes
        s2_in = config.fusion.stage2_input_dim  # concat_dim + 3 + 1

        self.fusion = MultiEncoderFusion(encoder_dims, proj_dim, dropout)
        self.stage1 = Stage1Classifier(self.fusion.concat_dim, hidden_dim, n_classes=3, dropout=dropout)
        self.beat_head = nn.Sequential(
            nn.Linear(self.fusion.concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        if mode == "online":
            self.stage2 = Stage2LSTM(s2_in, hidden_dim, n_classes=n5, dropout=dropout)
        else:
            self.stage2 = Stage2Transformer(s2_in, hidden_dim, n_classes=n5, dropout=dropout)

    def _build_stage2_input(self, fused: torch.Tensor, stage1_logits: torch.Tensor,
                             rel_pos: torch.Tensor) -> torch.Tensor:
        stage1_probs = F.softmax(stage1_logits, dim=-1)
        if rel_pos.dim() == 1:
            rel_pos = rel_pos.unsqueeze(-1)
        return torch.cat([fused, stage1_probs, rel_pos], dim=-1)

    def forward(self, embeddings: torch.Tensor, rel_pos: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        embeddings: (B, raw_concat_dim) or (B, T, raw_concat_dim)
        rel_pos:    (B,) or (B, T)
        """
        batch_seq = embeddings.dim() == 3
        if batch_seq:
            B, T, D = embeddings.shape
            emb_flat = embeddings.reshape(B * T, D)
            rp_flat  = rel_pos.reshape(B * T)
        else:
            emb_flat = embeddings
            rp_flat  = rel_pos

        fused = self.fusion(emb_flat)                   # (B[*T], concat_dim)
        s1_logits = self.stage1(fused)                  # (B[*T], 3)
        beat_pred = self.beat_head(fused).squeeze(-1)   # (B[*T],)
        s2_in = self._build_stage2_input(fused, s1_logits, rp_flat)  # (B[*T], s2_in)

        if batch_seq:
            s2_in = s2_in.reshape(B, T, -1)
            fused = fused.reshape(B, T, -1)
            s1_logits = s1_logits.reshape(B, T, -1)
            beat_pred = beat_pred.reshape(B, T)
            if self.mode == "online":
                s2_logits, _ = self.stage2(s2_in)
            else:
                s2_logits = self.stage2(s2_in)
        else:
            if self.mode == "online":
                s2_logits, _ = self.stage2(s2_in.unsqueeze(1))
                s2_logits = s2_logits.squeeze(1)
            else:
                s2_logits = self.stage2(s2_in.unsqueeze(1)).squeeze(1)

        return {"stage1_logits": s1_logits, "stage2_logits": s2_logits,
                "beat_pred": beat_pred, "fused": fused}

    def forward_step(self, embedding: torch.Tensor, rel_pos: float,
                     lstm_hidden: Any = None) -> tuple[torch.Tensor, Any]:
        """Online single-step. embedding: (1, raw_concat_dim)."""
        assert self.mode == "online", "forward_step only available in online mode"
        fused = self.fusion(embedding)
        s1_logits = self.stage1(fused)
        s2_in = self._build_stage2_input(
            fused, s1_logits,
            torch.tensor([[rel_pos]], device=embedding.device, dtype=torch.float32)
        )
        logits, new_hidden = self.stage2.forward_step(s2_in.unsqueeze(1), lstm_hidden)
        return logits, new_hidden
