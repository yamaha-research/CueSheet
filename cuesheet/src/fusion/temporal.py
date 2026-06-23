"""Temporal smoothers: HMM (causal + Viterbi) and CRF."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class HMMSmoother:
    """Wraps a simple HMM for post-processing per-second class posteriors.

    Offline: full-sequence Viterbi decoding.
    Online:  causal forward algorithm, one step at a time.

    The default transition matrix has a uniform self-transition prior
    (stay = 0.95). Callers that need to model "short, easy-to-enter"
    classes — like Applause sitting between long Performance runs —
    can pass a custom `transition_matrix` whose rows sum to 1.
    """

    def __init__(self, num_states: int = 5, min_segment_sec: int = 10,
                 transition_matrix: np.ndarray | None = None):
        self.num_states = num_states
        self.min_segment_sec = min_segment_sec

        if transition_matrix is None:
            # Strong self-transition prior — segments last at least ~10s
            stay = 0.95
            leave = (1 - stay) / (num_states - 1)
            self.transition = np.full((num_states, num_states), leave)
            np.fill_diagonal(self.transition, stay)
        else:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            assert tm.shape == (num_states, num_states), (
                f"transition_matrix must be ({num_states}, {num_states}), "
                f"got {tm.shape}")
            row_sums = tm.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-3), (
                f"transition_matrix rows must sum to 1, got {row_sums}")
            self.transition = tm

        # Uniform start probability
        self.startprob = np.ones(num_states) / num_states

        # Near-identity emission (posterior IS the emission probability)
        # We use the model softmax directly as emission, so no separate matrix needed.

    # ── offline ───────────────────────────────────────────────────────────────

    def decode_viterbi(self, posteriors: np.ndarray) -> np.ndarray:
        """Viterbi decode a (T, num_states) posterior matrix."""
        T, K = posteriors.shape
        log_trans = np.log(self.transition + 1e-12)
        log_emit  = np.log(posteriors + 1e-12)

        viterbi = np.full((T, K), -np.inf)
        backptr = np.zeros((T, K), dtype=int)

        viterbi[0] = np.log(self.startprob + 1e-12) + log_emit[0]

        for t in range(1, T):
            scores = viterbi[t - 1, :, np.newaxis] + log_trans  # (K_from, K_to)
            backptr[t] = scores.argmax(axis=0)
            viterbi[t] = scores.max(axis=0) + log_emit[t]

        # backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(viterbi[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        return self._enforce_min_segment(path)

    def _enforce_min_segment(self, path: np.ndarray) -> np.ndarray:
        """Merge segments shorter than min_segment_sec into neighbours."""
        if self.min_segment_sec <= 1:
            return path
        path = path.copy()
        T = len(path)
        i = 0
        while i < T:
            j = i
            while j < T and path[j] == path[i]:
                j += 1
            seg_len = j - i
            if seg_len < self.min_segment_sec and i > 0:
                path[i:j] = path[i - 1]
            i = j
        return path


    # ── online ────────────────────────────────────────────────────────────────

    def init_alpha(self) -> np.ndarray:
        return self.startprob.copy()

    def forward_step(self, alpha: np.ndarray, posterior: np.ndarray) -> tuple[np.ndarray, int]:
        """One causal HMM forward step.

        alpha:     (K,) current forward variable
        posterior: (K,) softmax probabilities from model
        Returns:   (alpha_new, predicted_label)
        """
        alpha_new = (self.transition.T @ alpha) * (posterior + 1e-12)
        norm = alpha_new.sum()
        alpha_new = alpha_new / (norm + 1e-12)
        return alpha_new, int(np.argmax(alpha_new))

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit_transitions(self, label_sequences: list[np.ndarray]) -> None:
        """Estimate transition matrix from labeled sequences via counting."""
        counts = np.zeros((self.num_states, self.num_states)) + 1e-6  # Laplace
        for seq in label_sequences:
            for t in range(len(seq) - 1):
                if seq[t] >= 0 and seq[t + 1] >= 0:
                    counts[seq[t], seq[t + 1]] += 1
        self.transition = counts / counts.sum(axis=1, keepdims=True)


def enforce_min_segment_per_class(path: np.ndarray,
                                   min_seg_per_class: dict[int, int],
                                   default: int = 10) -> np.ndarray:
    """Drop-in replacement for HMMSmoother._enforce_min_segment when some
    classes (e.g. Applause) are intrinsically shorter than the default
    min-segment threshold."""
    out = np.asarray(path).copy()
    T = len(out)
    i = 0
    while i < T:
        j = i
        while j < T and out[j] == out[i]:
            j += 1
        seg_len = j - i
        cls = int(out[i])
        min_sec = min_seg_per_class.get(cls, default)
        if seg_len < min_sec and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


class CRFSmoother(nn.Module):
    """Linear-chain CRF via torchcrf — offline only."""

    def __init__(self, num_tags: int = 5):
        super().__init__()
        try:
            from torchcrf import CRF
            self.crf = CRF(num_tags, batch_first=True)
        except ImportError:
            raise ImportError("Install pytorch-crf: uv add pytorch-crf")
        self.num_tags = num_tags

    def compute_loss(self, emissions: torch.Tensor, tags: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """emissions: (B, T, num_tags), tags: (B, T) → scalar loss."""
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, emissions: torch.Tensor,
               mask: torch.Tensor | None = None) -> list[list[int]]:
        """Viterbi decode: (B, T, num_tags) → list of label sequences."""
        return self.crf.decode(emissions, mask=mask)
