"""Online causal Viterbi over the 6 concert segments.

Forward-only: at each tick we update log alpha using only past observations
and emit the current argmax. No backtrace, no future context, no heuristics.
This is the streaming counterpart of the offline batch Viterbi in
`src/fusion/temporal.HMMSmoother.decode_viterbi`; their outputs agree where
the offline decision is unaffected by future evidence and may differ at
transitions.

The transition matrix is the same one the offline pipeline uses,
`pann_baseline.build_transition_matrix(CLASSES)`, so the only structural
difference is "no future" vs "full sequence".
"""
from __future__ import annotations

import numpy as np


class OnlineCausalHMM:
    """One forward Viterbi step per observation."""

    def __init__(self, transition_matrix: np.ndarray, prior: np.ndarray | None = None):
        T = np.asarray(transition_matrix, dtype=np.float64)
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError(f"transition_matrix must be square, got {T.shape}")
        self.K = T.shape[0]
        self.log_T = np.log(T + 1e-12)
        if prior is None:
            self.log_alpha = -np.log(self.K) * np.ones(self.K)
        else:
            self.log_alpha = np.log(np.asarray(prior, dtype=np.float64) + 1e-12)

    def step(self, posterior: np.ndarray) -> int:
        """Update with one (K,) posterior; return the argmax state."""
        if posterior.shape != (self.K,):
            raise ValueError(
                f"posterior must have shape ({self.K},), got {posterior.shape}")
        log_obs = np.log(np.asarray(posterior, dtype=np.float64) + 1e-12)
        # scores[i, j] = log alpha[i] + log T[i, j]; max over i is the
        # standard forward Viterbi update.
        scores = self.log_alpha[:, None] + self.log_T
        self.log_alpha = scores.max(axis=0) + log_obs
        # Renormalize to keep numbers in range without changing the argmax.
        self.log_alpha -= self.log_alpha.max()
        return int(self.log_alpha.argmax())
