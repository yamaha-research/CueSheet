"""Per-second streaming audio pipeline: encoder -> 6-class -> online HMM.

Stateful. Push raw 32 kHz mono float32 samples as they arrive; every time a
new 1 second hop completes (and after the 10 second buffer is first full)
an Emission is returned.

No live capture here, only the streaming logic. `live/sources.py` provides
file replay and microphone capture that hand samples to this pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cuesheet"))
from scripts.bootstrap_labels import build_group_indices, load_audioset_index  # noqa: E402
from scripts.pann_baseline import (  # noqa: E402
    build_transition_matrix, map_groups_to_classes,
)

from live.audio_encoder import (  # noqa: E402
    AudioEncoder, HOP_SAMPLES, SAMPLE_RATE, WIN_SAMPLES,
)
from live.online_hmm import OnlineCausalHMM  # noqa: E402

CLASSES = ["Pre_Concert", "Performance", "MC_Talk", "Applause",
           "Intermission", "Ambient"]
AMBIENT_IDX = CLASSES.index("Ambient")

# Energy-gate baseline threshold (dBFS). Below this RMS the encoder is
# bypassed and the window is declared Ambient. Noise floors differ widely
# from room to room, so a single fixed number cannot fit every venue.
# AudioPipeline therefore runs a rolling noise-floor estimator: the first
# ADAPT_CALIB_SECONDS of audio are used as a baseline, and the threshold
# tracks (p20 of recent RMS) + ADAPT_MARGIN_DBFS thereafter.
DEFAULT_SILENCE_DBFS = -45.0      # fallback when adaptation is disabled
ADAPT_CALIB_SECONDS = 5           # initial calibration window
ADAPT_HISTORY_SECONDS = 60        # rolling window for the noise-floor p20
ADAPT_MARGIN_DBFS = 6.0           # threshold = noise_floor + margin


def _window_dbfs(window: np.ndarray) -> float:
    """RMS of a Float32 [-1, 1] window in dBFS."""
    rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)) + 1e-12)
    return 20.0 * np.log10(rms)


@dataclass
class Emission:
    t_sec: float            # window-start second, matches the offline timestamp
    post6: np.ndarray       # (6,) float32 — 6-class posterior after group map
    hmm_label: int          # online causal Viterbi argmax over the 6 classes
    rms_dbfs: float = 0.0          # input window RMS (diagnostic)
    silence_gated: bool = False    # encoder bypassed for this window
    audioset_top: list = None      # [{name: str, p: float}] top-K AudioSet raw
    silence_threshold_dbfs: float = 0.0   # effective dBFS threshold used this window

    def __post_init__(self):
        if self.audioset_top is None:
            self.audioset_top = []


def _load_audioset_names() -> list[str]:
    """Read AudioSet display names (column 2) from the panns_data CSV.

    Returns a 527-long list. Falls back to numeric ids if the CSV is
    absent — the pipeline still runs, only the on-screen panel is less
    descriptive.
    """
    import csv
    from pathlib import Path as _Path
    p = _Path.home() / "panns_data" / "class_labels_indices.csv"
    names = [f"AS-{i}" for i in range(527)]
    if not p.is_file():
        return names
    with open(p) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                idx = int(row[0])
                if 0 <= idx < 527:
                    names[idx] = row[2]
    return names


class AudioPipeline:
    def __init__(self, device: str = "cpu", model_name: str = "mn10_as",
                 silence_dbfs: float | None = DEFAULT_SILENCE_DBFS,
                 audioset_top_k: int = 5, adaptive_silence: bool = True):
        import os
        env = os.environ.get("CUESHEET_SILENCE_DBFS")
        if env is not None:
            if env.lower() in ("off", "none", "disable", "disabled"):
                silence_dbfs = None
                adaptive_silence = False
            elif env.lower() in ("auto", "adapt", "adaptive"):
                adaptive_silence = True
            else:
                try:
                    silence_dbfs = float(env)
                    adaptive_silence = False
                except ValueError:
                    pass
        # silence_dbfs is the *baseline* threshold. When adaptive_silence
        # is True, the effective threshold equals the rolling p20 of past
        # window-RMS + ADAPT_MARGIN_DBFS; the baseline is used during
        # the first ADAPT_CALIB_SECONDS as a safety floor and as the
        # fallback if no history has accumulated yet.
        self.silence_dbfs = silence_dbfs
        self.adaptive_silence = adaptive_silence
        self._rms_history: list[float] = []
        self.audioset_top_k = audioset_top_k
        self._audioset_names = _load_audioset_names()
        self.encoder = AudioEncoder(device=device, model_name=model_name)
        self.group_idx = build_group_indices(load_audioset_index())
        self.hmm = OnlineCausalHMM(build_transition_matrix(CLASSES))
        self._queue = np.empty(0, dtype=np.float32)
        self._emitted = 0

    def _adaptive_threshold(self) -> float | None:
        """Return the current effective silence threshold in dBFS, or
        None if the user has disabled silence-gating entirely.

        Behavior during the initial calibration window: use the static
        baseline (silence_dbfs). After that, return the rolling 20th
        percentile of recent RMS history plus ADAPT_MARGIN_DBFS, with
        the static baseline as a soft cap so a noisy room cannot drift
        the threshold up to silence the real signal.
        """
        if self.silence_dbfs is None:
            return None
        if not self.adaptive_silence:
            return self.silence_dbfs
        # Calibration: lean on the static baseline while we gather data
        if len(self._rms_history) < ADAPT_CALIB_SECONDS:
            return self.silence_dbfs
        keep = ADAPT_HISTORY_SECONDS
        hist = np.asarray(self._rms_history[-keep:], dtype=np.float64)
        # p20 resists occasional speech / applause spikes in the
        # window — it tracks the quiet floor without being pulled by
        # transient loud events.
        floor = float(np.percentile(hist, 20))
        return floor + ADAPT_MARGIN_DBFS

    @property
    def cold_start_seconds(self) -> float:
        """How many seconds of audio must arrive before the first Emission."""
        return WIN_SAMPLES / SAMPLE_RATE

    def push(self, samples: np.ndarray) -> list[Emission]:
        """Append samples (32 kHz mono float32, any length). Emit any
        predictions whose 10 second window now sits inside the buffer."""
        if samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got shape {samples.shape}")
        s = samples.astype(np.float32, copy=False)
        self._queue = (s if self._queue.size == 0
                       else np.concatenate([self._queue, s]))
        out: list[Emission] = []
        while len(self._queue) >= WIN_SAMPLES:
            window = self._queue[:WIN_SAMPLES]
            dbfs = _window_dbfs(window)
            self._rms_history.append(dbfs)
            if len(self._rms_history) > ADAPT_HISTORY_SECONDS * 2:
                # Trim only when way over the window to avoid touching
                # the list on every hop.
                self._rms_history = self._rms_history[-ADAPT_HISTORY_SECONDS:]
            thresh = self._adaptive_threshold()
            gated = (thresh is not None and dbfs < thresh)
            audioset_top: list = []
            if gated:
                # Quiet window — bypass the encoder, declare Ambient. The
                # HMM still steps with this synthetic posterior so the
                # state stays consistent when the next window crosses
                # back above threshold.
                post6 = np.zeros(len(CLASSES), dtype=np.float32)
                post6[AMBIENT_IDX] = 1.0
                audioset_top = [
                    {"name": "(silence-gated)", "p": 1.0},
                ]
            else:
                probs527 = self.encoder.predict(window)
                post6 = map_groups_to_classes(
                    probs527[None, :], self.group_idx, CLASSES,
                    beat_density=None)[0].astype(np.float32)
                # Top-K AudioSet classes for the on-screen "Encoder raw
                # output" panel. This mirrors what LiveCueSheet + the
                # precompute path expose so the same UI renders for the
                # AV-live source without special-casing.
                k = self.audioset_top_k
                top_idx = np.argpartition(-probs527, k - 1)[:k]
                top_idx = top_idx[np.argsort(-probs527[top_idx])]
                audioset_top = [
                    {"name": self._audioset_names[int(i)],
                     "p": float(probs527[int(i)])}
                    for i in top_idx
                ]
            label = self.hmm.step(post6)
            out.append(Emission(
                t_sec=float(self._emitted),
                post6=post6, hmm_label=label,
                rms_dbfs=dbfs, silence_gated=gated,
                audioset_top=audioset_top,
                silence_threshold_dbfs=float(thresh) if thresh is not None
                                       else float("-inf")))
            self._emitted += 1
            self._queue = self._queue[HOP_SAMPLES:]
        return out
