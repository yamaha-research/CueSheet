"""Audio sources for the streaming pipeline.

Two implementations share one interface:

  FileAudioSource  : replays a file at real time (1 second per second), so a
                     "live" run on a labeled show produces the same timing
                     as a real microphone would, for offline validation.
  LiveAudioSource  : captures from the system audio device via sounddevice
                     (PortAudio). Cross-platform; the macOS path uses
                     CoreAudio under the hood and works with the built-in
                     mic or any audio interface (a mixer bus exposed as an
                     audio device would be the same path).

Both deliver 32 kHz mono float32 chunks of length HOP_SAMPLES (1 second), to
be passed directly into AudioPipeline.push.
"""
from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np

from live.audio_encoder import HOP_SAMPLES, SAMPLE_RATE


class FileAudioSource:
    """Replays a file at real time. Yields HOP_SAMPLES per second."""

    def __init__(self, path: Path, paced: bool = True):
        import librosa
        self.path = Path(path)
        self.paced = paced
        wav, _ = librosa.core.load(str(self.path), sr=SAMPLE_RATE, mono=True)
        self._wav = wav.astype(np.float32, copy=False)
        self._cursor = 0
        self._t_started: float | None = None

    @property
    def duration_sec(self) -> float:
        return len(self._wav) / SAMPLE_RATE

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self._cursor + HOP_SAMPLES > len(self._wav):
            raise StopIteration
        chunk = self._wav[self._cursor: self._cursor + HOP_SAMPLES]
        self._cursor += HOP_SAMPLES
        if self.paced:
            if self._t_started is None:
                self._t_started = time.perf_counter()
            target = self._t_started + (self._cursor / SAMPLE_RATE)
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
        return chunk


class LiveAudioSource:
    """Captures from a sounddevice input stream. Hand each yielded chunk
    directly to AudioPipeline.push.

    sounddevice imports lazily so this module loads on machines where it is
    not installed; install with `uv add sounddevice` (and on macOS the
    PortAudio backend comes with the wheel)."""

    def __init__(self, device: int | str | None = None,
                 capture_rate: int = 48_000):
        """device: sounddevice index or substring of the input device name;
        None picks the system default input. capture_rate: 48 kHz matches
        common pro-audio output and most MacBook inputs; we resample
        internally to the 32 kHz the encoder expects."""
        import sounddevice as sd  # noqa: F401  imported lazily
        self.device = device
        self.capture_rate = capture_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._buf = np.empty(0, dtype=np.float32)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        # `indata` is (frames, channels) float32 at capture_rate. Take the
        # first channel (mono) and queue it; resampling happens on read.
        if status:
            # Underruns/overruns; print but keep going.
            print(f"[LiveAudioSource] status: {status}")
        self._q.put(indata[:, 0].copy())

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.capture_rate, channels=1, dtype="float32",
            device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        """Block until 1 second of 32 kHz mono audio is ready, then return it."""
        if self._stream is None:
            self.start()
        from scipy.signal import resample_poly
        # Target HOP_SAMPLES samples of 32 kHz mono. We need
        # HOP_SAMPLES * capture_rate / 32000 samples at the capture rate
        # before resampling.
        need_capture = int(HOP_SAMPLES * self.capture_rate / SAMPLE_RATE)
        while len(self._buf) < need_capture:
            chunk = self._q.get()
            self._buf = (chunk if self._buf.size == 0
                         else np.concatenate([self._buf, chunk]))
        capture = self._buf[:need_capture]
        self._buf = self._buf[need_capture:]
        # Polyphase resample capture_rate -> 32 kHz.
        # gcd-reduced ratio keeps the filter small. For 48000 -> 32000, ratio
        # 2/3.
        from math import gcd
        g = gcd(SAMPLE_RATE, self.capture_rate)
        up = SAMPLE_RATE // g
        down = self.capture_rate // g
        resampled = resample_poly(capture, up, down).astype(np.float32, copy=False)
        # Trim or pad to exactly HOP_SAMPLES (resample_poly can round).
        if len(resampled) > HOP_SAMPLES:
            resampled = resampled[:HOP_SAMPLES]
        elif len(resampled) < HOP_SAMPLES:
            resampled = np.pad(resampled, (0, HOP_SAMPLES - len(resampled)))
        return resampled
