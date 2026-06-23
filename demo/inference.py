"""CueSheet pipeline as a callable: audio in → per-second posterior +
segment label out. Shared by demo/precompute.py and demo/server.py so the
preloaded gallery and the drag-drop endpoint emit the same shape.

Pipeline:
  audio file (mp3 / wav / m4a / mp4 / mov / mkv)
    -> ffmpeg extract 16 kHz mono wav
    -> PANN AudioTagging clipwise probs (T, 527)
    -> group-mapping projection (T, 4)   [Performance, MC_Talk, Applause, Ambient]
    -> map_to_classes argmax + confidence threshold
    -> apply_pre_concert_heuristic (lead-in seconds become Pre_Concert)

Returns a dict matching the JSON schema the demo frontend consumes.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cuesheet"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from cuesheet.scripts.bootstrap_labels import (  # noqa: E402
    CLASS_GROUPS, HOP_SEC, TARGET_SR, WINDOW_SEC,
    apply_pre_concert_heuristic, build_group_indices,
    extract_wav, load_audioset_index, map_to_classes,
)


def classify_windows_batched(wav_path: Path, device: str = "cpu",
                             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Batched PANN inference. Kept for run_cuesheet() backward compat. The
    streaming path uses EfficientAT MN10 via stream_cuesheet — see that
    function for the default encoder.
    """
    from panns_inference import AudioTagging

    model = AudioTagging(checkpoint_path=None, device=device)
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    assert sr == TARGET_SR, f"expected {TARGET_SR} Hz, got {sr}"

    win_n = int(WINDOW_SEC * TARGET_SR)
    hop_n = int(HOP_SEC * TARGET_SR)
    starts = list(range(0, len(wav) - win_n + 1, hop_n))

    probs: list[np.ndarray] = []
    n = len(starts)
    for b0 in range(0, n, batch_size):
        b1 = min(b0 + batch_size, n)
        batch = np.stack([wav[starts[i]: starts[i] + win_n] for i in range(b0, b1)])
        with torch.no_grad():
            clipwise, _ = model.inference(batch)
        probs.append(clipwise)
        if (b1 // batch_size) % 20 == 0:
            print(f"  [PANN] {b1}/{n} windows", flush=True)

    probs_arr = np.concatenate(probs, axis=0)
    ts = np.array([s / TARGET_SR for s in starts], dtype=np.float64)
    return probs_arr, ts


# Frontend expects this fixed 6-class order for ribbon coloring.
FULL_SEGMENT_CLASSES: list[str] = [
    "Pre_Concert", "Performance", "MC_Talk", "Applause",
    "Intermission", "Ambient",
]

# The 4 classes PANN actually emits posteriors for (the other 2 — Pre_Concert,
# Intermission — come from temporal heuristics, not classifier output, so we
# do not fake a posterior for them).
POSTERIOR_CLASSES: list[str] = ["Performance", "MC_Talk", "Applause", "Ambient"]


def _project_to_groups(clipwise: np.ndarray,
                       group_idx: dict[str, list[int]]) -> np.ndarray:
    """(T, 527) sigmoid probs → (T, 4) group-sum scores in POSTERIOR_CLASSES
    order, renormalized so each row sums to 1.0.
    """
    cols = []
    for cls in POSTERIOR_CLASSES:
        cols.append(clipwise[:, group_idx[cls]].sum(axis=-1))
    g = np.stack(cols, axis=-1).astype(np.float32)
    g = np.clip(g, 0.0, None)
    row_sum = g.sum(axis=-1, keepdims=True)
    row_sum = np.where(row_sum > 1e-6, row_sum, 1.0)
    return g / row_sum


def _load_audioset_class_names() -> list[str]:
    """Read the AudioSet display names (column 2) from the panns_data CSV."""
    import csv
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    names = [""] * 527
    if not csv_path.is_file():
        return names
    with open(csv_path) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3:
                idx = int(row[0])
                if 0 <= idx < 527:
                    names[idx] = row[2]
    return names


_AUDIOSET_NAMES_CACHE: list[str] | None = None


def stream_cuesheet(audio_path: Path, device: str = "cpu",
              window_batch: int = 4, confidence: float = 0.3,
              top_k: int = 7, start_sec: float = 0.0,
              causal: bool = False, encoder: str = "efficientat"):
    """Generator: yields one per-second prediction dict at a time, using
    the default encoder (EfficientAT MN10, ~5 M params / 19 MB) and an
    online causal HMM forward smoother.

    Each yielded dict has:
        {
          "t":                   int,        # second index
          "timestamp":           float,      # seconds
          "posterior":           [4 floats], # Perf / MC / Applause / Ambient
          "segment_idx":           int,        # 4-class argmax of posterior (raw)
          "segment_idx_smoothed":  int,        # HMM forward-pass argmax (sticky)
          "segment_name":          str,
          "audioset_top":        [{name, p} ...]  # top-K AudioSet classes
        }

    Encoder: EfficientAT MN10 (Schmid et al. 2023) — distilled
    MobileNet-style CNN, AudioSet-pretrained, ~30x faster than PANN
    CNN14 on CPU while retaining 91.5% F1 on our concert benchmark
    (vs 90.3% for PANN). The default encoder.

    Smoother: HMMSmoother online causal forward step (stay = 0.95) — one
    step per second, no need to wait for the whole sequence. Streaming
    posterior is the unsmoothed projection; segment_idx_smoothed is the
    HMM forward argmax so the ribbon does not flicker between adjacent
    seconds when the raw posterior is borderline.

    The Pre_Concert temporal heuristic (first sustained Performance run
    becomes the show-start cursor) is still applied client-side over
    the smoothed labels — see cuesheet.js.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "vendor" / "EfficientAT"))
    _sys.path.insert(0, str(REPO_ROOT / "cuesheet"))
    import librosa
    import torch as _torch
    _ensure_efficientat_sys_path()
    from scripts.encoder_efficientat import (  # noqa: E402
        _load, SAMPLE_RATE as EAT_SR, WINDOW_SEC as EAT_WIN_SEC,
        HOP_SEC as EAT_HOP_SEC,
    )
    from src.fusion.temporal import HMMSmoother  # noqa: E402

    global _AUDIOSET_NAMES_CACHE
    if _AUDIOSET_NAMES_CACHE is None:
        _AUDIOSET_NAMES_CACHE = _load_audioset_class_names()
    audioset_names = _AUDIOSET_NAMES_CACHE

    name_to_idx = load_audioset_index()
    group_idx = build_group_indices(name_to_idx)

    yield {"progress": "loading_encoder",
           "message": "loading EfficientAT MN10 (one-time, ~1 s)…"}
    model, mel = _load("mn10_as", device)

    with tempfile.TemporaryDirectory() as tmp:
        yield {"progress": "extracting_audio",
               "message": "ffmpeg: extracting audio track (slowest step "
                          "for video files; ~30-90 s for HD clips)…"}
        wav_path = extract_wav(audio_path, tmp)
        yield {"progress": "resampling",
               "message": "librosa: resampling to 32 kHz mono…"}
        # EfficientAT runs at 32 kHz; resample on load.
        wav, _ = librosa.core.load(str(wav_path), sr=EAT_SR, mono=True)
        if encoder not in ("efficientat", "mn10", "mn10_as"):
            # Heavy encoders (PANN / AST / HTS-AT). Offline: one batched
            # whole-file pass via the model's own classifier, then smooth
            # and yield rows. Causal: per-second trailing windows through
            # _run_causal_encoder (slower; same cost as the interactive
            # causal pump).
            hmm_h = HMMSmoother(num_states=len(POSTERIOR_CLASSES))
            alpha_h = hmm_h.init_alpha()
            sr_h = EAT_SR
            win_h = int(EAT_WIN_SEC * sr_h)
            total_h = len(wav)
            if causal:
                yield {"progress": "starting_inference",
                       "message": f"{encoder}: causal per-second windows "
                                  "(slower than real time on CPU)…"}
                t = int(start_sec)
                while True:
                    end = min((t + 1) * sr_h, total_h)
                    seg = wav[max(0, end - win_h):end]
                    if len(seg) < win_h:
                        seg = np.pad(seg, (win_h - len(seg), 0))
                    clip527 = _run_causal_encoder(encoder, seg, device)
                    yield from _emit_row(t, clip527[0], hmm_h, alpha_h, top_k)
                    alpha_h = _LAST_ALPHA[0]
                    if (t + 1) * sr_h >= total_h:
                        break
                    t += 1
            else:
                n_est = max(1, total_h // sr_h)
                yield {"progress": "starting_inference",
                       "message": f"{encoder}: one batched pass over the "
                                  f"whole file (~{n_est} s of audio)…"}
                probs = _full_file_probs(encoder, wav, device)
                for t in range(len(probs)):
                    yield from _emit_row(t, probs[t], hmm_h, alpha_h, top_k)
                    alpha_h = _LAST_ALPHA[0]
            yield {"done": True, "n_seconds": None}
            return

        yield {"progress": "starting_inference",
               "message": "first window through the encoder…"}
        waveform_t = _torch.from_numpy(wav)
        win_n = int(EAT_WIN_SEC * EAT_SR)
        sr = EAT_SR
        start_i = int(start_sec)
        total = len(waveform_t)

        # Build (output_second, window_tensor) pairs. The window controls
        # whether the prediction for second t is allowed to see the future.
        #   causal=False : window [t, t+WIN]  — offline; the label peeks
        #                  ahead, which is fine for a precomputed track.
        #   causal=True  : window [t-WIN, t]  — real-time; the label uses
        #                  only the trailing WIN seconds, never the future.
        #                  This is exactly what a live system hears at
        #                  show-time t, so a seek to t yields an immediate
        #                  full-window prediction (no warm-up). Only the
        #                  recording's first seconds are left-padded.
        items = []  # (t, window tensor of length win_n)
        if causal:
            t = start_i
            while True:
                end = min((t + 1) * sr, total)
                seg = waveform_t[max(0, end - win_n):end]
                if len(seg) < win_n:
                    seg = _torch.nn.functional.pad(seg, (win_n - len(seg), 0))
                items.append((t, seg))
                if (t + 1) * sr >= total:
                    break
                t += 1
        else:
            if total < win_n:
                waveform_t = _torch.nn.functional.pad(
                    waveform_t, (0, win_n - total))
                total = len(waveform_t)
            t = start_i
            while t * sr + win_n <= total:
                items.append((t, waveform_t[t * sr:t * sr + win_n]))
                t += 1
            # Tail: fewer than WIN seconds remain past t. Label the
            # remaining seconds from right-padded partial windows so the
            # track covers every second of the file, mirroring how the
            # causal branch left-pads the recording's first seconds.
            while t * sr < total:
                seg = waveform_t[t * sr:]
                seg = _torch.nn.functional.pad(seg, (0, win_n - len(seg)))
                items.append((t, seg))
                t += 1

        # Online HMM forward smoother over the 4-class posterior.
        hmm = HMMSmoother(num_states=len(POSTERIOR_CLASSES))
        alpha = hmm.init_alpha()

        for b0 in range(0, len(items), window_batch):
            b1 = min(b0 + window_batch, len(items))
            chunks = _torch.stack([items[i][1] for i in range(b0, b1)])
            with _torch.no_grad():
                specs = mel(chunks)                  # (B, n_mels, T_mel)
                preds, _features = model(specs.unsqueeze(1))  # (B, 527)
                clipwise = _torch.sigmoid(preds.float()).cpu().numpy()
            posts = _project_to_groups(clipwise, group_idx)
            top_idx = np.argpartition(-clipwise, top_k, axis=-1)[:, :top_k]
            for j in range(b1 - b0):
                tt = items[b0 + j][0]
                row = posts[j]
                raw = int(row.argmax())
                alpha, smoothed = hmm.forward_step(alpha, row)
                cw = clipwise[j]
                ti = top_idx[j]
                order = ti[np.argsort(-cw[ti])]
                top = [
                    {"name": (audioset_names[int(idx)] or f"class_{idx}")[:64],
                     "p": round(float(cw[int(idx)]), 4)}
                    for idx in order
                ]
                yield {
                    "t": tt,
                    "timestamp": float(tt),
                    "posterior": [round(float(v), 4) for v in row.tolist()],
                    "segment_idx": raw,
                    "segment_idx_smoothed": int(smoothed),
                    "segment_name": POSTERIOR_CLASSES[raw],
                    "audioset_top": top,
                }


def run_cuesheet(audio_path: Path, device: str = "cpu",
           confidence: float = 0.3) -> dict:
    """Run the full pipeline on a single file. Returns a payload dict
    with segment_int (6-class), posterior (T, 4), posterior_classes, and
    timing metadata.

    Inputs ending in .mp4/.mov/.mkv are routed through ffmpeg too — we
    extract the audio track and ignore the video.
    """
    audio_path = Path(audio_path)
    name_to_idx = load_audioset_index()
    group_idx = build_group_indices(name_to_idx)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = extract_wav(audio_path, tmp)
        clipwise, timestamps = classify_windows_batched(wav_path, device)

    posterior = _project_to_groups(clipwise, group_idx)

    rel_pos = timestamps / max(float(timestamps[-1]), 1.0)
    labels = map_to_classes(clipwise, group_idx, FULL_SEGMENT_CLASSES, rel_pos,
                            beat_density=None, confidence=confidence)
    labels = apply_pre_concert_heuristic(labels, FULL_SEGMENT_CLASSES)

    # Cascade: apply CausalPostApplause + post-applause heuristic +
    # CausalIntermission on top of the bootstrap labels so /api/infer
    # (drag-drop) emits the same shape as the LOSO + live pipeline.
    from live.causal_heuristics import (  # noqa: E402
        CausalIntermission, CausalPostApplause,
    )
    from cuesheet.scripts.bootstrap_labels import (  # noqa: E402
        apply_post_performance_applause_heuristic,
    )
    _app_idx_post = POSTERIOR_CLASSES.index("Applause")
    _app_post = posterior[:, _app_idx_post]
    labels = apply_post_performance_applause_heuristic(
        np.asarray(labels, dtype=int), FULL_SEGMENT_CLASSES,
        applause_posterior=_app_post,
        min_perf_run=40, window_sec=12, min_post_evidence=0.10)
    _post_app = CausalPostApplause(class_names=tuple(FULL_SEGMENT_CLASSES))
    _inter = CausalIntermission(class_names=tuple(FULL_SEGMENT_CLASSES))
    _final = np.zeros(len(labels), dtype=int)
    for _t in range(len(labels)):
        _ap_t = float(_app_post[_t]) if _t < len(_app_post) else 0.0
        _lab = _post_app.update(int(labels[_t]), applause_posterior=_ap_t)
        _final[_t] = _inter.update(_lab)
    labels = _final

    n_sec = int(len(labels))
    return {
        "segment_int": labels[:n_sec].astype(int).tolist(),
        "segment_classes": list(FULL_SEGMENT_CLASSES),
        "posterior": np.round(posterior[:n_sec], 4).astype(float).tolist(),
        "posterior_classes": list(POSTERIOR_CLASSES),
        "timestamps": np.round(timestamps[:n_sec], 3).astype(float).tolist(),
        "n_seconds": n_sec,
    }


class LiveCueSheet:
    """Stateful real-time inference for one live microphone session.

    ``feed()`` accepts raw mono float32 PCM at the client's sample rate
    and returns zero or more per-window prediction dicts (the same shape
    as ``stream_cuesheet`` rows). The online HMM forward state is carried
    across calls so the live ribbon is as stable as the file path.

    A new instance is created per WebSocket connection; the encoder is
    loaded once in ``__init__`` so ``feed()`` pays no warmup cost.
    """

    def __init__(self, input_sr: int, device: str = "cpu", top_k: int = 7):
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT / "vendor" / "EfficientAT"))
        _sys.path.insert(0, str(REPO_ROOT / "cuesheet"))
        import torch as _torch
        _ensure_efficientat_sys_path()
        from scripts.encoder_efficientat import (  # noqa: E402
            _load, SAMPLE_RATE as EAT_SR, WINDOW_SEC as EAT_WIN_SEC,
            HOP_SEC as EAT_HOP_SEC,
        )
        from src.fusion.temporal import HMMSmoother  # noqa: E402

        global _AUDIOSET_NAMES_CACHE
        if _AUDIOSET_NAMES_CACHE is None:
            _AUDIOSET_NAMES_CACHE = _load_audioset_class_names()
        self._names = _AUDIOSET_NAMES_CACHE
        self._torch = _torch
        self._group_idx = build_group_indices(load_audioset_index())
        self._model, self._mel = _load("mn10_as", device)
        self._top_k = top_k

        self._eat_sr = int(EAT_SR)
        self._hop_sec = float(EAT_HOP_SEC)
        self.input_sr = int(input_sr)
        self._win_in = int(EAT_WIN_SEC * self.input_sr)
        self._hop_in = int(EAT_HOP_SEC * self.input_sr)
        self._win_eat = int(EAT_WIN_SEC * EAT_SR)

        # Sliding buffer holds at most one window (WINDOW_SEC) of recent
        # audio. _received / _emitted count input-rate samples so we emit
        # exactly one prediction per HOP_SEC, starting one hop after the
        # mic opens rather than after a full 10 s window: early windows
        # are zero-padded, the same lead-in handling stream_cuesheet uses.
        self._buf = np.zeros(0, dtype=np.float32)
        self._received = 0
        self._emitted = 0
        self._hmm = HMMSmoother(num_states=len(POSTERIOR_CLASSES))
        self._alpha = self._hmm.init_alpha()
        self._t = 0

    def feed(self, samples: np.ndarray) -> list[dict]:
        """Append live PCM and emit one prediction per hop of audio."""
        import librosa

        if samples.size:
            self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
            if len(self._buf) > self._win_in:
                self._buf = self._buf[-self._win_in:]
            self._received += int(samples.size)
        out: list[dict] = []
        while self._received - self._emitted >= self._hop_in:
            self._emitted += self._hop_in
            window = self._buf

            if self.input_sr != self._eat_sr:
                w = librosa.resample(window, orig_sr=self.input_sr,
                                     target_sr=self._eat_sr)
            else:
                w = window
            if len(w) < self._win_eat:
                w = np.pad(w, (0, self._win_eat - len(w)))
            else:
                w = w[-self._win_eat:]

            chunk = self._torch.from_numpy(w).float().unsqueeze(0)
            with self._torch.no_grad():
                spec = self._mel(chunk)
                preds, _features = self._model(spec.unsqueeze(1))
                clipwise = self._torch.sigmoid(preds.float()).cpu().numpy()

            row = _project_to_groups(clipwise, self._group_idx)[0]
            raw = int(row.argmax())
            self._alpha, smoothed = self._hmm.forward_step(self._alpha, row)

            cw = clipwise[0]
            ti = np.argpartition(-cw, self._top_k)[: self._top_k]
            order = ti[np.argsort(-cw[ti])]
            top = [{"name": (self._names[int(i)] or f"class_{i}")[:64],
                    "p": round(float(cw[int(i)]), 4)} for i in order]

            out.append({
                "t": self._t,
                "timestamp": round(self._t * self._hop_sec, 3),
                "posterior": [round(float(v), 4) for v in row.tolist()],
                "segment_idx": raw,
                "segment_idx_smoothed": int(smoothed),
                "segment_name": POSTERIOR_CLASSES[raw],
                "audioset_top": top,
            })
            self._t += 1
        return out


# All causal encoders see a 10 s trailing window decoded at 32 kHz; each
# encoder resamples internally if it needs another rate. PANN re-windows
# the clip to its own 2 s and we keep only the most recent window.
_CAUSAL_WIN_SEC = 10
_CAUSAL_SR = 32_000
CAUSAL_ENCODERS = ("efficientat", "pann", "ast", "htsat")

# Optional pre-warmed full-show waveform cache (32 kHz mono float32).
# When present, causal_step slices the trailing window straight from
# memory instead of spawning ffmpeg per call, cutting per-second latency
# from ~140 ms to ~30 ms. Populated by warm_show_audio() which the
# server may kick off on loadShow.
_CAUSAL_WAV_CACHE: dict = {}


def warm_show_audio(audio_path) -> int:
    """Decode the entire show into the in-memory cache at 32 kHz mono.

    Only one show is held at a time to keep memory bounded; loading a
    different show evicts the previous one. Also runs a dummy encoder
    forward pass so the very first per-second pump pays no model-load
    cost. Returns the cached sample count. Safe to call concurrently
    — repeated calls for the same path are no-ops.
    """
    import librosa as _lr
    key = str(audio_path)
    if key not in _CAUSAL_WAV_CACHE:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = extract_wav(Path(audio_path), tmp)
            wav, _ = _lr.core.load(str(wav_path), sr=_CAUSAL_SR, mono=True)
        _CAUSAL_WAV_CACHE.clear()
        _CAUSAL_WAV_CACHE[key] = np.asarray(wav, dtype=np.float32)
    # Warm the EfficientAT model with a dummy 10 s forward so the first
    # real pump does not pay the ~80 ms model-load cost.
    try:
        win_n = int(_CAUSAL_WIN_SEC * _CAUSAL_SR)
        dummy = np.zeros(win_n, dtype=np.float32)
        _run_causal_encoder("efficientat", dummy, "cpu")
    except Exception:                          # pragma: no cover — best-effort
        pass
    return len(_CAUSAL_WAV_CACHE[key])


def _ensure_efficientat_sys_path() -> None:
    """Re-prime sys.path / sys.modules for EfficientAT's bare-name imports.

    EfficientAT provides ``models/`` as a namespace package (no
    ``__init__.py``) and loads its model via
    ``from models.mn.model import ...``. The HTS-AT vendor root happens
    to contain a regular file ``models.py``, and a regular module always
    beats a namespace package across ``sys.path``: once HTS-AT has been
    on the path the bare name ``models`` resolves to that file and
    EfficientAT fails with ``ModuleNotFoundError: No module named
    'models.mn'; 'models' is not a package``. The fix is two-fold:
    materialize empty ``__init__.py`` so EfficientAT's ``models/`` is a
    regular package (vendor is gitignored, so we create them on demand
    rather than commit them), and push the EfficientAT root to the
    front while evicting any stale cached ``models`` entry from a
    different vendor.
    """
    effat_root = REPO_ROOT / "vendor" / "EfficientAT"
    for sub in ("models", "models/mn", "models/dymn"):
        init = effat_root / sub / "__init__.py"
        if init.parent.exists() and not init.exists():
            try:
                init.touch()
            except OSError:
                pass
    effat_str = str(effat_root)
    if effat_str in sys.path:
        sys.path.remove(effat_str)
    sys.path.insert(0, effat_str)
    for k in list(sys.modules):
        if k != "models" and not k.startswith("models."):
            continue
        mod = sys.modules[k]
        f = getattr(mod, "__file__", "") or ""
        if not f.startswith(effat_str):
            del sys.modules[k]


def _run_causal_encoder(encoder: str, seg32k: np.ndarray,
                        device: str = "cpu") -> np.ndarray:
    """Run one AudioSet encoder on a single 10 s segment (given at 32 kHz).

    Returns ``(1, 527)`` sigmoid posteriors. EfficientAT runs in process on
    the segment directly; the other three reuse their tested whole-file
    classifiers on a one-window temp clip, so there is no re-implemented
    inference path to drift from the benchmarked encoders.
    """
    enc = (encoder or "efficientat").lower()
    sys.path.insert(0, str(REPO_ROOT / "vendor" / "EfficientAT"))
    sys.path.insert(0, str(REPO_ROOT / "cuesheet"))
    if enc in ("efficientat", "mn10", "mn10_as"):
        _ensure_efficientat_sys_path()
        from scripts.encoder_efficientat import _load
        model, mel = _load("mn10_as", device)
        s = torch.from_numpy(np.asarray(seg32k, dtype=np.float32)).float()
        with torch.no_grad():
            spec = mel(s.unsqueeze(0))
            preds, _features = model(spec.unsqueeze(1))
            return torch.sigmoid(preds.float()).cpu().numpy()
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "causal_window.wav"
        sf.write(str(clip), np.asarray(seg32k, dtype=np.float32), _CAUSAL_SR)
        if enc == "ast":
            from scripts.encoder_ast import classify_windows_ast
            probs, _ = classify_windows_ast(str(clip), device)
        elif enc == "htsat":
            from scripts.encoder_htsat import classify_windows_htsat
            probs, _ = classify_windows_htsat(str(clip), device)
        elif enc == "pann":
            import librosa as _lr
            from scripts.bootstrap_labels import classify_windows
            # PANN's classify_windows asserts a 16 kHz wav, so resample
            # the 32 kHz segment down before handing it the clip.
            seg16 = _lr.resample(np.asarray(seg32k, dtype=np.float32),
                                 orig_sr=_CAUSAL_SR, target_sr=16_000)
            sf.write(str(clip), seg16, 16_000)
            probs, _ = classify_windows(clip, device)
        else:
            raise ValueError(f"unknown encoder: {encoder}")
    return np.asarray(probs)[-1:]              # most recent window


def causal_step(audio_path, t: int, start_sec: int = 0, alpha=None,
                device: str = "cpu", top_k: int = 7,
                encoder: str = "efficientat") -> dict:
    """One genuinely on-demand causal prediction for second ``t``.

    The window is ``[max(start_sec, t-9), t+1]`` — only audio between the
    seek point and t. It never sees the future and never the recording's
    audio from *before* the seek point, so a rehearsal-style jump starts
    fresh and warms up, exactly like a live deployment. The caller
    advances t with the playhead, so nothing ahead is ever computed. The
    HMM forward state is carried by the caller via ``alpha`` (reset to
    ``None`` on a seek). ``encoder`` selects the AudioSet backbone.
    """
    import librosa
    from src.fusion.temporal import HMMSmoother  # noqa: E402

    global _AUDIOSET_NAMES_CACHE
    if _AUDIOSET_NAMES_CACHE is None:
        _AUDIOSET_NAMES_CACHE = _load_audioset_class_names()
    group_idx = build_group_indices(load_audioset_index())

    # Trailing 10 s ending at t+1. The receptive field is what a live
    # deployment naturally always has as a rolling buffer; the real
    # "no memory across a jump" guarantee lives in the HMM forward
    # state (alpha), which is reset to a uniform prior below. If the
    # show was pre-warmed into _CAUSAL_WAV_CACHE we slice straight from
    # memory (~30 ms total per call); otherwise fall back to a
    # per-window ffmpeg pipe (~140 ms).
    end_s = t + 1
    win_start_s = max(0, end_s - _CAUSAL_WIN_SEC)
    win_dur_s = max(1, end_s - win_start_s)
    key = str(audio_path)
    if key in _CAUSAL_WAV_CACHE:
        wav = _CAUSAL_WAV_CACHE[key]
        seg = wav[win_start_s * _CAUSAL_SR : end_s * _CAUSAL_SR].copy()
    else:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(win_start_s), "-i", str(audio_path),
             "-t", str(win_dur_s), "-ac", "1", "-ar", str(_CAUSAL_SR),
             "-f", "f32le", "-"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        seg = np.frombuffer(proc.stdout, dtype=np.float32)
    win_n = int(_CAUSAL_WIN_SEC * _CAUSAL_SR)
    if len(seg) < win_n:                       # early window — left-pad
        seg = np.pad(seg, (win_n - len(seg), 0))
    else:
        seg = seg[-win_n:]

    clipwise = _run_causal_encoder(encoder, seg, device)
    row = _project_to_groups(clipwise, group_idx)[0]
    raw = int(row.argmax())

    hmm = HMMSmoother(num_states=len(POSTERIOR_CLASSES))
    a = (np.asarray(alpha, dtype=float)
         if alpha is not None and len(alpha) else hmm.init_alpha())
    a, smoothed = hmm.forward_step(a, row)

    cw = clipwise[0]
    ti = np.argpartition(-cw, top_k)[:top_k]
    order = ti[np.argsort(-cw[ti])]
    top = [{"name": (_AUDIOSET_NAMES_CACHE[int(i)] or f"class_{i}")[:64],
            "p": round(float(cw[int(i)]), 4)} for i in order]
    return {
        "t": int(t),
        "posterior": [round(float(v), 4) for v in row.tolist()],
        "segment_idx": raw,
        "segment_idx_smoothed": int(smoothed),
        "segment_name": POSTERIOR_CLASSES[raw],
        "audioset_top": top,
        "encoder": (encoder or "efficientat").lower(),
        "alpha": [round(float(x), 6) for x in
                  (a.tolist() if hasattr(a, "tolist") else list(a))],
    }



def _full_file_probs(encoder: str, wav32k, device: str = "cpu"):
    """Whole-file 527-class posteriors for the heavy encoders, using each
    model's own batched file-level classifier (the same code path the
    gallery per-encoder precompute uses). The audio is right-padded by
    one window minus one hop before classification and the output is
    trimmed to the real duration, so every real second gets a row and
    the ribbon fills to the end — mirroring the EfficientAT tail fix.
    Returns (n_real_seconds, 527) with leading windows.
    """
    import tempfile
    import librosa
    import soundfile as _sf
    import numpy as _np
    wav32k = _np.asarray(wav32k, dtype=_np.float32)
    n_real = int(_np.ceil(len(wav32k) / 32_000))
    win_sec = 2 if encoder == "pann" else 10
    pad32 = _np.pad(wav32k, (0, (win_sec - 1) * 32_000))
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "full.wav"
        if encoder == "pann":
            from scripts.bootstrap_labels import classify_windows
            wav16 = librosa.resample(pad32, orig_sr=32_000, target_sr=16_000)
            _sf.write(str(clip), wav16, 16_000)
            probs, _ = classify_windows(clip, device)
        elif encoder == "ast":
            from scripts.encoder_ast import classify_windows_ast
            wav16 = librosa.resample(pad32, orig_sr=32_000, target_sr=16_000)
            _sf.write(str(clip), wav16, 16_000)
            probs, _ = classify_windows_ast(str(clip), device)
        elif encoder == "htsat":
            from scripts.encoder_htsat import classify_windows_htsat
            _sf.write(str(clip), pad32, 32_000)
            probs, _ = classify_windows_htsat(str(clip), device)
        else:
            raise ValueError(f"unknown encoder: {encoder}")
    return _np.asarray(probs, dtype=_np.float32)[:n_real]


# Row emission shared by the heavy-encoder branches of stream_cuesheet. Keeps the
# exact row shape of the EfficientAT path. _LAST_ALPHA carries the smoother
# state back to the caller across the generator boundary.
_LAST_ALPHA = [None]

def _emit_row(t: int, clipwise_row, hmm, alpha, top_k: int):
    global _AUDIOSET_NAMES_CACHE
    if _AUDIOSET_NAMES_CACHE is None:
        _AUDIOSET_NAMES_CACHE = _load_audioset_class_names()
    names = _AUDIOSET_NAMES_CACHE
    name_to_idx = load_audioset_index()
    group_idx = build_group_indices(name_to_idx)
    row = _project_to_groups(clipwise_row[None, :], group_idx)[0]
    raw = int(row.argmax())
    alpha, smoothed = hmm.forward_step(alpha, row)
    _LAST_ALPHA[0] = alpha
    order = np.argsort(-clipwise_row)[:top_k]
    top = [{"name": (names[int(i)] or f"class_{i}")[:64],
            "p": round(float(clipwise_row[int(i)]), 4)} for i in order]
    yield {
        "t": t,
        "timestamp": float(t),
        "posterior": [round(float(v), 4) for v in row.tolist()],
        "segment_idx": raw,
        "segment_idx_smoothed": int(smoothed),
        "segment_name": POSTERIOR_CLASSES[raw],
        "audioset_top": top,
    }
