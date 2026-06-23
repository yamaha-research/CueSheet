"""CueSheet demo backend — drag-drop an mp3 into the browser UI, get
per-second predictions back.

Run:
    uv run uvicorn demo.server:app --reload --port 8000
    # then open http://localhost:8000/static/

The backend mounts the existing static frontend at `/static/`, the
precomputed showcase payloads at `/data/`, and the showcase audio
symlinks at `/audio/`. It adds one new endpoint:

    POST /api/infer
        multipart upload: file=<audio.mp3>
        returns: same JSON shape as data/{show_id}.json

For interactive latency (a user drops a ≤5-min clip), inference is
synchronous in the request handler. Longer clips are truncated to
MAX_INFER_SEC.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import (FastAPI, File, Form, HTTPException, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import (FileResponse, JSONResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

# Make the project root importable when launched via uvicorn.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_INFER_SEC = 7_200     # 2 hours — long enough for typical full-concert
                          # uploads while still bounding CPU runtime
MAX_UPLOAD_MB = 5_000     # hard cap on body size (5 GB; we stream to disk
                          # so this is bounded by disk free space, not RAM)

DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parent

# Catalog selection. CueSheet serves the gallery catalog (catalog.json by
# default), overridable through the CUESHEET_CATALOG env var.
# The /api/catalog endpoint, the audio-lookup helper in this file, and the
# static frontend (which fetches /data/<this filename>) all resolve through
# this single env-driven name, so switching catalogs is just
#   CUESHEET_CATALOG=other_catalog.json uvicorn demo.server:app ...
# with no code change.
import os
CATALOG_FILENAME = os.environ.get("CUESHEET_CATALOG", "catalog.json")

# ===== Remote fallback for gallery files on a partial-dataset machine =====
# If the local demo can't find a gallery asset (audio / video / meta JSON),
# stream-proxy from the same file on a configured remote. Set
# REMOTE_DEMO_BASE to that remote's base URL, or "" to disable; the
# default (localhost) is effectively off. Streaming through this server
# (instead of a 302) keeps everything same-origin so the browser's
# fetch() can report download progress without CORS preflight issues, and
# the request still benefits from range support on the remote where available.
import os  # noqa: E402
import httpx  # noqa: E402
REMOTE_DEMO_BASE = os.environ.get(
    "REMOTE_DEMO_BASE", "http://localhost:8000"
).rstrip("/")

# Shared async client (one per process). Long read timeout because big
# videos can take 20+ s to fully transit; connect timeout stays short so
# missing remotes fail fast.
_remote_http: httpx.AsyncClient | None = None


def _remote_client() -> httpx.AsyncClient:
    global _remote_http
    if _remote_http is None:
        _remote_http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=4.0, read=60.0, write=10.0, pool=4.0),
            follow_redirects=True,
        )
    return _remote_http


async def _proxy_to_remote(remote_url: str, request: Request) -> StreamingResponse:
    """Stream-fetch `remote_url` and forward to the caller. Range headers
    (used by <video> seeking) pass through both ways so partial downloads
    still work."""
    headers = {}
    for h in ("range", "accept", "accept-encoding"):
        v = request.headers.get(h)
        if v:
            headers[h] = v
    client = _remote_client()
    try:
        req = client.build_request("GET", remote_url, headers=headers)
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"remote unreachable: {e}") from e
    if resp.status_code >= 400:
        body = await resp.aread()
        await resp.aclose()
        return StreamingResponse(iter([body]), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type",
                                                             "application/octet-stream"))
    fwd_headers = {}
    for h in ("content-length", "content-range", "accept-ranges",
              "last-modified", "etag"):
        v = resp.headers.get(h)
        if v:
            fwd_headers[h] = v

    async def _gen():
        try:
            async for chunk in resp.aiter_raw(chunk_size=65536):
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        _gen(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers=fwd_headers,
    )

# ===== Upload session registry (causal-from-playhead on user uploads) =====
# An upload's temp dir + audio path stays alive for MAX_UPLOADS sessions so
# /api/infer_causal_step can re-decode windows from it on demand. Oldest
# upload is evicted (shutil.rmtree) when the cap is reached. Keys are
# 12-char hex ids the upload endpoint stamps into the streaming response's
# meta event; the frontend mirrors them into currentShow.show_id as
# "upload:<id>".
MAX_UPLOADS = 5
_UPLOAD_REGISTRY: dict[str, tuple[str, Path]] = {}


def _register_upload(upload_id: str, tmp_dir: str, audio_path: Path) -> None:
    import shutil
    _UPLOAD_REGISTRY[upload_id] = (tmp_dir, audio_path)
    while len(_UPLOAD_REGISTRY) > MAX_UPLOADS:
        old_id, (old_dir, _) = next(iter(_UPLOAD_REGISTRY.items()))
        del _UPLOAD_REGISTRY[old_id]
        shutil.rmtree(old_dir, ignore_errors=True)


def _resolve_show_audio(show: str) -> Path:
    """Resolve a show identifier to its on-disk audio file.

    Gallery shows pass their catalog ``show_id``; uploads pass
    ``upload:<upload_id>`` from the streaming response's meta event.
    """
    if show.startswith("upload:"):
        entry = _UPLOAD_REGISTRY.get(show[len("upload:"):])
        if entry is None:
            raise HTTPException(404, f"upload session expired: {show}")
        audio_path = entry[1]
        if not audio_path.exists():
            raise HTTPException(404, f"upload audio missing on disk: {show}")
        return audio_path
    catalog = json.loads((DEMO_ROOT / "data" / CATALOG_FILENAME).read_text())
    entry = next((c for c in catalog if c.get("show_id") == show), None)
    if entry is None:
        raise HTTPException(404, f"unknown show: {show}")
    audio_name = entry.get("audio_filename") or f"{show}.mp3"
    audio_path = DEMO_ROOT / "audio" / audio_name
    if not audio_path.exists():
        raise HTTPException(404, f"audio not found for show {show}")
    return audio_path


app = FastAPI(title="CueSheet demo backend")

# Static frontend + showcase payloads + showcase audio symlinks.
app.mount("/static", StaticFiles(directory=DEMO_ROOT / "static", html=True),
          name="static")


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Tell the browser to revalidate /static/* and /data/* on every request
    so we don't burn debugging time on users seeing a stale cuesheet.js
    from before the latest fix.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path.startswith("/data/") \
            or path == "/" or path in ("/demo", "/cuesheet", "/teaser"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response

app.mount("/data", StaticFiles(directory=DEMO_ROOT / "data"), name="data")


# Fall-back route for /data/{name} so requests for gallery metadata not
# present locally redirect to the canonical remote. StaticFiles only
# matches existing files; an unmatched request falls through to this
# handler (registered AFTER the mount on purpose).
@app.get("/api/catalog")
async def serve_active_catalog():
    """Return the catalog JSON the current deployment is configured to
    surface, picked at process start from CUESHEET_CATALOG (default
    catalog.json). The static frontend resolves shows through this
    endpoint so different deployments can serve the same JS bundle and
    differ only in which catalog file is loaded."""
    path = (DEMO_ROOT / "data" / CATALOG_FILENAME).resolve()
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"configured catalog file missing: {CATALOG_FILENAME}",
        )
    return FileResponse(path, media_type="application/json")


@app.get("/data/{filename:path}")
async def serve_data_fallback(filename: str, request: Request):
    local = (DEMO_ROOT / "data" / filename).resolve()
    if local.is_file():
        return FileResponse(local)
    if REMOTE_DEMO_BASE:
        # Remote /data is the cuesheet backend mount point matching ours.
        return await _proxy_to_remote(
            f"{REMOTE_DEMO_BASE}/data/{filename}", request,
        )
    raise HTTPException(status_code=404, detail=f"data not found: {filename}")


# Audio served via a custom route so we can follow symlinks into
# data/raw/ — Starlette's StaticFiles rejects symlink targets outside
# the mount directory as a security measure.
@app.get("/audio/{filename:path}")
async def serve_audio(filename: str, request: Request):
    audio_path = (DEMO_ROOT / "audio" / filename).resolve()
    if audio_path.is_file():
        return FileResponse(audio_path, media_type="audio/mpeg")
    if REMOTE_DEMO_BASE:
        # Remote also exposes /audio/ via the cuesheet backend (proxied
        # through the front proxy on :8000 to demo/server.py on :8001).
        return await _proxy_to_remote(
            f"{REMOTE_DEMO_BASE}/audio/{filename}", request,
        )
    raise HTTPException(status_code=404, detail=f"audio not found: {filename}")

# Video served the same way for gallery entries whose source is a YouTube
# mp4 (Bach, Boiler Room), used only as the gallery preview player.
@app.get("/video/{filename:path}")
async def serve_video(filename: str, request: Request):
    """Serve gallery video, with fall-backs for partial datasets.

    The catalog advertises one `<stem>.<ext>` per show, but on machines
    without the full download we may only have a 30-second preview clip
    (`<stem>_0-30.mp4`) or a same-stem `.mov` carried over from an
    earlier dataset. Try those alternates before giving up so the demo
    page degrades to a short preview instead of a broken player.
    """
    root = DEMO_ROOT / "video"
    candidates = [root / filename]
    stem = Path(filename).stem
    candidates += [
        root / f"{stem}_0-30.mp4",
        root / f"{stem}.mp4",
        root / f"{stem}.mov",
        root / f"{stem}.webm",
    ]
    video_path = next((c.resolve() for c in candidates if c.is_file()), None)
    if video_path is None:
        if REMOTE_DEMO_BASE:
            return await _proxy_to_remote(
                f"{REMOTE_DEMO_BASE}/video/{filename}", request,
            )
        raise HTTPException(status_code=404, detail=f"video not found: {filename}")
    ext = video_path.suffix.lower()
    mime = "video/mp4" if ext == ".mp4" else (
           "video/quicktime" if ext == ".mov" else (
           "video/webm" if ext == ".webm" else "application/octet-stream"))
    return FileResponse(video_path, media_type=mime)

# Annotator tool — bootstrap-then-correct boundary labeler. Mounted only
# when the labeler assets are present (optional component).
_LABELER_DIR = DEMO_ROOT / "labeler"
if _LABELER_DIR.is_dir():
    app.mount("/labeler", StaticFiles(directory=_LABELER_DIR),
              name="labeler")


# The debug-video passthrough lives at the bottom of this file so its
# /{name:path} pattern does not shadow the explicit /api/* and
# /<mount>/* routes declared below.


def _redirect_preserve_query(request: Request, target: str) -> RedirectResponse:
    qs = request.url.query
    return RedirectResponse(url=f"{target}?{qs}" if qs else target)


@app.get("/")
async def root(request: Request) -> RedirectResponse:
    # The paper's demo URL points at the root, so the root must open the
    # demo itself. The landing page lives one hop away at /about.
    return _redirect_preserve_query(request, "/static/cuesheet.html")


@app.get("/about")
async def about_entry(request: Request) -> RedirectResponse:
    return _redirect_preserve_query(request, "/static/landing.html")


# the demo's target: CueSheet segment detection.
@app.get("/demo")
async def demo_entry(request: Request) -> RedirectResponse:
    return _redirect_preserve_query(request, "/static/cuesheet.html")


@app.get("/cuesheet")
async def cuesheet_entry(request: Request) -> RedirectResponse:
    return _redirect_preserve_query(request, "/static/cuesheet.html")




@app.get("/teaser")
async def teaser_entry(request: Request) -> RedirectResponse:
    return _redirect_preserve_query(request, "/static/teaser.html")


@app.get("/api/gt_labels/{show_id}")
async def gt_labels_endpoint(show_id: str) -> JSONResponse:
    """Per-second hand-labeled ground truth for a gallery show.

    Returns ``{"labels": [int, ...], "classes": [str, ...]}`` if the
    show has a hand label track on disk
    (``data/labels/<show_id>_labels.npz``), else 404. The integers are
    the canonical 6-class order: 0 Pre_Concert, 1 Performance,
    2 MC_Talk, 3 Applause, 4 Intermission, 5 Ambient. The frontend
    overlays this on the ribbon so the user can see the model's
    per-second mistakes directly.
    """
    # Tight filename validation so a crafted show_id cannot escape the
    # data/labels directory.
    safe = "".join(c for c in show_id
                   if c.isalnum() or c in ("_", "-", "."))
    if safe != show_id or not safe:
        raise HTTPException(404)
    p = REPO_ROOT / "data" / "labels" / f"{show_id}_labels.npz"
    if not p.is_file():
        raise HTTPException(404)
    z = np.load(p)
    labels = z["labels"].astype(int).tolist()
    if "classes" in z.files:
        classes = [str(c) for c in z["classes"].tolist()]
    else:
        classes = ["Pre_Concert", "Performance", "MC_Talk", "Applause",
                   "Intermission", "Ambient"]
    return JSONResponse({"labels": labels, "classes": classes,
                          "n_seconds": len(labels)})


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "max_infer_sec": MAX_INFER_SEC,
    }


@app.post("/api/infer_stream")
async def infer_stream(file: UploadFile = File(...),
                       causal: str = Form("1"),
                       encoder: str = Form("efficientat"),
                       register_only: str = Form("0")) -> StreamingResponse:
    """Streaming CueSheet inference. Same input as /api/infer but emits one
    NDJSON line per per-second prediction as soon as PANN finishes that
    window, so the frontend can paint bars + ribbon in true real time.

    Wire format (application/x-ndjson):
        {"meta": {"posterior_classes": [...], "horizon_sec": N}}
        {"t": 0, "posterior": [...], "segment_idx": 1, "segment_name": "Performance"}
        {"t": 1, "posterior": [...], ...}
        ...
        {"done": true, "n_seconds": N, "timing_sec": X}
    """
    from demo.inference import POSTERIOR_CLASSES, stream_cuesheet

    suffix = Path(file.filename or "upload.mp3").suffix or ".mp3"
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    chunk_size = 4 * 1024 * 1024

    # Keep the upload around after streaming so causal-from-playhead can
    # re-decode windows from it on demand. The directory is registered
    # under an upload_id and evicted by _trim_uploads() once the dict
    # exceeds MAX_UPLOADS.
    import uuid as _uuid
    upload_id = _uuid.uuid4().hex[:12]
    tmp_dir = tempfile.mkdtemp(prefix="cuesheet_stream_")
    in_path = Path(tmp_dir) / f"upload{suffix}"
    total = 0
    with open(in_path, "wb") as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    413,
                    f"Upload exceeded {MAX_UPLOAD_MB} MB hard cap "
                    f"(received {total / 1024 / 1024:.1f} MB so far).",
                )
            out.write(chunk)
    _register_upload(upload_id, tmp_dir, in_path)

    def gen():
        t0 = time.perf_counter()
        n_emitted = 0
        try:
            yield (json.dumps({"meta": {
                "posterior_classes": list(POSTERIOR_CLASSES),
                "filename": file.filename,
                "max_infer_sec": MAX_INFER_SEC,
                "upload_id": upload_id,
            }}) + "\n").encode("utf-8")
            if register_only == "1":
                # Live-pace mode: the file is registered for the causal
                # per-second endpoint; no inference happens here. Labels are
                # computed one second at a time as the playhead advances.
                yield (json.dumps({"done": True, "registered": True,
                                   "n_seconds": 0}) + "\n").encode("utf-8")
                return
            try:
                # Uploads run the causal branch (trailing windows, left-padded head)
                # so every user-driven inference matches the paper's no-look-ahead
                # claim; the gallery's stored tracks remain the offline variant the
                # paper's Section 4 numbers were measured on, labeled "Precomputed"
                # in the UI.
                for row in stream_cuesheet(in_path, device="cpu",
                                     causal=(causal == "1"),
                                     encoder=encoder):
                    if row.get("t", -1) >= MAX_INFER_SEC:
                        break
                    yield (json.dumps(row) + "\n").encode("utf-8")
                    n_emitted += 1
            except Exception as exc:  # noqa: BLE001
                # Surface decode / inference failures back to the client
                # rather than slamming the connection shut mid-stream.
                yield (json.dumps({
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_seconds": n_emitted,
                }) + "\n").encode("utf-8")
                return
            yield (json.dumps({
                "done": True,
                "n_seconds": n_emitted,
                "timing_sec": round(time.perf_counter() - t0, 3),
                "upload_id": upload_id,
            }) + "\n").encode("utf-8")
        finally:
            # Keep the file — causal-from-playhead may still need it. The
            # eviction in _register_upload bounds total disk usage.
            pass

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )


_CAUSAL_WARMING: set[str] = set()


@app.post("/api/causal_warm")
async def causal_warm(show: str) -> JSONResponse:
    """Pre-warm the on-demand causal cache for a gallery show or upload.

    Triggers a background decode of the source audio at 32 kHz so
    subsequent ``/api/infer_causal_step`` calls slice straight from
    memory (~30 ms) instead of paying an ffmpeg pipe per call (~140 ms).
    Returns immediately; the decode runs in a worker thread. Repeated
    calls for the same source are no-ops. ``show`` is either a gallery
    show_id or ``upload:<upload_id>``.
    """
    from demo.inference import _CAUSAL_WAV_CACHE, warm_show_audio

    audio_path = _resolve_show_audio(show)
    key = str(audio_path)
    if key in _CAUSAL_WAV_CACHE:
        return JSONResponse({"status": "warm", "show": show})
    if key in _CAUSAL_WARMING:
        return JSONResponse({"status": "warming", "show": show})
    _CAUSAL_WARMING.add(key)

    async def _bg() -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, warm_show_audio, audio_path)
        finally:
            _CAUSAL_WARMING.discard(key)

    asyncio.create_task(_bg())
    return JSONResponse({"status": "started", "show": show})


@app.get("/api/infer_causal_step")
async def infer_causal_step(show: str, t: int, start: int = 0,
                            alpha: str = "",
                            encoder: str = "efficientat") -> JSONResponse:
    """One on-demand causal prediction for second ``t`` of a gallery
    show or user upload.

    The frontend calls this once per second as the playhead advances, so
    nothing ahead of the playhead is ever computed — no precompute, no
    cheating. The window is floored at the seek point ``start`` and never
    sees the future. ``alpha`` carries the HMM forward state between
    calls (empty on a fresh seek). ``encoder`` selects the AudioSet
    backbone (efficientat / pann / ast / htsat). ``show`` is either a
    gallery show_id or ``upload:<upload_id>`` from the stream meta event.
    """
    from demo.inference import CAUSAL_ENCODERS, causal_step

    enc = (encoder or "efficientat").lower()
    if enc not in CAUSAL_ENCODERS:
        raise HTTPException(400, f"unknown encoder: {encoder}")

    audio_path = _resolve_show_audio(show)

    a = [float(x) for x in alpha.split(",") if x.strip()] if alpha else None
    try:
        row = await asyncio.get_event_loop().run_in_executor(
            None, lambda: causal_step(audio_path, int(t), int(start), a,
                                      encoder=enc))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)
    return JSONResponse(row)


@app.websocket("/api/infer_live")
async def infer_live(ws: WebSocket) -> None:
    """Real-time CueSheet inference over a live microphone stream.

    Client protocol:
      1. open the socket and send one JSON text frame ``{"sr": <rate>}``
         to start a session at the browser's AudioContext sample rate
      2. stream raw little-endian float32 mono PCM as binary frames
      3. receive one JSON per-window prediction back per hop of audio,
         in the same shape as the /api/infer_stream rows

    Inference runs in a worker thread so the encoder forward pass does
    not block the event loop receiving the next audio frame.
    """
    await ws.accept()
    from demo.inference import LiveCueSheet, POSTERIOR_CLASSES

    sess: LiveCueSheet | None = None
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text, data = msg.get("text"), msg.get("bytes")
            if text is not None:
                cfg = json.loads(text)
                sr = int(cfg.get("sr", 48_000))
                sess = await asyncio.get_event_loop().run_in_executor(
                    None, LiveCueSheet, sr)
                await ws.send_json({"meta": {
                    "posterior_classes": list(POSTERIOR_CLASSES),
                    "input_sr": sr,
                }})
            elif data is not None and sess is not None:
                samples = np.frombuffer(data, dtype=np.float32)
                rows = await asyncio.get_event_loop().run_in_executor(
                    None, sess.feed, samples)
                for row in rows:
                    await ws.send_json(row)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"error": f"{type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001
            pass




@app.post("/api/infer")
async def infer(file: UploadFile = File(...)) -> JSONResponse:
    """Run the CueSheet pipeline on an uploaded audio or video file.
    Returns per-second segment labels + 4-class posterior so the cuesheet
    frontend can drive the ribbon and the animated bar chart on real,
    pipeline-emitted values.
    """
    from demo.inference import run_cuesheet  # late import: heavy deps

    suffix = Path(file.filename or "upload.mp3").suffix or ".mp3"
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    chunk_size = 4 * 1024 * 1024   # 4 MB chunks

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / f"upload{suffix}"

        # Stream the body to disk in chunks so a multi-GB video upload
        # does not sit in RAM. Bail out as soon as the running total
        # exceeds the hard cap.
        total = 0
        with open(in_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        413,
                        f"Upload exceeded {MAX_UPLOAD_MB} MB hard cap "
                        f"(received {total / 1024 / 1024:.1f} MB so far).",
                    )
                out.write(chunk)

        t0 = time.perf_counter()
        result = run_cuesheet(in_path, device="cpu")
        t_total = time.perf_counter() - t0

    n_sec = int(result["n_seconds"])
    if n_sec > MAX_INFER_SEC:
        for k in ("segment_int", "posterior", "timestamps"):
            if isinstance(result.get(k), list):
                result[k] = result[k][:MAX_INFER_SEC]
        n_sec = MAX_INFER_SEC

    return JSONResponse({
        "show_id": f"upload_{uuid.uuid4().hex[:8]}",
        "label": file.filename or "uploaded clip",
        "genre": "User upload",
        "n_seconds": n_sec,
        "segment_int": result["segment_int"][:n_sec],
        "segment_classes": result["segment_classes"],
        "posterior": result["posterior"][:n_sec],
        "posterior_classes": result["posterior_classes"],
        "timing": {
            "total_sec": round(t_total, 3),
            "real_time_factor": round(t_total / max(n_sec, 1), 4),
        },
    })


# ===== CueSheet live-recording sessions ===================================
# Saved live captures (audio + optional video + meta) live on disk under
# data/sessions/<YYYY-MM-DD-HHMMSS>/. Path is gitignored. Three endpoints:
#   POST /api/save_session              save a finished live capture
#   GET  /api/list_sessions             enumerate saved sessions
#   GET  /api/session/{sid}/{file}      stream back audio / video / meta.json
import datetime  # noqa: E402

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"


@app.post("/api/save_session")
async def save_session(
    audio: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    meta:  str = Form("{}"),
):
    """Persist a live-capture session to disk.

    `meta` is a JSON string; the client should include at least
    `{label, mode, duration_sec, segment_classes, segment_int}` so the
    sessions dropdown can show useful labels and load_session can
    rehydrate the segment ribbon without re-running inference.
    """
    if audio is None and video is None:
        raise HTTPException(status_code=400, detail="audio or video required")
    try:
        meta_obj = json.loads(meta)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"meta must be JSON: {e}")
    sid = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = SESSIONS_DIR / sid
    if out_dir.exists():                # avoid collisions on rapid double-save
        sid += "-" + uuid.uuid4().hex[:4]
        out_dir = SESSIONS_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    if audio is not None:
        ext = ".wav" if (audio.filename or "").lower().endswith(".wav") else ".webm"
        path = out_dir / f"audio{ext}"
        with path.open("wb") as f:
            f.write(await audio.read())
        saved["audio"] = path.name
    if video is not None:
        ext = ".webm" if (video.filename or "").lower().endswith(".webm") else ".mp4"
        path = out_dir / f"video{ext}"
        with path.open("wb") as f:
            f.write(await video.read())
        saved["video"] = path.name
    meta_obj["id"] = sid
    meta_obj["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta_obj["files"] = saved
    (out_dir / "meta.json").write_text(json.dumps(meta_obj, indent=2))
    return JSONResponse({"id": sid, "files": saved})


def _dir_size_bytes(path: Path) -> int:
    """Sum of all regular-file sizes under `path`."""
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total


@app.get("/api/list_sessions")
async def list_sessions():
    """Enumerate saved sessions, newest first. Returns at most 50 rows
    plus a `storage` summary (total bytes across ALL saved sessions, plus
    the disk's available free space)."""
    if not SESSIONS_DIR.exists():
        return JSONResponse({"sessions": [], "storage": {
            "total_bytes": 0, "n_sessions": 0,
        }})
    rows = []
    all_dirs = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
    for d in sorted(all_dirs, reverse=True)[:50]:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append({
            "id": d.name,
            "saved_at": meta.get("saved_at", d.name),
            "label": meta.get("label", ""),
            "mode": meta.get("mode", ""),
            "duration_sec": meta.get("duration_sec", 0),
            "files": meta.get("files", {}),
            "size_bytes": _dir_size_bytes(d),
        })
    total_bytes = sum(_dir_size_bytes(d) for d in all_dirs)
    try:
        import shutil as _sh
        free_bytes = _sh.disk_usage(SESSIONS_DIR).free
    except OSError:
        free_bytes = 0
    return JSONResponse({
        "sessions": rows,
        "storage": {
            "total_bytes": total_bytes,
            "n_sessions": len(all_dirs),
            "free_bytes_on_disk": free_bytes,
        },
    })


@app.get("/api/session/{sid}/{filename:path}")
async def get_session_file(sid: str, filename: str):
    """Serve audio / video / meta.json back to the client."""
    if "/" in sid or sid.startswith(".") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid path")
    path = SESSIONS_DIR / sid / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    mime = {
        ".wav":  "audio/wav",
        ".webm": "video/webm" if filename.startswith("video") else "audio/webm",
        ".mp4":  "video/mp4",
        ".json": "application/json",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=mime)


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    """Remove a saved session entirely."""
    if "/" in sid or sid.startswith("."):
        raise HTTPException(status_code=400, detail="invalid sid")
    target = SESSIONS_DIR / sid
    if not target.exists():
        raise HTTPException(status_code=404, detail="not found")
    import shutil
    shutil.rmtree(target)
    return JSONResponse({"deleted": sid})


# ---- catchall: must be the LAST route in this file ----
# The encoder-comparison debug videos sit at the repo root and are
# embedded via src="../debug_compare_*.mp4". This allowlisted passthrough
# resolves only those file patterns; every other path falls through to
# the explicit /api/* and /-mount routes above.
import re as _re_dbg_cmp
_DBG_PATTERNS = (
    _re_dbg_cmp.compile(r"^debug_compare_[A-Za-z0-9._-]+\.mp4$"),
)


@app.get("/{name:path}")
async def _maybe_debug_video(name: str):
    if not any(p.match(name) for p in _DBG_PATTERNS):
        raise HTTPException(status_code=404)
    p = REPO_ROOT / name
    if not p.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(p, media_type="video/mp4")
