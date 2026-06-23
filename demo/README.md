# CueSheet demo

A live browser demo of multi-genre concert segment detection. Pick a show
from the preloaded gallery, drop in your own audio or video file, or run
the pipeline on a live microphone. Every second of audio is labeled
with one of six concert segments and shown on a glanceable interface.

## Quick start (local)

Requirements: [uv](https://docs.astral.sh/uv/), `ffmpeg`, `git`.

```sh
bash demo/run.sh
```

Then open <http://localhost:8001/cuesheet>. The script clones the
EfficientAT encoder into `vendor/`, syncs dependencies, and starts the
server. Set `PORT` to use another port.

## HTTPS (mic / camera from other devices)

Browsers block `getUserMedia` on plain-HTTP origins other than
`http://localhost`. To use the live microphone or camera from another
device, either serve the demo behind HTTPS (any reverse proxy with a
certificate works) or SSH-tunnel the port to the client machine
(`ssh -L 8001:localhost:8001 <host>`) and open
<http://localhost:8001/cuesheet> there.

## Standalone (Docker)

```sh
docker build -t cuesheet -f demo/Dockerfile .
docker run --rm -p 8001:8001 cuesheet
```

Then open <http://localhost:8001/cuesheet>.

## What runs

- **Preloaded gallery.** Four concert recordings, one per genre, with
  precomputed per-second segment tracks.
- **File upload.** Drop an audio or video file and the pipeline streams
  per-second predictions back as it runs.
- **Live microphone.** Click `Live mic` to stream the microphone
  through the online causal pipeline in real time.
- **Fullscreen.** The `Fullscreen` button or the `F` key scales the
  interface to fill the screen.

## Model weights

- The **EfficientAT MN10** encoder weights download automatically on the
  first inference (network access needed once).
- AudioSet class display names come from
  `~/panns_data/class_labels_indices.csv` if present; without it the raw
  output panel shows numeric class ids instead of names.
