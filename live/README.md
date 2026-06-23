# live/ real-time streaming concert understanding

Streaming counterpart of the offline audio pipeline. The encoder, group
map, and online HMM all run per second on a single CPU (or with the
encoder on MPS/CUDA). Each piece is verified against the offline path on
a labeled show: streaming output equals offline output to within float-32
noise, and labels agree on tested slices.

## Sampling and timing decisions

| | Capture | Internal | Why |
|---|---|---|---|
| Audio | 48 kHz mono float32 (common capture default; MacBook input too) | resample to 32 kHz mono | EfficientAT MN10 pretrained at 32 kHz, 10 second clips. 48->32 polyphase resample is cheap |
| Audio window | — | 10 s context, 1 s hop | matches the offline pretrained input exactly. Hop = 1 s gives per-second output |
| Audio cold start | — | 10 s before first emission | the model needs a full 10 s of real audio |
| End-to-end audio latency | event -> segment | ~1.0 s | hop is dominant; encoder is ~12 ms |

## Files

| | What it does |
|---|---|
| `audio_encoder.py` | Streaming wrapper around the offline EfficientAT model + mel front-end. `predict(window)` per second |
| `online_hmm.py` | One-step forward Viterbi over the 6 segments, same transition matrix as offline |
| `audio_pipeline.py` | Audio: encoder -> 527->6 group map -> online causal Viterbi. Emits `Emission(t_sec, post6, hmm_label)` per second |
| `causal_heuristics.py` | Causal Pre_Concert and post-applause rules applied on top of the smoothed segment track |
| `sources.py` | `FileAudioSource` (paced file replay) and `LiveAudioSource` (sounddevice) |
| `run.py` | CLI: file replay or live microphone |

## Verification (on f_7ntJHYAmc, CPU)

Audio (5 min slice):
```
6-class posterior diff   : max 1.79e-07  mean 3.36e-09  PASS  (threshold 1e-5)
label agreement vs Viterbi (no heur)   : 100.00 %
label agreement vs full pipeline       : 100.00 %
latency per prediction : mean 11.7 ms  p95 12.5 ms  max 19.4 ms
RTF                    : 0.0117  (< 1.0 = real-time feasible)
```

The first 300 s of f_7ntJHYAmc are pre-event but the audio path emits
MC_Talk there (crowd chatter is what AudioSet can hear). A causal
Pre_Concert detector that mirrors the offline post-Viterbi heuristic
runs on top of the smoothed track to correct this lead-in.

## Run order

```bash
# real time replay (file, paced at 1 s/s)
uv run python -m live.run --file-audio data/raw/f_7ntJHYAmc.mp3

# live microphone
uv run python -m live.run --live-audio
```

## MacBook setup

Apple Silicon laptop, anything with PortAudio:
```bash
brew install portaudio                # PortAudio backend for sounddevice
uv add sounddevice                    # rest already in the project env
```

Audio encoder runs comfortably on CPU (~12 ms per window):
```bash
uv run python -m live.run --live-audio --encoder-device mps
```

## Browser viewer

The demo server (`demo/server.py`) serves the static frontend for the
real-time viewer:

  - `GET /static/cuesheet.html`: the real-time viewer (auto-served by the
    existing static mount). Shows the current segment, the 6-class posterior
    bar, and the segment ribbon
  - `WS /api/infer_live`: streams per-second JSON from a live microphone
    (the browser captures mic PCM and the server runs the audio pipeline)

Run:
```bash
# server
uv run uvicorn demo.server:app --port 8001

# open in browser:
# http://localhost:8001/static/cuesheet.html
```

## Audio device path

`LiveAudioSource` accepts any sounddevice input device by index or name
substring. Whatever the OS exposes as an audio input device is selectable
with `--audio-device "<name>"`. No code change.

## Backbone licenses

The backbone used at inference time is under a permissive open-source
license:

| Model | License | Notes |
|---|---|---|
| EfficientAT MN10 (audio encoder) | MIT (Schmid et al.) | Pretrained on AudioSet (CC BY 4.0 labels) |
| Our HMM, group mapping | Ours | Research code |

## Component status

| | Status |
|---|---|
| Streaming audio encoder | Verified bit-exact vs offline on a real show |
| Online HMM | 100 % label agreement on 5 min slice vs offline Viterbi+heuristics |
| Live capture | sounddevice (audio). Microphone path verified by running `--live-audio` |
| Audio device integration | Any sounddevice input device, no code change expected |

Future work: tighter post-applause smoothing on top of the existing
causal heuristics.
