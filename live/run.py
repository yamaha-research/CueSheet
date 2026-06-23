"""Run the streaming pipeline from files or from live capture.

Two modes:

    # File replay at 1 s/s (portable, no special hardware)
    uv run python -m live.run --file-audio data/raw/f_7ntJHYAmc.mp3

    # Live microphone (needs sounddevice)
    uv run python -m live.run --live-audio
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from live.audio_pipeline import CLASSES, AudioPipeline


def run_audio_only(audio_source, encoder_device: str):
    pipe = AudioPipeline(device=encoder_device)
    print(f"warming up for {pipe.cold_start_seconds:.0f} s ...")
    n = 0
    t_start = time.perf_counter()
    try:
        for chunk in audio_source:
            n += 1
            t = time.perf_counter()
            for e in pipe.push(chunk):
                wall = (time.perf_counter() - t) * 1000
                top = sorted(zip(CLASSES, e.post6.tolist()),
                             key=lambda x: -x[1])[:2]
                top_str = " ".join(f"{n}:{p:.2f}" for n, p in top)
                print(f"  t={e.t_sec:6.1f}s  segment={CLASSES[e.hmm_label]:13s} "
                      f"top: {top_str}  ({wall:.0f} ms)")
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    finally:
        elapsed = time.perf_counter() - t_start
        print(f"\nprocessed {n} audio chunks in {elapsed:.1f} s wall")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file-audio", type=Path)
    ap.add_argument("--live-audio", action="store_true")
    ap.add_argument("--audio-device", default=None,
                    help="sounddevice input device for --live-audio")
    ap.add_argument("--audio-capture-rate", type=int, default=48_000)
    ap.add_argument("--encoder-device", default="cpu",
                    help="torch device for encoders: cpu | cuda | mps")
    args = ap.parse_args()

    if (args.file_audio is None) == (not args.live_audio):
        ap.error("specify exactly one of --file-audio or --live-audio")

    # build audio source
    if args.file_audio:
        from live.sources import FileAudioSource
        audio_source = FileAudioSource(args.file_audio, paced=True)
    else:
        from live.sources import LiveAudioSource
        audio_source = LiveAudioSource(device=args.audio_device,
                                       capture_rate=args.audio_capture_rate)

    run_audio_only(audio_source, args.encoder_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
