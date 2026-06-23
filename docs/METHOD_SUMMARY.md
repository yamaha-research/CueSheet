# Concert Segment Detection Method

**Scope:** CueSheet concert segment detection (method summary)

## Problem statement

Per-second classification of a concert recording into six structural segments (`Pre_Concert`, `Performance`, `MC_Talk`, `Applause`, `Intermission`, `Ambient`) from the concert audio, running causally and in real time on a single CPU (target: a mixing console or edge device, no GPU).

## The audio idea: zero-training AudioSet group-map + HMM

A 1-second window goes through EfficientAT MN10 (~5 M params, AudioSet-pretrained, ~12 ms per window (~85x realtime) on CPU), producing a 527-class semantic posterior. We sum the posterior over four AudioSet subtrees (Music + Musical instrument → `Performance`; single-speaker Speech / Narration → `MC_Talk`; Clapping / Cheering / Crowd + the empirically-validated Rain triplet → `Applause`; Silence → `Ambient`) and smooth the resulting 6-class track with an online causal HMM (`stay_prob = 0.95`). **No training is required for the four acoustic classes, since AudioSet's ontology already aligns with them.** `Pre_Concert` and `Intermission` are temporal heuristics (before the first sustained `Performance` run; long-quiet stretches mid-show) because **AudioSet has no acoustic concept for either**, measured at ≤ 0.3 % / 0 % across the evaluated shows. Strong on Applause and music-vs-speech, structurally blind on Pre_Concert / Intermission.

## Default configuration

```
audio expert : EfficientAT MN10 + AudioSet group-map + HMM (stay 0.95)
heuristics   : causal Pre_Concert + post-applause (CausalPreConcert / CausalPostApplause)
               plus offline Pre_Concert / Intermission temporal heuristics
smoother     : online causal HMM on the per-second track
```

## CPU cost (desktop-class CPU)

| Expert | Cost / sec | Mode |
|---|---|---|
| Audio (EfficientAT MN10) | ~12 ms per window (~85x realtime) | always-on backbone |

The audio expert runs at ~12 ms per window (~85x realtime) on a desktop-class CPU, well within real-time budget on a single core with no GPU.

## What is implemented vs research scope

**Implemented:** Audio expert (EfficientAT MN10 + AudioSet group-map), online causal HMM smoothing, causal Pre_Concert + post-applause heuristics, the offline Pre_Concert / Intermission temporal heuristics, and the demo end-to-end with GT-vs-prediction ribbon, inline segment criteria panel, and ribbon click → seek → widget alignment.

**Research scope:** song-structure segmentation within the Performance segment, and genre-conditioned thresholds across all ten CONCERT-10 genres.

## Open weaknesses

1. **`Intermission` F1 is low.** AudioSet has no concept for it, and the post-applause causal heuristic does not cover audience-milling. This is the most concrete target for further work.
2. **`Pre_Concert` is a label problem, not an encoder one.** AudioSet has no acoustic concept for it, so the audio expert relies on a temporal heuristic that over-predicts; the structural signal is present but unsupervised.
3. **Recording quality varies across shows.** Noisier captures lower pooled audio-only and may understate the audio expert's strength on typical recordings.
4. **Heuristic thresholds are global.** Different venues / genres need different thresholds; the adaptive silence gate partially addresses this.
5. **CONCERT-10 v1.0 has four hand-labeled gallery shows.** Generalization to opera / festival / sponsor-heavy formats is an explicit revision trigger, not a blocker.

## Demo

- Live demo: <https://yonghyunk1m-cue-sheet.hf.space> (gallery with 30-second audible excerpts, user audio upload, encoder swap, live mic).
- The ribbon shows GT (top strip) vs prediction (bottom strip); the audible-window cyan marker on the public mirror pins the music-onset second exactly.
- The per-second decision trace in the Current Segment card surfaces which rule fired for the current second.

## References

- [ABLATION_RESULTS.md](ABLATION_RESULTS.md): per-component audio-pipeline ablation
- [scripts/full_pipeline_6class_4shows.py](../scripts/full_pipeline_6class_4shows.py): reproduces the component ablation
- [scripts/causal_alignment_check.py](../scripts/causal_alignment_check.py): causal (live-path) scoring convention
- [scripts/multi_encoder_precompute.py](../scripts/multi_encoder_precompute.py): per-encoder gallery posteriors
- [scripts/compute_encoder_scores.py](../scripts/compute_encoder_scores.py): per-encoder pooled wF1 numbers
