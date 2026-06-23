# Concert Segment Detection (CueSheet)

Per-second classification of concert recordings into six concert segments.
This repository holds the audio encoders, weak-label bootstrapping, the
fusion/temporal model code, and a zero-training baseline.

## Task

Label every second of a concert recording with one of six segments:

**Pre_Concert / Performance / MC_Talk / Applause / Intermission / Ambient**

Segment definitions:

- **Pre_Concert**: before the show starts. A temporal segment covering everything ahead of the first sustained Performance run.
- **Performance**: active music being performed. Mid-performance applause (for example after a solo while the band keeps playing) stays in Performance.
- **MC_Talk**: a single host speaking on stage. Not multi-person audience chatter and not PA announcements. Brief audience reactions during MC speech are absorbed into MC_Talk.
- **Applause**: audience clapping or cheering as a standalone segment, when the music has stopped.
- **Intermission**: a formal scheduled mid-show break. This segment has no AudioSet equivalent and needs human labels.
- **Ambient**: room tone between active events during the concert. Distinct from Pre_Concert, which is its own temporal segment.

## Quick start (no training)

The fastest way to segment a recording. No labels and no GPU training required.
PANN's CNN14 checkpoint is downloaded automatically by `panns-inference` on
first run.

```bash
# 1. Run the zero-training PANN baseline on an audio file.
uv run python scripts/pann_baseline.py path/to/your_concert.mp3 \
    --output predictions.json
```

`pann_baseline.py` maps PANN's per-window AudioSet probabilities onto the
six-class scheme, runs a Viterbi HMM smoother, and exports the resulting
segments as JSON. Pre_Concert and Intermission have no AudioSet equivalent
and stay near zero in this baseline.

To produce weak per-second labels from the same PANN probabilities (plus
librosa beat density), use the bootstrap labeler:

```bash
uv run python scripts/bootstrap_labels.py path/to/your_concert.mp3 \
    --output-dir data/labels/bootstrap
```

It writes a `<stem>_labels.npz` whose schema matches a Label Studio export,
so it can be consumed directly by the dataset loader.

## What's in here

| Path | Purpose |
|---|---|
| `scripts/pann_baseline.py` | Zero-training baseline: PANN AudioSet probabilities to six-class groups to HMM smoothing. Exports segments as JSON. |
| `scripts/bootstrap_labels.py` | Generate weak per-second labels by aggregating PANN AudioSet probabilities and librosa beat density into a `<stem>_labels.npz`. |
| `scripts/convert_labels.py` | Convert a Label Studio JSON-MIN export to a per-second label `.npz`. |
| `scripts/compare_to_gt.py` | Compare a predictions JSON against hand-noted ground-truth segments and print a per-second confusion breakdown over the covered ranges. |
| `scripts/encoder_ast.py` | AST (Audio Spectrogram Transformer) wrapper returning per-second AudioSet posteriors, with the same signature as the PANN classifier. |
| `scripts/encoder_efficientat.py` | EfficientAT (MobileNet-style, edge-oriented) AudioSet encoder adapter. Requires the vendored EfficientAT repo. |
| `scripts/encoder_htsat.py` | HTS-AT (Hierarchical Token-Semantic Audio Transformer) AudioSet encoder wrapper. Requires the vendored HTS-AT repo and checkpoint. |
| `src/audio/extract.py` | Extract per-second audio embeddings from an audio file using AST, PANN, CLAP, wav2vec2, HuBERT, or MERT. |
| `src/fusion/` | Multi-encoder fusion model, two-stage classifier, and HMM/CRF temporal smoothing (offline Viterbi and online causal). |
| `src/utils/` | Config loading, dataset/dataloader, and evaluation metrics. |
| `configs/default.yaml` | Single source of truth for window/hop, encoder list, class names, and loss weights. |

The encoder wrappers all return `(n_seconds, 527)` AudioSet posteriors on a
common 1-second grid, so they are interchangeable upstream of the group
mapping and HMM smoother.

## Configuration

`configs/default.yaml` is the single source of truth. To change the active
encoder set:

```yaml
fusion:
  active_encoders: [ast, pann]    # any subset of ast/pann/hubert/mert
```

## Hardware notes

- **Apple Silicon (MPS)**: use `num_workers=0` in dataloaders. Multi-worker dataloaders hang on MPS. This is already set in `src/utils/dataset.py`.
- **CUDA**: standard, no caveats.
- **CPU only**: feature extraction takes roughly 5 to 15 minutes per hour of audio per encoder.

## License

Apache-2.0. See [../LICENSE](../LICENSE).
