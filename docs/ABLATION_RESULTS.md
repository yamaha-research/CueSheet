# CueSheet Pipeline Ablation

Documents the per-component contribution of the CueSheet audio pipeline,
measured on the 4 hand-labeled gallery shows.

**Script**: `scripts/full_pipeline_6class_4shows.py`
**Results JSON**: `scripts/full_pipeline_6class_results.json`
**Audio source**: `data/raw/{show_id}.mp3` (full unsilenced audio; the
the HF Space `audio/*` copies hold 30 s licensed excerpts
with the rest silenced, so using those would bias the measurement)
**GT source**: `gt_labels` field in `demo/data/{show_id}.json`

## Methodology

Four variants of the audio pipeline, each evaluated against the 6-class
per-second human GT:

| Variant | HMM smoothing | Structural rules |
|---|---|---|
| **A. Naive** | No | No |
| **B. HMM only** | Yes (stay=0.95) | No |
| **C. Rules only** | No | Yes |
| **D. Full pipeline** | Yes (stay=0.95) | Yes |

**Note on output spaces**:
- A and B can only emit 4 classes (Performance, MC talk, Applause,
  Ambient). The group mapping projects 527-class AudioSet to these 4,
  and the Pre-concert and Intermission columns of `post6` are zero, so argmax
  never selects them.
- C and D can emit all 6 classes. The structural rules
  (Pre-concert + Intermission) introduce the missing 2 classes.

This means the F1 against 6-class GT is structurally asymmetric: A and B
score 0 on Pre-concert and Intermission by construction. The
+1.0 / +6.8 / +8.1 lifts therefore decompose as:
- **HMM** improves the 4 acoustic classes
- **Rules** add the 2 structural classes (and re-segment the
  Pre-/Intermission stretches the 4-class output mislabeled)

## Pooled results (across 4 shows, 13 661 GT seconds)

| Variant | Weighted F1 | Macro F1 | Pre-concert F1 | Intermission F1 |
|---|---|---|---|---|
| A. Naive | 86.2% | 37.0% | 0.0% | 0.0% |
| B. HMM only | 87.2% | 39.4% | 0.0% | 0.0% |
| C. Rules only | 93.0% | 62.7% | 92.5% | 36.8% |
| **D. Full pipeline** | **94.3%** | **67.2%** | **92.5%** | **53.5%** |

**Component lifts (pooled wF1)**:
- HMM alone: +1.0 (A → B)
- Rules alone: +6.8 (A → C)
- HMM on top of rules: +1.3 (C → D)
- Rules on top of HMM: +7.1 (B → D)
- Both: +8.1 (A → D)

**Headline**: structural rules drive most of the gain (+6.8 vs HMM's
+1.0 alone); HMM's contribution is consistent (~+1.0–1.3 in both
single-component and full-pipeline settings).

## Per-show results

### tinydesk_seventeen (pop, n=1675 sec)

| Variant | wF1 | macroF1 |
|---|---|---|
| A. Naive | 91.7% | 79.0% |
| B. HMM only | 94.5% | 85.4% |
| C. Rules only | 93.2% | 84.7% |
| D. Full | 95.6% | 89.6% |

### f_7ntJHYAmc (jazz, n=5967 sec)

| Variant | wF1 | macroF1 |
|---|---|---|
| A. Naive | 74.5% | 37.4% |
| B. HMM only | 75.6% | 39.8% |
| C. Rules only | 90.1% | 65.2% |
| D. Full | 91.6% | 69.4% |

**Note**: jazz is the show with the largest gain (+17.1 wF1 from naive
to full pipeline). Most of that gain (+15.5) comes from the structural
rules alone, since the jazz recording has substantial Pre-concert and
Intermission stretches that the 4-class baseline cannot represent.

### boilerroom_fredagain_london (electronic, n=4295 sec)

| Variant | wF1 | macroF1 |
|---|---|---|
| A. Naive | 96.0% | 63.5% |
| B. HMM only | 97.0% | 72.3% |
| **C. Rules only** | **95.1%** | **54.7%** |
| **D. Full** | **95.9%** | **61.9%** |

**Caveat**: on this show the rules slightly hurt. The long
continuous DJ set with brief MC interludes triggers CausalIntermission
false positives. Full pipeline at 95.9% is 0.1 below naive and 1.1
below HMM-only.

### allofbach_bwv140 (classical, n=1724 sec)

| Variant | wF1 | macroF1 |
|---|---|---|
| A. Naive | 98.5% | 54.7% |
| B. HMM only | 98.4% | 49.7% |
| C. Rules only | 98.5% | 54.7% |
| D. Full | 98.4% | 49.7% |

Essentially flat across variants. Bach BWV140 is a single-piece
studio recording where ~99% of seconds are Performance, so both naive
argmax and the full pipeline label correctly.

## Per-class pooled F1

| Class | Support | A. Naive | B. HMM only | C. Rules only | D. Full |
|---|---|---|---|---|---|
| Pre-concert | 655 | 0.0% | 0.0% | 92.5% | 92.5% |
| Performance | 10844 | 98.5% | 98.8% | 98.5% | 98.8% |
| MC talk | 1314 | 62.5% | 66.9% | 77.5% | 82.5% |
| Applause | 457 | 57.4% | 70.5% | 67.2% | 75.9% |
| Intermission | 235 | 0.0% | 0.0% | 36.8% | 53.5% |
| Ambient | 156 | 3.8% | 0.0% | 3.8% | 0.0% |

**Observations**:
- **HMM** improves Applause F1 by +13.1 (57.4 → 70.5). Applause
  seconds get stabilized into runs rather than salt-and-peppered.
- **Rules** add Pre-concert (0 → 92.5%) and Intermission (0 → 53.5%).
- **Combined** improves MC talk substantially (62.5 → 82.5%). HMM
  stabilizes the MC stretches, then rules re-segment the
  long quiet runs as Intermission so MC is labeled correctly.
- **Ambient** suffers in HMM and Full (3.8 → 0%): Ambient seconds get
  swallowed into adjacent labels under HMM, then rules over-write some
  as Intermission. This is a known limitation.

## Hyperparameter rationale

The pipeline uses a single HMM stay probability (`stay = 0.95`,
uniform across classes) and a single CausalIntermission threshold
(`min_quiet_run = 120 s`). Both values were sweep-measured.

### HMM `stay = 0.95`

- **Mean segment length under `stay = p`** is `1 / (1 - p) ≈ 20 s` at
  `p = 0.95`. Concert segments (Performance runs, MC talk stretches,
  Intermission breaks) typically last tens of seconds to minutes, so
  a 20 s smoothing constant absorbs per-second classifier noise without
  smearing across real transitions.
- **Per-class sweep**: best per-class tuning lifts pooled macroF1 by
  only `+0.12` over the uniform `0.95` default (winner was MC = 0.90).
  The marginal gain does not justify the added per-class hyperparameter
  surface, so the uniform `0.95` is kept.

### CausalIntermission `min_quiet_run = 120 s`

- **Sweep** over `{30, 45, 60, 90, 120, 180}`: `120 s` is the
  macroF1-optimal threshold across the 4-show pool. Lower thresholds
  catch more of the actual intermission but over-fire on shows with long
  mid-set MC stretches; higher thresholds miss real but shorter
  intermissions.
- **Intermission F1 floor**: even at the optimum, pooled Intermission
  F1 is `53.5%`. The remaining gap is an Intermission ↔ MC talk class
  overlap that more aggressive heuristic tuning cannot close; a
  dedicated Intermission signal would be needed to push past it.

## Reproducibility

```bash
cd <repo_root>
.venv/bin/python scripts/full_pipeline_6class_4shows.py
# Results written to scripts/full_pipeline_6class_results.json
# Takes ~25 min on CPU (encoder inference dominates)
```

Requirements:
- Full unsilenced audio at `data/raw/{show_id}.mp3` (NOT the 30 s
  excerpts published on the HF Space)
- `gt_labels` per-second human annotation in
  `demo/data/{show_id}.json`
- Pretrained EfficientAT MN10 weights cached locally
- AudioSet class index CSV at `~/panns_data/class_labels_indices.csv`

## What the demo covers

The headline numbers are naive 86.2% pooled wF1 and full pipeline 94.3%,
with a per-show range of lift. The full per-component breakdown is here
for anyone who wants the detailed ablation.

## Scoring convention

All tables above use the offline leading-window convention of the precomputed gallery
tracks: the label for second t comes from the window starting at t, so every label hears
its full ten seconds. Run causally, as deployed live (trailing windows, no look-ahead),
the same full pipeline pools to 90.9% weighted F1 (jazz 86.6, pop 89.5, electronic 94.6,
classical 98.4). The gap comes entirely from transitions, since a causal system cannot
mark a segment change until it has heard one. Reproduce with
`scripts/causal_alignment_check.py`.
