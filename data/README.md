# CONCERT-10 v1.0

A multi-genre concert dataset released as a URL manifest plus per-second labels.
We distribute links and annotations, not the source media.

```
data/
├── CONCERT-10.tsv                       # URL manifest (26 concerts)
└── labels/
    ├── bootstrap/<show_id>_labels.npz    # machine pseudo-labels, all 26 concerts
    └── human/<show_id>_labels.npz        # human-verified labels, 4 concerts so far (growing)
```

Fetch the audio with `scripts/download_data.sh` (yt-dlp). It writes
`data/raw/<show_id>.mp3` for every row in the manifest.

## `CONCERT-10.tsv`

Tab-separated. The first line is the column header, followed by 26 data rows.

| column | type | description |
| --- | --- | --- |
| `show_id` | str | stable id. Matches the `.npz` filename and the demo show JSONs. |
| `url` | str | source video URL (YouTube) |
| `genre` | str | one of the ten genre tags |
| `duration_sec` | int | length of the source recording, in seconds |
| `notes` | str | free-text (artist, event) |

26 concerts, about 21.8 h total (580 to 10414 s each), ten genres: Classical (7),
Jazz (3), and Christian/Gospel, Country, Dance/Electronic, Hip-Hop/R&B, Latin,
Pop, Rock/Alternative, World/Cross-cultural (2 each).

## Label files (`.npz` schema)

Both `bootstrap/` and `human/` use the same NumPy-archive layout. Each file holds
four arrays, one entry per second (`T` = number of seconds).

| array | dtype | shape | meaning |
| --- | --- | --- | --- |
| `labels` | int64 | `(T,)` | class index into `classes`, or `-1` (bootstrap only) |
| `classes` | str | `(6,)` | class names, in index order |
| `timestamps` | float64 | `(T,)` | window-start second (`0, 1, ..., T-1`) |
| `source` | str | `()` | provenance tag |

Class index order: `0` Pre_Concert, `1` Performance, `2` MC_Talk, `3` Applause,
`4` Intermission, `5` Ambient.

```python
import numpy as np

d = np.load("data/labels/human/f_7ntJHYAmc_labels.npz")
labels, classes = d["labels"], d["classes"]
names = [classes[i] if i >= 0 else "UNLABELED" for i in labels]   # -1 safe
```

### What a file looks like

```text
>>> d = np.load("data/labels/bootstrap/f_7ntJHYAmc_labels.npz")
>>> d.files
['labels', 'classes', 'timestamps', 'source']
>>> list(d["classes"])
['Pre_Concert', 'Performance', 'MC_Talk', 'Applause', 'Intermission', 'Ambient']
>>> str(d["source"]), d["labels"].shape
('pann_audioset_bootstrap', (6009,))

# decoded, seconds 652 to 656: a Performance to MC_Talk boundary with one -1
 t(s)  label  class
  652      1  Performance
  653      1  Performance
  654      1  Performance
  655     -1  UNLABELED      # bootstrap abstained at the transition
  656      2  MC_Talk
```

The `human/` files read the same way, except `source` is `human_verified` and
`labels` never contains `-1`.

## `labels/bootstrap/` (machine pseudo-labels, all 26 concerts)

Generated training-free by
[`cuesheet/scripts/bootstrap_labels.py`](../cuesheet/scripts/bootstrap_labels.py),
with `source = pann_audioset_bootstrap`:

1. decode to 16 kHz mono.
2. run **PANN CNN14** (frozen, AudioSet-pretrained) over 2 s windows at a 1 s
   hop, giving 527-class AudioSet posteriors, one row per second.
3. sum those posteriors into four acoustic concert groups by AudioSet keyword.
   Performance from Music and instruments, MC_Talk from single-speaker speech,
   Applause from clapping, cheering, and crowd, Ambient from silence.
4. per second, take the strongest group if its score clears a **0.3 confidence
   threshold**, otherwise emit `-1`.
5. relabel the lead-in (before the first sustained Performance or MC_Talk run)
   as Pre_Concert with a temporal rule.

No training and no human input. Two consequences to know.

**Intermission never appears** in bootstrap labels. It has no acoustic signature,
so the live pipeline adds it with a separate rule, not this script.

**`labels` contains `-1`** for low-confidence or unlabeled seconds. This happens
when the top group falls below the 0.3 threshold, or when the second is
Pre_Concert or Intermission, which AudioSet cannot match. `-1` is not a class,
and it is not Ambient. About **9.6%** of bootstrap seconds (7,544 of 78,468) are
`-1`. Treat it as an ignore index. Do not decode with `classes[label]` directly,
because Python negative indexing would silently turn `-1` into `Ambient` (the
last class).

The generator also records a per-second `confidence` array (the winning group's
summed AudioSet score, the value compared against the 0.3 gate) so you can
re-threshold or weight by it. The v1.0 release files predate this field and do
not include it. It ships with the regenerated v1.x labels.

## `labels/human/` (human-verified labels, 4 concerts so far)

The four gallery shows (`tinydesk_seventeen`, `f_7ntJHYAmc`,
`boilerroom_fredagain_london`, `allofbach_bwv140`) additionally have per-second
labels checked by a human, with `source = human_verified`. These are complete
six-class tracks with no `-1`, and they are the ground truth the paper's
Section 4 numbers are scored against. The same labels are embedded in the demo
payloads at [`demo/data/<show_id>.json`](../demo/data/) under `gt_labels`.

This set is a work in progress. We are hand-labeling the remaining CONCERT-10
concerts and will add them under `human/` in future v1.x releases.

## License

The CONCERT-10 manifest and per-second labels are released under **CC BY 4.0**.
Source media stays under each uploader's YouTube license (as of 2026-06-30). See
the [repository README](../README.md#license) for the full source-media
compliance note.
