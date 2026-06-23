"""Generate weak labels by aggregating PANN AudioSet probabilities per window.

Maps the 527 AudioSet classes onto four acoustically grounded concert segments
(see CLASS_GROUPS): MC_Talk, Performance, Applause, Ambient. The two
time-defined segments, Pre_Concert and Intermission, are not acoustic and are
added downstream by temporal rules. Windows with no confident group match are
rejected as -1 (low-confidence).

The output is a `<stem>_labels.npz` with the same schema as Label-Studio exports,
so it slots straight into ConcertWindowDataset as `pseudo_label_dir`.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.utils.config import load_config

TARGET_SR = 16_000
WINDOW_SEC = 2.0
HOP_SEC = 1.0

CLASS_GROUPS: dict[str, list[str]] = {
    # MC_Talk: single-speaker on-stage announcement. Excludes "Conversation"
    # (multi-person) and "Crowd" (which maps to Applause) so audience murmur
    # does not leak in.
    "MC_Talk":     ["Speech", "Narration, monologue", "Speech synthesizer",
                    "Male speech, man speaking", "Female speech, woman speaking",
                    "Child speech, kid speaking"],
    # Performance: any AudioSet class that names a music genre, ensemble,
    # vocal form, or instrument. Membership is semantic — the encoder calls
    # music music, and the Intermission rule handles between-set PA music from
    # the time pattern rather than from the audio.
    "Performance": ["Music", "Musical instrument",
                    # Ensembles / classical
                    "Orchestra", "Opera",
                    # Jazz / blues / soul
                    "Jazz", "Blues", "Rhythm and blues", "Funk",
                    "Soul music",
                    # Rock
                    "Heavy metal", "Punk rock", "Progressive rock",
                    "Rock and roll", "Psychedelic rock", "Grunge",
                    # World / pop
                    "Country", "Reggae", "Ska", "Techno",
                    "Disco", "Afrobeat",
                    # Generic vocal music
                    "Song", "Lullaby",
                    # Instruments / vocal tokens missed by keyword matching
                    "Bagpipes", "Wood block", "Rattle (instrument)",
                    "Wind chime", "Sitar", "Hi-hat", "Gong",
                    "Didgeridoo", "Theremin", "Beatboxing"],
    # Applause: audience-reaction subtree (clapping, cheering, shouting,
    # whistling). The "Rain" classes share applause's percussive
    # many-small-events timbre and never occur in indoor concert audio.
    "Applause":    ["Clapping", "Cheering", "Applause", "Crowd",
                    "Shout", "Whoop", "Yell", "Children shouting",
                    "Screaming",
                    "Laughter", "Chuckle, chortle",
                    "Whistle",
                    "Rain on surface", "Rain", "Raindrop"],
    # Ambient: AudioSet silence plus the venue-acoustic subtree (room and
    # hall reverb, public-space hum) and audience-babble classes that fire
    # during quiet moments between active events.
    "Ambient":     ["Silence",
                    "Inside, small room", "Inside, large room or hall",
                    "Inside, public space", "Reverberation", "Echo",
                    "Hubbub, speech noise, speech babble",
                    "Chatter", "Children playing",],
}


def load_audioset_index() -> dict[str, int]:
    csv_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return {row["display_name"]: int(row["index"]) for row in reader}


# Anything whose AudioSet display name contains one of these (case-insensitive)
# is rolled into the Performance group. Without this, PANN's specific-instrument
# probabilities (Saxophone / Drum kit / Piano / Violin / Singing / …) wouldn't
# count toward Performance and many active-music windows fall below threshold.
_PERFORMANCE_KEYWORDS = (
    "music",                                                         # general music + genres
    "musical instrument",
    "guitar", "drum", "piano", "violin", "viola", "cello",
    "double bass", "bass guitar",
    "trumpet", "saxophone", "trombone", "flute", "clarinet", "oboe",
    "horn", "tuba", "organ", "synthesizer", "harpsichord", "harp",
    "banjo", "mandolin", "ukulele", "accordion", "harmonica",
    "vibraphone", "marimba", "xylophone", "glockenspiel", "bell",
    "cymbal", "tambourine", "snare", "kick drum", "bass drum",
    "bongo", "conga", "timpani", "tabla",
    "string section", "string instrument", "brass instrument",
    "woodwind instrument", "percussion",
    "singing", "yodeling", "rapping", "humming", "chant", "vocal",
    "choir", "a capella",
)

# Substring matching against _PERFORMANCE_KEYWORDS catches some AudioSet
# classes that have a music-sounding word in their name but are NOT
# musical (animal vocalizations, vehicle horns, appliance bells, the
# Speech synthesizer assistive-tech class, etc.). These are enumerated
# below and explicitly subtracted from the Performance group after
# keyword expansion. Add a class here only when it has no musical sense
# in any reasonable interpretation.
_PERFORMANCE_EXCLUDE_NAMES = frozenset({
    # 'synthesizer' caught the speech-synthesis class
    "Speech synthesizer",
    # 'bell' caught animal / appliance / vehicle bells
    "Bellow",                                          # animal/loud noise
    "Belly laugh",                                     # laughter
    "Bicycle bell", "Doorbell",
    "Telephone bell ringing",
    # 'horn' caught vehicle horns
    "Vehicle horn, car horn, honking",
    "Air horn, truck horn",
    "Train horn", "Foghorn",
    # 'vocal' / 'song' caught animal vocalizations
    "Bird vocalization, bird call, bird song",
    "Whale vocalization",
})


def build_group_indices(name_to_idx: dict[str, int]) -> dict[str, list[int]]:
    """Resolve display-name groups → AudioSet integer indices.

    For 'Performance', we additionally pull in every class whose display name
    contains a musical-instrument or music-related keyword. That picks up
    ~70 specific-instrument classes that wouldn't otherwise contribute to
    the Performance score. Then we subtract _PERFORMANCE_EXCLUDE_NAMES
    so substring false positives (Speech synthesizer / Vehicle horn /
    Bird vocalization / etc.) do not pollute the group.
    """
    groups: dict[str, list[int]] = {}
    excluded_ids = {name_to_idx[n] for n in _PERFORMANCE_EXCLUDE_NAMES
                    if n in name_to_idx}
    for grp, names in CLASS_GROUPS.items():
        idxs = [name_to_idx[n] for n in names if n in name_to_idx]
        if grp == "Performance":
            idxs += [
                i for n, i in name_to_idx.items()
                if any(kw in n.lower() for kw in _PERFORMANCE_KEYWORDS)
            ]
            idxs = [i for i in idxs if i not in excluded_ids]
        groups[grp] = sorted(set(idxs))
    return groups


def extract_wav(src: Path, tmp_dir: str) -> Path:
    out = Path(tmp_dir) / (src.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(TARGET_SR), str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def classify_windows(wav_path: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    from panns_inference import AudioTagging
    model = AudioTagging(checkpoint_path=None, device=device)

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    assert sr == TARGET_SR, f"expected {TARGET_SR} Hz, got {sr}"

    win_n = int(WINDOW_SEC * TARGET_SR)
    hop_n = int(HOP_SEC * TARGET_SR)
    starts = list(range(0, len(wav) - win_n + 1, hop_n))

    probs, timestamps = [], []
    for i, s in enumerate(starts):
        chunk = wav[s : s + win_n][np.newaxis, :]
        with torch.no_grad():
            clipwise, _ = model.inference(chunk)
        probs.append(clipwise.squeeze(0))
        timestamps.append(s / TARGET_SR)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(starts)}")

    return np.stack(probs), np.array(timestamps)


def compute_ambient_blind_onset(posteriors: np.ndarray,
                                 class_names: list[str]) -> np.ndarray:
    """Argmax of the per-second posterior with the Ambient column
    zeroed. Use this as ``onset_labels`` for
    ``apply_pre_concert_heuristic`` when the Ambient group-map has
    been expanded beyond AudioSet's narrow ``Silence`` detector — the
    venue-acoustic subtree dominates Performance argmax in early-song
    seconds, drifts ``show_start`` late, and forces real Performance
    seconds to be relabeled Pre_Concert.

    Zeroing the Ambient channel for *onset detection only* lets
    Performance win the argmax whenever it has any non-Ambient mass.
    The cascade and HMM still see the full posterior — Ambient is only
    invisible to the show_start cursor.
    """
    if "Ambient" not in class_names:
        return posteriors.argmax(axis=-1)
    masked = posteriors.copy()
    masked[..., class_names.index("Ambient")] = 0.0
    return masked.argmax(axis=-1)


def apply_pre_concert_heuristic(labels: np.ndarray, class_names: list[str],
                                 active_classes=("Performance",),
                                 min_active_run: int = 10,
                                 onset_labels: np.ndarray | None = None) -> np.ndarray:
    """Re-label the show's lead-in as Pre_Concert.

    Pre_Concert is *temporally* defined — everything before music actually
    starts. We derive that from PANN's per-window labels without needing the
    user to hand-mark every pre-show region:

    1. Walk forward in time and find the first sustained run of Performance
       (>= `min_active_run` consecutive windows by default).
    2. Override every earlier window in ``labels`` to Pre_Concert, regardless
       of what the smoother locally thought it was.

    We deliberately use ONLY Performance, not MC_Talk. Pre-concert PA
    announcements ("please be seated") and brief host welcomes look identical
    to MC_Talk to PANN (single-person speech at a mic) but are temporally
    Pre_Concert. Music-onset is the unambiguous "show started" signal.

    The min-run guard avoids treating a single stray Music-classified window
    in the pre-show audience-chatter region as "the show started."

    ``onset_labels``: optional separate label sequence used *only* for
    detecting the music-onset cursor (the ``show_start`` index). The label
    overrides are still written into ``labels``. Pass the raw argmax of the
    6-class posterior here when ``labels`` came from an offline smoother
    (Viterbi) that back-smoothes Performance to t=0 on music-dominated
    shows — without a separate onset signal the heuristic gives up at line
    "show_start == 0" and Pre_Concert is silently dropped. Defaults to
    ``labels`` itself for back-compat.
    """
    if "Pre_Concert" not in class_names:
        return labels

    name_to_id = {c: i for i, c in enumerate(class_names)}
    active_ids = {name_to_id[c] for c in active_classes if c in name_to_id}

    onset_src = onset_labels if onset_labels is not None else labels
    if len(onset_src) != len(labels):
        raise ValueError(
            f"onset_labels length {len(onset_src)} != labels length {len(labels)}"
        )

    run_len = 0
    show_start = None
    for t, lab in enumerate(onset_src):
        if lab in active_ids:
            run_len += 1
            if run_len >= min_active_run:
                show_start = t - run_len + 1
                break
        else:
            run_len = 0

    if show_start is None or show_start == 0:
        return labels   # never found a sustained active run, or onset is at t=0

    labels = labels.copy()
    labels[:show_start] = name_to_id["Pre_Concert"]
    return labels


def apply_post_performance_applause_heuristic(
        labels: np.ndarray, class_names: list[str],
        applause_posterior: np.ndarray | None = None,
        min_perf_run: int = 40,
        window_sec: int = 12,
        min_post_evidence: float = 0.30,
) -> np.ndarray:
    """Structural Applause detector for shows where PANN's posterior is too
    weak for the boost to fire.

    Premise: applause happens *right after* sustained Performance ends. On
    close-mic / studio-style recordings PANN's
    AudioSet-trained Applause group fires with near-zero probability — but
    the structural position of applause in the show is unambiguous. We use
    that position as the dominant cue and require only a tiny amount of
    Applause-group posterior energy to confirm it's not silence.

    For each Performance run ≥ ``min_perf_run`` seconds, the next
    ``window_sec`` seconds are re-labeled as Applause **only if** all of:
      - the current label is not already Performance (we never overwrite
        a new song starting); and
      - if ``applause_posterior`` is provided, the per-second score for
        that second is ≥ ``min_post_evidence``. The default 0.30 floor
        is the LOSO macroF1 optimum; a lower floor is too permissive and
        relabels MC_Talk and Ambient seconds inside the post-Performance
        window.

    Pass ``applause_posterior=None`` to disable the evidence check (pure
    structural override).
    """
    if "Performance" not in class_names or "Applause" not in class_names:
        return labels
    name_to_id = {c: i for i, c in enumerate(class_names)}
    perf_id = name_to_id["Performance"]
    app_id = name_to_id["Applause"]

    out = labels.copy()
    in_run = False
    run_start = 0
    T = len(labels)
    for t in range(T):
        if labels[t] == perf_id and not in_run:
            in_run = True
            run_start = t
        elif labels[t] != perf_id and in_run:
            in_run = False
            run_len = t - run_start
            if run_len < min_perf_run:
                continue
            # Re-label the next window_sec seconds, but stop at the next
            # Performance boundary (don't overwrite the next song).
            end = min(t + window_sec, T)
            for j in range(t, end):
                if labels[j] == perf_id:
                    break
                if applause_posterior is not None:
                    if applause_posterior[j] < min_post_evidence:
                        continue
                out[j] = app_id
    # If the recording ended inside a Performance run, no post-window — fine.
    return out


def map_to_classes(probs: np.ndarray, group_idx: dict[str, list[int]],
                   class_names: list[str], rel_pos: np.ndarray,
                   beat_density: np.ndarray | None = None,
                   confidence: float = 0.3,
                   ambient_beat_thresh: float = 0.5) -> np.ndarray:
    """Pick the strongest group per window; return per-window class id (or -1).

    With `beat_density` provided, we add a beat-aware ambient rule: a window
    with essentially no beats and no dominant speech/music gets labeled Ambient
    even if PANN didn't find its 'Silence' class strongly. This recovers
    meaningful coverage between active segments.

    Applause maps directly to the Applause class, which keeps long applause
    from being miscategorized as Intermission. Pre_Concert and
    Intermission have no PANN equivalent and stay -1 (need human labels).
    """
    T = len(probs)
    group_score = np.stack([
        probs[:, group_idx[g]].sum(axis=-1) for g in CLASS_GROUPS
    ], axis=-1)
    best = group_score.argmax(axis=-1)
    best_score = group_score.max(axis=-1)

    speech_score = group_score[:, list(CLASS_GROUPS).index("MC_Talk")]
    music_score  = group_score[:, list(CLASS_GROUPS).index("Performance")]

    name_to_id = {c: i for i, c in enumerate(class_names)}
    out = np.full(T, -1, dtype=np.int64)
    group_names = list(CLASS_GROUPS)

    for t in range(T):
        # Beat-aware Ambient: very few beats + neither speech nor music dominant
        if (beat_density is not None and beat_density[t] < ambient_beat_thresh
                and speech_score[t] < 0.2 and music_score[t] < 0.2):
            out[t] = name_to_id.get("Ambient", -1)
            continue

        if best_score[t] < confidence:
            continue
        grp = group_names[best[t]]
        # MC_Talk / Performance / Applause / Ambient — each maps to its own class.
        if grp in name_to_id:
            out[t] = name_to_id[grp]

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path, help="audio or video file")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("data/labels/bootstrap"))
    p.add_argument("--device", default=None)
    p.add_argument("--confidence", type=float, default=0.3,
                   help="minimum group score to assign a label")
    args = p.parse_args()

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    name_to_idx = load_audioset_index()
    group_idx = build_group_indices(name_to_idx)
    print("AudioSet group sizes:")
    for g, idxs in group_idx.items():
        print(f"  {g:<12} {len(idxs)} classes")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = extract_wav(args.audio, tmp)
        print(f"Classifying {args.audio.name} on {device}...")
        probs, timestamps = classify_windows(wav_path, device)

    rel_pos = timestamps / max(timestamps[-1], 1.0)

    beat_path = Path("data/labels/beats") / f"{args.audio.stem}_beats.npz"
    beat_density = None
    if beat_path.exists():
        beat_density = np.load(beat_path)["beat_density"]
        print(f"  Using beat density from {beat_path.name}")

    labels = map_to_classes(probs, group_idx, config.fusion.classes, rel_pos,
                            beat_density=beat_density,
                            confidence=args.confidence)

    # Temporal heuristic: anything before the first sustained Performance/MC_Talk
    # run is Pre_Concert. Removes the need for humans to mark every show's lead-in.
    labels = apply_pre_concert_heuristic(labels, config.fusion.classes)

    # Per-second confidence: the winning group's summed AudioSet score, i.e. the
    # value map_to_classes compares against args.confidence before it emits -1.
    # Saved so consumers can re-threshold or weight by it.
    group_score = np.stack(
        [probs[:, group_idx[g]].sum(axis=-1) for g in CLASS_GROUPS], axis=-1)
    confidence = group_score.max(axis=-1).astype(np.float32)

    out_path = args.output_dir / f"{args.audio.stem}_labels.npz"
    np.savez(out_path, labels=labels, classes=config.fusion.classes,
             timestamps=timestamps, confidence=confidence,
             source="pann_audioset_bootstrap")

    coverage = (labels >= 0).mean()
    print(f"\nSaved → {out_path}")
    print(f"Coverage: {coverage:.1%} ({(labels >= 0).sum()}/{len(labels)} windows labeled)")
    for i, c in enumerate(config.fusion.classes):
        n = (labels == i).sum()
        print(f"  {c:<12} {n:>5}  ({n / len(labels):.1%})")


if __name__ == "__main__":
    main()
