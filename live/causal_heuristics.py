"""Causal (online, no-look-ahead) versions of the offline Pre_Concert and
post-applause heuristics from `scripts/bootstrap_labels.py`.

The offline versions walk the *full* per-second label array to find a
sustained Performance run and back-label everything before it as
`Pre_Concert` (or to find a Performance run's end and forward-label the
next ~12 s as `Applause`). Both are O(T) one-pass operations but they
look at future seconds to decide a past label, which breaks live use:
the scheduler needs to emit the final label for second `t` at time `t`
without revising earlier emissions.

This module provides two small state machines that produce the same
labels *as they arrive*, accepting that the price of being causal is a
small bounded error window at each transition:

  - `CausalPreConcert`: emits `Pre_Concert` until it has seen
    `min_active_run` consecutive `Performance` seconds, then commits to
    "concert started" and passes the encoder's labels through unchanged.
    Bounded error: the first `min_active_run - 1` Performance seconds
    have already been emitted as `Pre_Concert` and cannot be revised.
    With the default `min_active_run = 10`, that is a 9-second
    misclassification at the show's opening transition.

  - `CausalPostApplause`: tracks the current Performance run length;
    when it sees a `Performance -> non-Performance` transition with a
    run length >= `min_perf_run`, it opens an `applause_window_sec`
    override that re-labels subsequent non-Performance encoder
    predictions as `Applause` (with optional posterior-floor evidence
    check). The window closes early if Performance resumes.
    Bounded error: zero, because the transition is observed at the
    instant it happens.

Both state machines preserve the offline counterparts' rationale:
Pre_Concert is the show's lead-in, not an acoustic class; post-applause
is the structural position right after a long Performance ends, which
is unambiguous even when the AudioSet Applause group fires weakly.

Reference: `scripts/bootstrap_labels.py:apply_pre_concert_heuristic` and
`apply_post_performance_applause_heuristic` for the offline versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Default class order from live/audio_pipeline.py:CLASSES.
DEFAULT_CLASSES = (
    "Pre_Concert", "Performance", "MC_Talk", "Applause",
    "Intermission", "Ambient",
)


def _idx(class_names, name):
    """Return the index of `name` in `class_names`, or raise."""
    try:
        return class_names.index(name)
    except ValueError as exc:
        raise ValueError(
            f"class {name!r} not present in class_names={class_names!r}"
        ) from exc


@dataclass
class CausalPreConcert:
    """Online Pre_Concert state machine.

    Usage:
        st = CausalPreConcert(class_names=CLASSES)
        for sec_label in encoder_stream:
            emitted = st.update(sec_label)
            ...   # use `emitted` as the per-second segment

    The state machine has two states:
      `LEAD_IN` (initial) — emits `Pre_Concert` regardless of input.
      `STARTED`           — emits the encoder's label unchanged; never
                            re-enters `LEAD_IN`.

    The transition `LEAD_IN -> STARTED` fires the second
    `min_active_run` consecutive `active_classes` labels are observed.
    The default `active_classes = ("Performance",)` deliberately
    excludes `MC_Talk`: pre-show PA announcements look like MC_Talk to
    the encoder, but are temporally Pre_Concert. Music-onset is the
    unambiguous "show started" signal.
    """
    class_names: tuple = DEFAULT_CLASSES
    active_classes: tuple = ("Performance",)
    min_active_run: int = 10

    _pre_id: int = field(init=False)
    _active_ids: frozenset = field(init=False)
    _run_len: int = field(init=False, default=0)
    _started: bool = field(init=False, default=False)
    _emitted_count: int = field(init=False, default=0)

    def __post_init__(self):
        self._pre_id = _idx(list(self.class_names), "Pre_Concert")
        self._active_ids = frozenset(
            _idx(list(self.class_names), c) for c in self.active_classes
        )

    @property
    def started(self) -> bool:
        return self._started

    def update(self, encoder_label: int) -> int:
        """Consume one second's encoder label, return the emitted label.

        Once `started` is True, future calls pass `encoder_label`
        through unchanged. While `started` is False, all output is
        `Pre_Concert` regardless of the input.
        """
        self._emitted_count += 1
        if self._started:
            return encoder_label
        if encoder_label in self._active_ids:
            self._run_len += 1
            if self._run_len >= self.min_active_run:
                self._started = True
                return encoder_label
        else:
            self._run_len = 0
        return self._pre_id

    def reset(self) -> None:
        """Reset to LEAD_IN. Use at the start of a new show / session."""
        self._run_len = 0
        self._started = False
        self._emitted_count = 0


@dataclass
class CausalPostApplause:
    """Online "applause follows a long Performance" structural override.

    Usage:
        st = CausalPostApplause(class_names=CLASSES)
        for sec_label, applause_p in stream:
            emitted = st.update(sec_label, applause_posterior=applause_p)
            ...

    The state machine tracks the current Performance run length. When
    it observes a `Performance -> non-Performance` transition at the
    current second `t`, and the run that just ended was at least
    `min_perf_run` seconds long, it opens an override window of up to
    `applause_window_sec` seconds *starting at t*. Within the window,
    any non-Performance encoder prediction is re-labeled as `Applause`,
    optionally subject to a posterior floor (`min_post_evidence`) on
    the encoder's own Applause score to rule out true silence.

    The window closes immediately if Performance resumes (a new song
    starts) before it expires. Emissions after window closure are the
    encoder's labels unchanged.

    Bounded error: zero. Unlike the Pre_Concert state machine, this
    one observes the run-end transition at the exact instant it
    happens, so it never has to revise an earlier emission.
    """
    class_names: tuple = DEFAULT_CLASSES
    min_perf_run: int = 40     # run=40 win=12 is the macroF1 winner in the
                                # LOSO post-applause window sweep; values
                                # below 30 sit inside a flat zone.
    applause_window_sec: int = 12
    min_post_evidence: float = 0.10     # posterior floor; 0.10 is the LOSO
                                        # macroF1 optimum. Set negative to
                                        # disable the floor.

    _perf_id: int = field(init=False)
    _app_id: int = field(init=False)
    _perf_run_len: int = field(init=False, default=0)
    _window_remaining: int = field(init=False, default=0)

    def __post_init__(self):
        self._perf_id = _idx(list(self.class_names), "Performance")
        self._app_id = _idx(list(self.class_names), "Applause")

    @property
    def window_active(self) -> bool:
        return self._window_remaining > 0

    def update(self, encoder_label: int,
               applause_posterior: Optional[float] = None) -> int:
        """Consume one second's encoder label + optional Applause
        posterior; return the emitted label.

        `applause_posterior` is the encoder's per-second probability
        for the Applause class. When provided, the override only fires
        on seconds whose posterior is at least `min_post_evidence` (a
        very low floor — enough to rule out absolute silence and pure
        speech). Pass `None` to disable the evidence check.
        """
        is_perf = (encoder_label == self._perf_id)

        # Detect a Performance -> non-Performance transition by
        # looking at this second's label vs the just-ended run state.
        # Order of operations matters: we must open the window BEFORE
        # we decide what to emit for this same second.
        if is_perf:
            self._perf_run_len += 1
            # Performance resumed mid-window -> close the window early.
            self._window_remaining = 0
        else:
            ended_run_len = self._perf_run_len
            self._perf_run_len = 0
            if ended_run_len >= self.min_perf_run:
                self._window_remaining = self.applause_window_sec

        # Emit. Inside an active window, override non-Performance
        # encoder labels to Applause subject to the posterior floor.
        if self._window_remaining > 0 and not is_perf:
            self._window_remaining -= 1
            if applause_posterior is None \
                    or applause_posterior >= self.min_post_evidence:
                return self._app_id
        return encoder_label

    def reset(self) -> None:
        """Reset to clean state. Use at the start of a new show."""
        self._perf_run_len = 0
        self._window_remaining = 0


@dataclass
class CausalIntermission:
    """Online structural Intermission detector.

    Addresses a concrete weakness measured under LOSO: pooled
    Intermission F1 = 0.0 across every variant, because AudioSet
    has no acoustic concept for "audience milling between sets" and
    Kinetics has no action class for it either. The structural cue we
    use instead is **temporal + label-based**: an Intermission is a
    sustained run of cascade-labeled `Ambient` seconds mid-show, after
    the concert has started and before Performance resumes.

    Why label-based, not confidence-based: in the bundled pipeline the
    audio posterior is HMM-smoothed and almost always concentrates on
    a single class (mean `audio_post6.max()` ≈ 0.94 across the eval
    set). A confidence-based "audio uncertain" gate therefore fires on
    <1 % of seconds, far too rarely to catch the GT Intermission
    stretches. The `audio_segment` label itself, which IS what the audio
    expert reports, is the more useful signal: a sustained `Ambient`
    run between sets is exactly what we want to relabel.

    Causal definition:
      * The cascade has reported at least one sustained Performance
        run earlier in the show — the concert has started (`armed`).
      * The current run of contiguous `Ambient` (and optionally
        `MC_Talk`) labels has reached `min_quiet_run` in length.
      * Performance has not yet resumed.

    On the second the run length crosses the threshold, the state
    machine starts emitting `Intermission` for that second and every
    subsequent `Ambient` / `MC_Talk` second, until `Performance`
    resumes (which closes the intermission and re-arms the counter).

    Bounded error: the first `min_quiet_run - 1` seconds of the actual
    intermission are emitted as whatever the cascade decided (usually
    `Ambient`); only once the threshold is crossed do we relabel to
    `Intermission`. The 60-s default is a compromise — most concerts
    treat 60 s of sustained Ambient between sets as a real break.
    """
    class_names: tuple = DEFAULT_CLASSES
    min_quiet_run: int = 120           # macroF1-optimal threshold from the
                                       # sweep over [30, 45, 60, 90, 120,
                                       # 180]. Lower thresholds catch more of
                                       # the actual intermission but over-fire
                                       # on shows that have long mid-set MC
                                       # stretches.
    include_mc_talk: bool = True       # treat MC_Talk as part of the
                                       # quiet stretch (a long PA
                                       # announcement during the break)
    min_perf_run_arm: int = 10         # how many Performance seconds
                                       # before the detector arms
    enable_cold_start_arm: bool = False  # Off by default. A cold-start ARM
                                       # would handle streams picked up
                                       # MID-intermission, but there are three
                                       # problems with that framing:
                                       #
                                       # (1) In LIVE deployment, the
                                       # heuristic cascade runs
                                       # CausalPreConcert FIRST. While
                                       # CausalPreConcert._started is False
                                       # (no 10 P seen yet), it forces
                                       # Pre_Concert on every emission, so
                                       # any Intermission output from
                                       # CausalIntermission is silently
                                       # overridden. After _started flips
                                       # True, the regular ARM path has
                                       # already fired and cold-start is
                                       # moot. Net effect on the live path:
                                       # ZERO.
                                       #
                                       # (2) In OFFLINE deployment on a
                                       # full recording, the same dominance
                                       # holds: apply_pre_concert_heuristic
                                       # finds show_start and relabels the
                                       # prefix to Pre_Concert. Cold-start
                                       # ARM only "fires" on shows that
                                       # NEVER show a 10-second Performance
                                       # run at all — vanishingly rare.
                                       #
                                       # (3) The apparent pooled-macroF1 gain
                                       # from cold-start ARM came almost
                                       # entirely from one test slice
                                       # (gt_jazz_spring_2025_labels.npz,
                                       # 1407 seconds, 72% Intermission GT,
                                       # ZERO Performance), which carries
                                       # ~99 % of the pool's Intermission
                                       # GT mass. It games that single slice
                                       # rather than improving deployment
                                       # quality.
                                       #
                                       # The mechanism remains available
                                       # as an OFFLINE-batch opt-in via
                                       # this flag for users processing
                                       # known-curated partial recordings.
                                       # Default is OFF.
    cold_start_arm_run: int = 240      # Quiet-run threshold consulted only
                                       # when enable_cold_start_arm=True.

    _perf_id: int = field(init=False)
    _app_id: int = field(init=False)
    _ambient_id: int = field(init=False)
    _mc_id: int = field(init=False)
    _intermission_id: int = field(init=False)
    _armed: bool = field(init=False, default=False)
    _perf_run_len: int = field(init=False, default=0)
    _quiet_run_len: int = field(init=False, default=0)

    def __post_init__(self):
        names = list(self.class_names)
        self._perf_id = _idx(names, "Performance")
        self._app_id = _idx(names, "Applause")
        self._ambient_id = _idx(names, "Ambient")
        self._mc_id = _idx(names, "MC_Talk")
        self._intermission_id = _idx(names, "Intermission")

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def in_intermission(self) -> bool:
        return self._armed and self._quiet_run_len >= self.min_quiet_run

    def _is_quiet_class(self, label: int) -> bool:
        if label == self._ambient_id:
            return True
        if self.include_mc_talk and label == self._mc_id:
            return True
        return False

    def update(self, encoder_label: int,
               audio_conf: Optional[float] = None) -> int:
        """Consume one second's encoder label, return the emitted label.

        The optional `audio_conf` argument is kept for API symmetry with
        the other heuristics but is not used by this state machine
        (label-based gating proved far more reliable on this data; see
        the docstring).
        """
        is_perf = (encoder_label == self._perf_id)
        is_app = (encoder_label == self._app_id)

        if is_perf:
            self._perf_run_len += 1
            if self._perf_run_len >= self.min_perf_run_arm:
                self._armed = True
            self._quiet_run_len = 0
            return encoder_label

        self._perf_run_len = 0

        # Applause is a known transitional class — it does not reset
        # the quiet stretch but does not extend it either.
        if is_app:
            return encoder_label

        if self._is_quiet_class(encoder_label):
            self._quiet_run_len += 1
            if self._armed and self._quiet_run_len >= self.min_quiet_run:
                return self._intermission_id
            # OPT-IN cold-start ARM (default disabled — see the
            # docstring on `enable_cold_start_arm` for the rationale).
            # Only useful for OFFLINE batch processing of curated
            # partial recordings; has no effect in live deployment.
            if (self.enable_cold_start_arm
                    and not self._armed
                    and self._quiet_run_len >= self.cold_start_arm_run):
                self._armed = True
                self._quiet_run_len = max(self._quiet_run_len,
                                          self.min_quiet_run)
                return self._intermission_id
        else:
            # Pre_Concert, Intermission already, or any other non-quiet
            # class breaks the run.
            self._quiet_run_len = 0
        return encoder_label

    def reset(self) -> None:
        self._armed = False
        self._perf_run_len = 0
        self._quiet_run_len = 0


__all__ = ["CausalPreConcert", "CausalPostApplause", "CausalIntermission",
           "DEFAULT_CLASSES"]
