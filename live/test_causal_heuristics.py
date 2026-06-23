"""Unit tests for the causal heuristics in live/causal_heuristics.py.

Each test starts with a string of one-letter labels for readability
(P=Performance, M=MC_Talk, A=Applause, B=Ambient, I=Intermission,
Z=Pre_Concert), feeds them through the state machine one second at a
time, and asserts the emitted-label sequence matches the expected
string. The same scenarios that the offline counterparts in
scripts/bootstrap_labels.py operate on are run here, so any divergence
between the offline (full-array) and online (streaming) outputs is
caught immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from live.causal_heuristics import (
    CausalIntermission,
    CausalPostApplause,
    CausalPreConcert,
    DEFAULT_CLASSES,
)


# Letter <-> class id mapping for compact test scenarios.
LETTER = {
    "Z": "Pre_Concert", "P": "Performance", "M": "MC_Talk",
    "A": "Applause", "I": "Intermission", "B": "Ambient",
}
NAME_TO_LETTER = {v: k for k, v in LETTER.items()}


def _seq_to_ids(seq: str) -> list[int]:
    return [DEFAULT_CLASSES.index(LETTER[c]) for c in seq]


def _ids_to_seq(ids: list[int]) -> str:
    return "".join(NAME_TO_LETTER[DEFAULT_CLASSES[i]] for i in ids)


# ---------------------------------------------------------------------------
# CausalPreConcert tests
# ---------------------------------------------------------------------------

def test_pre_concert_no_active_run_stays_in_lead_in():
    """A pre-show with chatter (B) and one isolated P should NOT trigger
    the start-of-concert transition; everything stays Pre_Concert."""
    st = CausalPreConcert(min_active_run=10)
    inp = "BBBPMBBB" + "B" * 5
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    assert out == "Z" * len(inp), f"expected all Pre_Concert, got {out}"
    assert not st.started


def test_pre_concert_short_run_resets_counter():
    """A 9-second Performance run (1 short of min) should not trigger
    the transition; the show must still be marked Pre_Concert until a
    sustained 10-sec run arrives."""
    st = CausalPreConcert(min_active_run=10)
    inp = "P" * 9 + "MBP" + "P" * 10 + "MBB"   # 9P fall short, then 10P trigger
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    # Indices 0..8 = 9 P (run_len reaches 9, never triggers).
    # Index 9 = M, index 10 = B -> run_len resets to 0.
    # Index 11 = P (run_len 1); a new active run begins.
    # Indices 12..20 = 9 more P -> at index 20, run_len reaches 10 and
    # the LEAD_IN -> STARTED transition fires; emit Performance for
    # index 20 onward.
    # So emissions 0..19 = Pre_Concert (20 Z), 20..21 = PP (the 10th
    # consecutive P plus one more from the run), 22 = M, 23..24 = BB.
    expected = "Z" * 20 + "PPMBB"
    assert out == expected, f"expected {expected}, got {out}"
    assert st.started


def test_pre_concert_active_run_locks_started():
    """Once STARTED, future non-Performance seconds stay unchanged
    (no return to Pre_Concert even if there's a long MC_Talk gap)."""
    st = CausalPreConcert(min_active_run=10)
    inp = "P" * 10 + "M" * 30 + "P" * 5
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    # First 10 P: indices 0..8 = Pre_Concert (counter not yet 10), index 9 = the
    # 10th consecutive P triggers; emit P at index 9.
    # Indices 10..39 (30 M): STARTED, emit M.
    # Indices 40..44 (5 P): emit P.
    expected = "Z" * 9 + "P" + "M" * 30 + "P" * 5
    assert out == expected, f"expected {expected!r} (len {len(expected)}), got {out!r} (len {len(out)})"


def test_pre_concert_mc_talk_does_not_trigger_start():
    """MC_Talk run before music should not be enough to leave LEAD_IN.
    Only Performance counts as 'show started'."""
    st = CausalPreConcert(min_active_run=10)
    inp = "M" * 60 + "B" * 5     # 60 sec of host PA + audience hush
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    assert out == "Z" * len(inp), \
        f"MC_Talk should not start the show, but got {out}"
    assert not st.started


def test_pre_concert_reset_restarts_state_machine():
    """reset() between shows should return to LEAD_IN."""
    st = CausalPreConcert(min_active_run=10)
    for _ in range(15):
        st.update(_seq_to_ids("P")[0])
    assert st.started
    st.reset()
    assert not st.started
    out = _ids_to_seq([st.update(i) for i in _seq_to_ids("BBB")])
    assert out == "ZZZ"


# ---------------------------------------------------------------------------
# CausalPostApplause tests
# ---------------------------------------------------------------------------

def test_post_applause_short_perf_no_override():
    """Performance run < min_perf_run: no override window should open."""
    st = CausalPostApplause(min_perf_run=20, applause_window_sec=12,
                            min_post_evidence=-1.0)   # disable floor
    inp = "P" * 10 + "BBBB"
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    assert out == inp, \
        f"short run should not trigger override; expected {inp}, got {out}"


def test_post_applause_long_perf_opens_window():
    """Performance run >= min_perf_run, then non-Performance: next
    `applause_window_sec` non-Performance seconds become Applause."""
    st = CausalPostApplause(min_perf_run=20, applause_window_sec=12,
                            min_post_evidence=-1.0)
    inp = "P" * 25 + "M" * 15
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    # 25 P pass through. Then transition at index 25: window opens with
    # 12 remaining. Index 25 is the first non-Perf; override to A. Then
    # 11 more M -> A. Indices 25..36 = A (12 Applause). Indices 37..39
    # = M (window closed). So expected output:
    expected = "P" * 25 + "A" * 12 + "M" * 3
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_post_applause_window_closes_on_performance_resume():
    """Performance resuming mid-window should close the override
    immediately so a new song doesn't get mis-labeled Applause."""
    st = CausalPostApplause(min_perf_run=20, applause_window_sec=12,
                            min_post_evidence=-1.0)
    inp = "P" * 25 + "M" * 5 + "P" * 10
    out = "".join(_ids_to_seq([st.update(i)]) for i in _seq_to_ids(inp))
    # First 25 P pass. Then 5 M get overridden to A (indices 25..29).
    # Then 10 P resume: window closes at first P; subsequent P pass
    # through unchanged.
    expected = "P" * 25 + "A" * 5 + "P" * 10
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_post_applause_posterior_floor_blocks_silence():
    """A second whose Applause posterior is below `min_post_evidence`
    is left unchanged even inside the override window (defends against
    re-labeling pure silence as Applause)."""
    st = CausalPostApplause(min_perf_run=20, applause_window_sec=12,
                            min_post_evidence=0.10)
    # 25 P followed by 5 sec of pure silence (Applause posterior ~ 0).
    inp_ids = _seq_to_ids("P" * 25 + "B" * 5)
    posteriors = [0.5] * 25 + [0.0] * 5   # silence has 0 applause energy
    out = []
    for lbl, p in zip(inp_ids, posteriors):
        out.append(st.update(lbl, applause_posterior=p))
    out_str = _ids_to_seq(out)
    # No override should fire on the silent seconds; they stay B.
    expected = "P" * 25 + "B" * 5
    assert out_str == expected, \
        f"posterior floor should have blocked override; got {out_str}"


# ---------------------------------------------------------------------------
# CausalIntermission tests (cold-start ARM)
# ---------------------------------------------------------------------------

def test_intermission_armed_by_performance_then_fires_after_quiet():
    """Regular ARM path: 10 Performance seconds arms; 120 quiet seconds
    later, Intermission starts emitting."""
    st = CausalIntermission(min_quiet_run=120, cold_start_arm_run=10_000)
    inp = "P" * 15 + "B" * 130
    out_ids = [st.update(i) for i in _seq_to_ids(inp)]
    out = _ids_to_seq(out_ids)
    # First 15 P emit as P (Performance), then 119 B emit as B before the
    # 120th quiet second triggers Intermission, and the remaining 10 B
    # emit as I.
    expected = "P" * 15 + "B" * 119 + "I" * 11
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_intermission_cold_start_arms_without_prior_performance():
    """cold-start ARM is opt-in. With enable_cold_start_arm=True,
    the detector arms itself after cold_start_arm_run consecutive quiet
    seconds even without a preceding Performance run. With the default
    enable_cold_start_arm=False (covered by the next test) the same
    sequence emits raw Ambient."""
    st = CausalIntermission(min_quiet_run=120, cold_start_arm_run=60,
                            enable_cold_start_arm=True)
    inp = "B" * 70
    out_ids = [st.update(i) for i in _seq_to_ids(inp)]
    out = _ids_to_seq(out_ids)
    expected = "B" * 59 + "I" * 11
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_intermission_cold_start_default_off_emits_raw_quiet():
    """default: enable_cold_start_arm=False means the state machine
    NEVER arms without a preceding Performance run. The cold-start
    scenario emits raw Ambient indefinitely. This is the behavior in
    production paths."""
    st = CausalIntermission(min_quiet_run=120, cold_start_arm_run=60)
    inp = "B" * 200
    out_ids = [st.update(i) for i in _seq_to_ids(inp)]
    out = _ids_to_seq(out_ids)
    assert out == "B" * 200, f"expected all B, got {out!r}"


def test_intermission_cold_start_does_not_fire_below_threshold():
    """A short quiet stretch below cold_start_arm_run should NOT trigger
    cold-start (otherwise pre-show chatter would be relabeled
    Intermission)."""
    st = CausalIntermission(min_quiet_run=120, cold_start_arm_run=240)
    inp = "B" * 100 + "M" * 50   # 150 quiet seconds < 240 threshold
    out_ids = [st.update(i) for i in _seq_to_ids(inp)]
    out = _ids_to_seq(out_ids)
    expected = "B" * 100 + "M" * 50
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_intermission_cold_start_reset_to_performance_clears_state():
    """Performance during a cold-start ARM-ed Intermission closes the
    intermission and re-arms the regular state machine."""
    st = CausalIntermission(min_quiet_run=120, cold_start_arm_run=60,
                            enable_cold_start_arm=True)
    inp = "B" * 80 + "P" * 15 + "B" * 130
    out_ids = [st.update(i) for i in _seq_to_ids(inp)]
    out = _ids_to_seq(out_ids)
    # Cold-start arms at 60 B, emits I from second 60 through 80 (21 I)
    # Then 15 P close intermission
    # Then 119 B emit as B, second 120 (= 134) crosses min_quiet_run and
    # emits I for the rest.
    expected = "B" * 59 + "I" * 21 + "P" * 15 + "B" * 119 + "I" * 11
    assert out == expected, f"expected {expected!r}, got {out!r}"


def test_post_applause_reset_restarts():
    st = CausalPostApplause(min_perf_run=20, applause_window_sec=12,
                            min_post_evidence=-1.0)
    for _ in range(25):
        st.update(_seq_to_ids("P")[0])
    for _ in range(3):
        st.update(_seq_to_ids("B")[0])   # opens window
    assert st.window_active
    st.reset()
    assert not st.window_active


# ---------------------------------------------------------------------------
# Equivalence check: causal output converges to the offline output once
# the offline transition point has been crossed.
# ---------------------------------------------------------------------------

def test_pre_concert_causal_matches_offline_after_transition():
    """After the offline transition point, the causal output should
    match the offline output exactly. They only differ in the LEAD_IN
    region where the causal version emits the first
    `min_active_run - 1` Performance seconds as Pre_Concert."""
    import numpy as np
    from cuesheet.scripts.bootstrap_labels import (
        apply_pre_concert_heuristic,
    )

    inp = "BBBM" + "P" * 30 + "MAAB" + "P" * 10
    seq = _seq_to_ids(inp)
    offline = apply_pre_concert_heuristic(
        np.asarray(seq), list(DEFAULT_CLASSES),
        active_classes=("Performance",), min_active_run=10,
    )
    st = CausalPreConcert(min_active_run=10)
    causal = [st.update(i) for i in seq]

    # The offline version back-labels everything before the first
    # sustained run start (= index 4, the first P of the 30-P block)
    # as Pre_Concert. The causal version emits Pre_Concert until the
    # 10th consecutive P (index 13), so causal[4..12] = Pre_Concert
    # where offline = Performance. Past index 13 they must match.
    assert np.array_equal(np.asarray(causal[13:]), offline[13:]), \
        f"diverged post-transition: causal[13:]={_ids_to_seq(causal[13:])}, " \
        f"offline[13:]={_ids_to_seq(list(offline[13:]))}"


if __name__ == "__main__":
    import inspect
    n_pass = n_fail = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__],
                                              inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  ok    {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e!r}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
