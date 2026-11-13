"""dtw (attacker/dtw.py): the torch-free realignment core — banded DTW and
Levenshtein-style edit alignment against hand-computed known answers and the
module's frozen fixtures (the E-R8 alignment-search machinery check)."""

import math
import runpy
import warnings

import pytest

from attacker import dtw
from attacker.dtw import (FIXTURE_DTW, FIXTURE_EDIT_PAIRS, _row_dist,
                          dtw_distance, edit_alignment)

BASE = FIXTURE_DTW["base"]
SHIFTED = FIXTURE_DTW["shifted"]


# --- _row_dist ----------------------------------------------------------------------
def test_row_dist_hand_computed():
    assert _row_dist([0.0, 0.0], [3.0, 4.0]) == 5.0          # 3-4-5 triangle
    assert _row_dist([1.0], [1.0]) == 0.0
    assert _row_dist([1.0, 2.0], [3.0, 4.0]) == math.sqrt(8.0)
    assert _row_dist([0.0, 0.0], [3.0, 4.0]) == _row_dist([3.0, 4.0],
                                                          [0.0, 0.0])


# --- dtw_distance ---------------------------------------------------------------------
def test_dtw_identity_is_zero_cost_diagonal():
    cost, path = dtw_distance(BASE, BASE)
    assert cost == 0.0
    assert path == [(i, i) for i in range(len(BASE))]


def test_dtw_hand_computed_unequal_lengths():
    # a=[0,2], b=[0,1,2]: cost matrix forces the path through (0,0),(0,1),(1,2)
    # with total cost 1 over 3 steps -> 1/3 normalized
    cost, path = dtw_distance([[0.0], [2.0]], [[0.0], [1.0], [2.0]])
    assert cost == pytest.approx(1.0 / 3.0)
    assert path == [(0, 0), (0, 1), (1, 2)]


def test_dtw_single_pair():
    cost, path = dtw_distance([[1.0, 1.0]], [[4.0, 5.0]])
    assert cost == 5.0       # sqrt((1-4)^2 + (1-5)^2), one step
    assert path == [(0, 0)]


def test_dtw_empty_input_is_inf_empty_path():
    assert dtw_distance([], BASE) == (float("inf"), [])
    assert dtw_distance(BASE, []) == (float("inf"), [])
    assert dtw_distance([], []) == (float("inf"), [])


def test_dtw_fixture_pad_inserted_sequence():
    cost, path = dtw_distance(BASE, SHIFTED)
    # frozen known answer: the [9,9] pad row is absorbed by row 1
    assert cost == pytest.approx(2.006932429798716)
    assert path == [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    # the true rows map to their shifted positions
    for p in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        assert p in path
    # cheaper than the naive diagonal pairing (the defense's hoped-for cost)
    diag = sum(_row_dist(BASE[min(i, len(BASE) - 1)], SHIFTED[i])
               for i in range(len(SHIFTED))) / len(SHIFTED)
    assert cost < diag


def test_dtw_path_spans_and_is_monotone():
    _, path = dtw_distance(BASE, SHIFTED)
    assert path[0] == (0, 0)
    assert path[-1] == (len(BASE) - 1, len(SHIFTED) - 1)
    assert all(b0 <= b1 and a0 <= a1
               for (a0, b0), (a1, b1) in zip(path, path[1:]))
    # steps are unit moves (up / left / diagonal), never jumps
    assert all((a1 - a0, b1 - b0) in ((1, 0), (0, 1), (1, 1))
               for (a0, b0), (a1, b1) in zip(path, path[1:]))


def test_dtw_cost_is_symmetric():
    assert dtw_distance(BASE, SHIFTED)[0] == dtw_distance(SHIFTED, BASE)[0]


def test_dtw_window_zero_forces_the_diagonal():
    cost, path = dtw_distance(BASE, BASE, window=0)
    assert cost == 0.0
    assert path == [(i, i) for i in range(len(BASE))]


def test_dtw_window_is_clamped_to_length_difference():
    # window=0 with |n-m|=1 is widened to 1, so the endpoint stays reachable
    cost, path = dtw_distance([[0.0], [1.0]], [[0.0], [0.5], [1.0]], window=0)
    assert cost == pytest.approx(1.0 / 6.0)
    assert path == [(0, 0), (0, 1), (1, 2)]


def test_dtw_banded_matches_unbounded_when_optimum_fits():
    # the fixture's optimal path never leaves the |i-j| <= 1 band
    assert dtw_distance(BASE, SHIFTED, window=1) == dtw_distance(BASE, SHIFTED)
    # a band that excludes the optimum can only cost more (never less)
    cheap = dtw_distance(BASE, SHIFTED)[0]
    for w in (1, 2, 10):
        c, _ = dtw_distance(BASE, SHIFTED, window=w)
        assert c >= cheap or c == float("inf")


# --- edit_alignment ---------------------------------------------------------------------
def test_edit_alignment_fixture_pairs():
    # inserted pad row skipped, all five true rows matched (frozen fixture)
    assert edit_alignment(BASE, SHIFTED, gap_cost=2.0) == FIXTURE_EDIT_PAIRS


def test_edit_alignment_identity_is_diagonal():
    assert edit_alignment(BASE, BASE, gap_cost=2.0) == [(i, i)
                                                        for i in range(len(BASE))]


def test_edit_alignment_gap_cost_decides_match_vs_skip():
    a, b = [[0.0], [1.0]], [[1.0], [0.0]]
    # cheap gaps: pay two gaps (0.8) instead of two mismatches (2.0) — only
    # the identical rows pair up
    assert edit_alignment(a, b, gap_cost=0.4) == [(0, 1)]
    # expensive gaps: two mismatches (2.0) beat four gap penalties -> diagonal
    assert edit_alignment(a, b, gap_cost=10.0) == [(0, 0), (1, 1)]


def test_edit_alignment_pure_insertion():
    # b is a with one row inserted in the middle: that row is skipped
    assert edit_alignment([[0.0], [2.0]], [[0.0], [1.0], [2.0]],
                          gap_cost=0.5) == [(0, 0), (1, 2)]


def test_edit_alignment_skips_pads_at_both_ends():
    a = [[float(i)] for i in range(6)]
    b = [[-1.0]] + a + [[99.0]]
    assert edit_alignment(a, b, gap_cost=3.0) == [(i, i + 1) for i in range(6)]


def test_edit_alignment_longer_a_pairs_only_the_shared_row():
    a = [[float(i)] for i in range(6)]
    assert edit_alignment(a, [a[2]], gap_cost=3.0) == [(2, 0)]


def test_edit_alignment_empty_inputs():
    assert edit_alignment([], BASE, gap_cost=1.0) == []
    assert edit_alignment(BASE, [], gap_cost=1.0) == []
    # the gap_cost=None default also survives an empty side (fallback 1.0)
    assert edit_alignment([], BASE) == []
    assert edit_alignment([], []) == []


def test_edit_alignment_default_gap_is_scale_free():
    # gap_cost=None derives from the median pairwise row distance, so the
    # identity stays the diagonal even when the rows are scaled up 10x
    scaled = [[10.0 * x, 10.0 * y] for x, y in BASE]
    assert edit_alignment(scaled, scaled) == [(i, i)
                                              for i in range(len(BASE))]
    # and the pad-inserted fixture still aligns the five true rows
    assert edit_alignment(BASE, SHIFTED) == FIXTURE_EDIT_PAIRS


def test_edit_alignment_pairs_are_strictly_monotone():
    pairs = edit_alignment(BASE, SHIFTED, gap_cost=2.0)
    assert all(a0 < a1 and b0 < b1
               for (a0, b0), (a1, b1) in zip(pairs, pairs[1:]))


# --- self-test + CLI entry --------------------------------------------------------------
def test_self_test_passes():
    assert dtw.self_test() == 0


def test_module_runs_standalone_torchless():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # already in sys.modules
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("attacker.dtw", run_name="__main__")
    assert exc.value.code == 0
