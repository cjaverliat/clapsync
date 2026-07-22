import math

from clapsync.core.solver import (
    EDGE_MIN_SCORE,
    LOW_CONFIDENCE,
    Alignment,
    solve_offsets,
)


def _pair(offset: float, score: float = 50.0) -> list[tuple[float, float]]:
    return [(offset, score)]


def test_two_tracks_clean_pair():
    result = solve_offsets(2, {(0, 1): _pair(0.5)})
    assert isinstance(result, Alignment)
    assert result.offsets == [0.0, 0.5]
    assert math.isinf(result.confidence[0])
    assert result.confidence[1] == 50.0
    assert result.warnings == []


def test_two_tracks_gated_edge_isolates_track():
    result = solve_offsets(2, {(0, 1): _pair(0.5, score=EDGE_MIN_SCORE - 1)})
    assert result.offsets == [0.0, 0.0]
    assert result.confidence[1] == 0.0
    assert any("unrelated" in w for w in result.warnings)


def test_low_score_edge_warns_but_uses_offset():
    result = solve_offsets(2, {(0, 1): _pair(0.5, score=7.0)})
    assert result.offsets == [0.0, 0.5]
    assert result.confidence[1] == 7.0
    assert any("low sync confidence" in w for w in result.warnings)


def test_three_tracks_chain_through_strongest_edges():
    # 0-1 and 1-2 strong; 0-2 gated out. Offsets must chain: o2 = x01 + x12.
    pairs = {
        (0, 1): _pair(1.0),
        (1, 2): _pair(0.5),
        (0, 2): _pair(9.9, score=2.0),  # gated
    }
    result = solve_offsets(3, pairs)
    assert result.offsets[1] == 1.0
    assert abs(result.offsets[2] - 1.5) < 1e-9


def test_nonzero_reference_index_rebases():
    result = solve_offsets(2, {(0, 1): _pair(0.5)}, reference_index=1)
    assert result.offsets == [-0.5, 0.0]
    assert math.isinf(result.confidence[1])


def test_single_track():
    result = solve_offsets(1, {})
    assert result.offsets == [0.0]
    assert math.isinf(result.confidence[0])


def _clean_pairs(offsets: list[float], score: float = 50.0):
    """All-pairs peak lists consistent with the given global offsets."""
    n = len(offsets)
    return {
        (i, j): [(offsets[j] - offsets[i], score)]
        for i in range(n)
        for j in range(i + 1, n)
    }


def test_outlier_edge_routed_around_at_n4():
    true = [0.0, 1.0, 2.0, 3.0]
    pairs = _clean_pairs(true)
    # Edge 1-2 lies by 180 s with a *high* score (repeated-content trap).
    pairs[(1, 2)] = [(true[2] - true[1] + 180.0, 80.0)]
    result = solve_offsets(4, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6
    # The lie never contaminates offsets; consistency weight kept it off the tree.


def test_n3_single_bad_edge_detected_not_silent():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    pairs[(1, 2)] = [(true[2] - true[1] + 180.0, 80.0)]  # no alternate peak
    result = solve_offsets(3, pairs)
    # One triangle, all edges share the blame: every non-reference track
    # must carry an inconsistency warning.
    assert sum("inconsistent" in w for w in result.warnings) >= 1


def test_clean_n4_no_warnings_high_confidence():
    result = solve_offsets(4, _clean_pairs([0.0, -0.5, 1.25, 3.0]))
    assert result.warnings == []
    assert all(c >= LOW_CONFIDENCE for i, c in enumerate(result.confidence) if i != 0)


def test_repair_adopts_alternate_peak_at_n3():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    # Edge 1-2's best peak is the wrong repeat (+180 s) but the correct
    # offset survives as its second peak.
    pairs[(1, 2)] = [
        (true[2] - true[1] + 180.0, 80.0),
        (true[2] - true[1], 60.0),
    ]
    result = solve_offsets(3, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6
    assert not any("inconsistent" in w for w in result.warnings)


def test_repair_adopts_alternate_peak_at_n4():
    true = [0.0, 1.0, 2.0, 3.0]
    pairs = _clean_pairs(true)
    pairs[(0, 3)] = [
        (true[3] - true[0] - 42.0, 70.0),
        (true[3] - true[0], 55.0),
    ]
    result = solve_offsets(4, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6


def test_repair_skips_when_no_alternate_closes_triangle():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    pairs[(1, 2)] = [
        (true[2] - true[1] + 180.0, 80.0),
        (true[2] - true[1] + 90.0, 60.0),  # also wrong
    ]
    result = solve_offsets(3, pairs)
    assert any("inconsistent" in w for w in result.warnings)


def test_zero_tracks_returns_empty():
    result = solve_offsets(0, {})
    assert result.offsets == []
    assert result.confidence == []
    assert result.warnings == []
