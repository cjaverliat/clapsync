from clapsync.core.timerange import (
    TimeRange, common_time_range, full_time_range,
)


def test_duration():
    assert TimeRange(1.0, 3.5).duration == 2.5


def test_full_range_is_union():
    # track A: [0, 10], track B: [2, 14]
    r = full_time_range(durations=[10.0, 12.0], offsets=[0.0, 2.0])
    assert r.start == 0.0 and r.end == 14.0


def test_common_range_is_intersection():
    # overlap is [2, 10]
    r = common_time_range(durations=[10.0, 12.0], offsets=[0.0, 2.0])
    assert r.start == 2.0 and r.end == 10.0


def test_common_range_empty_when_disjoint_is_zero_length():
    # A: [0,2], B: [5,7]  -> no overlap -> zero-length at the gap
    r = common_time_range(durations=[2.0, 2.0], offsets=[0.0, 5.0])
    assert r.duration == 0.0
    assert r.start == r.end
