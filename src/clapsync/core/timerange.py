"""Shared-timeline range math over aligned tracks (pure)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeRange:
    """A closed interval on the shared timeline, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def full_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Union: earliest track start to latest track end.

    Args:
        durations: Per-track source durations in seconds.
        offsets: Per-track offset on the shared timeline in seconds.

    Returns:
        The smallest range covering every track.
    """
    starts = offsets
    ends = [o + d for o, d in zip(offsets, durations)]
    return TimeRange(min(starts), max(ends))


def common_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Intersection: the region where every track has footage.

    When tracks do not all overlap the result is zero-length (start == end).

    Args:
        durations: Per-track source durations in seconds.
        offsets: Per-track offset on the shared timeline in seconds.

    Returns:
        The overlap range, clamped so end >= start.
    """
    start = max(offsets)
    end = min(o + d for o, d in zip(offsets, durations))
    if end < start:
        end = start
    return TimeRange(start, end)
