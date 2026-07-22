"""clapsync pure core: MFCC audio sync and time-range math (no I/O)."""
from clapsync.core.offsets import (
    PairAlignment,
    Refine,
    align_waveforms,
    find_offset,
    find_offset_peaks,
)
from clapsync.core.solver import LOW_CONFIDENCE, Alignment
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "align_waveforms",
    "find_offset",
    "find_offset_peaks",
    "Alignment",
    "PairAlignment",
    "LOW_CONFIDENCE",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "Refine",
]
