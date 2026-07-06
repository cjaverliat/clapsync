"""Trim/pad export of synced tracks (window math is pure; I/O is separate)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clapsync.core.timerange import TimeRange


@dataclass(frozen=True)
class ExportSettings:
    """Output settings for a sync/trim export."""

    trim: TimeRange
    output_dir: Path
    target_width: int | None = None
    target_height: int | None = None
    output_fps: float | None = None
    video_codec: str | None = None
    crf: int = 18
    audio_format: str | None = None


@dataclass(frozen=True)
class ExportResult:
    """Outcome for one exported track."""

    path: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def clip_window(
    offset: float, duration: float, trim: TimeRange
) -> tuple[float, float, float, float]:
    """Map a shared-timeline trim onto one track's local source range.

    Args:
        offset: Track offset on the shared timeline (seconds).
        duration: Track source duration (seconds).
        trim: Desired output range on the shared timeline.

    Returns:
        (local_start, local_end, pad_start, pad_end) in seconds. local_* index
        into the source; pad_* are black/silence gaps outside the source.
    """
    local_start = max(0.0, trim.start - offset)
    local_end = min(duration, trim.end - offset)
    pad_start = max(0.0, offset - trim.start)
    pad_end = max(0.0, trim.end - (offset + duration))
    return local_start, local_end, pad_start, pad_end


def frame_source_times(
    offset: float, trim: TimeRange, out_fps: float
) -> list[float]:
    """Source timestamps for each output frame on the shared trim grid.

    Output frame k sits at shared time trim.start + k / out_fps; the matching
    source time subtracts the track offset. Values outside [0, duration) mark
    frames the caller must black-pad. This keeps subframe offsets exact — the
    output grid is fixed, the source is sampled at the precise shifted time.

    Args:
        offset: Track offset on the shared timeline (seconds).
        trim: Output range on the shared timeline.
        out_fps: Output frame rate.

    Returns:
        One source timestamp per output frame.
    """
    n_frames = round(trim.duration * out_fps)
    return [trim.start + k / out_fps - offset for k in range(n_frames)]
