"""clapsync headless API: sync, time ranges, and muxed trim/export."""
from clapsync.core.export import (
    ExportResult,
    ExportSettings,
    export_tracks,
    sync_and_trim,
)
from clapsync.core.media import MediaInfo, probe
from clapsync.core.offsets import Method, Refine, find_offset
from clapsync.core.sync import compute_sync_offsets
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "MediaInfo",
    "probe",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "compute_sync_offsets",
    "ExportSettings",
    "ExportResult",
    "export_tracks",
    "sync_and_trim",
    "find_offset",
    "Method",
    "Refine",
]
