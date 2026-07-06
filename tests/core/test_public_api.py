def test_public_surface_importable():
    from clapsync.core import (
        MediaInfo, probe,
        TimeRange, common_time_range, full_time_range,
        compute_sync_offsets,
        ExportSettings, ExportResult, export_tracks, sync_and_trim,
        find_offset,
    )
    assert callable(sync_and_trim)
    assert callable(find_offset)
