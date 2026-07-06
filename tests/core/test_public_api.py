def test_public_surface_importable():
    from clapsync.app import (
        ExportResult,
        ExportSettings,
        compute_sync_offsets,
        export_tracks,
        sync_and_trim,
    )
    from clapsync.app.media import MediaInfo, probe
    from clapsync.core import (
        TimeRange,
        common_time_range,
        find_offset,
        full_time_range,
    )
    assert callable(sync_and_trim)
    assert callable(find_offset)
