from clapsync.core.timerange import TimeRange
from clapsync.app.export import clip_window, frame_source_times, ExportSettings
from pathlib import Path


def test_clip_window_interior():
    # offset 2.0, duration 10.0, trim [3, 8] on shared timeline
    ls, le, ps, pe = clip_window(2.0, 10.0, TimeRange(3.0, 8.0))
    assert (ls, le, ps, pe) == (1.0, 6.0, 0.0, 0.0)


def test_clip_window_leading_gap_pads_start():
    # track starts at offset 5 but trim starts at 3 -> 2 s of leading pad
    ls, le, ps, pe = clip_window(5.0, 10.0, TimeRange(3.0, 8.0))
    assert ls == 0.0
    assert ps == 2.0


def test_clip_window_trailing_gap_pads_end():
    # track [0,4], trim [1,6] -> 2 s trailing pad, local_end clamps to 4
    ls, le, ps, pe = clip_window(0.0, 4.0, TimeRange(1.0, 6.0))
    assert le == 4.0
    assert pe == 2.0


def test_frame_source_times_grid():
    # trim 0.4 s @ 25 fps => 10 frames; offset 0 => source times 0.00..0.36
    times = frame_source_times(0.0, TimeRange(0.0, 0.4), 25.0)
    assert len(times) == 10
    assert abs(times[0] - 0.0) < 1e-9
    assert abs(times[1] - 0.04) < 1e-9


def test_frame_source_times_shifted_by_offset():
    # offset 0.5 subtracts from every source time (track starts later)
    times = frame_source_times(0.5, TimeRange(0.0, 0.08), 25.0)
    assert abs(times[0] + 0.5) < 1e-9  # -0.5 -> before track start (caller pads)


def test_export_settings_defaults():
    s = ExportSettings(trim=TimeRange(0.0, 1.0), output_dir=Path("."))
    assert s.crf == 18
    assert s.video_codec is None
