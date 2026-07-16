import pytest

from clapsync.app import (
    ExportResult,
    ExportSettings,
    compute_sync_offsets,
    export_tracks,
    sync_and_trim,
)
from clapsync.app.export import export_media
from clapsync.app.media import probe
from clapsync.core import common_time_range
from clapsync.core.timerange import TimeRange


@pytest.mark.slow
def test_export_tracks_takes_paths(av_video, tmp_path):
    a, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="a.mp4")
    b, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="b.mp4")
    paths = [a, b]
    offsets = compute_sync_offsets(paths)
    durations = [probe(p).duration for p in paths]
    settings = ExportSettings(
        trim=common_time_range(durations, offsets), output_dir=tmp_path,
    )
    results = export_tracks(paths, offsets, settings)
    assert all(r.ok for r in results), [r.error for r in results]


@pytest.mark.slow
def test_sync_and_trim_two_videos_roundtrip(av_video, tmp_path):
    # Audio+video inputs: sync requires audio (video-only would fail hard).
    a, *_ = av_video(seconds=1.0, fps=30.0, name="a.mp4")
    b, *_ = av_video(seconds=1.0, fps=30.0, name="b.mp4")
    out = tmp_path / "out"
    out.mkdir()

    results = sync_and_trim([a, b], out, trim="common")
    assert len(results) == 2
    assert all(isinstance(r, ExportResult) for r in results)
    assert all(r.ok for r in results), [r.error for r in results]
    for r in results:
        assert r.path.exists() and r.path.stat().st_size > 0


@pytest.mark.slow
def test_sync_and_trim_requires_audio(rgb_video, tmp_path):
    # Video-only inputs must fail hard, not silently produce zero offsets.
    a, *_ = rgb_video(seconds=1.0, fps=30.0, name="a.mp4")
    b, *_ = rgb_video(seconds=1.0, fps=30.0, name="b.mp4")
    with pytest.raises(ValueError, match="without an audio stream"):
        sync_and_trim([a, b], tmp_path / "out", trim="common")


@pytest.mark.slow
def test_export_honors_resolution_fps_and_trim(av_video, tmp_path):
    """Streaming export resizes, retimes and trims to the requested settings."""
    src, *_ = av_video(seconds=4.0, fps=30.0, w=256, h=144, name="s.mp4")
    info = probe(src)
    out = tmp_path / "out"
    out.mkdir()

    results = export_media(
        [info], [0.0],
        ExportSettings(
            trim=TimeRange(1.0, 3.0), output_dir=out,
            target_width=128, target_height=72, output_fps=25.0,
        ),
    )
    assert results[0].ok, results[0].error
    got = probe(results[0].path)
    assert (got.width, got.height) == (128, 72)
    assert abs(got.fps - 25.0) < 0.1
    assert abs(got.duration - 2.0) < 0.15  # trim was 2 s


@pytest.mark.slow
def test_export_black_pads_offset_gap(av_video, tmp_path):
    """A track offset past the trim start is black-padded to the full range."""
    src, *_ = av_video(seconds=3.0, fps=30.0, w=128, h=72, name="s.mp4")
    info = probe(src)
    out = tmp_path / "out"
    out.mkdir()

    # offset 1 s, trim [0, 3] -> 1 s black head + 2 s source = 3 s output.
    results = export_media(
        [info], [1.0],
        ExportSettings(trim=TimeRange(0.0, 3.0), output_dir=out),
    )
    assert results[0].ok, results[0].error
    assert abs(probe(results[0].path).duration - 3.0) < 0.15


@pytest.mark.slow
def test_export_native_path_matches_source(av_video, tmp_path):
    """Native res+fps export (fast path) keeps the source dimensions and rate."""
    src, *_ = av_video(seconds=3.0, fps=30.0, w=192, h=108, name="s.mp4")
    info = probe(src)
    out = tmp_path / "out"
    out.mkdir()

    results = export_media(
        [info], [0.0],
        ExportSettings(trim=TimeRange(0.5, 2.5), output_dir=out),
    )
    assert results[0].ok, results[0].error
    got = probe(results[0].path)
    assert (got.width, got.height) == (192, 108)
    assert abs(got.fps - 30.0) < 0.1
    assert abs(got.duration - 2.0) < 0.15
