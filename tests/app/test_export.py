import pytest

from clapsync.app import export_tracks, compute_sync_offsets, ExportSettings
from clapsync.app.export import sync_and_trim, ExportResult
from clapsync.app.media import probe
from clapsync.core import common_time_range


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
