import pytest

from clapsync.core.export import sync_and_trim, ExportResult


@pytest.mark.slow
def test_sync_and_trim_two_videos_roundtrip(rgb_video, tmp_path):
    a, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, name="a.mp4")
    b, *_ = rgb_video(seconds=1.0, fps=30.0, name="b.mp4")
    out = tmp_path / "out"
    out.mkdir()

    results = sync_and_trim([a, b], out, trim="common")
    assert len(results) == 2
    assert all(isinstance(r, ExportResult) for r in results)
    assert all(r.ok for r in results), [r.error for r in results]
    for r in results:
        assert r.path.exists() and r.path.stat().st_size > 0
