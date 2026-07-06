import pytest

from clapsync.app.media import probe, MediaInfo


@pytest.mark.slow
def test_probe_audio_only(tone_wav):
    path, sr = tone_wav(seconds=0.5, sample_rate=48000)
    info = probe(path)
    assert isinstance(info, MediaInfo)
    assert info.kind == "audio"
    assert info.has_audio is True
    assert info.sample_rate == 48000
    assert info.fps is None
    assert abs(info.duration - 0.5) < 0.1


@pytest.mark.slow
def test_probe_video(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    info = probe(path)
    assert info.kind == "video"
    assert info.has_audio is False   # rgb_video fixture is video-only
    assert info.width == w and info.height == h
    assert abs(info.fps - 30.0) < 0.5
    assert abs(info.duration - 1.0) < 0.1
