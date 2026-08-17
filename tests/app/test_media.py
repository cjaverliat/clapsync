import pytest

from clapsync.app.media import family_display_order, is_av, probe, track_family


def test_track_family_and_is_av():
    assert track_family("audio") == "av"
    assert track_family("video") == "av"
    assert track_family("mocap") == "mocap"
    assert is_av("video") and is_av("audio")
    assert not is_av("mocap")


def test_family_display_order_av_first_then_c3d():
    # video, c3d, audio  ->  A/V (0, 2) first, then the c3d (1).
    order = family_display_order(["video", "mocap", "audio"])
    assert order == [0, 2, 1]
    # Every track appears exactly once — indices are permuted, never renamed.
    assert sorted(order) == [0, 1, 2]


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


def test_probe_non_media_file_raises(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 10)
    with pytest.raises(ValueError):
        probe(junk)


def test_probe_audio_only(tone_wav):
    path, rate = tone_wav(seconds=1.0, sample_rate=48000)
    info = probe(path)
    assert info.kind == "audio"
    assert info.has_audio is True
    assert info.sample_rate == rate
    assert info.width is None and info.height is None
    assert abs(info.duration - 1.0) < 0.1


def test_probe_video_without_audio(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    info = probe(path)
    assert info.kind == "video"
    assert info.has_audio is False
    assert (info.width, info.height) == (w, h)
    assert abs(info.fps - fps) < 0.1
    assert abs(info.duration - 1.0) < 0.1


def test_probe_muxed_av(av_video):
    path, fps, n, w, h, sr = av_video(seconds=1.0)
    info = probe(path)
    assert info.kind == "video"
    assert info.has_audio is True
    assert info.sample_rate == sr
    assert (info.width, info.height) == (w, h)
