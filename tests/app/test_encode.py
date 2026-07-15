import av
import pytest
import torch

from clapsync.app.encode import encode_clip, pick_video_codec


def _frames(n=30, h=48, w=64):
    f = torch.zeros((n, 3, h, w), dtype=torch.uint8)
    f[:, 1] = 180  # solid green
    return f


def _tone(seconds=1.0, rate=48000, channels=1):
    import math
    t = torch.arange(int(seconds * rate), dtype=torch.float32) / rate
    return torch.stack([torch.sin(2 * math.pi * 440 * t)] * channels)


def test_pick_video_codec():
    assert pick_video_codec("cpu") == "libx264"
    assert pick_video_codec("cuda") == "h264_nvenc"
    assert pick_video_codec("cuda:0") == "h264_nvenc"


def test_encode_requires_a_stream(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "x.mp4", None, None, None, None)


def test_encode_video_only(tmp_path):
    out = tmp_path / "v.mp4"
    encode_clip(out, _frames(), 30.0, None, None, video_codec="libx264")
    with av.open(str(out)) as c:
        assert not c.streams.audio
        v = c.streams.video[0]
        assert (v.width, v.height) == (64, 48)
        assert sum(1 for _ in c.decode(v)) == 30


def test_encode_muxed_av(tmp_path):
    out = tmp_path / "av.mp4"
    encode_clip(out, _frames(), 30.0, _tone(1.0), 48000, video_codec="libx264")
    with av.open(str(out)) as c:
        assert sum(1 for _ in c.decode(c.streams.video[0])) == 30
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.sample_rate == 48000
        assert sum(f.samples for f in c.decode(a)) == pytest.approx(48000, abs=300)


def test_encode_audio_only_wav_is_sample_exact(tmp_path):
    out = tmp_path / "a.wav"
    encode_clip(out, None, None, _tone(1.0), 48000)
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert sum(f.samples for f in c.decode(a)) == 48000


def test_encode_video_without_fps_raises(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "v.mp4", _frames(), None, None, None)


def test_encode_audio_without_rate_raises(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "a.wav", None, None, _tone(1.0), None)
