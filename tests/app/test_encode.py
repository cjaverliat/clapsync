from fractions import Fraction

import av
import pytest
import torch

from clapsync.app.encode import (
    _audio_codec_for,
    encode_clip,
    pick_video_codec,
    _quality_options,
)


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


def test_encode_video_keeps_fractional_fps_exact(tmp_path):
    """GoPro NTSC rate 60000/1001 (59.94) must not round to 60.

    Rounding it makes the container declare a rate the frames were not
    encoded at, desyncing an exported clip from its own muxed audio.
    """
    out = tmp_path / "ntsc.mp4"
    fps = 60000 / 1001
    encode_clip(out, _frames(n=12), fps, None, None, video_codec="libx264")
    with av.open(str(out)) as c:
        v = c.streams.video[0]
        assert v.average_rate == Fraction(60000, 1001)


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


def test_quality_options_nvenc_uses_cq():
    """h264_nvenc uses cq, not crf. NVENC ignores unknown options silently."""
    result = _quality_options("h264_nvenc", 18)
    assert result == {"cq": "18"}
    assert "crf" not in result


def test_quality_options_nvenc_with_different_crf():
    """Ensure non-default crf values are correctly mapped to cq."""
    result = _quality_options("h264_nvenc", 23)
    assert result == {"cq": "23"}
    assert "crf" not in result


def test_quality_options_libx264_uses_crf():
    """libx264 and other codecs use crf."""
    result = _quality_options("libx264", 18)
    assert result == {"crf": "18"}


def test_quality_options_other_codec_uses_crf():
    """Any codec other than h264_nvenc uses crf."""
    result = _quality_options("libx265", 21)
    assert result == {"crf": "21"}


def test_audio_codec_map_covers_supported_formats():
    from pathlib import Path
    assert _audio_codec_for(Path("x.wav")) == "pcm_s16le"
    assert _audio_codec_for(Path("x.flac")) == "flac"
    assert _audio_codec_for(Path("x.mp3")) == "libmp3lame"
    assert _audio_codec_for(Path("x.m4a")) == "aac"
    assert _audio_codec_for(Path("x.aac")) == "aac"


def test_encode_audio_only_sets_bitrate_for_lossy(tmp_path):
    import av
    out = tmp_path / "a.mp3"
    encode_clip(out, None, None, _tone(1.0), 48000, audio_bitrate=192000)
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        # libmp3lame honors the requested bit_rate on the stream.
        assert a.codec_context.bit_rate in (192000, pytest.approx(192000, abs=32000))
