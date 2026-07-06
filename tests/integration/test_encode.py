"""Integration tests for clapsync.io.encode."""
import pytest
import torch

from clapsync.io.encode import encode_clip, pick_video_codec


def test_pick_video_codec_is_pure():
    assert pick_video_codec("cpu") == "libx264"
    assert pick_video_codec("cuda") == "h264_nvenc"


@pytest.mark.slow
def test_encode_clip_muxes_audio_and_video(tmp_path):
    out = tmp_path / "clip.mp4"
    frames = torch.zeros((15, 3, 48, 64), dtype=torch.uint8)
    frames[:, 1] = 200  # green
    samples = torch.zeros((1, 24000), dtype=torch.float32)  # 0.5 s @ 48k
    encode_clip(out, frames, 30.0, samples, 48000, device="cpu")
    assert out.exists() and out.stat().st_size > 0

    from torchcodec.decoders import AudioDecoder, VideoDecoder

    assert VideoDecoder(str(out)).metadata.num_frames >= 14
    assert AudioDecoder(str(out)).metadata.sample_rate == 48000
