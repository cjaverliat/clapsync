"""CPU vs GPU coverage for decode/encode/export.

GPU cases are skipped when no CUDA device is present, so the suite stays green
on CPU-only machines while exercising both paths where a GPU exists. NVENC
requires frames >= ~256x144, so GPU encode tests use that size.
"""
import pytest
import torch

from torchcodec.decoders import VideoDecoder

from clapsync.app.decode import decode_frames_at
from clapsync.app.encode import encode_clip
from clapsync.app.export import _nvenc_available, sync_and_trim

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


@pytest.mark.slow
def test_encode_cpu_libx264(tmp_path):
    out = tmp_path / "cpu.mp4"
    frames = torch.zeros((10, 3, 144, 256), dtype=torch.uint8)
    frames[:, 1] = 150
    encode_clip(out, frames, 30.0, None, None, video_codec="libx264", device="cpu")
    assert VideoDecoder(str(out)).metadata.num_frames >= 9


@pytest.mark.slow
@cuda_only
def test_encode_gpu_nvenc(tmp_path):
    out = tmp_path / "gpu.mp4"
    frames = torch.zeros((10, 3, 144, 256), dtype=torch.uint8)
    frames[:, 2] = 150
    encode_clip(
        out, frames, 30.0, None, None, video_codec="h264_nvenc", device="cuda",
    )
    assert VideoDecoder(str(out)).metadata.num_frames >= 9


@pytest.mark.slow
@cuda_only
def test_nvenc_probe_true_on_gpu():
    _nvenc_available.cache_clear()
    assert _nvenc_available() is True
    _nvenc_available.cache_clear()


@pytest.mark.slow
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=cuda_only)]
)
def test_decode_frames_on_device(rgb_video, device):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=256, h=144)
    frames = decode_frames_at(path, [0.0, 0.5], device=device)
    assert frames.shape == (2, 3, h, w)
    assert frames.dtype == torch.uint8
    assert frames.device.type == device


@pytest.mark.slow
@cuda_only
def test_export_end_to_end_uses_gpu(av_video, tmp_path):
    # >= 256x144 so NVENC is actually exercised end-to-end (not the CPU
    # fallback that tiny clips trigger).
    a, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="a.mp4")
    b, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="b.mp4")
    out = tmp_path / "out"
    out.mkdir()
    results = sync_and_trim([a, b], out, trim="common")
    assert all(r.ok for r in results), [r.error for r in results]
    for r in results:
        assert r.path.exists() and r.path.stat().st_size > 0
