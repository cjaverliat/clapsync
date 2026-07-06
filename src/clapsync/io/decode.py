"""Media decode wrappers over torchcodec (audio waveforms, video frames)."""
from __future__ import annotations

from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder


def load_audio(
    path: Path, target_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Load a peak-normalized mono waveform from any audio/video file.

    Args:
        path: Source media path.
        target_rate: If set, decode/resample to this sample rate.

    Returns:
        (waveform, sample_rate) where waveform is float32 shape (1, N).
    """
    decoder = AudioDecoder(str(path), sample_rate=target_rate)
    samples = decoder.get_all_samples()
    data = samples.data  # (num_channels, N) float32 in [-1, 1]
    rate = samples.sample_rate

    mono = data.mean(dim=0, keepdim=True) if data.shape[0] > 1 else data
    peak = mono.abs().max()
    if peak > 0:
        mono = mono / peak
    return mono.contiguous(), int(rate)


def decode_frames_at(
    path: Path, seconds: list[float], device: str = "cpu"
) -> torch.Tensor:
    """Decode one frame per requested timestamp (NCHW uint8).

    Timestamps are clamped into the decodable stream range so callers can pass
    boundary times without raising.

    Args:
        path: Source video path.
        seconds: Presentation timestamps in seconds.
        device: "cpu" or "cuda" (nvdec).

    Returns:
        uint8 tensor of shape (len(seconds), C, H, W) on the requested device.
    """
    decoder = VideoDecoder(str(path), device=device)
    meta = decoder.metadata
    lo = meta.begin_stream_seconds or 0.0
    # Fall back to duration_seconds so the upper clamp still applies when
    # end_stream_seconds is absent; only skip it if neither is known.
    hi = meta.end_stream_seconds
    if hi is None:
        hi = meta.duration_seconds
    eps = 1e-3
    clamped = [
        min(max(s, lo), (hi - eps) if hi is not None else s) for s in seconds
    ]
    return decoder.get_frames_played_at(clamped).data
