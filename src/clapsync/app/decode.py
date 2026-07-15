"""Media decode wrappers over framepipe/PyAV (audio waveforms, video frames)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

# Progress is reported at most once per this fraction of a file. A 218 s GoPro
# clip decodes to ~10k AAC frames; calling back on each one costs real time and
# tells the GUI nothing it can draw.
_PROGRESS_STEP = 0.01


def load_audio(
    path: Path,
    target_rate: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """Load a peak-normalized mono waveform from any audio/video file.

    Args:
        path: Source media path.
        target_rate: If set, decode/resample to this sample rate.
        progress: Optional 0..1 callback reporting this file's decode fraction.
            Called as frames arrive, throttled to ~1% steps, ending at 1.0.

    Returns:
        (waveform, sample_rate) where waveform is float32 shape (1, N).

    Raises:
        ValueError: If the file has no audio stream.
    """
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        rate = int(target_rate or stream.codec_context.sample_rate)
        layout = stream.codec_context.layout
        # fltp gives to_ndarray() a planar (channels, samples) float32 array
        # regardless of what the codec natively emits.
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)

        time_base = float(stream.time_base)
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        last = -1.0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
            if progress is None or not duration or frame.pts is None:
                continue
            fraction = min(frame.pts * time_base / duration, 1.0)
            if fraction - last >= _PROGRESS_STEP:
                progress(fraction)
                last = fraction
        for resampled in resampler.resample(None):  # flush
            chunks.append(resampled.to_ndarray())

    if progress is not None:
        progress(1.0)

    if not chunks:
        raise ValueError(f"no audio samples decoded from {path}")
    data = torch.from_numpy(np.concatenate(chunks, axis=1))

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
