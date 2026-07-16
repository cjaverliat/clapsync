"""Media decode wrappers over framepipe/PyAV (audio waveforms, video frames)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch
from framepipe import IndexedFramePrefetcher, PyAvVideoDecoder

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
    boundary times without raising. The frame for time t is the last one whose
    pts <= t.

    Args:
        path: Source video path.
        seconds: Presentation timestamps in seconds; may be unordered and may
            repeat.
        device: "cpu" or "cuda" (nvdec).

    Returns:
        uint8 tensor of shape (len(seconds), C, H, W) on the requested device.
    """
    if not seconds:
        raise ValueError("decode_frames_at needs at least one timestamp")

    # Construct the decoder first and read its metadata (one probe), instead
    # of probing separately here and letting PyAvVideoDecoder._open probe
    # again. This also lets us drop the `start_frame` seek-before-open below:
    # IndexedFramePrefetcher seeks by itself on large forward gaps.
    decoder = PyAvVideoDecoder(str(path), device=device, batch_size=1)
    try:
        pts = decoder.metadata.pts  # (num_frames,) seconds, CPU
        # framepipe's pts is 0-based for CFR but stream-absolute (includes
        # stream.start_time) for VFR. Normalize both to a 0-based, source-
        # relative timeline so caller timestamps (which are source-relative)
        # map correctly regardless of container/framerate mode. No-op for
        # CFR, where pts[0] is already 0.
        pts = pts - pts[0]
        wanted = torch.tensor(seconds, dtype=pts.dtype).clamp_(
            float(pts[0]), float(pts[-1])
        )
        # last frame with pts <= t, matching torchcodec's get_frames_played_at
        idx = (torch.searchsorted(pts, wanted, right=True) - 1).clamp_(
            0, len(pts) - 1
        )

        # The prefetcher consumes a forward-ordered plan; sort, then invert
        # back to caller order so unordered/duplicate timestamps behave.
        order = torch.argsort(idx)
        plan = idx[order].tolist()

        batches = []
        with IndexedFramePrefetcher(decoder, plan) as stream:
            while (batch := stream.next_batch()) is not None:
                batches.append(batch.frames)
    except Exception:
        # If IndexedFramePrefetcher's constructor (or anything above) raises
        # before it takes ownership of `decoder`, nothing else will close it
        # — close it here. On the normal path the prefetcher's context
        # manager already closed it (close_decoder=True, the default), but
        # VideoDecoder.close() is idempotent, so this is never a double-close.
        decoder.close()
        raise

    frames = torch.cat(batches, dim=0)
    return frames[torch.argsort(order)]
