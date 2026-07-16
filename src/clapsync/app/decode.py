"""Media decode wrappers over framepipe/PyAV (audio waveforms, video frames)."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch
from framepipe import IndexedFramePrefetcher, PyAvVideoDecoder

logger = logging.getLogger(__name__)

# Progress is reported at most once per this fraction of a file, so a long clip
# doesn't fire thousands of callbacks the GUI can't draw between.
_PROGRESS_STEP = 0.01
# stdout read granularity while draining ffmpeg. Small enough that even a short
# clip yields many progress ticks; large enough that a 12 min clip is a few
# hundred reads, not thousands.
_READ_CHUNK = 1 << 16


def load_audio(
    path: Path,
    target_rate: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """Load a peak-normalized mono waveform from any audio/video file.

    Decodes with an ``ffmpeg`` subprocess (downmix + resample happen in C, no
    per-frame Python round-trip — an order of magnitude faster than decoding
    frame by frame through PyAV on large 4K containers). The peak-normalized
    result is cached on disk keyed by (path, mtime, size, target_rate), so a
    file that has already been loaded returns instantly on the next open.

    Args:
        path: Source media path.
        target_rate: If set, decode/resample to this sample rate.
        progress: Optional 0..1 callback reporting this file's decode fraction.
            Throttled to ~1% steps, always ending at exactly 1.0.

    Returns:
        (waveform, sample_rate) where waveform is float32 shape (1, N).

    Raises:
        ValueError: If the file has no audio stream.
        RuntimeError: If the ffmpeg binary is unavailable or decoding fails.
    """
    path = Path(path)
    key = _cache_key(path, target_rate)

    cached = _cache_load(key)
    if cached is not None:
        if progress is not None:
            progress(1.0)
        return cached

    rate, duration = _audio_meta(path, target_rate)
    wave = _decode_with_ffmpeg(path, rate, duration, progress)
    _cache_store(key, wave, rate)
    return wave, rate


def _audio_meta(path: Path, target_rate: int | None) -> tuple[int, float]:
    """Resolve output sample rate and source duration via a metadata-only probe.

    Opening the container without decoding is effectively free; it gives the
    native sample rate (when no target is requested) and the duration used to
    scale the progress bar.

    Raises:
        ValueError: If the file has no audio stream.
    """
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        rate = int(target_rate or stream.codec_context.sample_rate)
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
    return rate, duration


def _decode_with_ffmpeg(
    path: Path,
    rate: int,
    duration: float,
    progress: Callable[[float], None] | None,
) -> torch.Tensor:
    """Stream mono f32le PCM out of ffmpeg and peak-normalize it.

    Progress is driven by bytes received (each mono f32 sample is 4 bytes)
    against the expected total from ``duration``, not ffmpeg's own ``-progress``
    output — byte counting stays fine-grained even on short clips.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot decode audio")

    cmd = [
        ffmpeg, "-nostdin", "-v", "quiet",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate),
        "-f", "f32le", "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    expected = int(duration * rate * 4) if duration else 0
    buf = bytearray()
    last = -1.0
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(_READ_CHUNK)
        if not chunk:
            break
        buf += chunk
        if progress is not None and expected:
            fraction = min(len(buf) / expected, 1.0)
            if fraction - last >= _PROGRESS_STEP:
                progress(fraction)
                last = fraction
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio from {path}")

    if progress is not None:
        progress(1.0)

    if not buf:
        raise ValueError(f"no audio samples decoded from {path}")
    samples = np.frombuffer(bytes(buf), dtype=np.float32)
    wave = torch.from_numpy(samples.copy()).unsqueeze(0)  # (1, N)

    peak = wave.abs().max()
    if peak > 0:
        wave = wave / peak
    return wave.contiguous()


def _cache_dir() -> Path:
    """Per-user cache directory for decoded waveforms."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    directory = Path(base) / "clapsync" / "audio-cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cache_key(path: Path, target_rate: int | None) -> str:
    """Content-addressed key: any edit to the file (mtime/size) invalidates it."""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{target_rate}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_load(key: str) -> tuple[torch.Tensor, int] | None:
    """Return the cached (waveform, rate) for ``key``, or None on miss/corruption."""
    cache_file = _cache_dir() / f"{key}.npz"
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file) as data:
            wave = torch.from_numpy(data["wave"])
            return wave, int(data["rate"])
    except Exception:  # noqa: BLE001 - a corrupt cache entry just re-decodes
        logger.debug("ignoring unreadable audio cache entry %s", cache_file)
        return None


def _cache_store(key: str, wave: torch.Tensor, rate: int) -> None:
    """Best-effort persist; a failed write (e.g. full disk) is non-fatal."""
    cache_file = _cache_dir() / f"{key}.npz"
    try:
        np.savez(cache_file, wave=wave.numpy(), rate=np.int64(rate))
    except Exception:  # noqa: BLE001 - caching is an optimization, never required
        logger.debug("could not write audio cache entry %s", cache_file)


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
