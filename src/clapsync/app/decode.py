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

from clapsync.app import media

logger = logging.getLogger(__name__)

# Progress is reported at most once per this fraction of a file, so a long clip
# doesn't fire thousands of callbacks the GUI can't draw between.
_PROGRESS_STEP = 0.01
# stdout read granularity while draining ffmpeg. Small enough that even a short
# clip yields many progress ticks; large enough that a 12 min clip is a few
# hundred reads, not thousands.
_READ_CHUNK = 1 << 16
# Preview proxy policy. Sources at or below _PROXY_SOURCE_MAX_HEIGHT already
# decode in real time across a mosaic (1080p measured at ~3.4x), so they play
# as-is; taller ones (e.g. 5.3K GoPro) are transcoded once to a light
# _PREVIEW_HEIGHT/_PREVIEW_FPS proxy and cached.
_PROXY_SOURCE_MAX_HEIGHT = 1080
_PREVIEW_HEIGHT = 480
_PREVIEW_FPS = 30
# Bumped whenever the transcode parameters change, so proxies cached under the
# old settings are regenerated instead of served stale.
_PROXY_VERSION = 4


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


def _cache_dir(kind: str = "audio-cache") -> Path:
    """Per-user cache directory for decoded media (waveforms, preview proxies)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    directory = Path(base) / "clapsync" / kind
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


def ensure_preview_proxy(
    path: Path, progress: Callable[[float], None] | None = None
) -> Path:
    """Return a 480p/30fps proxy of ``path``, transcoding+caching on first use.

    High-resolution source (e.g. 5.3K 60fps GoPro HEVC) decodes far too slowly
    to sustain real-time multi-track mosaic preview. A one-time single-process
    ffmpeg transcode (GPU decode -> GPU scale -> GPU encode; no frame ever
    crosses into Python) produces a light proxy that plays with headroom. The
    proxy is cached on disk keyed by (path, mtime, size), so later opens return
    it instantly.

    Sources at or below 1080p already play in real time, so they are returned
    unchanged; only taller footage is proxied.

    Args:
        path: Source video path.
        progress: Optional 0..1 callback reporting the transcode fraction.

    Returns:
        Path to a playable proxy, or ``path`` itself when no proxy is needed.

    Raises:
        RuntimeError: If ffmpeg is unavailable or transcoding fails.
    """
    path = Path(path)
    info = media.probe(path)
    if info.kind != "video" or (info.height or 0) <= _PROXY_SOURCE_MAX_HEIGHT:
        if progress is not None:
            progress(1.0)
        return path

    proxy = _cache_dir("video-cache") / f"{_proxy_key(path)}.mp4"
    if proxy.exists():
        if progress is not None:
            progress(1.0)
        return proxy

    _transcode_preview_proxy(path, proxy, info.duration, progress)
    return proxy


def _proxy_key(path: Path) -> str:
    """Content-addressed key: any edit to the file (mtime/size) invalidates it."""
    st = path.stat()
    raw = (
        f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"
        f"|{_PREVIEW_HEIGHT}|{_PREVIEW_FPS}|{_PROXY_VERSION}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _transcode_preview_proxy(
    path: Path,
    proxy: Path,
    duration: float,
    progress: Callable[[float], None] | None,
) -> None:
    """Transcode ``path`` into a 480p/30fps proxy at ``proxy`` (atomically).

    A single ffmpeg process does GPU decode, scale and encode. The CPU path is
    tried only if the GPU pipeline fails to start (no NVDEC/NVENC). Output is
    written to a temp file and renamed into place, so a killed transcode never
    leaves a half-written proxy that a later run would treat as a cache hit.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; cannot build preview proxy")

    tmp = proxy.with_suffix(".partial.mp4")
    enc = [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an", "-progress", "pipe:1", "-nostats", str(tmp),
    ]
    head = [ffmpeg, "-nostdin", "-y", "-v", "error"]
    # Encode with libx264, not NVENC: NVENC-produced streams intermittently fall
    # back to *software* decode under PyAV's CUDA hwaccel, whereas libx264
    # streams reliably NVDEC-decode across a mosaic. Decode+scale still run on
    # the GPU (scale_cuda), so transcode stays decode-bound (same speed as
    # all-NVENC); only the light 480p encode is on the CPU. bilinear scale is
    # used over lanczos: it is ~2x faster on scale_cuda and the difference is
    # slight once the display resamples again.
    gpu = head + [
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(path),
        "-vf",
        f"scale_cuda=-2:{_PREVIEW_HEIGHT}:interp_algo=bilinear"
        f",hwdownload,format=nv12,fps={_PREVIEW_FPS}",
        *enc,
    ]
    # CPU fallback when no NVIDIA GPU is present: software decode/scale/encode.
    cpu = head + [
        "-i", str(path),
        "-vf", f"scale=-2:{_PREVIEW_HEIGHT}:flags=bilinear,fps={_PREVIEW_FPS}",
        *enc,
    ]

    if not _run_transcode(gpu, duration, progress):
        if not _run_transcode(cpu, duration, progress):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed to transcode preview proxy from {path}")
    os.replace(tmp, proxy)
    if progress is not None:
        progress(1.0)


def _run_transcode(
    cmd: list[str],
    duration: float,
    progress: Callable[[float], None] | None,
) -> bool:
    """Run one ffmpeg transcode, driving ``progress`` from its ``-progress`` feed.

    Returns True on a clean (exit 0) transcode, False otherwise, so the caller
    can fall back to a software pipeline.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    last = -1.0
    assert proc.stdout is not None
    for line in proc.stdout:
        if progress is None or duration <= 0:
            continue
        key, _, value = line.strip().partition("=")
        if key != "out_time_us":
            continue
        try:
            seconds = int(value) / 1_000_000
        except ValueError:
            continue  # ffmpeg emits "N/A" before the first frame lands
        fraction = min(seconds / duration, 1.0)
        if fraction - last >= _PROGRESS_STEP:
            progress(fraction)
            last = fraction
    return proc.wait() == 0
