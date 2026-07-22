"""Media decode wrappers (ffmpeg audio waveforms, preview proxies)."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch

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


def _stderr_tail(err_file, limit: int = 4000) -> str:
    """Read a captured ffmpeg stderr temp file, returning its trailing chars.

    stderr is captured to a temp file rather than a pipe so a chatty ffmpeg
    can't fill a pipe buffer and deadlock against the stdout we stream. The tail
    is what carries the actual failure (missing codec, corrupt input, OOM).
    """
    err_file.seek(0)
    text = err_file.read().decode("utf-8", "replace").strip()
    return text[-limit:]


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
    key = _content_key(path, target_rate)

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

    # -v error (not quiet) so the captured stderr carries the failure reason.
    cmd = [
        ffmpeg, "-nostdin", "-v", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(rate),
        "-f", "f32le", "pipe:1",
    ]
    expected = int(duration * rate * 4) if duration else 0
    buf = bytearray()
    last = -1.0
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_file
        )
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
            logger.error(
                "ffmpeg audio decode failed for %s:\n%s",
                path, _stderr_tail(err_file),
            )
            raise RuntimeError(f"ffmpeg failed to decode audio from {path}")

    if progress is not None:
        progress(1.0)

    if not buf:
        raise ValueError(f"no audio samples decoded from {path}")
    # frombuffer views the bytearray; the single .copy() detaches it, so the
    # waveform is materialized once instead of twice.
    samples = np.frombuffer(buf, dtype=np.float32)
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


def _content_key(path: Path, *extras: object) -> str:
    """Content-addressed key: any edit to the file (mtime/size) invalidates it.

    ``extras`` fold decode parameters into the key so e.g. the same file at
    two target rates caches separately.
    """
    st = path.stat()
    parts = [str(path.resolve()), str(st.st_mtime_ns), str(st.st_size)]
    parts += [str(e) for e in extras]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


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


def source_needs_preview_proxy(info: media.MediaInfo) -> bool:
    """Return True if a source is high-res enough to warrant a preview proxy.

    Sources at or below the proxy height threshold play in real time across a
    mosaic and are used directly; taller footage is transcoded to a light
    proxy. This is the same test ``ensure_preview_proxy`` applies before
    transcoding.

    Args:
        info: Probed source metadata.
    """
    return info.kind == "video" and (info.height or 0) > _PROXY_SOURCE_MAX_HEIGHT


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
    if not source_needs_preview_proxy(info):
        logger.debug("proxy: %s within envelope, using source directly", path.name)
        if progress is not None:
            progress(1.0)
        return path

    key = _content_key(path, _PREVIEW_HEIGHT, _PREVIEW_FPS, _PROXY_VERSION)
    proxy = _cache_dir("video-cache") / f"{key}.mp4"
    if proxy.exists():
        logger.info("proxy: cache hit for %s -> %s", path.name, proxy.name)
        if progress is not None:
            progress(1.0)
        return proxy

    logger.info(
        "proxy: transcoding %s (%sx%s @%sfps) -> %dp/%dfps",
        path.name, info.width, info.height,
        f"{info.fps:.1f}" if info.fps else "?",
        _PREVIEW_HEIGHT, _PREVIEW_FPS,
    )
    _transcode_preview_proxy(path, proxy, info.duration, progress)
    return proxy


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

    if _run_transcode(gpu, duration, progress):
        logger.info("proxy: %s transcoded on GPU pipeline", path.name)
    elif _run_transcode(cpu, duration, progress):
        logger.info("proxy: %s transcoded on CPU pipeline (GPU unavailable)", path.name)
    else:
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
    can fall back to a software pipeline. A failing pipeline's ffmpeg stderr is
    logged (the GPU path failing is expected on non-NVIDIA hosts, so it is a
    warning, not an error).
    """
    last = -1.0
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err_file, text=True
        )
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
        if proc.wait() != 0:
            pipeline = "GPU" if "-hwaccel" in cmd else "CPU"
            logger.warning(
                "ffmpeg %s transcode pipeline failed:\n%s",
                pipeline, _stderr_tail(err_file),
            )
            return False
    return True
