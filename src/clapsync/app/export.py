"""Trim/pad export of synced tracks (window math is pure; I/O is separate)."""
from __future__ import annotations

import functools
import logging
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal

import av
import numpy as np
import torch

from clapsync.app.decode import load_audio
from clapsync.app.encode import (
    _AUDIO_BLOCK,
    _audio_codec_for,
    _quality_options,
    encode_clip,
    pick_video_codec,
)
from clapsync.app.media import MediaInfo, probe
from clapsync.app.sync import offsets_from_media
from clapsync.core.offsets import Refine
from clapsync.core.timerange import TimeRange, common_time_range, full_time_range

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    """Return True if h264_nvenc can actually encode on this machine.

    CUDA being present is not sufficient — the NVENC path can fail to
    initialize (driver/SDK/build mismatch). Probe once by encoding a tiny
    clip to a temp file; cache the verdict for the process. On failure the
    export falls back to CPU libx264.
    """
    if not torch.cuda.is_available():
        return False
    # Probe at a realistic size: NVENC rejects tiny frames (min ~256x144), so a
    # 16x16 probe would false-negative on machines where NVENC actually works.
    frames = torch.zeros((2, 3, 144, 256), dtype=torch.uint8)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            encode_clip(
                Path(tmp) / "nvenc_probe.mp4",
                frames, 30.0, None, None, device="cuda",
            )
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "use CPU"
        logger.warning("h264_nvenc unavailable (%s) — using CPU libx264", exc)
        return False


@dataclass(frozen=True)
class ExportSettings:
    """Output settings for a sync/trim export."""

    trim: TimeRange
    output_dir: Path
    target_width: int | None = None
    target_height: int | None = None
    output_fps: float | None = None
    video_codec: str | None = None
    crf: int = 18
    audio_format: str | None = None


@dataclass(frozen=True)
class ExportResult:
    """Outcome for one exported track."""

    path: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def clip_window(
    offset: float, duration: float, trim: TimeRange
) -> tuple[float, float, float, float]:
    """Map a shared-timeline trim onto one track's local source range.

    Args:
        offset: Track offset on the shared timeline (seconds).
        duration: Track source duration (seconds).
        trim: Desired output range on the shared timeline.

    Returns:
        (local_start, local_end, pad_start, pad_end) in seconds. local_* index
        into the source; pad_* are black/silence gaps outside the source.
    """
    local_start = max(0.0, trim.start - offset)
    local_end = min(duration, trim.end - offset)
    pad_start = max(0.0, offset - trim.start)
    pad_end = max(0.0, trim.end - (offset + duration))
    return local_start, local_end, pad_start, pad_end


def frame_source_times(
    offset: float, trim: TimeRange, out_fps: float
) -> list[float]:
    """Source timestamps for each output frame on the shared trim grid.

    Output frame k sits at shared time trim.start + k / out_fps; the matching
    source time subtracts the track offset. Values outside [0, duration) mark
    frames the caller must black-pad. This keeps subframe offsets exact — the
    output grid is fixed, the source is sampled at the precise shifted time.

    Args:
        offset: Track offset on the shared timeline (seconds).
        trim: Output range on the shared timeline.
        out_fps: Output frame rate.

    Returns:
        One source timestamp per output frame.
    """
    n_frames = round(trim.duration * out_fps)
    return [trim.start + k / out_fps - offset for k in range(n_frames)]


def _video_filter_chain(
    window: tuple[float, float, float, float],
    out_w: int,
    out_h: int,
    out_fps: float,
    native_size: bool,
    native_fps: bool,
) -> str:
    """Build the libavfilter chain that trims, retimes, resizes and pads a track.

    ``trim`` selects the in-source span; ``setpts`` rebases it to a 0-based
    output timeline while subtracting the exact (fractional) local start, so the
    sub-frame offset is preserved as real PTS rather than snapped to a frame.
    ``scale``/``fps`` are omitted on the native fast path (just trim + pad),
    and ``tpad`` fills the black gaps outside the source.
    """
    local_start, local_end, pad_start, pad_end = window
    parts = [
        f"trim=start={local_start:.6f}:end={local_end:.6f}",
        f"setpts=PTS-{local_start:.6f}/TB",
    ]
    if not native_size:
        parts.append(f"scale={out_w}:{out_h}")
    if not native_fps:
        parts.append(f"fps={out_fps}")
    if pad_start > 1e-6 or pad_end > 1e-6:
        parts.append(
            f"tpad=start_duration={pad_start:.6f}"
            f":stop_duration={pad_end:.6f}:color=black"
        )
    return ",".join(parts)


def _stream_encode_video(
    container: av.container.OutputContainer,
    vstream: av.stream.Stream,
    info: MediaInfo,
    window: tuple[float, float, float, float],
    out_w: int,
    out_h: int,
    out_fps: float,
    native_size: bool,
    native_fps: bool,
) -> None:
    """Decode → filter → encode one video track into ``vstream``.

    Frames flow one at a time through a libavfilter graph and out to the
    encoder, so peak memory is a handful of frames regardless of clip length —
    unlike decoding the whole clip into a tensor first. The source is seeked to
    the keyframe at/before the trim start so a late trim doesn't decode from 0.
    ``vstream`` must already be added to ``container`` (with any audio stream)
    before muxing begins, or the container header is finalized without them.
    """
    local_start = window[0]

    with av.open(str(info.path)) as source:
        istream = source.streams.video[0]
        istream.thread_type = "AUTO"
        if local_start > 0:
            source.seek(
                round(local_start / istream.time_base),
                stream=istream,
                backward=True,
            )

        graph = av.filter.Graph()
        # Explicit buffer args (not template=): tpad needs a frame_rate to know
        # how many black frames a pad duration is; a template buffer omits it,
        # silently dropping the pad.
        source_fps = Fraction(info.fps or out_fps).limit_denominator(65535)
        buffer = graph.add(
            "buffer",
            f"width={istream.codec_context.width}"
            f":height={istream.codec_context.height}"
            f":pix_fmt={istream.codec_context.format.name}"
            f":time_base={istream.time_base}"
            f":frame_rate={source_fps}",
        )
        node = buffer
        for spec in _video_filter_chain(
            window, out_w, out_h, out_fps, native_size, native_fps
        ).split(","):
            name, _, args = spec.partition("=")
            step = graph.add(name, args or None)
            node.link_to(step)
            node = step
        sink = graph.add("buffersink")
        node.link_to(sink)
        graph.configure()

        def drain() -> None:
            while True:
                try:
                    out_frame = sink.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    return
                for packet in vstream.encode(out_frame):
                    container.mux(packet)

        try:
            for frame in source.decode(istream):
                try:
                    buffer.push(frame)
                except av.error.EOFError:
                    break  # trim end reached; graph closed the stream
                drain()
        finally:
            try:
                buffer.push(None)
            except av.error.EOFError:
                pass
            drain()
            for packet in vstream.encode():
                container.mux(packet)


def _mux_waveform_audio(
    container: av.container.OutputContainer,
    astream: av.stream.Stream,
    layout: str,
    audio: torch.Tensor,
    rate: int,
) -> None:
    """Mux the pre-trimmed waveform (small, already in memory) into ``astream``."""
    resampler = av.AudioResampler(
        format=astream.codec_context.format.name, layout=layout, rate=rate
    )
    samples = audio.cpu().numpy().astype(np.float32)
    pts = 0
    for start in range(0, samples.shape[1], _AUDIO_BLOCK):
        block = np.ascontiguousarray(samples[:, start : start + _AUDIO_BLOCK])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = pts
        pts += block.shape[1]
        for resampled in resampler.resample(frame):
            for packet in astream.encode(resampled):
                container.mux(packet)
    for resampled in resampler.resample(None):
        for packet in astream.encode(resampled):
            container.mux(packet)
    for packet in astream.encode():
        container.mux(packet)


def _export_video_track(
    info: MediaInfo,
    offset: float,
    trim: TimeRange,
    settings: ExportSettings,
    out_path: Path,
    device: str,
) -> None:
    """Stream one synced video track (+audio) to ``out_path`` at flat memory.

    Video is trimmed/retimed/resized/padded by a libavfilter graph and encoded
    frame-by-frame; when the output matches the source resolution and fps the
    scale/fps stages are dropped (a plain trim, sub-frame alignment intact).
    Audio is the already-trimmed waveform, muxed alongside. NVENC failures fall
    back to CPU libx264 unless a codec was pinned.
    """
    out_fps = settings.output_fps or info.fps or 30.0
    out_w = settings.target_width or info.width
    out_h = settings.target_height or info.height
    native_size = out_w == info.width and out_h == info.height
    native_fps = abs(out_fps - (info.fps or out_fps)) < 1e-3
    window = clip_window(offset, info.duration, trim)
    audio = None
    rate = None
    if info.has_audio:
        audio, rate = _build_audio_samples(info, offset, trim)

    def write(codec: str) -> None:
        with av.open(str(out_path), mode="w") as container:
            # Add every stream before muxing any packet — the first mux
            # finalizes the container header, and a stream added afterward gets
            # no time_base (mux then raises "Cannot rebase to zero time").
            vstream = container.add_stream(
                codec, rate=Fraction(out_fps).limit_denominator(65535)
            )
            vstream.width = out_w
            vstream.height = out_h
            vstream.pix_fmt = "yuv420p"
            vstream.options = _quality_options(codec, settings.crf)

            astream = None
            layout = None
            if audio is not None:
                layout = "mono" if audio.shape[0] == 1 else "stereo"
                astream = container.add_stream(
                    _audio_codec_for(out_path), rate=rate, layout=layout
                )

            _stream_encode_video(
                container, vstream, info, window, out_w, out_h, out_fps,
                native_size, native_fps,
            )
            if astream is not None:
                _mux_waveform_audio(container, astream, layout, audio, rate)

    codec = settings.video_codec or pick_video_codec(device)
    try:
        write(codec)
    except Exception as exc:  # noqa: BLE001
        # NVENC can reject a clip (e.g. below its minimum resolution). Fall back
        # to CPU libx264 unless the caller pinned a codec.
        if not str(device).startswith("cuda") or settings.video_codec is not None:
            raise
        logger.warning(
            "GPU encode failed for %s (%s) — retrying on CPU", info.path, exc
        )
        write("libx264")


def _build_audio_samples(
    info: MediaInfo,
    offset: float,
    trim: TimeRange,
) -> tuple[torch.Tensor, int]:
    """Trim + silence-pad the track audio to the shared trim range.

    Args:
        info: Probed source track (must have audio).
        offset: Track offset on the shared timeline (seconds).
        trim: Output range on the shared timeline.

    Returns:
        (samples, sample_rate) where samples has shape (channels, N).
    """
    wave, rate = load_audio(info.path)
    local_start, local_end, pad_start, pad_end = clip_window(
        offset, info.duration, trim,
    )
    s0 = max(0, round(local_start * rate))
    s1 = min(wave.shape[1], round(local_end * rate))
    core = wave[:, s0:s1] if s1 > s0 else wave[:, :0]
    total = round(trim.duration * rate)
    pad_front = round(pad_start * rate)
    out = torch.zeros((core.shape[0], total), dtype=core.dtype)
    end = min(total, pad_front + core.shape[1])
    out[:, pad_front:end] = core[:, : end - pad_front]
    return out, rate


def export_media(
    media: list[MediaInfo],
    offsets: list[float],
    settings: ExportSettings,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Encode already-probed tracks (no re-probe); see export_tracks."""
    device = "cuda" if _nvenc_available() else "cpu"
    trim = settings.trim
    results: list[ExportResult] = []
    n = len(media)

    for i, (info, offset) in enumerate(zip(media, offsets)):
        if is_cancelled is not None and is_cancelled():
            break
        try:
            if info.kind == "video":
                out_path = settings.output_dir / f"{info.path.stem}_synced.mp4"
                _export_video_track(
                    info, offset, trim, settings, out_path, device,
                )
            else:
                audio, rate = _build_audio_samples(info, offset, trim)
                ext = (
                    settings.audio_format
                    or info.path.suffix.lstrip(".")
                    or "wav"
                )
                out_path = settings.output_dir / f"{info.path.stem}_synced.{ext}"
                encode_clip(
                    out_path, None, None, audio, rate, device="cpu",
                )
            results.append(ExportResult(out_path))
        except Exception as exc:  # noqa: BLE001 — record per-track failure
            logger.exception("export failed for %s", info.path)
            results.append(ExportResult(
                settings.output_dir / f"{info.path.stem}_synced",
                str(exc),
            ))
        if progress is not None:
            progress((i + 1) / n)

    return results


def export_tracks(
    paths: list[Path],
    offsets: list[float],
    settings: ExportSettings,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Trim+pad every track to settings.trim and encode one file each.

    Video tracks produce a muxed A/V MP4; audio-only tracks produce an audio
    file. Video and audio share the exact subframe trim origin.

    Args:
        paths: Input media file paths.
        offsets: Per-track shared-timeline offsets (seconds).
        settings: Output resolution/fps/codec/dir.
        progress: Optional 0..1 callback.
        is_cancelled: Optional cancel check.

    Returns:
        One ExportResult per track (in input order).
    """
    if is_cancelled is not None and is_cancelled():
        return []
    media = [probe(p) for p in paths]
    return export_media(media, offsets, settings, progress, is_cancelled)


def sync_and_trim(
    paths: list[Path],
    output_dir: Path,
    *,
    refine: Refine = "parabolic",
    trim: Literal["common", "full"] = "common",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Probe, sync, pick a trim range, and export — the one-call convenience.

    The inputs are probed once here and the resulting MediaInfo is reused for
    both alignment and export, so each file is only probed a single time.

    Args:
        paths: Input media files.
        output_dir: Destination directory (created if missing).
        refine: Peak refinement.
        trim: "common" (overlap) or "full" (union) output range.
        reference_index: Reference track.
        progress: Optional 0..1 callback (spans sync then export).
        is_cancelled: Optional cancel check.

    Returns:
        One ExportResult per input.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    media = [probe(p) for p in paths]

    def sync_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 * f)

    offsets = offsets_from_media(
        media,
        reference_index=reference_index,
        refine=refine,
        progress=sync_progress,
        is_cancelled=is_cancelled,
    )

    durations = [m.duration for m in media]
    rng = (common_time_range if trim == "common" else full_time_range)(
        durations, offsets,
    )
    settings = ExportSettings(trim=rng, output_dir=output_dir)

    def export_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 + 0.5 * f)

    return export_media(
        media, offsets, settings,
        progress=export_progress, is_cancelled=is_cancelled,
    )
