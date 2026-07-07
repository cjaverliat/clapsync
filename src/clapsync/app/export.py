"""Trim/pad export of synced tracks (window math is pure; I/O is separate)."""
from __future__ import annotations

import functools
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch

from clapsync.app.decode import decode_frames_at, load_audio
from clapsync.app.encode import encode_clip
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


def _build_video_frames(
    info: MediaInfo,
    offset: float,
    trim: TimeRange,
    out_fps: float,
    out_w: int,
    out_h: int,
) -> torch.Tensor:
    """Assemble the output frame tensor on the trim grid, black-padding gaps.

    Frames are assembled on CPU (uniform device for stacking); encode_clip moves
    the batch to the encode device.

    Args:
        info: Probed source track.
        offset: Track offset on the shared timeline (seconds).
        trim: Output range on the shared timeline.
        out_fps: Output frame rate.
        out_w: Output width in pixels.
        out_h: Output height in pixels.

    Returns:
        uint8 tensor of shape (N, 3, out_h, out_w) on CPU.
    """
    times = frame_source_times(offset, trim, out_fps)
    in_range = [0.0 <= t < info.duration for t in times]
    wanted_idx = [k for k, ok in enumerate(in_range) if ok]
    decoded_by_idx: dict[int, torch.Tensor] = {}
    if wanted_idx:
        decoded = decode_frames_at(
            info.path, [times[k] for k in wanted_idx], device="cpu",
        )
        for k, frame in zip(wanted_idx, decoded):
            decoded_by_idx[k] = frame

    black = torch.zeros((3, out_h, out_w), dtype=torch.uint8)
    frames = []
    for k, ok in enumerate(in_range):
        if not ok:
            frames.append(black)
            continue
        frame = decoded_by_idx[k]
        if frame.shape[-1] != out_w or frame.shape[-2] != out_h:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0).float(),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0, 255).to(torch.uint8)
        frames.append(frame)
    return torch.stack(frames)


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
                out_fps = settings.output_fps or info.fps or 30.0
                out_w = settings.target_width or info.width
                out_h = settings.target_height or info.height
                frames = _build_video_frames(
                    info, offset, trim, out_fps, out_w, out_h,
                )
                audio = None
                rate = None
                if info.has_audio:
                    audio, rate = _build_audio_samples(info, offset, trim)
                ext = "mp4"
                out_path = settings.output_dir / f"{info.path.stem}_synced.{ext}"
                try:
                    encode_clip(
                        out_path, frames, out_fps, audio, rate,
                        video_codec=settings.video_codec, crf=settings.crf,
                        device=device,
                    )
                except Exception as exc:  # noqa: BLE001
                    # NVENC can still reject an individual clip (e.g. below its
                    # minimum resolution). Fall back to CPU libx264 for this
                    # clip unless the caller pinned a specific codec.
                    if device != "cuda" or settings.video_codec is not None:
                        raise
                    logger.warning(
                        "GPU encode failed for %s (%s) — retrying on CPU",
                        info.path, exc,
                    )
                    encode_clip(
                        out_path, frames, out_fps, audio, rate,
                        video_codec="libx264", crf=settings.crf, device="cpu",
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
