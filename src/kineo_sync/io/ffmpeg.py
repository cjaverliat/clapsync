from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from kineo_sync.io.video import get_video_info

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _log_ffmpeg_info() -> None:
    path = shutil.which("ffmpeg") or "ffmpeg"
    try:
        out = subprocess.check_output([path, "-version"], stderr=subprocess.STDOUT, text=True)
        version_line = out.splitlines()[0] if out else "unknown"
    except Exception:
        version_line = "unavailable"
    logger.info("ffmpeg: %s  (%s)", version_line, path)


def export_synced_video(
    input_path: Path,
    output_path: Path,
    local_start_s: float,
    local_end_s: float,
    pad_start: float = 0.0,
    pad_end: float = 0.0,
    target_width: int | None = None,
    target_height: int | None = None,
    output_fps: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    info = get_video_info(input_path)
    source_fps = info.fps
    effective_fps = output_fps if output_fps is not None else source_fps
    out_w = max(2, (target_width if target_width is not None else info.width) & ~1)
    out_h = max(2, (target_height if target_height is not None else info.height) & ~1)

    has_video_content = local_end_s > local_start_s
    needs_padding = pad_start > 0.0 or pad_end > 0.0

    total_duration = (local_end_s - local_start_s if has_video_content else 0.0) + pad_start + pad_end
    if total_duration <= 0:
        raise ValueError(f"Invalid export range: nothing to output")

    if not has_video_content:
        _run_ffmpeg_black(output_path, total_duration, out_w, out_h, effective_fps, info.has_audio, progress_callback, is_cancelled)
    elif needs_padding or not (out_w == info.width and out_h == info.height and abs(effective_fps - source_fps) < 0.01):
        _run_ffmpeg_transcode(
            input_path, output_path,
            local_start_s, local_end_s - local_start_s,
            pad_start, pad_end,
            out_w, out_h, effective_fps, info.has_audio,
            progress_callback, is_cancelled,
        )
    else:
        _run_ffmpeg_stream_copy(
            input_path, output_path,
            local_start_s, local_end_s - local_start_s,
            info.has_audio,
            progress_callback, is_cancelled,
        )

    if progress_callback is not None:
        progress_callback(1.0)


def _run_ffmpeg(
    cmd: list[str],
    duration: float,
    progress_callback: Callable[[float], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    _log_ffmpeg_info()
    logger.info("ffmpeg command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_lines: list[str] = []

    def _read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            logger.debug("ffmpeg: %s", line.rstrip())

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    cancelled = False
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if is_cancelled is not None and is_cancelled():
            proc.kill()
            cancelled = True
            break

        # -progress pipe:1 emits key=value lines
        if line.startswith("out_time_ms=") and progress_callback is not None and duration > 0:
            try:
                out_time_s = float(line.split("=", 1)[1]) / 1_000_000.0
                progress_callback(min(1.0, out_time_s / duration))
            except (ValueError, IndexError):
                pass

    proc.wait()
    stderr_thread.join(timeout=2.0)
    stderr = "".join(stderr_lines)

    if not cancelled and proc.returncode != 0:
        logger.error("ffmpeg failed (rc=%d):\n%s", proc.returncode, stderr)
        raise RuntimeError(f"ffmpeg failed (exit code {proc.returncode}):\n{stderr}")


def _run_ffmpeg_stream_copy(
    input_path: Path,
    output_path: Path,
    start_s: float,
    duration: float,
    has_audio: bool,
    progress_callback: Callable[[float], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.6f}",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]
    _run_ffmpeg(cmd, duration, progress_callback, is_cancelled)


def _run_ffmpeg_transcode(
    input_path: Path,
    output_path: Path,
    start_s: float,
    duration: float,
    pad_start: float,
    pad_end: float,
    out_w: int,
    out_h: int,
    output_fps: float,
    has_audio: bool,
    progress_callback: Callable[[float], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    vf = f"scale={out_w}:{out_h}"
    if pad_start > 0.0 or pad_end > 0.0:
        vf += f",tpad=start_duration={pad_start:.6f}:start_mode=black:stop_duration={pad_end:.6f}:stop_mode=black"

    total_duration = duration + pad_start + pad_end

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.6f}",
        "-i", str(input_path),
        "-t", f"{duration:.6f}",
        "-vf", vf,
        "-r", f"{output_fps:.6f}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        af_parts = []
        if pad_start > 0.0:
            af_parts.append(f"adelay={int(pad_start * 1000)}:all=1")
        if pad_end > 0.0:
            af_parts.append(f"apad=pad_dur={pad_end:.6f}")
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]
        cmd += ["-c:a", "aac"]
    else:
        cmd += ["-an"]
    cmd += [
        "-t", f"{total_duration:.6f}",
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]
    _run_ffmpeg(cmd, total_duration, progress_callback, is_cancelled)


def _run_ffmpeg_black(
    output_path: Path,
    duration: float,
    out_w: int,
    out_h: int,
    output_fps: float,
    has_audio: bool,
    progress_callback: Callable[[float], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=black:size={out_w}x{out_h}:rate={output_fps:.6f}",
    ]
    if has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc"]
    cmd += [
        "-t", f"{duration:.6f}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += [
        "-progress", "pipe:1",
        "-nostats",
        str(output_path),
    ]
    _run_ffmpeg(cmd, duration, progress_callback, is_cancelled)
