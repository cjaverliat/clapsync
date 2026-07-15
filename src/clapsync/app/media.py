"""Media probing: unify audio-only and video files behind MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
from framepipe.metadata import extract_video_metadata


@dataclass(frozen=True)
class MediaInfo:
    """Stream metadata for one input file."""

    path: Path
    duration: float
    has_audio: bool
    kind: Literal["audio", "video"]
    sample_rate: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def _audio_meta(path: Path) -> tuple[bool, int | None, float | None]:
    """Return (has_audio, sample_rate, duration) or (False, None, None).

    A container with no audio stream is not an error; av raises only when the
    file itself is undecodable, which the caller reports as "no streams".
    """
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return False, None, None
            stream = container.streams.audio[0]
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration
                else None
            )
            return True, int(stream.codec_context.sample_rate), duration
    except av.FFmpegError:
        return False, None, None


def probe(path: Path) -> MediaInfo:
    """Probe a media file via framepipe (video) and PyAV (audio) metadata.

    A file with a decodable video stream is kind="video"; otherwise it is
    treated as audio-only.

    Args:
        path: Source media path.

    Returns:
        A populated MediaInfo.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has neither a video nor an audio stream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    has_audio, sample_rate, audio_dur = _audio_meta(path)

    # extract_video_metadata indexes streams.video[0] unguarded, so audio-only
    # input raises bare IndexError. We catch it to detect "no video stream" and
    # fall back to audio-only.
    try:
        vmeta = extract_video_metadata(str(path))
    except (av.FFmpegError, ValueError, RuntimeError, IndexError):
        vmeta = None

    if vmeta is not None:
        return MediaInfo(
            path=path,
            duration=vmeta.duration,
            has_audio=has_audio,
            kind="video",
            sample_rate=sample_rate,
            width=vmeta.width,
            height=vmeta.height,
            fps=vmeta.average_fps,
        )

    if has_audio:
        return MediaInfo(
            path=path,
            duration=audio_dur if audio_dur is not None else 0.0,
            has_audio=True,
            kind="audio",
            sample_rate=sample_rate,
        )

    raise ValueError(f"No decodable audio or video stream in {path}")
