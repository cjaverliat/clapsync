"""Media probing: unify audio-only and video files behind MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torchcodec.decoders import AudioDecoder, VideoDecoder


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

    torchcodec raises RuntimeError (undecodable) or ValueError (no such stream)
    when there is no audio; only those are caught so OSError/PermissionError
    and other real failures propagate.
    """
    try:
        meta = AudioDecoder(str(path)).metadata
    except (RuntimeError, ValueError):
        return False, None, None
    return True, int(meta.sample_rate), meta.duration_seconds


def probe(path: Path) -> MediaInfo:
    """Probe a media file via torchcodec metadata.

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

    try:
        vmeta = VideoDecoder(str(path)).metadata
    except (RuntimeError, ValueError):
        vmeta = None

    if vmeta is not None:
        return MediaInfo(
            path=path,
            duration=vmeta.duration_seconds,
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
