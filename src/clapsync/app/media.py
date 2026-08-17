"""Media probing: unify audio-only and video files behind MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
from framepipe.metadata import extract_video_metadata

from clapsync.app import mocap


@dataclass(frozen=True)
class MediaInfo:
    """Stream metadata for one input file."""

    path: Path
    duration: float
    has_audio: bool
    kind: Literal["audio", "video", "mocap"]
    sample_rate: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    point_rate: float | None = None


def track_family(kind: str) -> Literal["av", "mocap"]:
    """Group a track kind into a family.

    "av" (audio, video) tracks drive audio sync — they are reference-eligible,
    MFCC-aligned, and define the trim window. "mocap" (c3d) tracks carry no
    audio and cannot self-sync; they are placed by the clapperboard link (the
    c3d against the A/V clap).
    """
    return "av" if kind in ("audio", "video") else "mocap"


def is_av(kind: str) -> bool:
    """True for audio/video tracks — the ones that drive audio sync."""
    return track_family(kind) == "av"


def family_display_order(kinds: list[str]) -> list[int]:
    """Row order for track lanes: A/V first, then the mocap (c3d) tracks.

    This is display order only — the returned values are the original track
    indices, each appearing exactly once, so every data lookup still keys by the
    track's own index.

    Args:
        kinds: Track kinds in input order.

    Returns:
        Track indices permuted into display order.
    """
    av = [i for i, kind in enumerate(kinds) if is_av(kind)]
    mocap = [i for i, kind in enumerate(kinds) if kind == "mocap"]
    return av + mocap


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

    A ``.c3d`` file is kind="mocap"; a file with a decodable video stream is
    kind="video"; otherwise it is treated as audio-only.

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

    if path.suffix.lower() == ".c3d":
        point_rate, n_frames = mocap.probe_c3d(path)
        return MediaInfo(
            path=path,
            duration=n_frames / point_rate if point_rate else 0.0,
            has_audio=False,
            kind="mocap",
            point_rate=point_rate,
        )

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
