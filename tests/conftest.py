"""Shared test fixtures.

Synthetic media are generated with PyAV so decode/probe/export tests have
known-content inputs without checking binaries into git. These deliberately do
not use clapsync's own encode_clip — a bug there must fail the tests, not
silently produce matching fixtures.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch


def _tone(seconds: float, sample_rate: int, freq: float) -> torch.Tensor:
    """Mono sine tone, shape (1, N), float in [-1, 1]."""
    n = int(seconds * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


def _add_audio_stream(container, sample_rate: int, codec: str):
    """Add an audio stream to `container` without writing any frames yet.

    Streams must be added before the first packet of *any* stream is muxed —
    muxing writes the container header, and adding a stream afterward makes
    PyAV raise "Cannot rebase to zero time." So callers that mux more than one
    stream (see `av_video`) must add every stream up front, before writing.
    """
    return container.add_stream(codec, rate=sample_rate, layout="mono")


def _write_audio(container, stream, samples: torch.Tensor, sample_rate: int):
    """Encode (1, N) float samples into an already-added audio stream."""
    resampler = av.AudioResampler(
        format=stream.codec_context.format.name, layout="mono", rate=sample_rate,
    )
    data = samples.numpy().astype(np.float32)
    pts = 0
    for start in range(0, data.shape[1], 1024):
        block = np.ascontiguousarray(data[:, start : start + 1024])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = pts
        pts += block.shape[1]
        for resampled in resampler.resample(frame):
            for packet in stream.encode(resampled):
                container.mux(packet)
    for resampled in resampler.resample(None):
        for packet in stream.encode(resampled):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


def _add_video_stream(container, w: int, h: int, fps: float):
    """Add a libx264 video stream to `container` without writing frames yet.

    See `_add_audio_stream` for why stream creation is split from writing.
    """
    # Exact rational, not int(round(...)) — see clapsync.app.encode.encode_clip
    # for why rounding a fractional fps (e.g. NTSC 59.94) is wrong. Harmless
    # at the fps=30.0 these fixtures use today, but kept consistent.
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(65535))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    return stream


def _write_video(container, stream, frames: torch.Tensor):
    """Encode (N, 3, H, W) uint8 frames into an already-added video stream."""
    for i in range(frames.shape[0]):
        rgb = np.ascontiguousarray(frames[i].permute(1, 2, 0).numpy())
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Factory: write a mono tone WAV, return (path, sample_rate)."""

    def _make(seconds: float = 1.0, sample_rate: int = 48000,
              freq: float = 440.0, name: str = "tone.wav") -> tuple[Path, int]:
        path = tmp_path / name
        with av.open(str(path), mode="w") as container:
            stream = _add_audio_stream(container, sample_rate, "pcm_s16le")
            _write_audio(container, stream, _tone(seconds, sample_rate, freq),
                         sample_rate)
        return path, sample_rate

    return _make


@pytest.fixture
def rgb_video(tmp_path: Path):
    """Factory: write a solid-color video, return (path, fps, n, w, h)."""

    def _make(seconds: float = 1.0, fps: float = 30.0,
              w: int = 64, h: int = 48, name: str = "vid.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 0] = 255  # solid red
        with av.open(str(path), mode="w") as container:
            stream = _add_video_stream(container, w, h, fps)
            _write_video(container, stream, frames)
        return path, fps, n, w, h

    return _make


@pytest.fixture
def indexed_video(tmp_path: Path):
    """Factory: write a video whose frames are distinguishable by content.

    Frame i is a solid gray level of i * _INDEXED_STEP, so a decoded frame's
    mean brightness recovers which source frame it is — unlike a solid-color
    fixture (e.g. `rgb_video`), where every frame compares equal and ordering
    bugs (a broken searchsorted/sort/inverse-permutation) go undetected. Kept
    short (n <= 12 by default) so i * _INDEXED_STEP stays under 256 and
    distinct per frame.
    """

    def _make(n: int = 8, fps: float = 10.0, w: int = 64, h: int = 48,
              name: str = "idx.mp4"):
        assert n * _INDEXED_STEP <= 255, "index encoding would wrap"
        path = tmp_path / name
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        for i in range(n):
            frames[i] = i * _INDEXED_STEP
        with av.open(str(path), mode="w") as container:
            stream = _add_video_stream(container, w, h, fps)
            _write_video(container, stream, frames)
        return path, fps, n, w, h

    return _make


# Gray level per frame for `indexed_video`; a decoded frame's mean brightness
# divided by this recovers its source frame index (see that fixture).
_INDEXED_STEP = 20


def frame_index(frame: torch.Tensor) -> int:
    """Recover the source index of a frame produced by `indexed_video`."""
    return int(round(float(frame.float().mean()) / _INDEXED_STEP))


@pytest.fixture
def vfr_video_with_offset(tmp_path: Path):
    """Factory: write a VFR video with a nonzero stream start_time.

    Frame i is a solid gray level of i * _INDEXED_STEP (see `indexed_video`),
    so decoded frames are identifiable by content. Unlike `indexed_video`
    (CFR), this fixture assigns each frame an explicit pts on an explicit
    time_base: the first half is spaced uniformly (10 fps) and the second
    half uses growing, non-uniform deltas, so PyAV's guessed frame rate
    diverges from the declared average rate and
    `extract_video_metadata(...).is_vfr` is True. All pts are offset by
    `start_seconds`, so `pts[0] != 0` — the combination (VFR + nonzero
    start_time) is what makes framepipe's metadata.pts stream-absolute
    instead of 0-based (see framepipe.metadata.extract_video_metadata), which
    is the case decode_frames_at must normalize.
    """

    def _make(n: int = 8, w: int = 64, h: int = 48, name: str = "vfr_off.mp4",
              start_seconds: float = 5.0):
        assert n * _INDEXED_STEP <= 255, "index encoding would wrap"
        assert n >= 4, "need at least two frames per half"
        path = tmp_path / name
        time_base = Fraction(1, 90000)
        start_pts = round(start_seconds / time_base)
        half = n // 2
        # First half: uniform 0.1 s spacing (nominal 10 fps). Second half:
        # growing deltas, so the actual frame timing is not periodic.
        deltas = [9000] * (half - 1)
        deltas += [9000 + 4500 * (k + 1) for k in range(n - half)]
        pts_ticks = [start_pts]
        for d in deltas:
            pts_ticks.append(pts_ticks[-1] + d)

        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        for i in range(n):
            frames[i] = i * _INDEXED_STEP

        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=Fraction(10))
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            for i in range(n):
                rgb = np.ascontiguousarray(frames[i].permute(1, 2, 0).numpy())
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = pts_ticks[i]
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        return path, n, w, h, [t * float(time_base) for t in pts_ticks]

    return _make


@pytest.fixture
def av_video(tmp_path: Path):
    """Factory: write a muxed audio+video file, return (path, fps, n, w, h, sr)."""

    def _make(seconds: float = 1.0, fps: float = 30.0, w: int = 64, h: int = 48,
              sample_rate: int = 48000, freq: float = 440.0, name: str = "av.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 2] = 200  # solid blue
        with av.open(str(path), mode="w") as container:
            # Both streams must be added before either is written — see
            # `_add_audio_stream`.
            vstream = _add_video_stream(container, w, h, fps)
            astream = _add_audio_stream(container, sample_rate, "aac")
            _write_video(container, vstream, frames)
            _write_audio(container, astream, _tone(seconds, sample_rate, freq),
                         sample_rate)
        return path, fps, n, w, h, sample_rate

    return _make
