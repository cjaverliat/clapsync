"""Shared test fixtures.

Synthetic media are generated with raw torchcodec encoders so decode/probe/
export tests have known-content inputs without checking binaries into git.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch


def _tone(seconds: float, sample_rate: int, freq: float) -> torch.Tensor:
    """Mono sine tone, shape (1, N), float in [-1, 1]."""
    n = int(seconds * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Factory: write a mono tone WAV, return (path, sample_rate)."""
    from torchcodec.encoders import AudioEncoder

    def _make(seconds: float = 1.0, sample_rate: int = 48000,
              freq: float = 440.0, name: str = "tone.wav") -> tuple[Path, int]:
        path = tmp_path / name
        AudioEncoder(_tone(seconds, sample_rate, freq),
                     sample_rate=sample_rate).to_file(str(path))
        return path, sample_rate

    return _make


@pytest.fixture
def rgb_video(tmp_path: Path):
    """Factory: write a solid-color video, return (path, fps, n, w, h)."""
    from torchcodec.encoders import VideoEncoder

    def _make(seconds: float = 1.0, fps: float = 30.0,
              w: int = 64, h: int = 48, name: str = "vid.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 0] = 255  # solid red
        VideoEncoder(frames, frame_rate=fps).to_file(str(path), codec="libx264")
        return path, fps, n, w, h

    return _make
