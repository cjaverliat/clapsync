"""Muxed audio+video encoding via the torchcodec multi-stream Encoder."""
from __future__ import annotations

from pathlib import Path

import torch
from torchcodec.encoders import Encoder


def pick_video_codec(device: str) -> str:
    """Return the H.264 encoder name for a device.

    Args:
        device: Torch device string, e.g. "cpu", "cuda", "cuda:0".

    Returns:
        "h264_nvenc" for CUDA devices, "libx264" otherwise.
    """
    return "h264_nvenc" if str(device).startswith("cuda") else "libx264"


def encode_clip(
    out_path: Path,
    video_frames: torch.Tensor | None,
    video_fps: float | None,
    audio_samples: torch.Tensor | None,
    sample_rate: int | None,
    *,
    video_codec: str | None = None,
    crf: int = 18,
    device: str = "cpu",
) -> None:
    """Encode and mux one clip to a single container file.

    At least one of video_frames or audio_samples must be provided.

    Args:
        out_path: Destination path; container format inferred from suffix.
        video_frames: (N, C, H, W) uint8 frames, or None for audio-only.
        video_fps: Output frame rate; required when video_frames is given.
        audio_samples: (C, M) float samples in [-1, 1], or None for
            video-only.
        sample_rate: Audio sample rate in Hz; required when audio_samples
            is given.
        video_codec: Override encoder name; defaults to pick_video_codec(device).
        crf: Constant rate factor for the video stream (quality).
        device: "cpu" or "cuda[:<index>]"; video frames are moved here
            before encoding.

    Raises:
        ValueError: If required companions (fps/sample_rate) are missing, or
            if neither stream is provided.
    """
    if video_frames is None and audio_samples is None:
        raise ValueError("encode_clip needs at least one stream")

    encoder = Encoder()
    vstream = None
    astream = None

    if video_frames is not None:
        if video_fps is None:
            raise ValueError("video_fps is required when video_frames is given")
        _, _channels, height, width = video_frames.shape
        codec = video_codec or pick_video_codec(device)
        vstream = encoder.add_video(
            height=height,
            width=width,
            frame_rate=video_fps,
            device=device,
            codec=codec,
            crf=crf,
        )

    if audio_samples is not None:
        if sample_rate is None:
            raise ValueError(
                "sample_rate is required when audio_samples is given"
            )
        astream = encoder.add_audio(
            sample_rate=sample_rate,
            num_channels=audio_samples.shape[0],
        )

    with encoder.open_file(str(out_path)):
        if vstream is not None:
            frames = video_frames.to(device)
            vstream.add_frames(frames)
        if astream is not None:
            astream.add_samples(audio_samples)
