"""Muxed audio+video encoding via PyAV."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

# Audio codec per container; PyAV needs it named explicitly. Verified
# round-trips: pcm_s16le/libmp3lame/flac are sample-exact, aac pads +128.
_AUDIO_CODEC_FOR_SUFFIX = {
    ".wav": "pcm_s16le",
    ".m4a": "aac",
    ".mp4": "aac",
    ".mkv": "aac",
    ".mov": "aac",
    ".mp3": "libmp3lame",
    ".flac": "flac",
}
_DEFAULT_AUDIO_CODEC = "aac"

# Frames are handed to the encoder in blocks of this many samples.
_AUDIO_BLOCK = 1024


def pick_video_codec(device: str) -> str:
    """Return the H.264 encoder name for a device.

    Args:
        device: Torch device string, e.g. "cpu", "cuda", "cuda:0".

    Returns:
        "h264_nvenc" for CUDA devices, "libx264" otherwise.
    """
    return "h264_nvenc" if str(device).startswith("cuda") else "libx264"


def _audio_codec_for(out_path: Path) -> str:
    return _AUDIO_CODEC_FOR_SUFFIX.get(out_path.suffix.lower(), _DEFAULT_AUDIO_CODEC)


def _quality_options(codec: str, crf: int) -> dict[str, str]:
    """NVENC has no crf; its constant-quality knob is cq."""
    return {"cq": str(crf)} if codec == "h264_nvenc" else {"crf": str(crf)}


def _mux_audio_samples(
    container: av.container.OutputContainer,
    astream: av.stream.Stream,
    layout: str,
    samples: torch.Tensor,
    rate: int,
) -> None:
    """Resample, encode and mux (C, N) float samples into ``astream``.

    Flushes both the resampler and the encoder, so the stream is complete
    once this returns. ``astream`` must already be added to ``container``.
    """
    resampler = av.AudioResampler(
        format=astream.codec_context.format.name, layout=layout, rate=rate
    )
    data = samples.cpu().numpy().astype(np.float32)
    pts = 0
    for start in range(0, data.shape[1], _AUDIO_BLOCK):
        block = np.ascontiguousarray(data[:, start : start + _AUDIO_BLOCK])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = pts
        pts += block.shape[1]
        for resampled in resampler.resample(frame):
            for packet in astream.encode(resampled):
                container.mux(packet)
    for resampled in resampler.resample(None):  # flush resampler
        for packet in astream.encode(resampled):
            container.mux(packet)
    for packet in astream.encode():  # flush encoder
        container.mux(packet)


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
        crf: Constant quality for the video stream (mapped to cq on NVENC).
        device: "cpu" or "cuda[:<index>]"; selects the default encoder. Does
            not move the frame tensor — h264_nvenc uploads from host memory
            itself.

    Raises:
        ValueError: If required companions (fps/sample_rate) are missing, or
            if neither stream is provided.
    """
    if video_frames is None and audio_samples is None:
        raise ValueError("encode_clip needs at least one stream")
    if video_frames is not None and video_fps is None:
        raise ValueError("video_fps is required when video_frames is given")
    if audio_samples is not None and sample_rate is None:
        raise ValueError("sample_rate is required when audio_samples is given")

    out_path = Path(out_path)
    with av.open(str(out_path), mode="w") as container:
        vstream = None
        astream = None
        layout = None

        if video_frames is not None:
            if video_frames.shape[0] == 0:
                raise ValueError("encode_clip got no video frames")
            _, _channels, height, width = video_frames.shape
            codec = video_codec or pick_video_codec(device)
            # Exact rational, not int(round(...)): rounding a GoPro's 59.94
            # (60000/1001) up to 60 makes the container declare a rate the
            # frames were not encoded at, desyncing video from its own audio
            # (0.218s drift on a 218s clip). PyAV accepts a Fraction directly.
            rate = Fraction(video_fps).limit_denominator(65535)
            vstream = container.add_stream(codec, rate=rate)
            vstream.width = width
            vstream.height = height
            vstream.pix_fmt = "yuv420p"
            vstream.options = _quality_options(codec, crf)

        if audio_samples is not None:
            layout = "mono" if audio_samples.shape[0] == 1 else "stereo"
            acodec = _audio_codec_for(out_path)
            astream = container.add_stream(acodec, rate=sample_rate, layout=layout)

        if vstream is not None:
            frames = video_frames.cpu()
            for i in range(frames.shape[0]):
                rgb = frames[i].permute(1, 2, 0).numpy()  # CHW -> HWC
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(rgb), format="rgb24",
                )
                for packet in vstream.encode(frame):
                    container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)

        if astream is not None:
            _mux_audio_samples(
                container, astream, layout, audio_samples, sample_rate,
            )
