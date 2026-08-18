"""Find sync offsets and the common/full time range from audio waveforms.

The caller loads the audio and passes tensors to the pure core: clapsync.core
does no file I/O. This example reads PCM WAV with the standard library so it
depends on nothing but numpy and torch — any decoder works (soundfile, PyAV,
torchaudio with torchcodec installed), the core only needs (channels, samples)
tensors and their sample rates.

Usage:
    pixi run python examples/find_sync.py clip_a.wav clip_b.wav clip_c.wav
"""
from __future__ import annotations

import sys
import wave

import numpy as np
import torch

from clapsync.core import align_waveforms, common_time_range, full_time_range

_DTYPES = {1: np.uint8, 2: np.int16, 4: np.int32}


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    """Read a PCM WAV file.

    Args:
        path: Path to a PCM (uncompressed) .wav file.

    Returns:
        (waveform, sample_rate); waveform is (channels, samples) float32 in
        [-1, 1].

    Raises:
        ValueError: If the file is not 8/16/32-bit PCM.
    """
    with wave.open(path, "rb") as handle:
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    dtype = _DTYPES.get(width)
    if dtype is None:
        raise ValueError(f"{path}: unsupported sample width {width * 8}-bit")

    data = np.frombuffer(raw, dtype=dtype).reshape(-1, channels).T
    if dtype is np.uint8:  # 8-bit PCM is unsigned, centred on 128
        samples = (data.astype(np.float32) - 128.0) / 128.0
    else:
        samples = data.astype(np.float32) / float(np.iinfo(dtype).max)
    return torch.from_numpy(samples), rate


def main(paths: list[str]) -> None:
    waveforms = []
    rates = []
    durations = []
    for path in paths:
        wave_tensor, rate = load_wav(path)
        waveforms.append(wave_tensor)
        rates.append(rate)
        durations.append(wave_tensor.shape[-1] / rate)

    alignment = align_waveforms(waveforms, rates)
    offsets = alignment.offsets

    for path, offset, conf in zip(paths, offsets, alignment.confidence):
        conf_str = "ref" if conf == float("inf") else f"{conf:.1f}"
        print(f"{path:30s} offset = {offset:+.4f} s  confidence = {conf_str}")
    for warning in alignment.warnings:
        print(f"  ! {warning}")

    common = common_time_range(durations, offsets)
    full = full_time_range(durations, offsets)
    print(
        f"\ncommon overlap : {common.start:.3f} -> {common.end:.3f} s "
        f"({common.duration:.3f} s)"
    )
    print(
        f"full timeline  : {full.start:.3f} -> {full.end:.3f} s "
        f"({full.duration:.3f} s)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: find_sync.py <clip1.wav> <clip2.wav> [clip3.wav ...]")
    main(sys.argv[1:])
