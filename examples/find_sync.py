"""Find sync offsets and the common/full time range from audio waveforms.

The caller loads audio (here with torchaudio) and passes tensors to the pure
core — clapsync.core does no file I/O. Pass audio files; torchaudio.load can
also read a video container's audio track, but that depends on its ffmpeg
backend, so extract audio first if in doubt.

Usage:
    pixi run python examples/find_sync.py clip_a.wav clip_b.wav clip_c.wav
"""
from __future__ import annotations

import sys

import torchaudio

from clapsync.core import align_waveforms, common_time_range, full_time_range


def main(paths: list[str]) -> None:
    waveforms = []
    rates = []
    durations = []
    for path in paths:
        wave, rate = torchaudio.load(path)
        waveforms.append(wave)
        rates.append(rate)
        durations.append(wave.shape[-1] / rate)

    offsets = align_waveforms(waveforms, rates)

    for path, offset in zip(paths, offsets):
        print(f"{path:30s} offset = {offset:+.4f} s")

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
        sys.exit("usage: find_sync.py <clip1> <clip2> [clip3 ...]")
    main(sys.argv[1:])
