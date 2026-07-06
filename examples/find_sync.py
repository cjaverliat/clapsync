"""Find sync offsets and the common/full time range for a set of clips.

Works for audio-only files or videos (the audio track drives alignment). Every
input must have an audio stream — audio-based sync needs audio.

Usage:
    pixi run python examples/find_sync.py clip_a.mp4 clip_b.mp4 clip_c.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

from clapsync.core import (
    common_time_range,
    compute_sync_offsets,
    full_time_range,
    probe,
)


def main(paths: list[str]) -> None:
    media = [probe(Path(p)) for p in paths]
    offsets = compute_sync_offsets(media)

    for info, offset in zip(media, offsets):
        print(f"{info.path.name:30s} offset = {offset:+.4f} s  ({info.kind})")

    durations = [m.duration for m in media]
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
