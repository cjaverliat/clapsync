"""One call: sync a set of clips and export trimmed, aligned copies.

`sync_and_trim` probes the inputs, computes audio offsets, picks a trim range
(the common overlap by default), and writes one synced clip per input. Video
inputs produce muxed A/V MP4s; audio-only inputs produce audio files.

Usage:
    pixi run python examples/auto_sync_trim.py -o synced/ cam_a.mp4 cam_b.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clapsync.app import sync_and_trim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Clips to sync")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--trim", choices=["common", "full"], default="common")
    args = parser.parse_args()

    results = sync_and_trim(
        args.inputs,
        args.output,
        trim=args.trim,
        progress=lambda f: print(f"\r{f * 100:5.1f}%", end="", flush=True),
    )
    print()
    for result in results:
        status = "ok " if result.ok else f"ERR {result.error}"
        print(f"{status}\t{result.path}")


if __name__ == "__main__":
    main()
