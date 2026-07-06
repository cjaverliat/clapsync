"""Manual pipeline: probe -> sync -> pick range -> export at a custom size/fps.

Shows the individual steps that `sync_and_trim` wraps, so you can customize the
trim range, output resolution, frame rate, or codec. Here we export the common
overlap at a fixed resolution and frame rate.

Usage:
    pixi run python examples/export_custom.py -o out/ --width 1280 --height 720 \
        cam_a.mp4 cam_b.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clapsync.app import ExportSettings, compute_sync_offsets, export_tracks
from clapsync.app.media import probe
from clapsync.core import common_time_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = args.inputs
    offsets = compute_sync_offsets(paths)
    durations = [probe(p).duration for p in paths]
    trim = common_time_range(durations, offsets)

    settings = ExportSettings(
        trim=trim,
        output_dir=args.output,
        target_width=args.width,
        target_height=args.height,
        output_fps=args.fps,
    )
    results = export_tracks(
        paths,
        offsets,
        settings,
        progress=lambda f: print(f"\r{f * 100:5.1f}%", end="", flush=True),
    )
    print()
    for result in results:
        status = "ok " if result.ok else f"ERR {result.error}"
        print(f"{status}\t{result.path}")


if __name__ == "__main__":
    main()
