"""Headless CLI for clapsync: sync and synctrim subcommands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clapsync.app import compute_sync_offsets, sync_and_trim
from clapsync.app.media import probe
from clapsync.core import common_time_range, full_time_range


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", type=Path, help="Media files")
    p.add_argument("--refine", choices=["none", "parabolic"], default="parabolic")
    p.add_argument("--reference", type=int, default=0, help="Reference index")


def _cmd_sync(args: argparse.Namespace) -> int:
    offsets = compute_sync_offsets(
        args.inputs,
        reference_index=args.reference,
        refine=args.refine,
        progress=lambda f: print(
            f"\rsync {f * 100:3.0f}%", end="", file=sys.stderr
        ),
    )
    print(file=sys.stderr)
    durations = [probe(p).duration for p in args.inputs]
    common = common_time_range(durations, offsets)
    full = full_time_range(durations, offsets)
    for path, off in zip(args.inputs, offsets):
        print(f"{path.name}\toffset={off:+.4f}s")
    print(
        f"common range: {common.start:.4f}..{common.end:.4f}s"
        f" ({common.duration:.4f}s)"
    )
    print(
        f"full range:   {full.start:.4f}..{full.end:.4f}s"
        f" ({full.duration:.4f}s)"
    )
    return 0


def _cmd_synctrim(args: argparse.Namespace) -> int:
    results = sync_and_trim(
        args.inputs,
        args.output,
        refine=args.refine,
        trim=args.trim,
        reference_index=args.reference,
        progress=lambda f: print(f"\r{f * 100:3.0f}%", end="", file=sys.stderr),
    )
    print(file=sys.stderr)
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK " if r.ok else f"ERR {r.error}"
        print(f"{status}\t{r.path}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Run the clapsync CLI and return an exit code.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        0 on success, 1 if any export failed, 2 if no subcommand given.
    """
    parser = argparse.ArgumentParser(prog="clapsync-cli")
    sub = parser.add_subparsers(dest="command")

    p_sync = sub.add_parser("sync", help="Print offsets and time ranges")
    _add_common(p_sync)
    p_sync.set_defaults(func=_cmd_sync)

    p_trim = sub.add_parser("synctrim", help="Sync and export trimmed clips")
    _add_common(p_trim)
    p_trim.add_argument("-o", "--output", type=Path, required=True)
    p_trim.add_argument("--trim", choices=["common", "full"], default="common")
    p_trim.set_defaults(func=_cmd_synctrim)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
