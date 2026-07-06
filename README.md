# 👏 clapsync

**Sync and trim multi-camera footage by its audio.**

Point clapsync at several clips of the same moment — filmed by different cameras
or recorders that started at different times — and it aligns them by listening to
their audio (MFCC fingerprinting with sub-frame refinement), then exports trimmed,
perfectly synchronized copies. Use it from Python, the command line, or the GUI.

Great for multicam shoots, interviews, concerts, and anything you clapped a
slate in front of. 👏

---

## Install

clapsync uses [pixi](https://pixi.sh) for its environment (Python, torch,
torchcodec, ffmpeg libraries — all handled for you):

```bash
pixi install
```

## Quick start (Python)

```python
from clapsync.core import sync_and_trim

# Probe, align by audio, trim to the common overlap, and export synced clips.
results = sync_and_trim(
    ["cam_a.mp4", "cam_b.mp4", "cam_c.mp4"],
    output_dir="synced/",
)
for r in results:
    print("ok" if r.ok else r.error, r.path)
```

Every input needs an audio track — that's what the alignment listens to.

## Command line

```bash
# Print each clip's offset and the common / full time range
pixi run clapsync-cli sync cam_a.mp4 cam_b.mp4 cam_c.mp4

# Sync and export trimmed, aligned copies into out/
pixi run clapsync-cli synctrim -o out/ cam_a.mp4 cam_b.mp4 cam_c.mp4
```

## GUI

```bash
pixi run clapsync
```

Opens a picker, computes offsets, then lets you preview, fine-tune the alignment,
set the trim range, and export. See
[docs/clapsync/clapsync.md](docs/clapsync/clapsync.md) for the full manual.

## The API

Everything the GUI and CLI do is available headless from `clapsync.core`:

| Function | What it does |
|---|---|
| `probe(path)` | Read a clip's metadata (`MediaInfo`: duration, fps, audio, …). |
| `compute_sync_offsets(media)` | Per-clip offset in seconds on a shared timeline. |
| `common_time_range` / `full_time_range` | The overlap / union of all clips. |
| `export_tracks(media, offsets, settings)` | Trim + export with full control (resolution, fps, codec). |
| `sync_and_trim(paths, out)` | The one-call convenience: probe → sync → trim → export. |

Offsets are sub-frame accurate and kept as floating-point seconds end to end, so
audio and video stay locked even when the true offset falls between frames.

Runnable examples in [`examples/`](examples/):

- [`find_sync.py`](examples/find_sync.py) — offsets + common/full range
- [`auto_sync_trim.py`](examples/auto_sync_trim.py) — one-call sync & export
- [`export_custom.py`](examples/export_custom.py) — manual pipeline, custom size/fps

```bash
pixi run python examples/auto_sync_trim.py -o synced/ cam_a.mp4 cam_b.mp4
```

## How it works

1. **Listen** — decode each clip's audio and cross-correlate it against a
   reference (MFCC by default; robust to different mics and gain).
2. **Refine** — parabolic peak interpolation pins the offset below one frame.
3. **Range** — compute where all clips overlap (or their full span).
4. **Export** — trim each clip to that range, pad gaps, and re-encode aligned.
   GPU (NVENC) is used automatically when available, with a CPU fallback.

Media decode/encode runs on [torchcodec](https://github.com/meta-pytorch/torchcodec)
(GPU-accelerated); no `ffmpeg` command-line calls.

## Requirements

- A CUDA-capable GPU is recommended (torchcodec + torch on CUDA 13.0 / cu130).
- torch and torchcodec must share the same CUDA build; torchcodec ≥ 0.13 ships on
  cu126 and cu130. pixi pins a working combination for you.

## Build a standalone binary

```bash
pixi run build-clapsync
```

## Layout

```
src/clapsync/
  core/    # headless API: probe, sync, time ranges, export (Qt-free)
  io/      # torchcodec decode / encode wrappers
  gui/     # PySide6 app (thin wrapper around core)
  cli.py   # sync / synctrim commands
examples/  # runnable API examples
```
