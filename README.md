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
framepipe, PyAV, ffmpeg libraries — all handled for you):

```bash
pixi install
```

## Quick start (Python)

```python
from clapsync.app import sync_and_trim

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

### Pure core (`clapsync.core`) — no media libraries required

Install with `pip install clapsync` (numpy + torch + torchaudio only).

| Function / class | What it does |
|---|---|
| `align_waveforms(waveforms, rates)` | Per-clip offset in seconds from raw audio tensors. |
| `find_offset(ref, target, rate)` | Single-pair offset with sub-frame parabolic refinement. |
| `TimeRange(start, end)` | Value type for a half-open time interval. |
| `common_time_range(durations, offsets)` | The overlap where all clips are present. |
| `full_time_range(durations, offsets)` | The union spanning every clip. |

### App layer (`clapsync.app`) — requires `clapsync[app]` (framepipe + PyAV)

| Function / class | What it does |
|---|---|
| `load_audio(path)` | Decode a file's audio track to a tensor via PyAV. |
| `compute_sync_offsets(paths)` | Probe + load + align; returns per-path offsets (seconds). |
| `export_tracks(paths, offsets, settings)` | Trim + export with full control (resolution, fps, codec). |
| `sync_and_trim(paths, out)` | One-call convenience: probe → sync → trim → export. |
| `ExportSettings` / `ExportResult` | Configuration and result types for export. |

`probe(path)` (in `clapsync.app.media`) reads a clip's metadata (`MediaInfo`:
duration, fps, …) — internal, but available if you need it.

Offsets are sub-frame accurate and kept as floating-point seconds end to end, so
audio and video stay locked even when the true offset falls between frames.

Runnable examples in [`examples/`](examples/):

- [`find_sync.py`](examples/find_sync.py) — pure: torchaudio + `clapsync.core` only (no media libraries)
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

Media decode/encode runs on [framepipe](https://github.com/cjaverliat/framepipe)
for video decode and [PyAV](https://github.com/PyAV-Org/PyAV) for audio and
muxing (both GPU-accelerated where applicable); no `ffmpeg` command-line calls.

## Requirements

- A CUDA-capable GPU is recommended — framepipe's `device="cuda:0"` decode and
  the `h264_nvenc` export path both use it.

## Build a standalone binary

```bash
pixi run build-clapsync
```

## Build the Windows installer

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`) and a sibling `../framepipe`
checkout.

```bash
pixi run build-installer
```

Produces `outputs/clapsync-setup-<version>.exe` — a small online installer.
It installs per-user to `%LOCALAPPDATA%\clapsync`, then downloads the locked
Python/CUDA environment (~2 GB download, ~8 GB on disk) at install time, so
internet is required during install. No NVIDIA GPU is needed to install;
without one, clapsync runs on the CPU (slower sync and export).

The installer is unsigned, so Windows SmartScreen shows an "unrecognized
app" warning on first run — choose **More info → Run anyway**.

## Layout

```
src/clapsync/
  core/    # pure: MFCC align, time-range math (numpy/torch/torchaudio only)
  app/     # file I/O, probe, framepipe/PyAV decode/encode, export, sync_and_trim
  gui/     # PySide6 app (thin wrapper around app layer)
  cli.py   # sync / synctrim commands
examples/  # runnable API examples
```
