# 👏 clapsync

**Auto synchronize and trim video, audio and motion capture recordings.**

**clapsync** allows synchronizing multiple audio/video tracks automatically using audio cues, and exports trimmed copies that start at the same
instant and have the same duration. Optionally, **clapsync** can also synchronize motion capture files (`.c3d`), as long as it contains a clapperboard. To bridge between the audio-less motion capture file and the audio, it detects when the two arms of the clapperboard meet, and synchronize it with the clapperboard sound from the audio tracks.

Clapsync can be used via a GUI, from the command line, or as a headless python
library:

- **[The app](#the-app-gui)**: a desktop GUI, with a Windows installer.
- **[The Python library](#the-python-library)**: the same engine, importable.
- **[The command line](#command-line)**: `clapsync-cli sync` / `synctrim`, for
  scripts and batch runs.

---

## The app (GUI)

### Install on Windows

Download `clapsync-setup-<version>.exe` from the
[**releases page**](https://github.com/cjaverliat/clapsync/releases/latest) and
run it.

It installs into `%LOCALAPPDATA%\clapsync` for your user only, so no admin
rights are needed. The setup is a small online installer: it downloads the
Python/CUDA environment as it runs (~2 GB download, ~8 GB on disk), so keep
an internet connection during install.

clapsync uses NVIDIA GPU if it finds one to run faster, and falls back to the CPU otherwise.

> The installer is unsigned, so Windows SmartScreen warns about an
> *"unrecognized app"* on first run. Choose **More info → Run anyway**.

### Run from source (any platform)

```bash
pixi install
pixi run clapsync
```

### What you do in the app

1. **Pick your files.** Videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.mts`,
   `.m2ts`, `.webm`, `.flv`, `.wmv`), audio (`.wav`, `.mp3`, `.flac`, `.m4a`,
   `.aac`) and motion capture (`.c3d`). The first file is the reference;
   everything else aligns to it.
2. **Let it sync.** clapsync compares the audio tracks and computes every
   offset. Uncertain files are flagged for review.
3. **Check and fix.** A grid preview plays every track together, so bad
   alignment shows at once. Drag the playhead, zoom the timeline, or
   double-click a track to type an exact offset.
4. **Set the trim range** using the two handles and **export**. Every file comes out aligned, padded
   where a recorder wasn't rolling yet, and identical in duration.

### Motion capture (`.c3d`)

Add `.c3d` files next to your videos. clapsync places them on the shared
timeline for you:

- It finds the **clap sound** in the camera and recorder audio: the loudest
  sharp, broadband transient, with speech, music and handling noise filtered
  out. Tracks are cross-checked, so the real slate wins.
- It finds the **clapperboard closing** in the marker data: the gap between top
  and bottom arm markers collapsing at peak closing speed.
- Matching the two gives the mocap offset, accurate below one frame.

Markers are recognized by name when their labels hold a clapperboard keyword
(`clap` or `slate`) plus a direction (`top`/`upper`/`up` vs
`bottom`/`lower`/`low`/`down`). Otherwise the app asks you to pick the top and
bottom markers yourself.

On export, a `.c3d` yields a `<name>_trim.txt` beside the synced clips, holding
the trim's `start_frame` and `end_frame` **in the c3d's own numbering**. Cut the
take in your mocap tool; clapsync never rewrites your file. Frames outside the
capture are reported as they fall, possibly negative, so padding stays visible
instead of being clamped away.

If no clap is found, the track stays at offset 0 and you get a warning. clapsync
never guesses.

---

## The Python library

### Install

```bash
pip install clapsync          # pure core: numpy + torch + torchaudio
pip install clapsync[app]     # + media I/O and c3d (PyAV, py-c3d)
pip install clapsync[gui]     # + the desktop app (PySide6)
```

For development, [pixi](https://pixi.sh) handles the whole environment (Python,
torch/CUDA, PyAV, ffmpeg libraries):

```bash
pixi install
```

### Quick start

```python
from pathlib import Path

from clapsync.app import sync_and_trim

# Probe, align by audio, trim to the common overlap, and export synced clips.
results = sync_and_trim(
    [Path("cam_a.mp4"), Path("cam_b.mp4"), Path("cam_c.mp4")],
    output_dir=Path("synced/"),
)
for r in results:
    print("ok" if r.ok else r.error, r.path)
```

Every audio/video input needs an audio track. That's what the alignment listens
to.

### Sync video and motion capture together

```python
from pathlib import Path

from clapsync.app import compute_sync_offsets

alignment = compute_sync_offsets(
    [Path("cam_a.mp4"), Path("cam_b.mp4"), Path("take_01.c3d")],
)
print(alignment.offsets)     # seconds, on the shared timeline
print(alignment.confidence)  # per track; low values want a manual check
print(alignment.warnings)    # e.g. "no clapperboard motion detected"
```

The c3d is placed by its clapperboard. For every track,
`shared_time = local_time + offset`.

### Pure core (`clapsync.core`), no media libraries required

| Function / class | What it does |
|---|---|
| `align_waveforms(waveforms, rates)` | Solve all-pairs offsets via consistency-weighted MST; returns an `Alignment` (offsets, per-track confidence, warnings). |
| `find_offset(ref, target, rate)` | Single-pair offset with sub-frame parabolic refinement. |
| `TimeRange(start, end)` | Value type for a half-open time interval. |
| `common_time_range(durations, offsets)` | The overlap where all clips are present. |
| `full_time_range(durations, offsets)` | The union spanning every clip. |

Clap detection (`clapsync.core.clap`) is pure too: arrays in, results out.

| Function | What it does |
|---|---|
| `detect_clap_sound(wave, rate)` | Rank clap-sound candidates in a waveform, gating out non-claps. |
| `detect_clap_motions(top, bottom, rate)` | Rank clapperboard snaps from top/bottom marker centroids. |
| `classify_clap_markers(labels)` | Split marker labels into (top, bottom) groups by name. |
| `clapperboard_reliability(...)` | Per-frame mask rejecting badly-tracked clapperboard geometry. |

### App layer (`clapsync.app`), requires `clapsync[app]`

| Function / class | What it does |
|---|---|
| `load_audio(path)` | Decode a file's audio track to a tensor via PyAV. |
| `compute_sync_offsets(paths)` | Probe + align audio/video *and* c3d; returns an `Alignment`. |
| `export_tracks(paths, offsets, settings)` | Trim + export with full control (resolution, fps, codec). |
| `sync_and_trim(paths, out)` | One-call convenience: probe → sync → trim → export. |
| `ExportSettings` / `ExportResult` | Configuration and result types for export. |
| `mocap.load_c3d(path)` / `write_c3d(path, data)` | Read/write c3d markers and analog data as a plain `MocapData`. |

`probe(path)` (in `clapsync.app.media`) reads a clip's metadata (`MediaInfo`:
duration, fps, …). Internal, but available if you need it.

Offsets stay floating-point seconds end to end, so audio and video keep their
sub-frame alignment even when the true offset falls between frames.

### Examples

Runnable scripts in [`examples/`](examples/):

- [`find_sync.py`](examples/find_sync.py): pure, torchaudio + `clapsync.core` only, no media libraries
- [`auto_sync_trim.py`](examples/auto_sync_trim.py): one-call sync & export
- [`export_custom.py`](examples/export_custom.py): manual pipeline, custom size/fps

```bash
pixi run python examples/auto_sync_trim.py -o synced/ cam_a.mp4 cam_b.mp4
```

---

## Command line

```bash
# Print each track's offset and the common / full time range
pixi run clapsync-cli sync cam_a.mp4 cam_b.mp4 mic.wav

# Sync and export trimmed, aligned copies into out/
pixi run clapsync-cli synctrim -o out/ cam_a.mp4 cam_b.mp4 mic.wav

# Motion capture works here too: the c3d is bridged by the clapperboard
pixi run clapsync-cli synctrim -o out/ cam_a.mp4 mic.wav take_01.c3d
```

Options for both subcommands:

| Flag | What it does |
|---|---|
| `--reference N` | Index of the reference track (default `0`, the first input). |
| `--refine {none,parabolic}` | Sub-frame peak refinement (default `parabolic`). |
| `--trim {common,full}` | `synctrim` only: export the overlap or the full span (default `common`). |
| `-o, --output DIR` | `synctrim` only, required: output directory. |
| `-v, --verbose` | Debug logging. Goes before the subcommand. |

The c3d's clapperboard markers must be identifiable by name here, since the CLI
has no marker picker. If they aren't, that track is left at offset 0 with a
warning.