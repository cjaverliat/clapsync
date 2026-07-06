# clapsync

Multi-camera audio-based synchronization and trimming tool. Aligns several videos
recorded simultaneously by different cameras (via MFCC audio fingerprinting), lets you
fine-tune the offsets manually, and exports trimmed, synchronized clips.

## Environment: pixi

Run everything through [pixi](https://pixi.sh) — no bare `python`/`pip`/`conda`.

```bash
pixi install         # create the environment
pixi run clapsync    # launch the app
```

## Usage

`pixi run clapsync` opens the video-selection dialog. See
[docs/clapsync/clapsync.md](docs/clapsync/clapsync.md) for the full user manual.

## Layout

```
src/clapsync/
  app.py                 # entry point (clapsync)
  sync_editor.py         # main editor window
  offset_worker.py       # audio-offset computation (background)
  export_dialog.py       # export settings + worker
  audio_sync.py          # MFCC-based offset finder (torch/torchaudio)
  gui/                   # PySide6 widgets (timeline, video player, selection)
  io/                    # video/audio/ffmpeg I/O helpers
```

## Dependencies

PySide6, PyAV, numpy, torch + torchaudio (cu128), and an `ffmpeg` binary (provided by pixi).

## Build a standalone binary

```bash
pixi run build-clapsync
```
