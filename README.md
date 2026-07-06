# kineo-sync

Multi-camera audio-based synchronization and trimming tool. Aligns several videos
recorded simultaneously by different cameras (via MFCC audio fingerprinting), lets you
fine-tune the offsets manually, and exports trimmed, synchronized clips ready for
`kineo-reconstruct`.

Extracted as a standalone project from [kineo](../kineo-pro) (`kineo.sync_trim`).

## Environment: pixi

Run everything through [pixi](https://pixi.sh) — no bare `python`/`pip`/`conda`.

```bash
pixi install            # create the environment
pixi run kineo-sync     # launch the app
```

## Usage

`pixi run kineo-sync` opens the video-selection dialog. See
[docs/kineo-sync/kineo-sync.md](docs/kineo-sync/kineo-sync.md) for the full user manual.

## Layout

```
src/kineo_sync/
  app.py                 # entry point (kineo-sync)
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
pixi run build-kineo-sync
```
