# clapsync — Project 1: Headless torchcodec core & CLI

Date: 2026-07-06
Status: Approved design (pending spec review)

## Context & decomposition

The overall goal — decouple a Qt-free headless API from the PySide6 GUI and
standardize media I/O on **torchcodec ≥0.13** (dropping the ffmpeg binary,
ffprobe, PyAV, and framepipe) — is split into two projects:

- **Project 1 (this spec):** the headless `core/` API + CLI + torchcodec I/O
  (probe, audio load, decode, and muxed export). Independently testable with no
  GUI. The GUI keeps framepipe **only for its real-time preview** meanwhile, but
  its offset-computation and export paths are rewired to the new core.
- **Project 2 (separate, next):** reimplement the real-time synced mosaic
  playback engine (framepipe `VideoGroupDecoder`) on N torchcodec
  `VideoDecoder`s, rewire GUI workers, and remove framepipe entirely.

torchcodec links the ffmpeg *libraries* internally, so "drop ffmpeg" means
removing the CLI/subprocess and ffprobe dependencies — codecs remain.

## Requirements

The headless API must support, for both audio-only files and video files:

1. Find the audio sync point (per-track offset on a shared timeline).
2. Find the common time range (all tracks overlap) and the full range.
3. Automatically sync **and** trim/export tracks (audio + video, muxed).

## Architecture

### The boundary: plain callbacks

Core functions are Qt-free and cross the boundary with plain callables:

- `progress: Callable[[float], None] | None` — fraction in `[0, 1]`.
- `is_cancelled: Callable[[], bool] | None` — cooperative cancel check.

GUI workers adapt these to Qt signals; the CLI adapts them to a stderr bar. No
Qt import ever appears under `core/`.

### Module layout

```
src/clapsync/
  core/                      # headless, Qt-free — the API
    __init__.py              # re-exports the public surface
    media.py                 # MediaInfo + probe() (torchcodec metadata)
    offsets.py               # pure MFCC/envelope finders + parabolic refine
    sync.py                  # compute_sync_offsets(...)
    timerange.py             # TimeRange, common_time_range(), full_time_range()
    export.py                # ExportSettings, ExportResult, export_tracks(), sync_and_trim()
  io/
    decode.py                # torchcodec AudioDecoder / VideoDecoder wrappers
    encode.py                # torchcodec multi-stream Encoder (muxed A/V)
  gui/                       # unchanged in Project 1 except worker rewiring
    workers.py               # QThread adapters: core callbacks -> Qt signals
    export_dialog.py         # widgets only (worker logic removed)
    sync_editor.py, timeline_widget.py, video_player.py, video_selection_dialog.py
  cli.py                     # argparse headless entry (sync / synctrim)
  app.py                     # GUI entry
```

`io/audio.py` and `io/ffmpeg.py` are removed (replaced by `decode.py` /
`encode.py`). `audio_sync.py` moves to `core/offsets.py`.

## Media I/O (torchcodec ≥0.13)

### Probe (`core/media.py`)

```python
@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    has_audio: bool
    kind: Literal["audio", "video"]
    sample_rate: int | None = None   # audio
    width: int | None = None         # video
    height: int | None = None        # video
    fps: float | None = None         # video

def probe(path: Path) -> MediaInfo:
    """Read stream metadata via torchcodec decoders.

    Tries VideoDecoder(path).metadata (average_fps, width, height,
    duration_seconds); presence of an audio stream and its sample_rate via
    AudioDecoder(path).metadata. Files with no video stream are kind="audio".
    """
```

### Decode (`io/decode.py`)

```python
def load_audio(path, target_rate: int | None = None) -> tuple[torch.Tensor, int]:
    """Mono-mix float32 waveform (1, N) + sample rate via AudioDecoder.
    Peak-normalized, matching current sync behavior. torchcodec can resample
    on decode when target_rate is set."""

def decode_frames_in_range(path, start_s, stop_s, fps=None, device="cpu") -> torch.Tensor:
    """(N, C, H, W) uint8 frames via VideoDecoder.get_frames_played_in_range.
    device='cuda' decodes on GPU (nvdec)."""
```

Progress: torchcodec decodes in memory rather than streaming a subprocess, so
per-track audio-load progress is coarse (0 → 1 per file) instead of sample-level.
Acceptable; the sync loop's granularity stays per-track.

### Export (`io/encode.py` + `core/export.py`)

Encode uses the torchcodec multi-stream `Encoder` to mux one MP4 per track:

```python
enc = Encoder()
vstream = enc.add_video(height=h, width=w, frame_rate=out_fps,
                        device="cuda", codec="h264_nvenc", crf=18)   # or libx264/cpu
astream = enc.add_audio(sample_rate=sr, num_channels=ch)             # if track has audio
with enc.open_file(out_path):
    vstream.add_frames(video_frames)     # (N, C, H, W) uint8, placed on out_fps grid
    astream.add_samples(audio_samples)   # (C, M) float, trimmed + silence-padded
```

CUDA (`h264_nvenc`) when available, else CPU `libx264`. Codec/CRF configurable
via `ExportSettings`.

## Public API (`clapsync.core`)

```python
# offsets.py  (pure — no I/O, no Qt)
Method = Literal["mfcc", "envelope"]
Refine = Literal["none", "parabolic"]

def find_offset(ref_waveform, ref_rate, waveform, rate, fps,
                method: Method = "mfcc", refine: Refine = "parabolic",
                ...) -> tuple[int, float]:
    """(lag_frames, lag_seconds). lag_seconds is subframe-accurate."""

# sync.py
def compute_sync_offsets(
    media: list[MediaInfo], reference_index: int = 0,
    method: Method = "mfcc", refine: Refine = "parabolic",
    progress=None, is_cancelled=None,
) -> list[float]:
    """Per-track offset in seconds on a shared timeline. offset[ref] == 0.
    Float seconds throughout — never quantized to whole frames."""

# timerange.py
@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    @property
    def duration(self) -> float: ...

def common_time_range(media, offsets) -> TimeRange:   # intersection
def full_time_range(media, offsets)   -> TimeRange:   # union

# export.py
@dataclass(frozen=True)
class ExportSettings:
    trim: TimeRange
    output_dir: Path
    target_width: int | None = None      # video only
    target_height: int | None = None     # video only
    output_fps: float | None = None       # video only
    video_codec: str | None = None        # None -> auto (nvenc if CUDA else libx264)
    crf: int = 18
    audio_format: str | None = None       # audio-only tracks; None = pass-through

@dataclass(frozen=True)
class ExportResult:
    path: Path
    error: str | None = None
    @property
    def ok(self) -> bool: return self.error is None

def export_tracks(media, offsets, settings, progress=None, is_cancelled=None) -> list[ExportResult]:
    """Trim+pad each track to settings.trim on the shared timeline.
    Video -> muxed MP4 (A/V), audio-only -> audio file. Subframe-exact."""

def sync_and_trim(paths: list[Path], output_dir: Path, *,
                  method="mfcc", refine="parabolic",
                  trim: Literal["common", "full"] = "common",
                  reference_index: int = 0,
                  progress=None, is_cancelled=None) -> list[ExportResult]:
    """probe -> compute_sync_offsets -> common/full range -> export_tracks."""
```

The six requirements (audio + video × {find sync, find range, auto sync+trim})
map onto the *same* functions; media kind is a field, not a separate code path.

## Subframe correctness

MFCC lag is subframe (~5 ms hop, refined below). It must survive to output:

- Offsets are **float seconds** everywhere. Reference offset is `0.0`.
- `lag_frames` is informational (GUI display) only — export never uses it.
- Per track, the export window on the shared timeline is `settings.trim`. Local
  source range = `trim - offset`; leading gap (`offset - trim.start`, if > 0)
  is padding. Because export is now **tensor-based**, we build the output frame
  sequence by sampling source frames at the exact output-fps timestamps
  (`k / out_fps` shifted by the subframe offset) via
  `get_frames_played_at`/`get_frames_played_in_range`, holding/black-padding at
  boundaries. Audio samples are trimmed and silence-padded to the sample exact
  count. Video and audio therefore share one subframe-accurate origin — perfect
  lock, no frame-grid snapping.

## Parabolic refinement (strict improvement)

Coarse cross-correlation returns an integer-index peak (~5 ms MFCC hop / one
frame for envelope). With `refine="parabolic"` (default), fit a parabola to the
three correlation samples straddling the peak and take its analytic maximum for
sub-index precision. Pure, deterministic, ~free. `refine="none"` reproduces the
old integer-peak behavior.

## Migration (behavior-preserving where behavior is unchanged)

| From | To |
|---|---|
| `audio_sync.py` | `core/offsets.py` (+ parabolic step) |
| `offset_worker.OffsetWorker` compute loop | `core/sync.compute_sync_offsets` |
| `offset_worker.compute_offsets_with_progress` | `gui/workers.py` adapter |
| `export_dialog.ExportWorker.run` | `core/export.export_tracks` |
| `export_dialog.ExportDialog` | `gui/export_dialog.py` (widgets only) |
| `io/audio.py` (ffmpeg subprocess) | `io/decode.py` (torchcodec) |
| `io/ffmpeg.py` (ffmpeg subprocess) | `io/encode.py` (torchcodec Encoder) |
| `get_video_info` / ffprobe | `core/media.probe` (torchcodec metadata) |

GUI preview keeps framepipe `VideoGroupDecoder` until Project 2. `pyproject.toml`
adds `torchcodec >=0.13`; removes the `ffmpeg` binary dep, `av`; framepipe stays
(preview only) until Project 2.

## Testing

- **Pure unit** — `core/offsets.py`: synthetic clap with a known injected lag
  (incl. subframe, e.g. 12.3 ms), both methods; assert recovery within sub-hop
  error with `refine="parabolic"`, integer-hop with `refine="none"`.
- **Pure unit** — `core/timerange.py`: intersection/union over crafted inputs.
- **Pure unit** — export window math (local range, pad, frame-grid sampling
  timestamps) extracted as pure helpers from `export.py`.
- **Integration (slow, needs torchcodec + a codec)** — a couple: audio-offset
  round-trip after real export within <1 ms; muxed video export produces a
  non-empty MP4 of expected duration with an audio stream.

## CLAUDE.md cleanup (throughout)

Grouped/alphabetized imports (`__future__` → stdlib → third-party →
first-party); `X | None` over `Optional`; Google-format docstrings on the public
API; 80-col; f-strings; pure-first (window/grid math extracted pure from I/O).
Changes stay surgical — no unrelated GUI refactoring.

## Out of scope (Project 1)

- The GUI real-time playback engine rewrite and framepipe removal (Project 2).
- Sample-level / GCC-PHAT fine refinement.
- Video codecs/containers beyond H.264 (nvenc/libx264) MP4. Audio-only tracks
  support pass-through plus user-selected formats (wav/aac/flac/…).
