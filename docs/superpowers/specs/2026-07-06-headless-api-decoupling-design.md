# clapsync — Headless API decoupling & refactor

Date: 2026-07-06
Status: Approved design (pending spec review)

## Goal

Extract a Qt-free **headless API** from the current PySide6-coupled code so the
same logic drives (a) the existing GUI and (b) a new CLI and any scripted use.
Audio-based sync is the shared core; "audio track" and "video" inputs differ
only in what gets trimmed/exported. Apply CLAUDE.md (Google Python style)
cleanup throughout. Behavior is preserved except for two strict improvements:
parabolic sub-hop offset refinement and an explicit subframe-preserving export
contract.

## Requirements

The headless API must support, for both audio-only files and video files:

1. Find audio sync point (per-track offset on a shared timeline).
2. Find the common time range (where all tracks overlap) and the full range.
3. Automatically sync **and** trim/export tracks.

The GUI becomes a thin wrapper around this API (decoupled). A CLI wraps the
same API.

## Architecture

### The boundary: plain callbacks

Core functions are Qt-free and cross the boundary with plain callables — the
pattern `io/ffmpeg.py` and `io/audio.py` already use:

- `progress: Callable[[float], None] | None` — fraction in `[0, 1]`.
- `is_cancelled: Callable[[], bool] | None` — cooperative cancel check.

GUI workers adapt these to Qt signals; the CLI adapts them to a stderr/tqdm
bar. No Qt import ever appears under `core/`.

### Module layout

```
src/clapsync/
  core/                      # headless, Qt-free — the API
    __init__.py              # re-exports the public surface
    media.py                 # MediaInfo + probe() — unifies audio-only & video
    offsets.py               # pure MFCC/envelope finders + parabolic refine
    sync.py                  # compute_sync_offsets(...)
    timerange.py             # TimeRange, common_time_range(), full_time_range()
    export.py                # ExportSettings, ExportResult, export_tracks(), sync_and_trim()
  io/
    audio.py                 # load_audio_from_media (was load_audio_from_video)
    ffmpeg.py                # video export + new audio-only trim/pad branch
    probe.py                 # ffprobe helpers (audio sample-rate, stream presence)
  gui/
    workers.py               # QThread adapters: core callbacks -> Qt signals
    export_dialog.py         # widgets only (worker logic removed)
    sync_editor.py           # main window (moved under gui/)
    timeline_widget.py  video_player.py  video_selection_dialog.py   # unchanged
  cli.py                     # argparse headless entry (sync / synctrim subcommands)
  app.py                     # GUI entry (unchanged role)
```

## Public API (`clapsync.core`)

```python
# media.py
@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    has_audio: bool
    kind: Literal["audio", "video"]
    width: int | None = None      # video only
    height: int | None = None     # video only
    fps: float | None = None      # video only

def probe(path: Path) -> MediaInfo:
    """framepipe for video, ffprobe for audio-only files."""

# offsets.py  (pure — no I/O, no Qt)
Method = Literal["mfcc", "envelope"]
Refine = Literal["none", "parabolic"]

def find_offset(
    ref_waveform, ref_rate, waveform, rate, fps,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    ...
) -> tuple[int, float]:
    """(lag_frames, lag_seconds). lag_seconds is subframe-accurate."""

# sync.py
def compute_sync_offsets(
    media: list[MediaInfo],
    reference_index: int = 0,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
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

def common_time_range(media: list[MediaInfo], offsets: list[float]) -> TimeRange:
    """Intersection — the region where every track has footage."""

def full_time_range(media: list[MediaInfo], offsets: list[float]) -> TimeRange:
    """Union — earliest start to latest end."""

# export.py
@dataclass(frozen=True)
class ExportSettings:
    trim: TimeRange
    output_dir: Path
    target_width: int | None = None      # video only
    target_height: int | None = None     # video only
    output_fps: float | None = None       # video only
    audio_format: str | None = None       # audio only; None = pass-through container

@dataclass(frozen=True)
class ExportResult:
    path: Path
    error: str | None = None
    @property
    def ok(self) -> bool: return self.error is None

def export_tracks(
    media: list[MediaInfo],
    offsets: list[float],
    settings: ExportSettings,
    progress=None, is_cancelled=None,
) -> list[ExportResult]:
    """Trim+pad each track to settings.trim on the shared timeline.
    Audio -> audio file, video -> video file. Subframe-exact."""

def sync_and_trim(
    paths: list[Path],
    output_dir: Path,
    *,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    trim: Literal["common", "full"] = "common",
    reference_index: int = 0,
    progress=None, is_cancelled=None,
) -> list[ExportResult]:
    """probe -> compute_sync_offsets -> common/full range -> export_tracks."""
```

The six requirements (audio + video x {find sync, find range, auto sync+trim})
map onto the *same* functions; media kind is a field, not a separate code path.

## Subframe correctness

MFCC lag is subframe (~5 ms hop). This must survive to the output:

- Offsets are **float seconds** everywhere. Reference track offset is `0.0`.
- `lag_frames` is informational (GUI display) only — export never uses it.
- Video export: subframe `pad_start`/`pad_end` via `tpad` (video) plus a
  matching audio delay; fps conversion applied **last** with a PTS-preserving
  `fps` filter so wall-clock timing (hence cross-camera lock) survives
  requantization. Frames are held/extended at boundaries, never snapped to the
  output frame grid.
- Audio export: trim + exact silence pad (`adelay`/`apad`); no frame grid, so
  trivially subframe-exact. Output container is pass-through by default
  (`ExportSettings.audio_format = None`) or user-selected (e.g. `"wav"`,
  `"aac"`, `"flac"`).

## Parabolic refinement (strict improvement)

Coarse cross-correlation returns an integer-index peak (~5 ms MFCC hop / one
frame for envelope). With `refine="parabolic"` (default), fit a parabola to the
three correlation samples straddling the peak and take its analytic maximum for
sub-index precision. Pure, deterministic, ~free (no extra FFT). `refine="none"`
reproduces the old integer-peak behavior. Sample-level fine cross-correlation
was considered and dropped (YAGNI) but the `Refine` type leaves room for it.

## Migration (behavior-preserving)

| From | To |
|---|---|
| `audio_sync.py` | `core/offsets.py` (+ parabolic step) |
| `offset_worker.OffsetWorker` compute loop | `core/sync.compute_sync_offsets` |
| `offset_worker.compute_offsets_with_progress` | `gui/workers.py` adapter |
| `export_dialog.ExportWorker.run` | `core/export.export_tracks` |
| `export_dialog.ExportDialog` | `gui/export_dialog.py` (widgets only) |
| `io/ffmpeg.export_synced_video` | keep + audio-only branch |
| `io/audio._get_audio_sample_rate` | `io/probe.py` |
| `sync_editor.py`, `app.py` | move under `gui/`; `app.py` stays GUI entry |

`framepipe` remains the video probe. Audio-only files are probed via ffprobe.

## Testing

No tests exist today. Add:

- **Pure unit** — `core/offsets.py`: synthetic clap signal with a known
  injected lag (incl. subframe, e.g. 12.3 ms), both methods, assert recovery
  within sub-hop error with `refine="parabolic"`; assert `refine="none"` gives
  integer-hop result.
- **Pure unit** — `core/timerange.py`: intersection/union math over crafted
  offsets+durations.
- **Pure unit** — export boundary math (`local_start/end`, `pad_start/end`)
  extracted as a pure helper from the worker.
- **Integration (slow, needs ffmpeg binary)** — a couple: audio round-trip
  offset recovery after real export within <1 ms; video export produces a
  non-empty file of expected duration.

## CLAUDE.md cleanup (throughout)

Grouped/alphabetized imports (`__future__` -> stdlib -> third-party ->
first-party); `X | None` over `Optional`; Google-format docstrings on the new
public API; 80-col; f-strings; pure-first (export boundary math extracted pure
from I/O). Changes stay surgical — no unrelated refactoring of GUI widgets.

## Out of scope

- Sample-level / GCC-PHAT fine refinement.
- Any GUI redesign beyond moving files and swapping worker internals to call
  core.
- New video export codecs/containers beyond the current H.264/AAC MP4. Audio
  tracks support pass-through plus common user-selected formats (wav/aac/flac/…).
