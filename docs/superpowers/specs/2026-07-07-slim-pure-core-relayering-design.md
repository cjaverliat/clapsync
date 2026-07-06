# clapsync — Slim pure core + app layer relayering

Date: 2026-07-07
Status: Approved design (pending spec review)

## Goal

Refactor clapsync into a **slim, pure `clapsync.core` library** (audio sync from
waveform tensors + time-range math on caller-provided data) with all file I/O,
probing, torchcodec encoding, CLI, and GUI moved into a separate `clapsync.app`
layer that consumes the core. This removes the `MediaInfo`/`probe` boilerplate
from callers, follows CLAUDE.md (pure functions first), and lets the pure sync
library be imported and used without torchcodec.

Trigger: the current public API forces `[probe(p) for p in paths]` boilerplate
and mixes pure algorithm with file I/O. The slim core takes tensors and plain
data; the app layer handles files.

## Non-goals

- Project 2 (real-time GUI playback engine on torchcodec; framepipe removal).
- Lossless stream-copy trim (a separate future option).
- Any change to the sync algorithm itself (MFCC cross-correlation + parabolic
  sub-hop refinement + sign convention are preserved exactly).

## Architecture

Two layers in one package:

- **`clapsync.core`** — pure. Depends only on `torch`, `torchaudio`, `numpy`.
  No file I/O, no torchcodec, no Qt. Importing it never imports torchcodec.
- **`clapsync.app`** — file loading, probing, torchcodec decode/encode, export
  orchestration. Consumes `clapsync.core`. `cli.py`, `gui/`, and the GUI entry
  point are app-level.

### Package layout

```
src/clapsync/
  core/                 # PURE (torch / torchaudio / numpy only)
    __init__.py         # align_waveforms, find_offset, TimeRange,
                        #   common_time_range, full_time_range, Refine
    offsets.py          # find_offset (MFCC only) + align_waveforms + parabolic
    timerange.py        # TimeRange, common_time_range, full_time_range
  app/                  # file / IO / export — consumes core
    __init__.py         # load_audio, compute_sync_offsets, ExportSettings,
                        #   ExportResult, export_tracks, sync_and_trim
    media.py            # MediaInfo, probe          (internal to app)
    decode.py           # load_audio                (moved from io/decode.py)
    encode.py           # encode_clip               (moved from io/encode.py)
    export.py           # ExportSettings, ExportResult, export_tracks, sync_and_trim
    sync.py             # compute_sync_offsets(paths)
  cli.py                # CLI entry (clapsync-cli)
  gui/
    app.py              # GUI entry (MOVED from clapsync/app.py)
    workers.py, sync_editor.py, export_dialog.py,
    timeline_widget.py, video_player.py, video_selection_dialog.py
```

`io/` is removed (its two modules move into `app/`). The GUI entry moves from
`clapsync/app.py` to `clapsync/gui/app.py` to free the `app` name for the
subpackage (a module and a package of the same name cannot coexist).

## Pure core API (`clapsync.core`)

```python
Refine = Literal["none", "parabolic"]

def align_waveforms(
    waveforms: list[torch.Tensor],
    rates: list[int],
    *,
    refine: Refine = "parabolic",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
) -> list[float]:
    """Align each waveform to a reference by MFCC cross-correlation.

    Args:
        waveforms: Per-track audio, each shape (channels, samples) or (samples,).
        rates: Per-track sample rate in Hz (parallel to waveforms).
        refine: Peak refinement ("parabolic" or "none").
        reference_index: Track whose timeline is the origin.
        progress: Optional 0..1 callback.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0. Positive
        means the track leads (starts before) the reference. Used as
        offset with the convention shared_time = local_time + offset.
    """

def find_offset(
    ref_waveform: torch.Tensor, ref_rate: int,
    waveform: torch.Tensor, rate: int,
    *,
    refine: Refine = "parabolic",
    n_mfcc: int = 13, n_fft: int = 2048,
    hop_duration: float = 0.005, win_duration: float = 0.04,
    n_mels: int = 128, mel_scale: Literal["htk", "slaney"] = "htk",
) -> float:
    """Pairwise MFCC lag in seconds. Positive = query leads the reference."""

@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float
    @property
    def duration(self) -> float: ...

def common_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Intersection — where every track overlaps. Caller supplies durations."""

def full_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Union — earliest start to latest end."""
```

Removed vs the previous API: `Method` (MFCC only now), `fps` (MFCC lag in
seconds needs no frame rate), `lag_frames` (`find_offset` returns a bare float),
`MediaInfo`/`probe` (not in core). No audio-presence check in core — the caller
provides valid tensors; the app layer enforces "must have audio".

## App API (`clapsync.app`)

```python
def load_audio(path: Path, target_rate: int | None = None) -> tuple[torch.Tensor, int]:
    """Peak-normalized mono waveform + sample rate via torchcodec."""

def compute_sync_offsets(
    paths: list[Path], *,
    refine: Refine = "parabolic", reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Probe (internal) -> hard-fail if any track lacks audio -> load_audio each
    -> core.align_waveforms. Offsets in seconds; reference == 0.0."""

@dataclass(frozen=True)
class ExportSettings:
    trim: TimeRange
    output_dir: Path
    target_width: int | None = None
    target_height: int | None = None
    output_fps: float | None = None
    video_codec: str | None = None
    crf: int = 18
    audio_format: str | None = None

@dataclass(frozen=True)
class ExportResult:
    path: Path
    error: str | None = None
    @property
    def ok(self) -> bool: ...

def export_tracks(
    paths: list[Path], offsets: list[float], settings: ExportSettings,
    progress=None, is_cancelled=None,
) -> list[ExportResult]:
    """Probe internally, trim+pad each track to settings.trim, encode. Video ->
    muxed A/V MP4 (NVENC when available, CPU fallback); audio-only -> audio file."""

def sync_and_trim(
    paths: list[Path], output_dir: Path, *,
    refine: Refine = "parabolic",
    trim: Literal["common", "full"] = "common",
    reference_index: int = 0,
    progress=None, is_cancelled=None,
) -> list[ExportResult]:
    """probe -> compute_sync_offsets -> common/full range -> export_tracks."""
```

`MediaInfo` and `probe` remain in `app/media.py` but are internal (not
re-exported from `clapsync.app`). Export functions take paths and probe
internally.

## Dependencies

```toml
[project]
dependencies = ["numpy", "torch>=2.11", "torchaudio>=2.11"]   # pure core

[project.optional-dependencies]
app = ["torchcodec>=0.13"]
gui = ["clapsync[app]", "PySide6>=6.7,<6.8"]
```

`pip install clapsync` installs the slim sync library (no torchcodec).
`clapsync[gui]` installs the full app. The pixi workspace keeps its cu130 index
pins in `[tool.pixi.pypi-dependencies]` and installs both extras for dev/GPU.

## Migration (moves + rewrites)

| From | To | Change |
|---|---|---|
| `core/offsets.py` | `core/offsets.py` | remove envelope (`_build_envelope`, `_fft_correlate`, `find_offset_envelope`); `find_offset` returns `float`, drop `fps`/`Method`; add `align_waveforms` |
| `core/sync.py` | `app/sync.py` | rewrite: `compute_sync_offsets(paths)` loads audio + calls `core.align_waveforms`; keeps audio-required hard-fail + progress/cancel |
| `core/media.py` | `app/media.py` | move; internal |
| `core/export.py` | `app/export.py` | move; `export_tracks`/`sync_and_trim` take paths; use `core.common_time_range`/`full_time_range` |
| `io/decode.py` | `app/decode.py` | move (`io/` removed) |
| `io/encode.py` | `app/encode.py` | move |
| `core/__init__.py` | `core/__init__.py` | shrink to pure symbols |
| — | `app/__init__.py` | new app public surface |
| `app.py` | `gui/app.py` | GUI entry move |
| `cli.py` | `cli.py` | imports -> `clapsync.app` + `clapsync.core` |
| `gui/workers.py`, `gui/sync_editor.py`, `gui/export_dialog.py` | same | imports -> `clapsync.app` + `clapsync.core` |
| `pyproject.toml` | | dependency split + `[project.scripts]` `clapsync = "clapsync.gui.app:main"` |

`find_offset` now returns a bare float (was `(lag_frames, lag_seconds)`); its
only callers are `align_waveforms`/`compute_sync_offsets` and tests — all
updated.

## Testing

- **`tests/core/` — pure, no `slow` marks:**
  - `test_offsets.py`: `find_offset` MFCC recovers a known lag from a click pair
    (correct sign — delayed query -> negative); `_parabolic_peak` interpolation +
    edge clamp; `refine="none"` gives integer-hop result. Envelope test removed.
  - `test_align.py`: `align_waveforms` over 3 synthetic click tracks with known
    lags returns correct signed offsets, `offset[reference_index] == 0.0`,
    progress reaches 1.0.
  - `test_timerange.py`: intersection/union math (unchanged).
- **`tests/app/` — slow (needs codecs/GPU):** decode, encode, probe, export
  round-trip, CLI. Moved from `tests/integration/`; import paths updated to
  `clapsync.app.*`. `av_video`/`tone_wav`/`rgb_video` conftest fixtures unchanged.

## Examples

- `examples/find_sync.py` — **pure**: load with `torchaudio.load`, call
  `clapsync.core.align_waveforms`, print offsets and `common_time_range` /
  `full_time_range`. No clapsync file I/O.
- `examples/auto_sync_trim.py` — `clapsync.app.sync_and_trim(paths, out)`.
- `examples/export_custom.py` — `clapsync.app`: `compute_sync_offsets` +
  `common_time_range` + `ExportSettings` + `export_tracks`.

README API section updated: `clapsync.core` (pure) table + `clapsync.app`
(files) table; the quick-start snippet uses `clapsync.app.sync_and_trim`.

## Behavior preservation

The MFCC offset algorithm, parabolic refinement, sign convention
(`shared = local + offset`; positive lag = leads), subframe export handling,
NVENC-probe + per-clip CPU fallback, and audio-required hard-fail are all
preserved. Only the envelope method is removed and the API is re-layered.
