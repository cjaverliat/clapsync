# Slim Pure Core + App Layer Relayering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split clapsync into a pure `clapsync.core` (MFCC audio sync on tensors + time-range math on caller data) and a `clapsync.app` layer (files, probe, torchcodec encode, export) that consumes it, removing `MediaInfo`/`probe` boilerplate from callers.

**Architecture:** `core/` depends only on torch/torchaudio/numpy and does no I/O. `app/` (moved from the old `core`-impure modules + `io/`) handles loading/probing/encoding and calls into core. `cli.py` and `gui/` are app entry points. The MFCC sync algorithm, parabolic refinement, sign convention, subframe export, and NVENC-probe+fallback are all preserved.

**Tech Stack:** Python 3.10–3.12, torch/torchaudio ≥2.11 (cu130), torchcodec ≥0.13, PySide6 (GUI), pytest.

## Global Constraints

- `clapsync.core` imports ONLY `torch`, `torchaudio`, `numpy`, stdlib. No torchcodec, no Qt, no file I/O. Importing `clapsync.core` must not import torchcodec.
- MFCC only. No `envelope` method, no `Method` type, no `fps` parameter, no `lag_frames`. `find_offset` returns a bare `float` (lag seconds).
- Sign convention (unchanged): positive lag = query LEADS reference; `offset[i]=lag`; `offset[reference_index]==0.0`; export uses `shared = local + offset`.
- Audio-required hard-fail stays in the app layer: `compute_sync_offsets`/`sync_and_trim` raise `ValueError` listing any path whose file has no audio stream.
- `MediaInfo`/`probe` live in `app/media.py` and are NOT re-exported from `clapsync.app` (internal).
- A module and a package named `app` cannot coexist: the GUI entry `clapsync/app.py` moves to `clapsync/gui/app.py`.
- CLAUDE.md/Google style: grouped imports (`__future__`→stdlib→third-party→first-party), Google docstrings on public API, `X | None`, 80-col, f-strings, pure-first.
- Run everything via `pixi run ...`. The suite must stay green after every task: `pixi run pytest -q -m 'slow or not slow'`.

---

## File Structure (end state)

```
src/clapsync/
  core/  __init__.py  offsets.py  timerange.py            # PURE
  app/   __init__.py  media.py  decode.py  encode.py  export.py  sync.py
  cli.py
  gui/   app.py  workers.py  sync_editor.py  export_dialog.py
         timeline_widget.py  video_player.py  video_selection_dialog.py
tests/
  core/  test_offsets.py  test_align.py  test_timerange.py  test_public_api.py
  app/   test_decode.py  test_encode.py  test_media.py  test_export.py
         test_cli.py  test_cpu_gpu.py
```

---

## Task 1: Relayer — move GUI entry, create `clapsync.app`, repoint imports (behavior identical)

Pure relocation: no logic changes. `find_offset` still returns a tuple and `compute_sync_offsets` still takes `list[MediaInfo]` at the end of this task. The suite must pass unchanged.

**Files:**
- Move: `src/clapsync/app.py` → `src/clapsync/gui/app.py`
- Move: `src/clapsync/io/decode.py` → `src/clapsync/app/decode.py`
- Move: `src/clapsync/io/encode.py` → `src/clapsync/app/encode.py`
- Move: `src/clapsync/core/media.py` → `src/clapsync/app/media.py`
- Move: `src/clapsync/core/export.py` → `src/clapsync/app/export.py`
- Move: `src/clapsync/core/sync.py` → `src/clapsync/app/sync.py`
- Create: `src/clapsync/app/__init__.py`
- Modify: `src/clapsync/core/__init__.py`, `src/clapsync/cli.py`, `src/clapsync/gui/workers.py`, `src/clapsync/gui/sync_editor.py`, `pyproject.toml`
- Move: `tests/integration/*` → `tests/app/*`
- Remove: `src/clapsync/io/` (now empty)

**Interfaces:**
- Produces: `clapsync.app` exporting `load_audio`, `compute_sync_offsets`, `ExportSettings`, `ExportResult`, `export_tracks`, `sync_and_trim` (same signatures as today). `clapsync.app.media.probe`/`MediaInfo` importable (internal). `clapsync.core` still exports `find_offset`, `TimeRange`, `common_time_range`, `full_time_range`, `Method`, `Refine`.

- [ ] **Step 1: Move the GUI entry point**

```bash
git mv src/clapsync/app.py src/clapsync/gui/app.py
```

In `src/clapsync/gui/app.py`, the imports that were `from clapsync.gui.video_selection_dialog import VideoSelectionDialog`, `from clapsync.gui.sync_editor import SyncEditorWindow`, `from clapsync.gui.workers import compute_offsets_with_progress` stay valid (already `clapsync.gui.*`). No edit needed beyond the move.

In `pyproject.toml`, update the script entry:

```toml
[project.scripts]
clapsync = "clapsync.gui.app:main"
clapsync-cli = "clapsync.cli:main"
```

- [ ] **Step 2: Create the app package and move modules into it**

```bash
mkdir -p src/clapsync/app
touch src/clapsync/app/__init__.py
git mv src/clapsync/io/decode.py  src/clapsync/app/decode.py
git mv src/clapsync/io/encode.py  src/clapsync/app/encode.py
git mv src/clapsync/core/media.py src/clapsync/app/media.py
git mv src/clapsync/core/export.py src/clapsync/app/export.py
git mv src/clapsync/core/sync.py  src/clapsync/app/sync.py
git rm -r src/clapsync/io 2>/dev/null || rmdir src/clapsync/io
```

- [ ] **Step 3: Fix imports inside the moved modules**

In `src/clapsync/app/sync.py`:

```python
from clapsync.app.decode import load_audio
```
(was `from clapsync.io.decode import load_audio`). Its `from clapsync.core.media import MediaInfo` becomes `from clapsync.app.media import MediaInfo`. Its `from clapsync.core.offsets import Method, Refine, find_offset` stays.

In `src/clapsync/app/export.py`, update these import lines:

```python
from clapsync.app.media import MediaInfo, probe
from clapsync.app.decode import decode_frames_at, load_audio
from clapsync.app.encode import encode_clip
```
(were `clapsync.core.media`, `clapsync.io.decode`, `clapsync.io.encode`). Its `from clapsync.core.sync import compute_sync_offsets` becomes `from clapsync.app.sync import compute_sync_offsets`. Its `from clapsync.core.timerange import ...` and `from clapsync.core.offsets import Method, Refine` stay.

`src/clapsync/app/media.py`, `decode.py`, `encode.py` have no `clapsync.*` imports to change (verify with grep in Step 6).

- [ ] **Step 4: Write `app/__init__.py` and shrink `core/__init__.py`**

`src/clapsync/app/__init__.py`:

```python
"""clapsync app layer: file loading, probing, and muxed trim/export."""
from clapsync.app.decode import load_audio
from clapsync.app.export import (
    ExportResult,
    ExportSettings,
    export_tracks,
    sync_and_trim,
)
from clapsync.app.sync import compute_sync_offsets

__all__ = [
    "load_audio",
    "compute_sync_offsets",
    "ExportSettings",
    "ExportResult",
    "export_tracks",
    "sync_and_trim",
]
```

Replace `src/clapsync/core/__init__.py` with the core-only surface:

```python
"""clapsync pure core: MFCC audio sync and time-range math (no I/O)."""
from clapsync.core.offsets import Method, Refine, find_offset
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "find_offset",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "Method",
    "Refine",
]
```

- [ ] **Step 5: Repoint callers (cli, gui workers, sync_editor)**

In `src/clapsync/cli.py`, the import block that pulled everything from `clapsync.core` splits. Use:

```python
from clapsync.app import compute_sync_offsets, sync_and_trim
from clapsync.app.media import probe
from clapsync.core import common_time_range, full_time_range
```

In `src/clapsync/gui/workers.py`, change:

```python
from clapsync.app import (
    ExportResult,
    ExportSettings,
    compute_sync_offsets,
    export_tracks,
)
from clapsync.app.media import MediaInfo, probe
```
(was a single `from clapsync.core import (...)`). Keep the rest of the file unchanged.

In `src/clapsync/gui/sync_editor.py`, change the core imports to:

```python
from clapsync.app import ExportSettings, ExportResult
from clapsync.app.media import probe
from clapsync.core import TimeRange, common_time_range
```
(match whatever subset it currently imports; `ExportWorker` comes from `clapsync.gui.workers` as before, `ExportDialog` from `clapsync.gui.export_dialog`).

- [ ] **Step 6: Move tests and verify no stale imports**

```bash
mkdir -p tests/app
git mv tests/integration/__init__.py tests/app/__init__.py
git mv tests/integration/test_decode.py  tests/app/test_decode.py
git mv tests/integration/test_encode.py  tests/app/test_encode.py
git mv tests/integration/test_media.py   tests/app/test_media.py
git mv tests/integration/test_export.py  tests/app/test_export.py
git mv tests/integration/test_cli.py     tests/app/test_cli.py
git mv tests/integration/test_cpu_gpu.py tests/app/test_cpu_gpu.py
rmdir tests/integration 2>/dev/null || true
```

In the moved test files, replace import prefixes: `clapsync.io.decode`→`clapsync.app.decode`, `clapsync.io.encode`→`clapsync.app.encode`, `clapsync.core.media`→`clapsync.app.media`, `clapsync.core.export`→`clapsync.app.export`. Any `from clapsync.core import ...` of app symbols (probe/export/sync) becomes `from clapsync.app import ...` / `from clapsync.app.media import probe`.

Verify nothing references the old locations:

```bash
grep -rn "clapsync\.io\|clapsync\.core\.media\|clapsync\.core\.export\|clapsync\.core\.sync" src tests
```
Expected: no hits (all references now point at `clapsync.app.*`).

- [ ] **Step 7: Run the full suite (must be green, unchanged behavior)**

Run: `pixi run pytest -q -m 'slow or not slow'`
Expected: all pass (same count as before the task, 35).

Import smoke:

Run: `pixi run python -c "import clapsync.core; import clapsync.app; import clapsync.gui.app, clapsync.cli; print('ok')"`
Expected: `ok`, no ImportError.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: split clapsync into pure core/ and app/ layers (moves only)"
```

---

## Task 2: Slim the core — MFCC-only offsets, `find_offset` → float, add `align_waveforms`

**Files:**
- Modify: `src/clapsync/core/offsets.py`, `src/clapsync/core/__init__.py`, `src/clapsync/app/sync.py`
- Test: `tests/core/test_offsets.py` (update), `tests/core/test_align.py` (create)

**Interfaces:**
- Consumes: `_parabolic_peak`, `_compute_mfcc`, `_mfcc_cross_correlate`, `_to_mono_f64` (already in offsets.py).
- Produces:
  - `find_offset(ref_waveform, ref_rate, waveform, rate, *, refine="parabolic", n_mfcc=13, n_fft=2048, hop_duration=0.005, win_duration=0.04, n_mels=128, mel_scale="htk") -> float`
  - `align_waveforms(waveforms: list[Tensor], rates: list[int], *, refine="parabolic", reference_index=0, progress=None) -> list[float]`
  - `Refine = Literal["none", "parabolic"]` (keep). `Method` removed.

- [ ] **Step 1: Write the failing tests**

Update `tests/core/test_offsets.py` — replace the envelope test and adjust `find_offset` to the float return. Full file:

```python
import numpy as np
import torch

from clapsync.core.offsets import find_offset, _parabolic_peak


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_parabolic_peak_interpolates_between_samples():
    corr = np.array([0.0, 1.0, 1.2, 0.3])
    frac = _parabolic_peak(corr, 2)
    assert 1.5 < frac < 2.5
    assert frac != 2.0


def test_parabolic_peak_clamps_at_edges():
    corr = np.array([5.0, 1.0, 0.0])
    assert _parabolic_peak(corr, 0) == 0.0


def test_find_offset_returns_float_and_recovers_subframe_lag():
    # Sign convention: positive = query leads. Here sub is 12.3 ms LATER than
    # ref, so the lag must be negative.
    sr = 48000
    n = sr * 2
    true_lag = 0.0123
    ref = _click_track(n, click_at=sr)
    sub = _click_track(n, click_at=sr + int(true_lag * sr))
    lag = find_offset(ref, sr, sub, sr)
    assert isinstance(lag, float)
    assert abs(lag - (-true_lag)) < 0.005


def test_find_offset_none_refine_is_integer_hop():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)
    sub = _click_track(n, click_at=sr + int(0.20 * sr))
    lag_none = find_offset(ref, sr, sub, sr, refine="none")
    lag_par = find_offset(ref, sr, sub, sr, refine="parabolic")
    # Both near -0.20 s; parabolic is at least as close.
    assert abs(lag_par - (-0.20)) <= abs(lag_none - (-0.20)) + 1e-9
```

Create `tests/core/test_align.py`:

```python
import numpy as np
import torch

from clapsync.core.offsets import align_waveforms


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_align_waveforms_reference_zero_and_signed_offsets():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)                    # 1.000 s
    later = _click_track(n, click_at=sr + int(0.20 * sr))  # +200 ms (delayed)
    earlier = _click_track(n, click_at=sr - int(0.10 * sr))  # -100 ms (leads)

    offsets = align_waveforms([ref, later, earlier], [sr, sr, sr])
    assert offsets[0] == 0.0
    assert abs(offsets[1] - (-0.20)) < 0.02   # delayed -> negative
    assert abs(offsets[2] - (0.10)) < 0.02    # leads -> positive


def test_align_waveforms_progress_reaches_one():
    sr = 48000
    n = sr
    tracks = [_click_track(n, click_at=n // 2) for _ in range(3)]
    seen = []
    align_waveforms(tracks, [sr, sr, sr], progress=seen.append)
    assert seen and seen[-1] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/core/test_offsets.py tests/core/test_align.py -q`
Expected: FAIL — `find_offset` currently returns a tuple / takes `fps`; `align_waveforms` does not exist.

- [ ] **Step 3: Rewrite `core/offsets.py`**

Delete these from `src/clapsync/core/offsets.py`: the entire "Envelope method" section (`_build_envelope`, `_fft_correlate`, `find_offset_envelope`), the `Method` alias, and the `find_offset` dispatcher + `find_offset_mfcc`. Keep `_to_mono_f64`, `_parabolic_peak`, `_compute_mfcc`, `_mfcc_cross_correlate`. Keep `Refine`.

Add the new `find_offset` (former `find_offset_mfcc` body, float return, no fps) in the MFCC section:

```python
def find_offset(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    *,
    refine: Refine = "parabolic",
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
) -> float:
    """Temporal offset between two waveforms via MFCC cross-correlation.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate: Reference sample rate in Hz.
        waveform: Query audio, resampled to ref_rate if rates differ.
        rate: Query sample rate in Hz.
        refine: "parabolic" (sub-hop interpolation) or "none" (integer hop).
        n_mfcc, n_fft, hop_duration, win_duration, n_mels, mel_scale: MFCC params.

    Returns:
        Lag in seconds. Positive means the query leads (starts before) the
        reference; negative means it starts later.
    """
    if ref_rate != rate:
        waveform = AF.resample(waveform, orig_freq=rate, new_freq=ref_rate)

    ref_mono = _to_mono_f64(ref_waveform)
    sub_mono = _to_mono_f64(waveform)

    hop_length = int(ref_rate * hop_duration)
    win_length = int(ref_rate * win_duration)

    mfcc_ref = _compute_mfcc(
        ref_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale,
    )
    mfcc_sub = _compute_mfcc(
        sub_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale,
    )

    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_sub)
    peak_idx = int(np.argmax(corr))
    peak = _parabolic_peak(corr, peak_idx) if refine == "parabolic" else float(peak_idx)

    # Sign convention: positive lag = query leads the reference.
    lag_hops = peak - (mfcc_sub.shape[1] - 1)
    return lag_hops * hop_length / ref_rate


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
        waveforms: Per-track audio tensors.
        rates: Per-track sample rate in Hz (parallel to waveforms).
        refine: Peak refinement ("parabolic" or "none").
        reference_index: Track whose timeline is the origin (offset 0.0).
        progress: Optional 0..1 callback.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0. Positive
        means the track leads the reference (shared = local + offset).
    """
    n = len(waveforms)
    ref_wave = waveforms[reference_index]
    ref_rate = rates[reference_index]

    offsets = [0.0] * n
    for i in range(n):
        if i != reference_index:
            offsets[i] = find_offset(
                ref_wave, ref_rate, waveforms[i], rates[i], refine=refine,
            )
        if progress is not None:
            progress((i + 1) / n)
    return offsets
```

Ensure the imports at the top of `offsets.py` include `from typing import Callable, Literal` and keep `import numpy as np`, `import torch`, `import torchaudio.functional as AF`, `from torchaudio.transforms import MFCC`. Remove `import logging`/`logger` only if no longer used (envelope removal may orphan the debug logs — delete orphaned `logger` lines).

- [ ] **Step 4: Update `core/__init__.py` (drop `Method`)**

```python
"""clapsync pure core: MFCC audio sync and time-range math (no I/O)."""
from clapsync.core.offsets import Refine, align_waveforms, find_offset
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "align_waveforms",
    "find_offset",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "Refine",
]
```

- [ ] **Step 5: Rewire `app/sync.py` to `align_waveforms`**

Replace the body of `compute_sync_offsets` in `src/clapsync/app/sync.py` so it loads waveforms and delegates to `align_waveforms` (still takes `list[MediaInfo]` in this task — the paths rewrite is Task 3). Its imports become:

```python
from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo
from clapsync.core.offsets import Refine, align_waveforms
```

Body:

```python
def compute_sync_offsets(
    media: list[MediaInfo],
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Align probed tracks by loading their audio and cross-correlating.

    Args:
        media: Probed tracks; each must have an audio stream.
        reference_index: Track whose timeline is the origin.
        refine: Peak refinement.
        progress: Optional 0..1 callback.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0.
    """
    missing = [str(m.path) for m in media if not m.has_audio]
    if missing:
        raise ValueError(
            "cannot sync tracks without an audio stream: " + ", ".join(missing)
        )

    n = len(media)
    waveforms: list[torch.Tensor] = []
    rates: list[int] = []
    for i, info in enumerate(media):
        if is_cancelled is not None and is_cancelled():
            return [0.0] * n
        wave, rate = load_audio(info.path)
        waveforms.append(wave)
        rates.append(rate)
        if progress is not None:
            progress(0.9 * (i + 1) / n)

    offsets = align_waveforms(
        waveforms, rates, refine=refine, reference_index=reference_index,
    )
    if progress is not None:
        progress(1.0)
    return offsets
```

Add `import torch` and `from typing import Callable` at the top if not present. Remove the now-unused `find_offset`, `Method`, and `ref_fps` references.

- [ ] **Step 6: Run tests**

Run: `pixi run pytest tests/core/test_offsets.py tests/core/test_align.py -q`
Expected: all pass.

Run: `pixi run pytest -q -m 'slow or not slow'`
Expected: all pass (envelope test gone; align tests added). Note the total shifts accordingly.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(core): MFCC-only find_offset->float + align_waveforms; drop envelope/fps"
```

---

## Task 3: Path-taking app API; `MediaInfo` internal

**Files:**
- Modify: `src/clapsync/app/sync.py`, `src/clapsync/app/export.py`, `src/clapsync/cli.py`, `src/clapsync/gui/workers.py`, `src/clapsync/gui/sync_editor.py`
- Test: `tests/app/test_export.py`, `tests/app/test_cli.py` (update to pass paths)

**Interfaces:**
- Produces:
  - `compute_sync_offsets(paths: list[Path], *, refine="parabolic", reference_index=0, progress=None, is_cancelled=None) -> list[float]`
  - `export_tracks(paths: list[Path], offsets: list[float], settings: ExportSettings, progress=None, is_cancelled=None) -> list[ExportResult]`
  - `sync_and_trim(paths: list[Path], output_dir: Path, *, refine="parabolic", trim="common", reference_index=0, progress=None, is_cancelled=None) -> list[ExportResult]`
  - `clapsync.app` re-exports these + `load_audio`, `ExportSettings`, `ExportResult`. `probe`/`MediaInfo` stay in `app.media`, not re-exported.

- [ ] **Step 1: Update the app tests to pass paths (write-first)**

In `tests/app/test_export.py`, the roundtrip test already calls `sync_and_trim([a, b], out, ...)` with paths — no change needed there. If any test constructed `MediaInfo`/probed manually before calling `export_tracks`, change it to pass paths directly, e.g.:

```python
from clapsync.app import sync_and_trim, ExportResult, compute_sync_offsets
# export_tracks now takes paths:
from clapsync.app import export_tracks, ExportSettings
from clapsync.core import common_time_range
```

Add a direct `export_tracks(paths, ...)` test:

```python
import pytest
from clapsync.app import export_tracks, compute_sync_offsets, ExportSettings
from clapsync.core import common_time_range
from clapsync.app.media import probe


@pytest.mark.slow
def test_export_tracks_takes_paths(av_video, tmp_path):
    a, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="a.mp4")
    b, *_ = av_video(seconds=1.0, fps=30.0, w=256, h=144, name="b.mp4")
    paths = [a, b]
    offsets = compute_sync_offsets(paths)
    durations = [probe(p).duration for p in paths]
    settings = ExportSettings(
        trim=common_time_range(durations, offsets), output_dir=tmp_path,
    )
    results = export_tracks(paths, offsets, settings)
    assert all(r.ok for r in results), [r.error for r in results]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run pytest tests/app/test_export.py::test_export_tracks_takes_paths -q`
Expected: FAIL — `export_tracks` currently takes `media`, not paths.

- [ ] **Step 3: Make `compute_sync_offsets` take paths**

In `src/clapsync/app/sync.py`, change the signature and probe internally:

```python
from pathlib import Path

from clapsync.app.decode import load_audio
from clapsync.app.media import probe
from clapsync.core.offsets import Refine, align_waveforms


def compute_sync_offsets(
    paths: list[Path],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Probe, load audio, and align paths by MFCC cross-correlation.

    Raises:
        ValueError: If any input has no audio stream.
    """
    media = [probe(p) for p in paths]
    missing = [str(m.path) for m in media if not m.has_audio]
    if missing:
        raise ValueError(
            "cannot sync tracks without an audio stream: " + ", ".join(missing)
        )

    n = len(media)
    waveforms = []
    rates = []
    for i, info in enumerate(media):
        if is_cancelled is not None and is_cancelled():
            return [0.0] * n
        wave, rate = load_audio(info.path)
        waveforms.append(wave)
        rates.append(rate)
        if progress is not None:
            progress(0.9 * (i + 1) / n)

    offsets = align_waveforms(
        waveforms, rates, refine=refine, reference_index=reference_index,
    )
    if progress is not None:
        progress(1.0)
    return offsets
```

- [ ] **Step 4: Make `export_tracks`/`sync_and_trim` take paths**

In `src/clapsync/app/export.py`, change `export_tracks` to probe internally:

```python
def export_tracks(
    paths: list[Path],
    offsets: list[float],
    settings: ExportSettings,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Trim+pad every track to settings.trim and encode one file each."""
    media = [probe(p) for p in paths]
    device = "cuda" if _nvenc_available() else "cpu"
    trim = settings.trim
    results: list[ExportResult] = []
    n = len(media)

    for i, (info, offset) in enumerate(zip(media, offsets)):
        # ... existing per-track body unchanged ...
```
Keep the entire existing loop body (video/audio branches, nvenc per-clip fallback, ExportResult recording) verbatim — only the signature and the added `media = [probe(p) for p in paths]` line change.

Change `sync_and_trim` to pass paths through:

```python
def sync_and_trim(
    paths: list[Path],
    output_dir: Path,
    *,
    refine: Refine = "parabolic",
    trim: Literal["common", "full"] = "common",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """probe -> compute_sync_offsets -> common/full range -> export_tracks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def sync_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 * f)

    offsets = compute_sync_offsets(
        paths, reference_index=reference_index, refine=refine,
        progress=sync_progress, is_cancelled=is_cancelled,
    )

    durations = [probe(p).duration for p in paths]
    rng = (common_time_range if trim == "common" else full_time_range)(
        durations, offsets,
    )
    settings = ExportSettings(trim=rng, output_dir=output_dir)

    def export_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 + 0.5 * f)

    return export_tracks(
        paths, offsets, settings,
        progress=export_progress, is_cancelled=is_cancelled,
    )
```
Remove the old `MediaInfo`-based `method=`/media threading. Drop the now-unused `Method` import; keep `probe`, `Refine`, `common_time_range`, `full_time_range`.

- [ ] **Step 5: Rewire GUI workers and sync_editor to paths**

In `src/clapsync/gui/workers.py`:
- `OffsetWorker.run`: replace `media = [probe(p) for p in self._paths]; compute_sync_offsets(media, ...)` with `offsets = compute_sync_offsets(self._paths, progress=lambda f: self.progress_value.emit(int(f * 1000)))`. Remove the `probe`/`MediaInfo` imports if now unused.
- `ExportWorker.__init__` takes `(paths: list[Path], offsets, settings)` instead of `(media, offsets, settings)`; store `self._paths`. In `run`, call `export_tracks(self._paths, self._offsets, self._settings, progress=_on_progress, is_cancelled=lambda: self._cancelled)`. Update the status math `len(self._paths)`.

In `src/clapsync/gui/sync_editor.py` `_on_export`: replace `media = [probe(info.path) for info in self._video_infos]` and `ExportWorker(media, ...)` with:

```python
paths = [info.path for info in self._video_infos]
worker = ExportWorker(paths, self._offsets, settings)
```
Remove the `from clapsync.app.media import probe` import if it becomes unused.

- [ ] **Step 6: Update the CLI to paths**

In `src/clapsync/cli.py` `_cmd_sync`, replace the probe+compute with:

```python
from clapsync.app.media import probe  # durations for range display

offsets = compute_sync_offsets(
    args.inputs, reference_index=args.reference, refine=args.refine,
    progress=lambda f: print(f"\rsync {f*100:3.0f}%", end="", file=sys.stderr),
)
durations = [probe(p).duration for p in args.inputs]
common = common_time_range(durations, offsets)
full = full_time_range(durations, offsets)
```
`_cmd_synctrim` already calls `sync_and_trim(args.inputs, args.output, ...)`; drop the `--method` argument and any `method=` kwarg (MFCC only now) — remove `--method` from `_add_common`, keep `--refine`.

- [ ] **Step 7: Run the suite**

Run: `pixi run pytest -q -m 'slow or not slow'`
Expected: all pass.

Import smoke:

Run: `pixi run python -c "import clapsync.gui.app, clapsync.cli; from clapsync.app import compute_sync_offsets, export_tracks, sync_and_trim; print('ok')"`
Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(app): path-taking compute_sync_offsets/export_tracks/sync_and_trim; MediaInfo internal"
```

---

## Task 4: Dependency split, examples, README, final verify

**Files:**
- Modify: `pyproject.toml`, `README.md`
- Rewrite: `examples/find_sync.py`, `examples/auto_sync_trim.py`, `examples/export_custom.py`

**Interfaces:**
- Consumes: `clapsync.core.align_waveforms`, `common_time_range`, `full_time_range`; `clapsync.app.sync_and_trim`, `compute_sync_offsets`, `export_tracks`, `ExportSettings`.

- [ ] **Step 1: Split dependencies in `pyproject.toml`**

Replace the `[project].dependencies` and add optional groups:

```toml
dependencies = [
    "numpy",
    "torch>=2.11",
    "torchaudio>=2.11",
]

[project.optional-dependencies]
app = ["torchcodec>=0.13"]
gui = ["clapsync[app]", "PySide6>=6.7,<6.8"]
```

Keep the `[tool.pixi.pypi-dependencies]` cu130 index pins for torch/torchaudio/torchcodec as-is (dev/GPU). Ensure PySide6 stays available to the pixi env (it is already listed under pixi pypi-dependencies). No pixi environment change is required — pixi installs the workspace with its pinned set.

- [ ] **Step 2: Rewrite `examples/find_sync.py` as the pure path**

```python
"""Find sync offsets and the common/full time range from audio waveforms.

The caller loads audio (here with torchaudio) and passes tensors to the pure
core — clapsync.core does no file I/O.

Usage:
    pixi run python examples/find_sync.py clip_a.wav clip_b.mp4 clip_c.wav
"""
from __future__ import annotations

import sys

import torchaudio

from clapsync.core import align_waveforms, common_time_range, full_time_range


def main(paths: list[str]) -> None:
    waveforms = []
    rates = []
    durations = []
    for path in paths:
        wave, rate = torchaudio.load(path)
        waveforms.append(wave)
        rates.append(rate)
        durations.append(wave.shape[-1] / rate)

    offsets = align_waveforms(waveforms, rates)

    for path, offset in zip(paths, offsets):
        print(f"{path:30s} offset = {offset:+.4f} s")

    common = common_time_range(durations, offsets)
    full = full_time_range(durations, offsets)
    print(
        f"\ncommon overlap : {common.start:.3f} -> {common.end:.3f} s "
        f"({common.duration:.3f} s)"
    )
    print(
        f"full timeline  : {full.start:.3f} -> {full.end:.3f} s "
        f"({full.duration:.3f} s)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: find_sync.py <clip1> <clip2> [clip3 ...]")
    main(sys.argv[1:])
```

- [ ] **Step 3: Point the other two examples at `clapsync.app`**

In `examples/auto_sync_trim.py` change the import to `from clapsync.app import sync_and_trim` (call is unchanged). In `examples/export_custom.py` change imports to:

```python
from clapsync.app import ExportSettings, compute_sync_offsets, export_tracks
from clapsync.app.media import probe
from clapsync.core import common_time_range
```
and, since `export_tracks` takes paths, replace the `media = [probe(p) ...]` / `export_tracks(media, ...)` usage with:

```python
paths = args.inputs
offsets = compute_sync_offsets(paths)
durations = [probe(p).duration for p in paths]
trim = common_time_range(durations, offsets)
settings = ExportSettings(
    trim=trim, output_dir=args.output,
    target_width=args.width, target_height=args.height, output_fps=args.fps,
)
results = export_tracks(paths, offsets, settings,
                        progress=lambda f: print(f"\r{f*100:5.1f}%", end="", flush=True))
```

- [ ] **Step 4: Update the README API section**

In `README.md`, split the API table into a pure-core table and an app table, and update the quick-start snippet import to `from clapsync.app import sync_and_trim`. Update the `examples/` bullet for `find_sync.py` to note it is pure (torchaudio + `clapsync.core`). Update the Layout block to show `core/` (pure) and `app/`.

- [ ] **Step 5: Reinstall + verify examples compile + full suite**

Run: `pixi install`
Expected: resolves (exit 0).

Run: `pixi run python -m py_compile examples/*.py`
Expected: no output (compiles).

Run: `pixi run python examples/auto_sync_trim.py --help >/dev/null && echo ok`
Expected: `ok`.

Run: `pixi run pytest -q -m 'slow or not slow'`
Expected: all pass.

Run: `pixi run python -c "import clapsync.core, sys; assert 'torchcodec' not in sys.modules; print('core is torchcodec-free')"`
Expected: `core is torchcodec-free` (importing core must not import torchcodec).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "build+docs: split deps (core/app/gui extras); pure find_sync example; README API"
```

---

## Self-Review Notes

- **Spec coverage:** pure core (T2), app layer + moves (T1), path-taking API + MediaInfo internal (T3), dep split + examples + README (T4), GUI-entry move + naming-clash fix (T1), `find_offset`→float / MFCC-only / no fps (T2), audio hard-fail preserved (T2/T3), sign convention preserved + tested (T2). All spec sections mapped.
- **`core` torchcodec-free** asserted directly in T4 Step 5.
- **Behavior preservation:** MFCC/parabolic/sign/subframe/NVENC-fallback bodies are moved or kept verbatim; only envelope is deleted and signatures re-layered.
- **Green between tasks:** T1 is a pure move (same behavior); T2 swaps algorithm surface with matching test updates; T3 changes call shape with matching test updates; T4 is docs/deps.
