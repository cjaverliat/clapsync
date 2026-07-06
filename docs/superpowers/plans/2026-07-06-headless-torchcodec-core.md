# Headless torchcodec Core & CLI — Implementation Plan (Project 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a Qt-free headless `core/` API (sync, common/full time range, muxed export) plus a CLI, with all media I/O on torchcodec ≥0.13, and rewire the GUI's offset/export paths to it.

**Architecture:** `core/` holds pure logic (`offsets`, `timerange`, window math) and thin orchestrators (`sync`, `export`) that cross the boundary with plain `progress`/`is_cancelled` callbacks. `io/decode.py` and `io/encode.py` wrap torchcodec `AudioDecoder`/`VideoDecoder` and the multi-stream `Encoder`. The GUI keeps framepipe for real-time preview but calls the new core for offsets and export via Qt-adapter workers.

**Tech Stack:** Python 3.10–3.12, torch ≥2.11 (cu128), torchaudio, torchcodec ≥0.13 (cu128), PySide6 (GUI only), pytest.

## Global Constraints

- torchcodec pinned `>=0.13` (multi-stream `Encoder` for A/V muxing); torch/torchaudio `>=2.11,<2.12` on the **cu128** index. CUDA 12.8, Python `>=3.10,<3.13` unchanged.
- ffmpeg conda dep is **kept** (torchcodec's shared-lib backend). **No** ffmpeg CLI / `ffprobe` / `subprocess` calls in new code.
- No Qt import anywhere under `src/clapsync/core/` or `src/clapsync/io/`.
- Offsets are **float seconds** end-to-end; never quantize to whole frames for export. `lag_frames` is display-only.
- Boundary contract: `progress: Callable[[float], None] | None` (fraction 0..1), `is_cancelled: Callable[[], bool] | None`.
- CLAUDE.md / Google style: grouped+alphabetized imports (`__future__`→stdlib→third-party→first-party), `X | None` over `Optional`, Google docstrings on public API, 80-col, f-strings, pure-first.
- Reference track offset is exactly `0.0`.

---

## File Structure

Create:
- `src/clapsync/core/__init__.py` — public surface re-exports
- `src/clapsync/core/offsets.py` — pure offset finders + parabolic refine (from `audio_sync.py`)
- `src/clapsync/core/timerange.py` — `TimeRange`, `common_time_range`, `full_time_range`
- `src/clapsync/core/media.py` — `MediaInfo`, `probe`
- `src/clapsync/core/sync.py` — `compute_sync_offsets`
- `src/clapsync/core/export.py` — `ExportSettings`, `ExportResult`, window math, `export_tracks`, `sync_and_trim`
- `src/clapsync/io/decode.py` — torchcodec decode wrappers
- `src/clapsync/io/encode.py` — torchcodec `Encoder` wrapper
- `src/clapsync/cli.py` — argparse entry (`sync`, `synctrim`)
- `src/clapsync/gui/workers.py` — Qt adapters around core
- `tests/conftest.py`, `tests/core/test_*.py`, `tests/integration/test_*.py`

Modify:
- `pyproject.toml` — deps, pytest config, `clapsync-cli` script
- `src/clapsync/gui/export_dialog.py` — widgets only (drop `ExportWorker`)
- `src/clapsync/gui/sync_editor.py` — call `core`/`workers` (moved under gui/)
- `src/clapsync/app.py` — use `workers` adapter

Remove (after callers migrated):
- `src/clapsync/audio_sync.py`, `src/clapsync/offset_worker.py`
- `src/clapsync/io/audio.py`, `src/clapsync/io/ffmpeg.py`
- `src/clapsync/export_dialog.py`, `src/clapsync/sync_editor.py` (moved into gui/)

---

## Task 1: Dependency bump & pytest scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: pytest markers `slow` (needs codecs/GPU); `pixi` env with torch 2.11 + torchcodec 0.13.

- [ ] **Step 1: Edit `pyproject.toml` dependency versions**

In `[tool.pixi.pypi-dependencies]` change the torch lines and add torchcodec:

```toml
torch = { version = ">=2.11.0,<2.12.0", index = "https://download.pytorch.org/whl/cu128" }
torchaudio = { version = ">=2.11.0,<2.12.0", index = "https://download.pytorch.org/whl/cu128" }
torchcodec = { version = ">=0.13.0,<0.14.0", index = "https://download.pytorch.org/whl/cu128" }
```

Add a pytest dev dependency under `[tool.pixi.feature.dev.pypi-dependencies]`:

```toml
pytest = "*"
```

Add the CLI entry point under `[project.scripts]`:

```toml
clapsync-cli = "clapsync.cli:main"
```

Add pytest config at the end of the file:

```toml
[tool.pytest.ini_options]
markers = ["slow: requires torchcodec decode/encode with real codecs"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create test package + shared fixtures**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
"""Shared test fixtures.

Synthetic media are generated with raw torchcodec encoders so decode/probe/
export tests have known-content inputs without checking binaries into git.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch


def _tone(seconds: float, sample_rate: int, freq: float) -> torch.Tensor:
    """Mono sine tone, shape (1, N), float in [-1, 1]."""
    n = int(seconds * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Factory: write a mono tone WAV, return (path, sample_rate)."""
    from torchcodec.encoders import AudioEncoder

    def _make(seconds: float = 1.0, sample_rate: int = 48000,
              freq: float = 440.0, name: str = "tone.wav") -> tuple[Path, int]:
        path = tmp_path / name
        AudioEncoder(_tone(seconds, sample_rate, freq),
                     sample_rate=sample_rate).to_file(str(path))
        return path, sample_rate

    return _make


@pytest.fixture
def rgb_video(tmp_path: Path):
    """Factory: write a solid-color video, return (path, fps, n, w, h)."""
    from torchcodec.encoders import VideoEncoder

    def _make(seconds: float = 1.0, fps: float = 30.0,
              w: int = 64, h: int = 48, name: str = "vid.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 0] = 255  # solid red
        VideoEncoder(frames, frame_rate=fps).to_file(str(path), codec="libx264")
        return path, fps, n, w, h

    return _make
```

- [ ] **Step 3: Install the environment**

Run: `pixi install`
Expected: resolves torch 2.11 + torchcodec 0.13 (cu128). Heavy download; may take minutes.

- [ ] **Step 4: Verify torchcodec imports**

Run: `pixi run python -c "import torch, torchcodec; from torchcodec.encoders import Encoder, AudioEncoder, VideoEncoder; from torchcodec.decoders import AudioDecoder, VideoDecoder; print(torch.__version__, torchcodec.__version__)"`
Expected: prints `2.11.x 0.13.x`, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "build: bump torch>=2.11, add torchcodec>=0.13 and pytest scaffold"
```

---

## Task 2: `core/timerange.py` (pure)

**Files:**
- Create: `src/clapsync/core/__init__.py` (empty for now), `src/clapsync/core/timerange.py`
- Test: `tests/core/test_timerange.py`, `tests/core/__init__.py`

**Interfaces:**
- Produces:
  - `TimeRange(start: float, end: float)` with `.duration -> float`
  - `common_time_range(durations: list[float], offsets: list[float]) -> TimeRange`
  - `full_time_range(durations: list[float], offsets: list[float]) -> TimeRange`
  - (durations/offsets are parallel lists; each track spans `[offset, offset+duration]` on the shared timeline.)

- [ ] **Step 1: Write the failing test**

`tests/core/__init__.py`: empty. `tests/core/test_timerange.py`:

```python
from clapsync.core.timerange import (
    TimeRange, common_time_range, full_time_range,
)


def test_duration():
    assert TimeRange(1.0, 3.5).duration == 2.5


def test_full_range_is_union():
    # track A: [0, 10], track B: [2, 14]
    r = full_time_range(durations=[10.0, 12.0], offsets=[0.0, 2.0])
    assert r.start == 0.0 and r.end == 14.0


def test_common_range_is_intersection():
    # overlap is [2, 10]
    r = common_time_range(durations=[10.0, 12.0], offsets=[0.0, 2.0])
    assert r.start == 2.0 and r.end == 10.0


def test_common_range_empty_when_disjoint_is_zero_length():
    # A: [0,2], B: [5,7]  -> no overlap -> zero-length at the gap
    r = common_time_range(durations=[2.0, 2.0], offsets=[0.0, 5.0])
    assert r.duration == 0.0
    assert r.start == r.end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/core/test_timerange.py -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.core.timerange`.

- [ ] **Step 3: Write minimal implementation**

`src/clapsync/core/__init__.py`: empty file (populated in Task 10).

`src/clapsync/core/timerange.py`:

```python
"""Shared-timeline range math over aligned tracks (pure)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeRange:
    """A closed interval on the shared timeline, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def full_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Union: earliest track start to latest track end.

    Args:
        durations: Per-track source durations in seconds.
        offsets: Per-track offset on the shared timeline in seconds.

    Returns:
        The smallest range covering every track.
    """
    starts = offsets
    ends = [o + d for o, d in zip(offsets, durations)]
    return TimeRange(min(starts), max(ends))


def common_time_range(durations: list[float], offsets: list[float]) -> TimeRange:
    """Intersection: the region where every track has footage.

    When tracks do not all overlap the result is zero-length (start == end).

    Args:
        durations: Per-track source durations in seconds.
        offsets: Per-track offset on the shared timeline in seconds.

    Returns:
        The overlap range, clamped so end >= start.
    """
    start = max(offsets)
    end = min(o + d for o, d in zip(offsets, durations))
    if end < start:
        end = start
    return TimeRange(start, end)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/core/test_timerange.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/__init__.py src/clapsync/core/timerange.py tests/core
git commit -m "feat(core): add TimeRange with common/full range math"
```

---

## Task 3: `core/offsets.py` — move + parabolic refine (pure)

**Files:**
- Create: `src/clapsync/core/offsets.py` (content from `src/clapsync/audio_sync.py`)
- Test: `tests/core/test_offsets.py`
- (Do NOT delete `audio_sync.py` yet — removed in Task 12 once no importer remains.)

**Interfaces:**
- Consumes: nothing (pure, torch/numpy only).
- Produces:
  - `Method = Literal["mfcc", "envelope"]`, `Refine = Literal["none", "parabolic"]`
  - `find_offset(ref_waveform, ref_rate, waveform, rate, fps, method="mfcc", refine="parabolic", **mfcc_kwargs) -> tuple[int, float]` returning `(lag_frames, lag_seconds)`.
  - `_parabolic_peak(corr: np.ndarray, peak: int) -> float` (fractional peak index).

- [ ] **Step 1: Write the failing test**

`tests/core/test_offsets.py`:

```python
import numpy as np
import torch

from clapsync.core.offsets import find_offset, _parabolic_peak


def _click_track(n: int, click_at: int, sr: int) -> torch.Tensor:
    """Silence with a single sharp click sample, shape (1, n)."""
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_parabolic_peak_interpolates_between_samples():
    # Symmetric-ish parabola peaking between index 1 and 2.
    corr = np.array([0.0, 1.0, 1.2, 0.3])
    frac = _parabolic_peak(corr, 2)
    assert 1.5 < frac < 2.5
    assert frac != 2.0  # actually refined off the integer grid


def test_parabolic_peak_clamps_at_edges():
    corr = np.array([5.0, 1.0, 0.0])
    assert _parabolic_peak(corr, 0) == 0.0


def test_find_offset_recovers_known_lag_envelope():
    sr, fps = 48000, 25.0
    n = sr * 2
    ref = _click_track(n, click_at=sr, sr=sr)            # click at 1.000 s
    sub = _click_track(n, click_at=sr + int(0.20 * sr), sr=sr)  # +200 ms
    lag_frames, lag_s = find_offset(ref, sr, sub, sr, fps, method="envelope")
    assert abs(lag_s - 0.20) < 1.0 / fps


def test_mfcc_subframe_offset_parabolic_beats_none():
    sr, fps = 48000, 25.0
    n = sr * 2
    true_lag = 0.0123  # 12.3 ms, well under one 25 fps frame (40 ms)
    ref = _click_track(n, click_at=sr, sr=sr)
    sub = _click_track(n, click_at=sr + int(true_lag * sr), sr=sr)
    _, lag_par = find_offset(ref, sr, sub, sr, fps, method="mfcc", refine="parabolic")
    _, lag_none = find_offset(ref, sr, sub, sr, fps, method="mfcc", refine="none")
    # Parabolic lands closer to the true subframe lag than raw hop-grid argmax.
    assert abs(lag_par - true_lag) <= abs(lag_none - true_lag) + 1e-9
    assert abs(lag_par - true_lag) < 0.005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/core/test_offsets.py -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.core.offsets`.

- [ ] **Step 3: Create `core/offsets.py` from `audio_sync.py` and add parabolic refine**

Copy the entire current contents of `src/clapsync/audio_sync.py` into `src/clapsync/core/offsets.py`, then apply these edits:

(a) Add the `Refine` alias next to `Method`:

```python
Method = Literal["mfcc", "envelope"]
Refine = Literal["none", "parabolic"]
```

(b) Add the pure helper (place it in the "Shared helpers" section under `_to_mono_f64`):

```python
def _parabolic_peak(corr: np.ndarray, peak: int) -> float:
    """Refine an integer correlation peak to sub-sample precision.

    Fits a parabola through corr[peak-1:peak+2] and returns the fractional
    index of its vertex. Clamps to the integer peak at array edges or when the
    three points are colinear.

    Args:
        corr: 1D correlation array.
        peak: Index of the integer-grid maximum.

    Returns:
        Fractional peak index in [peak-0.5, peak+0.5].
    """
    if peak <= 0 or peak >= len(corr) - 1:
        return float(peak)
    y0, y1, y2 = float(corr[peak - 1]), float(corr[peak]), float(corr[peak + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(peak)
    return peak + 0.5 * (y0 - y2) / denom
```

(c) In `find_offset_envelope`, add a `refine: Refine = "parabolic"` parameter (after `fps`) and replace the peak/lag block:

```python
    peak_idx = int(np.argmax(corr))
    peak = _parabolic_peak(corr, peak_idx) if refine == "parabolic" else float(peak_idx)

    # Zero-lag is at index len(env_sub) due to the prepended zero in _fft_correlate
    lag_frac = peak - len(env_sub)
    lag_seconds = lag_frac / fps
    lag_frames = round(lag_frac)
```

Update the `logger.debug` line to log `lag_frames` and `lag_seconds` (drop the old `lag_frames` integer-only formatting if it references removed locals).

(d) In `find_offset_mfcc`, add `refine: Refine = "parabolic"` (after `fps`, before `n_mfcc`) and replace the peak/lag block:

```python
    peak_idx = int(np.argmax(corr))
    peak = _parabolic_peak(corr, peak_idx) if refine == "parabolic" else float(peak_idx)

    # Zero-lag is at index T_sub - 1 (see _mfcc_cross_correlate docstring)
    lag_hops = peak - (mfcc_sub.shape[1] - 1)
    lag_seconds = lag_hops * hop_length / ref_rate
    lag_frames = round(lag_seconds * fps)
```

Update its `logger.debug` call to use the new float `lag_hops`/`lag_seconds`.

(e) In `find_offset`, add `refine: Refine = "parabolic"` (after `method`) and forward it to both branches:

```python
    if method == "mfcc":
        return find_offset_mfcc(
            ref_waveform, ref_rate, waveform, rate, fps,
            refine=refine,
            n_mfcc=n_mfcc, n_fft=n_fft, hop_duration=hop_duration,
            win_duration=win_duration, n_mels=n_mels, mel_scale=mel_scale,
        )
    if method == "envelope":
        return find_offset_envelope(
            ref_waveform, ref_rate, waveform, rate, fps, refine=refine,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/core/test_offsets.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/offsets.py tests/core/test_offsets.py
git commit -m "feat(core): offsets module with parabolic sub-hop peak refinement"
```

---

## Task 4: `io/encode.py` — torchcodec Encoder wrapper

**Files:**
- Create: `src/clapsync/io/encode.py`
- Test: `tests/integration/test_encode.py`, `tests/integration/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pick_video_codec(device: str) -> str` (pure): `"h264_nvenc"` for cuda else `"libx264"`.
  - `encode_clip(out_path: Path, video_frames: torch.Tensor | None, video_fps: float | None, audio_samples: torch.Tensor | None, sample_rate: int | None, *, video_codec: str | None = None, crf: int = 18, device: str = "cpu") -> None` — muxes one file. `video_frames` is `(N, C, H, W)` uint8; `audio_samples` is `(C, M)` float.

- [ ] **Step 1: Write the failing test**

`tests/integration/__init__.py`: empty. `tests/integration/test_encode.py`:

```python
import pytest
import torch

from clapsync.io.encode import pick_video_codec, encode_clip


def test_pick_video_codec_is_pure():
    assert pick_video_codec("cpu") == "libx264"
    assert pick_video_codec("cuda") == "h264_nvenc"


@pytest.mark.slow
def test_encode_clip_muxes_audio_and_video(tmp_path):
    out = tmp_path / "clip.mp4"
    frames = torch.zeros((15, 3, 48, 64), dtype=torch.uint8)
    frames[:, 1] = 200  # green
    samples = torch.zeros((1, 24000), dtype=torch.float32)  # 0.5 s @ 48k
    encode_clip(out, frames, 30.0, samples, 48000, device="cpu")
    assert out.exists() and out.stat().st_size > 0

    from torchcodec.decoders import VideoDecoder, AudioDecoder
    assert VideoDecoder(str(out)).metadata.num_frames >= 14
    assert AudioDecoder(str(out)).metadata.sample_rate == 48000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/integration/test_encode.py::test_pick_video_codec_is_pure -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.io.encode`.

- [ ] **Step 3: Write the implementation**

`src/clapsync/io/encode.py`:

```python
"""Muxed audio+video encoding via the torchcodec multi-stream Encoder."""
from __future__ import annotations

from pathlib import Path

import torch
from torchcodec.encoders import Encoder


def pick_video_codec(device: str) -> str:
    """Return the H.264 encoder name for a device ("cuda" -> NVENC)."""
    return "h264_nvenc" if str(device).startswith("cuda") else "libx264"


def encode_clip(
    out_path: Path,
    video_frames: torch.Tensor | None,
    video_fps: float | None,
    audio_samples: torch.Tensor | None,
    sample_rate: int | None,
    *,
    video_codec: str | None = None,
    crf: int = 18,
    device: str = "cpu",
) -> None:
    """Encode and mux one clip to a single container file.

    Args:
        out_path: Destination path; container inferred from suffix (.mp4).
        video_frames: (N, C, H, W) uint8 frames, or None for audio-only.
        video_fps: Output frame rate; required when video_frames is given.
        audio_samples: (C, M) float samples in [-1, 1], or None for video-only.
        sample_rate: Audio sample rate; required when audio_samples is given.
        video_codec: Override encoder name; defaults per device.
        crf: Constant rate factor for the video stream.
        device: "cpu" or "cuda"; frames are moved here before encoding.
    """
    encoder = Encoder()
    vstream = None
    astream = None

    if video_frames is not None:
        if video_fps is None:
            raise ValueError("video_fps is required when video_frames is given")
        _, channels, height, width = video_frames.shape
        codec = video_codec or pick_video_codec(device)
        vstream = encoder.add_video(
            height=height, width=width, frame_rate=video_fps,
            device=device, codec=codec, crf=crf,
        )

    if audio_samples is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio_samples is given")
        astream = encoder.add_audio(
            sample_rate=sample_rate, num_channels=audio_samples.shape[0],
        )

    if vstream is None and astream is None:
        raise ValueError("encode_clip needs at least one stream")

    with encoder.open_file(str(out_path)):
        if vstream is not None:
            frames = video_frames.to(device) if device != "cpu" else video_frames
            vstream.add_frames(frames)
        if astream is not None:
            astream.add_samples(audio_samples)
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/integration/test_encode.py -v`
Expected: `test_pick_video_codec_is_pure` passes; the `slow` test passes if libx264 is available (`pixi run pytest -m slow` to include it).

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/io/encode.py tests/integration
git commit -m "feat(io): torchcodec muxed audio+video encode_clip"
```

---

## Task 5: `io/decode.py` — torchcodec decode wrappers

**Files:**
- Create: `src/clapsync/io/decode.py`
- Test: `tests/integration/test_decode.py`

**Interfaces:**
- Consumes: torchcodec decoders; `tone_wav`/`rgb_video` fixtures.
- Produces:
  - `load_audio(path: Path, target_rate: int | None = None) -> tuple[torch.Tensor, int]` — mono `(1, N)` float32, peak-normalized, plus sample rate.
  - `decode_frames_at(path: Path, seconds: list[float], device: str = "cpu") -> torch.Tensor` — `(len(seconds), C, H, W)` uint8, one frame per requested time (clamped to stream bounds).

- [ ] **Step 1: Write the failing test**

`tests/integration/test_decode.py`:

```python
import pytest
import torch

from clapsync.io.decode import load_audio, decode_frames_at


@pytest.mark.slow
def test_load_audio_returns_mono_normalized(tone_wav):
    path, sr = tone_wav(seconds=0.5, sample_rate=48000)
    wav, rate = load_audio(path)
    assert rate == 48000
    assert wav.shape[0] == 1
    assert wav.abs().max() <= 1.0 + 1e-6
    assert wav.shape[1] > 0


@pytest.mark.slow
def test_load_audio_resamples(tone_wav):
    path, _ = tone_wav(seconds=0.5, sample_rate=48000)
    wav, rate = load_audio(path, target_rate=16000)
    assert rate == 16000
    assert abs(wav.shape[1] - 8000) < 200


@pytest.mark.slow
def test_decode_frames_at_returns_requested_count(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    frames = decode_frames_at(path, [0.0, 0.5, 0.9])
    assert frames.shape == (3, 3, h, w)
    assert frames.dtype == torch.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/integration/test_decode.py -m slow -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.io.decode`.

- [ ] **Step 3: Write the implementation**

`src/clapsync/io/decode.py`:

```python
"""Media decode wrappers over torchcodec (audio waveforms, video frames)."""
from __future__ import annotations

from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder


def load_audio(
    path: Path, target_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Load a peak-normalized mono waveform from any audio/video file.

    Args:
        path: Source media path.
        target_rate: If set, decode/resample to this sample rate.

    Returns:
        (waveform, sample_rate) where waveform is float32 shape (1, N).
    """
    decoder = AudioDecoder(str(path), sample_rate=target_rate)
    samples = decoder.get_all_samples()
    data = samples.data  # (num_channels, N) float32 in [-1, 1]
    rate = samples.sample_rate

    mono = data.mean(dim=0, keepdim=True) if data.shape[0] > 1 else data
    peak = mono.abs().max()
    if peak > 0:
        mono = mono / peak
    return mono.contiguous(), int(rate)


def decode_frames_at(
    path: Path, seconds: list[float], device: str = "cpu"
) -> torch.Tensor:
    """Decode one frame per requested timestamp (NCHW uint8).

    Timestamps are clamped into the decodable stream range so callers can pass
    boundary times without raising.

    Args:
        path: Source video path.
        seconds: Presentation timestamps in seconds.
        device: "cpu" or "cuda" (nvdec).

    Returns:
        uint8 tensor of shape (len(seconds), C, H, W) on the requested device.
    """
    decoder = VideoDecoder(str(path), device=device)
    meta = decoder.metadata
    lo = meta.begin_stream_seconds or 0.0
    hi = meta.end_stream_seconds
    eps = 1e-3
    clamped = [
        min(max(s, lo), (hi - eps) if hi is not None else s) for s in seconds
    ]
    return decoder.get_frames_played_at(clamped).data
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/integration/test_decode.py -m slow -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/io/decode.py tests/integration/test_decode.py
git commit -m "feat(io): torchcodec load_audio and decode_frames_at"
```

---

## Task 6: `core/media.py` — MediaInfo & probe

**Files:**
- Create: `src/clapsync/core/media.py`
- Test: `tests/integration/test_media.py`

**Interfaces:**
- Consumes: torchcodec decoders + metadata.
- Produces:
  - `MediaInfo(path, duration, has_audio, kind, sample_rate=None, width=None, height=None, fps=None)` (frozen).
  - `probe(path: Path) -> MediaInfo`.

- [ ] **Step 1: Write the failing test**

`tests/integration/test_media.py`:

```python
import pytest

from clapsync.core.media import probe, MediaInfo


@pytest.mark.slow
def test_probe_audio_only(tone_wav):
    path, sr = tone_wav(seconds=0.5, sample_rate=48000)
    info = probe(path)
    assert isinstance(info, MediaInfo)
    assert info.kind == "audio"
    assert info.has_audio is True
    assert info.sample_rate == 48000
    assert info.fps is None
    assert abs(info.duration - 0.5) < 0.1


@pytest.mark.slow
def test_probe_video(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    info = probe(path)
    assert info.kind == "video"
    assert info.width == w and info.height == h
    assert abs(info.fps - 30.0) < 0.5
    assert abs(info.duration - 1.0) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/integration/test_media.py -m slow -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.core.media`.

- [ ] **Step 3: Write the implementation**

`src/clapsync/core/media.py`:

```python
"""Media probing: unify audio-only and video files behind MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torchcodec.decoders import AudioDecoder, VideoDecoder


@dataclass(frozen=True)
class MediaInfo:
    """Stream metadata for one input file."""

    path: Path
    duration: float
    has_audio: bool
    kind: Literal["audio", "video"]
    sample_rate: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None


def _audio_meta(path: Path) -> tuple[bool, int | None, float | None]:
    """Return (has_audio, sample_rate, duration) or (False, None, None)."""
    try:
        meta = AudioDecoder(str(path)).metadata
    except Exception:
        return False, None, None
    return True, int(meta.sample_rate), meta.duration_seconds


def probe(path: Path) -> MediaInfo:
    """Probe a media file via torchcodec metadata.

    A file with a decodable video stream is kind="video"; otherwise it is
    treated as audio-only.

    Args:
        path: Source media path.

    Returns:
        A populated MediaInfo.

    Raises:
        ValueError: If the file has neither a video nor an audio stream.
    """
    path = Path(path)
    has_audio, sample_rate, audio_dur = _audio_meta(path)

    try:
        vmeta = VideoDecoder(str(path)).metadata
    except Exception:
        vmeta = None

    if vmeta is not None:
        return MediaInfo(
            path=path,
            duration=vmeta.duration_seconds,
            has_audio=has_audio,
            kind="video",
            sample_rate=sample_rate,
            width=vmeta.width,
            height=vmeta.height,
            fps=vmeta.average_fps,
        )

    if has_audio:
        return MediaInfo(
            path=path,
            duration=audio_dur or 0.0,
            has_audio=True,
            kind="audio",
            sample_rate=sample_rate,
        )

    raise ValueError(f"No decodable audio or video stream in {path}")
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/integration/test_media.py -m slow -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/media.py tests/integration/test_media.py
git commit -m "feat(core): MediaInfo + torchcodec probe (audio/video)"
```

---

## Task 7: `core/sync.py` — compute_sync_offsets

**Files:**
- Create: `src/clapsync/core/sync.py`
- Test: `tests/core/test_sync.py`

**Interfaces:**
- Consumes: `MediaInfo` (Task 6); `load_audio` (Task 5); `find_offset` (Task 3).
- Produces: `compute_sync_offsets(media: list[MediaInfo], reference_index: int = 0, method="mfcc", refine="parabolic", progress=None, is_cancelled=None) -> list[float]`.

- [ ] **Step 1: Write the failing test (uses monkeypatch — no real media)**

`tests/core/test_sync.py`:

```python
import torch

import clapsync.core.sync as sync_mod
from clapsync.core.media import MediaInfo
from clapsync.core.sync import compute_sync_offsets


def _info(name, dur, fps=25.0):
    return MediaInfo(path=name, duration=dur, has_audio=True,
                     kind="video", sample_rate=48000, width=8, height=8, fps=fps)


def test_reference_offset_is_zero_and_others_signed(monkeypatch):
    media = [_info("a", 5.0), _info("b", 5.0), _info("c", 5.0)]
    # Record which track's audio is being loaded so find_offset can map lags.
    current = {"name": None}

    def fake_load(path, target_rate=None):
        current["name"] = path
        return torch.zeros(1, 100), 48000

    lags = {"b": 0.5, "c": -0.3}  # b lags ref +0.5 s, c leads -0.3 s

    def fake_find(rw, rr, w, r, fps, method, refine):
        return 0, lags[current["name"]]

    monkeypatch.setattr(sync_mod, "load_audio", fake_load)
    monkeypatch.setattr(sync_mod, "find_offset", fake_find)

    offsets = compute_sync_offsets(media)
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 0.5) < 1e-9
    assert abs(offsets[2] + 0.3) < 1e-9


def test_progress_reaches_one(monkeypatch):
    media = [_info("a", 1.0), _info("b", 1.0)]
    monkeypatch.setattr(sync_mod, "load_audio",
                        lambda path, target_rate=None: (torch.zeros(1, 10), 48000))
    monkeypatch.setattr(sync_mod, "find_offset", lambda *a, **k: (0, 0.1))
    seen = []
    compute_sync_offsets(media, progress=seen.append)
    assert seen and seen[-1] == 1.0
```

> The first test threads the loaded track name through `fake_load` so
> `fake_find` returns a per-track lag — verifies mapping without real audio.
> `find_offset` is called with positional `(rw, rr, w, r, fps)` + keyword
> `method`/`refine`, matching `compute_sync_offsets`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/core/test_sync.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` for `clapsync.core.sync`.

- [ ] **Step 3: Write the implementation**

`src/clapsync/core/sync.py`:

```python
"""Compute per-track sync offsets on a shared timeline (audio-based)."""
from __future__ import annotations

import logging
from typing import Callable

from clapsync.core.media import MediaInfo
from clapsync.core.offsets import Method, Refine, find_offset
from clapsync.io.decode import load_audio

logger = logging.getLogger(__name__)


def compute_sync_offsets(
    media: list[MediaInfo],
    reference_index: int = 0,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Align every track to a reference by audio cross-correlation.

    Args:
        media: Probed tracks; each must have an audio stream to be aligned.
        reference_index: Track whose timeline is the origin (offset 0).
        method: Offset finder ("mfcc" or "envelope").
        refine: Peak refinement ("parabolic" or "none").
        progress: Optional 0..1 progress callback.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0. Positive
        means the track starts after the reference. Tracks that fail to load or
        correlate get 0.0.
    """
    n = len(media)
    ref_fps = media[reference_index].fps or 30.0

    ref_wave, ref_rate = load_audio(media[reference_index].path)
    if progress is not None:
        progress(1.0 / n)

    lags: list[float] = [0.0] * n
    for i, info in enumerate(media):
        if is_cancelled is not None and is_cancelled():
            break
        if i == reference_index:
            continue
        try:
            wave, rate = load_audio(info.path)
            _, lag_s = find_offset(
                ref_wave, ref_rate, wave, rate, ref_fps,
                method=method, refine=refine,
            )
        except Exception as exc:  # noqa: BLE001 — one bad track must not abort all
            logger.warning("offset failed for %s: %s — using 0.0", info.path, exc)
            lag_s = 0.0
        lags[i] = lag_s
        if progress is not None:
            progress((i + 1) / n)

    if progress is not None:
        progress(1.0)
    return lags
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/core/test_sync.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/sync.py tests/core/test_sync.py
git commit -m "feat(core): compute_sync_offsets over MediaInfo tracks"
```

---

## Task 8: `core/export.py` — window math (pure)

**Files:**
- Create: `src/clapsync/core/export.py` (window helpers + dataclasses only in this task)
- Test: `tests/core/test_export_math.py`

**Interfaces:**
- Consumes: `TimeRange` (Task 2).
- Produces:
  - `ExportSettings` and `ExportResult` dataclasses (fields per spec).
  - `clip_window(offset: float, duration: float, trim: TimeRange) -> tuple[float, float, float, float]` returning `(local_start, local_end, pad_start, pad_end)`.
  - `frame_source_times(offset: float, trim: TimeRange, out_fps: float) -> list[float]` — per-output-frame source timestamps (may be negative / past-duration; caller pads).

- [ ] **Step 1: Write the failing test**

`tests/core/test_export_math.py`:

```python
from clapsync.core.timerange import TimeRange
from clapsync.core.export import clip_window, frame_source_times, ExportSettings
from pathlib import Path


def test_clip_window_interior():
    # offset 2.0, duration 10.0, trim [3, 8] on shared timeline
    ls, le, ps, pe = clip_window(2.0, 10.0, TimeRange(3.0, 8.0))
    assert (ls, le, ps, pe) == (1.0, 6.0, 0.0, 0.0)


def test_clip_window_leading_gap_pads_start():
    # track starts at offset 5 but trim starts at 3 -> 2 s of leading pad
    ls, le, ps, pe = clip_window(5.0, 10.0, TimeRange(3.0, 8.0))
    assert ls == 0.0
    assert ps == 2.0


def test_clip_window_trailing_gap_pads_end():
    # track [0,4], trim [1,6] -> 2 s trailing pad, local_end clamps to 4
    ls, le, ps, pe = clip_window(0.0, 4.0, TimeRange(1.0, 6.0))
    assert le == 4.0
    assert pe == 2.0


def test_frame_source_times_grid():
    # trim 0.4 s @ 25 fps => 10 frames; offset 0 => source times 0.00..0.36
    times = frame_source_times(0.0, TimeRange(0.0, 0.4), 25.0)
    assert len(times) == 10
    assert abs(times[0] - 0.0) < 1e-9
    assert abs(times[1] - 0.04) < 1e-9


def test_frame_source_times_shifted_by_offset():
    # offset 0.5 subtracts from every source time (track starts later)
    times = frame_source_times(0.5, TimeRange(0.0, 0.08), 25.0)
    assert abs(times[0] + 0.5) < 1e-9  # -0.5 -> before track start (caller pads)


def test_export_settings_defaults():
    s = ExportSettings(trim=TimeRange(0.0, 1.0), output_dir=Path("."))
    assert s.crf == 18
    assert s.video_codec is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/core/test_export_math.py -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.core.export`.

- [ ] **Step 3: Write the dataclasses + pure helpers**

`src/clapsync/core/export.py`:

```python
"""Trim/pad export of synced tracks (window math is pure; I/O is separate)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clapsync.core.timerange import TimeRange


@dataclass(frozen=True)
class ExportSettings:
    """Output settings for a sync/trim export."""

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
    """Outcome for one exported track."""

    path: Path
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def clip_window(
    offset: float, duration: float, trim: TimeRange
) -> tuple[float, float, float, float]:
    """Map a shared-timeline trim onto one track's local source range.

    Args:
        offset: Track offset on the shared timeline (seconds).
        duration: Track source duration (seconds).
        trim: Desired output range on the shared timeline.

    Returns:
        (local_start, local_end, pad_start, pad_end) in seconds. local_* index
        into the source; pad_* are black/silence gaps outside the source.
    """
    local_start = max(0.0, trim.start - offset)
    local_end = min(duration, trim.end - offset)
    pad_start = max(0.0, offset - trim.start)
    pad_end = max(0.0, trim.end - (offset + duration))
    return local_start, local_end, pad_start, pad_end


def frame_source_times(
    offset: float, trim: TimeRange, out_fps: float
) -> list[float]:
    """Source timestamps for each output frame on the shared trim grid.

    Output frame k sits at shared time trim.start + k / out_fps; the matching
    source time subtracts the track offset. Values outside [0, duration) mark
    frames the caller must black-pad. This keeps subframe offsets exact — the
    output grid is fixed, the source is sampled at the precise shifted time.

    Args:
        offset: Track offset on the shared timeline (seconds).
        trim: Output range on the shared timeline.
        out_fps: Output frame rate.

    Returns:
        One source timestamp per output frame.
    """
    n_frames = round(trim.duration * out_fps)
    return [trim.start + k / out_fps - offset for k in range(n_frames)]
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/core/test_export_math.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/export.py tests/core/test_export_math.py
git commit -m "feat(core): export window + frame-grid math (pure)"
```

---

## Task 9: `core/export.py` — export_tracks & sync_and_trim

**Files:**
- Modify: `src/clapsync/core/export.py` (append orchestration)
- Test: `tests/integration/test_export.py`

**Interfaces:**
- Consumes: `probe` (T6), `compute_sync_offsets` (T7), `common_time_range`/`full_time_range` (T2), `clip_window`/`frame_source_times` (T8), `load_audio`/`decode_frames_at` (T5), `encode_clip` (T4).
- Produces:
  - `export_tracks(media, offsets, settings, progress=None, is_cancelled=None) -> list[ExportResult]`
  - `sync_and_trim(paths, output_dir, *, method="mfcc", refine="parabolic", trim="common", reference_index=0, progress=None, is_cancelled=None) -> list[ExportResult]`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_export.py`:

```python
import pytest

from clapsync.core.export import sync_and_trim, ExportResult


@pytest.mark.slow
def test_sync_and_trim_two_videos_roundtrip(rgb_video, tmp_path):
    a, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, name="a.mp4")
    b, *_ = rgb_video(seconds=1.0, fps=30.0, name="b.mp4")
    out = tmp_path / "out"
    out.mkdir()

    results = sync_and_trim([a, b], out, trim="common")
    assert len(results) == 2
    assert all(isinstance(r, ExportResult) for r in results)
    assert all(r.ok for r in results), [r.error for r in results]
    for r in results:
        assert r.path.exists() and r.path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/integration/test_export.py -m slow -v`
Expected: FAIL with `ImportError: cannot import name 'sync_and_trim'`.

- [ ] **Step 3: Append the orchestration to `core/export.py`**

Add these imports at the top of `src/clapsync/core/export.py` (keep grouping):

```python
import logging
from typing import Callable, Literal

import torch

from clapsync.core.media import MediaInfo, probe
from clapsync.core.offsets import Method, Refine
from clapsync.core.sync import compute_sync_offsets
from clapsync.core.timerange import common_time_range, full_time_range
from clapsync.io.decode import decode_frames_at, load_audio
from clapsync.io.encode import encode_clip
```

Add `logger = logging.getLogger(__name__)` after the imports, then append:

```python
def _build_video_frames(
    info: MediaInfo, offset: float, trim: TimeRange, out_fps: float,
    out_w: int, out_h: int,
) -> torch.Tensor:
    """Assemble the output frame tensor on the trim grid, black-padding gaps.

    Frames are assembled on CPU (uniform device for stacking); encode_clip moves
    the batch to the encode device.
    """
    times = frame_source_times(offset, trim, out_fps)
    in_range = [0.0 <= t < info.duration for t in times]
    sampled = {}
    wanted = [t for t, ok in zip(times, in_range) if ok]
    if wanted:
        decoded = decode_frames_at(info.path, wanted, device="cpu")
        for t, frame in zip(wanted, decoded):
            sampled[t] = frame

    black = torch.zeros((3, out_h, out_w), dtype=torch.uint8)
    frames = []
    for t, ok in zip(times, in_range):
        if not ok:
            frames.append(black)
            continue
        frame = sampled[t]
        if frame.shape[-1] != out_w or frame.shape[-2] != out_h:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0).float(), size=(out_h, out_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0).clamp(0, 255).to(torch.uint8)
        frames.append(frame)
    return torch.stack(frames)


def _build_audio_samples(
    info: MediaInfo, offset: float, trim: TimeRange,
) -> tuple[torch.Tensor, int]:
    """Trim + silence-pad the track audio to the shared trim range."""
    wave, rate = load_audio(info.path)
    local_start, local_end, pad_start, pad_end = clip_window(
        offset, info.duration, trim,
    )
    s0 = max(0, round(local_start * rate))
    s1 = min(wave.shape[1], round(local_end * rate))
    core = wave[:, s0:s1] if s1 > s0 else wave[:, :0]
    total = round(trim.duration * rate)
    pad_front = round(pad_start * rate)
    out = torch.zeros((core.shape[0], total), dtype=core.dtype)
    end = min(total, pad_front + core.shape[1])
    out[:, pad_front:end] = core[:, : end - pad_front]
    return out, rate


def export_tracks(
    media: list[MediaInfo],
    offsets: list[float],
    settings: ExportSettings,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Trim+pad every track to settings.trim and encode one file each.

    Video tracks produce a muxed A/V MP4; audio-only tracks produce an audio
    file. Video and audio share the exact subframe trim origin.

    Args:
        media: Probed tracks.
        offsets: Per-track shared-timeline offsets (seconds).
        settings: Output resolution/fps/codec/dir.
        progress: Optional 0..1 callback.
        is_cancelled: Optional cancel check.

    Returns:
        One ExportResult per track (in input order).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trim = settings.trim
    results: list[ExportResult] = []
    n = len(media)

    for i, (info, offset) in enumerate(zip(media, offsets)):
        if is_cancelled is not None and is_cancelled():
            break
        try:
            if info.kind == "video":
                out_fps = settings.output_fps or info.fps or 30.0
                out_w = settings.target_width or info.width
                out_h = settings.target_height or info.height
                frames = _build_video_frames(
                    info, offset, trim, out_fps, out_w, out_h,
                )
                audio = None
                rate = None
                if info.has_audio:
                    audio, rate = _build_audio_samples(info, offset, trim)
                ext = "mp4"
                out_path = settings.output_dir / f"{info.path.stem}_synced.{ext}"
                encode_clip(
                    out_path, frames, out_fps, audio, rate,
                    video_codec=settings.video_codec, crf=settings.crf,
                    device=device,
                )
            else:
                audio, rate = _build_audio_samples(info, offset, trim)
                ext = settings.audio_format or info.path.suffix.lstrip(".") or "wav"
                out_path = settings.output_dir / f"{info.path.stem}_synced.{ext}"
                encode_clip(
                    out_path, None, None, audio, rate, device="cpu",
                )
            results.append(ExportResult(out_path))
        except Exception as exc:  # noqa: BLE001 — record per-track failure
            logger.exception("export failed for %s", info.path)
            results.append(ExportResult(
                settings.output_dir / f"{info.path.stem}_synced", str(exc),
            ))
        if progress is not None:
            progress((i + 1) / n)

    return results


def sync_and_trim(
    paths: list[Path],
    output_dir: Path,
    *,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    trim: Literal["common", "full"] = "common",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[ExportResult]:
    """Probe, sync, pick a trim range, and export — the one-call convenience.

    Args:
        paths: Input media files.
        output_dir: Destination directory (created if missing).
        method: Offset finder.
        refine: Peak refinement.
        trim: "common" (overlap) or "full" (union) output range.
        reference_index: Reference track.
        progress: Optional 0..1 callback (spans sync then export).
        is_cancelled: Optional cancel check.

    Returns:
        One ExportResult per input.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    media = [probe(p) for p in paths]

    def sync_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 * f)

    offsets = compute_sync_offsets(
        media, reference_index=reference_index, method=method, refine=refine,
        progress=sync_progress, is_cancelled=is_cancelled,
    )

    durations = [m.duration for m in media]
    rng = (common_time_range if trim == "common" else full_time_range)(
        durations, offsets,
    )
    settings = ExportSettings(trim=rng, output_dir=output_dir)

    def export_progress(f: float) -> None:
        if progress is not None:
            progress(0.5 + 0.5 * f)

    return export_tracks(
        media, offsets, settings,
        progress=export_progress, is_cancelled=is_cancelled,
    )
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/integration/test_export.py -m slow -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/export.py tests/integration/test_export.py
git commit -m "feat(core): export_tracks + sync_and_trim (subframe-exact, muxed)"
```

---

## Task 10: `core/__init__.py` — public surface

**Files:**
- Modify: `src/clapsync/core/__init__.py`
- Test: `tests/core/test_public_api.py`

**Interfaces:**
- Produces: importable names on `clapsync.core`.

- [ ] **Step 1: Write the failing test**

`tests/core/test_public_api.py`:

```python
def test_public_surface_importable():
    from clapsync.core import (
        MediaInfo, probe,
        TimeRange, common_time_range, full_time_range,
        compute_sync_offsets,
        ExportSettings, ExportResult, export_tracks, sync_and_trim,
        find_offset,
    )
    assert callable(sync_and_trim)
    assert callable(find_offset)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/core/test_public_api.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `core/__init__.py`**

```python
"""clapsync headless API: sync, time ranges, and muxed trim/export."""
from clapsync.core.export import (
    ExportResult,
    ExportSettings,
    export_tracks,
    sync_and_trim,
)
from clapsync.core.media import MediaInfo, probe
from clapsync.core.offsets import Method, Refine, find_offset
from clapsync.core.sync import compute_sync_offsets
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "MediaInfo",
    "probe",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "compute_sync_offsets",
    "ExportSettings",
    "ExportResult",
    "export_tracks",
    "sync_and_trim",
    "find_offset",
    "Method",
    "Refine",
]
```

- [ ] **Step 4: Run tests + full suite**

Run: `pixi run pytest -v` (fast) and `pixi run pytest -m slow -v` (integration)
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/__init__.py tests/core/test_public_api.py
git commit -m "feat(core): public API surface"
```

---

## Task 11: `cli.py` — headless CLI

**Files:**
- Create: `src/clapsync/cli.py`
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: `probe`, `compute_sync_offsets`, `common_time_range`, `full_time_range`, `sync_and_trim`.
- Produces: `main(argv: list[str] | None = None) -> int`. Subcommands `sync` (print offsets + ranges) and `synctrim` (run sync_and_trim).

- [ ] **Step 1: Write the failing test**

`tests/integration/test_cli.py`:

```python
import pytest

from clapsync.cli import main


@pytest.mark.slow
def test_cli_sync_prints_offsets(rgb_video, capsys):
    a, *_ = rgb_video(seconds=1.0, name="a.mp4")
    b, *_ = rgb_video(seconds=1.0, name="b.mp4")
    rc = main(["sync", str(a), str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "offset" in out.lower()


@pytest.mark.slow
def test_cli_synctrim_writes_files(rgb_video, tmp_path):
    a, *_ = rgb_video(seconds=1.0, name="a.mp4")
    b, *_ = rgb_video(seconds=1.0, name="b.mp4")
    out = tmp_path / "out"
    rc = main(["synctrim", str(a), str(b), "-o", str(out)])
    assert rc == 0
    assert list(out.glob("*_synced.mp4"))


def test_cli_no_command_returns_error(capsys):
    rc = main([])
    assert rc == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/integration/test_cli.py::test_cli_no_command_returns_error -v`
Expected: FAIL with `ModuleNotFoundError: clapsync.cli`.

- [ ] **Step 3: Write the implementation**

`src/clapsync/cli.py`:

```python
"""Headless CLI for clapsync: sync and synctrim subcommands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clapsync.core import (
    common_time_range,
    compute_sync_offsets,
    full_time_range,
    probe,
    sync_and_trim,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", type=Path, help="Media files")
    p.add_argument("--method", choices=["mfcc", "envelope"], default="mfcc")
    p.add_argument("--refine", choices=["none", "parabolic"], default="parabolic")
    p.add_argument("--reference", type=int, default=0, help="Reference index")


def _cmd_sync(args: argparse.Namespace) -> int:
    media = [probe(p) for p in args.inputs]
    offsets = compute_sync_offsets(
        media, reference_index=args.reference,
        method=args.method, refine=args.refine,
        progress=lambda f: print(f"\rsync {f*100:3.0f}%", end="", file=sys.stderr),
    )
    print(file=sys.stderr)
    durations = [m.duration for m in media]
    common = common_time_range(durations, offsets)
    full = full_time_range(durations, offsets)
    for info, off in zip(media, offsets):
        print(f"{info.path.name}\toffset={off:+.4f}s")
    print(f"common range: {common.start:.4f}..{common.end:.4f}s "
          f"({common.duration:.4f}s)")
    print(f"full range:   {full.start:.4f}..{full.end:.4f}s "
          f"({full.duration:.4f}s)")
    return 0


def _cmd_synctrim(args: argparse.Namespace) -> int:
    results = sync_and_trim(
        args.inputs, args.output,
        method=args.method, refine=args.refine, trim=args.trim,
        reference_index=args.reference,
        progress=lambda f: print(f"\r{f*100:3.0f}%", end="", file=sys.stderr),
    )
    print(file=sys.stderr)
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK " if r.ok else f"ERR {r.error}"
        print(f"{status}\t{r.path}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
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
```

- [ ] **Step 4: Run tests**

Run: `pixi run pytest tests/integration/test_cli.py -v` (add `-m slow` to include the export ones)
Expected: fast test passes; slow tests pass with codecs.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/cli.py tests/integration/test_cli.py
git commit -m "feat(cli): sync and synctrim subcommands"
```

---

## Task 12: Rewire GUI to core; remove dead modules

**Files:**
- Create: `src/clapsync/gui/workers.py`
- Move: `src/clapsync/sync_editor.py` → `src/clapsync/gui/sync_editor.py`
- Modify: `src/clapsync/gui/export_dialog.py` (was `src/clapsync/export_dialog.py`; move + strip worker), `src/clapsync/app.py`
- Remove: `src/clapsync/audio_sync.py`, `src/clapsync/offset_worker.py`, `src/clapsync/io/audio.py`, `src/clapsync/io/ffmpeg.py`, `src/clapsync/export_dialog.py`, `src/clapsync/sync_editor.py`

**Interfaces:**
- Consumes: `core.compute_sync_offsets`, `core.probe`, `core.ExportSettings`, `core.export_tracks`, `core.common_time_range`.
- Produces:
  - `gui/workers.py`: `OffsetWorker(QObject)` with `progress_value(int)`, `finished(list)`, `failed(str)` signals + `run()`; `compute_offsets_with_progress(paths, parent=None) -> list[float] | None`; `ExportWorker(QObject)` wrapping `export_tracks`.

- [ ] **Step 1: Create `gui/workers.py` (Qt adapters around core)**

```python
"""Qt worker adapters that run headless core operations off the GUI thread."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from clapsync.core import (
    ExportResult,
    ExportSettings,
    MediaInfo,
    compute_sync_offsets,
    export_tracks,
    probe,
)

logger = logging.getLogger(__name__)


class OffsetWorker(QObject):
    progress_value = Signal(int)   # 0..1000
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths

    @Slot()
    def run(self) -> None:
        try:
            media = [probe(p) for p in self._paths]
            offsets = compute_sync_offsets(
                media, progress=lambda f: self.progress_value.emit(int(f * 1000)),
            )
            self.finished.emit(offsets)
        except Exception as exc:  # noqa: BLE001
            logger.exception("offset worker failed")
            self.failed.emit(str(exc))


def compute_offsets_with_progress(
    paths: list[Path], parent=None
) -> list[float] | None:
    """Run OffsetWorker on a QThread behind a modal progress dialog."""
    dialog = QProgressDialog("Analyzing audio…", "Cancel", 0, 1000, parent)
    dialog.setWindowTitle("Computing Offsets")
    dialog.setMinimumWidth(420)
    dialog.setValue(0)
    dialog.show()

    result: list[float] | None = None
    error: str | None = None

    worker = OffsetWorker(paths)
    thread = QThread()
    worker.moveToThread(thread)

    def on_value(v: int) -> None:
        dialog.setValue(v)

    def on_finished(offsets: list) -> None:
        nonlocal result
        result = offsets
        dialog.setValue(1000)
        thread.quit()

    def on_failed(msg: str) -> None:
        nonlocal error
        error = msg
        thread.quit()

    worker.progress_value.connect(on_value)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    thread.started.connect(worker.run)
    thread.start()

    while thread.isRunning():
        QApplication.processEvents()
        if dialog.wasCanceled():
            thread.requestInterruption()
            thread.quit()
            thread.wait(3000)
            return None
    thread.wait()

    if error is not None:
        QMessageBox.critical(parent, "Error", f"Failed to compute offsets:\n{error}")
        return None
    return result


class ExportWorker(QObject):
    progress = Signal(float)
    status = Signal(str)
    finished = Signal(object)   # list[ExportResult]

    def __init__(
        self, media: list[MediaInfo], offsets: list[float],
        settings: ExportSettings,
    ) -> None:
        super().__init__()
        self._media = media
        self._offsets = offsets
        self._settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        def status(i: int) -> None:
            self.status.emit(f"Exporting {i + 1}/{len(self._media)}")

        results = export_tracks(
            self._media, self._offsets, self._settings,
            progress=lambda f: self.progress.emit(f),
            is_cancelled=lambda: self._cancelled,
        )
        self.finished.emit(results)
```

- [ ] **Step 2: Move + strip `export_dialog.py`**

Run:

```bash
git mv src/clapsync/export_dialog.py src/clapsync/gui/export_dialog.py
```

In `src/clapsync/gui/export_dialog.py` delete the `ExportResult` class and the entire `ExportWorker` class (now provided by `workers.py` / `core`). Keep only `_fmt` and `ExportDialog`. Change the import of `VideoInfo` from framepipe to `from clapsync.core import MediaInfo` and rename the type hints `list[VideoInfo]` → `list[MediaInfo]` (field access `.width/.height/.fps/.path/.duration` is identical on `MediaInfo`). Remove the now-unused `from clapsync.io.ffmpeg import export_synced_video` import.

- [ ] **Step 3: Move `sync_editor.py` and rewire imports**

Run:

```bash
git mv src/clapsync/sync_editor.py src/clapsync/gui/sync_editor.py
```

In `src/clapsync/gui/sync_editor.py`:

(a) Replace the export-related imports:

```python
from clapsync.core import ExportSettings, ExportResult, common_time_range
from clapsync.gui.export_dialog import ExportDialog
from clapsync.gui.workers import ExportWorker
```

(b) Keep `from framepipe import VideoInfo, get_video_info, get_display_size` for the preview path (framepipe stays in Project 1). The `_video_infos` list still comes from framepipe `get_video_info` for preview.

(c) In `_on_export`, build settings from the dialog and pass core types. Replace the `ExportWorker(...)` construction block with:

```python
target_width, target_height, output_fps, output_dir = dialog.get_export_params()
if not output_dir.exists():
    QMessageBox.warning(self, "Export", f"Directory does not exist: {output_dir}")
    return

media = [probe(info.path) for info in self._video_infos]
settings = ExportSettings(
    trim=TimeRange(self._trim_start, self._trim_end),
    output_dir=output_dir,
    target_width=target_width,
    target_height=target_height,
    output_fps=output_fps,
)
worker = ExportWorker(media, self._offsets, settings)
```

Add imports `from clapsync.core import probe, TimeRange` at the top. The existing progress-dialog wiring and `on_finished(results)` handler are unchanged (results are still `list[ExportResult]` with `.ok/.path/.error`).

- [ ] **Step 4: Update `app.py`**

In `src/clapsync/app.py` change the two imports:

```python
from clapsync.gui.sync_editor import SyncEditorWindow
from clapsync.gui.workers import compute_offsets_with_progress
```

(The rest of `main()` is unchanged.)

- [ ] **Step 5: Remove dead modules**

```bash
git rm src/clapsync/audio_sync.py src/clapsync/offset_worker.py \
       src/clapsync/io/audio.py src/clapsync/io/ffmpeg.py
```

- [ ] **Step 6: Verify no dangling imports**

Run: `pixi run python -c "import clapsync.app, clapsync.gui.sync_editor, clapsync.gui.export_dialog, clapsync.gui.workers, clapsync.cli"`
Expected: no ImportError.

Run: `pixi run pytest -v`
Expected: all fast tests pass.

- [ ] **Step 7: Smoke-test the GUI entry (manual)**

Run: `pixi run clapsync`
Expected: selection dialog opens; picking clips computes offsets (via new core) and opens the editor; Export writes muxed MP4s. (Preview still uses framepipe — expected in Project 1.)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(gui): rewire offsets/export to headless core; remove ffmpeg/subprocess modules"
```

---

## Self-Review Notes

- **Spec coverage:** find sync (T3/T7), common+full range (T2), auto sync+trim (T9), CLI sync/synctrim (T11), torchcodec decode/probe/encode (T4–T6), muxed A/V (T4/T9), subframe correctness (T3 parabolic + T8 grid + T9 build), GUI decoupling (T12), deps (T1). All requirements mapped.
- **framepipe:** intentionally retained for GUI preview only (Project 2 removes it); `VideoInfo`/`get_video_info`/`get_display_size` still imported in `sync_editor.py`.
- **Audio format:** `ExportSettings.audio_format` honored for audio-only tracks (T9); video tracks always `.mp4`.
- **Cancellation:** cooperative via `is_cancelled` across sync + export; GUI `ExportWorker.cancel()` sets the flag.
