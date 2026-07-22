# Audio Track Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let clapsync sync/preview/export any mix of video and audio files (video+audio, audio+audio, video+video), with audible per-track-mute preview, waveform cells for audio tracks, per-track export selection, and audio export parameters (spec: `docs/superpowers/specs/2026-07-21-audio-track-support-design.md`).

**Architecture:** The backend already treats every input as "a thing with an audio stream" (`MediaInfo.kind`, audio sync, audio-only export). All gaps are in the GUI plus two small backend export knobs. New GUI pieces: a `WaveformWidget` (static waveform + playhead) for audio cells, an `audio_engine.py` mixing unmuted tracks through `QAudioSink`, a per-row mute toggle on the timeline, and export-dialog track selection + audio params. `SyncEditorWindow` maps video tracks to a subset of mosaic cells and treats the audio engine as a fallback playback clock.

**Tech Stack:** Python 3, PySide6 (incl. QtMultimedia — QAudioSink, verified present in the pixi env), PyAV, numpy, torch, pytest via pixi.

## Global Constraints

- Supported audio output formats: **wav, mp3, flac, m4a, ogg** (+ `.aac` accepted as input). No opus/aiff/wma.
- Codec map additions: `.ogg → libvorbis`, `.aac → aac`. "Same as source" with an unmapped suffix → **wav** output.
- Preview mixes to **mono**; export keeps source channels. Preview waveform uses the **16 kHz** cached sync waveform; playback engine uses **48 kHz**, stored **int16** in RAM.
- Mix of unmuted tracks = **mean** (no clipping). All muted → engine pushes **silence** (stays a valid clock).
- Drift re-sync threshold: **100 ms** (video position is master when video frames flow).
- Audio export params apply to **audio-only outputs**; video (mp4) keeps aac. `ExportSettings` gains `audio_sample_rate: int | None = None`, `audio_bitrate: int | None = None` (bps).
- Bitrate choices 128/192/256/320 kbps (default 320), enabled only for lossy formats (mp3, m4a, ogg). Sample rate: Native / 44100 / 48000.
- Test runner: `pixi run python -m pytest <path> -v` from repo root `C:/Users/Charles/PycharmProjects/clapsync`. Windows/Git-Bash, forward slashes.
- Branch `feat/audio-tracks` (already created from `feat/sync-confidence @ a920d5e`); commit after each task. Do NOT create/switch branches.
- Known pre-existing flake (ignore): `tests/gui/test_playback_perf.py::test_playback_sustains_30fps[1080p]` (GPU-perf benchmark). Deselect it in full-suite runs.

---

### Task 1: Backend audio export params (encode + settings)

**Files:**
- Modify: `src/clapsync/app/encode.py`
- Modify: `src/clapsync/app/export.py`
- Test: `tests/app/test_encode.py`, `tests/app/test_export.py`

**Interfaces:**
- Produces:
  - `_AUDIO_CODEC_FOR_SUFFIX` gains `.ogg → "libvorbis"`, `.aac → "aac"`.
  - `_audio_codec_for(out_path)` unchanged signature; unmapped suffix still returns `_DEFAULT_AUDIO_CODEC` (used by the mp4 mux path). A NEW helper `resolve_audio_output(stem, fmt)` — see below — decides the audio-only output path/extension and applies the wav fallback.
  - `encode_clip(..., audio_bitrate: int | None = None)` — sets `astream.bit_rate` for lossy codecs.
  - `ExportSettings` gains `audio_sample_rate: int | None = None`, `audio_bitrate: int | None = None`.
  - `export_media`'s audio-only branch honors `audio_format` (already present), `audio_sample_rate`, `audio_bitrate`.

- [ ] **Step 1: Write failing tests (encode)**

Append to `tests/app/test_encode.py`:

```python
from clapsync.app.encode import _audio_codec_for


def test_audio_codec_map_covers_supported_formats():
    from pathlib import Path
    assert _audio_codec_for(Path("x.wav")) == "pcm_s16le"
    assert _audio_codec_for(Path("x.flac")) == "flac"
    assert _audio_codec_for(Path("x.mp3")) == "libmp3lame"
    assert _audio_codec_for(Path("x.m4a")) == "aac"
    assert _audio_codec_for(Path("x.ogg")) == "libvorbis"
    assert _audio_codec_for(Path("x.aac")) == "aac"


def test_encode_audio_only_ogg_roundtrips(tmp_path):
    import av
    out = tmp_path / "a.ogg"
    encode_clip(out, None, None, _tone(1.0), 48000)
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.name in ("vorbis", "libvorbis")


def test_encode_audio_only_sets_bitrate_for_lossy(tmp_path):
    import av
    out = tmp_path / "a.mp3"
    encode_clip(out, None, None, _tone(1.0), 48000, audio_bitrate=192000)
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        # libmp3lame honors the requested bit_rate on the stream.
        assert a.codec_context.bit_rate in (192000, pytest.approx(192000, abs=32000))
```

- [ ] **Step 2: Run, confirm fail**

Run: `pixi run python -m pytest tests/app/test_encode.py -k "codec_map or ogg or bitrate" -v`
Expected: FAIL (ogg not in map → falls back to aac in an ogg container / bitrate param missing).

- [ ] **Step 3: Implement encode changes**

In `src/clapsync/app/encode.py`:

```python
_AUDIO_CODEC_FOR_SUFFIX = {
    ".wav": "pcm_s16le",
    ".m4a": "aac",
    ".mp4": "aac",
    ".mkv": "aac",
    ".mov": "aac",
    ".mp3": "libmp3lame",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".aac": "aac",
}
```

Add lossy-codec set and a bitrate param. Change `encode_clip`'s signature to add `audio_bitrate: int | None = None` (keyword, after `crf`), and where the audio stream is created:

```python
        if audio_samples is not None:
            layout = "mono" if audio_samples.shape[0] == 1 else "stereo"
            acodec = _audio_codec_for(out_path)
            astream = container.add_stream(acodec, rate=sample_rate, layout=layout)
            if audio_bitrate is not None and acodec in _LOSSY_AUDIO_CODECS:
                astream.bit_rate = audio_bitrate
```

with, near the codec map:

```python
# Codecs that take a target bit rate; pcm/flac are lossless and ignore it.
_LOSSY_AUDIO_CODECS = {"aac", "libmp3lame", "libvorbis"}
```

Add a resolver used by the export layer (keeps the wav-fallback rule in one place):

```python
def resolve_audio_output(stem: str, fmt: str | None, src_suffix: str) -> Path:
    """Pick the audio-only output filename for a track.

    fmt is the user's chosen extension (without dot), or None for
    "same as source". An unmapped/unknown result falls back to wav rather
    than muxing an arbitrary codec into an unsupported container.
    """
    ext = (fmt or src_suffix.lstrip(".") or "wav").lower()
    if f".{ext}" not in _AUDIO_CODEC_FOR_SUFFIX:
        ext = "wav"
    return Path(f"{stem}_synced.{ext}")
```

(Import `Path` in encode.py if not already imported.)

- [ ] **Step 4: Wire export_media (export.py)**

Add the two fields to `ExportSettings`:

```python
    audio_format: str | None = None
    audio_sample_rate: int | None = None
    audio_bitrate: int | None = None
```

In `export_media`'s audio-only branch, replace the ext/encode block:

```python
            else:
                rate = settings.audio_sample_rate
                audio, src_rate = _build_audio_samples(info, offset, trim)
                out_rate = rate or src_rate
                from clapsync.app.encode import resolve_audio_output
                out_path = settings.output_dir / resolve_audio_output(
                    info.path.stem, settings.audio_format, info.path.suffix
                ).name
                encode_clip(
                    out_path, None, None, audio, out_rate,
                    audio_bitrate=settings.audio_bitrate, device="cpu",
                )
```

Note: `encode_clip` creates the audio stream at `out_rate`; the existing `_mux_audio_samples` `AudioResampler` resamples the samples (decoded at `src_rate`) to the stream rate. Verify: the resampler in `_mux_audio_samples` is constructed with `rate=rate` where `rate` is the stream rate passed to `encode_clip` — confirm this argument threads `out_rate`, not `src_rate`. If `_mux_audio_samples` currently assumes in==out rate, pass `out_rate` explicitly and let it resample.

- [ ] **Step 5: Write failing test (export sample-rate)**

Append to `tests/app/test_export.py` (uses the `tone_wav` fixture):

```python
def test_export_audio_only_resamples_and_picks_format(tone_wav, tmp_path):
    from clapsync.app.export import export_tracks, ExportSettings
    from clapsync.core import common_time_range
    import av
    a, sr = tone_wav(seconds=1.0, sample_rate=48000, name="a.wav")
    b, _ = tone_wav(seconds=1.0, sample_rate=48000, freq=660.0, name="b.wav")
    offsets = [0.0, 0.0]
    durations = [1.0, 1.0]
    out = tmp_path / "out"
    out.mkdir()
    settings = ExportSettings(
        trim=common_time_range(durations, offsets), output_dir=out,
        audio_format="flac", audio_sample_rate=44100,
    )
    results = export_tracks([a, b], offsets, settings)
    assert all(r.ok for r in results), [r.error for r in results]
    outs = sorted(out.glob("*_synced.flac"))
    assert len(outs) == 2
    with av.open(str(outs[0])) as c:
        assert c.streams.audio[0].codec_context.sample_rate == 44100
```

- [ ] **Step 6: Run all backend audio tests, confirm pass**

Run: `pixi run python -m pytest tests/app/test_encode.py tests/app/test_export.py -v -m "not slow"`
Expected: PASS. Fix `_mux_audio_samples` rate threading if the 44100 assertion fails.

- [ ] **Step 7: Commit**

```bash
git add src/clapsync/app/encode.py src/clapsync/app/export.py tests/app/test_encode.py tests/app/test_export.py
git commit -m "feat(export): audio format/sample-rate/bitrate params + ogg/aac codecs"
```

---

### Task 2: Media selection dialog (accept audio)

**Files:**
- Rename: `src/clapsync/gui/video_selection_dialog.py` → `src/clapsync/gui/media_selection_dialog.py`
- Modify: `src/clapsync/gui/app.py`
- Test: none (Qt dialog; manual — but keep a public `get_video_paths` alias for compatibility)

**Interfaces:**
- Produces: `MediaSelectionDialog` with method `get_media_paths() -> list[Path]` (keep `get_video_paths = get_media_paths` alias so nothing else breaks). File filter includes audio.

- [ ] **Step 1: Rename the file and class**

```bash
git mv src/clapsync/gui/video_selection_dialog.py src/clapsync/gui/media_selection_dialog.py
```

In `media_selection_dialog.py`: rename `class VideoSelectionDialog` → `class MediaSelectionDialog`; window title `"clapsync — Select Media"`; label `"Select two or more media files (video or audio) to synchronize:"`; button `"Add Videos…"` → `"Add Media…"`; file dialog title `"Select Media Files"` and filter:

```python
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Media Files",
            "",
            "Media Files (*.mp4 *.mov *.avi *.mkv *.mts *.m2ts *.webm *.flv *.wmv "
            "*.wav *.mp3 *.flac *.m4a *.aac *.ogg);;All Files (*)",
        )
```

Rename `get_video_paths` → `get_media_paths`, and add at the end of the class body:

```python
    # Back-compat alias.
    get_video_paths = get_media_paths
```

- [ ] **Step 2: Update the import site**

In `src/clapsync/gui/app.py`:

```python
from clapsync.gui.media_selection_dialog import MediaSelectionDialog
```

and `sel = MediaSelectionDialog()` / `video_paths = sel.get_media_paths()`.

- [ ] **Step 3: Import-check**

Run: `pixi run python -c "import clapsync.gui.app; from clapsync.gui.media_selection_dialog import MediaSelectionDialog"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add -A src/clapsync/gui/
git commit -m "feat(gui): media selection dialog accepts audio files"
```

---

### Task 3: Waveform downsampling (pure fn) + WaveformWidget

**Files:**
- Create: `src/clapsync/gui/waveform_widget.py`
- Test: `tests/gui/test_waveform.py`

**Interfaces:**
- Produces:
  - `downsample_peaks(samples: np.ndarray, buckets: int) -> np.ndarray` — shape `(buckets, 2)` of (min, max) per bucket; pure, unit-tested.
  - `WaveformWidget(QWidget)` with `set_waveform(samples: np.ndarray)`, `set_window(offset_s, duration_s, trim_start, trim_end)` (for x-mapping), `set_playhead(global_s)`.

- [ ] **Step 1: Write failing test (pure fn)**

Create `tests/gui/test_waveform.py`:

```python
import numpy as np
from clapsync.gui.waveform_widget import downsample_peaks


def test_downsample_peaks_shape_and_extremes():
    x = np.linspace(-1.0, 1.0, 1000).astype(np.float32)
    peaks = downsample_peaks(x, 10)
    assert peaks.shape == (10, 2)
    # first bucket min is near -1, last bucket max near +1
    assert peaks[0, 0] <= -0.9
    assert peaks[-1, 1] >= 0.9
    # min <= max in every bucket
    assert np.all(peaks[:, 0] <= peaks[:, 1])


def test_downsample_peaks_handles_short_input():
    x = np.array([0.5, -0.5], dtype=np.float32)
    peaks = downsample_peaks(x, 10)
    assert peaks.shape == (10, 2)
    assert np.isfinite(peaks).all()


def test_downsample_peaks_empty_is_zeros():
    peaks = downsample_peaks(np.zeros(0, dtype=np.float32), 5)
    assert peaks.shape == (5, 2)
    assert np.all(peaks == 0.0)
```

- [ ] **Step 2: Run, confirm fail**

Run: `pixi run python -m pytest tests/gui/test_waveform.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/clapsync/gui/waveform_widget.py`:

```python
"""Static waveform display for an audio track's mosaic cell."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


def downsample_peaks(samples: np.ndarray, buckets: int) -> np.ndarray:
    """Reduce a mono waveform to per-pixel (min, max) peak pairs.

    Returns an array of shape (buckets, 2). Buckets beyond the sample count
    (or an empty input) are zero. Cheap enough to recompute on resize.
    """
    out = np.zeros((buckets, 2), dtype=np.float32)
    n = samples.shape[0]
    if n == 0 or buckets <= 0:
        return out
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    for i in range(buckets):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg = samples[lo:hi]
        out[i, 0] = float(seg.min())
        out[i, 1] = float(seg.max())
    return out


class WaveformWidget(QWidget):
    """Paints a static waveform with a moving playhead. Display only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 90)
        self.setStyleSheet("background-color: #111;")
        self._samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._peaks: np.ndarray | None = None
        self._off = 0.0
        self._dur = 0.0
        self._trim0 = 0.0
        self._trim1 = 0.0
        self._playhead = 0.0

    def set_waveform(self, samples: np.ndarray) -> None:
        self._samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._peaks = None
        self.update()

    def set_window(self, offset_s: float, duration_s: float,
                   trim_start: float, trim_end: float) -> None:
        self._off, self._dur = offset_s, duration_s
        self._trim0, self._trim1 = trim_start, trim_end
        self.update()

    def set_playhead(self, global_s: float) -> None:
        self._playhead = global_s
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._peaks = None  # re-bucket to the new width

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        mid = h / 2
        if self._peaks is None or self._peaks.shape[0] != w:
            self._peaks = downsample_peaks(self._samples, max(1, w))
        p.setPen(QPen(QColor("#4A90D9"), 1))
        for x in range(self._peaks.shape[0]):
            ymin = mid - self._peaks[x, 1] * mid
            ymax = mid - self._peaks[x, 0] * mid
            p.drawLine(x, int(ymin), x, int(ymax))
        # Playhead within the track's own [offset, offset+duration] span,
        # mapped across the full widget width.
        if self._dur > 0:
            frac = (self._playhead - self._off) / self._dur
            if 0.0 <= frac <= 1.0:
                px = int(frac * w)
                p.setPen(QPen(QColor("#D93025"), 2))
                p.drawLine(px, 0, px, h)
        p.end()
```

- [ ] **Step 4: Run, confirm pass**

Run: `pixi run python -m pytest tests/gui/test_waveform.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/gui/waveform_widget.py tests/gui/test_waveform.py
git commit -m "feat(gui): WaveformWidget + downsample_peaks for audio cells"
```

---

### Task 4: Audio mix (pure fn) + audio engine

**Files:**
- Create: `src/clapsync/gui/audio_engine.py`
- Test: `tests/gui/test_audio_mix.py`

**Interfaces:**
- Produces:
  - `mix_block(tracks: list[np.ndarray], offsets: list[float], muted: list[bool], start_s: float, n: int, rate: int) -> np.ndarray` — pure. Returns `(n,)` float32 mono in [-1,1], the mean of unmuted tracks sliced at their offset-shifted position; zeros where a track has no samples. All muted / no tracks → zeros.
  - `AudioEngine(QObject)` — `set_tracks(waveforms, offsets)`, `set_muted(list[bool])`, `play()`, `pause()`, `seek(global_s)`, `position_s() -> float`, signal `position_changed(float)`. Uses `QAudioSink` at 48 kHz mono int16. `enabled` False if no output device.

- [ ] **Step 1: Write failing test (pure fn)**

Create `tests/gui/test_audio_mix.py`:

```python
import numpy as np
from clapsync.gui.audio_engine import mix_block


def _ramp(n, val):
    return np.full(n, val, dtype=np.float32)


def test_mix_mean_of_unmuted():
    rate = 100
    a = _ramp(200, 1.0)
    b = _ramp(200, 0.0)
    out = mix_block([a, b], [0.0, 0.0], [False, False], 0.0, 50, rate)
    assert out.shape == (50,)
    assert np.allclose(out, 0.5)


def test_mix_skips_muted():
    rate = 100
    a = _ramp(200, 1.0)
    b = _ramp(200, 0.0)
    out = mix_block([a, b], [0.0, 0.0], [False, True], 0.0, 50, rate)
    assert np.allclose(out, 1.0)  # only a contributes


def test_mix_respects_offset_and_pads_silence():
    rate = 100
    a = _ramp(100, 1.0)  # 1s of ones
    # a starts at offset +1.0s; querying global [0,0.5s) => before a starts => silence
    out = mix_block([a], [1.0], [False], 0.0, 50, rate)
    assert np.allclose(out, 0.0)
    # querying global [1.0,1.5s) => inside a => ones
    out2 = mix_block([a], [1.0], [False], 1.0, 50, rate)
    assert np.allclose(out2, 1.0)


def test_mix_all_muted_is_silence():
    out = mix_block([_ramp(100, 1.0)], [0.0], [True], 0.0, 20, 100)
    assert np.allclose(out, 0.0)
```

- [ ] **Step 2: Run, confirm fail**

Run: `pixi run python -m pytest tests/gui/test_audio_mix.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/clapsync/gui/audio_engine.py`:

```python
"""Audible preview: mixes unmuted tracks through QAudioSink on a wall clock."""
from __future__ import annotations

import logging

import numpy as np
import torch
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal

logger = logging.getLogger(__name__)

_RATE = 48000
_MUTED_FILL = np.float32(0.0)


def mix_block(
    tracks: list[np.ndarray],
    offsets: list[float],
    muted: list[bool],
    start_s: float,
    n: int,
    rate: int,
) -> np.ndarray:
    """Mean-mix n samples of the unmuted tracks starting at global start_s.

    Each track's local index is (global - offset). Out-of-range regions
    contribute silence. Returns (n,) float32 in [-1, 1]; all-muted -> zeros.
    """
    out = np.zeros(n, dtype=np.float32)
    active = 0
    for wave, off, is_muted in zip(tracks, offsets, muted):
        if is_muted:
            continue
        active += 1
        local0 = int(round((start_s - off) * rate))
        seg = np.zeros(n, dtype=np.float32)
        src_lo = max(0, local0)
        src_hi = min(wave.shape[0], local0 + n)
        if src_hi > src_lo:
            dst_lo = src_lo - local0
            seg[dst_lo:dst_lo + (src_hi - src_lo)] = wave[src_lo:src_hi]
        out += seg
    if active > 0:
        out /= active
    return out
```

Then the `AudioEngine` (a `QIODevice`-pull sink driven by `QAudioSink`). Key design: subclass `QIODevice`, override `readData` to serve `mix_block` from a running sample cursor; `QAudioSink.start(device)` pulls. `position_s()` = cursor / rate. `seek` resets the cursor and restarts the sink.

```python
class _MixDevice(QIODevice):
    def __init__(self, engine: "AudioEngine") -> None:
        super().__init__()
        self._engine = engine

    def readData(self, maxlen: int) -> bytes:
        # maxlen is bytes; int16 mono => 2 bytes/sample.
        n = max(0, maxlen // 2)
        if n == 0:
            return b""
        block = self._engine._pull(n)  # float32 (n,)
        i16 = np.clip(block, -1.0, 1.0)
        i16 = (i16 * 32767.0).astype(np.int16)
        return i16.tobytes()

    def writeData(self, data) -> int:  # never written to
        return 0

    def bytesAvailable(self) -> int:
        return 1 << 20  # always claim data available (continuous stream)

    def isSequential(self) -> bool:
        return True


class AudioEngine(QObject):
    position_changed = Signal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[np.ndarray] = []
        self._offsets: list[float] = []
        self._muted: list[bool] = []
        self._cursor = 0            # samples from time 0 on the shared timeline
        self._base = 0              # cursor value at last seek (== _cursor start)
        self.enabled = True
        self._sink = None
        self._dev = None
        try:
            from PySide6.QtMultimedia import (
                QAudioFormat, QAudioSink, QMediaDevices,
            )
            out = QMediaDevices.defaultAudioOutput()
            if out.isNull():
                raise RuntimeError("no default audio output")
            fmt = QAudioFormat()
            fmt.setSampleRate(_RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self._sink = QAudioSink(out, fmt)
            self._dev = _MixDevice(self)
            self._dev.open(QIODevice.OpenModeFlag.ReadOnly)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio preview disabled (%s)", exc)
            self.enabled = False
        # Emit position on a light timer while playing.
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(
            lambda: self.position_changed.emit(self.position_s())
        )

    def set_tracks(self, waveforms: list[torch.Tensor], offsets: list[float]) -> None:
        # Mono float32 numpy at _RATE (waveforms already decoded at 48k).
        self._tracks = [
            w.reshape(-1).to(torch.float32).cpu().numpy() for w in waveforms
        ]
        self._offsets = list(offsets)
        if not self._muted or len(self._muted) != len(self._tracks):
            self._muted = [False] * len(self._tracks)

    def set_muted(self, muted: list[bool]) -> None:
        self._muted = list(muted)

    def set_offsets(self, offsets: list[float]) -> None:
        self._offsets = list(offsets)

    def _pull(self, n: int) -> np.ndarray:
        block = mix_block(
            self._tracks, self._offsets, self._muted,
            self._cursor / _RATE, n, _RATE,
        )
        self._cursor += n
        return block

    def position_s(self) -> float:
        return self._cursor / _RATE

    def seek(self, global_s: float) -> None:
        self._cursor = max(0, int(round(global_s * _RATE)))
        if self.enabled and self._sink is not None and self._sink.state().name != "StoppedState":
            # Restart the pull so the new cursor takes effect immediately.
            self._sink.stop()
            self._sink.start(self._dev)

    def play(self) -> None:
        if not self.enabled or self._sink is None:
            return
        self._sink.start(self._dev)
        self._timer.start()

    def pause(self) -> None:
        if not self.enabled or self._sink is None:
            return
        self._sink.stop()
        self._timer.stop()
```

(Note: `_pull` advances `_cursor` by the pulled sample count. This is the wall-clock model — the sink pulls at real time. Drift correction is the editor's job via `seek`.)

- [ ] **Step 4: Run mix tests, confirm pass; import-check the engine**

Run: `pixi run python -m pytest tests/gui/test_audio_mix.py -v && pixi run python -c "import clapsync.gui.audio_engine"`
Expected: PASS + clean import.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/gui/audio_engine.py tests/gui/test_audio_mix.py
git commit -m "feat(gui): audio mixing engine (QAudioSink) + pure mix_block"
```

---

### Task 5: Timeline mute toggle

**Files:**
- Modify: `src/clapsync/gui/timeline_widget.py`

**Interfaces:**
- Produces: `TrackState.muted: bool = False`; `TimelineWidget.mute_changed = Signal(int, bool)` (track index, muted); a speaker icon painted at the right end of each row; click hit-test toggles it.

- [ ] **Step 1: Add the field + signal**

`TrackState` dataclass gains `muted: bool = False`. On `TimelineWidget` add `mute_changed = Signal(int, bool)`.

- [ ] **Step 2: Paint a speaker/mute icon per row**

In `_draw_tracks`, after the lock icon block, draw a speaker glyph near the row's right edge within the visible viewport. Compute an icon rect in *widget* coordinates (not scrolled with the bar — anchor it to the row's vertical center and a fixed x from the right edge of the content area). Store the last-painted rects in `self._mute_hitboxes: dict[int, QRectF]` for hit-testing. Draw a filled speaker when unmuted, a speaker with a slash when muted (muted uses a dimmer color). Keep it simple — a small triangle + arcs, or reuse a unicode glyph via `painter.drawText` (`"🔊"` / `"🔇"`).

Minimal glyph approach:

```python
        # in __init__: self._mute_hitboxes: dict[int, QRectF] = {}
        # at start of _draw_tracks: self._mute_hitboxes.clear()
        # inside the per-track loop, after drawing label/lock:
        icon = "🔇" if track.muted else "🔊"
        box = QRectF(w - 26, rect.center().y() - 9, 18, 18)
        painter.setPen(QPen(QColor("white"), 1))
        f = painter.font(); f.setPixelSize(14); painter.setFont(f)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, icon)
        self._mute_hitboxes[track.index] = box
```

(`w` is the content width passed into `_draw_tracks`.)

- [ ] **Step 3: Hit-test in mousePressEvent**

At the TOP of `TimelineWidget.mousePressEvent` (base class) AND ensure the `SyncTrimTimelineWidget.mousePressEvent` override calls through: before the trim-handle checks, test the mute hitboxes:

```python
        pos = QPointF(event.position().x(), event.position().y())
        for idx, box in self._mute_hitboxes.items():
            if box.contains(pos):
                track = next(t for t in self._tracks if t.index == idx)
                track.muted = not track.muted
                self.mute_changed.emit(idx, track.muted)
                self.update()
                return
```

Put this shared check in a helper `self._maybe_toggle_mute(event) -> bool` on `TimelineWidget` and call it first in both `mousePressEvent`s (base and `SyncTrimTimelineWidget`), returning early if it handled the click.

- [ ] **Step 4: Import-check + manual note**

Run: `pixi run python -c "import clapsync.gui.timeline_widget"`
Expected: clean. (Visual behavior verified in Task 8 GUI smoke.)

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/gui/timeline_widget.py
git commit -m "feat(gui): per-track mute toggle on the timeline"
```

---

### Task 6: SyncEditorWindow integration

**Files:**
- Modify: `src/clapsync/gui/sync_editor.py`

**Interfaces:**
- Consumes: `WaveformWidget` (Task 3), `AudioEngine` (Task 4), `TrackState.muted`/`mute_changed` (Task 5), `load_audio` (`clapsync.app.decode`).
- Produces: an editor that (a) builds a `WaveformWidget` for audio tracks and a `VideoPlayerWidget` for video tracks, (b) maps video frames to the right cells, (c) drives the audio engine as playback audio + fallback clock, (d) no longer crashes on `None` dimensions.

- [ ] **Step 1: Fix the `%d`-on-None probe log**

In the probe loop, guard the dimensions log:

```python
            if info.kind == "video":
                logger.debug(
                    "probe done: %s  duration=%.2fs  %dx%d",
                    path.name, info.duration, info.width, info.height,
                )
            else:
                logger.debug(
                    "probe done: %s  duration=%.2fs  audio", path.name, info.duration,
                )
```

Also the `total = max(info.duration + off ...)` line is kind-agnostic — leave it.

- [ ] **Step 2: Build audio cells + video-slot map**

In `_build_ui`, when creating cells, branch on `info.kind`: video → `VideoPlayerWidget` (as today, appended to `self._players` AND recorded in `self._video_slots: list[int]` = the track index for that player); audio → `WaveformWidget` (append to a parallel `self._waveforms: dict[int, WaveformWidget]` keyed by track index; load its 16 kHz waveform via `load_audio(info.path, target_rate=16000)` and `set_waveform`). Keep the mosaic grid packing over ALL tracks (video + audio cells interleaved by track order).

Change `_players` from "one per track" to "one per video track", and keep `self._video_slots` parallel to `_players` so `_on_frames_ready` can map `frames[k]` → `self._players[k]`.

- [ ] **Step 3: Only pass video tracks to VideoGroupWorker**

In `_load_all_videos`, build `paths`/`offsets` from video tracks only (`[info.path for info in self._video_infos if info.kind == "video"]` with matching offsets by index). If there are zero video tracks, do NOT start the group worker (`self._group_worker` stays None). `_on_frames_ready` maps by video-slot order.

- [ ] **Step 4: Wire the audio engine**

Construct `self._audio = AudioEngine(self)`. Load each track's 48 kHz waveform (`load_audio(info.path, target_rate=48000)`) and `self._audio.set_tracks(waveforms, self._offsets)`. Connect:
- play/pause: in `_on_play_pause`, also `self._audio.play()` / `self._audio.pause()`.
- seek: in `_sync_seek_all`, also `self._audio.seek(global_s)`.
- offsets changed: `self._audio.set_offsets(offsets)`.
- `self._timeline.mute_changed.connect(lambda idx, m: self._on_mute(idx, m))` where `_on_mute` updates a `self._muted` list and calls `self._audio.set_muted(self._muted)`.

- [ ] **Step 5: Clock + drift + waveform playhead**

- On every position update (`_on_position_changed`), call `wf.set_playhead(global_s)` for each waveform widget.
- Drift: in `_on_frames_ready` (video is master when frames flow), if `abs(video_ts - self._audio.position_s()) > 0.1`, call `self._audio.seek(video_ts)`.
- Zero-video sessions: there is no `frames_ready`; drive the playhead/time-label from `self._audio.position_changed` instead. Connect `self._audio.position_changed` to a slot that updates `self._global_pos`, the timeline playhead, the time label, and applies the loop/stop logic at `trim_end` (mirroring `_on_position_changed`).
- Video EOF with audio extending past: in `_on_eof`, if `self._global_pos < self._trim_end - 0.05` and the audio engine is enabled, do NOT stop — let `position_changed` carry playback to `trim_end`.

- [ ] **Step 6: Stop the engine on close**

In `closeEvent`, `self._audio.pause()` before stopping the group worker.

- [ ] **Step 7: Import-check + non-slow suite**

Run: `pixi run python -c "import clapsync.gui.sync_editor" && pixi run python -m pytest -m "not slow" -q`
Expected: clean import; suite passes (GUI logic untested by unit tests here — validated in Task 8).

- [ ] **Step 8: Commit**

```bash
git add src/clapsync/gui/sync_editor.py
git commit -m "feat(gui): audio tracks in the editor — waveform cells, audible mix, drift resync"
```

---

### Task 7: Export dialog — track selection + audio params

**Files:**
- Modify: `src/clapsync/gui/export_dialog.py`, `src/clapsync/gui/sync_editor.py` (call site)

**Interfaces:**
- Produces: `ExportDialog.get_export_params() -> ExportParams` dataclass with `selected_indices: list[int]`, `target_width`, `target_height`, `output_fps`, `output_dir`, `audio_format: str | None`, `audio_sample_rate: int | None`, `audio_bitrate: int | None`. The editor filters `media`/`offsets` by `selected_indices` and sets the audio fields on `ExportSettings`.

- [ ] **Step 1: Track checkbox list**

Add a `QListWidget` (or column of `QCheckBox`) with one checkable row per track (label = `info.path.name`), all checked by default. Wire `itemChanged`/`stateChanged` to a recompute + OK-enable handler. OK disabled when zero selected.

- [ ] **Step 2: Resolution/fps only over selected video tracks; hide when none**

Move the resolution/fps construction into a method `_recompute_video_options()` that considers only *selected* tracks with `kind == "video"`. If none, hide the resolution + fps rows (and skip the `min(... width*height)` computation that crashes on audio-only). Call it on init and on any checkbox change.

- [ ] **Step 3: Audio params section**

Add three rows: Format (`Same as source`/wav/flac/mp3/m4a/ogg), Sample rate (`Native`/44100/48000), Bitrate (128/192/256/320 kbps, default 320). Enable Bitrate only when the chosen format is lossy (mp3/m4a/ogg) — connect the format combo's `currentIndexChanged` to toggle the bitrate combo's enabled state.

- [ ] **Step 4: Return a dataclass**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExportParams:
    selected_indices: list[int]
    target_width: int | None
    target_height: int | None
    output_fps: float | None
    output_dir: Path
    audio_format: str | None
    audio_sample_rate: int | None
    audio_bitrate: int | None
```

`get_export_params` builds it. `audio_format` is None for "Same as source"; `audio_sample_rate` None for "Native"; `audio_bitrate` None when the format is lossless or bitrate disabled. `target_width/height/output_fps` None when no video track is selected.

- [ ] **Step 5: Update the editor call site**

In `sync_editor.py::_on_export`, replace the tuple unpack with the dataclass; build `selected_media = [media[i] for i in params.selected_indices]` and `selected_offsets = [self._offsets[i] for i in params.selected_indices]`; pass audio fields into `ExportSettings`; hand the filtered lists to `ExportWorker`.

- [ ] **Step 6: Import-check + non-slow suite**

Run: `pixi run python -c "import clapsync.gui.export_dialog, clapsync.gui.sync_editor" && pixi run python -m pytest -m "not slow" -q`
Expected: clean; suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/clapsync/gui/export_dialog.py src/clapsync/gui/sync_editor.py
git commit -m "feat(gui): export dialog track selection + audio format/rate/bitrate"
```

---

### Task 8: Full verification + real-footage acceptance

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `pixi run python -m pytest -q --deselect "tests/gui/test_playback_perf.py::test_playback_sustains_30fps[1080p]"`
Expected: all pass.

- [ ] **Step 2: Backend audio-only sync via CLI (real files)**

Extract an audio track from one Volvo cam (`ffmpeg -i "<cam1>" -vn -ac 1 a1.wav`, into the scratchpad) and sync it against a Volvo video:

Run: `pixi run clapsync-cli sync "<Volvo cam2 mp4>" "<a1.wav>"`
Expected: a sensible offset + confidence, no crash (proves audio+video sync end to end headlessly).

- [ ] **Step 3: GUI smoke — video+audio**

Launch `pixi run clapsync`. Select 2 Volvo videos + one extracted `.wav`. Confirm: dialog accepts the wav; editor opens without crashing; the audio track shows a waveform cell; timeline shows its row with a mute toggle; Play produces audible mixed sound; muting a track silences it; the waveform playhead tracks position; export dialog lists all three tracks as checkboxes with the audio params section.

- [ ] **Step 4: GUI smoke — audio only**

Launch again, select 2+ `.wav` files only. Confirm: no video mosaic crash (zero video tracks → no group worker); playback is driven by the audio engine; export produces audio files in the chosen format.

- [ ] **Step 5: Export check**

From the video+audio session, export with audio format = flac, sample rate = 48000, one track deselected. Confirm the output dir has the expected files (deselected track absent; audio track written as `.flac` at 48 kHz; video track as mp4).

- [ ] **Step 6: Merge decision**

Use superpowers:finishing-a-development-branch.

## Self-review notes

- Spec coverage: media dialog → Task 2; waveform cell → Task 3; video-worker mapping → Task 6 (Step 2-3); audio engine + mix + mute-silence clock → Task 4; drift/EOF clock → Task 6 (Step 5); timeline mute → Task 5; export dialog selection + audio params + hide-res-on-audio-only → Task 7; export backend fields + codec map → Task 1; error handling (QAudioSink disable) → Task 4 (`enabled` flag); testing (pure fns) → Tasks 1/3/4.
- Deviations from spec: (a) `MediaSelectionDialog` keeps a `get_video_paths` alias for a smaller blast radius; (b) mute icon uses a unicode glyph rather than a custom-painted speaker (simpler, same behavior); (c) engine stores 48 kHz waveforms as float32 numpy rather than int16 (the spec's int16 is a memory optimization — noted as a follow-up; revisit if RAM is a problem on long sessions). Flag (c) to the reviewer explicitly.
- Type consistency: `mix_block(tracks, offsets, muted, start_s, n, rate)` and `downsample_peaks(samples, buckets)` signatures identical across their tasks and tests; `ExportSettings.audio_format/audio_sample_rate/audio_bitrate` used identically in Tasks 1 and 7; `ExportParams` fields consumed exactly as produced.
- Known accepted-risk carryovers from spec: mono preview, mute click, seek transient, positional index alignment on probe failure.
```
