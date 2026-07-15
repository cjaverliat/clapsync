# Drop torchcodec — framepipe + PyAV Media I/O — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `torchcodec` dependency from clapsync entirely, serving video decode + video metadata from `framepipe` and everything else (audio decode, audio metadata, muxed A/V encode, test fixtures) from PyAV, with no change to sync offsets or export output.

**Architecture:** `framepipe` covers video only — it has no audio decoder and no encoder. So `probe()` becomes a two-source function (framepipe `extract_video_metadata` for the video stream, PyAV for the audio stream), `decode_frames_at()` moves to framepipe's `PyAvVideoDecoder` + `IndexedFramePrefetcher`, and `load_audio()` / `encode_clip()` / test fixtures move to PyAV. No `ffmpeg` CLI or `ffprobe` subprocess is introduced; PyAV binds the same ffmpeg shared libraries torchcodec was using.

**Tech Stack:** Python 3.10–3.12, torch/torchaudio ≥2.11 (cu130), framepipe (local editable, `../framepipe`), PyAV 18, PySide6 (GUI), pytest, pixi.

## Global Constraints

- **No `torchcodec` import may remain** in `src/`, `tests/`, or `pyproject.toml` when this plan completes.
- **No ffmpeg CLI / `ffprobe` / `subprocess`** in new code. PyAV only.
- `clapsync.core` stays pure — it imports only torch/torchaudio/numpy/stdlib. Importing `clapsync.core` must not import PyAV or framepipe.
- **Behavior parity is the bar:** sync offsets and export byte-content must not shift. Measured tolerances are in each task's verification step.
- Run everything with `pixi run` from `C:/Users/Charles/PycharmProjects/clapsync`.
- Public signatures of `load_audio`, `decode_frames_at`, `encode_clip`, `probe` do **not** change — only their internals. `gui/`, `cli.py`, and `export.py` call sites stay untouched.

## Measured Baseline (verified 2026-07-15, before any change)

These are real numbers from this machine against `E:/Datasets/Guedelon/Carriere Luc/GoPro1/GX010037.MP4` (3.28 GB, 5312×2988, 59.94 fps, 13089 frames, 218.37 s, AAC 48 kHz stereo). Later tasks assert against them.

| Measurement | torchcodec (baseline) | Replacement | Verdict |
|---|---|---|---|
| Audio decode, warm cache | 0.48 s | PyAV 1.75 s | 3.7× slower; cold disk read of the 3.28 GB file is ~19 s, so decode is I/O-dominated. Acceptable. |
| Audio waveform values | — | max abs diff **2.9e-08** | Identical; sync will not move. |
| `decode_frames_at`, 7 timestamps | 41.6 s | framepipe **800.6 s** unseeked / fast when pre-seeked | Needs Task 1. |
| Single late frame (13000) | — | framepipe **766.1 s** unseeked, **1.4 s** pre-seeked; raw PyAV seek 0.98 s | Needs Task 1. |
| Frame pixel values | — | max abs diff **0.0** (all 7 frames) | Bit-identical, including out-of-range clamping. |
| Video metadata read | ~ (approximate seek_mode) | framepipe 0.22 s | Faster. |
| AAC encode round-trip | 48000 → 48128 (**+128**) | PyAV 48000 → 48128 (**+128**) | Exact parity. |
| wav / mp3 / flac round-trip | wav +0 | PyAV +0 / +0 / +0 | Exact. |
| Encoders present | — | `libx264`, `h264_nvenc`, `aac`, `libmp3lame`, `flac`, `pcm_s16le` | All available. |

**Known non-parity to accept:** none identified. AAC padding matches torchcodec exactly.

### Offset baseline — the number the port must reproduce

Measured 2026-07-15 on the six test files, against the current torchcodec code at `edbf535` (`sync took 69.7 s`, probe 5.9 s):

```
offsets: [0.0, -3.7834, 5.9064, 14.6969, 20.0706, 19.567]
```

| File | Duration | Offset (s) |
|---|---|---|
| `GoPro1/GX010037.MP4` | 218.4 s | 0.0 (reference) |
| `GoPro2/GX010035.MP4` | 227.7 s | -3.7834 |
| `GoPro3/GX010035.MP4` | 225.8 s | 5.9064 |
| `GoPro4/GX010036.MP4` | 220.0 s | 14.6969 |
| `GoPro5/GX010034.MP4` | 210.2 s | 20.0706 |
| `GoPro6/GX010038.MP4` | 204.3 s | 19.567 |

All six probe as `video 5312x2988, audio=True, 48000 Hz`. **Task 7 Step 7 must reproduce these to within 1e-3 s.** If they move, the PyAV waveform is not equivalent — stop and diff the waveform, do not adjust the expectation.

### Progress baseline (the feature being preserved)

Same run, per-file tick counts from the `progress` callback: **45, 47, 47, 45, 44, 42** — one per 5 s window. Every file climbed `0.000 -> 1.000` monotonically with 40+ intermediate values, i.e. no 0→100 jump. The PyAV port (Task 2) should roughly double these (~100/file, throttled to 1% steps) and must never regress to one tick per file.

## Prior Art / Context

- `edbf535 feat(sync): fine-grained per-file audio loading progress` (2026-07-15) added per-file 0→100% audio progress by decoding in 5 s windows via torchcodec's `get_samples_played_in_range`. **Task 2 must preserve this behavior** — it is the feature the user asked for. PyAV gets it more directly: progress falls out of frames already being decoded, with no windowing or seeking.
- `docs/superpowers/specs/2026-07-06-headless-api-decoupling-design.md` moved this project *onto* torchcodec and *off* the ffmpeg CLI. This plan reverses the torchcodec half only; the "no ffmpeg CLI" rule from that spec still stands.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `../framepipe/src/framepipe/frame_prefetcher.py` | **Task 1** — `IndexedFramePrefetcher` seeks to its first planned index. |
| `src/clapsync/app/decode.py` | **Tasks 2, 4** — `load_audio` (PyAV), `decode_frames_at` (framepipe). |
| `src/clapsync/app/media.py` | **Task 3** — `probe` via framepipe video meta + PyAV audio meta. |
| `src/clapsync/app/encode.py` | **Task 5** — `encode_clip` via PyAV muxer; container→codec map. |
| `tests/conftest.py` | **Task 6** — `tone_wav` / `rgb_video` / `av_video` fixtures on PyAV. |
| `pyproject.toml`, `README.md` | **Task 7** — drop torchcodec dep + docs. |

---

## Task 1: framepipe — `IndexedFramePrefetcher` seeks to its first index

**Repo:** `C:/Users/Charles/PycharmProjects/framepipe` (separate repo, editable-installed into clapsync's pixi env — changes take effect immediately).

**Why:** `IndexedFramePrefetcher._locate` seeks on *backward* steps but **forward-walks** every batch to reach a target ahead of the decoder — so a plan starting at frame 13000 decodes frames 0..13000 first.

Measured on `GX010037.MP4` (13089 frames, 5312×2988):

| Fetch | Time |
|---|---|
| `IndexedFramePrefetcher(dec, [0])` | 0.31 s |
| `IndexedFramePrefetcher(dec, [13000])` | **766.12 s** |
| Same, decoder pre-seeked via `start_frame=13000` | **1.40 s** |
| `dec.seek_to_index(13000)` + `next_batch()` | 2.03 s |
| Raw PyAV `seek` + decode to the same frame | 0.98 s |

~780× off the raw-seek floor. The decoder itself is fine — only `_locate`'s forward branch needs a threshold.

**Exact defect** — `src/framepipe/frame_prefetcher.py:390`:

```python
if want >= self._decoder.position:
    # Forward walk, dropping unwanted batches until want is in window.
    while True:
        held = self._decoder.next_batch()
        ...
```

The forward branch is unconditional, so an arbitrarily large forward gap is paid one batch at a time. The seek path below it (line 402) is already correct and stays as-is.

**Files:**
- Modify: `src/framepipe/frame_prefetcher.py` (`IndexedFramePrefetcher._locate`, ~line 383-406)
- Modify: `tests/conftest.py` (add a clip long enough to cross the threshold — existing fixtures are 12 frames)
- Test: `tests/test_indexed_prefetcher.py`

**Interfaces:**
- Consumes: `VideoDecoder.seek_to_index(int)`, `VideoDecoder.position` (both already used by `_locate`).
- Produces: no signature change. `IndexedFramePrefetcher(decoder, indices)` becomes fast for plans that start late or jump forward. Docstring claim "other backward steps pay a `seek_to_index`" widens to include large forward steps.

- [ ] **Step 1: Add a long fixture**

Existing fixtures (`cfr_video`, `cfr_offset_video`, `vfr_video`) are 12 frames — too short to cross a 32-frame threshold. Append to `tests/conftest.py`, matching the existing `_ffmpeg` helper style:

```python
@pytest.fixture(scope="session")
def long_cfr_video(tmp_path_factory):
    """A 120-frame CFR clip at 10 fps, GOP 30 — long enough that a forward
    jump crosses SEEK_AHEAD_THRESHOLD and spans several keyframes.

    Unlike the 12-frame fixtures this does not encode the frame index in
    luminance (20*N wraps past 255 above frame 12); tests here assert on
    FrameBatch.indices, not on content.
    """
    path = tmp_path_factory.mktemp("long_cfr") / "long_cfr.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "nullsrc=s=64x64:r=10:d=12",
        "-vf", "format=yuv420p",
        "-g", "30", str(path),
    )
    return path
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_indexed_prefetcher.py`. That module is already `pytestmark`-gated on CUDA and uses `NvVideoDecoder`; follow both conventions. Assert *decode work*, not wall-clock — timing assertions are flaky:

```python
def test_indexed_prefetcher_seeks_over_large_forward_gap(long_cfr_video):
    """A plan starting deep into the clip must seek, not decode from 0."""
    dec = NvVideoDecoder(long_cfr_video, batch_size=1)
    calls = {"decode": 0, "seek": []}

    real_next, real_seek = dec.next_batch, dec.seek_to_index

    def counting_next():
        calls["decode"] += 1
        return real_next()

    def counting_seek(i):
        calls["seek"].append(i)
        return real_seek(i)

    dec.next_batch = counting_next
    dec.seek_to_index = counting_seek

    with IndexedFramePrefetcher(dec, [100, 102]) as stream:
        got, _, _ = _drain_indexed(stream)

    assert got == [100, 102]
    assert calls["seek"] and calls["seek"][0] == 100, "must seek to frame 100"
    # Forward-walking from 0 costs ~100 batches; seeking costs a keyframe's worth.
    assert calls["decode"] < 40, f"decoded {calls['decode']} batches — not seeking"


def test_indexed_prefetcher_small_forward_gap_does_not_seek(long_cfr_video):
    """Below the threshold, decoding forward is cheaper than a seek."""
    dec = NvVideoDecoder(long_cfr_video, batch_size=1)
    seeks = []
    real_seek = dec.seek_to_index
    dec.seek_to_index = lambda i: (seeks.append(i), real_seek(i))[1]

    with IndexedFramePrefetcher(dec, [0, 2, 4]) as stream:
        got, _, _ = _drain_indexed(stream)

    assert got == [0, 2, 4]
    assert not seeks, f"small forward steps must not seek, got {seeks}"
```

- [ ] **Step 3: Run them and watch the first fail**

Run: `pixi run python -m pytest tests/test_indexed_prefetcher.py -k seeks -v`
Expected: `test_indexed_prefetcher_seeks_over_large_forward_gap` FAILs — `calls["seek"]` is empty and `calls["decode"]` is ~100. `test_indexed_prefetcher_small_forward_gap_does_not_seek` PASSes already (it is the regression guard for Step 4).

- [ ] **Step 4: Add the threshold to `_locate`**

In `src/framepipe/frame_prefetcher.py`, add the constant near the top of the module (beside the other module-level names):

```python
# A forward jump shorter than this is cheaper to decode through than to seek:
# seeking lands on the preceding keyframe and re-decodes to the target anyway.
# Sized to a typical GOP.
_SEEK_AHEAD_THRESHOLD = 32
```

Then narrow the forward branch at line 390 so only *near* targets walk; everything else falls through to the existing seek path:

```python
        if self._decoder.position <= want < self._decoder.position + _SEEK_AHEAD_THRESHOLD:
            # Near-ahead: forward walk, dropping unwanted batches until want is
            # in window. Cheaper than a seek that would rewind to a keyframe.
            while True:
                held = self._decoder.next_batch()
                if held is None:
                    raise RuntimeError(
                        f"decoder exhausted before frame {want}; its frame "
                        "count disagrees with metadata.num_frames"
                    )
                if int(held.indices[0]) <= want <= int(held.indices[-1]):
                    return held
        # Behind the window, or far ahead: pay a seek. The next batch starts at want.
        self._decoder.seek_to_index(want)
```

Leave lines 383-389 (the held-window check) and 402-406 (the seek + `next_batch`) untouched.

- [ ] **Step 5: Update the class docstring**

The `IndexedFramePrefetcher` docstring says "other backward steps pay a ``seek_to_index`` on the wrapped decoder." That is now incomplete. Change that sentence to:

```
    and reorder / random access (``[0, 2, 1, 5, 4]``). Repeated *consecutive*
    indices are served from the held window without re-decode; backward steps
    and large forward jumps pay a ``seek_to_index`` on the wrapped decoder,
    while short forward gaps decode through.
```

- [ ] **Step 6: Run the new tests and the full framepipe suite**

Run: `pixi run python -m pytest tests/test_indexed_prefetcher.py -v`
Expected: all PASS, including both new tests.

Run: `pixi run python -m pytest`
Expected: no new failures vs. the pre-change run. Pay attention to `test_seeking.py` and any decimation test using `range(0, n, 2)` — a stride of 2 is below the threshold and must still take the walk path, so those must be unaffected.

- [ ] **Step 7: Verify the real-file speedup**

Run:
```bash
pixi run python -u -c "
import time
from framepipe import IndexedFramePrefetcher, PyAvVideoDecoder
P='E:/Datasets/Guedelon/Carriere Luc/GoPro1/GX010037.MP4'
dec=PyAvVideoDecoder(P, device='cpu', batch_size=1)
t0=time.time()
with IndexedFramePrefetcher(dec,[13000]) as st:
    b=st.next_batch()
print('late frame: %.2fs idx=%s'%(time.time()-t0, b.indices.tolist()))
"
```
Expected: completes in **under 10 s** with `idx=[13000]`. This exact snippet measured **766.12 s** before the fix.

- [ ] **Step 8: Commit (in the framepipe repo)**

```bash
cd C:/Users/Charles/PycharmProjects/framepipe
git add src/framepipe/frame_prefetcher.py tests/test_indexed_prefetcher.py tests/conftest.py
git commit -m "fix(prefetcher): seek over large forward gaps instead of decoding through

IndexedFramePrefetcher._locate seeked backward steps but forward-walked
every batch to reach a target ahead of the decoder, so a plan starting at
frame 13000 decoded 13000 frames first: over 200s on a 4K clip, vs ~1.4s
seeked. Walk only within _SEEK_AHEAD_THRESHOLD frames, where re-decoding
from the preceding keyframe would cost more than the walk."
```

---

## Task 2: `load_audio` on PyAV (preserves the per-file progress feature)

**Files:**
- Modify: `src/clapsync/app/decode.py:1-55`
- Test: `tests/app/test_decode.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `load_audio(path: Path, target_rate: int | None = None, progress: Callable[[float], None] | None = None) -> tuple[torch.Tensor, int]` — unchanged signature. Returns `(waveform, sample_rate)`, waveform float32 shape `(1, N)`, peak-normalized mono. `sync.py:51` and `export.py:186` call it as-is.

**Behavior to preserve:** `progress` receives a monotonically increasing 0..1 fraction *during* the file's decode (commit `edbf535`). It must climb progressively, never jump 0→1. Updates are throttled to ~1% steps so a 218 s file emits ~100 callbacks rather than 10236 (one per AAC frame) — the GUI cannot use more, and unthrottled calls cost measurable time.

- [ ] **Step 1: Write the failing tests**

Add to `tests/app/test_decode.py` (create it if absent; `tone_wav` is an existing fixture in `tests/conftest.py`):

```python
import torch

from clapsync.app.decode import load_audio


def test_load_audio_shape_and_rate(tone_wav):
    path, rate = tone_wav(seconds=1.0, sample_rate=48000)
    wave, got_rate = load_audio(path)
    assert got_rate == rate
    assert wave.shape[0] == 1
    assert wave.dtype == torch.float32
    assert abs(wave.shape[1] - 48000) <= 64  # codec padding tolerance


def test_load_audio_is_peak_normalized(tone_wav):
    path, _ = tone_wav(seconds=1.0)
    wave, _ = load_audio(path)
    assert abs(float(wave.abs().max()) - 1.0) < 1e-5


def test_load_audio_progress_climbs_progressively(tone_wav):
    path, _ = tone_wav(seconds=8.0)
    seen = []
    load_audio(path, progress=seen.append)

    assert len(seen) >= 5, f"too coarse: {seen}"
    assert seen == sorted(seen), "progress must be monotonic"
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen[-1] == 1.0, "must finish at exactly 1.0"
    assert seen[0] < 0.5, "must not jump straight to the end"


def test_load_audio_target_rate_resamples(tone_wav):
    path, _ = tone_wav(seconds=1.0, sample_rate=48000)
    wave, rate = load_audio(path, target_rate=16000)
    assert rate == 16000
    assert abs(wave.shape[1] - 16000) <= 64


def test_load_audio_without_progress_still_works(tone_wav):
    path, _ = tone_wav(seconds=1.0)
    wave, _ = load_audio(path)
    assert wave.shape[1] > 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pixi run python -m pytest tests/app/test_decode.py -v`
Expected: the progress test FAILs (`len(seen) >= 5` — torchcodec's 5 s windows give only 2 ticks on an 8 s file, and `seen[-1] == 1.0` may not hold exactly). Others may pass against the current torchcodec code; that is fine — they are the parity net.

- [ ] **Step 3: Replace `load_audio`**

Rewrite `src/clapsync/app/decode.py`'s imports and `load_audio`. Delete `_AUDIO_CHUNK_S` — windowing is gone; PyAV reports progress per decoded frame.

```python
"""Media decode wrappers over framepipe/PyAV (audio waveforms, video frames)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch

# Progress is reported at most once per this fraction of a file. A 218 s GoPro
# clip decodes to ~10k AAC frames; calling back on each one costs real time and
# tells the GUI nothing it can draw.
_PROGRESS_STEP = 0.01


def load_audio(
    path: Path,
    target_rate: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """Load a peak-normalized mono waveform from any audio/video file.

    Args:
        path: Source media path.
        target_rate: If set, decode/resample to this sample rate.
        progress: Optional 0..1 callback reporting this file's decode fraction.
            Called as frames arrive, throttled to ~1% steps, ending at 1.0.

    Returns:
        (waveform, sample_rate) where waveform is float32 shape (1, N).

    Raises:
        ValueError: If the file has no audio stream.
    """
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        rate = int(target_rate or stream.codec_context.sample_rate)
        layout = stream.codec_context.layout
        # fltp gives to_ndarray() a planar (channels, samples) float32 array
        # regardless of what the codec natively emits.
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)

        time_base = float(stream.time_base)
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else 0.0
        )
        last = -1.0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
            if progress is None or not duration or frame.pts is None:
                continue
            fraction = min(frame.pts * time_base / duration, 1.0)
            if fraction - last >= _PROGRESS_STEP:
                progress(fraction)
                last = fraction
        for resampled in resampler.resample(None):  # flush
            chunks.append(resampled.to_ndarray())

    if progress is not None:
        progress(1.0)

    if not chunks:
        raise ValueError(f"no audio samples decoded from {path}")
    data = torch.from_numpy(np.concatenate(chunks, axis=1))

    mono = data.mean(dim=0, keepdim=True) if data.shape[0] > 1 else data
    peak = mono.abs().max()
    if peak > 0:
        mono = mono / peak
    return mono.contiguous(), int(rate)
```

Leave `decode_frames_at` and its `torchcodec` import alone for now — Task 4 handles it. The file will import both `av` and `torchcodec` between Tasks 2 and 4; that is expected and temporary.

- [ ] **Step 4: Run the tests**

Run: `pixi run python -m pytest tests/app/test_decode.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify waveform parity against torchcodec on a real file**

This is the load-bearing check — if waveforms drift, every sync offset drifts.

Run:
```bash
pixi run python -u -c "
import torch, numpy as np, av
from clapsync.app.decode import load_audio
from torchcodec.decoders import AudioDecoder
P='E:/Datasets/Guedelon/Carriere Luc/GoPro1/GX010037.MP4'
new,_ = load_audio(P)
d=AudioDecoder(P).get_all_samples()
old=d.data.mean(dim=0,keepdim=True) if d.data.shape[0]>1 else d.data
old=old/old.abs().max()
print('shapes', tuple(new.shape), tuple(old.shape))
n=min(new.shape[1], old.shape[1])
print('max abs diff:', float((new[:,:n]-old[:,:n]).abs().max()))
"
```
Expected: identical shapes and **max abs diff < 1e-6** (measured 2.9e-08 during planning).

- [ ] **Step 6: Verify the progress feature end-to-end on the user's 6 files**

Run:
```bash
pixi run python -u -c "
from pathlib import Path
from clapsync.app.media import probe
from clapsync.app.sync import offsets_from_media
P=[Path('E:/Datasets/Guedelon/Carriere Luc/GoPro%d/GX0100%s.MP4'%(i,s))
   for i,s in [(1,'37'),(2,'35'),(3,'35'),(4,'36'),(5,'34'),(6,'38')]]
media=[probe(p) for p in P]
ticks=[]
offs=offsets_from_media(media, progress=ticks.append, status=lambda s: print('STATUS', s))
print('offsets', [round(o,4) for o in offs])
print('progress ticks:', len(ticks))
"
```
Expected: `STATUS Loading audio (1/6)…` through `(6/6)`, then `Aligning waveforms…`. **Record the printed offsets** — Task 7 re-checks them. Ticks should number in the hundreds (progress climbing within each file), not 6.

- [ ] **Step 7: Commit**

```bash
git add src/clapsync/app/decode.py tests/app/test_decode.py
git commit -m "refactor(decode): load_audio on PyAV instead of torchcodec

Progress now comes from frames already being decoded rather than 5s
windowed re-reads, so the per-file bar is finer (~100 updates vs 44) and
needs no seeking. Waveforms match torchcodec to 2.9e-08, so offsets are
unchanged. Throttled to ~1% steps: a 218s clip has ~10k AAC frames and
calling back on each costs more than it reports."
```

---

## Task 3: `probe` on framepipe + PyAV

**Files:**
- Modify: `src/clapsync/app/media.py` (whole file)
- Test: `tests/app/test_media.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `probe(path: Path) -> MediaInfo` — unchanged. `MediaInfo` fields unchanged: `path, duration, has_audio, kind, sample_rate, width, height, fps`. Callers: `export.py:299,333`, `sync.py:96`, `cli.py:29`, `gui/sync_editor.py:83`.

**Error contract to preserve:** missing file → `FileNotFoundError`; a file with neither stream → `ValueError`. PyAV cooperates: `av.open` on a missing path raises `FileNotFoundError` natively, on junk raises `av.InvalidDataError`, and a video with no audio yields `streams.audio == []` with no exception.

- [ ] **Step 1: Write the failing tests**

Replace `tests/app/test_media.py` with:

```python
import pytest

from clapsync.app.media import probe


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


def test_probe_non_media_file_raises(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 10)
    with pytest.raises(ValueError):
        probe(junk)


def test_probe_audio_only(tone_wav):
    path, rate = tone_wav(seconds=1.0, sample_rate=48000)
    info = probe(path)
    assert info.kind == "audio"
    assert info.has_audio is True
    assert info.sample_rate == rate
    assert info.width is None and info.height is None
    assert abs(info.duration - 1.0) < 0.1


def test_probe_video_without_audio(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    info = probe(path)
    assert info.kind == "video"
    assert info.has_audio is False
    assert (info.width, info.height) == (w, h)
    assert abs(info.fps - fps) < 0.1
    assert abs(info.duration - 1.0) < 0.1


def test_probe_muxed_av(av_video):
    path, fps, n, w, h, sr = av_video(seconds=1.0)
    info = probe(path)
    assert info.kind == "video"
    assert info.has_audio is True
    assert info.sample_rate == sr
    assert (info.width, info.height) == (w, h)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run python -m pytest tests/app/test_media.py -v`
Expected: these pass against the current torchcodec `probe` (they are a parity net written before the swap). If any FAIL now, stop — the test is wrong, not the code. Fix the test before continuing.

- [ ] **Step 3: Rewrite `media.py`**

```python
"""Media probing: unify audio-only and video files behind MediaInfo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
from framepipe.metadata import extract_video_metadata


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
    """Return (has_audio, sample_rate, duration) or (False, None, None).

    A container with no audio stream is not an error; av raises only when the
    file itself is undecodable, which the caller reports as "no streams".
    """
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return False, None, None
            stream = container.streams.audio[0]
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration
                else None
            )
            return True, int(stream.codec_context.sample_rate), duration
    except av.FFmpegError:
        return False, None, None


def probe(path: Path) -> MediaInfo:
    """Probe a media file via framepipe (video) and PyAV (audio) metadata.

    A file with a decodable video stream is kind="video"; otherwise it is
    treated as audio-only.

    Args:
        path: Source media path.

    Returns:
        A populated MediaInfo.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has neither a video nor an audio stream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    has_audio, sample_rate, audio_dur = _audio_meta(path)

    try:
        vmeta = extract_video_metadata(str(path))
    except (av.FFmpegError, ValueError, RuntimeError):
        vmeta = None

    if vmeta is not None:
        return MediaInfo(
            path=path,
            duration=vmeta.duration,
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
            duration=audio_dur if audio_dur is not None else 0.0,
            has_audio=True,
            kind="audio",
            sample_rate=sample_rate,
        )

    raise ValueError(f"No decodable audio or video stream in {path}")
```

- [ ] **Step 4: Run the tests**

Run: `pixi run python -m pytest tests/app/test_media.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify against a real GoPro file**

Run:
```bash
pixi run python -u -c "
from pathlib import Path
from clapsync.app.media import probe
i=probe(Path('E:/Datasets/Guedelon/Carriere Luc/GoPro1/GX010037.MP4'))
print(i.kind, i.width, i.height, round(i.duration,2), round(i.fps,3), i.has_audio, i.sample_rate)
"
```
Expected exactly: `video 5312 2988 218.37 59.94 True 48000`.

- [ ] **Step 6: Commit**

```bash
git add src/clapsync/app/media.py tests/app/test_media.py
git commit -m "refactor(media): probe via framepipe video meta + PyAV audio meta"
```

---

## Task 4: `decode_frames_at` on framepipe

**Files:**
- Modify: `src/clapsync/app/decode.py` (`decode_frames_at`; remove the torchcodec import)
- Test: `tests/app/test_decode.py`

**Depends on:** Task 1 (without the seek fix this is ~19× slower than torchcodec).

**Interfaces:**
- Consumes: `load_audio` from Task 2 (same file); framepipe `PyAvVideoDecoder`, `IndexedFramePrefetcher`, `extract_video_metadata`.
- Produces: `decode_frames_at(path: Path, seconds: list[float], device: str = "cpu") -> torch.Tensor` — unchanged. Returns uint8 `(len(seconds), C, H, W)` on `device`. Caller: `export.py:147`.

**Semantics to preserve:** torchcodec's `get_frames_played_at` returns, for time `t`, the last frame whose pts ≤ t. Out-of-range timestamps are clamped into the stream range rather than raising. `export.py:147` passes **dense, ordered** times (`frame_source_times` over the trim range), but the function must still tolerate unordered and duplicate inputs — verified during planning: querying `[0.0, 12.5, 60.0, 123.456, 218.0, 1e9, -5.0]` produced frames bit-identical to torchcodec (`0.0` max diff), including the duplicate frame-0 that `-5.0` and `0.0` both clamp to.

- [ ] **Step 1: Write the failing tests**

Append to `tests/app/test_decode.py`:

```python
from clapsync.app.decode import decode_frames_at


def test_decode_frames_at_shape_and_order(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=2.0, fps=30.0, w=64, h=48)
    frames = decode_frames_at(path, [0.0, 0.5, 1.0], device="cpu")
    assert frames.shape == (3, 3, h, w)
    assert frames.dtype == torch.uint8
    assert int(frames[0, 0].float().mean()) > 200  # fixture is solid red


def test_decode_frames_at_clamps_out_of_range(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0)
    frames = decode_frames_at(path, [-5.0, 1e9], device="cpu")
    assert frames.shape[0] == 2  # clamped, not raised


def test_decode_frames_at_unordered_and_duplicate(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=2.0, fps=30.0)
    frames = decode_frames_at(path, [1.0, 0.0, 1.0], device="cpu")
    assert frames.shape[0] == 3
    # same timestamp -> same frame, and order is caller order
    assert torch.equal(frames[0], frames[2])
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run python -m pytest tests/app/test_decode.py -k decode_frames -v`
Expected: PASS against current torchcodec code (parity net, as in Task 3). If one FAILs, the test is wrong — fix it first.

- [ ] **Step 3: Replace `decode_frames_at`**

Swap `from torchcodec.decoders import AudioDecoder, VideoDecoder` for the framepipe imports at the top of `decode.py` — after this step the file has **no torchcodec import**:

```python
from framepipe import IndexedFramePrefetcher, PyAvVideoDecoder
from framepipe.metadata import extract_video_metadata
```

```python
def decode_frames_at(
    path: Path, seconds: list[float], device: str = "cpu"
) -> torch.Tensor:
    """Decode one frame per requested timestamp (NCHW uint8).

    Timestamps are clamped into the decodable stream range so callers can pass
    boundary times without raising. The frame for time t is the last one whose
    pts <= t.

    Args:
        path: Source video path.
        seconds: Presentation timestamps in seconds; may be unordered and may
            repeat.
        device: "cpu" or "cuda" (nvdec).

    Returns:
        uint8 tensor of shape (len(seconds), C, H, W) on the requested device.
    """
    if not seconds:
        raise ValueError("decode_frames_at needs at least one timestamp")

    pts = extract_video_metadata(str(path)).pts  # (num_frames,) seconds, CPU
    wanted = torch.tensor(seconds, dtype=pts.dtype).clamp_(
        float(pts[0]), float(pts[-1])
    )
    # last frame with pts <= t, matching torchcodec's get_frames_played_at
    idx = (torch.searchsorted(pts, wanted, right=True) - 1).clamp_(0, len(pts) - 1)

    # The prefetcher consumes a forward-ordered plan; sort, then invert back to
    # caller order so unordered/duplicate timestamps behave.
    order = torch.argsort(idx)
    plan = idx[order].tolist()

    decoder = PyAvVideoDecoder(
        str(path), device=device, batch_size=1, start_frame=int(plan[0]),
    )
    batches = []
    with IndexedFramePrefetcher(decoder, plan) as stream:
        while (batch := stream.next_batch()) is not None:
            batches.append(batch.frames)

    frames = torch.cat(batches, dim=0)
    return frames[torch.argsort(order)]
```

`start_frame=int(plan[0])` keeps this fast even against a framepipe without Task 1's fix.

- [ ] **Step 4: Run the tests**

Run: `pixi run python -m pytest tests/app/test_decode.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify frame parity against torchcodec on a real file**

Run:
```bash
pixi run python -u -c "
import time, torch
from clapsync.app.decode import decode_frames_at
from torchcodec.decoders import VideoDecoder
P='E:/Datasets/Guedelon/Carriere Luc/GoPro1/GX010037.MP4'
S=[0.0, 12.5, 60.0, 123.456, 218.0, 1e9, -5.0]
t0=time.time(); new=decode_frames_at(P,S,device='cpu'); t_new=time.time()-t0
d=VideoDecoder(P, device='cpu'); m=d.metadata
hi=m.end_stream_seconds or m.duration_seconds
old=d.get_frames_played_at([min(max(s,m.begin_stream_seconds or 0.0),hi-1e-3) for s in S]).data
print('new %.1fs shape=%s'%(t_new, tuple(new.shape)))
print('max abs diff:', float((new.float()-old.float()).abs().max()))
"
```
Expected: **max abs diff exactly 0.0**, and `new` completes in well under the 41.6 s torchcodec baseline. If the diff is nonzero the seconds→index mapping is wrong — do not proceed.

- [ ] **Step 6: Confirm torchcodec is gone from this module**

Run: `pixi run python -c "import clapsync.app.decode, sys; assert 'torchcodec' not in sys.modules, 'still imports torchcodec'; print('decode.py is torchcodec-free')"`
Expected: `decode.py is torchcodec-free`.

- [ ] **Step 7: Commit**

```bash
git add src/clapsync/app/decode.py tests/app/test_decode.py
git commit -m "refactor(decode): decode_frames_at on framepipe instead of torchcodec

Maps timestamps to frame indices with searchsorted (last frame with
pts <= t, matching get_frames_played_at) and pre-seeks via start_frame.
Frames are bit-identical to torchcodec on a real 4K clip."
```

---

## Task 5: `encode_clip` on PyAV

**Files:**
- Modify: `src/clapsync/app/encode.py` (whole file)
- Test: `tests/app/test_encode.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `pick_video_codec(device: str) -> str` and `encode_clip(out_path, video_frames, video_fps, audio_samples, sample_rate, *, video_codec=None, crf=18, device="cpu") -> None` — both unchanged. Callers: `export.py:39,232,247,259`.

**Behavior notes:**
- `encode_clip` infers the container from `out_path.suffix` and must now also pick the **audio codec**, which torchcodec did implicitly. Verified round-trips: `wav`→`pcm_s16le` (+0 samples), `m4a`/`mp4`→`aac` (+128, exactly matching torchcodec), `mp3`→`libmp3lame` (+0), `flac`→`flac` (+0).
- `device` now selects the **codec only**. torchcodec required frames on the encode device; PyAV's `h264_nvenc` accepts CPU frames and uploads internally. Keep the parameter — `export.py:209` and `_nvenc_available()` pass it — but drop the `.to(device)`.
- `h264_nvenc` does not accept `crf`; its quality knob is `cq`. torchcodec passed `crf` regardless and ffmpeg ignored it. Map it properly here.
- `export.py:237-250` relies on a failed NVENC encode **raising** so it can retry on CPU. Do not swallow encode errors.

- [ ] **Step 1: Write the failing tests**

Replace `tests/app/test_encode.py` with:

```python
import av
import pytest
import torch

from clapsync.app.encode import encode_clip, pick_video_codec


def _frames(n=30, h=48, w=64):
    f = torch.zeros((n, 3, h, w), dtype=torch.uint8)
    f[:, 1] = 180  # solid green
    return f


def _tone(seconds=1.0, rate=48000, channels=1):
    import math
    t = torch.arange(int(seconds * rate), dtype=torch.float32) / rate
    return torch.stack([torch.sin(2 * math.pi * 440 * t)] * channels)


def test_pick_video_codec():
    assert pick_video_codec("cpu") == "libx264"
    assert pick_video_codec("cuda") == "h264_nvenc"
    assert pick_video_codec("cuda:0") == "h264_nvenc"


def test_encode_requires_a_stream(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "x.mp4", None, None, None, None)


def test_encode_video_only(tmp_path):
    out = tmp_path / "v.mp4"
    encode_clip(out, _frames(), 30.0, None, None, video_codec="libx264")
    with av.open(str(out)) as c:
        assert not c.streams.audio
        v = c.streams.video[0]
        assert (v.width, v.height) == (64, 48)
        assert sum(1 for _ in c.decode(v)) == 30


def test_encode_muxed_av(tmp_path):
    out = tmp_path / "av.mp4"
    encode_clip(out, _frames(), 30.0, _tone(1.0), 48000, video_codec="libx264")
    with av.open(str(out)) as c:
        assert sum(1 for _ in c.decode(c.streams.video[0])) == 30
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert a.codec_context.sample_rate == 48000
        assert sum(f.samples for f in c.decode(a)) == pytest.approx(48000, abs=300)


def test_encode_audio_only_wav_is_sample_exact(tmp_path):
    out = tmp_path / "a.wav"
    encode_clip(out, None, None, _tone(1.0), 48000)
    with av.open(str(out)) as c:
        a = c.streams.audio[0]
        assert sum(f.samples for f in c.decode(a)) == 48000


def test_encode_video_without_fps_raises(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "v.mp4", _frames(), None, None, None)


def test_encode_audio_without_rate_raises(tmp_path):
    with pytest.raises(ValueError):
        encode_clip(tmp_path / "a.wav", None, None, _tone(1.0), None)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run python -m pytest tests/app/test_encode.py -v`
Expected: mostly PASS on current torchcodec code (parity net). `test_encode_audio_only_wav_is_sample_exact` is the sharpest — it must stay exact after the swap.

- [ ] **Step 3: Rewrite `encode.py`**

```python
"""Muxed audio+video encoding via PyAV."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch

# Audio codec per container. torchcodec inferred this from the extension; PyAV
# needs it named. Verified round-trips: pcm_s16le/libmp3lame/flac are
# sample-exact, aac pads +128 (identical to torchcodec's aac padding).
_AUDIO_CODEC_FOR_SUFFIX = {
    ".wav": "pcm_s16le",
    ".m4a": "aac",
    ".mp4": "aac",
    ".mkv": "aac",
    ".mov": "aac",
    ".mp3": "libmp3lame",
    ".flac": "flac",
}
_DEFAULT_AUDIO_CODEC = "aac"

# Frames are handed to the encoder in blocks of this many samples.
_AUDIO_BLOCK = 1024


def pick_video_codec(device: str) -> str:
    """Return the H.264 encoder name for a device.

    Args:
        device: Torch device string, e.g. "cpu", "cuda", "cuda:0".

    Returns:
        "h264_nvenc" for CUDA devices, "libx264" otherwise.
    """
    return "h264_nvenc" if str(device).startswith("cuda") else "libx264"


def _audio_codec_for(out_path: Path) -> str:
    return _AUDIO_CODEC_FOR_SUFFIX.get(out_path.suffix.lower(), _DEFAULT_AUDIO_CODEC)


def _quality_options(codec: str, crf: int) -> dict[str, str]:
    """NVENC has no crf; its constant-quality knob is cq."""
    return {"cq": str(crf)} if codec == "h264_nvenc" else {"crf": str(crf)}


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

    At least one of video_frames or audio_samples must be provided.

    Args:
        out_path: Destination path; container format inferred from suffix.
        video_frames: (N, C, H, W) uint8 frames, or None for audio-only.
        video_fps: Output frame rate; required when video_frames is given.
        audio_samples: (C, M) float samples in [-1, 1], or None for
            video-only.
        sample_rate: Audio sample rate in Hz; required when audio_samples
            is given.
        video_codec: Override encoder name; defaults to pick_video_codec(device).
        crf: Constant quality for the video stream (mapped to cq on NVENC).
        device: "cpu" or "cuda[:<index>]"; selects the default encoder. Unlike
            the old torchcodec path this does not move the frame tensor —
            h264_nvenc uploads from host memory itself.

    Raises:
        ValueError: If required companions (fps/sample_rate) are missing, or
            if neither stream is provided.
    """
    if video_frames is None and audio_samples is None:
        raise ValueError("encode_clip needs at least one stream")
    if video_frames is not None and video_fps is None:
        raise ValueError("video_fps is required when video_frames is given")
    if audio_samples is not None and sample_rate is None:
        raise ValueError("sample_rate is required when audio_samples is given")

    out_path = Path(out_path)
    with av.open(str(out_path), mode="w") as container:
        vstream = None
        astream = None
        resampler = None
        layout = None

        if video_frames is not None:
            _, _channels, height, width = video_frames.shape
            codec = video_codec or pick_video_codec(device)
            vstream = container.add_stream(codec, rate=int(round(video_fps)))
            vstream.width = width
            vstream.height = height
            vstream.pix_fmt = "yuv420p"
            vstream.options = _quality_options(codec, crf)

        if audio_samples is not None:
            layout = "mono" if audio_samples.shape[0] == 1 else "stereo"
            acodec = _audio_codec_for(out_path)
            astream = container.add_stream(acodec, rate=sample_rate, layout=layout)
            resampler = av.AudioResampler(
                format=astream.codec_context.format.name,
                layout=layout,
                rate=sample_rate,
            )

        if vstream is not None:
            frames = video_frames.cpu()
            for i in range(frames.shape[0]):
                rgb = frames[i].permute(1, 2, 0).numpy()  # CHW -> HWC
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(rgb), format="rgb24",
                )
                for packet in vstream.encode(frame):
                    container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)

        if astream is not None:
            samples = audio_samples.cpu().numpy().astype(np.float32)
            pts = 0
            for start in range(0, samples.shape[1], _AUDIO_BLOCK):
                block = np.ascontiguousarray(
                    samples[:, start : start + _AUDIO_BLOCK]
                )
                frame = av.AudioFrame.from_ndarray(
                    block, format="fltp", layout=layout,
                )
                frame.sample_rate = sample_rate
                frame.pts = pts
                pts += block.shape[1]
                for resampled in resampler.resample(frame):
                    for packet in astream.encode(resampled):
                        container.mux(packet)
            for resampled in resampler.resample(None):  # flush resampler
                for packet in astream.encode(resampled):
                    container.mux(packet)
            for packet in astream.encode():  # flush encoder
                container.mux(packet)
```

- [ ] **Step 4: Run the tests**

Run: `pixi run python -m pytest tests/app/test_encode.py -v`
Expected: all PASS. If `test_encode_audio_only_wav_is_sample_exact` fails, the resampler flush is dropping a partial block — fix that, do not relax the test.

- [ ] **Step 5: Verify the NVENC path still works**

Run:
```bash
pixi run python -u -c "
import torch, tempfile, os
from pathlib import Path
from clapsync.app.encode import encode_clip
f=torch.zeros((10,3,144,256),dtype=torch.uint8); f[:,1]=200
d=tempfile.mkdtemp(); p=Path(d)/'nv.mp4'
encode_clip(p, f, 30.0, None, None, device='cuda')
print('nvenc OK, %.1f KB'%(os.path.getsize(p)/1e3))
"
```
Expected: `nvenc OK` with a nonzero size. This exercises the same path `export.py:_nvenc_available()` probes.

- [ ] **Step 6: Commit**

```bash
git add src/clapsync/app/encode.py tests/app/test_encode.py
git commit -m "refactor(encode): encode_clip on PyAV instead of torchcodec

Names the audio codec per container (torchcodec inferred it): wav/mp3/flac
are sample-exact, aac pads +128 exactly as torchcodec did. Maps crf to cq
on NVENC, which has no crf. device now selects the codec only — PyAV's
h264_nvenc uploads from host memory, so frames stay on the CPU."
```

---

## Task 6: Test fixtures on PyAV

**Files:**
- Modify: `tests/conftest.py` (whole file)

**Interfaces:**
- Consumes: nothing (fixtures must not depend on `clapsync.app.encode`, or a bug there would silently produce matching-but-wrong fixtures).
- Produces: unchanged fixture contracts — `tone_wav(seconds, sample_rate, freq, name) -> (path, sample_rate)`, `rgb_video(seconds, fps, w, h, name) -> (path, fps, n, w, h)`, `av_video(seconds, fps, w, h, sample_rate, freq, name) -> (path, fps, n, w, h, sample_rate)`.

- [ ] **Step 1: Rewrite `conftest.py`**

```python
"""Shared test fixtures.

Synthetic media are generated with PyAV so decode/probe/export tests have
known-content inputs without checking binaries into git. These deliberately do
not use clapsync's own encode_clip — a bug there must fail the tests, not
silently produce matching fixtures.
"""
from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np
import pytest
import torch


def _tone(seconds: float, sample_rate: int, freq: float) -> torch.Tensor:
    """Mono sine tone, shape (1, N), float in [-1, 1]."""
    n = int(seconds * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return torch.sin(2 * math.pi * freq * t).unsqueeze(0)


def _write_audio(container, samples: torch.Tensor, sample_rate: int, codec: str):
    """Encode (1, N) float samples as one audio stream."""
    stream = container.add_stream(codec, rate=sample_rate, layout="mono")
    resampler = av.AudioResampler(
        format=stream.codec_context.format.name, layout="mono", rate=sample_rate,
    )
    data = samples.numpy().astype(np.float32)
    pts = 0
    for start in range(0, data.shape[1], 1024):
        block = np.ascontiguousarray(data[:, start : start + 1024])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = pts
        pts += block.shape[1]
        for resampled in resampler.resample(frame):
            for packet in stream.encode(resampled):
                container.mux(packet)
    for resampled in resampler.resample(None):
        for packet in stream.encode(resampled):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


def _write_video(container, frames: torch.Tensor, fps: float):
    """Encode (N, 3, H, W) uint8 frames as one libx264 stream."""
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = frames.shape[3]
    stream.height = frames.shape[2]
    stream.pix_fmt = "yuv420p"
    for i in range(frames.shape[0]):
        rgb = np.ascontiguousarray(frames[i].permute(1, 2, 0).numpy())
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Factory: write a mono tone WAV, return (path, sample_rate)."""

    def _make(seconds: float = 1.0, sample_rate: int = 48000,
              freq: float = 440.0, name: str = "tone.wav") -> tuple[Path, int]:
        path = tmp_path / name
        with av.open(str(path), mode="w") as container:
            _write_audio(container, _tone(seconds, sample_rate, freq),
                         sample_rate, "pcm_s16le")
        return path, sample_rate

    return _make


@pytest.fixture
def rgb_video(tmp_path: Path):
    """Factory: write a solid-color video, return (path, fps, n, w, h)."""

    def _make(seconds: float = 1.0, fps: float = 30.0,
              w: int = 64, h: int = 48, name: str = "vid.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 0] = 255  # solid red
        with av.open(str(path), mode="w") as container:
            _write_video(container, frames, fps)
        return path, fps, n, w, h

    return _make


@pytest.fixture
def av_video(tmp_path: Path):
    """Factory: write a muxed audio+video file, return (path, fps, n, w, h, sr)."""

    def _make(seconds: float = 1.0, fps: float = 30.0, w: int = 64, h: int = 48,
              sample_rate: int = 48000, freq: float = 440.0, name: str = "av.mp4"):
        path = tmp_path / name
        n = int(seconds * fps)
        frames = torch.zeros((n, 3, h, w), dtype=torch.uint8)
        frames[:, 2] = 200  # solid blue
        with av.open(str(path), mode="w") as container:
            _write_video(container, frames, fps)
            _write_audio(container, _tone(seconds, sample_rate, freq),
                         sample_rate, "aac")
        return path, fps, n, w, h, sample_rate

    return _make
```

Note `_write_video` and `_write_audio` both add their stream to an already-open container, so `av_video` muxes both into one file.

- [ ] **Step 2: Run the whole suite**

Run: `pixi run python -m pytest -v`
Expected: all PASS. Every fixture consumer (`tests/app/test_media.py`, `test_decode.py`, `test_encode.py`, `test_export.py`, `test_sync.py`, `test_cpu_gpu.py`) exercises these.

- [ ] **Step 3: Port `tests/app/test_cpu_gpu.py`**

This is the last torchcodec import in `tests/` (Task 5 already replaced `test_encode.py` wholesale). It uses `VideoDecoder(...).metadata.num_frames` in two places as a read-back assertion.

Replace the import at line 10:

```python
# was: from torchcodec.decoders import VideoDecoder
import av
```

Add a helper below the `cuda_only` marker (line 18) — `num_frames` is not in a container header for a freshly muxed file, so count decoded frames, which is what the assertion actually means:

```python
def _frame_count(path) -> int:
    """Decoded frame count — a muxed file's header has no reliable frame count."""
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))
```

Then replace both assertions (lines 27 and 39):

```python
    assert _frame_count(out) >= 9
```

Leave `test_decode_frames_on_device`, `test_nvenc_probe_true_on_gpu`, and `test_export_end_to_end_uses_gpu` alone — they touch no torchcodec.

- [ ] **Step 4: Confirm `tests/` is torchcodec-free**

Run: `pixi run grep -rn torchcodec tests/`
Expected: no matches.

Run: `pixi run python -m pytest -v`
Expected: all PASS, including the `slow`-marked GPU cases on this machine.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: generate synthetic media with PyAV instead of torchcodec

Fixtures deliberately do not use clapsync's own encode_clip — a bug there
must fail the tests, not silently produce matching fixtures."
```

---

## Task 7: Drop the dependency and update docs

**Files:**
- Modify: `pyproject.toml:6-18,44,78`
- Modify: `README.md:18,62,74,78,92,109,114-115,129`

**Interfaces:**
- Consumes: Tasks 1–6 complete (no torchcodec imports remain anywhere).
- Produces: an environment with no torchcodec installed.

- [ ] **Step 1: Prove nothing imports torchcodec**

Run: `pixi run grep -rn torchcodec src/ tests/`
Expected: **no matches**. If any remain, go back to the owning task — do not edit deps first.

- [ ] **Step 2: Update `pyproject.toml`**

Replace the `app` extra (line 18) — framepipe is a git dependency, so consumers need it by URL:

```toml
app = ["av>=18", "framepipe @ git+https://github.com/cjaverliat/framepipe"]
```

Delete the torchcodec line from `[tool.pixi.pypi-dependencies]` (line 44). Leave the local editable `framepipe = { path = "../framepipe", editable = true }` entry as-is for dev.

Update the comment block at lines 6-9, which currently explains a torchcodec/torch CUDA pairing that no longer exists:

```toml
# Abstract runtime deps for consumers. Media I/O is framepipe (video decode)
# plus PyAV (audio, muxing); both bind the ffmpeg shared libraries. No CUDA
# index is pinned here on purpose — the consuming project installs torch from
# the appropriate CUDA wheel index. This workspace's own cu130 pins live in
# [tool.pixi.pypi-dependencies] for dev.
```

Update the pytest marker (line 78):

```toml
markers = ["slow: requires framepipe/PyAV decode/encode with real codecs"]
```

- [ ] **Step 3: Re-resolve the environment**

Run: `pixi install`
Expected: resolves without torchcodec. May take minutes.

Run: `pixi run python -c "import torchcodec" `
Expected: `ModuleNotFoundError` — proof it is actually gone, not merely unimported.

- [ ] **Step 4: Update `README.md`**

Rewrite the torchcodec references at lines 18, 62, 74, 78, 92, 109, 114-115, 129. Read each line in context first. Substance to carry:
- The app layer needs `clapsync[app]` → framepipe + PyAV (not torchcodec).
- `clapsync.core` stays pure — it needs neither.
- Line 109's "Media decode/encode runs on torchcodec" → framepipe for video decode, PyAV for audio and muxing.
- Lines 114-115's torch/torchcodec shared-CUDA-build constraint is **obsolete** — delete it. Keep the "CUDA-capable GPU recommended" advice; framepipe's `device="cuda:0"` and `h264_nvenc` still use it.
- Line 129's architecture comment → `app/ # file I/O, probe, framepipe/PyAV decode/encode, export, sync_and_trim`.

- [ ] **Step 5: Verify `core` stays pure**

Run: `pixi run python -c "import clapsync.core, sys; assert 'av' not in sys.modules and 'framepipe' not in sys.modules; print('core is media-lib-free')"`
Expected: `core is media-lib-free`.

- [ ] **Step 6: Run the full suite**

Run: `pixi run python -m pytest -v`
Expected: all PASS.

- [ ] **Step 7: End-to-end on the user's six GoPro files**

Run:
```bash
pixi run python -u -c "
from pathlib import Path
from clapsync.app.media import probe
from clapsync.app.sync import offsets_from_media
P=[Path('E:/Datasets/Guedelon/Carriere Luc/GoPro%d/GX0100%s.MP4'%(i,s))
   for i,s in [(1,'37'),(2,'35'),(3,'35'),(4,'36'),(5,'34'),(6,'38')]]
media=[probe(p) for p in P]
ticks=[]
offs=offsets_from_media(media, progress=ticks.append, status=lambda s: print('STATUS', s))
print('offsets', [round(o,4) for o in offs])
print('ticks', len(ticks))
"
```
Expected: offsets **match the values recorded in Task 2 Step 6** to within 1e-3 s. The status line still steps `Loading audio (1/6)…` → `(6/6)…`, and tick count is in the hundreds.

- [ ] **Step 8: Launch the GUI and watch the bar**

Run: `pixi run clapsync`

Load the six GoPro files and start a sync. Confirm by eye: the label steps `Loading audio (1/6)` → `(6/6)` while the bar sweeps **0→100% within each file** and resets per file — not one jump per file. Then run an export and confirm the output plays with audio and video in sync. This is the only step that checks the actual user-visible feature; do not skip it.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml README.md pixi.lock
git commit -m "build: drop torchcodec — media I/O is framepipe + PyAV

Video decode and metadata come from framepipe, audio decode and all
muxing from PyAV. Both bind the same ffmpeg shared libraries torchcodec
used, so no ffmpeg CLI is introduced. Removes the torch/torchcodec
shared-CUDA-build constraint."
```

---

## Verification Summary

| Requirement | Task |
|---|---|
| No torchcodec in `src/` | Tasks 2, 3, 4, 5 (asserted T4 S6) |
| No torchcodec in `tests/` | Task 6 (asserted T6 S3) |
| No torchcodec dep / installed | Task 7 (asserted T7 S1, S3) |
| framepipe used where possible (video) | Tasks 3, 4 |
| PyAV elsewhere (audio, encode, fixtures) | Tasks 2, 5, 6 |
| No ffmpeg CLI / subprocess | All — PyAV only |
| Per-file 0→100% audio progress preserved | Task 2 (S6), Task 7 (S7, S8) |
| Sync offsets unchanged | Task 2 S5 (waveform 1e-6), Task 7 S7 (offsets 1e-3) |
| Frames unchanged | Task 4 S5 (bit-identical) |
| Export unchanged | Task 5 S4 (sample-exact wav), Task 7 S8 (GUI export) |
| NVENC path intact | Task 5 S5, Task 7 S8 |
| `core` stays pure | Task 7 S5 |
| framepipe seek bug fixed | Task 1 (S6) |

## Risks

- **Task 1 touches a second repo.** framepipe is editable-installed, so a regression there hits clapsync immediately and silently. Run framepipe's own suite (T1 S5) before moving on.
- **PyAV audio decode is 3.7× slower than torchcodec warm** (1.75 s vs 0.48 s per 218 s file). Across six files that is ~10 s vs ~3 s of CPU — but cold reads of these 3.28 GB files cost ~19 s each, so the change is invisible in practice. If it ever matters, `stream.thread_type = "AUTO"` is already set and the remaining cost is the per-frame `to_ndarray`.
- **`export.py:_build_video_frames` stacks the entire clip in RAM** (`torch.stack(frames)`, line 168). At 5312×2988 that is ~48 MB/frame — a 10 s trim is ~14 GB. This is a **pre-existing** bug, untouched by this plan, but Task 7 Step 8's export test may hit it on a long trim. Use a short trim; file it separately.
