# c3d mocap sync + trim support — design

**Date:** 2026-07-22
**Branch:** `feat/c3d-mocap-sync`
**Status:** approved design, pending spec review

## Goal

Add motion-capture (`.c3d`) files as first-class sources in clapsync. A c3d
file carries no audio or video, so it cannot join the existing MFCC
cross-correlation sync. Instead it is synced to the audio/video (A/V) world
through a **clapperboard clap**, treated as a shared physical clock:

- The clap **sound** is detected in the A/V tracks (audio) → an anchor time on
  the shared timeline.
- The clap **motion** is detected in the c3d (clapperboard arm markers meeting)
  → a local time in the mocap stream.
- The c3d offset is the difference, placing the mocap on the same timeline as
  everything else.

The feature also renders an animated 3D marker preview and exports a
synced + trimmed `.c3d` alongside the existing A/V outputs.

Non-goals: retargeting, skeleton solving, editing marker data, supporting
mocap formats other than c3d, syncing multiple c3d files to each other
directly (each c3d syncs to the A/V clap independently).

## Key decisions (locked)

- **Clap anchor:** auto-pick, gated by cross-track agreement (see below).
- **c3d library:** `c3d` (py-c3d, pure-Python, pip). Fallback to `ezc3d` only
  if py-c3d fails to round-trip the user's real files (validated in Phase A).
- **3D preview:** portable 2D `QPainter` orthographic turntable projection of
  the markers. (Chosen over `QOpenGLWidget` during implementation: the endorsed
  premortem mandated a QPainter fallback for the mixed installer target anyway,
  the marker count is tiny so no GPU is needed, and it needs no GL context —
  which RDP / no-GPU machines cannot be assumed to provide.)
- **Export padding:** trim to the shared window, pad regions outside c3d
  coverage with **invalid** frames (residual = −1), matching audio silence-pad.
- **A/V offset convention is trusted and locked:** `shared = local + offset`
  (positive offset shifts a track's content later on the shared timeline). No
  re-audit of the A/V path. Only the new c3d bridge formula gets a sign test.

## Architecture

Existing pipeline is unchanged for A/V:
`select → probe → decode audio → MFCC cross-corr → manual edit → trim/pad export`.

c3d support adds a parallel lane that bridges into the same offset list.

### New modules

**`src/clapsync/app/mocap.py`** — I/O only (py-c3d), no analysis.

```python
@dataclass(frozen=True)
class MocapData:
    point_rate: float                 # POINT:RATE, Hz
    first_frame: int                  # c3d FIRST_FRAME (1-based origin)
    labels: list[str]                 # marker labels, order matches points
    points: np.ndarray                # (n_frames, n_markers, 3), float32
    valid: np.ndarray                 # (n_frames, n_markers) bool; False where
                                      # residual < 0 (occluded/invalid)
    analog: AnalogData | None         # analog channels, own rate (see export)
    events: list[C3dEvent]            # label + absolute frame index
    raw_params: object                # opaque param groups preserved for write
    @property
    def duration(self) -> float: ...  # n_frames / point_rate

def load_c3d(path: Path) -> MocapData: ...
def write_c3d(path: Path, data: MocapData) -> None: ...  # preserves units,
    # param groups, analog, events; writes residual = -1 for invalid frames
```

`write_c3d` must round-trip units (`POINT:UNITS`), param groups, analog data,
and events. Invalid/pad frames are written with residual = −1 and camera mask
0 so downstream tools do not read padded coordinates as real markers.

**`src/clapsync/core/clap.py`** — PURE, no I/O, fully unit-testable.

```python
@dataclass(frozen=True)
class ClapCandidate:
    time: float        # seconds, local to the stream, sub-sample refined
    score: float       # combined feature quality, higher = more clap-like

# --- audio ---
def detect_clap_sound(wave: np.ndarray, rate: int) -> list[ClapCandidate]:
    """Ranked clap-sound candidates, outliers gated out. Pure numpy."""

# --- c3d marker naming ---
def classify_clap_markers(
    labels: list[str],
) -> tuple[list[int], list[int]] | None:
    """(top_indices, bottom_indices) or None if names are ambiguous."""

# --- c3d motion ---
def detect_clap_motion(
    top_pts: np.ndarray,     # (n_frames, 3) centroid of top markers
    bottom_pts: np.ndarray,  # (n_frames, 3) centroid of bottom markers
    rate: float,
) -> ClapCandidate | None:
    """Gap-collapse / peak closing-velocity, occlusion-robust, sub-frame."""
```

### `detect_clap_sound` algorithm

All pure numpy — no scipy, no librosa (matches repo: torchaudio/numpy only).

1. **Candidate onsets** — pure-numpy STFT (frame → `np.fft.rfft`), spectral
   flux (sum of positive magnitude differences between frames), peak-pick the
   flux curve. Yields a handful of candidate onset times, not one.
2. **Features per candidate** over a short burst window (~30–50 ms) around the
   onset:
   - **Attack sharpness** — 10%→90% rise time of the RMS envelope
     (rectify + moving-RMS; no Hilbert). Short rise = clap.
   - **Duration** — time the envelope stays within −10 dB of peak (or
     decay-to-−20 dB time). Short burst = clap.
   - **Spectral flatness** (Wiener entropy = geometric-mean / arithmetic-mean
     of the burst power spectrum). High = broadband/noise-like = clap; the
     strongest discriminator against tonal voice/music.
   - **Crest factor** (peak / RMS) — high for impulsive signals.
3. **Gate** — hard-reject candidates outside plausible clap ranges on each axis
   (rise time, duration, flatness). Kills obvious non-claps.
4. **Score** survivors — weighted, normalized sub-score combination → ranked.
   Time is sub-sample refined (parabolic interp on the flux/envelope peak,
   reusing the existing `_parabolic_peak` idea).

Thresholds are module constants (like `EDGE_MIN_SCORE`), tuned against the
user's real file.

### Sync bridge (extends `src/clapsync/app/sync.py`)

The A/V MFCC path is unchanged. A new orchestration step runs **only when at
least one c3d input is present**:

1. Partition inputs into A/V and mocap, preserving original indices.
2. Guard: if the caller-supplied `reference_index` points at a c3d, reject
   (mocap has no audio; reference must be A/V).
3. Run the existing MFCC solve on A/V tracks. Handle `n == 1` A/V track (single
   video + one c3d — the primary use case): that track is trivially the
   reference at offset 0; do not call all-pairs correlation.
4. **Anchor `T_sound`** (feature quality × cross-track agreement):
   - Run `detect_clap_sound` on each A/V track's waveform.
   - Lift each candidate to shared time: `t_shared = t_local + offset_track`,
     using only tracks with good MFCC confidence (isolated/low-confidence
     tracks are excluded — their offset is untrustworthy).
   - Cluster shared-times across tracks. The anchor is the cluster with ≥2
     agreeing tracks (within tolerance) **and** the best summed feature score.
   - No agreeing cluster → no reliable anchor → all c3d offsets 0 + warning.
5. **Per c3d:**
   - Markers already resolved to `(top_idx, bottom_idx)` before this step (see
     GUI threading note). Compute top/bottom centroids per frame (ignoring
     invalid markers), run `detect_clap_motion` → `T_c3d`.
   - `offset_c3d = T_sound − T_c3d`.
   - No motion clap found → offset 0 + warning.
6. Merge A/V and c3d results back into offset / confidence / warning lists in
   **original input order** (parallel to the media list, as everything else in
   the codebase expects).

### Data model (`src/clapsync/app/media.py`)

- Extend `MediaInfo.kind` to `Literal["audio", "video", "mocap"]`.
- Add `point_rate: float | None`.
- `probe()` recognizes `.c3d`: returns `kind="mocap"`, `has_audio=False`,
  `duration = n_frames / point_rate`, `point_rate` set. Does not raise the
  existing "no stream" `ValueError` for c3d.

### Export (extends `src/clapsync/app/export.py`)

New `kind == "mocap"` branch in the `export_media` loop, parallel to
video/audio:

1. Parse the c3d (`load_c3d`).
2. Reuse the pure `clip_window(offset, duration, trim)` to get
   `(local_start, local_end, pad_start, pad_end)` in seconds.
3. **Points:** convert seconds → frame indices (`round(t * point_rate)`), slice
   the marker array, allocate a full-window buffer of
   `round(trim.duration * point_rate)` frames, place the core at
   `pad_front`, mark all pad frames invalid (residual −1). Mirrors
   `_build_audio_samples`.
4. **Analog:** trim on the analog rate (`point_rate * analog_per_frame`) in
   lockstep with points; pad analog to match. Points-only trimming would
   desync force-plate/EMG data.
5. **Events:** rebase event frame indices to the trimmed origin; drop events
   that fall outside the window.
6. Write `{stem}_synced.c3d` via `write_c3d`.

`ExportDialog` lists mocap tracks as exportable but excludes them from the
output-resolution / fps computation (currently derived from `kind=="video"`
only).

### GUI

- **File filter:** add `*.c3d` to the `MediaSelectionDialog` filter string.
- **Marker resolution before compute (threading fix):** parse each c3d and run
  `classify_clap_markers` on the **main thread, before** spawning the
  `OffsetWorker`. If classification returns `None`, show a modal
  `MarkerSelectionDialog` (list of labels → pick top set + bottom set) up
  front. Resolved `(top_idx, bottom_idx)` per c3d is passed into the worker.
  This avoids any cross-thread dialog from inside the worker (which would
  deadlock against the `processEvents` spin in `_run_with_progress`).
- **`C3DMarkerPreviewWidget(QOpenGLWidget)`** — animated 3D point cloud plus a
  top/bottom skeleton segment, driven by the playhead. Plugs into the
  `sync_editor.py` kind-branch (`~:167`) alongside `VideoPlayerWidget` /
  `WaveformWidget`; fed from `_on_position_changed`. Maps shared time → c3d
  frame: `frame = round((shared - offset) * point_rate)`, honoring
  `first_frame`. Skips invalid/NaN markers. Auto-fits the camera to marker
  bounds. **Wraps GL init in try/except; on failure falls back to a 2D
  QPainter-projected view** (portability: RDP, no-GPU, mixed installer
  hardware).
- **Kind icon:** add a mocap icon for `TrackState.kind` and the mute panel
  (`track_panel.py`, currently binary audio/video).

### CLI (`src/clapsync/cli.py`)

c3d files flow through the existing positional `inputs` once `probe()` accepts
them — no new required flag. `synctrim` gains c3d outputs automatically via the
export loop. Marker auto-classification runs headless; ambiguous names →
offset 0 + warning (no interactive dialog in CLI). Reject a c3d
`--reference`.

### Dependencies

- Add `c3d` (py-c3d) to `[tool.pixi.pypi-dependencies]` and the `app`
  optional-extra in `pyproject.toml`.
- Mirror into `installer/pixi.toml` so it ships in the Windows binary.
- Regenerate `pixi.lock` via `pixi install`.
- No new GUI dep: `QOpenGLWidget` ships with PySide6.

## Phasing (each phase independently verifiable)

**Phase A — parse + detect core.** `app/mocap.py` (`load_c3d`/`write_c3d`) and
`core/clap.py` (all three functions). **Validate py-c3d against the user's real
c3d first** (load + round-trip write) before building on it; fall back to ezc3d
if it fails. Unit tests with synthetic signals: impulse passes; sine burst
fails flatness; noise swell fails attack; speech-like fails flatness/duration;
motion detector on a synthetic gap-collapse (including an occlusion-dropout
case). Verify: tests green + real-file round-trip.

**Phase B — sync bridge + CLI.** Partition/merge with order preservation,
consistency-gated anchor, `n==1` guard, reference guard, `probe()` c3d support,
`MediaInfo` extension. Integration test: synthetic A/V + c3d with a known
injected delay confirms the bridge sign and value. Verify: `synctrim` on
real (video + c3d) produces a correct offset.

**Phase C — export.** mocap branch in `export_media` with points + analog +
events trimmed in lockstep and invalid-frame padding. Verify: exported
`{stem}_synced.c3d` loads in a c3d reader, frame count matches the trim window,
analog stays aligned, no spurious markers at origin.

**Phase D — GUI.** File filter, pre-compute marker resolution +
`MarkerSelectionDialog`, `C3DMarkerPreviewWidget` (GL + QPainter fallback),
kind icons, `ExportDialog` inclusion. Verify: load video + c3d in the app,
preview animates in sync with the playhead, export succeeds.

## Risks & mitigations (from premortem)

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Wrong clap transient → silent bad sync | Multi-feature gate + cross-track agreement anchor; disagreement → offset 0 + warn, never silent-trust |
| 2 | c3d bridge sign error | Integration test with known synthetic delay (Phase B); A/V convention already locked |
| 3 | py-c3d lossy round-trip (analog/events/units) | `write_c3d` preserves param groups/units; export trims analog + rebases events in lockstep |
| 4 | py-c3d fails on real mocap-system c3d | Validate user's real file in Phase A before building; ezc3d fallback |
| 5 | Primary case (1 video + 1 c3d) hits MFCC n==1 edge | Explicit `n==1` guard; first integration test |
| 6 | Marker dialog mid-worker deadlock | Resolve markers on main thread before spawning worker |
| 7 | Matcher too strict → auto never fires | `{clap,clapper,clapperboard,slate,cb}` × `{top/up/upper,down/bottom/low/lower}`, case-insensitive |
| 8 | QOpenGLWidget crashes on some hardware | try/except GL init → QPainter 2D fallback |
| 9 | Index/order + reference integrity | Preserve original order on merge; reject non-A/V reference |
| 10 | Anchor on low-confidence A/V track | Restrict anchor candidates to confidently-synced tracks |
| 11 | One giant PR | Phased A–D, each shippable |

## Test surface (pure, no hardware)

- `detect_clap_sound`: impulse / sine-burst / noise-swell / speech-like fixtures.
- `classify_clap_markers`: exact names, loose synonyms, ambiguous → None.
- `detect_clap_motion`: synthetic gap-collapse, occlusion dropout at impact.
- `clip_window` at mocap frame rate (existing pure fn, new inputs).
- Sync bridge: synthetic A/V + c3d known-delay integration test.
- Export: frame-count / analog-alignment / invalid-pad assertions on a
  round-tripped file.
