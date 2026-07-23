# FBX mocap animation tracks — design

**Date:** 2026-07-23
**Status:** approved (brainstorming)

## Problem

Users capture mocap that produces two artifacts from the same take: a `.c3d`
marker file (used today for clap-based sync) and a `.fbx` skeletal animation.
The `.fbx` currently cannot be added to a project. We want to add `.fbx` tracks
that ride the shared timeline by inheriting the c3d's clap-derived offset, and
to trim them to the export window using bpy (Blender's Python API).

## Key decisions

- **FBX ↔ c3d:** same take. An fbx inherits its c3d's shared-timeline offset
  exactly — no independent sync. fbx frame 0 = c3d frame 0.
- **Multiplicity:** single-c3d assumption. Every fbx follows *the* c3d. If no
  c3d is present, warn and leave the fbx unsynced (offset 0, low confidence).
- **Preview:** none. fbx is a timeline lane only (no mosaic cell, no waveform,
  no skeleton render). The c3d marker preview already visualizes the motion.
- **bpy:** in-process, via a pixi env on Python 3.11 + bpy 5.0. Used for fbx
  metadata read (probe) and fbx trim (export).
- **Export trim:** pad to the full trim window, freezing the pose — hold the
  first captured pose backward into a leading gap and the last pose forward into
  a trailing gap. Matches the c3d export (which already pads to the full window),
  so the two same-take outputs stay frame-aligned.
- **A/V neutrality:** fbx is non-A/V, exactly like c3d. It is excluded from
  every audio/video sync path (reference selection, MFCC alignment, sound
  anchor, default trim range, clap sound-marker lanes).
- **Track families:** two families — **A/V** (`audio`, `video`) and **mocap**
  (`c3d`, `fbx`). The family drives sync/trim logic *and* the visual layout:
  tracks are ordered by family (A/V first, then mocap), with each fbx placed
  immediately under its c3d parent (the fbx is enslaved to it).

## Components

### 1. Environment (prerequisite — gated)

Bump the pixi env Python **3.12 → 3.11** and add **bpy 5.0** (the cp311 wheel;
bpy has no cp312 build). `pyproject.toml`: set the Python constraint to
`>=3.11,<3.12`, add `bpy>=5,<5.1` to the `app` dependency group. The
`libpython-static >=3.10.20,<4` pin already spans 3.11.

**Gate — verify before writing any feature code:**
- torch + CUDA, PyAV, framepipe, PySide6 all resolve and install on 3.11.
- `python -c "import PySide6; import bpy"` succeeds in one process (headless
  bpy must coexist with Qt).
- The full test suite passes, including the slow GPU perf test
  (`test_playback_sustains_30fps`) — the GPU decode/encode path must be intact.

If the GPU stack breaks on 3.11, stop and reassess (fall back to a dedicated
bpy subprocess env) rather than pushing feature code onto a broken base.

### 2. Probe (`app/media.py` + new `app/fbx.py`)

`probe()` maps `.fbx` → `kind="fbx"`. New module `app/fbx.py` owns all bpy
usage; `import bpy` is deferred *inside* functions (mirrors the deferred
torch/framepipe imports), so merely importing clapsync stays cheap.

```
def probe_fbx(path: Path) -> tuple[float, int]:
    """(fps, n_frames) for an fbx's animation. Resets the bpy scene first."""
```

`MediaInfo` for an fbx carries `duration = n_frames / fps` and `fps`. Every
bpy entry point calls `bpy.ops.wm.read_factory_settings(use_empty=True)` before
importing, so bpy's global scene state never leaks between calls.

### 3. Track families (`app/media.py`)

Model two families and route every family-dependent check through them:

```
def track_family(kind: str) -> str:
    """"av" for audio/video (drive audio sync); "mocap" for c3d/fbx."""
    return "av" if kind in ("audio", "video") else "mocap"

def is_av(kind: str) -> bool:
    return track_family(kind) == "av"
```

`"mocap"` and `"fbx"` are the mocap family — non-A/V "supplementary" tracks
placed by the clap link. Replace existing `kind != "mocap"` / `kind == "mocap"`
A/V gates with `is_av(...)` / `not is_av(...)` at:

- `sync_editor._reference_index` — reference must be A/V.
- `app/sync.align_media` — only A/V tracks are MFCC-aligned; mocap/fbx are
  bridged afterward.
- `mocap_sync._sound_candidates` — A/V audio only (already excludes fbx via
  `has_audio`, but make the kind gate explicit).
- `export.sync_and_trim` default-trim `av` list and `sync_editor`/timeline
  default-trim overlap — computed over A/V tracks only.
- `sync_editor._rebuild_clap_markers` av_tracks (sound-flag lanes).

### 4. Offset inheritance (`app/mocap_sync.py`)

fbx has no audio and no markers, so it cannot self-sync. After the c3d
offset(s) are computed, assign each fbx offset **= the c3d offset**. With the
single-c3d assumption there is one c3d offset to inherit.

- One c3d present → fbx offset = that c3d's offset; confidence = the c3d's
  synced confidence.
- No c3d present → offset 0, confidence 0, warning
  (`"<fbx>: no c3d to sync against — left unsynced (offset 0)"`).

This lives alongside `bridge_mocap_offsets`; the bridge (or its caller) emits
offsets/confidence/warnings for fbx tracks parallel to the c3d handling.

### 5. GUI (`sync_editor.py`, `timeline_widget.py`)

fbx tracks join `_video_infos` with `kind="fbx"`. They render as a **timeline
lane only** — no mosaic cell (skipped like `kind == "audio"`), no waveform, no
preview widget. The track panel lists name + kind.

**Offset slaving:** an fbx's offset is always its c3d's offset. The fbx lane is
not independently draggable. Whenever the c3d offset changes — clap-link apply,
c3d drag, `offsets_changed` — the fbx offset is updated to match before the
change propagates to audio/timeline/export. Concretely, a
`_sync_fbx_to_c3d(offsets)` step rewrites fbx entries to the c3d offset inside
`_on_offsets_changed` and `_apply_clap_link`.

### 5b. Visual grouping (`sync_editor.py`, `timeline_widget.py`, `track_panel.py`)

Lanes are ordered by family, not input order: **A/V tracks first, then the
mocap family**, with each fbx placed immediately under its c3d parent. Under the
single-c3d assumption this is `[a/v tracks…, c3d, fbx…]`.

The order is a *display* concern only — the canonical track index (used by
`offsets`, clap markers, export) must not change. Today the timeline positions a
lane vertically by `track.index` (`_track_rect`: `y = HEADER_H + index *
(TRACK_H + TRACK_PAD)`). To reorder visually without touching indices, decouple
display row from index: compute a `display_order` list of track indices grouped
by family, and position lanes by their row in that list while keeping
`track.index` intact for all data lookups. The track panel iterates the same
`display_order`. Mosaic cells (A/V video only) are unaffected.

### 6. Export (`app/export.py` → `_export_fbx_track` → `app/fbx.py`)

`export_media` dispatches `kind == "fbx"` to `_export_fbx_track(info, offset,
trim, out_path)`, output `{stem}_synced.fbx`. Window math reuses the pure
`clip_window(offset, duration, trim)` → (local_start, local_end, pad_start,
pad_end); convert seconds→frames with the fbx fps.

`app/fbx.py` trim algorithm (bpy, in-process):

```
def trim_fbx(src, dst, window, fps) -> None:
    # window = (local_start, local_end, pad_start, pad_end) in seconds,
    # from clip_window(); local_end is an absolute local time, not a length.
```

1. `read_factory_settings(use_empty=True)`; `import_scene.fbx(src)`.
2. Read source fps; frames: `f0 = round(local_start*fps)`,
   `pad_front = round(pad_start*fps)`, and total output frames
   `total = round((pad_start + (local_end - local_start) + pad_end) * fps)`
   — which equals `round(trim.duration * fps)` by construction.
3. Shift every fcurve keyframe by `delta = pad_front - f0`, so the source
   window start lands at output frame `pad_front`.
4. Set every fcurve's extrapolation to **CONSTANT** — this *is* the freeze:
   before the first key and after the last key the pose is held.
5. `scene.frame_start = 0`, `scene.frame_end = total` — the full window,
   spanning the pad regions.
6. `export_scene.fbx(dst, bake_anim=True, bake_anim_step=1,
   bake_anim_use_all_bones=True)` over the scene range. The exporter samples
   every frame; pad regions come out as the held pose automatically.

Frame-aligned with the padded c3d output (same fps → same frame count/origin
when the c3d and fbx share point/anim rate; if rates differ, both still span
the same shared window).

No-c3d fbx (offset 0, unsynced) still exports with offset 0 as a best effort;
the unsynced warning was surfaced at sync time.

## Data flow

```
add files ─▶ probe() ─▶ MediaInfo(kind=fbx, duration, fps)
                          │
align_media (A/V only) ─▶ av offsets
                          │
bridge_mocap_offsets ─▶ c3d offset ─▶ fbx offset := c3d offset
                          │
GUI: c3d moves ─▶ _sync_fbx_to_c3d ─▶ fbx follows
                          │
export_media ─▶ _export_fbx_track ─▶ trim_fbx (bpy) ─▶ {stem}_synced.fbx
```

## Testing

- `clip_window` window math — already covered.
- `track_family` routing — unit tests that a project with an fbx picks an A/V
  reference, excludes fbx from the default trim, and leaves fbx out of MFCC
  alignment.
- `display_order` — unit test: given mixed inputs, the order is A/V tracks then
  the mocap family with each fbx directly under its c3d, while `track.index`
  values are unchanged.
- Offset inheritance — unit test: fbx offset == c3d offset; no-c3d → warning +
  offset 0.
- fbx trim — gated integration test (`pytest.mark` on bpy availability). bpy
  builds a tiny armature with two keyframes, `trim_fbx` runs, the result is
  re-imported and asserted: output frame range = full window, leading/trailing
  pad frames equal the first/last captured pose (freeze), interior keys shifted
  correctly.

## Risks / limitations

- **3.11 GPU-stack rebuild** — the single biggest risk; the step-1 gate must
  pass before anything else. Fallback: dedicated bpy subprocess env.
- **bpy + PySide6 in one process** — verified in step 1.
- **fbx variety** — v1 handles the common case (armature action / fcurve
  animation). Exotic NLA stacks, multiple stacked actions, or shape-key-only
  animation are out of scope for v1 and noted as a limitation.
- **fps mismatch** — if the fbx anim fps differs from the c3d point rate, the
  two outputs span the same shared window but have different frame counts;
  acceptable (each is internally consistent and time-aligned).

## Out of scope (v1)

- fbx skeleton preview / mosaic rendering.
- Multiple c3d takes with per-fbx parent selection.
- Independent fbx sync (fbx that is not the same take as a c3d).
- Per-fbx local frame nudge on top of the c3d offset.
