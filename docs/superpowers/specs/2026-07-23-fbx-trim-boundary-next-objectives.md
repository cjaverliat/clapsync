# FBX trim boundary fix + remaining plumbing — next objectives

**Date:** 2026-07-23
**Status:** open — resume here
**Feature spec:** `2026-07-23-fbx-mocap-tracks-design.md`

Foundations are committed (`75de06f`): the py3.13 + bpy 5.2 env, the PySide6 6.8
worker-signal fix, the `track_family`/`is_av` model + `.fbx` kind, and a working
`probe_fbx()`. What remains is (1) getting `trim_fbx()` correct and (2) wiring
the feature through the app. This doc captures the hard-won bpy findings so the
next session doesn't rediscover them.

## Objective 1 — fix the fbx trim boundary (the blocker)

`app/fbx.py::trim_fbx` must trim an fbx animation to the shared export window and
freeze-pad it (hold the first pose backward into a leading gap, the last pose
forward into a trailing gap), frame-aligned with the padded c3d export. It does
not yet land correctly on Blender 5's animation model.

### Verified bpy 5.2 (Blender 5.0) facts

- **Slotted actions.** `action.fcurves` is **gone**. Fcurves live at
  `action.layers[].strips[].channelbags[].fcurves`. `_iter_fcurves()` in
  `app/fbx.py` already handles both this and the legacy path — reuse it.
- **`action.frame_range`** still exists (read-only, derived from keyframes) and
  is what `probe_fbx` uses successfully.
- **FBX export animation modes** (`bpy.ops.export_scene.fbx`):
  - Default `bake_anim_use_all_actions=True` **does** export animation, but
    bakes each action over **its own** frame range — it ignores `scene.frame_start/end`.
  - `bake_anim_use_all_actions=False` **exported no animation at all** in
    headless tests (slotted-action + exporter interaction). Avoid for now.
  - Setting the **action's manual frame range** — `action.use_frame_range=True;
    action.frame_start=f0; action.frame_end=f1` — **is honored** by the default
    export mode and is the most promising lever: it produced the correct output
    **frame count** and **auto-rebased** the output to start at frame 1.

### The remaining bug

With `action.use_frame_range` set to a 6→24 window on a linear test animation
(object z = 0→30 over source frames 1→31, so source frame f has z = f−1):

| output frame | expected z (source 6/15/24) | actual z |
|---|---|---|
| 1  | 5.0  | 2.2  |
| 10 | 14.0 | 13.5 |
| 19 | 23.0 | 25.9 |

Output range `(1, 19)` = 19 frames is **correct** (24−6+1). But the content is
offset ~3 frames at the boundaries (output frame 1 samples ~source frame 3.2;
ends overshoot). `bake_anim_simplify_factor=0.0` did **not** fix it, so it is not
curve simplification — it is a bake-boundary/margin behavior in the exporter.

### Things to try next (roughly in order)

1. **Test against a real skeletal-mocap fbx**, not the synthetic cube rig. The
   cube animates the object transform; real mocap animates pose bones, which the
   FBX baker treats differently. The offset may be a rig artifact. This is the
   single most important next step — get a real fbx from the user.
2. **Inspect the baked keyframes directly** in the trimmed action (positions +
   values of `keyframe_points`) instead of `fcurve.evaluate()`, to separate a
   key-placement error from an interpolation/handle artifact.
3. **`bake_anim_step`** and any bake-margin: confirm step=1.0 and look for an
   exporter option that pads the bake range.
4. **Drop `use_frame_range`, delete out-of-window keyframes + insert exact
   boundary keys, then export** with default bake — make the action's *natural*
   `frame_range` equal the window rather than relying on the manual range.
5. **Rebase + freeze-pad** once the core clip is exact:
   - rebase so the window start lands at output frame `pad_front` (currently the
     manual-range export rebases to frame 1 — reconcile with `pad_front`);
   - set every fcurve `extrapolation='CONSTANT'` and extend the exported range to
     the full window (`total = round(trim.duration * fps)`) so pad regions bake
     as the held pose. `_iter_fcurves` already sets CONSTANT — verify it takes.
6. **Success test (gated on bpy):** build a small pose-bone-animated armature,
   trim to a window with non-zero `pad_start`/`pad_end`, re-import, and assert:
   output frame count = `round(trim.duration*fps)`; interior frames match the
   source window sample-for-sample; leading/trailing pad frames equal the
   first/last captured pose (freeze).

### Reference: the pure window math (already correct)

`export.clip_window(offset, duration, trim)` → `(local_start, local_end,
pad_start, pad_end)` in seconds. `local_end` is an absolute local time, not a
length. Frames: `f0 = round(local_start*fps)`, `pad_front = round(pad_start*fps)`,
`total = round((pad_start + (local_end-local_start) + pad_end) * fps)` =
`round(trim.duration*fps)`.

## Objective 2 — route the track-family model through the kind gates

Replace each A/V-only `kind != "mocap"` / `kind == "mocap"` check with
`is_av(...)` / `not is_av(...)` so `fbx` is excluded from audio sync exactly like
`c3d`:

- `sync_editor._reference_index` — reference must be A/V.
- `app/sync.align_media` — only A/V tracks are MFCC-aligned.
- `mocap_sync._sound_candidates` — A/V audio only (make the kind gate explicit).
- `export.sync_and_trim` default-trim `av` list; `sync_editor`/timeline
  default-trim overlap.
- `sync_editor._rebuild_clap_markers` av_tracks (sound-flag lanes).

Add a unit test: a project containing an fbx picks an A/V reference, excludes fbx
from the default trim, and leaves it out of MFCC alignment.

## Objective 3 — fbx offset inheritance

In `app/mocap_sync.py` (or its caller): after the c3d offset is computed, set each
fbx offset **= the c3d offset** (single-c3d assumption). No c3d → offset 0,
confidence 0, warning. Unit test: fbx offset == c3d offset; no-c3d → warning.

## Objective 4 — GUI (timeline lane + family grouping)

- fbx probed via `probe_fbx` at add time (needs bpy — keep the import deferred;
  probing several fbx opens/closes the bpy scene each time, so consider batching
  or a progress note if slow).
- fbx = timeline lane only: no mosaic cell (skip like `kind == "audio"`), no
  waveform, no preview.
- **Offset slaving:** fbx offset always tracks its c3d. Not independently
  draggable. Add `_sync_fbx_to_c3d(offsets)` and call it in `_on_offsets_changed`
  and `_apply_clap_link` before propagating.
- **Visual family grouping:** order lanes A/V first, then the mocap family with
  each fbx directly under its c3d. This is *display* only — decouple the lane row
  from `track.index` (today `_track_rect` positions by `track.index`); introduce
  a `display_order` list grouped by family and position by row in it, keeping
  `track.index` for all data lookups. The track panel iterates the same order.
  Unit test: order is A/V then mocap-with-fbx-under-c3d; `track.index` unchanged.

## Objective 5 — export dispatch

`export.export_media`: dispatch `kind == "fbx"` to `_export_fbx_track(info,
offset, trim, out_path)` → `fbx.trim_fbx(src, dst, clip_window(...), fps)`,
output `{stem}_synced.fbx`.

## Known caveats to carry forward

- **PySide6 6.8 footgun:** queued signals emitted from a non-Qt thread (e.g. the
  decode python thread) are silently dropped. All worker→GUI signals now route
  through the QThread's `run()` loop (see `video_player.py`). Any *new*
  worker signal must follow the same pattern.
- **bpy in the GUI process:** bpy 5.2 coexists with PySide6 6.8 + torch (verified
  at import). Keep all bpy usage confined to `app/fbx.py`, imported lazily.
- **triton 3.6** (bumped from 3.5 for the cp313 wheel) builds the framepipe
  NV12→RGB kernel fine; GPU perf holds 30 fps. Watch for the `alloca LNK2019` the
  old pin comment warned about if triton is bumped further.
