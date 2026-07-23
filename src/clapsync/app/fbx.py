"""FBX mocap animation I/O via in-process bpy (Blender's Python API).

An fbx carries the same take as a c3d, so it rides the shared timeline by
inheriting the c3d's clap-derived offset (see app.mocap_sync). This module only
reads an fbx's animation metadata (to place its timeline lane) and trims it to
the export window; the offset math lives in the pure export window helpers.

bpy is a heavy import with global scene state, so it is imported lazily inside
each function and the scene is reset before every load — nothing leaks between
calls. All bpy usage in the project is confined here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _reset_and_import(path: Path):
    """Clear bpy's global scene and import ``path``; return the bpy module."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(path))
    return bpy


def _iter_fcurves(action):
    """Yield every fcurve of an action across Blender's action models.

    Blender 4.4+ moved fcurves into slotted actions
    (action.layers[].strips[].channelbags[].fcurves); older versions exposed
    action.fcurves directly. Support both.
    """
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                yield from channelbag.fcurves


def _crop_and_shift(fcurve, lo: int, hi: int, delta: int) -> None:
    """Keep an fcurve's keyframes in [lo, hi] and shift them by ``delta`` frames.

    A real mocap fcurve carries a key on every frame — tens of thousands, times
    a hundred objects — so a per-key Python loop crosses the C boundary millions
    of times. The keys are read, filtered, and rebuilt as flat numpy arrays via
    ``foreach_get``/``foreach_set`` (a handful of boundary crossings total). Only
    ``co`` is carried: the export bakes one sample per integer frame where a key
    already sits, so interpolation handles never affect the output. CONSTANT
    extrapolation holds the first/last kept key across the pad regions.
    """
    keyframes = fcurve.keyframe_points
    n = len(keyframes)
    fcurve.extrapolation = "CONSTANT"
    if n == 0:
        return
    co = np.empty(n * 2, dtype=np.float64)
    keyframes.foreach_get("co", co)
    co = co.reshape(n, 2)
    kept = co[(co[:, 0] >= lo - 0.5) & (co[:, 0] <= hi + 0.5)]
    kept[:, 0] += delta
    keyframes.clear()
    if kept.shape[0]:
        keyframes.add(kept.shape[0])
        keyframes.foreach_set("co", kept.ravel())
    fcurve.update()


def _has_keys(action) -> bool:
    """True if any fcurve of ``action`` carries a keyframe."""
    return any(len(fc.keyframe_points) for fc in _iter_fcurves(action))


def _anim_frame_range(bpy) -> tuple[float, float]:
    """Keyframe span across every keyed action, in frames (start, end).

    Mocap fbx often carry a keyless placeholder action (e.g. a static "Position"
    slot with frame_range (0, 0)); it must not drag the span origin to frame 0,
    so keyless actions are skipped.
    """
    lo: float | None = None
    hi: float | None = None
    for action in bpy.data.actions:
        if not _has_keys(action):
            continue
        start, end = action.frame_range
        lo = start if lo is None else min(lo, start)
        hi = end if hi is None else max(hi, end)
    if lo is None:
        scene = bpy.context.scene
        return float(scene.frame_start), float(scene.frame_end)
    return float(lo), float(hi)


def probe_fbx(path: Path) -> tuple[float, int]:
    """Read an fbx's animation frame rate and length.

    Args:
        path: FBX file path.

    Returns:
        (fps, n_frames): animation frames per second and the number of frames
        spanned by the animation (inclusive of both ends).
    """
    bpy = _reset_and_import(Path(path))
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    start, end = _anim_frame_range(bpy)
    n_frames = int(round(end - start)) + 1
    return float(fps), n_frames


def trim_fbx(
    src: Path,
    dst: Path,
    window: tuple[float, float, float, float],
    fps: float,
) -> None:
    """Trim + freeze-pad an fbx animation to the shared export window.

    Mirrors the c3d trim (``export._export_mocap_track``): the source frames
    covering the kept span are shifted so the window's first frame lands at
    output frame ``pad_front + 1`` (output numbering restarts at 1, as c3d's
    ``first_frame``), and the leading/trailing pad frames are filled by holding
    the first/last captured pose (CONSTANT extrapolation). The result is
    frame-aligned with the padded c3d export.

    The window comes from ``export.clip_window`` (seconds): ``local_start`` and
    ``local_end`` are absolute local times bounding the kept span; ``pad_start``
    and ``pad_end`` are the gaps outside the source that must be filled. Frame
    indices are taken relative to the animation's own start frame (mocap fbx
    commonly begin at frame 2, not 1).

    Each action's manual frame range is set to the full output window so the FBX
    exporter's per-action bake samples the shifted keys plus the CONSTANT-held
    pads (the default ``bake_anim_use_all_actions`` bakes each action over its
    own range; ``use_frame_range`` overrides that range and is honored).

    Args:
        src: Source fbx path.
        dst: Destination fbx path.
        window: (local_start, local_end, pad_start, pad_end) in seconds.
        fps: Animation frame rate (from ``probe_fbx``).
    """
    local_start, local_end, pad_start, pad_end = window
    bpy = _reset_and_import(Path(src))

    src0 = int(round(_anim_frame_range(bpy)[0]))
    f0 = round(local_start * fps)
    f1 = round(local_end * fps)
    pad_front = round(pad_start * fps)
    total = round((pad_start + (local_end - local_start) + pad_end) * fps)

    lo = src0 + f0  # first kept source frame (inclusive)
    hi = src0 + f1 - 1  # last kept source frame (inclusive)
    delta = (pad_front + 1) - lo  # shift lo -> output frame pad_front + 1

    for action in bpy.data.actions:
        for fcurve in _iter_fcurves(action):
            _crop_and_shift(fcurve, lo, hi, delta)
        if hasattr(action, "use_frame_range"):
            action.use_frame_range = True
            action.frame_start = 1
            action.frame_end = total

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = total
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0

    bpy.ops.export_scene.fbx(
        filepath=str(dst),
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_step=1.0,
        # No key reduction: mocap fidelity over file size.
        bake_anim_simplify_factor=0.0,
        add_leaf_bones=False,
    )
