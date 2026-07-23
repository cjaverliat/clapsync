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


def _anim_frame_range(bpy) -> tuple[float, float]:
    """Keyframe span across every imported action, in frames (start, end)."""
    lo: float | None = None
    hi: float | None = None
    for action in bpy.data.actions:
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

    WIP — the keyframe-shift + scene-range + CONSTANT-extrapolation approach
    below does not yet land correctly on Blender 5's slotted actions + FBX
    exporter (a ~few-frame boundary offset in the baked output). See
    docs/superpowers/specs/2026-07-23-fbx-trim-boundary-next-objectives.md for
    the findings and the plan to finish this. Not wired into export yet.


    The window comes from ``export.clip_window`` (seconds): ``local_start`` and
    ``local_end`` are absolute local times bounding the kept span; ``pad_start``
    and ``pad_end`` are the gaps outside the source that must be filled. The kept
    span is shifted so ``local_start`` lands at output frame ``pad_start``, every
    fcurve is set to CONSTANT extrapolation (which holds the first pose backward
    into a leading gap and the last pose forward into a trailing gap), and the
    scene range is set to the full window so the exporter bakes the held pose
    across the pads. Frame-aligned with the padded c3d export.

    Args:
        src: Source fbx path.
        dst: Destination fbx path.
        window: (local_start, local_end, pad_start, pad_end) in seconds.
        fps: Animation frame rate (from ``probe_fbx``).
    """
    local_start, local_end, pad_start, pad_end = window
    bpy = _reset_and_import(Path(src))

    f0 = round(local_start * fps)
    pad_front = round(pad_start * fps)
    total = round((pad_start + (local_end - local_start) + pad_end) * fps)
    delta = pad_front - f0

    for action in bpy.data.actions:
        for fcurve in _iter_fcurves(action):
            for keyframe in fcurve.keyframe_points:
                keyframe.co.x += delta
                keyframe.handle_left.x += delta
                keyframe.handle_right.x += delta
            fcurve.extrapolation = "CONSTANT"
            fcurve.update()

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = total
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0

    bpy.ops.export_scene.fbx(
        filepath=str(dst),
        bake_anim=True,
        bake_anim_use_all_bones=True,
        # Bake the active action across the *scene* range (not each action's own
        # range), so the trimmed/rebased window and its freeze-pads are what gets
        # sampled.
        bake_anim_use_all_actions=False,
        bake_anim_step=1.0,
        add_leaf_bones=False,
    )
