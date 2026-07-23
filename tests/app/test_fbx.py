"""FBX probe + trim tests, gated on bpy (Blender's Python API) being importable.

bpy is a heavy optional dependency; when absent the whole module skips. The
fixtures build a tiny in-memory rig (two dense per-frame-keyed empties, mirroring
how real optical mocap imports: object-transform channels keyed on every frame)
so the tests exercise the real trim path without a multi-megabyte asset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

bpy = pytest.importorskip("bpy")

from clapsync.app import fbx


FPS = 30
# Source anim starts at frame 2 (like real mocap fbx), spans 2..(2+N-1); the
# object's z equals its 0-based sample index, so source frame f has z = f - 2.
SRC0 = 2
N = 40


def _build_fixture(path: Path) -> None:
    """Write an fbx: two empties, z = sample index, keyed on every frame."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for name in ("A", "B"):
        obj = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(obj)
        for i in range(N):
            obj.location = (0.0, 0.0, float(i))
            obj.keyframe_insert("location", frame=SRC0 + i)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = SRC0
    scene.frame_end = SRC0 + N - 1
    bpy.ops.export_scene.fbx(
        filepath=str(path), bake_anim=True, bake_anim_use_all_bones=True,
        bake_anim_simplify_factor=0.0, add_leaf_bones=False,
    )


def _z_curve():
    """The location.z fcurve of the first empty in the loaded scene."""
    for action in bpy.data.actions:
        for fcurve in fbx._iter_fcurves(action):
            if fcurve.data_path.endswith("location") and fcurve.array_index == 2:
                if len(fcurve.keyframe_points):
                    return fcurve
    return None


def test_probe_fbx_reads_fps_and_length(tmp_path: Path):
    src = tmp_path / "src.fbx"
    _build_fixture(src)

    probe_fps, n_frames = fbx.probe_fbx(src)

    assert probe_fps == float(FPS)
    assert n_frames == N  # keyless placeholder actions must not extend the span


def test_trim_fbx_interior_exact_and_freeze_pads(tmp_path: Path):
    src = tmp_path / "src.fbx"
    dst = tmp_path / "dst.fbx"
    _build_fixture(src)

    # Keep source samples [10, 30) (z 10..29), pad 4 before / 3 after.
    f0, keep, pad_front, pad_back = 10, 20, 4, 3
    local_start = f0 / FPS
    local_end = (f0 + keep) / FPS
    pad_start = pad_front / FPS
    pad_end = pad_back / FPS
    total = round((pad_start + (local_end - local_start) + pad_end) * FPS)

    fbx.trim_fbx(src, dst, (local_start, local_end, pad_start, pad_end), FPS)

    fbx._reset_and_import(dst)
    curve = _z_curve()

    # Scene range is not round-tripped through fbx; the baked animation spans
    # output frames 1..total, which is what must match round(trim.duration*fps).
    key_xs = [round(kp.co.x) for kp in curve.keyframe_points]
    assert min(key_xs) == 1
    assert max(key_xs) == total

    first_z, last_z = float(f0), float(f0 + keep - 1)  # z at kept ends

    # Leading pad holds the first kept pose.
    for frame in range(1, pad_front + 1):
        assert curve.evaluate(frame) == pytest.approx(first_z, abs=1e-3)
    # Interior matches the source window sample-for-sample.
    for k in range(keep):
        out_frame = pad_front + 1 + k
        assert curve.evaluate(out_frame) == pytest.approx(float(f0 + k), abs=1e-3)
    # Trailing pad holds the last kept pose.
    for frame in range(pad_front + keep + 1, total + 1):
        assert curve.evaluate(frame) == pytest.approx(last_z, abs=1e-3)
