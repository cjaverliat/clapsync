"""fbx.py must route all bpy work through an injectable executor.

bpy is main-thread-only; the GUI runs sync/export on worker threads, so the
bpy calls in fbx.py are submitted to a swappable executor (default: inline) that
the GUI replaces with a main-thread marshaller. These tests pin that contract
without needing bpy itself — they assert the work is *submitted*, not run inline.
"""
from __future__ import annotations

from pathlib import Path

from clapsync.app import fbx


def test_probe_fbx_submits_work_to_executor(monkeypatch):
    seen = []
    monkeypatch.setattr(fbx, "_bpy_executor", lambda fn: seen.append(fn) or (99.0, 7))

    result = fbx.probe_fbx(Path("nonexistent.fbx"))

    assert result == (99.0, 7)  # returns whatever the executor returns
    assert len(seen) == 1 and callable(seen[0])  # bpy body submitted, not run


def test_trim_fbx_submits_work_to_executor(monkeypatch):
    seen = []
    monkeypatch.setattr(fbx, "_bpy_executor", lambda fn: seen.append(fn))

    fbx.trim_fbx(Path("a.fbx"), Path("b.fbx"), (0.0, 1.0, 0.0, 0.0), 30.0)

    assert len(seen) == 1 and callable(seen[0])


def test_set_and_reset_bpy_executor(monkeypatch):
    calls = []
    fbx.set_bpy_executor(lambda fn: calls.append(fn) or (1.0, 1))
    try:
        fbx.probe_fbx(Path("z.fbx"))
        assert len(calls) == 1
    finally:
        fbx.reset_bpy_executor()
    # After reset the default inline executor runs the body (which imports bpy);
    # only assert the executor slot was restored, not that bpy is present.
    assert fbx._bpy_executor is fbx._run_inline
