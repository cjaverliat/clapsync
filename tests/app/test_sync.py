from pathlib import Path

import torch

import clapsync.app.sync as sync_mod
from clapsync.app.media import MediaInfo
from clapsync.app.sync import compute_sync_offsets
from clapsync.core.solver import Alignment


def _info(name, dur, fps=25.0):
    return MediaInfo(path=Path(name), duration=dur, has_audio=True,
                     kind="video", sample_rate=48000, width=8, height=8, fps=fps)


def test_reference_offset_is_zero_and_others_signed(monkeypatch):
    paths = [Path("a"), Path("b"), Path("c")]

    monkeypatch.setattr(sync_mod, "probe", lambda p: _info(str(p), 5.0))

    def fake_load(path, target_rate=None, progress=None):
        return torch.zeros(1, 100), 48000

    def fake_align(waveforms, rates, *, refine="parabolic", reference_index=0, progress=None):
        return Alignment([0.0, 0.5, -0.3], [float("inf"), 42.0, 37.0], [])

    monkeypatch.setattr(sync_mod, "load_audio", fake_load)
    monkeypatch.setattr(sync_mod, "align_waveforms", fake_align)

    alignment = compute_sync_offsets(paths)
    offsets = alignment.offsets
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 0.5) < 1e-9
    assert abs(offsets[2] + 0.3) < 1e-9
    assert alignment.confidence[1] == 42.0


def test_media_is_reused_and_probe_not_called(monkeypatch):
    paths = [Path("a"), Path("b")]
    pre = [_info("a", 5.0), _info("b", 5.0)]

    def _boom(_p):
        raise AssertionError("probe must not run when media is supplied")

    monkeypatch.setattr(sync_mod, "probe", _boom)
    monkeypatch.setattr(
        sync_mod, "load_audio",
        lambda *a, **k: (torch.zeros(1, 10), 48000),
    )
    monkeypatch.setattr(
        sync_mod, "align_waveforms",
        lambda *a, **k: Alignment([0.0, 0.2], [float("inf"), 9.0], []),
    )

    alignment = compute_sync_offsets(paths, media=pre)

    assert alignment.offsets[0] == 0.0


def test_progress_reaches_one(monkeypatch):
    paths = [Path("a"), Path("b")]

    monkeypatch.setattr(sync_mod, "probe", lambda p: _info(str(p), 1.0))

    def fake_load(path, target_rate=None, progress=None):
        if progress is not None:
            progress(1.0)  # this file fully loaded
        return torch.zeros(1, 10), 48000

    def fake_align(*a, progress=None, **k):
        if progress is not None:
            progress(1.0)
        return Alignment([0.0, 0.1], [float("inf"), 50.0], [])

    monkeypatch.setattr(sync_mod, "load_audio", fake_load)
    monkeypatch.setattr(sync_mod, "align_waveforms", fake_align)
    seen = []
    compute_sync_offsets(paths, progress=seen.append)
    assert seen and seen[-1] == 1.0
    # Each file drives the bar 0..1 on its own (reset then climb), so 1.0
    # appears once per file, not just at the very end.
    assert seen.count(1.0) >= len(paths)
