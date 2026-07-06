import torch

import clapsync.app.sync as sync_mod
from clapsync.app.media import MediaInfo
from clapsync.app.sync import compute_sync_offsets


def _info(name, dur, fps=25.0):
    return MediaInfo(path=name, duration=dur, has_audio=True,
                     kind="video", sample_rate=48000, width=8, height=8, fps=fps)


def test_reference_offset_is_zero_and_others_signed(monkeypatch):
    media = [_info("a", 5.0), _info("b", 5.0), _info("c", 5.0)]

    def fake_load(path, target_rate=None):
        return torch.zeros(1, 100), 48000

    def fake_align(waveforms, rates, *, refine="parabolic", reference_index=0, progress=None):
        return [0.0, 0.5, -0.3]

    monkeypatch.setattr(sync_mod, "load_audio", fake_load)
    monkeypatch.setattr(sync_mod, "align_waveforms", fake_align)

    offsets = compute_sync_offsets(media)
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 0.5) < 1e-9
    assert abs(offsets[2] + 0.3) < 1e-9


def test_progress_reaches_one(monkeypatch):
    media = [_info("a", 1.0), _info("b", 1.0)]
    monkeypatch.setattr(sync_mod, "load_audio",
                        lambda path, target_rate=None: (torch.zeros(1, 10), 48000))
    monkeypatch.setattr(sync_mod, "align_waveforms",
                        lambda *a, **k: [0.0, 0.1])
    seen = []
    compute_sync_offsets(media, progress=seen.append)
    assert seen and seen[-1] == 1.0
