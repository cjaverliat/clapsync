import torch

import clapsync.core.sync as sync_mod
from clapsync.core.media import MediaInfo
from clapsync.core.sync import compute_sync_offsets


def _info(name, dur, fps=25.0):
    return MediaInfo(path=name, duration=dur, has_audio=True,
                     kind="video", sample_rate=48000, width=8, height=8, fps=fps)


def test_reference_offset_is_zero_and_others_signed(monkeypatch):
    media = [_info("a", 5.0), _info("b", 5.0), _info("c", 5.0)]
    # Record which track's audio is being loaded so find_offset can map lags.
    current = {"name": None}

    def fake_load(path, target_rate=None):
        current["name"] = path
        return torch.zeros(1, 100), 48000

    lags = {"b": 0.5, "c": -0.3}  # b lags ref +0.5 s, c leads -0.3 s

    def fake_find(rw, rr, w, r, fps, method, refine):
        return 0, lags[current["name"]]

    monkeypatch.setattr(sync_mod, "load_audio", fake_load)
    monkeypatch.setattr(sync_mod, "find_offset", fake_find)

    offsets = compute_sync_offsets(media)
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 0.5) < 1e-9
    assert abs(offsets[2] + 0.3) < 1e-9


def test_progress_reaches_one(monkeypatch):
    media = [_info("a", 1.0), _info("b", 1.0)]
    monkeypatch.setattr(sync_mod, "load_audio",
                        lambda path, target_rate=None: (torch.zeros(1, 10), 48000))
    monkeypatch.setattr(sync_mod, "find_offset", lambda *a, **k: (0, 0.1))
    seen = []
    compute_sync_offsets(media, progress=seen.append)
    assert seen and seen[-1] == 1.0
