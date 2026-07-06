import numpy as np
import torch

from clapsync.core.offsets import align_waveforms


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_align_waveforms_reference_zero_and_signed_offsets():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)                    # 1.000 s
    later = _click_track(n, click_at=sr + int(0.20 * sr))  # +200 ms (delayed)
    earlier = _click_track(n, click_at=sr - int(0.10 * sr))  # -100 ms (leads)

    offsets = align_waveforms([ref, later, earlier], [sr, sr, sr])
    assert offsets[0] == 0.0
    assert abs(offsets[1] - (-0.20)) < 0.02   # delayed -> negative
    assert abs(offsets[2] - (0.10)) < 0.02    # leads -> positive


def test_align_waveforms_progress_reaches_one():
    sr = 48000
    n = sr
    tracks = [_click_track(n, click_at=n // 2) for _ in range(3)]
    seen = []
    align_waveforms(tracks, [sr, sr, sr], progress=seen.append)
    assert seen and seen[-1] == 1.0
