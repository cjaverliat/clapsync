import numpy as np
import torch

from clapsync.core.offsets import find_offset, _parabolic_peak


def _click_track(n: int, click_at: int, sr: int) -> torch.Tensor:
    """Silence with a single sharp click sample, shape (1, n)."""
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_parabolic_peak_interpolates_between_samples():
    # Symmetric-ish parabola peaking between index 1 and 2.
    corr = np.array([0.0, 1.0, 1.2, 0.3])
    frac = _parabolic_peak(corr, 2)
    assert 1.5 < frac < 2.5
    assert frac != 2.0  # actually refined off the integer grid


def test_parabolic_peak_clamps_at_edges():
    corr = np.array([5.0, 1.0, 0.0])
    assert _parabolic_peak(corr, 0) == 0.0


def test_find_offset_recovers_known_lag_envelope():
    # Sign convention: positive lag = query LEADS the reference, negative =
    # query is delayed/later. Here sub's click is 200 ms LATER than ref, so the
    # expected lag is NEGATIVE (-0.20). This matches the export clip_window
    # convention shared = local + offset; do not flip the sign.
    sr, fps = 48000, 25.0
    n = sr * 2
    ref = _click_track(n, click_at=sr, sr=sr)                   # click at 1.000 s
    sub = _click_track(n, click_at=sr + int(0.20 * sr), sr=sr)  # +200 ms (later)
    lag_frames, lag_s = find_offset(ref, sr, sub, sr, fps, method="envelope")
    assert abs(lag_s - (-0.20)) < 1.0 / fps


def test_mfcc_subframe_offset_parabolic_beats_none():
    # sub delayed by +true_lag => expected lag is -true_lag (see convention in
    # test_find_offset_recovers_known_lag_envelope).
    sr, fps = 48000, 25.0
    n = sr * 2
    true_lag = 0.0123  # 12.3 ms, well under one 25 fps frame (40 ms)
    expected = -true_lag
    ref = _click_track(n, click_at=sr, sr=sr)
    sub = _click_track(n, click_at=sr + int(true_lag * sr), sr=sr)
    _, lag_par = find_offset(ref, sr, sub, sr, fps, method="mfcc", refine="parabolic")
    _, lag_none = find_offset(ref, sr, sub, sr, fps, method="mfcc", refine="none")
    # Parabolic lands closer to the true subframe lag than raw hop-grid argmax.
    assert abs(lag_par - expected) <= abs(lag_none - expected) + 1e-9
    assert abs(lag_par - expected) < 0.005
