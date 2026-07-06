import numpy as np
import torch

from clapsync.core.offsets import find_offset, _parabolic_peak


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def test_parabolic_peak_interpolates_between_samples():
    corr = np.array([0.0, 1.0, 1.2, 0.3])
    frac = _parabolic_peak(corr, 2)
    assert 1.5 < frac < 2.5
    assert frac != 2.0


def test_parabolic_peak_clamps_at_edges():
    corr = np.array([5.0, 1.0, 0.0])
    assert _parabolic_peak(corr, 0) == 0.0


def test_find_offset_returns_float_and_recovers_subframe_lag():
    # Sign convention: positive = query leads. Here sub is 12.3 ms LATER than
    # ref, so the lag must be negative.
    sr = 48000
    n = sr * 2
    true_lag = 0.0123
    ref = _click_track(n, click_at=sr)
    sub = _click_track(n, click_at=sr + int(true_lag * sr))
    lag = find_offset(ref, sr, sub, sr)
    assert isinstance(lag, float)
    assert abs(lag - (-true_lag)) < 0.005


def test_find_offset_none_refine_is_integer_hop():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)
    sub = _click_track(n, click_at=sr + int(0.20 * sr))
    lag_none = find_offset(ref, sr, sub, sr, refine="none")
    lag_par = find_offset(ref, sr, sub, sr, refine="parabolic")
    # Both near -0.20 s; parabolic within 5 ms of none
    # (for exact-grid lags, integer argmax is already exact, so parabolic may
    # introduce a tiny Hann-window asymmetry; 5 ms tolerance covers that).
    assert abs(lag_par - (-0.20)) <= abs(lag_none - (-0.20)) + 5e-3
