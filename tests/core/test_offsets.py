import numpy as np
import torch

from clapsync.core.offsets import find_offset, _parabolic_peak
from clapsync.core.offsets import PairAlignment, find_offset_peaks


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


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
    # Independently pin correctness: integer-hop lag lands on -0.20 s.
    assert abs(lag_none - (-0.20)) < 0.01
    # Both near -0.20 s; parabolic within 5 ms of none
    # (for exact-grid lags, integer argmax is already exact, so parabolic may
    # introduce a tiny Hann-window asymmetry; 5 ms tolerance covers that).
    assert abs(lag_par - (-0.20)) <= abs(lag_none - (-0.20)) + 5e-3


def test_find_offset_peaks_matched_content_scores_high():
    # Same noise content shifted by 0.25 s: the peak must tower over the curve.
    sr = 16000
    base = _noise(sr * 4, seed=1)
    ref = torch.from_numpy(base).unsqueeze(0)
    sub = torch.from_numpy(np.roll(base, int(0.25 * sr))).unsqueeze(0)
    result = find_offset_peaks(ref, sr, sub, sr)
    assert isinstance(result, PairAlignment)
    offset, score = result.peaks[0]
    assert abs(offset - (-0.25)) < 0.02
    assert score > 20.0


def test_find_offset_peaks_unrelated_content_scores_low():
    sr = 16000
    ref = torch.from_numpy(_noise(sr * 4, seed=2)).unsqueeze(0)
    sub = torch.from_numpy(_noise(sr * 4, seed=3)).unsqueeze(0)
    matched = find_offset_peaks(ref, sr, ref.clone(), sr).peaks[0][1]
    unrelated = find_offset_peaks(ref, sr, sub, sr).peaks[0][1]
    assert unrelated < matched / 4
    assert unrelated < 10.0


def test_find_offset_peaks_exclusion_window_separates_candidates():
    # A repeated identical burst yields two peaks at least 0.5 s apart.
    sr = 16000
    burst = _noise(sr // 2, seed=4)
    timeline = _noise(sr * 6, seed=5) * 0.05
    timeline[sr : sr + len(burst)] += burst
    timeline[4 * sr : 4 * sr + len(burst)] += burst
    ref = torch.from_numpy(timeline).unsqueeze(0)
    sub = torch.from_numpy(burst).unsqueeze(0)
    result = find_offset_peaks(ref, sr, sub, sr)
    assert len(result.peaks) >= 2
    offsets = [p[0] for p in result.peaks[:2]]
    assert abs(offsets[0] - offsets[1]) >= 0.5
    # Scores are sorted best-first.
    scores = [p[1] for p in result.peaks]
    assert scores == sorted(scores, reverse=True)


def test_find_offset_still_returns_float():
    sr = 16000
    base = _noise(sr * 2, seed=6)
    ref = torch.from_numpy(base).unsqueeze(0)
    lag = find_offset(ref, sr, ref.clone(), sr)
    assert isinstance(lag, float)
    assert abs(lag) < 0.02


def test_mfcc_correlate_recovers_full_length_shift():
    # Full-length equal-length content shifted by a known lag must be
    # recovered. This fails before mean-centering: the MFCC c0 (log-energy)
    # DC ridge pins the correlation peak at zero lag regardless of content.
    sr = 16000
    base = _noise(sr * 4, seed=11)
    ref = torch.from_numpy(base).unsqueeze(0)
    sub = torch.from_numpy(np.roll(base, int(0.30 * sr))).unsqueeze(0)
    result = find_offset_peaks(ref, sr, sub, sr)
    offset, score = result.peaks[0]
    assert abs(offset - (-0.30)) < 0.02   # rolled later -> query lags -> negative
    assert score > 20.0
