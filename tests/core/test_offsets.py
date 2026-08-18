import numpy as np
import torch

from clapsync.core.offsets import align_waveforms, find_offset, _parabolic_peak
from clapsync.core.offsets import PairAlignment, find_offset_peaks
from clapsync.core.solver import EDGE_MIN_SCORE


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


def _codec_limited(x: np.ndarray, sr: int, cutoff: float = 20000.0) -> torch.Tensor:
    """Brick-wall `x` above `cutoff`, the way a lossy codec bands-limits audio.

    Every camera writes AAC/MP3, so a real 48 kHz track carries no content in
    the top few kHz of its own Nyquist range.
    """
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    spectrum[freqs > cutoff] = 0.0
    limited = np.fft.irfft(spectrum, n=x.size)
    return torch.from_numpy(limited.astype(np.float32)).unsqueeze(0)


def _overlapping_pair(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Two 6 s takes sharing a 3 s burst, each with its own room noise."""
    rng = np.random.default_rng(7)
    burst = rng.standard_normal(int(sr * 3.0)) * 0.3
    takes = []
    for lead in (1.0, 2.5):  # cam_b starts its burst 1.5 s later
        x = rng.standard_normal(int(sr * 6.0)) * 0.05
        start = int(lead * sr)
        x[start : start + burst.size] += burst
        takes.append(x)
    return takes[0], takes[1]


def test_find_offset_peaks_scores_high_on_band_limited_48khz():
    # Regression: the MFCC analysis band must not follow the input rate. With
    # f_max left at Nyquist, a 48 kHz track whose content stops at a codec
    # cutoff is analysed with mel filters over an empty region, and the score
    # for a correct peak collapsed from ~22 to ~3 — under the solver's edge
    # gate, so align_waveforms called a matching pair unrelated audio.
    sr = 48000
    a, b = _overlapping_pair(sr)
    ref = _codec_limited(a, sr)
    sub = _codec_limited(b, sr)
    offset, score = find_offset_peaks(ref, sr, sub, sr).peaks[0]
    assert abs(offset - (-1.5)) < 0.02
    assert score > EDGE_MIN_SCORE


def test_align_waveforms_syncs_band_limited_48khz_without_warning():
    sr = 48000
    a, b = _overlapping_pair(sr)
    result = align_waveforms([_codec_limited(a, sr), _codec_limited(b, sr)],
                             [sr, sr])
    assert abs(result.offsets[1] - (-1.5)) < 0.02
    assert not result.warnings
