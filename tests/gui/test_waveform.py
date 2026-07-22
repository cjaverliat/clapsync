import numpy as np

from clapsync.gui.waveform_widget import downsample_peaks


def test_downsample_peaks_shape_and_extremes():
    x = np.linspace(-1.0, 1.0, 1000).astype(np.float32)
    peaks = downsample_peaks(x, 10)
    assert peaks.shape == (10, 2)
    # first bucket min is near -1, last bucket max near +1
    assert peaks[0, 0] <= -0.9
    assert peaks[-1, 1] >= 0.9
    # min <= max in every bucket
    assert np.all(peaks[:, 0] <= peaks[:, 1])


def test_downsample_peaks_handles_short_input():
    x = np.array([0.5, -0.5], dtype=np.float32)
    peaks = downsample_peaks(x, 10)
    assert peaks.shape == (10, 2)
    assert np.isfinite(peaks).all()


def test_downsample_peaks_empty_is_zeros():
    peaks = downsample_peaks(np.zeros(0, dtype=np.float32), 5)
    assert peaks.shape == (5, 2)
    assert np.all(peaks == 0.0)
