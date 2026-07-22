import numpy as np
from clapsync.gui.audio_engine import mix_block


def _ramp(n, val):
    return np.full(n, val, dtype=np.float32)


def test_mix_mean_of_unmuted():
    rate = 100
    a = _ramp(200, 1.0)
    b = _ramp(200, 0.0)
    out = mix_block([a, b], [0.0, 0.0], [False, False], 0.0, 50, rate)
    assert out.shape == (50,)
    assert np.allclose(out, 0.5)


def test_mix_skips_muted():
    rate = 100
    a = _ramp(200, 1.0)
    b = _ramp(200, 0.0)
    out = mix_block([a, b], [0.0, 0.0], [False, True], 0.0, 50, rate)
    assert np.allclose(out, 1.0)  # only a contributes


def test_mix_respects_offset_and_pads_silence():
    rate = 100
    a = _ramp(100, 1.0)  # 1s of ones
    # a starts at offset +1.0s; querying global [0,0.5s) => before a starts => silence
    out = mix_block([a], [1.0], [False], 0.0, 50, rate)
    assert np.allclose(out, 0.0)
    # querying global [1.0,1.5s) => inside a => ones
    out2 = mix_block([a], [1.0], [False], 1.0, 50, rate)
    assert np.allclose(out2, 1.0)


def test_mix_all_muted_is_silence():
    out = mix_block([_ramp(100, 1.0)], [0.0], [True], 0.0, 20, 100)
    assert np.allclose(out, 0.0)
