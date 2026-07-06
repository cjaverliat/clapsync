"""Integration tests for clapsync.io.decode."""
import pytest
import torch

from clapsync.io.decode import load_audio, decode_frames_at


@pytest.mark.slow
def test_load_audio_returns_mono_normalized(tone_wav):
    path, sr = tone_wav(seconds=0.5, sample_rate=48000)
    wav, rate = load_audio(path)
    assert rate == 48000
    assert wav.shape[0] == 1
    assert wav.abs().max() <= 1.0 + 1e-6
    assert wav.shape[1] > 0


@pytest.mark.slow
def test_load_audio_resamples(tone_wav):
    path, _ = tone_wav(seconds=0.5, sample_rate=48000)
    wav, rate = load_audio(path, target_rate=16000)
    assert rate == 16000
    assert abs(wav.shape[1] - 8000) < 200


@pytest.mark.slow
def test_decode_frames_at_returns_requested_count(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0, w=64, h=48)
    frames = decode_frames_at(path, [0.0, 0.5, 0.9])
    assert frames.shape == (3, 3, h, w)
    assert frames.dtype == torch.uint8
