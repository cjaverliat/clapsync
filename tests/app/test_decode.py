"""Integration tests for clapsync.app.decode."""
import pytest
import torch

from clapsync.app.decode import load_audio, decode_frames_at


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


def test_load_audio_shape_and_rate(tone_wav):
    path, rate = tone_wav(seconds=1.0, sample_rate=48000)
    wave, got_rate = load_audio(path)
    assert got_rate == rate
    assert wave.shape[0] == 1
    assert wave.dtype == torch.float32
    assert abs(wave.shape[1] - 48000) <= 64  # codec padding tolerance


def test_load_audio_is_peak_normalized(tone_wav):
    path, _ = tone_wav(seconds=1.0)
    wave, _ = load_audio(path)
    assert abs(float(wave.abs().max()) - 1.0) < 1e-5


def test_load_audio_progress_climbs_progressively(tone_wav):
    path, _ = tone_wav(seconds=8.0)
    seen = []
    load_audio(path, progress=seen.append)

    assert len(seen) >= 5, f"too coarse: {seen}"
    assert seen == sorted(seen), "progress must be monotonic"
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen[-1] == 1.0, "must finish at exactly 1.0"
    assert seen[0] < 0.5, "must not jump straight to the end"


def test_load_audio_target_rate_resamples(tone_wav):
    path, _ = tone_wav(seconds=1.0, sample_rate=48000)
    wave, rate = load_audio(path, target_rate=16000)
    assert rate == 16000
    assert abs(wave.shape[1] - 16000) <= 64


def test_load_audio_without_progress_still_works(tone_wav):
    path, _ = tone_wav(seconds=1.0)
    wave, _ = load_audio(path)
    assert wave.shape[1] > 0
