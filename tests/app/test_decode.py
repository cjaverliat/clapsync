"""Integration tests for clapsync.app.decode."""
from pathlib import Path

import pytest
import torch
from framepipe.metadata import extract_video_metadata

from clapsync.app.decode import (
    ensure_preview_proxy,
    load_audio,
    source_needs_preview_proxy,
)
from clapsync.app.media import MediaInfo


def _video_info(height: int) -> MediaInfo:
    return MediaInfo(
        path=Path("clip.mp4"),
        duration=1.0,
        has_audio=True,
        kind="video",
        width=height * 16 // 9,
        height=height,
    )


def test_source_needs_preview_proxy_above_1080p():
    assert source_needs_preview_proxy(_video_info(1440))


def test_source_needs_preview_proxy_at_1080p_passes_through():
    assert not source_needs_preview_proxy(_video_info(1080))


def test_source_needs_preview_proxy_ignores_audio_only():
    info = MediaInfo(
        path=Path("a.wav"),
        duration=1.0,
        has_audio=True,
        kind="audio",
    )
    assert not source_needs_preview_proxy(info)


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


# ensure_preview_proxy caches to LOCALAPPDATA/clapsync/video-cache; point that
# at a tmp dir so the tests never touch (or hit) the real user cache.
@pytest.fixture
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))


def test_ensure_preview_proxy_passes_through_1080p_source(
    rgb_video, _isolated_cache
):
    """A source at/below 1080p decodes fine, so it is returned unchanged."""
    path, *_ = rgb_video(seconds=0.5, fps=30.0, w=1920, h=1080, name="hd.mp4")
    assert ensure_preview_proxy(path) == path


@pytest.mark.slow
def test_ensure_preview_proxy_downscales_and_caches_large_source(
    rgb_video, _isolated_cache
):
    """A >1080p source is transcoded to a 480-tall proxy and reused on re-open."""
    path, *_ = rgb_video(seconds=0.5, fps=30.0, w=2560, h=1440, name="big.mp4")

    proxy = ensure_preview_proxy(path)
    assert proxy != path
    assert proxy.exists()
    assert extract_video_metadata(str(proxy)).height == 480

    # Second call is a cache hit: same proxy path, no re-transcode.
    assert ensure_preview_proxy(path) == proxy


@pytest.mark.slow
def test_ensure_preview_proxy_progress_reaches_one(rgb_video, _isolated_cache):
    path, *_ = rgb_video(seconds=1.0, fps=30.0, w=2560, h=1440, name="prog.mp4")
    seen = []
    ensure_preview_proxy(path, progress=seen.append)
    assert seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen)
