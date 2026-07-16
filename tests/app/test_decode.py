"""Integration tests for clapsync.app.decode."""
import pytest
import torch
from framepipe.metadata import extract_video_metadata

from clapsync.app.decode import ensure_preview_proxy, load_audio, decode_frames_at
from tests.conftest import frame_index


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


def test_decode_frames_at_shape_and_order(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=2.0, fps=30.0, w=64, h=48)
    frames = decode_frames_at(path, [0.0, 0.5, 1.0], device="cpu")
    assert frames.shape == (3, 3, h, w)
    assert frames.dtype == torch.uint8
    assert int(frames[0, 0].float().mean()) > 200  # fixture is solid red


def test_decode_frames_at_clamps_out_of_range(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=1.0, fps=30.0)
    frames = decode_frames_at(path, [-5.0, 1e9], device="cpu")
    assert frames.shape[0] == 2  # clamped, not raised


def test_decode_frames_at_unordered_and_duplicate(rgb_video):
    path, fps, n, w, h = rgb_video(seconds=2.0, fps=30.0)
    frames = decode_frames_at(path, [1.0, 0.0, 1.0], device="cpu")
    assert frames.shape[0] == 3
    # same timestamp -> same frame, and order is caller order
    assert torch.equal(frames[0], frames[2])


# The tests below use `indexed_video`, whose frames are distinguishable by
# content (frame i's mean brightness is i * 20). `rgb_video` is solid red, so
# every frame compares equal and cannot catch a broken searchsorted / sort /
# inverse-permutation in decode_frames_at — these pin the actual mapping.


def test_decode_frames_at_ordered_distinct_returns_correct_frames(indexed_video):
    path, fps, n, w, h = indexed_video(n=8, fps=10.0)
    # pts for frame i is i/fps = i * 0.1
    frames = decode_frames_at(path, [0.0, 0.3, 0.6], device="cpu")
    assert [frame_index(f) for f in frames] == [0, 3, 6]


def test_decode_frames_at_unordered_returns_caller_order(indexed_video):
    path, fps, n, w, h = indexed_video(n=8, fps=10.0)
    # Requested out of order; result must match caller order (index 6,0,3),
    # not sorted order (0,3,6) — this is what a broken inverse permutation
    # would silently produce.
    frames = decode_frames_at(path, [0.6, 0.0, 0.3], device="cpu")
    assert [frame_index(f) for f in frames] == [6, 0, 3]


def test_decode_frames_at_duplicate_timestamps_identify_correct_frames(indexed_video):
    path, fps, n, w, h = indexed_video(n=8, fps=10.0)
    frames = decode_frames_at(path, [0.3, 0.0, 0.3], device="cpu")
    assert [frame_index(f) for f in frames] == [3, 0, 3]
    assert torch.equal(frames[0], frames[2])  # duplicate timestamp -> same frame
    assert not torch.equal(frames[0], frames[1])  # different timestamp -> different frame


def test_decode_frames_at_clamps_to_first_and_last_by_content(indexed_video):
    path, fps, n, w, h = indexed_video(n=8, fps=10.0)
    frames = decode_frames_at(path, [-5.0, 1e9], device="cpu")
    assert [frame_index(f) for f in frames] == [0, 7]


# VFR-with-offset regression (Finding A): framepipe's metadata.pts is
# 0-based for CFR but stream-absolute (includes stream.start_time) for VFR.
# decode_frames_at receives source-relative caller times, so a VFR file whose
# stream starts at a nonzero timestamp must have its pts normalized to
# 0-based before the caller's times are searchsorted into it -- otherwise
# every t below the start offset clamps to pts[0] and returns frame 0 for
# everything.


def test_extract_video_metadata_confirms_vfr_fixture_reproduces_the_bug(
    vfr_video_with_offset,
):
    """Sanity-check the fixture itself before trusting any test built on it."""
    from framepipe.metadata import extract_video_metadata

    path, n, w, h, expected_pts = vfr_video_with_offset()
    meta = extract_video_metadata(str(path))
    assert meta.is_vfr is True, "fixture must be VFR to reproduce Finding A"
    assert float(meta.pts[0]) != 0.0, "fixture must have a nonzero start_time"


def test_decode_frames_at_vfr_with_offset_does_not_collapse_to_frame_zero(
    vfr_video_with_offset,
):
    """The regression: every t < start_offset used to clamp to frame 0."""
    path, n, w, h, expected_pts = vfr_video_with_offset()
    # expected_pts are the fixture's actual (stream-absolute) pts; the
    # source-relative timeline decode_frames_at works in starts at 0.
    relative_pts = [t - expected_pts[0] for t in expected_pts]

    frames = decode_frames_at(path, relative_pts, device="cpu")
    got = [frame_index(f) for f in frames]

    # Must not collapse: at least one requested (non-first) time must not
    # map to frame 0.
    assert got != [0] * n, f"collapsed to frame 0 for every timestamp: {got}"
    # Non-decreasing requested times -> non-decreasing frame indices.
    assert got == sorted(got)
    # Asking for each frame's own pts must return that exact frame.
    assert got == list(range(n))


def test_decode_frames_at_vfr_with_offset_first_and_last_by_content(
    vfr_video_with_offset,
):
    path, n, w, h, expected_pts = vfr_video_with_offset()
    duration = expected_pts[-1] - expected_pts[0]
    frames = decode_frames_at(path, [0.0, duration], device="cpu")
    assert [frame_index(f) for f in frames] == [0, n - 1]


def test_decode_frames_at_vfr_with_offset_clamps_out_of_range_by_content(
    vfr_video_with_offset,
):
    path, n, w, h, expected_pts = vfr_video_with_offset()
    frames = decode_frames_at(path, [-5.0, 1e9], device="cpu")
    assert [frame_index(f) for f in frames] == [0, n - 1]


def test_decode_frames_at_cfr_behavior_unchanged(indexed_video):
    """CFR normalization (pts - pts[0]) is a no-op: results unchanged."""
    path, fps, n, w, h = indexed_video(n=8, fps=10.0)
    frames = decode_frames_at(path, [0.0, 0.3, 0.6], device="cpu")
    assert [frame_index(f) for f in frames] == [0, 3, 6]


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
