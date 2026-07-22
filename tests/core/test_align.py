import numpy as np
import torch

from clapsync.core.offsets import align_waveforms
from clapsync.core.solver import LOW_CONFIDENCE, Alignment


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def _noise_track(n: int, seed: int) -> torch.Tensor:
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def test_align_waveforms_reference_zero_and_signed_offsets():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)                       # 1.000 s
    later = _click_track(n, click_at=sr + int(0.20 * sr))    # +200 ms (delayed)
    earlier = _click_track(n, click_at=sr - int(0.10 * sr))  # -100 ms (leads)

    result = align_waveforms([ref, later, earlier], [sr, sr, sr])
    assert isinstance(result, Alignment)
    offsets = result.offsets
    assert offsets[0] == 0.0
    assert abs(offsets[1] - (-0.20)) < 0.02   # delayed -> negative
    assert abs(offsets[2] - (0.10)) < 0.02    # leads -> positive


def test_align_waveforms_progress_reaches_one():
    sr = 48000
    n = sr
    tracks = [_click_track(n, click_at=n // 2) for _ in range(3)]
    seen = []
    align_waveforms(tracks, [sr, sr, sr], progress=seen.append)
    assert seen and seen[-1] == 1.0


def test_related_tracks_have_high_confidence():
    sr = 16000
    base = np.random.default_rng(7).standard_normal(sr * 4).astype(np.float32)
    a = torch.from_numpy(base).unsqueeze(0)
    b = torch.from_numpy(np.roll(base, int(0.3 * sr))).unsqueeze(0)
    c = torch.from_numpy(np.roll(base, int(-0.2 * sr))).unsqueeze(0)
    result = align_waveforms([a, b, c], [sr, sr, sr])
    assert all(conf > LOW_CONFIDENCE for conf in result.confidence[1:])
    assert result.warnings == []


def test_unrelated_track_isolated_with_warning():
    sr = 16000
    base = np.random.default_rng(8).standard_normal(sr * 6).astype(np.float32)
    a = torch.from_numpy(base).unsqueeze(0)
    b = torch.from_numpy(np.roll(base, int(0.3 * sr))).unsqueeze(0)
    stranger = _noise_track(sr * 6, seed=9)
    result = align_waveforms([a, b, stranger], [sr, sr, sr])
    assert result.confidence[2] < LOW_CONFIDENCE
    assert any("track 2" in w for w in result.warnings)
    # The related pair is unaffected by the stranger.
    assert abs(result.offsets[1] - (-0.3)) < 0.02


def test_repeated_content_trap_repaired():
    """A clip matching the louder wrong repeat is fixed via triangle repair.

    ref carries a burst at 1 s and a LOUDER copy at 4 s, so the clip's best
    correlation peak is the wrong occurrence; witness carries the burst once
    (attenuated repeat), giving unambiguous edges that expose the lie. The
    clip's correct offset survives as its second peak and must be adopted.
    """
    sr = 16000
    rng = np.random.default_rng(10)
    burst_len = int(1.5 * sr)                         # 1.5 s burst (see below)
    burst = rng.standard_normal(burst_len).astype(np.float32)
    bed_ref = rng.standard_normal(sr * 6).astype(np.float32) * 0.05
    bed_wit = rng.standard_normal(sr * 6).astype(np.float32) * 0.05

    ref_arr = bed_ref.copy()
    ref_arr[sr : sr + burst_len] += burst
    ref_arr[4 * sr : 4 * sr + burst_len] += 1.5 * burst  # louder wrong repeat

    wit_arr = bed_wit.copy()
    wit_arr[sr : sr + burst_len] += burst             # correct occurrence only

    # burst_len is 1.5 s (not 1 s) so the ref<->witness edge's primary peak
    # is the correct (near-zero) offset rather than tying/losing to the
    # wrong-repeat echo; the ref<->clip edge remains fooled (its primary
    # peak still matches the louder 4 s repeat), which is what the repair
    # pass is meant to fix using the clip's second peak.
    clip = torch.from_numpy(burst).unsqueeze(0)       # clip of the burst
    ref = torch.from_numpy(ref_arr).unsqueeze(0)
    wit = torch.from_numpy(wit_arr).unsqueeze(0)

    result = align_waveforms([ref, wit, clip], [sr, sr, sr])
    # Correct clip offset: its content is at t=1s in ref -> o_clip = +1.0
    # (shared = local + offset; clip local 0 aligns with shared 1.0).
    assert abs(result.offsets[2] - 1.0) < 0.05
