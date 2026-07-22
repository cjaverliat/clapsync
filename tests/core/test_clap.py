"""Pure clap-detection tests on synthetic acoustic and kinematic signals."""
from __future__ import annotations

import numpy as np

from clapsync.core.clap import (
    centroid,
    clapperboard_reliability,
    classify_clap_markers,
    detect_clap_motion,
    detect_clap_motions,
    detect_clap_sound,
)


# ---------------------------------------------------------------------------
# Marker classification
# ---------------------------------------------------------------------------

def test_classify_exact_names():
    labels = ["clapperboard_top", "clapperboard_down", "hand"]
    assert classify_clap_markers(labels) == ([0], [1])


def test_classify_loose_synonyms_case_insensitive():
    labels = ["Slate_Upper", "SLATE_bottom", "clap_TOP_L", "clap_low_R"]
    top, bottom = classify_clap_markers(labels)
    assert top == [0, 2]
    assert bottom == [1, 3]


def test_classify_returns_none_when_ambiguous():
    # Clap markers present but no direction -> cannot split.
    assert classify_clap_markers(["clapper_a", "clapper_b"]) is None
    # No clap markers at all.
    assert classify_clap_markers(["head", "hand", "foot_top"]) is None


# ---------------------------------------------------------------------------
# Acoustic detection
# ---------------------------------------------------------------------------

def _burst(rate: int, n: int, at: int, length: int, signal: np.ndarray) -> np.ndarray:
    x = np.random.default_rng(1).standard_normal(n).astype(np.float64) * 1e-3
    x[at : at + length] += signal
    return x


def test_detect_clap_sound_finds_broadband_burst():
    rate, n = 16000, 16000
    at = 8000
    length = int(0.008 * rate)
    noise = np.random.default_rng(2).standard_normal(length)
    x = _burst(rate, n, at, length, noise)

    cands = detect_clap_sound(x, rate)
    assert cands, "expected a clap candidate"
    assert abs(cands[0].time - at / rate) < 0.02
    assert cands[0].score > 0.0


def test_detect_clap_sound_rejects_tonal_burst():
    rate, n = 16000, 16000
    at = 6000
    length = int(0.2 * rate)  # sustained 1 kHz tone, smoothly ramped (no click)
    ramp = int(0.005 * rate)
    env = np.ones(length)
    env[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    env[-ramp:] = env[:ramp][::-1]
    t = np.arange(length) / rate
    tone = np.sin(2 * np.pi * 1000 * t) * env * 3.0  # low HF content -> rejected
    x = _burst(rate, n, at, length, tone)

    assert detect_clap_sound(x, rate) == []


def test_detect_clap_sound_rejects_slow_swell():
    rate, n = 16000, 16000
    at = 4000
    length = int(0.4 * rate)  # 400 ms ramp -> slow attack, long duration
    ramp = np.linspace(0, 1, length) ** 2
    swell = np.random.default_rng(3).standard_normal(length) * ramp
    x = _burst(rate, n, at, length, swell)

    assert detect_clap_sound(x, rate) == []


# ---------------------------------------------------------------------------
# Kinematic detection
# ---------------------------------------------------------------------------

def _snap_gap(n: int, snap_start: int, snap_end: int, size: float) -> tuple:
    """Top/bottom centroids held apart, then snapped together at snap_end."""
    top = np.zeros((n, 3))
    bottom = np.zeros((n, 3))
    for f in range(n):
        if f < snap_start:
            frac = 1.0
        elif f < snap_end:
            frac = 1.0 - (f - snap_start) / (snap_end - snap_start)
        else:
            frac = 0.0
        top[f, 2] = 0.5 * size * frac
        bottom[f, 2] = -0.5 * size * frac
    return top, bottom


def test_detect_clap_motion_finds_snap():
    rate = 100.0
    top, bottom = _snap_gap(200, snap_start=90, snap_end=100, size=20.0)
    cand = detect_clap_motion(top, bottom, rate)
    assert cand is not None
    assert abs(cand.time - 100 / rate) < 0.05


def test_detect_clap_motion_survives_occlusion_at_impact():
    rate = 100.0
    top, bottom = _snap_gap(200, snap_start=90, snap_end=100, size=20.0)
    top[98:103] = np.nan  # markers drop out right at contact
    bottom[98:103] = np.nan
    cand = detect_clap_motion(top, bottom, rate)
    assert cand is not None
    assert abs(cand.time - 100 / rate) < 0.1


def test_clapperboard_reliability_accepts_valid_rejects_garbage():
    # Valid: bottom coplanar (z=0 plane), top colinear (a line at z=10).
    bottom = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]], float)
    top = np.array([[[0, 0, 10], [1, 0, 10], [2, 0, 10], [3, 0, 10]]], float)
    valid = np.ones((1, 4), dtype=bool)
    assert clapperboard_reliability(top, valid, bottom, valid)[0]

    rng = np.random.default_rng(0)
    garbage_b = rng.standard_normal((1, 4, 3)) * 100
    garbage_t = rng.standard_normal((1, 4, 3)) * 100
    assert not clapperboard_reliability(garbage_t, valid, garbage_b, valid)[0]


def test_clapperboard_reliability_rejects_detached_top():
    # Bottom is a plane (z=0); top is colinear but tilted along the plane normal
    # (z) — a detached top that is not parallel to the plane.
    bottom = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]], float)
    top = np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]], float)
    valid = np.ones((1, 4), dtype=bool)
    assert not clapperboard_reliability(top, valid, bottom, valid)[0]


def test_clapperboard_reliability_marks_occluded_frames_unreliable():
    bottom = np.zeros((1, 4, 3))
    top = np.zeros((1, 4, 3))
    valid = np.ones((1, 4), dtype=bool)
    valid[0, 0] = False  # one occluded marker
    assert not clapperboard_reliability(top, valid, bottom, valid)[0]


def test_detect_motions_drops_snap_in_unreliable_region():
    rate = 100.0
    top, bottom = _snap_gap(200, snap_start=90, snap_end=100, size=20.0)
    assert detect_clap_motions(top, bottom, rate)  # found without a mask
    unreliable = np.ones(200, dtype=bool)
    unreliable[80:120] = False  # the snap sits in a bad-geometry stretch
    assert detect_clap_motions(top, bottom, rate, unreliable) == []


def test_detect_clap_motion_none_without_collapse():
    rate = 100.0
    top = np.zeros((100, 3))
    bottom = np.zeros((100, 3))
    top[:, 2] = 10.0
    bottom[:, 2] = -10.0  # constant 20-unit gap, never collapses
    assert detect_clap_motion(top, bottom, rate) is None


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

def test_centroid_ignores_invalid_markers():
    points = np.array([[[0, 0, 0], [2, 2, 2]]], dtype=float)  # 1 frame, 2 markers
    valid = np.array([[True, False]])
    out = centroid(points, valid)
    assert np.allclose(out[0], [0, 0, 0])


def test_centroid_nan_when_all_invalid():
    points = np.ones((1, 2, 3))
    valid = np.array([[False, False]])
    out = centroid(points, valid)
    assert np.isnan(out[0]).all()
