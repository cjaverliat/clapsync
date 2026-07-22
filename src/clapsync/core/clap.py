"""Pure clapperboard-clap detection: acoustic and kinematic.

No I/O, no global state — every function maps arrays to results and is testable
with synthetic signals. Feeds the sync bridge, which turns detected clap times
into a c3d offset on the shared timeline.

Acoustic detection scores each candidate transient on the clap's signature —
sharp attack, short duration, flat (broadband) spectrum, high crest factor —
and gates out non-claps (speech, music, rumble, handling noise). Kinematic
detection finds the clapperboard arms meeting: the gap between the top- and
bottom-arm marker centroids collapses at peak closing velocity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- acoustic detection tuning ---------------------------------------------
_STFT_WIN = 1024
_STFT_HOP = 256
_ENVELOPE_MS = 5.0        # moving-RMS window for the sample-domain envelope
_BURST_MS = 40.0          # analysis window after onset for spectral features
_MAX_ATTACK_S = 0.010     # 10%->peak rise slower than this is not a clap
_MAX_DURATION_S = 0.120   # decay-to-10% longer than this is not a clap
_MIN_FLATNESS = 0.20      # spectral flatness below this is tonal, not a clap
_MIN_CREST = 3.0          # peak/RMS below this is not impulsive

# --- kinematic detection tuning --------------------------------------------
_MIN_COLLAPSE_RATIO = 0.5  # gap must shrink to <= this fraction of its span
_MIN_CLOSING_SHARPNESS = 3.0  # peak closing speed vs. median motion

# --- marker naming ---------------------------------------------------------
_CLAP_KEYWORDS = ("clap", "slate")
_TOP_KEYWORDS = ("top", "upper", "up")
_BOTTOM_KEYWORDS = ("bottom", "lower", "low", "down")


@dataclass(frozen=True)
class ClapCandidate:
    """A detected clap event.

    Attributes:
        time: Event time in seconds, local to the stream, sub-sample refined.
        score: Quality in (0, 1]; higher is more clap-like.
    """

    time: float
    score: float


# ---------------------------------------------------------------------------
# Marker classification
# ---------------------------------------------------------------------------

def classify_clap_markers(
    labels: list[str],
) -> tuple[list[int], list[int]] | None:
    """Split clapperboard marker labels into (top, bottom) index groups by name.

    A marker qualifies when its label contains a clapperboard keyword
    (``clap``/``slate``) and a direction keyword. Matching is case-insensitive.

    Args:
        labels: Marker labels in file order.

    Returns:
        (top_indices, bottom_indices), or None if either group is empty (the
        caller then asks the user to pick markers manually).
    """
    top: list[int] = []
    bottom: list[int] = []
    for i, label in enumerate(labels):
        name = label.lower()
        if not any(kw in name for kw in _CLAP_KEYWORDS):
            continue
        if any(kw in name for kw in _TOP_KEYWORDS):
            top.append(i)
        elif any(kw in name for kw in _BOTTOM_KEYWORDS):
            bottom.append(i)
    if not top or not bottom:
        return None
    return top, bottom


# ---------------------------------------------------------------------------
# Acoustic detection
# ---------------------------------------------------------------------------

def detect_clap_sound(wave: np.ndarray, rate: int) -> list[ClapCandidate]:
    """Rank clap-sound candidates in a waveform, gating out non-claps.

    Args:
        wave: Mono (or multi-channel, averaged) samples.
        rate: Sample rate (Hz).

    Returns:
        Candidates sorted by descending score. Empty if none pass the gate.
    """
    x = _to_mono(wave)
    if x.size < _STFT_WIN * 2:
        return []

    flux = _spectral_flux(x)
    onset_frames = _pick_peaks(flux)
    if onset_frames.size == 0:
        return []

    env = _moving_rms(x, max(1, int(_ENVELOPE_MS * 1e-3 * rate)))
    burst = max(1, int(_BURST_MS * 1e-3 * rate))
    # An onset frame's window spans _STFT_WIN samples, so the true envelope peak
    # can sit up to a full window past the frame start — search that far.
    search = _STFT_WIN

    candidates: list[ClapCandidate] = []
    for frame in onset_frames:
        onset = int(frame) * _STFT_HOP
        peak = _local_peak(env, onset, search)
        cand = _score_candidate(x, env, peak, burst, rate)
        if cand is not None:
            candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _score_candidate(
    x: np.ndarray, env: np.ndarray, peak: int, burst: int, rate: int
) -> ClapCandidate | None:
    """Feature-gate one onset and return a scored candidate, or None."""
    peak_val = env[peak]
    if peak_val <= 0.0:
        return None

    thresh = 0.1 * peak_val  # 10% of peak == -20 dB
    start = peak
    while start > 0 and env[start] > thresh:
        start -= 1
    end = peak
    while end < len(env) - 1 and env[end] > thresh:
        end += 1
    attack = (peak - start) / rate
    duration = (end - start) / rate

    window = x[peak : peak + burst]
    if window.size < 8:
        return None
    flatness = _spectral_flatness(window)
    rms = float(np.sqrt(np.mean(window * window))) + 1e-12
    crest = float(np.max(np.abs(window))) / rms

    if (attack > _MAX_ATTACK_S or duration > _MAX_DURATION_S
            or flatness < _MIN_FLATNESS or crest < _MIN_CREST):
        return None

    # Each sub-score is in (0, 1]; the product demands quality on every axis.
    sharpness = 1.0 - attack / _MAX_ATTACK_S
    shortness = 1.0 - duration / _MAX_DURATION_S
    score = flatness * sharpness * shortness * min(1.0, crest / (2 * _MIN_CREST))
    if score <= 0.0:
        return None

    time = _parabolic_vertex(env, peak) / rate
    return ClapCandidate(time=time, score=float(score))


def _spectral_flux(x: np.ndarray) -> np.ndarray:
    """Onset strength: summed positive frame-to-frame magnitude change."""
    window = np.hanning(_STFT_WIN)
    n = 1 + (len(x) - _STFT_WIN) // _STFT_HOP
    idx = np.arange(_STFT_WIN)[None, :] + _STFT_HOP * np.arange(n)[:, None]
    frames = x[idx] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    diff = np.diff(mag, axis=0)
    return np.maximum(diff, 0.0).sum(axis=1)


def _pick_peaks(flux: np.ndarray) -> np.ndarray:
    """Local maxima of the flux curve above a robust threshold."""
    if flux.size < 3:
        return np.empty(0, dtype=int)
    median = np.median(flux)
    mad = np.median(np.abs(flux - median)) + 1e-12
    threshold = median + 5.0 * 1.4826 * mad
    hi = flux > threshold
    local = np.zeros_like(hi)
    local[1:-1] = (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    return np.nonzero(hi & local)[0]


def _spectral_flatness(window: np.ndarray) -> float:
    """Wiener entropy of a burst: geometric mean / arithmetic mean of power."""
    spec = np.abs(np.fft.rfft(window * np.hanning(len(window)))) ** 2
    spec = spec[1:]  # drop DC
    if spec.size == 0 or not np.any(spec):
        return 0.0
    geo = np.exp(np.mean(np.log(spec + 1e-12)))
    arith = np.mean(spec) + 1e-12
    return float(geo / arith)


# ---------------------------------------------------------------------------
# Kinematic detection
# ---------------------------------------------------------------------------

def detect_clap_motion(
    top_pts: np.ndarray, bottom_pts: np.ndarray, rate: float
) -> ClapCandidate | None:
    """Find the clapperboard arms meeting: gap collapse at peak closing speed.

    Args:
        top_pts: (n_frames, 3) top-arm marker centroid; NaN where occluded.
        bottom_pts: (n_frames, 3) bottom-arm marker centroid; NaN where occluded.
        rate: Point sampling rate (Hz).

    Returns:
        The clap candidate (time from frame 0), or None if no clear collapse.
    """
    gap = np.linalg.norm(top_pts - bottom_pts, axis=1)
    finite = np.isfinite(gap)
    if finite.sum() < 3:
        return None

    valid_gap = gap[finite]
    span = float(valid_gap.max() - valid_gap.min())
    if span <= 0.0 or valid_gap.min() > _MIN_COLLAPSE_RATIO * valid_gap.max():
        return None  # gap never collapses -> not a clap

    # Interpolate across dropouts so velocity and the sub-frame refine stay
    # finite even when the markers occlude right at contact.
    filled = _fill_nan(gap)

    # Closing velocity (positive as the gap shrinks). Occlusion right at impact
    # can NaN the minimum, so key the event on peak closing speed, which occurs
    # just before contact and survives a dropout at the very bottom.
    velocity = -np.gradient(filled)
    peak = int(np.argmax(velocity))
    median_speed = np.median(np.abs(velocity)) + 1e-12
    sharpness = velocity[peak] / median_speed
    if sharpness < _MIN_CLOSING_SHARPNESS:
        return None

    # The impact is where closing decelerates to zero after the velocity peak;
    # fall back to the peak itself if no crossing is found within the tail.
    impact = peak
    for f in range(peak, len(velocity) - 1):
        if velocity[f] <= 0.0:
            impact = f
            break
    time = _parabolic_vertex(-filled, impact) / rate
    score = float(min(1.0, sharpness / (2 * _MIN_CLOSING_SHARPNESS)))
    return ClapCandidate(time=time, score=score)


def centroid(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-frame mean of the valid markers in a group.

    Args:
        points: (n_frames, n_group_markers, 3).
        valid: (n_frames, n_group_markers) bool.

    Returns:
        (n_frames, 3); rows with no valid marker are NaN.
    """
    mask = valid[:, :, None]
    count = valid.sum(axis=1)
    summed = np.where(mask, points, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = summed / count[:, None]
    out[count == 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_mono(wave: np.ndarray) -> np.ndarray:
    x = np.asarray(wave, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[1] else x.mean(axis=1)
    return x


def _moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-RMS envelope, same length as x."""
    power = x * x
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(power, kernel, mode="same"))


def _local_peak(env: np.ndarray, center: int, radius: int) -> int:
    lo = max(0, center - radius)
    hi = min(len(env), center + radius + 1)
    return lo + int(np.argmax(env[lo:hi]))


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs so gradients are defined across dropouts."""
    x = x.copy()
    nan = np.isnan(x)
    if nan.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    return x


def _parabolic_vertex(y: np.ndarray, peak: int) -> float:
    """Sub-sample index of the maximum near ``peak`` by parabolic fit."""
    if peak <= 0 or peak >= len(y) - 1:
        return float(peak)
    y0, y1, y2 = float(y[peak - 1]), float(y[peak]), float(y[peak + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(peak)
    return peak + 0.5 * (y0 - y2) / denom
