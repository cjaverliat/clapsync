from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torchaudio.functional as AF
from torchaudio.transforms import MFCC

from clapsync.core.solver import Alignment, Peaks, solve_offsets

Refine = Literal["none", "parabolic"]

# Peak-candidate extraction. K and the exclusion window bound how many
# distinct correlation lobes are reported per pair; the robust score scale
# matches the BBC audio-offset-finder "standard score" (1.4826*MAD == sigma
# for Gaussian data), so its published 5/10 thresholds apply.
_MAX_PEAKS = 5
_EXCLUSION_S = 0.5

# Upper edge of the MFCC analysis band. Pinned in Hz rather than left at the
# input's Nyquist so the features do not change with the sample rate (hop and
# window are already specified as durations for the same reason). Camera audio
# is lossy-coded and brick-walled well below Nyquist; mel filters spanning that
# empty region flatten the log-mel vector, and the 13 retained cepstral
# coefficients then describe the informative bands far more coarsely, which
# drops a correct peak's score under the solver's edge gate.
_F_MAX_HZ = 8000.0


@dataclass(frozen=True)
class PairAlignment:
    """Ranked correlation-peak candidates for one track pair.

    peaks: (offset_seconds, score) tuples, best first. peaks[0][0] is the
    primary offset estimate; peaks[0][1] its robust standard score.
    """

    peaks: list[tuple[float, float]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_mono_f64(waveform: torch.Tensor) -> np.ndarray:
    """Force mono and return a float64 numpy array.

    The float64 cast happens before any arithmetic to avoid overflow on
    integer-typed inputs (e.g. int16 where abs(−32768) overflows back to −32768).
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
    return waveform.to(torch.float64).cpu().numpy()


def _parabolic_peak(corr: np.ndarray, peak: int) -> float:
    """Refine an integer correlation peak to sub-sample precision.

    Fits a parabola through corr[peak-1:peak+2] and returns the fractional
    index of its vertex. Clamps to the integer peak at array edges or when the
    three points are colinear.

    Args:
        corr: 1D correlation array.
        peak: Index of the integer-grid maximum.

    Returns:
        Fractional peak index in [peak-0.5, peak+0.5].
    """
    if peak <= 0 or peak >= len(corr) - 1:
        return float(peak)
    y0, y1, y2 = float(corr[peak - 1]), float(corr[peak]), float(corr[peak + 1])
    denom = y0 - 2.0 * y1 + y2
    # Near-colinear/flat triplet: interpolation is ill-conditioned, keep the
    # integer peak rather than dividing by a tiny denominator.
    if abs(denom) < 1e-12:
        return float(peak)
    return peak + 0.5 * (y0 - y2) / denom


def _peak_candidates(
    corr: np.ndarray,
    sub_len: int,
    hop_length: int,
    rate: int,
    refine: Refine,
) -> list[tuple[float, float]]:
    """Extract up to _MAX_PEAKS distinct peaks with robust standard scores.

    Score = (peak - median) / (1.4826 * MAD): the BBC-style standard score
    made robust to sidelobes (music beats, reverb) by using median/MAD
    instead of mean/std. Candidates are greedily taken in height order with
    an exclusion window so shoulders of one lobe are not reported twice.

    Args:
        corr: 1D cross-correlation array (see _mfcc_cross_correlate).
        sub_len: Number of MFCC frames in the query clip (T_sub).
        hop_length: Hop length in samples used to compute the MFCCs.
        rate: Sample rate in Hz the hop_length is expressed against.
        refine: "parabolic" (sub-hop interpolation) or "none" (integer hop).

    Returns:
        Up to _MAX_PEAKS (offset_seconds, score) tuples, best first.
    """
    med = float(np.median(corr))
    mad = float(np.median(np.abs(corr - med)))
    scale = 1.4826 * mad + 1e-12
    exclusion = max(1, round(_EXCLUSION_S * rate / hop_length))

    order = np.argsort(corr)[::-1]
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= _MAX_PEAKS:
            break
        if all(abs(int(idx) - c) >= exclusion for c in chosen):
            chosen.append(int(idx))

    peaks: list[tuple[float, float]] = []
    for idx in chosen:
        pos = _parabolic_peak(corr, idx) if refine == "parabolic" else float(idx)
        lag_hops = pos - (sub_len - 1)
        offset = lag_hops * hop_length / rate
        score = (float(corr[idx]) - med) / scale
        peaks.append((offset, score))
    return peaks


# ---------------------------------------------------------------------------
# MFCC method
# ---------------------------------------------------------------------------

def _compute_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    mel_scale: Literal["htk", "slaney"],
    f_max: float,
) -> torch.Tensor:
    """Compute MFCC features for a mono audio array.

    The input is a 1D float64 numpy array (from _to_mono_f64). It is cast to
    float32 and given a channel dimension before being passed to torchaudio's
    MFCC transform, which expects (..., time).

    Args:
        audio: 1D float64 numpy array of mono audio samples.
        sample_rate: Sample rate in Hz. Must match the rate the audio was loaded at.
        n_mfcc: Number of MFCC coefficients to return.
        n_fft: FFT size for the mel spectrogram.
        hop_length: Hop length in samples.
        win_length: Window length in samples.
        n_mels: Number of mel filter banks.
        mel_scale: Mel scale type ("htk" or "slaney").
        f_max: Upper edge of the mel filterbank in Hz, clamped to Nyquist.

    Returns:
        Float32 tensor of shape (n_mfcc, T) on CPU.
    """
    # float64 -> float32 for torchaudio; unsqueeze adds the required channel dim
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, N)
    mfcc_fn = MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": n_mels,
            "mel_scale": mel_scale,
            "f_max": min(f_max, sample_rate / 2),
        },
    )
    return mfcc_fn(waveform).squeeze(0)  # (n_mfcc, T)


def _mfcc_cross_correlate(ref: torch.Tensor, sub: torch.Tensor) -> np.ndarray:
    """FFT-based cross-correlation summed across MFCC coefficients.

    Each coefficient is normalized to unit variance before correlation so that
    no single cepstral band dominates the alignment signal (low-order MFCC
    coefficients carry much more energy than high-order ones without this step).

    Uses the time-reversal trick: convolving ref with sub[::-1] is equivalent
    to cross-correlating ref with sub, without needing conjugate multiplication
    in the frequency domain.

    Args:
        ref: (n_mfcc, T_ref) MFCC tensor for the reference clip.
        sub: (n_mfcc, T_sub) MFCC tensor for the query clip.

    Returns:
        float64 array of length T_ref + T_sub - 1.
        Zero-lag is at index T_sub - 1; a peak at index k gives
        lag = k - (T_sub - 1) hops.
    """
    # Per-coefficient mean-centering then variance normalization. Subtracting
    # the mean makes this a true cross-covariance (Google ICASSP'14 eq. 4):
    # without it, MFCC c0 (log-energy) carries a large DC term whose triangular
    # autocorrelation ridge pins the peak at zero lag for low-transient content.
    ref_c = ref - ref.mean(dim=1, keepdim=True)
    sub_c = sub - sub.mean(dim=1, keepdim=True)
    ref_std = ref_c.std(dim=1, keepdim=True).clamp(min=1e-8)
    sub_std = sub_c.std(dim=1, keepdim=True).clamp(min=1e-8)
    ref_n   = ref_c / ref_std                  # (n_mfcc, T_ref)
    sub_rev = sub_c.flip(dims=[1]) / sub_std   # (n_mfcc, T_sub), time-reversed

    # FFT size: next power of 2 >= linear convolution output length
    valid_len = ref.shape[1] + sub.shape[1] - 1
    fft_size  = 1
    while fft_size < valid_len:
        fft_size <<= 1

    Ref_f = torch.fft.rfft(ref_n,   n=fft_size, dim=1)  # (n_mfcc, F)
    Sub_f = torch.fft.rfft(sub_rev, n=fft_size, dim=1)  # (n_mfcc, F)

    # Pointwise multiply then sum across MFCC coefficients -> single correlation signal
    corr_f = (Ref_f * Sub_f).sum(dim=0)                        # (F,)
    corr   = torch.fft.irfft(corr_f, n=fft_size)[:valid_len]  # (valid_len,)

    return corr.numpy().astype(np.float64)


def find_offset_peaks(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    *,
    refine: Refine = "parabolic",
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
    f_max: float = _F_MAX_HZ,
) -> PairAlignment:
    """Temporal offset between two waveforms via MFCC cross-correlation.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate: Reference sample rate in Hz.
        waveform: Query audio, resampled to ref_rate if rates differ.
        rate: Query sample rate in Hz.
        refine: "parabolic" (sub-hop interpolation) or "none" (integer hop).
        n_mfcc, n_fft, hop_duration, win_duration, n_mels, mel_scale, f_max:
            MFCC params. f_max is clamped to Nyquist.

    Returns:
        PairAlignment ranking up to _MAX_PEAKS candidate lags, best first.
        Positive means the query leads (starts before) the reference;
        negative means it starts later.
    """
    if ref_rate != rate:
        waveform = AF.resample(waveform, orig_freq=rate, new_freq=ref_rate)

    ref_mono = _to_mono_f64(ref_waveform)
    sub_mono = _to_mono_f64(waveform)

    hop_length = int(ref_rate * hop_duration)
    win_length = int(ref_rate * win_duration)

    mfcc_ref = _compute_mfcc(
        ref_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale, f_max,
    )
    mfcc_sub = _compute_mfcc(
        sub_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale, f_max,
    )

    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_sub)
    return PairAlignment(
        _peak_candidates(corr, mfcc_sub.shape[1], hop_length, ref_rate, refine)
    )


def find_offset(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    *,
    refine: Refine = "parabolic",
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
    f_max: float = _F_MAX_HZ,
) -> float:
    """Primary temporal offset between two waveforms (see find_offset_peaks)."""
    return find_offset_peaks(
        ref_waveform, ref_rate, waveform, rate,
        refine=refine, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_duration=hop_duration, win_duration=win_duration,
        n_mels=n_mels, mel_scale=mel_scale, f_max=f_max,
    ).peaks[0][0]


def align_waveforms(
    waveforms: list[torch.Tensor],
    rates: list[int],
    *,
    refine: Refine = "parabolic",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
) -> Alignment:
    """Align tracks via an all-pairs MFCC correlation graph and MST solve.

    Every pair is correlated (MFCCs are computed once per track); edges are
    score-gated, triangle-consistency weighted, repaired from alternate
    peaks, and solved with an MST rooted at the reference (see
    clapsync.core.solver).

    Args:
        waveforms: Per-track audio tensors.
        rates: Per-track sample rate in Hz (parallel to waveforms).
        refine: Peak refinement ("parabolic" or "none").
        reference_index: Track whose timeline is the origin (offset 0.0).
        progress: Optional 0..1 callback (spans the pairwise correlations).

    Returns:
        Alignment; offsets[reference_index] == 0.0, confidence per track
        (+inf for the reference, 0.0 for isolated tracks), warnings for
        tracks needing manual verification.
    """
    n = len(waveforms)
    if n <= 1:
        if progress is not None:
            progress(1.0)
        return Alignment(
            [0.0] * n, [float("inf")] * n, [],
        )

    rate = rates[reference_index]
    hop_length = int(rate * 0.005)
    win_length = int(rate * 0.04)

    monos = []
    for wave, r in zip(waveforms, rates):
        if r != rate:
            wave = AF.resample(wave, orig_freq=r, new_freq=rate)
        monos.append(_to_mono_f64(wave))
    mfccs = [
        _compute_mfcc(m, rate, 13, 2048, hop_length, win_length, 128, "htk",
                      _F_MAX_HZ)
        for m in monos
    ]

    pairs: dict[tuple[int, int], Peaks] = {}
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr = _mfcc_cross_correlate(mfccs[i], mfccs[j])
            pairs[(i, j)] = _peak_candidates(
                corr, mfccs[j].shape[1], hop_length, rate, refine
            )
            done += 1
            if progress is not None:
                progress(done / total)

    return solve_offsets(n, pairs, reference_index)
