from typing import Callable, Literal

import numpy as np
import torch
import torchaudio.functional as AF
from torchaudio.transforms import MFCC

Refine = Literal["none", "parabolic"]


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
    # Per-coefficient variance normalization
    ref_std = ref.std(dim=1, keepdim=True).clamp(min=1e-8)
    sub_std = sub.std(dim=1, keepdim=True).clamp(min=1e-8)
    ref_n   = ref / ref_std                  # (n_mfcc, T_ref)
    sub_rev = sub.flip(dims=[1]) / sub_std   # (n_mfcc, T_sub), time-reversed

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
) -> float:
    """Temporal offset between two waveforms via MFCC cross-correlation.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate: Reference sample rate in Hz.
        waveform: Query audio, resampled to ref_rate if rates differ.
        rate: Query sample rate in Hz.
        refine: "parabolic" (sub-hop interpolation) or "none" (integer hop).
        n_mfcc, n_fft, hop_duration, win_duration, n_mels, mel_scale: MFCC params.

    Returns:
        Lag in seconds. Positive means the query leads (starts before) the
        reference; negative means it starts later.
    """
    if ref_rate != rate:
        waveform = AF.resample(waveform, orig_freq=rate, new_freq=ref_rate)

    ref_mono = _to_mono_f64(ref_waveform)
    sub_mono = _to_mono_f64(waveform)

    hop_length = int(ref_rate * hop_duration)
    win_length = int(ref_rate * win_duration)

    mfcc_ref = _compute_mfcc(
        ref_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale,
    )
    mfcc_sub = _compute_mfcc(
        sub_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length,
        n_mels, mel_scale,
    )

    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_sub)
    peak_idx = int(np.argmax(corr))
    peak = _parabolic_peak(corr, peak_idx) if refine == "parabolic" else float(peak_idx)

    # Sign convention: positive lag = query leads the reference.
    lag_hops = peak - (mfcc_sub.shape[1] - 1)
    return lag_hops * hop_length / ref_rate


def align_waveforms(
    waveforms: list[torch.Tensor],
    rates: list[int],
    *,
    refine: Refine = "parabolic",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
) -> list[float]:
    """Align each waveform to a reference by MFCC cross-correlation.

    Args:
        waveforms: Per-track audio tensors.
        rates: Per-track sample rate in Hz (parallel to waveforms).
        refine: Peak refinement ("parabolic" or "none").
        reference_index: Track whose timeline is the origin (offset 0.0).
        progress: Optional 0..1 callback.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0. Positive
        means the track leads the reference (shared = local + offset).
    """
    n = len(waveforms)
    ref_wave = waveforms[reference_index]
    ref_rate = rates[reference_index]

    offsets = [0.0] * n
    for i in range(n):
        if i != reference_index:
            offsets[i] = find_offset(
                ref_wave, ref_rate, waveforms[i], rates[i], refine=refine,
            )
        if progress is not None:
            progress((i + 1) / n)
    return offsets
