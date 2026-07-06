import logging
from typing import Literal

import numpy as np
import torch
import torchaudio.functional as AF
from torchaudio.transforms import MFCC

logger = logging.getLogger(__name__)

Method = Literal["mfcc", "envelope"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_mono_f64(waveform: torch.Tensor) -> np.ndarray:
    """
    Force mono and return a float64 numpy array.

    The float64 cast happens before any arithmetic to avoid overflow on
    integer-typed inputs (e.g. int16 where abs(−32768) overflows back to −32768).
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
    return waveform.to(torch.float64).cpu().numpy()


# ---------------------------------------------------------------------------
# Envelope method
# ---------------------------------------------------------------------------

def _build_envelope(audio: np.ndarray, sample_rate: int, fps: float) -> np.ndarray:
    """
    Build a per-video-frame amplitude envelope from a mono float64 audio array.

    Each frame's value is the L1 norm (sum of absolute sample values) of all
    audio samples in [round(i * sr/fps), round((i+1) * sr/fps)). The rounding
    convention matches MLT's mlt_sample_calculator, avoiding cumulative drift
    for non-integer samples-per-frame rates (e.g. 29.97 fps at 48000 Hz).

    Args:
        audio:       1D float64 array of audio samples.
        sample_rate: Sample rate in Hz.
        fps:         Video frame rate in Hz.

    Returns:
        float64 array of shape (n_frames,), one scalar per video frame.
    """
    spf = sample_rate / fps
    n_frames = int(len(audio) / spf)
    boundaries = np.round(np.arange(n_frames + 1) * spf).astype(np.int64)

    cumsum = np.concatenate([[0.0], np.cumsum(np.abs(audio))])
    return cumsum[boundaries[1:]] - cumsum[boundaries[:-1]]


def _fft_correlate(main: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """
    FFT-based cross-correlation of two amplitude envelopes.

    Matches FFTCorrelation::correlate() + convolve() from fftCorrelation.cpp:

    1. Normalize each envelope by its maximum absolute value (floored at 1.0
       to guard against silent clips).
    2. Time-reverse the sub envelope. Convolving with a time-reversed kernel
       is mathematically equivalent to cross-correlation, without needing
       conjugate multiplication in the frequency domain.
    3. Zero-pad both to the next power of 2 >= 2 * max(len(main), len(sub))
       to prevent circular wrap-around artifacts.
    4. Forward real FFT of both, pointwise complex multiply, inverse real FFT.
    5. Prepend a zero to the output. This replicates the C++ convention:
           *out_convolved = 0;
           copy(convolved.begin(), convolved.begin() + out_size - 1, out + 1);
       which shifts the zero-lag position from index len(sub)-1 to len(sub).

    Args:
        main: Reference amplitude envelope (not modified).
        sub:  Query amplitude envelope to align against main.

    Returns:
        float64 cross-correlation array of length len(main) + len(sub).
        Zero-lag is at index len(sub); a peak at index k gives
        lag = k - len(sub) frames.
    """
    max_main = max(1.0, float(np.abs(main).max()))
    max_sub  = max(1.0, float(np.abs(sub).max()))

    main_norm = main      / max_main
    sub_rev   = sub[::-1] / max_sub   # time-reversal: convolution == correlation

    # Power of 2 >= 2 * max(L, R) so linear and circular convolution agree
    largest  = max(len(main_norm), len(sub_rev))
    fft_size = 64
    while fft_size // 2 < largest:
        fft_size <<= 1

    Main_f = np.fft.rfft(main_norm, n=fft_size)
    Sub_f  = np.fft.rfft(sub_rev,   n=fft_size)

    # Standard complex multiply — no conjugate needed because sub is already reversed
    convolved = np.fft.irfft(Main_f * Sub_f, n=fft_size)

    # Linear convolution produces exactly len(main) + len(sub) - 1 valid samples.
    # We prepend one zero (matching the C++ convention), so total length is
    # len(main) + len(sub).  Taking one extra sample beyond this would read into
    # the circular-wrap region of the FFT output.
    valid_len  = len(main) + len(sub) - 1
    out_size   = valid_len + 1          # +1 for the prepended zero
    result     = np.empty(out_size, dtype=np.float64)
    result[0]  = 0.0                    # matches: *out_convolved = 0
    result[1:] = convolved[:valid_len]  # matches: copy(convolved, out + 1)
    return result


def find_offset_envelope(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    fps: float,
) -> tuple[int, float]:
    """
    Find the temporal offset between two audio waveforms using per-frame L1
    amplitude envelope cross-correlation.

    Resolution is one video frame (1/fps seconds). Works best with clean
    recordings that contain sharp transients (claps, slates) and consistent
    gain across cameras.

    Both envelopes are built at the same *fps* so that the integer lag returned
    maps directly to reference-clip video frames. The caller is responsible for
    passing the reference clip's frame rate so that lag_frames has a well-defined
    meaning.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate:     Sample rate of ref_waveform in Hz.
        waveform:     Audio to align, shape (channels, samples) or (samples,).
        rate:         Sample rate of waveform in Hz. Resampled to ref_rate if needed.
        fps:          Reference clip's video frame rate in Hz (e.g. 25.0, 29.97, 30.0).

    Returns:
        (lag_frames, lag_seconds): signed lag. Positive means waveform starts
        after ref_waveform and should be delayed by that many frames to align.
    """
    if ref_rate != rate:
        waveform = AF.resample(waveform, orig_freq=rate, new_freq=ref_rate)

    env_ref = _build_envelope(_to_mono_f64(ref_waveform), ref_rate, fps)
    env_sub = _build_envelope(_to_mono_f64(waveform),     ref_rate, fps)

    corr     = _fft_correlate(main=env_ref, sub=env_sub)
    peak_idx = int(np.argmax(corr))

    # Zero-lag is at index len(env_sub) due to the prepended zero in _fft_correlate
    lag_frames  = peak_idx - len(env_sub)
    lag_seconds = lag_frames / fps

    logger.debug(
        "[envelope] sizes: ref=%d sub=%d  corr_size=%d  peak=%d  lag=%+d frames (%+.3f s)",
        len(env_ref), len(env_sub), len(corr), peak_idx, lag_frames, lag_seconds,
    )

    return lag_frames, lag_seconds


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
    """
    Compute MFCC features for a mono audio array.

    The input is a 1D float64 numpy array (from _to_mono_f64). It is cast to
    float32 and given a channel dimension before being passed to torchaudio's
    MFCC transform, which expects (..., time).

    Args:
        audio:       1D float64 numpy array of mono audio samples.
        sample_rate: Sample rate in Hz. Must match the rate the audio was loaded at.
        n_mfcc:      Number of MFCC coefficients to return.
        n_fft:       FFT size for the mel spectrogram.
        hop_length:  Hop length in samples.
        win_length:  Window length in samples.
        n_mels:      Number of mel filter banks.
        mel_scale:   Mel scale type ("htk" or "slaney").

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
    """
    FFT-based cross-correlation summed across MFCC coefficients.

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


def find_offset_mfcc(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    fps: float,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,   # 5 ms -> ~8x finer than a 25 fps frame
    win_duration: float = 0.04,    # 40 ms, standard for speech/music MFCC
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
) -> tuple[int, float]:
    """
    Find the temporal offset between two audio waveforms using MFCC cross-correlation.

    Both waveforms are converted to mono float64 via _to_mono_f64 before any
    processing. The query waveform is resampled to ref_rate when rates differ,
    guaranteeing both MFCC transforms use identical hop/win lengths so that
    each MFCC frame represents the same duration in both clips.

    MFCC features are more robust than raw amplitude envelopes when the two
    recordings differ in gain or frequency response (e.g. different camera
    microphones with different EQ curves). The hop-based resolution (5 ms by
    default) is ~8× finer than a 25 fps video frame.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate:     Sample rate of ref_waveform in Hz.
        waveform:     Audio to align, shape (channels, samples) or (samples,).
        rate:         Sample rate of waveform in Hz. Resampled to ref_rate if needed.
        fps:          Reference clip's video frame rate in Hz (e.g. 25.0, 29.97, 30.0).
                      Used only to convert the sub-frame-accurate lag in seconds to
                      the nearest integer frame count.
        n_mfcc:       Number of MFCC coefficients.
        n_fft:        FFT size for the mel spectrogram.
        hop_duration: MFCC hop size in seconds. Controls time resolution of the
                      correlation; at 5 ms this is ~8x finer than a 25 fps frame.
        win_duration: MFCC analysis window in seconds.
        n_mels:       Number of mel filter banks.
        mel_scale:    Mel scale type ("htk" or "slaney").

    Returns:
        (lag_frames, lag_seconds): signed lag at video-frame and sub-frame resolution.
        Positive means waveform starts after ref_waveform.
    """
    if ref_rate != rate:
        waveform = AF.resample(waveform, orig_freq=rate, new_freq=ref_rate)

    ref_mono = _to_mono_f64(ref_waveform)
    sub_mono = _to_mono_f64(waveform)

    # Derive integer hop/win lengths from ref_rate so both clips share the same grid
    hop_length = int(ref_rate * hop_duration)
    win_length = int(ref_rate * win_duration)

    mfcc_ref = _compute_mfcc(ref_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length, n_mels, mel_scale)
    mfcc_sub = _compute_mfcc(sub_mono, ref_rate, n_mfcc, n_fft, hop_length, win_length, n_mels, mel_scale)

    corr     = _mfcc_cross_correlate(mfcc_ref, mfcc_sub)
    peak_idx = int(np.argmax(corr))

    # Zero-lag is at index T_sub - 1 (see _mfcc_cross_correlate docstring)
    lag_hops    = peak_idx - (mfcc_sub.shape[1] - 1)
    lag_seconds = lag_hops * hop_length / ref_rate
    lag_frames  = round(lag_seconds * fps)

    logger.debug(
        "[mfcc] sizes: ref=%d sub=%d hops  corr_size=%d  peak=%d  "
        "lag=%+d hops (%+.3f s, %+d frames @ %.4f fps)",
        mfcc_ref.shape[1], mfcc_sub.shape[1], len(corr), peak_idx,
        lag_hops, lag_seconds, lag_frames, fps,
    )

    return lag_frames, lag_seconds


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def find_offset(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    fps: float,
    method: Method = "mfcc",
    # MFCC-only params (ignored when method="envelope")
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
) -> tuple[int, float]:
    """
    Find the temporal offset between two audio waveforms.

    Dispatches to find_offset_mfcc or find_offset_envelope depending on
    *method*. Both return the same (lag_frames, lag_seconds) contract so
    callers can switch methods without any other code changes.

    Args:
        ref_waveform: Reference audio, shape (channels, samples) or (samples,).
        ref_rate:     Sample rate of ref_waveform in Hz.
        waveform:     Audio to align, shape (channels, samples) or (samples,).
        rate:         Sample rate of waveform in Hz. Resampled to ref_rate if needed.
        fps:          Reference clip's video frame rate in Hz (e.g. 25.0, 29.97, 30.0).
        method:       "mfcc" (default) or "envelope". See find_offset_mfcc and
                      find_offset_envelope for a full comparison of tradeoffs.
        n_mfcc:       [mfcc] Number of MFCC coefficients.
        n_fft:        [mfcc] FFT size for the mel spectrogram.
        hop_duration: [mfcc] Hop size in seconds (time resolution of correlation).
        win_duration: [mfcc] Analysis window in seconds.
        n_mels:       [mfcc] Number of mel filter banks.
        mel_scale:    [mfcc] Mel scale type ("htk" or "slaney").

    Returns:
        (lag_frames, lag_seconds): signed lag. Positive means waveform starts
        after ref_waveform and should be delayed by that many frames to align.

    Raises:
        ValueError: If *method* is not "mfcc" or "envelope".
    """
    if method == "mfcc":
        return find_offset_mfcc(
            ref_waveform, ref_rate, waveform, rate, fps,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_duration=hop_duration,
            win_duration=win_duration,
            n_mels=n_mels,
            mel_scale=mel_scale,
        )
    if method == "envelope":
        return find_offset_envelope(ref_waveform, ref_rate, waveform, rate, fps)

    raise ValueError(f"Unknown method {method!r}. Expected 'mfcc' or 'envelope'.")