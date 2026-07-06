"""Compute per-track sync offsets on a shared timeline (audio-based)."""
from __future__ import annotations

import logging
from typing import Callable

from clapsync.core.media import MediaInfo
from clapsync.core.offsets import Method, Refine, find_offset
from clapsync.io.decode import load_audio

logger = logging.getLogger(__name__)


def compute_sync_offsets(
    media: list[MediaInfo],
    reference_index: int = 0,
    method: Method = "mfcc",
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Align every track to a reference by audio cross-correlation.

    Args:
        media: Probed tracks; each must have an audio stream to be aligned.
        reference_index: Track whose timeline is the origin (offset 0).
        method: Offset finder ("mfcc" or "envelope").
        refine: Peak refinement ("parabolic" or "none").
        progress: Optional 0..1 progress callback.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0. Positive
        means the track starts after the reference. Tracks that fail to load or
        correlate get 0.0.
    """
    n = len(media)
    ref_fps = media[reference_index].fps or 30.0

    ref_wave, ref_rate = load_audio(media[reference_index].path)
    if progress is not None:
        progress(1.0 / n)

    lags: list[float] = [0.0] * n
    for i, info in enumerate(media):
        if is_cancelled is not None and is_cancelled():
            break
        if i == reference_index:
            continue
        try:
            wave, rate = load_audio(info.path)
            _, lag_s = find_offset(
                ref_wave, ref_rate, wave, rate, ref_fps,
                method=method, refine=refine,
            )
        except Exception as exc:  # noqa: BLE001 — one bad track must not abort all
            logger.warning("offset failed for %s: %s — using 0.0", info.path, exc)
            lag_s = 0.0
        lags[i] = lag_s
        if progress is not None:
            progress((i + 1) / n)

    if progress is not None:
        progress(1.0)
    return lags
