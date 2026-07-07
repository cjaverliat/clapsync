"""Compute per-track sync offsets on a shared timeline (audio-based)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch

from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo, probe
from clapsync.core.offsets import Refine, align_waveforms

logger = logging.getLogger(__name__)


def offsets_from_media(
    media: list[MediaInfo],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Load audio for already-probed tracks and align them (no re-probe).

    Internal helper so callers that already hold MediaInfo (e.g. sync_and_trim)
    do not probe the same files twice.

    Raises:
        ValueError: If any track has no audio stream.
    """
    missing = [str(m.path) for m in media if not m.has_audio]
    if missing:
        raise ValueError(
            "cannot sync tracks without an audio stream: " + ", ".join(missing)
        )

    n = len(media)
    waveforms: list[torch.Tensor] = []
    rates: list[int] = []
    for i, info in enumerate(media):
        if is_cancelled is not None and is_cancelled():
            return [0.0] * n
        wave, rate = load_audio(info.path)
        waveforms.append(wave)
        rates.append(rate)
        if progress is not None:
            progress(0.9 * (i + 1) / n)

    offsets = align_waveforms(
        waveforms, rates, refine=refine, reference_index=reference_index,
    )
    if progress is not None:
        progress(1.0)
    return offsets


def compute_sync_offsets(
    paths: list[Path],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Probe, load audio, and align paths by MFCC cross-correlation.

    Args:
        paths: Input media file paths.
        reference_index: Track whose timeline is the origin.
        refine: Peak refinement.
        progress: Optional 0..1 callback.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0.

    Raises:
        ValueError: If any input has no audio stream.
    """
    if is_cancelled is not None and is_cancelled():
        return [0.0] * len(paths)
    media = [probe(p) for p in paths]
    return offsets_from_media(
        media, reference_index=reference_index, refine=refine,
        progress=progress, is_cancelled=is_cancelled,
    )
