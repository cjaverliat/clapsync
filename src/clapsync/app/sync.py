"""Compute per-track sync offsets on a shared timeline (audio-based)."""
from __future__ import annotations

import logging
from typing import Callable

import torch

from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo
from clapsync.core.offsets import Refine, align_waveforms

logger = logging.getLogger(__name__)


def compute_sync_offsets(
    media: list[MediaInfo],
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Align probed tracks by loading their audio and cross-correlating.

    Args:
        media: Probed tracks; each must have an audio stream.
        reference_index: Track whose timeline is the origin.
        refine: Peak refinement.
        progress: Optional 0..1 callback.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0.
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
