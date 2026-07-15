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
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Load audio for already-probed tracks and align them (no re-probe).

    Internal helper so callers that already hold MediaInfo (e.g. sync_and_trim)
    do not probe the same files twice. ``status`` receives a human-readable
    label for the current stage.

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
        if status is not None:
            status(f"Loading audio ({i + 1}/{n})…")
        if progress is not None:
            progress(0.0)  # reset the bar; it sweeps 0..1 for this one file
        wave, rate = load_audio(info.path, progress=progress)
        waveforms.append(wave)
        rates.append(rate)

    if status is not None:
        status("Aligning waveforms…")
    if progress is not None:
        progress(0.0)  # new phase -> reset the bar; align reports its own 0..1
    return align_waveforms(
        waveforms, rates, refine=refine, reference_index=reference_index,
        progress=progress,
    )


def compute_sync_offsets(
    paths: list[Path],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    progress: Callable[[float], None] | None = None,
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Probe, load audio, and align paths by MFCC cross-correlation.

    Args:
        paths: Input media file paths.
        reference_index: Track whose timeline is the origin.
        refine: Peak refinement.
        progress: Optional 0..1 callback.
        status: Optional callback receiving a label for the current stage.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Per-track offset in seconds; offset[reference_index] == 0.0.

    Raises:
        ValueError: If any input has no audio stream.
    """
    if is_cancelled is not None and is_cancelled():
        return [0.0] * len(paths)
    media = []
    for i, p in enumerate(paths):
        if status is not None:
            status(f"Probing files ({i + 1}/{len(paths)})…")
        media.append(probe(p))
    return offsets_from_media(
        media, reference_index=reference_index, refine=refine,
        progress=progress, status=status, is_cancelled=is_cancelled,
    )
