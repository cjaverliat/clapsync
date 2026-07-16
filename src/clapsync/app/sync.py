"""Compute per-track sync offsets on a shared timeline (audio-based)."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
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
    target_rate: int | None = 16000,
    progress: Callable[[float], None] | None = None,
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Load audio for already-probed tracks and align them (no re-probe).

    Internal helper so callers that already hold MediaInfo (e.g. sync_and_trim)
    do not probe the same files twice. ``status`` receives a human-readable
    label for the current stage.

    Args:
        target_rate: Decode/resample audio to this rate before alignment. MFCC
            sync needs far less than the native 48 kHz, so downsampling shrinks
            the align stage. Pass None to keep each track's native rate.

    Raises:
        ValueError: If any track has no audio stream.
    """
    missing = [str(m.path) for m in media if not m.has_audio]
    if missing:
        raise ValueError(
            "cannot sync tracks without an audio stream: " + ", ".join(missing)
        )

    n = len(media)
    if is_cancelled is not None and is_cancelled():
        return [0.0] * n

    waveforms: list[torch.Tensor | None] = [None] * n
    rates: list[int] = [0] * n

    # Files decode concurrently — PyAV releases the GIL during decode, so this
    # is real parallelism, not just interleaving. The single progress bar can't
    # sweep one file at a time anymore, so it reports the mean of every file's
    # decode fraction; `done` tracks completions for the status label.
    lock = threading.Lock()
    fracs = [0.0] * n
    done = 0

    def report(i: int, fraction: float) -> None:
        # Only wired up when `progress` is set (see load_one), so no guard here.
        with lock:
            fracs[i] = fraction
            progress(sum(fracs) / n)

    def load_one(i: int, info: MediaInfo) -> None:
        nonlocal done
        wave, rate = load_audio(
            info.path,
            target_rate=target_rate,
            progress=(lambda f, i=i: report(i, f)) if progress else None,
        )
        waveforms[i] = wave
        rates[i] = rate
        if status is not None:
            with lock:
                done += 1
                status(f"Loading audio ({done}/{n})…")

    if status is not None:
        status(f"Loading audio ({n} files)…")

    max_workers = min(n, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(load_one, i, info) for i, info in enumerate(media)
        ]
        for future in concurrent.futures.as_completed(futures):
            if is_cancelled is not None and is_cancelled():
                for pending in futures:
                    pending.cancel()
                return [0.0] * n
            future.result()  # surface any load_audio exception

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
    target_rate: int | None = 16000,
    progress: Callable[[float], None] | None = None,
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[float]:
    """Probe, load audio, and align paths by MFCC cross-correlation.

    Args:
        paths: Input media file paths.
        reference_index: Track whose timeline is the origin.
        refine: Peak refinement.
        target_rate: Decode/resample rate for alignment; None keeps native.
            See offsets_from_media.
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
        if progress is not None:
            progress(i / len(paths))
        media.append(probe(p))
    if progress is not None:
        progress(1.0)  # probing done; the load phase reports its own 0..1
    return offsets_from_media(
        media, reference_index=reference_index, refine=refine,
        target_rate=target_rate, progress=progress, status=status,
        is_cancelled=is_cancelled,
    )
