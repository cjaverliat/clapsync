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
from clapsync.app.mocap_sync import MarkerChoices, bridge_mocap_offsets
from clapsync.core.offsets import Refine, align_waveforms
from clapsync.core.solver import Alignment

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
) -> Alignment:
    """Load audio for already-probed tracks and align them (no re-probe).

    Internal helper so callers that already hold MediaInfo (e.g. sync_and_trim)
    do not probe the same files twice. ``status`` receives a human-readable
    label for the current stage.

    Args:
        target_rate: Decode/resample audio to this rate before alignment. MFCC
            sync needs far less than the native 48 kHz, so downsampling shrinks
            the align stage. Pass None to keep each track's native rate.

    Returns:
        Alignment (offsets, per-track confidence, warnings); offsets[reference_index] == 0.0.

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
        return Alignment([0.0] * n, [0.0] * n, [])

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
                return Alignment([0.0] * n, [0.0] * n, [])
            future.result()  # surface any load_audio exception

    if status is not None:
        status("Aligning waveforms…")
    if progress is not None:
        progress(0.0)  # new phase -> reset the bar; align reports its own 0..1
    return align_waveforms(
        waveforms, rates, refine=refine, reference_index=reference_index,
        progress=progress,
    )


def align_media(
    media: list[MediaInfo],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    target_rate: int | None = 16000,
    marker_choices: MarkerChoices | None = None,
    progress: Callable[[float], None] | None = None,
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Alignment:
    """Align mixed audio/video and motion-capture tracks on one timeline.

    Audio/video tracks are synced by MFCC cross-correlation; motion-capture
    (c3d) tracks are bridged onto the same timeline via the clapperboard clap.
    Results are assembled in the original input order.

    Args:
        marker_choices: Optional per-track clapperboard marker groups for c3d
            files whose markers cannot be identified by name (GUI-supplied).

    Returns:
        Alignment (offsets, confidence, warnings) parallel to ``media``.

    Raises:
        ValueError: If an audio/video track has no audio stream.
    """
    n = len(media)
    av_idx = [i for i, m in enumerate(media) if m.kind != "mocap"]
    mocap_idx = [i for i, m in enumerate(media) if m.kind == "mocap"]

    if not mocap_idx:
        return offsets_from_media(
            media, reference_index=reference_index, refine=refine,
            target_rate=target_rate, progress=progress, status=status,
            is_cancelled=is_cancelled,
        )
    if not av_idx:
        return Alignment(
            [0.0] * n, [0.0] * n,
            ["no audio/video track to sync motion capture against"],
        )

    warnings: list[str] = []
    # The reference anchors MFCC sync, so it must carry audio — never a c3d, and
    # never a silent video. Fall back to the first audio-bearing track.
    valid_ref = (
        0 <= reference_index < n
        and media[reference_index].kind != "mocap"
        and media[reference_index].has_audio
    )
    if not valid_ref:
        audio_refs = [i for i in av_idx if media[i].has_audio]
        # Reference fallback is normal (e.g. a c3d listed first) — log it, but
        # keep it out of `warnings`, which surfaces as a sync-quality alert.
        reference_index = audio_refs[0] if audio_refs else av_idx[0]

    logger.info(
        "sync reference: %s (%s)",
        media[reference_index].path.name, media[reference_index].kind,
    )
    av_media = [media[i] for i in av_idx]
    av_align = offsets_from_media(
        av_media, reference_index=av_idx.index(reference_index), refine=refine,
        target_rate=target_rate, progress=progress, status=status,
        is_cancelled=is_cancelled,
    )

    if status is not None:
        status("Detecting clapperboard…")
    mocap_offsets, mocap_conf, mocap_warn = bridge_mocap_offsets(
        av_media, av_align, [media[i] for i in mocap_idx], mocap_idx,
        marker_choices, target_rate=target_rate or 16000,
    )

    offsets = [0.0] * n
    confidence = [0.0] * n
    for local, i in enumerate(av_idx):
        offsets[i] = av_align.offsets[local]
        confidence[i] = av_align.confidence[local]
    for local, i in enumerate(mocap_idx):
        offsets[i] = mocap_offsets[local]
        confidence[i] = mocap_conf[local]
    warnings.extend(av_align.warnings)
    warnings.extend(mocap_warn)
    return Alignment(offsets, confidence, warnings)


def compute_sync_offsets(
    paths: list[Path],
    *,
    reference_index: int = 0,
    refine: Refine = "parabolic",
    target_rate: int | None = 16000,
    marker_choices: MarkerChoices | None = None,
    progress: Callable[[float], None] | None = None,
    status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Alignment:
    """Probe and align paths (audio/video by MFCC, c3d by clapperboard).

    Args:
        paths: Input media file paths.
        reference_index: Track whose timeline is the origin (must be A/V; a
            mocap reference falls back to the first A/V track).
        refine: Peak refinement.
        target_rate: Decode/resample rate for alignment; None keeps native.
        marker_choices: Optional c3d clapperboard marker groups (GUI-supplied).
        progress: Optional 0..1 callback.
        status: Optional callback receiving a label for the current stage.
        is_cancelled: Optional cooperative cancel check.

    Returns:
        Alignment (offsets, per-track confidence, warnings).

    Raises:
        ValueError: If an audio/video input has no audio stream.
    """
    if is_cancelled is not None and is_cancelled():
        n = len(paths)
        return Alignment([0.0] * n, [0.0] * n, [])
    media = []
    for i, p in enumerate(paths):
        if status is not None:
            status(f"Probing files ({i + 1}/{len(paths)})…")
        if progress is not None:
            progress(i / len(paths))
        media.append(probe(p))
    if progress is not None:
        progress(1.0)  # probing done; the load phase reports its own 0..1
    return align_media(
        media, reference_index=reference_index, refine=refine,
        target_rate=target_rate, marker_choices=marker_choices,
        progress=progress, status=status, is_cancelled=is_cancelled,
    )
