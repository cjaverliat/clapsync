"""Bridge motion-capture tracks onto the shared timeline via a clapperboard.

The MFCC path syncs audio/video tracks to each other. A c3d file has no audio,
so it is placed by matching the clap it shares with the A/V world:

    offset_c3d = T_sound - T_c3d

where ``T_sound`` is the clap's time on the shared timeline (detected in the
A/V audio, lifted through each track's offset) and ``T_c3d`` is the clap's local
time in the mocap (the clapperboard arms meeting). This is consistent with the
locked convention ``shared = local + offset``.

Any step that cannot find a reliable clap leaves the track at offset 0 and
records a warning, rather than guessing.
"""
from __future__ import annotations

import logging

import numpy as np

from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo
from clapsync.app.mocap import load_c3d
from clapsync.core import clap
from clapsync.core.solver import Alignment, LOW_CONFIDENCE

logger = logging.getLogger(__name__)

# Detected clap times from different mics, once lifted to the shared timeline,
# should agree to within a couple of frames; cluster within this tolerance.
_ANCHOR_TOL_S = 0.08
# Nominal confidence for a mocap track that synced via clap, on the MFCC score
# scale (above LOW_CONFIDENCE so the GUI does not flag it for manual review).
_SYNCED_CONFIDENCE = 2.0 * LOW_CONFIDENCE

# Marker choices: global-track-index -> (top_indices, bottom_indices), supplied
# by the GUI when auto-classification needs manual resolution.
MarkerChoices = dict[int, tuple[list[int], list[int]]]


def detect_sound_anchor(
    av_media: list[MediaInfo], av_align: Alignment, target_rate: int = 16000
) -> float | None:
    """Find the clap time on the shared timeline from the A/V audio.

    Detects clap candidates in each confidently-synced A/V track, lifts them to
    shared time, and keeps the time cluster that several mics agree on (or the
    single track's best candidate when only one A/V track exists).

    Returns:
        Shared-timeline clap time in seconds, or None if no reliable anchor.
    """
    usable = [
        k for k, m in enumerate(av_media)
        if m.has_audio and av_align.confidence[k] >= LOW_CONFIDENCE
    ]
    if not usable:  # e.g. a single reference track, or all low-confidence
        usable = [k for k, m in enumerate(av_media) if m.has_audio]
    if not usable:
        return None

    entries: list[tuple[float, float, int]] = []  # (shared_time, score, track)
    for k in usable:
        wave, rate = load_audio(av_media[k].path, target_rate=target_rate)
        samples = wave.reshape(-1).cpu().numpy()
        for cand in clap.detect_clap_sound(samples, rate)[:3]:
            entries.append((cand.time + av_align.offsets[k], cand.score, k))

    if not entries:
        return None
    return _best_cluster(entries, single_ok=len(usable) == 1)


def _best_cluster(
    entries: list[tuple[float, float, int]], single_ok: bool
) -> float | None:
    """Score-weighted time of the cluster the most mics agree on."""
    entries.sort()
    clusters: list[dict] = []
    for time, score, track in entries:
        if clusters and time - clusters[-1]["tmax"] <= _ANCHOR_TOL_S:
            c = clusters[-1]
            c["times"].append(time)
            c["scores"].append(score)
            c["tracks"].add(track)
            c["tmax"] = time
        else:
            clusters.append(
                {"times": [time], "scores": [score], "tracks": {track},
                 "tmax": time}
            )

    accepted = [
        c for c in clusters if len(c["tracks"]) >= 2 or single_ok
    ]
    if not accepted:
        return None
    best = max(accepted, key=lambda c: (len(c["tracks"]), sum(c["scores"])))
    weights = np.array(best["scores"])
    times = np.array(best["times"])
    return float((weights * times).sum() / weights.sum())


def bridge_mocap_offsets(
    av_media: list[MediaInfo],
    av_align: Alignment,
    mocap_media: list[MediaInfo],
    mocap_indices: list[int],
    marker_choices: MarkerChoices | None,
    target_rate: int = 16000,
) -> tuple[list[float], list[float], list[str]]:
    """Compute shared-timeline offsets for the mocap tracks.

    Args:
        av_media: The audio/video tracks (already synced).
        av_align: Their alignment (offsets/confidence on the shared timeline).
        mocap_media: The c3d tracks to place.
        mocap_indices: Global input indices of ``mocap_media`` (for messages
            and marker-choice lookup).
        marker_choices: Optional per-track (top, bottom) marker index groups;
            falls back to name-based auto-classification.
        target_rate: Audio decode rate for clap-sound detection.

    Returns:
        (offsets, confidence, warnings) parallel to ``mocap_media``.
    """
    anchor = detect_sound_anchor(av_media, av_align, target_rate)
    offsets: list[float] = []
    confidence: list[float] = []
    warnings: list[str] = []

    for local, info in enumerate(mocap_media):
        name = info.path.name
        if anchor is None:
            offsets.append(0.0)
            confidence.append(0.0)
            warnings.append(
                f"{name}: no clapperboard sound found in audio/video — "
                f"motion capture left unsynced (offset 0)"
            )
            continue

        global_idx = mocap_indices[local]
        offset, warning = _offset_for_track(info, anchor, marker_choices,
                                             global_idx)
        offsets.append(offset)
        confidence.append(0.0 if warning else _SYNCED_CONFIDENCE)
        if warning:
            warnings.append(warning)

    return offsets, confidence, warnings


def _offset_for_track(
    info: MediaInfo,
    anchor: float,
    marker_choices: MarkerChoices | None,
    global_idx: int,
) -> tuple[float, str | None]:
    """Offset for one c3d track; (0.0, warning) when the clap is not found."""
    name = info.path.name
    data = load_c3d(info.path)

    choice = marker_choices.get(global_idx) if marker_choices else None
    if choice is None:
        choice = clap.classify_clap_markers(data.labels)
    if choice is None:
        return 0.0, (
            f"{name}: clapperboard markers not identified by name — "
            f"motion capture left unsynced (offset 0)"
        )

    top_idx, bottom_idx = choice
    top = clap.centroid(data.points[:, top_idx], data.valid[:, top_idx])
    bottom = clap.centroid(data.points[:, bottom_idx], data.valid[:, bottom_idx])
    motion = clap.detect_clap_motion(top, bottom, data.point_rate)
    if motion is None:
        return 0.0, (
            f"{name}: no clapperboard motion detected — motion capture left "
            f"unsynced (offset 0)"
        )

    return anchor - motion.time, None
