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


# How many clap candidates to keep per track when surfacing timeline markers.
_MARKERS_PER_TRACK = 4


def detect_sound_anchor(
    av_media: list[MediaInfo], av_align: Alignment, target_rate: int = 16000
) -> float | None:
    """Find the clap time on the shared timeline from the A/V audio.

    Returns:
        Shared-timeline clap time in seconds, or None if no reliable anchor.
    """
    entries = _sound_candidates(
        av_media, av_align, target_rate, per_track=_MARKERS_PER_TRACK,
    )
    if not entries:
        return None
    usable = {k for _t, _s, k in entries}
    anchor, _winners = _best_cluster(entries, single_ok=len(usable) == 1)
    return anchor


def _sound_candidates(
    av_media: list[MediaInfo], av_align: Alignment, target_rate: int,
    per_track: int,
) -> list[tuple[float, float, int]]:
    """Detect clap candidates in confidently-synced A/V tracks.

    Returns:
        (shared_time, score, av_track_index) for each candidate.
    """
    usable = [
        k for k, m in enumerate(av_media)
        if m.has_audio and av_align.confidence[k] >= LOW_CONFIDENCE
    ]
    if not usable:  # e.g. a single reference track, or all low-confidence
        usable = [k for k, m in enumerate(av_media) if m.has_audio]

    entries: list[tuple[float, float, int]] = []
    for k in usable:
        wave, rate = load_audio(av_media[k].path, target_rate=target_rate)
        samples = wave.reshape(-1).cpu().numpy()
        for cand in clap.detect_clap_sound(samples, rate)[:per_track]:
            entries.append((cand.time + av_align.offsets[k], cand.score, k))
    return entries


def _best_cluster(
    entries: list[tuple[float, float, int]], single_ok: bool
) -> tuple[float | None, set[tuple[float, int]]]:
    """Anchor time and the winning cluster's (shared_time, track) members."""
    entries.sort()
    clusters: list[dict] = []
    for time, score, track in entries:
        if clusters and time - clusters[-1]["tmax"] <= _ANCHOR_TOL_S:
            c = clusters[-1]
            c["members"].append((time, track))
            c["scores"].append(score)
            c["tracks"].add(track)
            c["tmax"] = time
        else:
            clusters.append(
                {"members": [(time, track)], "scores": [score],
                 "tracks": {track}, "tmax": time}
            )

    accepted = [c for c in clusters if len(c["tracks"]) >= 2 or single_ok]
    if not accepted:
        return None, set()
    # Prefer the clap the most tracks agree on; on a tie prefer the earliest
    # (a slate is typically clapped at the start of a take). members are
    # time-sorted, so members[0][0] is the cluster's earliest detection.
    best = max(
        accepted, key=lambda c: (len(c["tracks"]), -c["members"][0][0])
    )
    weights = np.array(best["scores"])
    times = np.array([t for t, _track in best["members"]])
    anchor = float((weights * times).sum() / weights.sum())
    return anchor, set(best["members"])


def motion_clap_time(info: MediaInfo) -> float | None:
    """Local time of a c3d's clapperboard motion clap, or None if not found."""
    if info.kind != "mocap":
        return None
    data = load_c3d(info.path)
    choice = clap.classify_clap_markers(data.labels)
    if choice is None:
        return None
    top, bottom = choice
    motion = clap.detect_clap_motion(
        clap.centroid(data.points[:, top], data.valid[:, top]),
        clap.centroid(data.points[:, bottom], data.valid[:, bottom]),
        data.point_rate,
    )
    return motion.time if motion is not None else None


def clap_markers(
    media: list[MediaInfo], alignment: Alignment, target_rate: int = 16000
) -> list[tuple[int, float, bool]]:
    """Detected clap positions for the timeline.

    Surfaces every clap-sound candidate found in the A/V tracks, plus each
    synced c3d's motion clap, as (track_index, shared_time, is_winner). A c3d
    marker sits at its *actual* synced position (offset + motion time), so it
    always lands on the c3d lane and matches the preview animation. A/V
    candidates are winners when they agree with a synced c3d's clap.
    """
    av_idx = [i for i, m in enumerate(media) if m.kind != "mocap"]
    mocap_idx = [i for i, m in enumerate(media) if m.kind == "mocap"]
    if not mocap_idx or not av_idx:
        return []

    av_media = [media[i] for i in av_idx]
    av_align = Alignment(
        [alignment.offsets[i] for i in av_idx],
        [alignment.confidence[i] for i in av_idx],
        [],
    )
    entries = _sound_candidates(
        av_media, av_align, target_rate, per_track=_MARKERS_PER_TRACK,
    )

    markers: list[tuple[int, float, bool]] = []
    clap_positions: list[float] = []
    for i in mocap_idx:
        motion = motion_clap_time(media[i])
        if motion is not None and alignment.confidence[i] > 0:
            position = alignment.offsets[i] + motion
            clap_positions.append(position)
            markers.append((i, position, True))

    for shared, _score, k in entries:
        winner = any(
            abs(shared - pos) <= _ANCHOR_TOL_S for pos in clap_positions
        )
        markers.append((av_idx[k], shared, winner))
    return markers


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
