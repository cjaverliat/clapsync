"""Lane layout decouples display row from track.index (family grouping).

Runs headless via Qt's offscreen platform. Guards the two changes that make
family grouping safe: lanes are positioned by display row, while get_offsets
stays keyed by track.index so the emitted offset list is never permuted.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from clapsync.gui.timeline_widget import (
    HEADER_H,
    TRACK_H,
    TRACK_PAD,
    SyncTrimTimelineWidget,
    TrackState,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["test"])


def _tracks_in_display_order() -> list[TrackState]:
    # Global indices: 0=video 1=c3d 2=fbx 3=audio. Display order groups A/V
    # (0, 3) first, then the c3d (1) with its fbx (2) beneath it. offset_s is
    # set to the global index so a permuted get_offsets would be detectable.
    kinds = {0: "video", 1: "mocap", 2: "fbx", 3: "audio"}
    order = [0, 3, 1, 2]
    return [
        TrackState(index=i, label=str(i), offset_s=float(i),
                   duration_s=1.0, kind=kinds[i])
        for i in order
    ]


def test_lane_row_follows_display_order(qapp):
    timeline = SyncTrimTimelineWidget()
    timeline.set_tracks(_tracks_in_display_order())

    assert timeline._row_by_index == {0: 0, 3: 1, 1: 2, 2: 3}
    for row, track in enumerate(timeline._tracks):
        rect = timeline._track_rect(track)
        expected_y = HEADER_H + TRACK_PAD + row * (TRACK_H + TRACK_PAD)
        assert rect.y() == expected_y


def test_get_offsets_stays_keyed_by_track_index(qapp):
    timeline = SyncTrimTimelineWidget()
    timeline.set_tracks(_tracks_in_display_order())

    # Global (index) order, not the [0, 3, 1, 2] display order.
    assert timeline.get_offsets() == [0.0, 1.0, 2.0, 3.0]
