from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from clapsync.app import ExportResult, ExportSettings
from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo, family_display_order, is_av, probe
from clapsync.app.mocap import load_c3d
from clapsync.core import TimeRange
from clapsync.core.timerange import common_time_range, full_time_range
from clapsync.core.clap import (
    centroid,
    clapperboard_reliability,
    classify_clap_markers,
    detect_clap_motions,
)
from clapsync.gui import icons
from clapsync.gui.c3d_preview import C3DMarkerPreviewWidget
from clapsync.gui.export_dialog import ExportDialog, fmt_time
from clapsync.gui.progress import run_blocking_with_progress
from clapsync.gui.qt_preview import QtPreviewController
from clapsync.gui.timeline_widget import SyncTrimTimelineWidget, TrackState
from clapsync.gui.track_panel import TrackHeaderPanel
from clapsync.gui.workers import ExportWorker

logger = logging.getLogger(__name__)


def _decode_lane_waveforms(
    infos: list[MediaInfo],
) -> dict[int, np.ndarray]:
    """Decode a 16 kHz lane waveform for every audio-bearing track.

    Standalone audio files and video tracks with an audio stream alike get a
    waveform for their timeline lane. Pure decode (no Qt), so it can run off the
    GUI thread behind a busy dialog.
    """
    waves: dict[int, np.ndarray] = {}
    for i, info in enumerate(infos):
        if not info.has_audio:
            continue
        try:
            wave, _rate = load_audio(info.path, target_rate=16000)
            waves[i] = wave.reshape(-1).cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to load waveform for %s: %s", info.path.name, exc
            )
    return waves


class _MosaicContainer(QWidget):
    """QWidget that keeps a floating overlay label sized to fill its entire area."""

    def __init__(self, overlay: QLabel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay = overlay

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._overlay.setGeometry(0, 0, self.width(), self.height())


class SyncEditorWindow(QMainWindow):
    def __init__(
        self,
        video_paths: list[Path],
        offsets: list[float],
        output_dir: Path | None = None,
        use_proxies: bool = False,
        low_confidence: list[bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("clapsync")
        self.resize(1200, 700)

        self._use_proxies: bool = use_proxies
        self._low_confidence: list[bool] = list(low_confidence or [])
        # Clap-link state (only used with a c3d). One link per c3d track:
        # (sound_shared_s, movement_local_s); offset = sound - movement.
        self._auto_move: dict[int, float] = {}  # c3d track -> best movement local
        self._clap: dict[int, tuple[float, float]] = {}
        self._video_infos: list[MediaInfo] = []
        self._offsets: list[float] = list(offsets)
        self._trim_start: float = 0.0
        self._trim_end: float = 0.0
        self._global_pos: float = 0.0
        self._output_dir: Path = output_dir or Path.home()
        self._export_thread: QThread | None = None

        # Qt-native preview transport: one QMediaPlayer per audible track, a
        # master (the reference) that free-runs and is audible, the rest slaved
        # to its position. Built after probing, before the mosaic cells (they
        # embed each video track's QVideoWidget).
        self._preview: QtPreviewController | None = None
        self._is_playing: bool = False
        self._muted: list[bool] = []

        for path in video_paths:
            logger.debug("probe: %s", path.name)
            try:
                info = probe(path)
            except Exception as exc:
                QMessageBox.warning(self, "Warning", f"Could not read {path.name}: {exc}")
                continue
            if info.kind == "video":
                logger.debug(
                    "probe done: %s  duration=%.2fs  %dx%d",
                    path.name, info.duration, info.width, info.height,
                )
            else:
                logger.debug(
                    "probe done: %s  duration=%.2fs  %s",
                    path.name, info.duration, info.kind,
                )
            self._video_infos.append(info)

        if not self._video_infos:
            QMessageBox.critical(self, "Error", "No readable video files.")
            return

        total = max(info.duration + off for info, off in zip(self._video_infos, self._offsets))
        self._trim_end = total
        self._muted: list[bool] = [False] * len(self._video_infos)
        self._has_mocap = any(i.kind == "mocap" for i in self._video_infos)

        self._preview = QtPreviewController(
            self._video_infos, self._offsets, self._reference_index(),
            self._use_proxies, muted=self._muted, parent=self,
        )
        self._preview.position_changed.connect(self._on_position_changed)
        self._preview.eof.connect(self._on_eof)

        self._build_ui()
        self._init_clap_links()
        self._wire_signals()
        self._init_timeline()
        self._preview.seek(self._trim_start)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Video preview ─────────────────────────────────────────────────────
        # Audio waveforms go in their own stacked panel below, so only the
        # visual sources (video, motion capture) tile the mosaic.
        _MIN_CELL_W = 200
        # Only visual sources tile the mosaic: video and c3d (marker preview).
        # Audio has its waveform lane, no cell.
        n = sum(
            1 for info in self._video_infos
            if info.kind != "audio"
        )
        available_w = self.width() - 16
        if n > 0 and available_w // n >= _MIN_CELL_W:
            cols = n
        else:
            cols = max(1, math.ceil(math.sqrt(n)))

        # Overlay label (created first so _MosaicContainer can hold a reference)
        self._loading_overlay = QLabel("Loading…")
        self._loading_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 160);"
            "color: white;"
            "font-size: 18px;"
            "font-weight: bold;"
        )
        self._loading_overlay.setVisible(False)

        mosaic = _MosaicContainer(self._loading_overlay)
        mosaic.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._mosaic = mosaic

        grid = QGridLayout(mosaic)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        self._mocap_previews: dict[int, C3DMarkerPreviewWidget] = {}

        # Audio waveforms render inside their timeline lanes, not as widgets.
        # Decode every audio-bearing track's lane waveform up front, off the GUI
        # thread behind a busy dialog, so the mosaic build below (which touches
        # Qt widgets and must stay on the GUI thread) doesn't freeze the window
        # while it loads.
        self._track_waves = run_blocking_with_progress(
            lambda: _decode_lane_waveforms(self._video_infos),
            "Loading waveforms…",
        )

        slot = 0  # grid position among visual (non-audio) tracks
        for i, info in enumerate(self._video_infos):
            if info.kind == "audio":
                continue

            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)

            name_lbl = QLabel(info.path.name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("font-size: 10px; color: #555;")
            cell_layout.addWidget(name_lbl)

            if info.kind == "video":
                player = self._preview.video_widget(i)
                player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                player.setMinimumSize(160, 90)
                cell_layout.addWidget(player, stretch=1)
            else:  # mocap
                preview = self._build_mocap_preview(i, info)
                cell_layout.addWidget(preview, stretch=1)
                self._mocap_previews[i] = preview

            grid.addWidget(cell, slot // cols, slot % cols)
            slot += 1

        # Attach overlay as child of mosaic (after grid children so raise_() works)
        self._loading_overlay.setParent(mosaic)
        self._loading_overlay.setGeometry(0, 0, mosaic.width(), mosaic.height())
        self._loading_overlay.raise_()

        if slot > 0:  # at least one visual source
            root.addWidget(mosaic, stretch=3)

        # ── Controls row (above timeline) ─────────────────────────────────────
        controls_widget = QWidget()
        controls_widget.setFixedHeight(36)
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)

        self._prev_frame_btn = QPushButton("◀")
        self._prev_frame_btn.setFixedWidth(32)
        self._prev_frame_btn.setToolTip("Previous frame (←)")
        controls.addWidget(self._prev_frame_btn)

        self._play_btn = QPushButton(icons.icon("play"), "  Play")
        self._play_btn.setFixedWidth(90)
        controls.addWidget(self._play_btn)

        self._next_frame_btn = QPushButton("▶")
        self._next_frame_btn.setFixedWidth(32)
        self._next_frame_btn.setToolTip("Next frame (→)")
        controls.addWidget(self._next_frame_btn)

        self._time_label = QLabel("0:00.000 / 0:00.000")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self._time_label)
        controls.addStretch()

        self._zoom_out_btn = QPushButton(icons.icon("zoom-out"), "")
        self._zoom_in_btn = QPushButton(icons.icon("zoom-in"), "")
        self._zoom_fit_btn = QPushButton(icons.icon("zoom-fit"), "")
        for btn, tip in ((self._zoom_out_btn, "Zoom out"),
                         (self._zoom_in_btn, "Zoom in"),
                         (self._zoom_fit_btn, "Fit timeline")):
            btn.setFixedWidth(32)
            btn.setToolTip(tip)
            controls.addWidget(btn)

        self._loop_checkbox = QCheckBox("Loop")
        self._loop_checkbox.setChecked(True)
        controls.addWidget(self._loop_checkbox)
        root.addWidget(controls_widget)

        # ── Timeline (fixed track panel + scrolling timeline) ─────────────────
        self._timeline = SyncTrimTimelineWidget()
        self._timeline.setFixedHeight(200)
        self._track_panel = TrackHeaderPanel()
        self._track_panel.setFixedHeight(200)
        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(0, 0, 0, 0)
        timeline_row.setSpacing(0)
        timeline_row.addWidget(self._track_panel)
        timeline_row.addWidget(self._timeline, stretch=1)
        root.addLayout(timeline_row)

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        # Clap-link controls only make sense with a motion-capture track.
        self._set_sound_btn: QPushButton | None = None
        self._set_c3d_btn: QPushButton | None = None
        if self._has_mocap:
            self._set_c3d_btn = QPushButton("Set c3d clap (at playhead)")
            self._set_sound_btn = QPushButton("Set clap sound (at playhead)")
            self._set_c3d_btn.setToolTip(
                "Move the c3d so the frame at the playhead is its clap, aligned "
                "to the clap sound"
            )
            self._set_sound_btn.setToolTip(
                "Set the clap-sound point to the playhead and resync the c3d"
            )
            bottom.addWidget(self._set_c3d_btn)
            bottom.addWidget(self._set_sound_btn)
        bottom.addStretch()
        self._export_btn = QPushButton(icons.icon("export"), "Export…")
        self._export_btn.setFixedWidth(100)
        bottom.addWidget(self._export_btn)
        root.addLayout(bottom)

    def _build_mocap_preview(
        self, index: int, info: MediaInfo
    ) -> C3DMarkerPreviewWidget:
        """Load a c3d, build its marker preview, and record its motion clap.

        The motion clap's local time lets a user re-anchor this track to any
        clap marker they click on the timeline.
        """
        data = load_c3d(info.path)
        preview = C3DMarkerPreviewWidget(data.points, data.valid)
        preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview.setMinimumSize(160, 90)
        choice = classify_clap_markers(data.labels)
        if choice is not None:
            preview.set_groups(*choice)
            top, bottom = choice
            reliable = clapperboard_reliability(
                data.points[:, top], data.valid[:, top],
                data.points[:, bottom], data.valid[:, bottom],
            )
            motions = detect_clap_motions(
                centroid(data.points[:, top], data.valid[:, top]),
                centroid(data.points[:, bottom], data.valid[:, bottom]),
                data.point_rate, reliable,
            )
            if motions:
                self._auto_move[index] = motions[0].time  # best snap
        return preview

    def _wire_signals(self) -> None:
        self._play_btn.clicked.connect(self._on_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._on_play_pause)
        self._prev_frame_btn.clicked.connect(lambda: self._step_frame(-1))
        self._next_frame_btn.clicked.connect(lambda: self._step_frame(1))
        self._zoom_in_btn.clicked.connect(lambda: self._timeline.zoom_by(1.3))
        self._zoom_out_btn.clicked.connect(lambda: self._timeline.zoom_by(1 / 1.3))
        self._zoom_fit_btn.clicked.connect(self._timeline.zoom_to_fit)
        # Application-wide so a focused scrollbar/button doesn't swallow arrows.
        for key, direction in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(lambda d=direction: self._step_frame(d))
        self._timeline.playhead_changed.connect(self._on_playhead_seek)
        self._timeline.offsets_changed.connect(self._on_offsets_changed)
        if self._set_c3d_btn is not None:
            self._set_c3d_btn.clicked.connect(self._on_set_c3d_clap)
        if self._set_sound_btn is not None:
            self._set_sound_btn.clicked.connect(self._on_set_clap_sound)
        self._timeline.trim_changed.connect(self._on_trim_changed)
        self._timeline.vscroll_changed.connect(self._track_panel.set_scroll_y)
        self._track_panel.mute_changed.connect(self._on_mute)
        self._export_btn.clicked.connect(self._on_export)
        # The playhead and c3d preview are driven by the preview controller's
        # position_changed signal (emitted on its render tick while playing and
        # on every seek), so no separate render timer lives here.

    def _reference_index(self) -> int:
        """The sync-reference track: the first audio-bearing track, never a c3d.

        Mirrors align_media's reference choice so the timeline locks the track
        that is actually pinned to offset 0.
        """
        for i, info in enumerate(self._video_infos):
            if is_av(info.kind) and info.has_audio:
                return i
        for i, info in enumerate(self._video_infos):
            if is_av(info.kind):
                return i
        return 0

    def _init_timeline(self) -> None:
        reference = self._reference_index()
        # Lanes are grouped by family (A/V first, then c3d); the TrackState
        # keeps its global index, so data lookups are unaffected.
        order = family_display_order([info.kind for info in self._video_infos])
        tracks = [
            TrackState(
                index=i,
                label=self._video_infos[i].path.stem,
                offset_s=self._offsets[i],
                duration_s=self._video_infos[i].duration,
                locked=i == reference,
                muted=self._muted[i] if i < len(self._muted) else False,
                kind=self._video_infos[i].kind,
                warn=i < len(self._low_confidence) and self._low_confidence[i],
            )
            for i in order
        ]
        self._timeline.set_tracks(tracks)
        self._track_panel.set_tracks(tracks)
        self._timeline.set_track_waveforms(self._track_waves)
        self._rebuild_clap_markers()
        self._trim_start, self._trim_end = self._timeline.get_trim()

    def _update_time_label(self, current: float) -> None:
        self._time_label.setText(
            f"{fmt_time(current)} / {fmt_time(self._trim_end)}"
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _restart_from_trim_start(self) -> None:
        """Loop: seek every track back to the trim start and resume."""
        self._global_pos = self._trim_start
        self._preview.seek(self._trim_start)
        self._preview.play()

    def _show_play(self) -> None:
        self._play_btn.setIcon(icons.icon("play"))
        self._play_btn.setText("  Play")

    def _show_pause(self) -> None:
        self._play_btn.setIcon(icons.icon("pause"))
        self._play_btn.setText("  Pause")

    def _stop_playback(self) -> None:
        """Stop: pause the transport and reset the play button."""
        self._is_playing = False
        self._preview.pause()
        self._show_play()

    @Slot()
    def _on_eof(self) -> None:
        if self._is_playing and self._loop_checkbox.isChecked():
            self._restart_from_trim_start()
            return
        self._stop_playback()
        logger.debug("SyncEditorWindow: EOF — playback stopped")

    @Slot()
    def _on_play_pause(self) -> None:
        if self._is_playing:
            self._is_playing = False
            self._preview.pause()
            self._show_play()
        else:
            if self._global_pos >= self._trim_end - 0.05 or self._global_pos < self._trim_start:
                self._preview.seek(self._trim_start)
                self._global_pos = self._trim_start
            self._is_playing = True
            self._preview.play()
            self._show_pause()

    def _frame_step(self) -> float:
        """Seconds per frame at the finest rate present (video fps / c3d rate)."""
        rates = [i.fps for i in self._video_infos if i.fps]
        rates += [i.point_rate for i in self._video_infos if i.point_rate]
        return 1.0 / max(rates) if rates else 1.0 / 30.0

    def _step_frame(self, direction: int) -> None:
        """Nudge the playhead one frame; pauses playback for precise control."""
        if self._is_playing:
            self._stop_playback()
        self._on_playhead_seek(self._global_pos + direction * self._frame_step())

    @Slot(float)
    def _on_playhead_seek(self, global_s: float) -> None:
        global_s = max(self._trim_start, min(self._trim_end, global_s))
        self._global_pos = global_s
        # seek() repositions every player (and keeps them playing if they were);
        # it also emits position_changed, but update the UI here too so a paused
        # scrub reflects immediately without waiting on the player's async pos.
        self._preview.seek(global_s)
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)
        self._update_mocap_previews(global_s)

    def _update_mocap_previews(self, global_s: float) -> None:
        """Show each c3d's frame for the shared time (offset -> local -> frame)."""
        for idx, preview in self._mocap_previews.items():
            rate = self._video_infos[idx].point_rate or 0.0
            preview.set_frame(round((global_s - self._offsets[idx]) * rate))

    @Slot(list)
    def _on_offsets_changed(self, offsets: list[float]) -> None:
        self._offsets = offsets
        self._timeline.set_offsets(offsets)
        self._preview.set_offsets(offsets)
        # Trim bounds may have been adjusted by _clamp_trim_to_tracks in the timeline.
        self._trim_start, self._trim_end = self._timeline.get_trim()
        clamped = max(self._trim_start, min(self._trim_end, self._global_pos))
        self._global_pos = clamped
        self._preview.seek(clamped)
        self._timeline.set_playhead(clamped)
        self._update_time_label(clamped)

    def _mocap_indices(self) -> list[int]:
        return [i for i, inf in enumerate(self._video_infos)
                if inf.kind == "mocap"]

    def _init_clap_links(self) -> None:
        """Seed each c3d's clap link from the auto sync (best movement ↔ sound).

        offset is already the auto result, so the sound point is offset + best
        movement; the link is kept as (sound, movement) and reproduced exactly.
        """
        for idx in self._mocap_indices():
            move = self._auto_move.get(idx, 0.0)
            sound = self._offsets[idx] + move  # the auto sound position
            self._clap[idx] = (sound, move)

    def _rebuild_clap_markers(self) -> None:
        """Draw the single clap link per c3d: the movement flag and sound flag.

        Both sit at the sound time (offset = sound - movement keeps them
        aligned) — a vertical line showing where the c3d clap meets the sound.
        """
        if not self._has_mocap:
            self._timeline.set_clap_markers([])
            return
        markers: list[tuple[int, float, bool, str]] = []
        av_tracks = [i for i, inf in enumerate(self._video_infos)
                     if is_av(inf.kind)]
        for idx, (sound, move) in self._clap.items():
            markers.append((idx, self._offsets[idx] + move, True, "movement"))
            for i in av_tracks:
                markers.append((i, sound, True, "sound"))
        self._timeline.set_clap_markers(markers)

    def _on_set_c3d_clap(self) -> None:
        """Set each c3d's clap to the frame under the playhead, keep the sound.

        Moves the c3d so its clap frame lands on the current sound point.
        """
        for idx in self._mocap_indices():
            sound = self._clap.get(idx, (self._global_pos, 0.0))[0]
            move = self._global_pos - self._offsets[idx]  # local time at playhead
            self._clap[idx] = (sound, move)
            self._offsets[idx] = sound - move
        self._apply_clap_link()

    def _on_set_clap_sound(self) -> None:
        """Set the sound point to the playhead, keep the c3d clap; resync c3d."""
        sound = self._global_pos
        for idx in self._mocap_indices():
            move = self._clap.get(idx, (sound, 0.0))[1]
            self._clap[idx] = (sound, move)
            self._offsets[idx] = sound - move
        self._apply_clap_link()

    def _apply_clap_link(self) -> None:
        """Push updated c3d offsets to the timeline, trim, previews, markers."""
        self._timeline.set_offsets(self._offsets)
        self._refit_trim()
        self._on_offsets_changed(self._offsets)
        self._rebuild_clap_markers()

    def _refit_trim(self) -> None:
        """Reset the trim to the overlap of every track after offsets change."""
        tracks = [(info.duration, self._offsets[i])
                  for i, info in enumerate(self._video_infos)]
        if not tracks:
            return
        durations = [d for d, _ in tracks]
        offsets = [o for _, o in tracks]
        rng = common_time_range(durations, offsets)
        if rng.duration <= 0:  # no shared overlap -> fall back to the union
            rng = full_time_range(durations, offsets)
        self._timeline.set_trim(rng.start, rng.end)

    @Slot(float, float)
    def _on_trim_changed(self, trim_start: float, trim_end: float) -> None:
        self._trim_start = trim_start
        self._trim_end = trim_end
        clamped = max(trim_start, min(trim_end, self._global_pos))
        if clamped != self._global_pos:
            self._global_pos = clamped
            self._preview.seek(clamped)
            self._timeline.set_playhead(clamped)
            self._update_time_label(clamped)

    @Slot(int, bool)
    def _on_mute(self, index: int, muted: bool) -> None:
        if 0 <= index < len(self._muted):
            self._muted[index] = muted
        self._preview.set_muted(index, muted)

    @Slot(float)
    def _on_position_changed(self, global_s: float) -> None:
        self._global_pos = global_s
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)
        self._update_mocap_previews(global_s)
        if self._is_playing and global_s >= self._trim_end - 0.05:
            if self._loop_checkbox.isChecked():
                self._restart_from_trim_start()
            else:
                self._stop_playback()

    @Slot()
    def _on_export(self) -> None:
        if self._export_thread and self._export_thread.isRunning():
            QMessageBox.information(self, "Export", "An export is already in progress.")
            return

        dialog = ExportDialog(
            self._video_infos,
            self._offsets,
            self._trim_start,
            self._trim_end,
            output_dir=self._output_dir,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        params = dialog.get_export_params()
        output_dir = params.output_dir

        if not output_dir.exists():
            QMessageBox.warning(self, "Export", f"Directory does not exist: {output_dir}")
            return

        selected_media = [self._video_infos[i] for i in params.selected_indices]
        selected_offsets = [self._offsets[i] for i in params.selected_indices]

        progress_dialog = QProgressDialog(
            "Exporting…", "Cancel", 0, 1000, self
        )
        progress_dialog.setWindowTitle("Export Progress")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)
        progress_dialog.show()

        settings = ExportSettings(
            trim=TimeRange(self._trim_start, self._trim_end),
            output_dir=output_dir,
            target_width=params.target_width,
            target_height=params.target_height,
            output_fps=params.output_fps,
            audio_format=params.audio_format,
            audio_sample_rate=params.audio_sample_rate,
            audio_bitrate=params.audio_bitrate,
        )
        worker = ExportWorker(selected_media, selected_offsets, settings)
        thread = QThread(self)
        worker.moveToThread(thread)

        def on_progress(p: float) -> None:
            progress_dialog.setValue(int(p * 1000))

        def on_status(msg: str) -> None:
            progress_dialog.setLabelText(msg)

        def on_finished(results: list[ExportResult]) -> None:
            was_canceled = progress_dialog.wasCanceled()
            progress_dialog.close()
            thread.quit()
            thread.wait()
            if was_canceled:
                return

            ok = [r for r in results if r.ok]
            failed = [r for r in results if not r.ok]

            if failed:
                errors = "\n".join(f"  • {r.error}" for r in failed)
                if ok:
                    for r in ok:
                        logger.info("Exported: %s", r.path)
                    for r in failed:
                        logger.error("Export failed: %s", r.error)
                    msg = (
                        f"{len(ok)} video(s) exported to:\n{output_dir}\n\n"
                        f"{len(failed)} failed:\n{errors}"
                    )
                    QMessageBox.warning(self, "Export Completed with Errors", msg)
                else:
                    for r in failed:
                        logger.error("Export failed: %s", r.error)
                    QMessageBox.critical(self, "Export Failed", f"All exports failed:\n{errors}")
            else:
                for r in ok:
                    logger.info("Exported: %s", r.path)
                logger.info("Export complete: %d video(s) written to %s", len(ok), output_dir)
                files = "\n".join(f"  • {r.path.name}" for r in ok)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"{len(ok)} video(s) exported to:\n{output_dir}\n\n{files}",
                )

        # Pass self (a GUI-thread QObject) as the connection context so these
        # slots run on the GUI thread. Without a context object a queued
        # connection to a bare closure is delivered on the *sender's* thread —
        # the worker thread — so touching the dialog or popping a QMessageBox
        # from on_finished froze the app instead of reporting the result.
        worker.progress.connect(self, on_progress)
        worker.status.connect(self, on_status)
        worker.finished.connect(self, on_finished)
        progress_dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)
        thread.started.connect(worker.run)
        thread.start()

        self._export_thread = thread
        self._export_worker = worker

    def closeEvent(self, event) -> None:
        if self._preview is not None:
            self._preview.stop()
        super().closeEvent(event)
