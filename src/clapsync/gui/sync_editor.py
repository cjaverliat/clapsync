from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, QTimer, Slot
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

from framepipe import VideoInfo, get_video_info, get_display_size
from clapsync.app import ExportResult, ExportSettings
from clapsync.core import TimeRange
from clapsync.gui.export_dialog import ExportDialog
from clapsync.gui.video_player import VideoPlayerWidget, PlaybackClock, VideoGroupWorker
from clapsync.gui.timeline_widget import SyncTrimTimelineWidget, TrackState
from clapsync.gui.workers import ExportWorker

logger = logging.getLogger(__name__)

# Low resolution cap for the mosaic preview — keeps decoding fast.
_MOSAIC_MAX_EDGE = 480


def _fmt(s: float) -> str:
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("clapsync")
        self.resize(1200, 700)

        self._video_infos: list[VideoInfo] = []
        self._offsets: list[float] = list(offsets)
        self._trim_start: float = 0.0
        self._trim_end: float = 0.0
        self._global_pos: float = 0.0
        self._output_dir: Path = output_dir or Path.home()
        self._export_thread: QThread | None = None

        self._group_worker: VideoGroupWorker | None = None
        self._group_thread: QThread | None = None
        self._is_playing: bool = False
        self._seek_gen: int = 0

        for path in video_paths:
            logger.debug("get_video_info: %s", path.name)
            try:
                info = get_video_info(path)
            except Exception as exc:
                QMessageBox.warning(self, "Warning", f"Could not read {path.name}: {exc}")
                continue
            logger.debug(
                "get_video_info done: %s  duration=%.2fs  %dx%d",
                path.name, info.duration, info.width, info.height,
            )
            self._video_infos.append(info)

        if not self._video_infos:
            QMessageBox.critical(self, "Error", "No readable video files.")
            return

        total = max(info.duration + off for info, off in zip(self._video_infos, self._offsets))
        self._trim_end = total

        self._build_ui()
        self._wire_signals()
        self._init_timeline()
        self._load_all_videos()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Video preview ─────────────────────────────────────────────────────
        _MIN_CELL_W = 200
        n = len(self._video_infos)
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

        self._players: list[VideoPlayerWidget] = []
        for i, info in enumerate(self._video_infos):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)

            name_lbl = QLabel(info.path.name)
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("font-size: 10px; color: #555;")
            cell_layout.addWidget(name_lbl)

            player = VideoPlayerWidget()
            player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            player.setMinimumSize(160, 90)
            cell_layout.addWidget(player, stretch=1)

            self._players.append(player)
            grid.addWidget(cell, i // cols, i % cols)

        # Attach overlay as child of mosaic (after grid children so raise_() works)
        self._loading_overlay.setParent(mosaic)
        self._loading_overlay.setGeometry(0, 0, mosaic.width(), mosaic.height())
        self._loading_overlay.raise_()

        root.addWidget(mosaic, stretch=3)

        # ── Controls row (above timeline) ─────────────────────────────────────
        controls_widget = QWidget()
        controls_widget.setFixedHeight(36)
        controls = QHBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(90)
        controls.addWidget(self._play_btn)

        self._time_label = QLabel("0:00.000 / 0:00.000")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self._time_label)
        controls.addStretch()

        self._loop_checkbox = QCheckBox("Loop")
        self._loop_checkbox.setChecked(True)
        controls.addWidget(self._loop_checkbox)
        root.addWidget(controls_widget)

        # ── Timeline ──────────────────────────────────────────────────────────
        self._timeline = SyncTrimTimelineWidget()
        self._timeline.setFixedHeight(200)
        root.addWidget(self._timeline)

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._export_btn = QPushButton("Export…")
        self._export_btn.setFixedWidth(100)
        bottom.addWidget(self._export_btn)
        root.addLayout(bottom)

    def _wire_signals(self) -> None:
        self._play_btn.clicked.connect(self._on_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._on_play_pause)
        self._timeline.playhead_changed.connect(self._on_playhead_seek)
        self._timeline.offsets_changed.connect(self._on_offsets_changed)
        self._timeline.trim_changed.connect(self._on_trim_changed)
        self._export_btn.clicked.connect(self._on_export)
        # Note: position tracking is done directly in _on_frames_ready, not via
        # individual player signals, to avoid double-updates.

        # Debounce for the loading overlay: only show after 150 ms so that fast
        # seeks (typically <100 ms) don't cause a visible blink.
        self._loading_debounce = QTimer(self)
        self._loading_debounce.setSingleShot(True)
        self._loading_debounce.setInterval(150)
        self._loading_debounce.timeout.connect(
            lambda: (
                self._loading_overlay.setVisible(True),
                self._loading_overlay.raise_(),
            )
        )

    def _init_timeline(self) -> None:
        tracks = [
            TrackState(
                index=i,
                label=info.path.stem,
                offset_s=self._offsets[i],
                duration_s=info.duration,
                locked=i == 0,
            )
            for i, info in enumerate(self._video_infos)
        ]
        self._timeline.set_tracks(tracks)
        self._trim_start, self._trim_end = self._timeline.get_trim()

    def _load_all_videos(self) -> None:
        paths = [info.path for info in self._video_infos]
        display_sizes = [
            get_display_size(info.width, info.height, _MOSAIC_MAX_EDGE)
            for info in self._video_infos
        ]

        for i, player in enumerate(self._players):
            player._offset_s = self._offsets[i]

        self._stop_group_worker()

        worker = VideoGroupWorker()
        thread = QThread(self)
        worker.moveToThread(thread)

        worker.frames_ready.connect(
            self._on_frames_ready, Qt.ConnectionType.QueuedConnection
        )
        worker.loading_changed.connect(
            self._on_loading_changed, Qt.ConnectionType.QueuedConnection
        )
        worker.eof_reached.connect(
            self._on_eof, Qt.ConnectionType.QueuedConnection
        )
        thread.started.connect(worker.run)

        self._group_worker = worker
        self._group_thread = thread
        thread.start()

        logger.info(
            "_load_all_videos: opening %d video(s) via VideoGroupWorker", len(paths)
        )
        worker.cmd("open", paths, self._offsets[:], display_sizes)
        self._sync_seek_all(self._trim_start)

    def _stop_group_worker(self) -> None:
        if self._group_worker is not None:
            self._group_worker.cmd("stop")
        if self._group_thread is not None:
            self._group_thread.quit()
            self._group_thread.wait(2000)
        self._group_worker = None
        self._group_thread = None

    def _update_time_label(self, current: float) -> None:
        self._time_label.setText(f"{_fmt(current)} / {_fmt(self._trim_end)}")

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object, float, int)
    def _on_frames_ready(
        self,
        frames: list[tuple[np.ndarray | None, float]],
        global_ts: float,
        seek_gen: int,
    ) -> None:
        for player, (frame, local_pts_s) in zip(self._players, frames):
            player.display_frame(frame, local_pts_s, global_ts)
        # Reject frames that were emitted before the most recent seek was
        # processed by the worker — their timestamps pre-date the new position.
        if global_ts >= 0.0 and seek_gen >= self._seek_gen:
            self._on_position_changed(global_ts)

    @Slot(bool)
    def _on_loading_changed(self, loading: bool) -> None:
        if loading:
            # Show overlay only if loading takes longer than the debounce interval.
            self._loading_debounce.start()
        else:
            self._loading_debounce.stop()
            self._loading_overlay.setVisible(False)
            # If the user paused while the worker was blocked in a seek/load, the
            # "pause" command may have been processed before the load started and a
            # stale "play" could still be in flight. Re-assert the paused state.
            if not self._is_playing and self._group_worker is not None:
                self._group_worker.cmd("pause")

    @Slot()
    def _on_eof(self) -> None:
        if self._is_playing and self._loop_checkbox.isChecked():
            self._sync_seek_all(self._trim_start)
            self._global_pos = self._trim_start
            clock = PlaybackClock.now(self._trim_start)
            if self._group_worker is not None:
                self._group_worker.cmd("play", clock)
            return
        self._is_playing = False
        self._play_btn.setText("▶  Play")
        logger.debug("SyncEditorWindow: EOF — playback stopped")

    @Slot()
    def _on_play_pause(self) -> None:
        if self._is_playing:
            self._is_playing = False
            if self._group_worker is not None:
                self._group_worker.cmd("pause")
            self._sync_seek_all(self._global_pos)
            self._play_btn.setText("▶  Play")
        else:
            if self._global_pos >= self._trim_end - 0.05 or self._global_pos < self._trim_start:
                self._sync_seek_all(self._trim_start)
                self._global_pos = self._trim_start
            clock = PlaybackClock.now(self._global_pos)
            self._is_playing = True
            if self._group_worker is not None:
                self._group_worker.cmd("play", clock)
            self._play_btn.setText("⏸  Pause")

    def _sync_seek_all(self, global_s: float) -> None:
        self._seek_gen += 1
        if self._group_worker is None:
            return
        logger.debug("SyncEditorWindow: seek_all  global_ts=%.4fs", global_s)
        self._group_worker.cmd("seek", global_s, self._seek_gen)

    @Slot(float)
    def _on_playhead_seek(self, global_s: float) -> None:
        global_s = max(self._trim_start, min(self._trim_end, global_s))
        was_playing = self._is_playing
        self._global_pos = global_s
        self._sync_seek_all(global_s)
        if was_playing:
            clock = PlaybackClock.now(global_s)
            if self._group_worker is not None:
                self._group_worker.cmd("play", clock)
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)

    @Slot(list)
    def _on_offsets_changed(self, offsets: list[float]) -> None:
        self._offsets = offsets
        for i, player in enumerate(self._players):
            if i < len(offsets):
                player._offset_s = offsets[i]
        if self._group_worker is not None:
            self._group_worker.cmd("update_offsets", offsets[:])
        # Trim bounds may have been adjusted by _clamp_trim_to_tracks in the timeline.
        self._trim_start, self._trim_end = self._timeline.get_trim()
        clamped = max(self._trim_start, min(self._trim_end, self._global_pos))
        self._global_pos = clamped
        self._sync_seek_all(clamped)
        self._timeline.set_playhead(clamped)
        self._update_time_label(clamped)

    @Slot(float, float)
    def _on_trim_changed(self, trim_start: float, trim_end: float) -> None:
        self._trim_start = trim_start
        self._trim_end = trim_end
        clamped = max(trim_start, min(trim_end, self._global_pos))
        if clamped != self._global_pos:
            self._global_pos = clamped
            self._sync_seek_all(clamped)
            self._timeline.set_playhead(clamped)
            self._update_time_label(clamped)
        if self._is_playing:
            clock = PlaybackClock.now(self._global_pos)
            if self._group_worker is not None:
                self._group_worker.cmd("play", clock)

    @Slot(float)
    def _on_position_changed(self, global_s: float) -> None:
        self._global_pos = global_s
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)
        if self._is_playing and global_s >= self._trim_end - 0.05:
            if self._loop_checkbox.isChecked():
                self._sync_seek_all(self._trim_start)
                self._global_pos = self._trim_start
                clock = PlaybackClock.now(self._trim_start)
                if self._group_worker is not None:
                    self._group_worker.cmd("play", clock)
            else:
                self._is_playing = False
                if self._group_worker is not None:
                    self._group_worker.cmd("pause")
                self._play_btn.setText("▶  Play")

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

        target_width, target_height, output_fps, output_dir = dialog.get_export_params()

        if not output_dir.exists():
            QMessageBox.warning(self, "Export", f"Directory does not exist: {output_dir}")
            return

        progress_dialog = QProgressDialog("Preparing…", "Cancel", 0, 1000, self)
        progress_dialog.setWindowTitle("Export Progress")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)
        progress_dialog.show()

        paths = [info.path for info in self._video_infos]
        settings = ExportSettings(
            trim=TimeRange(self._trim_start, self._trim_end),
            output_dir=output_dir,
            target_width=target_width,
            target_height=target_height,
            output_fps=output_fps,
        )
        worker = ExportWorker(paths, self._offsets, settings)
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

        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
        worker.status.connect(on_status, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
        progress_dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)
        thread.started.connect(worker.run)
        thread.start()

        self._export_thread = thread
        self._export_worker = worker

    def closeEvent(self, event) -> None:
        self._stop_group_worker()
        super().closeEvent(event)
