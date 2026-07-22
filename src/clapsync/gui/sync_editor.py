from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

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

from clapsync.app import ExportResult, ExportSettings
from clapsync.app.decode import load_audio
from clapsync.app.media import MediaInfo, probe
from clapsync.app.mocap import load_c3d
from clapsync.core import TimeRange
from clapsync.core.timerange import common_time_range, full_time_range
from clapsync.core.clap import centroid, classify_clap_markers, detect_clap_motions
from clapsync.gui import icons
from clapsync.gui.audio_engine import AudioEngine
from clapsync.gui.c3d_preview import C3DMarkerPreviewWidget
from clapsync.gui.export_dialog import ExportDialog, fmt_time
from clapsync.gui.video_player import VideoPlayerWidget, VideoGroupWorker
from clapsync.gui.timeline_widget import SyncTrimTimelineWidget, TrackState
from clapsync.gui.track_panel import TrackHeaderPanel
from clapsync.gui.waveform_widget import WaveformWidget
from clapsync.gui.workers import ExportWorker

logger = logging.getLogger(__name__)


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

        self._group_worker: VideoGroupWorker | None = None
        self._group_thread: QThread | None = None
        self._is_playing: bool = False
        self._seek_gen: int = 0
        self._video_eof_pending: bool = False
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

        self._build_ui()
        self._init_clap_links()
        self._wire_signals()
        self._init_timeline()
        self._init_audio()
        self._load_all_videos()

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
        n = sum(1 for info in self._video_infos if info.kind != "audio")
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
        self._video_slots: list[int] = []
        self._waveforms: dict[int, WaveformWidget] = {}
        self._mocap_previews: dict[int, C3DMarkerPreviewWidget] = {}

        # Vertical audio panel: full-width waveforms stack better than mosaic cells.
        audio_panel = QWidget()
        audio_layout = QVBoxLayout(audio_panel)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.setSpacing(3)

        slot = 0  # grid position among visual (non-audio) tracks
        for i, info in enumerate(self._video_infos):
            if info.kind == "audio":
                self._waveforms[i] = self._build_audio_row(audio_layout, info)
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
                player = VideoPlayerWidget()
                player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                player.setMinimumSize(160, 90)
                cell_layout.addWidget(player, stretch=1)
                self._players.append(player)
                self._video_slots.append(i)
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
        if self._waveforms:
            root.addWidget(audio_panel, stretch=1 if slot > 0 else 3)

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

    def _build_audio_row(
        self, layout: QVBoxLayout, info: MediaInfo
    ) -> WaveformWidget:
        """Add a full-width waveform row (name + waveform) to the audio panel."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        name_lbl = QLabel(info.path.name)
        name_lbl.setFixedWidth(150)
        name_lbl.setToolTip(info.path.name)
        name_lbl.setStyleSheet("font-size: 10px; color: #555;")

        wf = WaveformWidget()
        wf.setMinimumHeight(56)
        wf.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        try:
            waveform, _rate = load_audio(info.path, target_rate=16000)
            wf.set_waveform(waveform.reshape(-1).cpu().numpy())
        except Exception as exc:
            logger.warning(
                "failed to load waveform for %s: %s", info.path.name, exc
            )

        row_layout.addWidget(name_lbl)
        row_layout.addWidget(wf, stretch=1)
        layout.addWidget(row)
        return wf

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
            motions = detect_clap_motions(
                centroid(data.points[:, top], data.valid[:, top]),
                centroid(data.points[:, bottom], data.valid[:, bottom]),
                data.point_rate,
            )
            if motions:
                self._auto_move[index] = motions[0].time  # best snap
        return preview

    def _wire_signals(self) -> None:
        self._play_btn.clicked.connect(self._on_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(self._on_play_pause)
        self._prev_frame_btn.clicked.connect(lambda: self._step_frame(-1))
        self._next_frame_btn.clicked.connect(lambda: self._step_frame(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(
            lambda: self._step_frame(-1)
        )
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            lambda: self._step_frame(1)
        )
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

    def _reference_index(self) -> int:
        """The sync-reference track: the first audio-bearing track, never a c3d.

        Mirrors align_media's reference choice so the timeline locks the track
        that is actually pinned to offset 0.
        """
        for i, info in enumerate(self._video_infos):
            if info.kind != "mocap" and info.has_audio:
                return i
        for i, info in enumerate(self._video_infos):
            if info.kind != "mocap":
                return i
        return 0

    def _init_timeline(self) -> None:
        reference = self._reference_index()
        tracks = [
            TrackState(
                index=i,
                label=info.path.stem,
                offset_s=self._offsets[i],
                duration_s=info.duration,
                locked=i == reference,
                muted=self._muted[i] if i < len(self._muted) else False,
                kind=info.kind,
                warn=i < len(self._low_confidence) and self._low_confidence[i],
            )
            for i, info in enumerate(self._video_infos)
        ]
        self._timeline.set_tracks(tracks)
        self._track_panel.set_tracks(tracks)
        self._rebuild_clap_markers()
        self._trim_start, self._trim_end = self._timeline.get_trim()
        self._update_waveform_windows()

    def _update_waveform_windows(self) -> None:
        for idx, wf in self._waveforms.items():
            info = self._video_infos[idx]
            wf.set_window(
                self._offsets[idx], info.duration, self._trim_start, self._trim_end
            )

    def _init_audio(self) -> None:
        self._audio = AudioEngine(self)
        waveforms = []
        for info in self._video_infos:
            if info.kind == "mocap":
                waveforms.append(torch.zeros(1, 1))  # motion capture is silent
                continue
            try:
                w, _rate = load_audio(info.path, target_rate=48000)
            except Exception as exc:
                logger.warning(
                    "failed to load audio track for %s: %s", info.path.name, exc
                )
                w = torch.zeros(1, 1)
            waveforms.append(w)
        self._audio.set_tracks(waveforms, self._offsets)
        self._audio.set_muted(self._muted)
        self._audio.position_changed.connect(self._on_audio_tick)

    def _load_all_videos(self) -> None:
        paths = [self._video_infos[i].path for i in self._video_slots]
        video_offsets = [self._offsets[i] for i in self._video_slots]

        self._stop_group_worker()

        if not paths:
            # No video tracks: playback is driven entirely by the audio engine.
            self._sync_seek_all(self._trim_start)
            return

        worker = VideoGroupWorker(use_proxies=self._use_proxies)
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
        worker.cmd("open", paths, video_offsets)
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
        self._time_label.setText(
            f"{fmt_time(current)} / {fmt_time(self._trim_end)}"
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(object, float, int)
    def _on_frames_ready(
        self,
        frames: list[np.ndarray | None],
        global_ts: float,
        seek_gen: int,
    ) -> None:
        self._video_eof_pending = False
        for player, frame in zip(self._players, frames):
            player.display_frame(frame)
        # Reject frames that were emitted before the most recent seek was
        # processed by the worker — their timestamps pre-date the new position.
        if global_ts >= 0.0 and seek_gen >= self._seek_gen:
            self._on_position_changed(global_ts)
            # Re-lock audio to the video clock only on *genuine* drift. The
            # 0.5s threshold clears QAudioSink's normal buffer-latency lead
            # (position_s() counts samples pulled, ~100-200ms ahead of what's
            # audible), so this fires rarely — and seek() is now a cheap cursor
            # move, not a device restart.
            if self._audio.enabled and abs(global_ts - self._audio.position_s()) > 0.5:
                self._audio.seek(global_ts)

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

    def _restart_from_trim_start(self) -> None:
        """Loop: seek every track back to the trim start and resume."""
        self._video_eof_pending = False
        self._sync_seek_all(self._trim_start)
        self._global_pos = self._trim_start
        if self._group_worker is not None:
            self._group_worker.cmd("play")
        if self._audio.enabled:
            self._audio.play()

    def _show_play(self) -> None:
        self._play_btn.setIcon(icons.icon("play"))
        self._play_btn.setText("  Play")

    def _show_pause(self) -> None:
        self._play_btn.setIcon(icons.icon("pause"))
        self._play_btn.setText("  Pause")

    def _stop_playback(self) -> None:
        """Stop: pause the worker (a no-op at EOF) and reset the play button."""
        self._is_playing = False
        self._video_eof_pending = False
        if self._group_worker is not None:
            self._group_worker.cmd("pause")
        self._audio.pause()
        self._show_play()

    @Slot()
    def _on_eof(self) -> None:
        if self._is_playing and self._loop_checkbox.isChecked():
            self._restart_from_trim_start()
            return
        if (
            self._is_playing
            and self._global_pos < self._trim_end - 0.05
            and self._audio.enabled
        ):
            # Video ran out of frames before the trim end, but the audio mix
            # still has content — let it carry playback to trim_end instead
            # of stopping early.
            self._video_eof_pending = True
            return
        self._stop_playback()
        logger.debug("SyncEditorWindow: EOF — playback stopped")

    @Slot()
    def _on_play_pause(self) -> None:
        if self._is_playing:
            self._is_playing = False
            self._video_eof_pending = False
            if self._group_worker is not None:
                self._group_worker.cmd("pause")
            self._audio.pause()
            self._sync_seek_all(self._global_pos)
            self._show_play()
        else:
            if self._global_pos >= self._trim_end - 0.05 or self._global_pos < self._trim_start:
                self._sync_seek_all(self._trim_start)
                self._global_pos = self._trim_start
            self._is_playing = True
            if self._group_worker is not None:
                self._group_worker.cmd("play")
            if self._audio.enabled:
                self._audio.play()
            self._show_pause()

    def _sync_seek_all(self, global_s: float) -> None:
        self._seek_gen += 1
        self._audio.seek(global_s)
        if self._group_worker is None:
            return
        logger.debug("SyncEditorWindow: seek_all  global_ts=%.4fs", global_s)
        self._group_worker.cmd("seek", global_s, self._seek_gen)

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
        was_playing = self._is_playing
        self._global_pos = global_s
        self._sync_seek_all(global_s)
        if was_playing and self._group_worker is not None:
            self._group_worker.cmd("play")
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)

    @Slot(list)
    def _on_offsets_changed(self, offsets: list[float]) -> None:
        self._offsets = offsets
        self._audio.set_offsets(offsets)
        if self._group_worker is not None:
            video_offsets = [offsets[i] for i in self._video_slots]
            self._group_worker.cmd("update_offsets", video_offsets)
        # Trim bounds may have been adjusted by _clamp_trim_to_tracks in the timeline.
        self._trim_start, self._trim_end = self._timeline.get_trim()
        self._update_waveform_windows()
        clamped = max(self._trim_start, min(self._trim_end, self._global_pos))
        self._global_pos = clamped
        self._sync_seek_all(clamped)
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
                     if inf.kind != "mocap"]
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
        """Reset the trim to the audio/video overlap after offsets change."""
        av = [(info.duration, self._offsets[i])
              for i, info in enumerate(self._video_infos)
              if info.kind != "mocap"]
        if not av:
            return
        durations = [d for d, _ in av]
        offsets = [o for _, o in av]
        rng = common_time_range(durations, offsets)
        if rng.duration <= 0:  # no shared overlap -> fall back to the union
            rng = full_time_range(durations, offsets)
        self._timeline.set_trim(rng.start, rng.end)

    @Slot(float, float)
    def _on_trim_changed(self, trim_start: float, trim_end: float) -> None:
        self._trim_start = trim_start
        self._trim_end = trim_end
        self._update_waveform_windows()
        clamped = max(trim_start, min(trim_end, self._global_pos))
        if clamped != self._global_pos:
            self._global_pos = clamped
            self._sync_seek_all(clamped)
            self._timeline.set_playhead(clamped)
            self._update_time_label(clamped)
        if self._is_playing and self._group_worker is not None:
            self._group_worker.cmd("play")

    @Slot(int, bool)
    def _on_mute(self, index: int, muted: bool) -> None:
        if 0 <= index < len(self._muted):
            self._muted[index] = muted
        self._audio.set_muted(self._muted)

    def _on_audio_tick(self, position_s: float) -> None:
        # The audio engine is the fallback clock: it drives playback state
        # whenever there is no video (or the video has run dry but audio is
        # still carrying playback to the trim end). Otherwise video frames
        # (which arrive far more frequently) are the master clock.
        if not self._players or self._video_eof_pending:
            self._on_position_changed(position_s)

    @Slot(float)
    def _on_position_changed(self, global_s: float) -> None:
        self._global_pos = global_s
        self._timeline.set_playhead(global_s)
        self._update_time_label(global_s)
        for wf in self._waveforms.values():
            wf.set_playhead(global_s)
        for idx, preview in self._mocap_previews.items():
            rate = self._video_infos[idx].point_rate or 0.0
            preview.set_frame(round((global_s - self._offsets[idx]) * rate))
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

        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
        worker.status.connect(on_status, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
        progress_dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)
        thread.started.connect(worker.run)
        thread.start()

        self._export_thread = thread
        self._export_worker = worker

    def closeEvent(self, event) -> None:
        audio = getattr(self, "_audio", None)
        if audio is not None:
            audio.pause()
        self._stop_group_worker()
        super().closeEvent(event)
