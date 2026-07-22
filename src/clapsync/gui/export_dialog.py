from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clapsync.app.media import MediaInfo
from clapsync.gui import icons

logger = logging.getLogger(__name__)

_LOSSY_AUDIO_FORMATS = {"mp3", "m4a"}


def fmt_time(s: float) -> str:
    """Seconds → ``m:ss.mmm`` label (shared by the editor and this dialog)."""
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


@dataclass(frozen=True)
class ExportParams:
    selected_indices: list[int]
    target_width: int | None
    target_height: int | None
    output_fps: float | None
    output_dir: Path
    audio_format: str | None
    audio_sample_rate: int | None
    audio_bitrate: int | None


class ExportDialog(QDialog):
    def __init__(
        self,
        video_infos: list[MediaInfo],
        offsets: list[float],
        trim_start: float,
        trim_end: float,
        output_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(480)

        self._video_infos = video_infos
        self._offsets = offsets
        self._trim_start = trim_start
        self._trim_end = trim_end

        layout = QVBoxLayout(self)

        duration = trim_end - trim_start
        summary = (
            f"{len(video_infos)} track(s) — "
            f"trim: {fmt_time(trim_start)} → {fmt_time(trim_end)} "
            f"(duration: {fmt_time(duration)})"
        )
        layout.addWidget(QLabel(summary))

        # --- Track selection ---
        layout.addWidget(QLabel("Tracks:"))
        self._track_list = QListWidget()
        for info in video_infos:
            item = QListWidgetItem(info.path.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._track_list.addItem(item)
        self._track_list.itemChanged.connect(self._on_track_changed)
        layout.addWidget(self._track_list)

        # --- Resolution / fps rows (hidden when no video track selected) ---
        self._scale_row_widget = QWidget()
        scale_row = QHBoxLayout(self._scale_row_widget)
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.addWidget(QLabel("Resolution:"))
        self._scale_combo = QComboBox()
        scale_row.addWidget(self._scale_combo)
        scale_row.addStretch()
        layout.addWidget(self._scale_row_widget)

        self._fps_row_widget = QWidget()
        fps_row = QHBoxLayout(self._fps_row_widget)
        fps_row.setContentsMargins(0, 0, 0, 0)
        fps_row.addWidget(QLabel("Frame rate:"))
        self._fps_combo = QComboBox()
        fps_row.addWidget(self._fps_combo)
        fps_row.addStretch()
        layout.addWidget(self._fps_row_widget)

        self._resolutions: list[tuple[int, int]] = []
        self._fps_values: list[float] = []

        # --- Audio params ---
        audio_format_row = QHBoxLayout()
        audio_format_row.addWidget(QLabel("Audio format:"))
        self._audio_format_combo = QComboBox()
        self._audio_format_combo.addItems(
            ["Same as source", "wav", "flac", "mp3", "m4a"]
        )
        self._audio_format_combo.currentIndexChanged.connect(self._on_audio_format_changed)
        audio_format_row.addWidget(self._audio_format_combo)
        audio_format_row.addStretch()
        layout.addLayout(audio_format_row)

        sample_rate_row = QHBoxLayout()
        sample_rate_row.addWidget(QLabel("Sample rate:"))
        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItems(["Native", "44100", "48000"])
        sample_rate_row.addWidget(self._sample_rate_combo)
        sample_rate_row.addStretch()
        layout.addLayout(sample_rate_row)

        bitrate_row = QHBoxLayout()
        bitrate_row.addWidget(QLabel("Bitrate:"))
        self._bitrate_combo = QComboBox()
        self._bitrate_combo.addItems(["128 kbps", "192 kbps", "256 kbps", "320 kbps"])
        self._bitrate_combo.setCurrentIndex(3)
        self._bitrate_combo.setEnabled(False)
        bitrate_row.addWidget(self._bitrate_combo)
        bitrate_row.addStretch()
        layout.addLayout(bitrate_row)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self._dir_edit = QLineEdit(str(output_dir or Path.home()))
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton(icons.icon("browse"), "Browse…")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._recompute_video_options()

    def _selected_indices(self) -> list[int]:
        indices = []
        for i in range(self._track_list.count()):
            if self._track_list.item(i).checkState() == Qt.CheckState.Checked:
                indices.append(i)
        return indices

    def _on_track_changed(self, _item: QListWidgetItem) -> None:
        self._recompute_video_options()

    def _on_audio_format_changed(self, _index: int) -> None:
        fmt = self._audio_format_combo.currentText()
        self._bitrate_combo.setEnabled(fmt in _LOSSY_AUDIO_FORMATS)

    def _recompute_video_options(self) -> None:
        selected = self._selected_indices()
        video_infos = [
            self._video_infos[i] for i in selected if self._video_infos[i].kind == "video"
        ]

        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(len(selected) > 0)

        if not video_infos:
            self._scale_row_widget.setVisible(False)
            self._fps_row_widget.setVisible(False)
            self._resolutions = []
            self._fps_values = []
            return

        self._scale_row_widget.setVisible(True)
        self._fps_row_widget.setVisible(True)

        min_info = min(video_infos, key=lambda v: v.width * v.height)
        ref_w, ref_h = min_info.width, min_info.height
        # Native first (the default), then standard heights below it, each at the
        # source aspect ratio (widths rounded to even for the encoders). Upscaling
        # to a standard larger than the source is never offered.
        self._resolutions = [(ref_w, ref_h)]
        res_labels = [f"Native ({ref_w}×{ref_h})"]
        for std_h in (2160, 1440, 1080, 720, 480):
            if std_h >= ref_h:
                continue
            std_w = max(2, round(std_h * ref_w / ref_h)) & ~1
            self._resolutions.append((std_w, std_h))
            res_labels.append(f"{std_h}p ({std_w}×{std_h})")

        self._scale_combo.blockSignals(True)
        self._scale_combo.clear()
        self._scale_combo.addItems(res_labels)
        self._scale_combo.blockSignals(False)

        native_fps = min((info.fps for info in video_infos if info.fps), default=30.0)
        self._fps_values = [native_fps]
        fps_labels = [f"Native ({native_fps:g} fps)"]
        for std_fps in (60.0, 50.0, 30.0, 25.0, 24.0):
            if std_fps < native_fps - 0.01:
                self._fps_values.append(std_fps)
                fps_labels.append(f"{std_fps:g} fps")

        self._fps_combo.blockSignals(True)
        self._fps_combo.clear()
        self._fps_combo.addItems(fps_labels)
        self._fps_combo.blockSignals(False)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def get_export_params(self) -> ExportParams:
        if self._resolutions and self._fps_values:
            target_width, target_height = self._resolutions[self._scale_combo.currentIndex()]
            output_fps = self._fps_values[self._fps_combo.currentIndex()]
        else:
            target_width = target_height = None
            output_fps = None

        audio_format_text = self._audio_format_combo.currentText()
        audio_format = None if audio_format_text == "Same as source" else audio_format_text

        sample_rate_text = self._sample_rate_combo.currentText()
        audio_sample_rate = None if sample_rate_text == "Native" else int(sample_rate_text)

        audio_bitrate: int | None = None
        if self._bitrate_combo.isEnabled():
            kbps = int(self._bitrate_combo.currentText().split()[0])
            audio_bitrate = kbps * 1000

        return ExportParams(
            selected_indices=self._selected_indices(),
            target_width=target_width,
            target_height=target_height,
            output_fps=output_fps,
            output_dir=Path(self._dir_edit.text()),
            audio_format=audio_format,
            audio_sample_rate=audio_sample_rate,
            audio_bitrate=audio_bitrate,
        )
