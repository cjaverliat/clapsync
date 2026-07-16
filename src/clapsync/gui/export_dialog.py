from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clapsync.app.media import MediaInfo

logger = logging.getLogger(__name__)


def fmt_time(s: float) -> str:
    """Seconds → ``m:ss.mmm`` label (shared by the editor and this dialog)."""
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


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
            f"{len(video_infos)} video(s) — "
            f"trim: {fmt_time(trim_start)} → {fmt_time(trim_end)} "
            f"(duration: {fmt_time(duration)})"
        )
        layout.addWidget(QLabel(summary))

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

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Resolution:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(res_labels)
        scale_row.addWidget(self._scale_combo)
        scale_row.addStretch()
        layout.addLayout(scale_row)

        native_fps = min((info.fps for info in video_infos), default=30.0)
        self._fps_values = [native_fps]
        fps_labels = [f"Native ({native_fps:g} fps)"]
        for std_fps in (60.0, 50.0, 30.0, 25.0, 24.0):
            if std_fps < native_fps - 0.01:
                self._fps_values.append(std_fps)
                fps_labels.append(f"{std_fps:g} fps")

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Frame rate:"))
        self._fps_combo = QComboBox()
        self._fps_combo.addItems(fps_labels)
        fps_row.addWidget(self._fps_combo)
        fps_row.addStretch()
        layout.addLayout(fps_row)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self._dir_edit = QLineEdit(str(output_dir or Path.home()))
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self._dir_edit.text())
        if d:
            self._dir_edit.setText(d)

    def get_export_params(self) -> tuple[int, int, float, Path]:
        w, h = self._resolutions[self._scale_combo.currentIndex()]
        fps = self._fps_values[self._fps_combo.currentIndex()]
        return w, h, fps, Path(self._dir_edit.text())
