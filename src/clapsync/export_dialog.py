from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from framepipe import VideoInfo
from clapsync.io.ffmpeg import export_synced_video

logger = logging.getLogger(__name__)


def _fmt(s: float) -> str:
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


class ExportResult:
    __slots__ = ("path", "error")

    def __init__(self, path: Path, error: str | None = None) -> None:
        self.path = path
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


class ExportWorker(QObject):
    progress = Signal(float)
    status = Signal(str)
    # results: list[ExportResult]
    finished = Signal(object)

    def __init__(
        self,
        video_infos: list[VideoInfo],
        offsets: list[float],
        trim_start: float,
        trim_end: float,
        target_width: int,
        target_height: int,
        output_fps: float,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self._infos = video_infos
        self._offsets = offsets
        self._trim_start = trim_start
        self._trim_end = trim_end
        self._target_width = target_width
        self._target_height = target_height
        self._output_fps = output_fps
        self._output_dir = output_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        n = len(self._infos)
        results: list[ExportResult] = []

        for i, (info, offset) in enumerate(zip(self._infos, self._offsets)):
            if self._cancelled:
                break

            local_start = max(0.0, self._trim_start - offset)
            local_end = min(info.duration, self._trim_end - offset)
            pad_start = max(0.0, offset - self._trim_start)
            pad_end = max(0.0, self._trim_end - (offset + info.duration))
            out_path = self._output_dir / f"{info.path.stem}_synced.mp4"

            self.status.emit(f"Exporting video {i + 1}/{n}: {info.path.name}")

            def cb(p: float, _i: int = i, _n: int = n) -> None:
                self.progress.emit((_i + p) / _n)

            error: str | None = None
            try:
                export_synced_video(
                    info.path,
                    out_path,
                    local_start,
                    local_end,
                    pad_start=pad_start,
                    pad_end=pad_end,
                    target_width=self._target_width,
                    target_height=self._target_height,
                    output_fps=self._output_fps,
                    progress_callback=cb,
                    is_cancelled=lambda: self._cancelled,
                )
                if self._cancelled:
                    out_path.unlink(missing_ok=True)
                elif not out_path.exists() or out_path.stat().st_size == 0:
                    error = f"{info.path.name}: output file is missing or empty"
                    out_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if self._cancelled:
                    out_path.unlink(missing_ok=True)

            if not self._cancelled:
                results.append(ExportResult(out_path, error))

        self.finished.emit(results)


class ExportDialog(QDialog):
    def __init__(
        self,
        video_infos: list[VideoInfo],
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
            f"trim: {_fmt(trim_start)} → {_fmt(trim_end)} "
            f"(duration: {_fmt(duration)})"
        )
        layout.addWidget(QLabel(summary))

        min_info = min(video_infos, key=lambda v: v.width * v.height)
        ref_w, ref_h = min_info.width, min_info.height
        scales = (1.0, 0.75, 0.5, 0.25)
        self._resolutions = tuple(
            (max(2, int(ref_w * s) & ~1), max(2, int(ref_h * s) & ~1)) for s in scales
        )

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Resolution:"))
        self._scale_combo = QComboBox()
        for w, h in self._resolutions:
            self._scale_combo.addItem(f"{w}×{h}")
        scale_row.addWidget(self._scale_combo)
        scale_row.addStretch()
        layout.addLayout(scale_row)

        min_fps = min((info.fps for info in video_infos), default=30.0)
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Frame rate:"))
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setDecimals(3)
        self._fps_spin.setSingleStep(1.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setValue(min_fps)
        fps_row.addWidget(self._fps_spin)
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
        fps = self._fps_spin.value()
        return w, h, fps, Path(self._dir_edit.text())
