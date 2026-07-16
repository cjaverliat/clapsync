"""Qt worker adapters that run headless core operations off the GUI thread."""
from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from clapsync.app import (
    ExportResult,
    ExportSettings,
    compute_sync_offsets,
    export_tracks,
)

logger = logging.getLogger(__name__)


class OffsetWorker(QObject):
    progress_value = Signal(int)  # 0..1000
    status = Signal(str)  # human-readable current stage
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths

    @Slot()
    def run(self) -> None:
        try:
            offsets = compute_sync_offsets(
                self._paths,
                progress=lambda f: self.progress_value.emit(int(f * 1000)),
                status=lambda s: self.status.emit(s),
            )
            self.finished.emit(offsets)
        except Exception as exc:  # noqa: BLE001
            logger.exception("offset worker failed")
            self.failed.emit(str(exc))


def compute_offsets_with_progress(
    paths: list[Path], parent=None
) -> list[float] | None:
    """Run OffsetWorker on a QThread behind a modal progress dialog.

    Args:
        paths: Video file paths to analyze.
        parent: Optional Qt parent widget for the dialog.

    Returns:
        Computed offsets, or None if cancelled or an error occurred.
    """
    dialog = QProgressDialog("Preparing…", "Cancel", 0, 1000, parent)
    dialog.setWindowTitle("Computing Offsets")
    dialog.setMinimumWidth(420)
    # Each per-file load ends at progress(1.0) -> value 1000. Left at the
    # defaults the dialog would auto-reset and auto-hide on every file, so it
    # flickers closed and reopens between "Loading audio (i/N)". Keep it up;
    # it is dismissed when this function returns.
    dialog.setAutoReset(False)
    dialog.setAutoClose(False)
    dialog.setValue(0)
    dialog.show()

    result: list[float] | None = None
    error: str | None = None

    worker = OffsetWorker(paths)
    thread = QThread()
    worker.moveToThread(thread)

    def on_value(v: int) -> None:
        dialog.setValue(v)

    def on_status(msg: str) -> None:
        dialog.setLabelText(msg)

    def on_finished(offsets: list) -> None:
        nonlocal result
        result = offsets
        dialog.setValue(1000)
        thread.quit()

    def on_failed(msg: str) -> None:
        nonlocal error
        error = msg
        thread.quit()

    worker.progress_value.connect(on_value)
    worker.status.connect(on_status)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    thread.started.connect(worker.run)
    thread.start()

    while thread.isRunning():
        QApplication.processEvents()
        if dialog.wasCanceled():
            thread.requestInterruption()
            thread.quit()
            thread.wait(3000)
            return None
    thread.wait()

    dialog.close()  # autoClose is off, so dismiss it explicitly
    if error is not None:
        QMessageBox.critical(parent, "Error", f"Failed to compute offsets:\n{error}")
        return None
    return result


class ExportWorker(QObject):
    progress = Signal(float)
    status = Signal(str)
    finished = Signal(object)  # list[ExportResult]

    def __init__(
        self,
        paths: list[Path],
        offsets: list[float],
        settings: ExportSettings,
    ) -> None:
        super().__init__()
        self._paths = paths
        self._offsets = offsets
        self._settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        n = len(self._paths)

        def _on_progress(frac: float) -> None:
            # export_tracks reports (i+1)/n after each track; derive the
            # current item index so the dialog shows "Exporting i/n" — the
            # core API exposes only a float callback, not a status string.
            self.progress.emit(frac)
            done = min(n, max(1, math.ceil(frac * n)))
            self.status.emit(f"Exporting {done}/{n}")

        results = export_tracks(
            self._paths,
            self._offsets,
            self._settings,
            progress=_on_progress,
            is_cancelled=lambda: self._cancelled,
        )
        self.finished.emit(results)
