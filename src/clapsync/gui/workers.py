"""Qt worker adapters that run headless core operations off the GUI thread."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from clapsync.app import (
    ExportResult,
    ExportSettings,
    compute_sync_offsets,
)
from clapsync.app.decode import ensure_preview_proxy
from clapsync.app.export import export_media
from clapsync.app.media import probe

logger = logging.getLogger(__name__)

def _fmt_eta(seconds: float) -> str:
    """Format a remaining-time estimate as ``m:ss`` (or ``h:mm:ss``)."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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


class ProxyWorker(QObject):
    """Pre-generates preview proxies sequentially so the timeline opens warm.

    Transcodes run one at a time: proxy generation is bound by the single NVDEC
    decode engine, so running several at once only time-slices it (and risks
    VRAM exhaustion) for no throughput gain. Overall progress spans all files;
    the status line carries a running ETA from elapsed time against it.
    """

    progress_value = Signal(int)  # 0..1000
    status = Signal(str)
    finished = Signal(list)  # list[Path] proxy paths (unused; warms the cache)
    failed = Signal(str)

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self._paths = paths
        self._cancel = False

    def cancel(self) -> None:
        """Stop before the next file; an in-flight transcode still finishes."""
        self._cancel = True

    @Slot()
    def run(self) -> None:
        try:
            n = len(self._paths)
            start = time.monotonic()
            proxies = []
            for index, path in enumerate(self._paths):
                if self._cancel:
                    break

                def report(fraction: float, index: int = index) -> None:
                    overall = (index + fraction) / n
                    self.progress_value.emit(int(overall * 1000))
                    elapsed = time.monotonic() - start
                    # Withhold the ETA until there's enough signal to be useful.
                    if overall > 0.02:
                        eta = elapsed * (1.0 - overall) / overall
                        self.status.emit(
                            f"Generating preview proxy {index + 1}/{n}"
                            f"  (ETA {_fmt_eta(eta)})"
                        )
                    else:
                        self.status.emit(f"Generating preview proxy {index + 1}/{n}")

                proxies.append(ensure_preview_proxy(path, progress=report))
            self.finished.emit(proxies)
        except Exception as exc:  # noqa: BLE001
            logger.exception("proxy worker failed")
            self.failed.emit(str(exc))


def _run_with_progress(worker, title: str, parent):
    """Drive a progress-reporting worker on a QThread behind a modal dialog.

    The worker must expose ``progress_value(int 0..1000)``, ``status(str)``,
    ``finished(object)`` and ``failed(str)`` and a ``run`` slot. Shared by the
    offset and proxy pre-passes so both report progress identically.

    Returns:
        ``(result, error)``: the finished payload (``error`` None) on success,
        ``(None, message)`` on failure, or ``(None, None)`` if cancelled.
    """
    dialog = QProgressDialog("Preparing…", "Cancel", 0, 1000, parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(420)
    # Each per-file step ends at progress 1000. Left at the defaults the dialog
    # would auto-reset and auto-hide on every file, flickering closed and
    # reopening between "(i/N)" steps. Keep it up; dismissed on return.
    dialog.setAutoReset(False)
    dialog.setAutoClose(False)
    dialog.setValue(0)
    dialog.show()

    result = None
    error: str | None = None

    thread = QThread()
    worker.moveToThread(thread)

    def on_finished(payload) -> None:
        nonlocal result
        result = payload
        dialog.setValue(1000)
        thread.quit()

    def on_failed(msg: str) -> None:
        nonlocal error
        error = msg
        thread.quit()

    worker.progress_value.connect(dialog.setValue)
    worker.status.connect(dialog.setLabelText)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    thread.started.connect(worker.run)
    thread.start()

    while thread.isRunning():
        QApplication.processEvents()
        if dialog.wasCanceled():
            if hasattr(worker, "cancel"):
                worker.cancel()
            thread.requestInterruption()
            thread.quit()
            thread.wait(3000)
            return None, None
    thread.wait()

    dialog.close()  # autoClose is off, so dismiss it explicitly
    return result, error


def compute_offsets_with_progress(
    paths: list[Path], parent=None
) -> list[float] | None:
    """Compute sync offsets behind a modal progress dialog.

    Returns:
        Computed offsets, or None if cancelled or an error occurred.
    """
    result, error = _run_with_progress(
        OffsetWorker(paths), "Computing Offsets", parent
    )
    if error is not None:
        QMessageBox.critical(parent, "Error", f"Failed to compute offsets:\n{error}")
        return None
    return result


def prepare_proxies_with_progress(paths: list[Path], parent=None) -> bool:
    """Pre-generate preview proxies behind a modal progress dialog.

    Warms the on-disk proxy cache so the timeline's decode thread opens each
    track instantly instead of showing a black "Loading…" while it transcodes.

    Returns:
        True on success, False if cancelled or an error occurred.
    """
    result, error = _run_with_progress(
        ProxyWorker(paths), "Generating Preview Proxies", parent
    )
    if error is not None:
        QMessageBox.critical(
            parent, "Error", f"Failed to generate preview proxies:\n{error}"
        )
        return False
    return result is not None


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
        plural = "s" if n != 1 else ""

        # Probe first so the dialog names this phase instead of a vague
        # "Preparing…": on-disk metadata reads, no encoding yet.
        self.status.emit(f"Probing {n} source file{plural}…")
        media = [probe(p) for p in self._paths]
        if self._cancelled:
            self.finished.emit([])
            return

        self.status.emit(f"Exporting — 0/{n} tracks done, estimating…")
        start = time.monotonic()

        def _on_progress(frac: float) -> None:
            # export_media reports (i+1)/n after each track; there's no
            # sub-track signal, so the ETA is a linear extrapolation of the
            # per-track wall time and updates only on track boundaries.
            self.progress.emit(frac)
            done = min(n, max(1, math.ceil(frac * n)))
            if 0.0 < frac < 1.0:
                eta = (time.monotonic() - start) * (1.0 - frac) / frac
                self.status.emit(
                    f"Exporting — {done}/{n} tracks done  (ETA {_fmt_eta(eta)})"
                )
            else:
                self.status.emit(f"Exporting — {done}/{n} tracks done")

        results = export_media(
            media,
            self._offsets,
            self._settings,
            progress=_on_progress,
            is_cancelled=lambda: self._cancelled,
        )
        self.finished.emit(results)
