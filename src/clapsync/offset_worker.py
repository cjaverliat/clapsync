from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from clapsync.audio_sync import find_offset
from clapsync.io.audio import load_audio_from_video
from framepipe import get_video_info

logger = logging.getLogger(__name__)

# Each video occupies this many progress units: half for audio load, half for correlation.
_UNITS_PER_VIDEO = 200
_UNITS_AUDIO = 100
_UNITS_CORRELATE = 100


class OffsetWorker(QObject):
    progress = Signal(str)  # label text
    progress_value = Signal(int)  # absolute value in [0, n * _UNITS_PER_VIDEO]
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, video_paths: list[Path]) -> None:
        super().__init__()
        self._paths = video_paths

    @Slot()
    def run(self) -> None:
        try:
            paths = self._paths
            n = len(paths)

            # --- fps & duration detection (fast ffprobe calls) ---
            fps_values: list[float] = []
            durations: list[float] = []
            for path in paths:
                try:
                    info = get_video_info(path)
                    logger.info("%s: fps=%.4f", path.name, info.fps)
                    fps_values.append(info.fps)
                    durations.append(info.duration)
                except Exception:
                    fps_values.append(30.0)
                    durations.append(0.0)

            ref_fps = fps_values[0]
            logger.info("Using ref fps=%.4f for cross-correlation", ref_fps)

            # --- reference video (index 0): audio load only, no correlation ---
            msg = f"Loading audio from {paths[0].name} (reference)…"
            logger.info(msg)
            self.progress.emit(msg)

            def _ref_cb(frac: float) -> None:
                self.progress_value.emit(int(frac * _UNITS_AUDIO))

            try:
                ref_waveform, ref_rate = load_audio_from_video(
                    paths[0],
                    duration_s=durations[0],
                    progress_callback=_ref_cb,
                )
                logger.info(
                    "%s: audio sample_rate=%d Hz, samples=%d",
                    paths[0].name, ref_rate, ref_waveform.shape[-1],
                )
            except (ValueError, RuntimeError) as e:
                warning = f"WARNING: {e} — using offset 0.0 for reference"
                logger.warning("%s — using offset 0.0 for reference", e)
                self.progress.emit(warning)
                ref_waveform, ref_rate = None, 44100

            # Advance through both audio and correlate slots for the reference
            self.progress_value.emit(_UNITS_PER_VIDEO)

            # --- secondary videos ---
            lags: list[float] = [0.0]
            for i, path in enumerate(paths[1:], 1):
                video_base = i * _UNITS_PER_VIDEO
                audio_end = video_base + _UNITS_AUDIO
                video_end = video_base + _UNITS_PER_VIDEO

                msg = f"Loading audio from {path.name}…"
                logger.info(msg)
                self.progress.emit(msg)

                def _vid_cb(frac: float, _base: int = video_base) -> None:
                    self.progress_value.emit(_base + int(frac * _UNITS_AUDIO))

                try:
                    waveform, rate = load_audio_from_video(
                        path,
                        duration_s=durations[i],
                        progress_callback=_vid_cb,
                    )
                    logger.info(
                        "%s: audio sample_rate=%d Hz, samples=%d",
                        path.name, rate, waveform.shape[-1],
                    )
                    self.progress_value.emit(audio_end)

                    if ref_waveform is None:
                        lag_s = 0.0
                    else:
                        msg = f"Cross-correlating {path.name}…"
                        logger.info(msg)
                        self.progress.emit(msg)
                        _, lag_s = find_offset(ref_waveform, ref_rate, waveform, rate, ref_fps)
                        logger.info("Lag for %s: %.4f s", path.name, lag_s)

                except (ValueError, RuntimeError) as e:
                    warning = f"WARNING: {e} — using offset 0.0 for {path.name}"
                    logger.warning("%s — using offset 0.0 for %s", e, path.name)
                    self.progress.emit(warning)
                    lag_s = 0.0

                lags.append(lag_s)
                self.progress_value.emit(video_end)

            shift = -lags[0]
            offsets = [lag + shift for lag in lags]
            logger.info("Computed offsets: %s", [f"{o:.4f}s" for o in offsets])
            self.finished.emit(offsets)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in offset worker")
            self.failed.emit(str(exc))


def compute_offsets_with_progress(
        video_paths: list[Path], parent=None
) -> list[float] | None:
    n = len(video_paths)
    total_units = n * _UNITS_PER_VIDEO

    progress_dialog = QProgressDialog("Analyzing audio…", "Cancel", 0, total_units, parent)
    progress_dialog.setWindowTitle("Computing Offsets")
    progress_dialog.setValue(0)
    progress_dialog.setMinimumWidth(420)
    progress_dialog.show()

    result: list[float] | None = None
    error_msg: str | None = None

    worker = OffsetWorker(video_paths)
    thread = QThread()
    worker.moveToThread(thread)

    def on_progress(msg: str) -> None:
        progress_dialog.setLabelText(msg)

    def on_progress_value(val: int) -> None:
        progress_dialog.setValue(val)

    def on_finished(offsets: list) -> None:
        nonlocal result
        result = offsets
        progress_dialog.setValue(total_units)
        thread.quit()

    def on_failed(msg: str) -> None:
        nonlocal error_msg
        error_msg = msg
        progress_dialog.setValue(total_units)
        thread.quit()

    worker.progress.connect(on_progress)
    worker.progress_value.connect(on_progress_value)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    thread.started.connect(worker.run)
    thread.start()

    while thread.isRunning():
        QApplication.processEvents()
        if progress_dialog.wasCanceled():
            thread.requestInterruption()
            thread.quit()
            thread.wait(3000)
            return None

    thread.wait()

    if error_msg is not None:
        QMessageBox.critical(parent, "Error", f"Failed to compute offsets:\n{error_msg}")
        return None

    return result
