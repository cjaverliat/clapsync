from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

from framepipe import GroupFrame, PyAvVideoDecoder, Resize, TrackSpec, VideoGroupDecoder

# Playback grid + preview downscale. All tracks resample to this fps and are
# shrunk on the GPU to this longest edge before crossing to host memory.
_FPS = 30.0
_MOSAIC_MAX_EDGE = 480
# GPU-resident NVDEC decode when a CUDA device is present, else software decode.
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_numpy(frame: torch.Tensor) -> np.ndarray:
    """(3, H, W) uint8 device tensor → (H, W, 3) uint8 host array for Qt."""
    return frame.permute(1, 2, 0).contiguous().cpu().numpy()


class VideoGroupWorker(QObject):
    """Drives a framepipe ``VideoGroupDecoder`` from a background thread.

    All tracks decode in parallel and emit as one atomic per-tick batch, so
    every ``frames_ready`` is synchronized to a single global timestamp.
    Pacing is wall-clock: the tick to show is derived from elapsed real time,
    so playback stays real-time and skips ticks (one seek) if decode lags.

    Commands are sent via ``cmd(*args)`` from any thread; ``run`` must be
    connected to a QThread's ``started`` signal.
    """

    # list[np.ndarray | None], global_ts, seek_gen  (None = black/gap track)
    frames_ready = Signal(object, float, int)
    loading_changed = Signal(bool)  # True while blocked in a user seek
    eof_reached = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cmds: queue.Queue = queue.Queue()
        self._seek_gen: int = 0

    def cmd(self, *args: Any) -> None:
        """Thread-safe: enqueue a command tuple."""
        self._cmds.put(args)

    @staticmethod
    def _open(paths: list[Path], offsets: list[float]) -> VideoGroupDecoder:
        resize = Resize(_MOSAIC_MAX_EDGE)
        tracks = [
            TrackSpec(
                PyAvVideoDecoder(p, device=_DEVICE, batch_size=8),
                offset=off,
                transform=resize,
            )
            for p, off in zip(paths, offsets)
        ]
        return VideoGroupDecoder(tracks, global_fps=_FPS)

    def _deliver(self, tick: GroupFrame) -> None:
        frames = [
            _to_numpy(f) if valid else None
            for f, valid in zip(tick.frames, tick.valid)
        ]
        self.frames_ready.emit(frames, tick.pts, self._seek_gen)

    @Slot()
    def run(self) -> None:
        group: VideoGroupDecoder | None = None
        offsets: list[float] = []
        playing = False
        t0 = 0.0
        base_idx = 0  # group tick index playback started from

        try:
            while True:
                stop = False
                while True:
                    try:
                        cmd = self._cmds.get_nowait()
                    except queue.Empty:
                        break
                    op = cmd[0]

                    if op == "open":
                        _, paths, offsets = cmd
                        if group is not None:
                            group.close()
                        try:
                            group = self._open(paths, list(offsets))
                        except Exception:
                            logger.exception("VideoGroupDecoder open failed")
                            group = None
                        playing = False

                    elif op == "seek":
                        ts, gen = cmd[1], cmd[2]
                        # Coalesce rapid seeks: keep only the last ts/gen.
                        while True:
                            try:
                                nxt = self._cmds.get_nowait()
                            except queue.Empty:
                                break
                            if nxt[0] == "seek":
                                _, ts, gen = nxt
                            else:
                                self._cmds.put(nxt)
                                break
                        self._seek_gen = gen
                        if group is not None:
                            self.loading_changed.emit(True)
                            group.seek_to_index(min(round(ts * _FPS), group.num_steps))
                            tick = group.next_tick()
                            if tick is not None:
                                self._deliver(tick)
                            self.loading_changed.emit(False)
                        t0, base_idx = time.perf_counter(), group.position if group else 0

                    elif op == "update_offsets":
                        new = list(cmd[1])
                        if group is not None:
                            for i, off in enumerate(new):
                                if i < len(offsets) and off != offsets[i]:
                                    group.set_offset(i, off)
                        offsets = new

                    elif op == "play":
                        playing = True
                        t0, base_idx = time.perf_counter(), group.position if group else 0

                    elif op == "pause":
                        playing = False

                    elif op == "stop":
                        stop = True
                        break

                if stop:
                    break

                if not playing or group is None:
                    try:
                        self._cmds.put(self._cmds.get(timeout=0.05))
                    except queue.Empty:
                        pass
                    continue

                # ── play the next tick in sequence ────────────────────────────
                tick = group.next_tick()
                if tick is None:
                    playing = False
                    self.eof_reached.emit()
                    continue
                self._deliver(tick)

                # Cap the rate at _FPS: sleep until this tick's wall slot. If
                # decode already ran past it (can't keep up), don't sleep — play
                # at the decode ceiling and let the clock drift, never seek to
                # catch up (a per-tick seek rebuilds every prefetcher: thrash).
                deadline = t0 + (group.position - base_idx) / _FPS
                while time.perf_counter() < deadline and self._cmds.empty():
                    QThread.msleep(1)
        except Exception:
            logger.exception("VideoGroupWorker crashed")
        finally:
            if group is not None:
                group.close()


class VideoPlayerWidget(QWidget):
    """Pure display widget for one video track.

    Receives frames via ``display_frame()`` — all decoding is managed
    externally by a ``VideoGroupWorker``.
    """

    position_changed = Signal(float)  # global seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._offset_s: float = 0.0
        self._global_pos: float = 0.0
        self._last_frame: np.ndarray | None = None

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: black;")
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMinimumSize(320, 180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def display_frame(self, frame: np.ndarray | None, global_pos: float) -> None:
        """Display a decoded frame. Call from the main/GUI thread only."""
        self._global_pos = global_pos
        if frame is not None:
            self.position_changed.emit(global_pos)
            self._display_frame(frame)
        else:
            self._show_black()

    def clear(self) -> None:
        self._label.clear()
        self._label.setStyleSheet("background-color: black;")
        self._last_frame = None

    @property
    def global_position(self) -> float:
        return self._global_pos

    def _show_black(self) -> None:
        self._display_frame(np.zeros((180, 320, 3), dtype=np.uint8))

    def _display_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        self._redraw_last_frame()

    def _redraw_last_frame(self) -> None:
        if self._last_frame is None:
            return
        frame = self._last_frame
        h, w = frame.shape[:2]
        lw, lh = self._label.width(), self._label.height()
        if lw <= 0 or lh <= 0:
            return
        image = QImage(frame.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(image).scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw_last_frame()
