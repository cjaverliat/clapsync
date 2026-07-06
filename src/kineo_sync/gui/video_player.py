from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

from kineo_sync.io.video import VideoGroupDecoder, get_display_size


@dataclass
class PlaybackClock:
    """Shared temporal reference for multi-player synchronization.

    Set wall_t0 = time.perf_counter() and global_t0 = current global position
    once, then pass the same instance to every play() call.
    Each worker computes: target_wall = wall_t0 + (global_ts - global_t0)
    """
    wall_t0: float
    global_t0: float

    @staticmethod
    def now(global_t0: float) -> PlaybackClock:
        return PlaybackClock(wall_t0=time.perf_counter(), global_t0=global_t0)


class VideoGroupWorker(QObject):
    """Drives a VideoGroupDecoder from a background thread.

    All N videos are decoded in parallel and emitted as one atomic batch,
    guaranteeing that every ``frames_ready`` emission is synchronized across
    all videos to the same global timestamp.

    Commands are sent via ``cmd(*args)`` from any thread; the ``run`` slot
    must be connected to a QThread's ``started`` signal.
    """

    # list[(rgb_frame | None, local_pts_s)], global_ts, seek_gen
    frames_ready = Signal(object, float, int)
    # True  → decoder is blocked (seek / lag recovery)
    # False → decoder is running normally
    loading_changed = Signal(bool)
    eof_reached = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cmds: queue.Queue = queue.Queue()
        self._seek_gen: int = 0

    def cmd(self, *args: Any) -> None:
        """Thread-safe: enqueue a command tuple."""
        self._cmds.put(args)

    @Slot()
    def run(self) -> None:  # noqa: C901  (intentionally long timing loop)
        group: VideoGroupDecoder | None = None
        playing: bool = False
        clock: PlaybackClock | None = None
        last_emit_wall: float = -1.0

        _DISPLAY_FPS = 60.0
        _DISPLAY_INTERVAL = 1.0 / _DISPLAY_FPS
        _DECODE_BUDGET_S = 0.008  # start decoding this many seconds before planned emit

        try:
            while True:
                # ── drain command queue ────────────────────────────────────
                stop = False
                while True:
                    try:
                        cmd = self._cmds.get_nowait()
                    except queue.Empty:
                        break

                    op = cmd[0]

                    if op == "open":
                        _, paths, offsets, display_sizes = cmd
                        if group is not None:
                            group.close()
                        try:
                            group = VideoGroupDecoder(paths, offsets, display_sizes)
                        except Exception:
                            logger.exception("VideoGroupDecoder open failed")
                            group = None
                        clock = None
                        playing = False

                    elif op == "seek":
                        # Coalesce rapid seeks: keep only the last timestamp/gen.
                        ts: float = cmd[1]
                        gen: int = cmd[2] if len(cmd) > 2 else self._seek_gen
                        while True:
                            try:
                                nxt = self._cmds.get_nowait()
                                if nxt[0] == "seek":
                                    ts = nxt[1]
                                    gen = nxt[2] if len(nxt) > 2 else gen
                                else:
                                    self._cmds.put(nxt)
                                    break
                            except queue.Empty:
                                break

                        self._seek_gen = gen
                        clock = None
                        last_emit_wall = -1.0
                        if group is not None:
                            logger.debug("VideoGroupWorker: seek  global_ts=%.4fs", ts)
                            self.loading_changed.emit(True)
                            frames = group.read_frames(ts)
                            self.loading_changed.emit(False)
                            self.frames_ready.emit(frames, ts, self._seek_gen)

                    elif op == "update_offsets":
                        if group is not None:
                            group.update_offsets(cmd[1])
                            logger.debug("VideoGroupWorker: offsets updated to %s", cmd[1])

                    elif op == "play":
                        playing = True
                        clock = cmd[1]
                        last_emit_wall = -1.0
                        logger.debug(
                            "VideoGroupWorker: play  global_t0=%.4fs", clock.global_t0
                        )

                    elif op == "pause":
                        playing = False
                        logger.debug("VideoGroupWorker: pause")

                    elif op == "stop":
                        stop = True
                        break

                if stop:
                    break

                # ── idle when not playing ─────────────────────────────────
                if not playing or group is None or clock is None:
                    try:
                        nxt = self._cmds.get(timeout=0.05)
                        self._cmds.put(nxt)
                    except queue.Empty:
                        pass
                    continue

                # ── sleep until decode window ─────────────────────────────
                # Sleep until _DECODE_BUDGET_S before the planned emission so
                # advance_to_group is called exactly once per display cycle.
                # Calling it in a tight loop lets high-fps decoders drift ahead
                # of real-time, eventually triggering hard seeks and big jumps.
                now = time.perf_counter()
                next_emit_wall = (
                    last_emit_wall + _DISPLAY_INTERVAL if last_emit_wall >= 0 else now
                )
                decode_wake = next_emit_wall - _DECODE_BUDGET_S

                if now < decode_wake:
                    sleep_ms = max(1, int((decode_wake - now) * 1000) - 1)
                    if not self._cmds.empty():
                        continue
                    QThread.msleep(sleep_ms)
                    continue

                # ── advance all decoders to the planned emission time ─────
                # Use the scheduled emission wall-time (not "now") as the
                # target so we don't overshoot when decode finishes early.
                effective_target_wall = max(next_emit_wall, time.perf_counter())
                global_ts_target = clock.global_t0 + (effective_target_wall - clock.wall_t0)

                # Predict whether advance_to_group will need a random seek so
                # the loading signal can wrap the call.
                current_global = group.current_global_ts
                will_seek = (
                    current_global < 0
                    or global_ts_target < current_global
                    or (global_ts_target - current_global) > 2.0
                )

                if will_seek:
                    self.loading_changed.emit(True)

                frames, had_seek = group.advance_to_group(global_ts_target)

                if will_seek:
                    self.loading_changed.emit(False)

                ref_frame, ref_pts_s = frames[0]
                if ref_frame is None:
                    playing = False
                    logger.debug("VideoGroupWorker: EOF reached")
                    self.eof_reached.emit()
                    continue

                actual_global_ts = ref_pts_s + group.offsets[0]

                # If decoded PTS drifted >100 ms from the planned target,
                # reset the clock anchor so future iterations stay accurate.
                if abs(actual_global_ts - global_ts_target) > 0.1:
                    logger.debug(
                        "VideoGroupWorker: resetting clock  drift=%.3fs",
                        actual_global_ts - global_ts_target,
                    )
                    clock = PlaybackClock(
                        wall_t0=time.perf_counter(),
                        global_t0=actual_global_ts,
                    )

                # Precise sleep until 2 ms before planned emission.
                deadline = next_emit_wall - 0.002
                remaining = deadline - time.perf_counter()
                if remaining > 0.001:
                    if not self._cmds.empty():
                        continue
                    QThread.msleep(max(1, int(remaining * 1000) - 1))
                if time.perf_counter() < deadline and not self._cmds.empty():
                    continue
                if time.perf_counter() < deadline:
                    QThread.msleep(1)

                self.frames_ready.emit(frames, actual_global_ts, self._seek_gen)
                last_emit_wall = time.perf_counter()

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def display_frame(
        self,
        frame: np.ndarray | None,
        local_pts_s: float,
        global_pos: float,
    ) -> None:
        """Display a decoded frame.  Call from the main/GUI thread only."""
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

    # ------------------------------------------------------------------
    # Internal display helpers
    # ------------------------------------------------------------------

    def _show_black(self) -> None:
        black = np.zeros((180, 320, 3), dtype=np.uint8)
        self._display_frame(black)

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
        data = frame.tobytes()
        image = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw_last_frame()
