from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from threading import Thread
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

    Display rate is decoupled from decode rate: frames are emitted on a steady
    ``_FPS`` wall-clock grid, re-emitting the most recent decoded frame when
    decode hasn't produced a new one, while decode runs best-effort but never
    ahead of the clock. When decode keeps up (e.g. 1080p) every emit is a fresh
    frame — full motion; when it can't (heavy 4K), the display holds ``_FPS``
    and motion degrades gracefully instead of the playback clock lagging.

    Commands are sent via ``cmd(*args)`` from any thread; ``run`` must be
    connected to a QThread's ``started`` signal.
    """

    # list[np.ndarray | None], global_ts, seek_gen  (None = black/gap track)
    frames_ready = Signal(object, float, int)
    # True while blocked in a user seek, or while playback decode trails the
    # clock (re-emitting a stale frame). False once resolved.
    loading_changed = Signal(bool)
    eof_reached = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cmds: queue.Queue = queue.Queue()
        self._seek_gen: int = 0
        # Shared between the decode thread (writer) and the emit loop (reader).
        self._latest: tuple[list[np.ndarray | None], float] | None = None
        self._playing: bool = False
        self._stop: bool = False

    def cmd(self, *args: Any) -> None:
        """Thread-safe: enqueue a command tuple."""
        self._cmds.put(args)

    @staticmethod
    def _open(paths: list[Path], offsets: list[float]) -> VideoGroupDecoder:
        resize = Resize(_MOSAIC_MAX_EDGE)
        # batch_size=1: each decoded batch is full-resolution RGB held on the
        # GPU until resize, so on 4K a larger batch multiplies both GPU memory
        # (measured 6 GB peak for 3 tracks at batch 8 vs ~1 GB at batch 1 —
        # enough to spill into WDDM shared host RAM and stall the machine) and
        # seek latency (a seek decodes a whole batch just to show one frame:
        # 4.3 s at batch 8 vs 0.2 s at batch 1). Preview is one-frame-per-tick,
        # so batching buys nothing here.
        tracks = [
            TrackSpec(
                PyAvVideoDecoder(p, device=_DEVICE, batch_size=1),
                offset=off,
                transform=resize,
            )
            for p, off in zip(paths, offsets)
        ]
        return VideoGroupDecoder(tracks, global_fps=_FPS)

    def _convert(self, tick: GroupFrame) -> list[np.ndarray | None]:
        return [
            _to_numpy(f) if valid else None
            for f, valid in zip(tick.frames, tick.valid)
        ]

    @Slot()
    def run(self) -> None:
        """Emit the latest decoded frame on a steady _FPS grid.

        Decode happens on ``_decode_loop`` (a separate thread) so a slow
        ``next_tick`` never stalls emission: when decode can't keep up the same
        frame is re-emitted, holding the display at _FPS.
        """
        self._stop = False
        decoder = Thread(target=self._decode_loop, daemon=True)
        decoder.start()
        last = None
        try:
            while not self._stop:
                cur = self._latest
                # Emit only when the decode thread produced a *new* tick (a fresh
                # tuple). The old loop re-emitted the held frame on a 30 fps grid
                # regardless of decode progress; on 4K, decode manages only a few
                # ticks/s, so that flooded the GUI thread (QueuedConnection) with
                # ~30 identical frame events/s. The backlog grows unbounded ->
                # climbing RAM and a frozen UI. Re-drawing the same pixmap buys
                # nothing: the QLabel already shows it. Emit rate now tracks
                # decode rate, so a slow decoder degrades motion instead of
                # drowning the event loop.
                if self._playing and cur is not None and cur is not last:
                    self.frames_ready.emit(cur[0], cur[1], self._seek_gen)
                    last = cur
                QThread.msleep(1)
        finally:
            self._stop = True
            decoder.join()

    def _decode_loop(self) -> None:
        group: VideoGroupDecoder | None = None
        offsets: list[float] = []
        t0 = 0.0
        base_idx = 0  # group tick index playback started from

        try:
            while not self._stop:
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
                        self._playing = False
                        self._latest = None

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
                                self._latest = (self._convert(tick), tick.pts)
                                self.frames_ready.emit(*self._latest, self._seek_gen)
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
                        self._playing = True
                        t0, base_idx = time.perf_counter(), group.position if group else 0

                    elif op == "pause":
                        self._playing = False

                    elif op == "stop":
                        self._stop = True
                        break

                if self._stop:
                    break

                if not self._playing or group is None:
                    try:
                        self._cmds.put(self._cmds.get(timeout=0.05))
                    except queue.Empty:
                        pass
                    continue

                # Decode toward the wall-clock tick, one step per loop, never
                # ahead of the clock. When decode can't hit real-time (heavy 4K),
                # position simply trails disp_tick and playback runs in slow
                # motion — no "loading" overlay: frames are advancing, the
                # display just isn't real-time. The overlay is reserved for true
                # blocking (a seek), emitted around seek_to_index above.
                now = time.perf_counter()
                disp_tick = base_idx + round((now - t0) * _FPS)
                if disp_tick >= group.num_steps and group.position >= group.num_steps:
                    self._playing = False
                    self.eof_reached.emit()
                    continue

                if group.position <= disp_tick and group.position < group.num_steps:
                    tick = group.next_tick()
                    if tick is not None:
                        self._latest = (self._convert(tick), tick.pts)
                else:
                    QThread.msleep(1)
        except Exception:
            logger.exception("VideoGroupWorker decode loop crashed")
        finally:
            self._stop = True
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
