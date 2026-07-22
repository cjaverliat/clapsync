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

from framepipe import (
    GroupFrame,
    NvVideoDecoder,
    PyAvVideoDecoder,
    TrackSpec,
    VideoGroupDecoder,
)

from clapsync.app.decode import ensure_preview_proxy

# Global playback frame rate; every track resamples to this grid.
_FPS = 30.0
# GPU-resident NVDEC decode when a CUDA device is present, else software decode.
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# NVDEC (PyNvVideoCodec) on GPU, PyAV on CPU. PyAV's CUDA hwaccel path yields
# intermittently corrupt frames on the mosaic, so the GPU path uses NVDEC via
# framepipe's NvVideoDecoder instead; NvVideoDecoder needs a CUDA device, so CPU
# hosts fall back to software PyAV decode.
_DECODER = NvVideoDecoder if _DEVICE.startswith("cuda") else PyAvVideoDecoder


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

    def __init__(self, use_proxies: bool = False) -> None:
        super().__init__()
        self._use_proxies: bool = use_proxies
        self._cmds: queue.Queue = queue.Queue()
        self._seek_gen: int = 0
        # Shared between the decode thread (writer) and the emit loop (reader).
        self._latest: tuple[list[np.ndarray | None], float] | None = None
        self._playing: bool = False
        self._stop: bool = False

    def cmd(self, *args: Any) -> None:
        """Thread-safe: enqueue a command tuple."""
        self._cmds.put(args)

    def _open(self, paths: list[Path], offsets: list[float]) -> VideoGroupDecoder:
        # With --use_proxies, play a 480p/30fps proxy rather than the source:
        # decoding raw 5.3K 60fps footage per track cannot hold real time across
        # a mosaic. ensure_preview_proxy transcodes+caches once (cache is warm
        # from the startup pre-pass); sources within the envelope pass through
        # unchanged. Without the flag, the source is decoded directly. The proxy
        # is already preview resolution, so no decode-side resize transform is
        # needed — the display widget does the single scale-to-fit.
        # batch_size=1: preview shows one frame per tick, and a larger batch only
        # inflates seek latency (a seek would decode a whole batch to show one
        # frame) for no benefit.
        tracks = [
            TrackSpec(
                _DECODER(
                    str(ensure_preview_proxy(Path(p)) if self._use_proxies else p),
                    device=_DEVICE,
                    batch_size=1,
                ),
                offset=off,
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
                        # First open of a high-res mosaic transcodes proxies,
                        # which can take minutes; hold the overlay until ready.
                        self.loading_changed.emit(True)
                        try:
                            group = self._open(paths, list(offsets))
                        except Exception:
                            logger.exception("VideoGroupDecoder open failed")
                            group = None
                        finally:
                            self.loading_changed.emit(False)
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


class _FrameLabel(QLabel):
    """Displays a frame scaled to fit, keeping aspect ratio, on every resize.

    Rescaling lives in the label's own ``resizeEvent`` so it reads the label's
    real current size (a parent widget's ``resizeEvent`` fires before the layout
    has resized its children, so it would scale to a stale size). An ``Ignored``
    size policy keeps the shown pixmap from feeding back into the layout as a
    minimum, so a mosaic cell can always shrink and the whole frame stays
    visible however small the window gets.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)

    def set_frame(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None:
            return
        if self.width() <= 0 or self.height() <= 0:
            return
        super().setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class VideoPlayerWidget(QWidget):
    """Pure display widget for one video track.

    Receives frames via ``display_frame()`` — all decoding is managed
    externally by a ``VideoGroupWorker``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._label = _FrameLabel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def display_frame(self, frame: np.ndarray | None) -> None:
        """Display a decoded frame (None = black). GUI thread only."""
        if frame is not None:
            self._display_frame(frame)
        else:
            self._show_black()

    def _show_black(self) -> None:
        self._display_frame(np.zeros((180, 320, 3), dtype=np.uint8))

    def _display_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        image = QImage(frame.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        # fromImage copies, so the frame buffer need not outlive this call.
        self._label.set_frame(QPixmap.fromImage(image))
