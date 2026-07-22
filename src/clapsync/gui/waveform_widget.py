"""Static waveform display for an audio track's mosaic cell."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


def downsample_peaks(samples: np.ndarray, buckets: int) -> np.ndarray:
    """Reduce a mono waveform to per-pixel (min, max) peak pairs.

    Returns an array of shape (buckets, 2). Buckets beyond the sample count
    (or an empty input) are zero. Cheap enough to recompute on resize.
    """
    out = np.zeros((buckets, 2), dtype=np.float32)
    n = samples.shape[0]
    if n == 0 or buckets <= 0:
        return out
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    for i in range(buckets):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg = samples[lo:hi]
        out[i, 0] = float(seg.min())
        out[i, 1] = float(seg.max())
    return out


class WaveformWidget(QWidget):
    """Paints a static waveform with a moving playhead. Display only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 90)
        self.setStyleSheet("background-color: #111;")
        self._samples: np.ndarray = np.zeros(0, dtype=np.float32)
        self._peaks: np.ndarray | None = None
        self._off = 0.0
        self._dur = 0.0
        self._trim0 = 0.0
        self._trim1 = 0.0
        self._playhead = 0.0

    def set_waveform(self, samples: np.ndarray) -> None:
        self._samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._peaks = None
        self.update()

    def set_window(self, offset_s: float, duration_s: float,
                   trim_start: float, trim_end: float) -> None:
        self._off, self._dur = offset_s, duration_s
        self._trim0, self._trim1 = trim_start, trim_end
        self.update()

    def set_playhead(self, global_s: float) -> None:
        self._playhead = global_s
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._peaks = None  # re-bucket to the new width

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        mid = h / 2
        if self._peaks is None or self._peaks.shape[0] != w:
            self._peaks = downsample_peaks(self._samples, max(1, w))
        p.setPen(QPen(QColor("#4A90D9"), 1))
        for x in range(self._peaks.shape[0]):
            ymin = mid - self._peaks[x, 1] * mid
            ymax = mid - self._peaks[x, 0] * mid
            p.drawLine(x, int(ymin), x, int(ymax))
        # Playhead within the track's own [offset, offset+duration] span,
        # mapped across the full widget width.
        if self._dur > 0:
            frac = (self._playhead - self._off) / self._dur
            if 0.0 <= frac <= 1.0:
                px = int(frac * w)
                p.setPen(QPen(QColor("#D93025"), 2))
                p.drawLine(px, 0, px, h)
        p.end()
