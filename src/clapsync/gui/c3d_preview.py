"""Animated 3D preview of c3d markers, driven by the timeline playhead.

Rendered with QPainter using an orthographic turntable projection (world +Z is
up). This is dependency-free and works on any hardware — no GL context, which
the mixed installer target (RDP, no-GPU machines) cannot be assumed to provide.
The marker count in a clapperboard capture is small, so 2D projection is
smooth without GPU acceleration.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_TOP_COLOR = QColor("#4FC3F7")
_BOTTOM_COLOR = QColor("#FF8A65")
_OTHER_COLOR = QColor("#BBBBBB")
_MARKER_RADIUS = 3.0


class C3DMarkerPreviewWidget(QWidget):
    """Draw one frame of marker positions; rotatable by dragging."""

    def __init__(self, points: np.ndarray, valid: np.ndarray,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points = points.astype(np.float64)
        self._valid = valid
        self._frame = 0
        self._top: list[int] = []
        self._bottom: list[int] = []
        self._azimuth = math.radians(30.0)
        self._elevation = math.radians(20.0)
        self._last_drag: QPointF | None = None
        self.setMinimumSize(160, 90)
        self.setStyleSheet("background-color: #101010;")
        self._center, self._extent = self._compute_bounds()

    def set_groups(self, top: list[int], bottom: list[int]) -> None:
        """Colorize the top- and bottom-arm marker groups."""
        self._top = top
        self._bottom = bottom
        self.update()

    def set_frame(self, frame: int) -> None:
        frame = max(0, min(self._points.shape[0] - 1, frame))
        if frame != self._frame:
            self._frame = frame
            self.update()

    def _compute_bounds(self) -> tuple[np.ndarray, float]:
        """Center and extent over all valid markers, for a stable camera."""
        pts = self._points[self._valid]
        if pts.size == 0:
            return np.zeros(3), 1.0
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.max(hi - lo)) or 1.0
        return center, extent

    def _project(self, p: np.ndarray) -> tuple[float, float]:
        """Orthographic turntable projection; +Z up."""
        x, y, z = p - self._center
        ca, sa = math.cos(self._azimuth), math.sin(self._azimuth)
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        horizontal = rx
        vertical = z * math.cos(self._elevation) + ry * math.sin(self._elevation)
        return horizontal, vertical

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101010"))

        w, h = self.width(), self.height()
        scale = 0.4 * min(w, h) / self._extent
        cx, cy = w / 2.0, h / 2.0
        frame_pts = self._points[self._frame]
        frame_valid = self._valid[self._frame]

        def screen(idx: int) -> QPointF:
            hx, vy = self._project(frame_pts[idx])
            return QPointF(cx + hx * scale, cy - vy * scale)

        # Connect the two arm centroids so the clap gap is visible.
        top_c = self._group_centroid(frame_pts, frame_valid, self._top)
        bottom_c = self._group_centroid(frame_pts, frame_valid, self._bottom)
        if top_c is not None and bottom_c is not None:
            painter.setPen(QPen(QColor("#666666"), 1))
            painter.drawLine(self._screen_point(top_c, scale, cx, cy),
                             self._screen_point(bottom_c, scale, cx, cy))

        for idx in range(frame_pts.shape[0]):
            if not frame_valid[idx]:
                continue
            color = _OTHER_COLOR
            if idx in self._top:
                color = _TOP_COLOR
            elif idx in self._bottom:
                color = _BOTTOM_COLOR
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(screen(idx), _MARKER_RADIUS, _MARKER_RADIUS)
        painter.end()

    def _group_centroid(self, frame_pts, frame_valid, group):
        if not group:
            return None
        valid = [i for i in group if frame_valid[i]]
        if not valid:
            return None
        return frame_pts[valid].mean(axis=0)

    def _screen_point(self, p, scale, cx, cy) -> QPointF:
        hx, vy = self._project(p)
        return QPointF(cx + hx * scale, cy - vy * scale)

    def mousePressEvent(self, event) -> None:
        self._last_drag = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._last_drag is None:
            return
        pos = event.position()
        self._azimuth += math.radians((pos.x() - self._last_drag.x()) * 0.5)
        self._elevation = max(
            math.radians(-89.0),
            min(math.radians(89.0),
                self._elevation + math.radians((pos.y() - self._last_drag.y()) * 0.5)),
        )
        self._last_drag = pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_drag = None
