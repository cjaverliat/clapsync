"""Fixed left panel showing per-track identity (name, kind, mute).

Mirrors the timeline's vertical track layout so each row lines up with its
clip bar, but stays put while the timeline scrolls horizontally. Vertical
scroll is kept in sync via the timeline's ``vscroll_changed`` signal.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from clapsync.gui import icons
from clapsync.gui.timeline_widget import (
    HEADER_H,
    TRACK_H,
    TRACK_PAD,
    TrackState,
)

PANEL_W = 190
_MUTE_W = 26


class TrackHeaderPanel(QWidget):
    """Stationary track-info column: name, kind icon, and a mute toggle."""

    mute_changed = Signal(int, bool)  # track index, muted

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(PANEL_W)
        self._tracks: list[TrackState] = []
        self._scroll_y = 0
        self._mute_hitboxes: dict[int, QRectF] = {}
        self._px_cache: dict[tuple[str, int, str], QPixmap] = {}

    def _px(self, name: str, size: int, color: str) -> QPixmap:
        """Cached icon pixmap — paintEvent runs often, icons are few."""
        key = (name, size, color)
        px = self._px_cache.get(key)
        if px is None:
            px = icons.pixmap(name, size, color)
            self._px_cache[key] = px
        return px

    def set_tracks(self, tracks: list[TrackState]) -> None:
        self._tracks = tracks
        self.update()

    def set_scroll_y(self, y: int) -> None:
        self._scroll_y = y
        self.update()

    def _row_rect(self, i: int) -> QRectF:
        y = HEADER_H + TRACK_PAD + i * (TRACK_H + TRACK_PAD) - self._scroll_y
        return QRectF(0, y, self.width(), TRACK_H)

    def _is_dark(self) -> bool:
        from PySide6.QtGui import QPalette
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        dark = self._is_dark()
        bg = QColor("#1E1E1E") if dark else QColor("#F5F5F5")
        header_bg = QColor("#2A2A2A") if dark else QColor("#E8E8E8")
        border = QColor("#444444") if dark else QColor("#CCCCCC")
        text_col = QColor("#DDDDDD") if dark else QColor("#333333")
        text_hex = text_col.name()

        painter.fillRect(0, 0, w, h, bg)
        painter.fillRect(0, 0, w, HEADER_H, header_bg)
        painter.setPen(QPen(border, 1))
        painter.drawLine(0, HEADER_H, w, HEADER_H)
        painter.drawLine(w - 1, 0, w - 1, h)  # divider against the timeline

        painter.setClipRect(0, HEADER_H, w, h - HEADER_H)
        self._mute_hitboxes.clear()
        for i, track in enumerate(self._tracks):
            rect = self._row_rect(i)
            if rect.bottom() < HEADER_H or rect.top() > h:
                continue

            # Colored swatch (matches the clip bar color).
            swatch = QRectF(rect.left() + 4, rect.top() + 4, 5, rect.height() - 8)
            painter.fillRect(swatch, track.color)

            # Kind icon.
            cy = rect.center().y()
            kind_name = (
                "mocap" if track.kind in ("mocap", "fbx")
                else "audio" if track.kind == "audio"
                else "video"
            )
            kind_px = self._px(kind_name, 14, text_hex)
            painter.drawPixmap(QPointF(rect.left() + 15, cy - 7), kind_px)

            # Name (elided), leaving room for the kind icon, an optional warning
            # marker, and the mute button.
            name_left = rect.left() + 32
            if track.warn:
                warn_px = self._px("warning", 14, "#E0A020")
                painter.drawPixmap(QPointF(name_left, cy - 7), warn_px)
                name_left += 18
            name_w = w - _MUTE_W - name_left - 4
            painter.setPen(QPen(text_col, 1))
            fm = QFontMetricsF(painter.font())
            name = fm.elidedText(
                track.label, Qt.TextElideMode.ElideRight, max(0.0, name_w)
            )
            painter.drawText(
                QRectF(name_left, rect.top(), name_w, rect.height()),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                name,
            )

            # Mute toggle.
            box = QRectF(w - _MUTE_W, cy - 9, 18, 18)
            mute_px = self._px(
                "mute" if track.muted else "unmute",
                16,
                "#888888" if track.muted else text_hex,
            )
            painter.drawPixmap(QPointF(box.left() + 1, cy - 8), mute_px)
            self._mute_hitboxes[track.index] = box

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        for idx, box in self._mute_hitboxes.items():
            if box.contains(pos):
                track = next(t for t in self._tracks if t.index == idx)
                track.muted = not track.muted
                self.mute_changed.emit(idx, track.muted)
                self.update()
                return
