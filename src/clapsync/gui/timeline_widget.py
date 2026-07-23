from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QEvent, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetricsF, QPainter, QPalette, QPen, QBrush, QPainterPath,
    QPixmap, QPolygonF,
)
from PySide6.QtWidgets import QInputDialog, QScrollBar, QWidget

from clapsync.gui.waveform_widget import downsample_peaks

HEADER_H = 28
TRACK_H = 40
TRACK_PAD = 6
SCROLLBAR_H = 14  # horizontal scrollbar height
SCROLLBAR_W = 14  # vertical scrollbar width
HANDLE_W = 10
MIN_ZOOM = 10.0    # px/s
MAX_ZOOM = 2000.0  # px/s
_SCROLL_MARGIN = 10  # pixels of padding before time=0

TRACK_COLORS = [
    QColor("#4A90D9"),
    QColor("#E67E22"),
    QColor("#2ECC71"),
    QColor("#9B59B6"),
    QColor("#E74C3C"),
    QColor("#1ABC9C"),
    QColor("#F39C12"),
]

LOCKED_TRACK_COLOR = QColor("#999999")
PLAYHEAD_COLOR = QColor("#D93025")
TRACK_BORDER_ALPHA = 160

_THEME_LIGHT = {
    "ruler_bg":       QColor("#E8E8E8"),
    "track_area_bg":  QColor("#F5F5F5"),
    "ruler_text":     QColor("#444444"),
    "ruler_tick":     QColor("#888888"),
    "ruler_border":   QColor("#CCCCCC"),
    "handle":         QColor("#333333"),
    "trim_overlay":   QColor(0, 0, 0, 50),
}

_THEME_DARK = {
    "ruler_bg":       QColor("#2A2A2A"),
    "track_area_bg":  QColor("#1E1E1E"),
    "ruler_text":     QColor("#CCCCCC"),
    "ruler_tick":     QColor("#666666"),
    "ruler_border":   QColor("#444444"),
    "handle":         QColor("#DDDDDD"),
    "trim_overlay":   QColor(0, 0, 0, 80),
}

def _scrollbar_qss(
    orient: str, bg: str, border: str, handle: str, hover: str
) -> str:
    """QScrollBar stylesheet for one orientation and palette."""
    side = "top" if orient == "horizontal" else "left"
    size = "height" if orient == "horizontal" else "width"
    min_size = "min-width" if orient == "horizontal" else "min-height"
    return f"""
QScrollBar:{orient} {{
    background: {bg};
    {size}: 14px;
    margin: 0;
    border: none;
    border-{side}: 1px solid {border};
}}
QScrollBar::handle:{orient} {{
    background: {handle};
    {min_size}: 20px;
    border-radius: 3px;
    margin: 2px;
}}
QScrollBar::handle:{orient}:hover {{
    background: {hover};
}}
QScrollBar::add-line:{orient},
QScrollBar::sub-line:{orient} {{
    width: 0;
    height: 0;
}}
QScrollBar::add-page:{orient},
QScrollBar::sub-page:{orient} {{
    background: none;
}}
"""


_SCROLLBAR_STYLE_LIGHT = _scrollbar_qss(
    "horizontal", "#E8E8E8", "#CCCCCC", "#AAAAAA", "#888888"
)
_SCROLLBAR_STYLE_DARK = _scrollbar_qss(
    "horizontal", "#2A2A2A", "#444444", "#555555", "#777777"
)
_SCROLLBAR_VERT_STYLE_LIGHT = _scrollbar_qss(
    "vertical", "#F5F5F5", "#CCCCCC", "#AAAAAA", "#888888"
)
_SCROLLBAR_VERT_STYLE_DARK = _scrollbar_qss(
    "vertical", "#1E1E1E", "#444444", "#555555", "#777777"
)


@dataclass
class TrackState:
    index: int
    label: str
    offset_s: float
    duration_s: float
    locked: bool = False
    muted: bool = False
    kind: str = "video"  # "video" or "audio" — shown by the track panel
    warn: bool = False  # low-confidence alignment — flagged in the track panel

    @property
    def color(self) -> QColor:
        if self.locked:
            return LOCKED_TRACK_COLOR
        return TRACK_COLORS[self.index % len(TRACK_COLORS)]

    @property
    def end_s(self) -> float:
        return self.offset_s + self.duration_s


class ZoomScrollBar(QWidget):
    """NLE-style zoom/pan bar: a window over the whole timeline.

    Drag the window body to pan; drag either edge handle to resize it (zoom).
    Emits the visible [start, end] range in seconds.
    """

    _HANDLE_W = 7        # px grab zone at each edge
    _MIN_THUMB_W = 18    # px minimum window width

    range_changed = Signal(float, float)  # visible v0, v1 in seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(SCROLLBAR_H)
        self.setMouseTracking(True)
        self._total = 1.0
        self._v0 = 0.0
        self._v1 = 1.0
        self._drag: str | None = None
        self._drag_x = 0.0
        self._drag_v = (0.0, 0.0)

    def set_view(self, total: float, v0: float, v1: float) -> None:
        self._total = max(total, 1e-6)
        self._v0 = max(0.0, min(v0, self._total))
        self._v1 = max(self._v0, min(v1, self._total))
        self.update()

    def _x_of(self, s: float) -> float:
        return s / self._total * self.width()

    def _dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self._dark()
        painter.fillRect(self.rect(), QColor("#202020") if dark else QColor("#E0E0E0"))
        x0 = self._x_of(self._v0)
        x1 = max(x0 + self._MIN_THUMB_W, self._x_of(self._v1))
        thumb = QRectF(x0, 2, x1 - x0, self.height() - 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#5A5A5A") if dark else QColor("#B0B0B0"))
        painter.drawRoundedRect(thumb, 3, 3)
        grip = QColor("#888888") if dark else QColor("#808080")
        painter.setPen(QPen(grip, 2))
        for gx in (x0 + 3, x1 - 3):
            painter.drawLine(int(gx), int(thumb.top() + 3),
                             int(gx), int(thumb.bottom() - 3))
        painter.end()

    def _zone(self, x: float) -> str | None:
        x0 = self._x_of(self._v0)
        x1 = max(x0 + self._MIN_THUMB_W, self._x_of(self._v1))
        if abs(x - x0) <= self._HANDLE_W:
            return "left"
        if abs(x - x1) <= self._HANDLE_W:
            return "right"
        if x0 <= x <= x1:
            return "body"
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = self._zone(event.position().x())
        self._drag_x = event.position().x()
        self._drag_v = (self._v0, self._v1)

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._drag is None:
            zone = self._zone(x)
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if zone in ("left", "right")
                else Qt.CursorShape.OpenHandCursor if zone == "body"
                else Qt.CursorShape.ArrowCursor
            )
            return
        ds = (x - self._drag_x) / max(1, self.width()) * self._total
        v0, v1 = self._drag_v
        min_w = self._MIN_THUMB_W / max(1, self.width()) * self._total
        if self._drag == "body":
            width = v1 - v0
            v0 = max(0.0, min(self._total - width, v0 + ds))
            v1 = v0 + width
        elif self._drag == "left":
            v0 = max(0.0, min(v1 - min_w, v0 + ds))
        else:
            v1 = min(self._total, max(v0 + min_w, v1 + ds))
        self._v0, self._v1 = v0, v1
        self.range_changed.emit(v0, v1)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None


class TimelineWidget(QWidget):
    playhead_changed = Signal(float)  # user seek in global seconds
    vscroll_changed = Signal(int)     # vertical scroll offset in pixels

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(HEADER_H + TRACK_H + TRACK_PAD * 2 + SCROLLBAR_H + 20)

        self._tracks: list[TrackState] = []
        self._track_waveforms: dict[int, np.ndarray] = {}  # track index -> mono
        self._wave_cache: dict[int, tuple] = {}  # index -> (sig, x0, peaks)
        self._wave_version: int = 0  # bumped when the waveform set changes
        # (track index, shared seconds, is_selected, kind) kind: sound|movement
        self._clap_markers: list[tuple[int, float, bool, str]] = []
        self._playhead_s: float = 0.0
        # Everything except the playhead is static during playback: cache it to a
        # pixmap so a 60 Hz playhead move blits instead of re-drawing every
        # waveform. Rebuilt only when _static_signature changes (see paintEvent).
        self._static_pixmap: QPixmap | None = None
        self._static_sig: tuple | None = None
        self._zoom: float = 100.0   # px per second
        self._scroll_x: float = _SCROLL_MARGIN  # pixel x of time=0
        self._scroll_y: float = 0.0  # vertical scroll offset in pixels

        self._drag_mode: str | None = None
        self._drag_start_x: float = 0.0
        self._drag_start_value: float = 0.0


        self._hscroll = ZoomScrollBar(self)
        self._hscroll.range_changed.connect(self._on_view_range_changed)

        self._vscroll = QScrollBar(Qt.Orientation.Vertical, self)
        self._vscroll.setStyleSheet(self._vscrollbar_style())
        self._vscroll.setFixedWidth(SCROLLBAR_W)
        self._vscroll.setSingleStep(max(1, TRACK_H // 2))
        self._vscroll.setVisible(False)
        self._vscroll.valueChanged.connect(self._on_vscrollbar_changed)

    def _is_dark(self) -> bool:
        return self.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def _theme(self) -> dict:
        return _THEME_DARK if self._is_dark() else _THEME_LIGHT

    def _scrollbar_style(self) -> str:
        return _SCROLLBAR_STYLE_DARK if self._is_dark() else _SCROLLBAR_STYLE_LIGHT

    def _vscrollbar_style(self) -> str:
        return _SCROLLBAR_VERT_STYLE_DARK if self._is_dark() else _SCROLLBAR_VERT_STYLE_LIGHT

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._vscroll.setStyleSheet(self._vscrollbar_style())
        self.update()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._vscroll.setStyleSheet(self._vscrollbar_style())
            self.update()

    # ----------------------------------------------------------------- layout helpers

    def _content_w(self) -> int:
        return self.width() - (SCROLLBAR_W if self._vscroll.isVisible() else 0)

    def _track_area_h(self) -> int:
        return self.height() - SCROLLBAR_H - HEADER_H

    def _tracks_content_h(self) -> int:
        n = len(self._tracks)
        return TRACK_PAD + n * (TRACK_H + TRACK_PAD)

    def _needs_vscroll(self) -> bool:
        return self._tracks_content_h() > self._track_area_h()

    # ----------------------------------------------------------------- track management

    def set_tracks(self, tracks: list[TrackState]) -> None:
        self._tracks = tracks
        self._playhead_s = 0.0
        self._update_vscrollbar()
        self._fit_zoom()
        self.update()

    def set_playhead(self, global_s: float) -> None:
        self._playhead_s = global_s
        self.update()

    def get_offsets(self) -> list[float]:
        return [t.offset_s for t in self._tracks]

    # ------------------------------------------------------------------ paint

    def _static_signature(self) -> tuple:
        """Everything the cached static pixmap depends on — but NOT the playhead.

        When this is unchanged the pixmap is reused, so a playhead move only
        re-blits. Subclasses extend it (super() + their own state).
        """
        return (
            self.width(), self.height(),
            round(self._zoom, 4),
            round(self._scroll_x, 2), round(self._scroll_y, 2),
            self._is_dark(), self._vscroll.isVisible(),
            self._wave_version,
            tuple(
                (t.index, round(t.offset_s, 4), round(t.duration_s, 4),
                 t.locked, t.muted, t.kind, t.warn)
                for t in self._tracks
            ),
            tuple(self._clap_markers),
        )

    def _paint_static(self, painter: QPainter) -> None:
        """Draw every layer except the playhead (the cached content)."""
        t = self._theme()
        w = self._content_w()
        h = self.height() - SCROLLBAR_H  # don't paint over the scrollbar
        track_area_h = h - HEADER_H

        painter.fillRect(0, HEADER_H, w, track_area_h, t["track_area_bg"])
        painter.fillRect(0, 0, w, HEADER_H, t["ruler_bg"])
        painter.setPen(QPen(t["ruler_border"], 1))
        painter.drawLine(0, HEADER_H, w, HEADER_H)

        self._draw_ruler(painter, w, t)

        # Clip track area so scrolled tracks don't overdraw the ruler or scrollbar.
        painter.save()
        painter.setClipRect(0, HEADER_H, w, track_area_h)
        self._draw_tracks(painter, w, track_area_h)
        painter.restore()

        self._paint_extras(painter, w, h, t)

        painter.save()
        painter.setClipRect(0, HEADER_H, w, track_area_h)
        self._draw_clap_markers(painter)
        painter.restore()

    def _ensure_static_pixmap(self) -> None:
        sig = self._static_signature()
        dpr = self.devicePixelRatioF()
        want = (max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
        have = self._static_pixmap
        if (have is not None and self._static_sig == sig
                and (have.width(), have.height()) == want):
            return
        pixmap = QPixmap(want[0], want[1])
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_static(painter)
        painter.end()
        self._static_pixmap = pixmap
        self._static_sig = sig

    def paintEvent(self, event) -> None:
        self._ensure_static_pixmap()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._static_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_playhead(painter, self.height() - SCROLLBAR_H)
        painter.end()

    def set_clap_markers(
        self, markers: list[tuple[int, float, bool, str]]
    ) -> None:
        """Set clap markers: (track index, shared seconds, is_selected, kind).

        kind is "sound" or "movement". Selected markers form the active clap
        link and are drawn bright; the rest are dim proposals/candidates.
        """
        self._clap_markers = list(markers)
        self.update()

    def _draw_clap_markers(self, painter: QPainter) -> None:
        """Draw clap markers per lane: sound (amber) and movement (cyan).

        The selected link members are bright with a solid line; other
        candidates are dim and dashed. Selected drawn last, on top.
        """
        by_index = {track.index: track for track in self._tracks}
        palette = {
            ("sound", True): QColor("#FFC107"),
            ("sound", False): QColor("#7A6A3A"),
            ("movement", True): QColor("#4FC3F7"),
            ("movement", False): QColor("#3A5A6A"),
        }
        for index, shared_s, selected, kind in sorted(
            self._clap_markers, key=lambda m: m[2]
        ):
            track = by_index.get(index)
            if track is None:
                continue
            rect = self._track_rect(track)
            x = self._s_to_x(shared_s)
            color = palette[(kind, selected)]
            style = Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine
            painter.setPen(QPen(color, 2 if selected else 1, style))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            size = 8 if selected else 6
            # Sound flags point down from the top; movement flags point up.
            if kind == "movement":
                flag = QPolygonF([
                    QPointF(x, rect.bottom()),
                    QPointF(x + size, rect.bottom()),
                    QPointF(x, rect.bottom() - size),
                ])
            else:
                flag = QPolygonF([
                    QPointF(x, rect.top()),
                    QPointF(x + size, rect.top()),
                    QPointF(x, rect.top() + size),
                ])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(flag)

    def _paint_extras(self, painter: QPainter, w: int, h: int, t: dict) -> None:
        """Hook for subclasses to paint additional overlays before the playhead."""

    def _draw_ruler(self, painter: QPainter, w: int, t: dict) -> None:
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)

        interval = self._tick_interval()
        total = self._total_duration()
        ts = 0.0
        while ts <= total + interval:
            x = self._s_to_x(ts)
            if 0 <= x <= w:
                painter.setPen(QPen(t["ruler_tick"], 1))
                painter.drawLine(int(x), HEADER_H - 8, int(x), HEADER_H)
                painter.setPen(QPen(t["ruler_text"], 1))
                label = self._format_time(ts)
                painter.drawText(int(x) + 2, HEADER_H - 10, label)
            ts += interval

    def _draw_lock_icon(self, painter: QPainter, cx: float, cy: float, size: float) -> None:
        """Draw a simple padlock centred at (cx, cy) using the given pixel size."""
        painter.save()

        bw = size * 0.72
        bh = size * 0.52
        bx = cx - bw / 2
        by = cy - bh / 2 + size * 0.10
        body = QRectF(bx, by, bw, bh)

        sw = size * 0.42
        sh = size * 0.46
        sx = cx - sw / 2
        sy = cy - sh / 2 - size * 0.18
        shackle = QRectF(sx, sy, sw, sh)

        pen = QPen(QColor("white"), max(1.2, size * 0.14))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(shackle, 0 * 16, 180 * 16)

        painter.setBrush(QBrush(QColor("white")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(body, 2, 2)

        painter.restore()

    def _draw_tracks(self, painter: QPainter, w: int, track_area_h: int) -> None:
        # Track identity (name, kind, mute) lives in the fixed TrackHeaderPanel
        # to the left; the bars carry only the clip block and a lock badge, so
        # they stay readable while scrolling.
        for i, track in enumerate(self._tracks):
            rect = self._track_rect(track)

            if rect.bottom() < HEADER_H or rect.top() > HEADER_H + track_area_h:
                continue
            if rect.right() < 0 or rect.left() > w:
                continue

            color = track.color
            painter.fillRect(rect, color)

            self._draw_track_waveform(painter, track, rect, w)

            darker = color.darker(140)
            painter.setPen(QPen(darker, 1))
            painter.drawRect(rect)

            if track.locked:
                icon_size = TRACK_H * 0.38
                icon_cx = rect.left() + icon_size * 0.5 + 6.0
                icon_cy = rect.center().y()
                self._draw_lock_icon(painter, icon_cx, icon_cy, icon_size)

    def set_track_waveforms(self, waveforms: dict[int, np.ndarray]) -> None:
        """Provide mono samples to render inside each track's clip bar."""
        self._track_waveforms = dict(waveforms)
        self._wave_cache.clear()
        self._wave_version += 1  # invalidate the static pixmap
        self.update()

    def _draw_track_waveform(
        self, painter: QPainter, track: TrackState, rect: QRectF, w: int
    ) -> None:
        """Draw the track's waveform inside its clip bar, slightly darker.

        Only the visible span is peak-reduced, cached against zoom/scroll so
        the per-playhead repaints stay cheap.
        """
        wave = self._track_waveforms.get(track.index)
        if wave is None or wave.size == 0 or rect.width() < 2:
            return
        x0 = max(int(rect.left()), 0)
        x1 = min(int(math.ceil(rect.right())), w)
        if x1 <= x0:
            return

        sig = (round(self._zoom, 4), round(self._scroll_x, 1), x0, x1)
        cached = self._wave_cache.get(track.index)
        if cached is not None and cached[0] == sig:
            peaks = cached[1]
        else:
            span = rect.width()
            n = wave.size
            lo = max(0, int((x0 - rect.left()) / span * n))
            hi = min(n, int((x1 - rect.left()) / span * n))
            peaks = downsample_peaks(wave[lo:hi], x1 - x0)
            self._wave_cache[track.index] = (sig, peaks)

        wave_color = track.color.darker(145)
        painter.setPen(QPen(wave_color, 1))
        mid = rect.center().y()
        half = rect.height() * 0.45
        for k in range(peaks.shape[0]):
            ymax = mid - float(peaks[k, 1]) * half
            ymin = mid - float(peaks[k, 0]) * half
            painter.drawLine(x0 + k, int(ymax), x0 + k, int(ymin))

    def _draw_playhead(self, painter: QPainter, h: int) -> None:
        x = self._s_to_x(self._playhead_s)
        pen = QPen(PLAYHEAD_COLOR, 2)
        painter.setPen(pen)
        painter.drawLine(int(x), 0, int(x), h)

        tri_size = 8
        path = QPainterPath()
        path.moveTo(x - tri_size / 2, 0)
        path.lineTo(x + tri_size / 2, 0)
        path.lineTo(x, tri_size)
        path.closeSubpath()
        painter.setBrush(QBrush(PLAYHEAD_COLOR))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

    # ---------------------------------------------------------- mouse events

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x = float(event.position().x())
        t = self._x_to_s(x)

        self._drag_mode = "playhead"
        self._playhead_s = max(0.0, t)
        self.playhead_changed.emit(self._playhead_s)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode is None:
            return

        x = float(event.position().x())

        if self._drag_mode == "playhead":
            t = self._x_to_s(x)
            self._playhead_s = max(0.0, t)
            self.playhead_changed.emit(self._playhead_s)

        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = None

    def wheelEvent(self, event) -> None:
        x = float(event.position().x())
        t_at_cursor = self._x_to_s(x)
        delta = event.angleDelta().y()

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
            self._scroll_x = x - t_at_cursor * new_zoom
            self._zoom = new_zoom
        else:
            self._scroll_x += delta * 0.5

        self._update_scrollbar()
        self.update()

    def zoom_by(self, factor: float, anchor_x: float | None = None) -> None:
        """Scale the horizontal zoom about a pixel anchor (viewport center)."""
        if anchor_x is None:
            anchor_x = self._content_w() / 2.0
        t = self._x_to_s(anchor_x)
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        self._scroll_x = anchor_x - t * new_zoom
        self._zoom = new_zoom
        self._update_scrollbar()
        self.update()

    def zoom_to_fit(self) -> None:
        """Fit the whole timeline in the viewport."""
        self._fit_zoom()
        self.update()

    # ---------------------------------------------------------------- helpers

    def _s_to_x(self, seconds: float) -> float:
        return self._scroll_x + seconds * self._zoom

    def _x_to_s(self, x: float) -> float:
        return (x - self._scroll_x) / self._zoom

    def _track_rect(self, track: TrackState) -> QRectF:
        i = track.index
        y = HEADER_H + TRACK_PAD + i * (TRACK_H + TRACK_PAD) - self._scroll_y
        x = self._s_to_x(track.offset_s)
        w = track.duration_s * self._zoom
        return QRectF(x, y, w, TRACK_H)

    def _total_duration(self) -> float:
        if not self._tracks:
            return 10.0
        return max(t.end_s for t in self._tracks)

    def _fit_zoom(self) -> None:
        total = self._total_duration()
        available = max(1, self._content_w() - _SCROLL_MARGIN * 2)
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, available / total))
        self._scroll_x = float(_SCROLL_MARGIN)
        self._update_scrollbar()

    def _update_scrollbar(self) -> None:
        if self.width() == 0:
            return
        total = self._total_duration()
        viewport_w = self._content_w()
        self._hscroll.set_view(total, self._x_to_s(0), self._x_to_s(viewport_w))

    def _update_vscrollbar(self) -> None:
        needs = self._needs_vscroll()
        self._vscroll.setVisible(needs)
        if needs:
            track_area_h = self._track_area_h()
            self._vscroll.setGeometry(self._content_w(), HEADER_H, SCROLLBAR_W, track_area_h)
            content_h = self._tracks_content_h()
            max_val = max(0, content_h - track_area_h)
            self._scroll_y = min(self._scroll_y, float(max_val))
            self._vscroll.blockSignals(True)
            self._vscroll.setRange(0, max_val)
            self._vscroll.setPageStep(track_area_h)
            self._vscroll.setSingleStep(max(1, TRACK_H // 2))
            self._vscroll.setValue(int(self._scroll_y))
            self._vscroll.blockSignals(False)
        else:
            self._scroll_y = 0.0

    def _on_view_range_changed(self, v0: float, v1: float) -> None:
        """Apply a visible range from the zoom scrollbar as zoom + scroll."""
        viewport_w = self._content_w()
        width_s = max(1e-6, v1 - v0)
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, viewport_w / width_s))
        self._scroll_x = -v0 * self._zoom
        self.update()
        self._update_scrollbar()  # reflect any zoom clamp back onto the thumb

    def _on_vscrollbar_changed(self, value: int) -> None:
        self._scroll_y = float(value)
        self.vscroll_changed.emit(value)
        self.update()

    def _tick_interval(self) -> float:
        candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
        min_px = 60.0
        for c in candidates:
            if c * self._zoom >= min_px:
                return c
        return candidates[-1]

    def _format_time(self, s: float) -> str:
        m = int(s) // 60
        sec = s - m * 60
        return f"{m}:{sec:05.2f}"

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        self._update_vscrollbar()
        cw = self._content_w()
        self._hscroll.setGeometry(0, h - SCROLLBAR_H, cw, SCROLLBAR_H)
        if self._vscroll.isVisible():
            track_area_h = self._track_area_h()
            self._vscroll.setGeometry(cw, HEADER_H, SCROLLBAR_W, track_area_h)
        if self._tracks:
            self._fit_zoom()
        else:
            self._update_scrollbar()


class SyncTrimTimelineWidget(TimelineWidget):
    """TimelineWidget extended with trim handles and per-track offset editing."""

    offsets_changed = Signal(list)       # list[float]
    trim_changed = Signal(float, float)  # trim_start, trim_end

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trim_start_s: float = 0.0
        self._trim_end_s: float = 0.0

    # ----------------------------------------------------------------- track management

    def set_tracks(self, tracks: list[TrackState]) -> None:
        super().set_tracks(tracks)
        total = max((t.end_s for t in tracks), default=10.0)
        # The default trim covers the audio/video overlap only; a c3d is placed
        # by its clap link and may move, so it should not shrink the window.
        av = [t for t in tracks if t.kind != "mocap"] or tracks
        shared_start = max((t.offset_s for t in av), default=0.0)
        shared_end   = min((t.end_s   for t in av), default=total)
        if shared_start < shared_end:
            self._trim_start_s = shared_start
            self._trim_end_s   = shared_end
        else:
            self._trim_start_s = 0.0
            self._trim_end_s   = total
        self._playhead_s = self._trim_start_s
        self.update()

    def _static_signature(self) -> tuple:
        # The trim overlay/handles live in the cached static layer, so a trim
        # change must invalidate the pixmap too.
        return super()._static_signature() + (
            round(self._trim_start_s, 4), round(self._trim_end_s, 4),
        )

    def get_trim(self) -> tuple[float, float]:
        return self._trim_start_s, self._trim_end_s

    def set_trim(self, start: float, end: float) -> None:
        """Reset the trim window (e.g. after re-anchoring changes the overlap)."""
        self._trim_start_s = start
        self._trim_end_s = end
        self.update()

    # ------------------------------------------------------------------ paint

    def _paint_extras(self, painter: QPainter, w: int, h: int, t: dict) -> None:
        self._draw_trim_overlay(painter, w, h, t)
        self._draw_trim_handles(painter, h, t)

    def _draw_trim_overlay(self, painter: QPainter, w: int, h: int, t: dict) -> None:
        trim_start_x = int(self._s_to_x(self._trim_start_s))
        trim_end_x = int(self._s_to_x(self._trim_end_s))

        if trim_start_x > 0:
            painter.fillRect(0, HEADER_H, trim_start_x, h - HEADER_H, t["trim_overlay"])
        if trim_end_x < w:
            painter.fillRect(trim_end_x, HEADER_H, w - trim_end_x, h - HEADER_H, t["trim_overlay"])

    def _draw_trim_handles(self, painter: QPainter, h: int, t: dict) -> None:
        handle_color = t["handle"]
        for t_s, side in ((self._trim_start_s, "start"), (self._trim_end_s, "end")):
            x = self._s_to_x(t_s)
            painter.setPen(QPen(handle_color, 2))
            painter.drawLine(int(x), HEADER_H, int(x), h)

            grab_w = 12
            grab_h = 20
            if side == "start":
                grab_rect = QRectF(x, HEADER_H, grab_w, grab_h)
            else:
                grab_rect = QRectF(x - grab_w, HEADER_H, grab_w, grab_h)

            painter.setBrush(QBrush(handle_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(grab_rect, 3, 3)

    # ---------------------------------------------------------- mouse events

    def set_offsets(self, offsets: list[float]) -> None:
        """Reposition track lanes to new offsets (e.g. after re-anchoring)."""
        for track in self._tracks:
            if track.index < len(offsets):
                track.offset_s = offsets[track.index]
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x = float(event.position().x())

        if abs(x - self._s_to_x(self._trim_start_s)) <= HANDLE_W:
            self._drag_mode = "trim_start"
            self._drag_start_x = x
            self._drag_start_value = self._trim_start_s
            return

        if abs(x - self._s_to_x(self._trim_end_s)) <= HANDLE_W:
            self._drag_mode = "trim_end"
            self._drag_start_x = x
            self._drag_start_value = self._trim_end_s
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode not in ("trim_start", "trim_end"):
            super().mouseMoveEvent(event)
            return

        x = float(event.position().x())
        delta_s = (x - self._drag_start_x) / self._zoom

        if self._drag_mode == "trim_start":
            new_val = max(0.0, min(self._trim_end_s - 0.001, self._drag_start_value + delta_s))
            self._trim_start_s = new_val
            self.trim_changed.emit(self._trim_start_s, self._trim_end_s)

        elif self._drag_mode == "trim_end":
            total = self._total_duration()
            new_val = max(self._trim_start_s + 0.001, min(total, self._drag_start_value + delta_s))
            self._trim_end_s = new_val
            self.trim_changed.emit(self._trim_start_s, self._trim_end_s)

        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        x = float(event.position().x())
        y = float(event.position().y())
        for track in self._tracks:
            if track.locked:
                continue
            rect = self._track_rect(track)
            if rect.contains(QPointF(x, y)):
                value, ok = QInputDialog.getDouble(
                    self,
                    "Set Offset",
                    f"Offset for '{track.label}' (seconds):",
                    track.offset_s,
                    -100000.0,
                    100000.0,
                    3,
                )
                if ok:
                    track.offset_s = value
                    self._clamp_trim_to_tracks()
                    self.offsets_changed.emit(self.get_offsets())
                    self._update_scrollbar()
                    self.update()
                return

    # ---------------------------------------------------------------- helpers

    def _clamp_trim_to_tracks(self) -> None:
        total = self._total_duration()
        self._trim_end_s = min(self._trim_end_s, total)
        self._trim_start_s = min(self._trim_start_s, self._trim_end_s - 0.001)
        self._trim_start_s = max(0.0, self._trim_start_s)
