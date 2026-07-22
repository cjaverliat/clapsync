"""Central icon definitions for the GUI.

Single source of truth so every button and painted control shares one icon
family (Material Design Icons via qtawesome), one default size, and a
theme-aware tint. Call sites use semantic names ("play", "mute") and never
touch a raw MDI id, keeping the icon set homogeneous and swappable in one
place.
"""
from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPalette, QPixmap
from PySide6.QtWidgets import QApplication

# Semantic name -> Material Design Icon id.
_MDI = {
    "play": "mdi6.play",
    "pause": "mdi6.pause",
    "export": "mdi6.tray-arrow-down",
    "add": "mdi6.plus",
    "remove": "mdi6.trash-can-outline",
    "move-up": "mdi6.arrow-up",
    "move-down": "mdi6.arrow-down",
    "browse": "mdi6.folder-open-outline",
    "mute": "mdi6.volume-off",
    "unmute": "mdi6.volume-high",
    "audio": "mdi6.music-note",
    "video": "mdi6.filmstrip",
    "warning": "mdi6.alert",
}


def theme_color() -> str:
    """Palette text color as a hex string, for tinting icons to the theme."""
    app = QApplication.instance()
    if app is not None:
        return app.palette().color(QPalette.ColorRole.WindowText).name()
    return "#333333"


def icon(name: str, color: str | None = None) -> QIcon:
    """Return a themed QIcon for a semantic name.

    Args:
        name: Semantic icon key (see the module's icon map).
        color: Optional hex override; defaults to the palette text color.

    Returns:
        A qtawesome QIcon.

    Raises:
        KeyError: If ``name`` is not a known icon.
    """
    return qta.icon(_MDI[name], color=color or theme_color())


def pixmap(name: str, size: int, color: str | None = None) -> QPixmap:
    """Return a square themed QPixmap, for custom-painted controls.

    Args:
        name: Semantic icon key (see the module's icon map).
        size: Edge length in device-independent pixels.
        color: Optional hex override; defaults to the palette text color.

    Returns:
        A rendered QPixmap of the icon.

    Raises:
        KeyError: If ``name`` is not a known icon.
    """
    return icon(name, color).pixmap(QSize(size, size))
