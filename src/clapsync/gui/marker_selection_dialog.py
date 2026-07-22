"""Manual clapperboard marker picker for c3d files not resolved by name."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QVBoxLayout,
    QWidget,
)


class MarkerSelectionDialog(QDialog):
    """Ask the user to pick the top- and bottom-arm clapperboard markers.

    Shown before offset computation (on the main thread) when a c3d file's
    markers cannot be classified by name, so the sync worker receives resolved
    marker groups and never has to raise a dialog itself.
    """

    def __init__(
        self, filename: str, labels: list[str], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select clapperboard markers")
        self._labels = labels

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            f"Clapperboard markers in <b>{filename}</b> could not be identified "
            f"automatically.\nSelect the markers on each arm (one or more each)."
        ))

        lists = QHBoxLayout()
        self._top = self._marker_list(labels, "Top arm")
        self._bottom = self._marker_list(labels, "Bottom arm")
        lists.addLayout(self._top[0])
        lists.addLayout(self._bottom[0])
        root.addLayout(lists)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._top[1].itemSelectionChanged.connect(self._update_ok)
        self._bottom[1].itemSelectionChanged.connect(self._update_ok)
        self._update_ok()

    def _marker_list(self, labels: list[str], title: str):
        column = QVBoxLayout()
        column.addWidget(QLabel(title))
        widget = QListWidget()
        widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        widget.addItems(labels)
        column.addWidget(widget)
        return column, widget

    def _selected_rows(self, widget: QListWidget) -> list[int]:
        return sorted(widget.row(item) for item in widget.selectedItems())

    def _update_ok(self) -> None:
        ok = bool(self._top[1].selectedItems()) and bool(
            self._bottom[1].selectedItems()
        )
        self._buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(ok)

    def get_selection(self) -> tuple[list[int], list[int]]:
        """Return (top_indices, bottom_indices) of the chosen markers."""
        return self._selected_rows(self._top[1]), self._selected_rows(
            self._bottom[1]
        )
