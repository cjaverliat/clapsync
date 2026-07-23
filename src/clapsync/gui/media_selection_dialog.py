from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clapsync.gui import icons


class MediaSelectionDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("clapsync — Select Media")
        self.setMinimumSize(560, 380)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(
            QLabel("Select two or more media files (video or audio) to synchronize:")
        )

        list_row = QHBoxLayout()

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._update_buttons)
        list_row.addWidget(self._list, stretch=1)

        side_btns = QVBoxLayout()
        side_btns.setSpacing(4)

        self._add_btn = QPushButton(icons.icon("add"), "Add Media…")
        self._remove_btn = QPushButton(icons.icon("remove"), "Remove")
        self._up_btn = QPushButton(icons.icon("move-up"), "Move Up")
        self._down_btn = QPushButton(icons.icon("move-down"), "Move Down")

        for btn in (self._add_btn, self._remove_btn, self._up_btn, self._down_btn):
            side_btns.addWidget(btn)
        side_btns.addStretch()

        list_row.addLayout(side_btns)
        layout.addLayout(list_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._add_btn.clicked.connect(self._add_videos)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._up_btn.clicked.connect(self._move_up)
        self._down_btn.clicked.connect(self._move_down)

        self._update_buttons()

    def _add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Media Files",
            "",
            "Media Files (*.mp4 *.mov *.avi *.mkv *.mts *.m2ts *.webm *.flv *.wmv "
            "*.wav *.mp3 *.flac *.m4a *.aac *.c3d *.fbx);;All Files (*)",
        )
        existing = {self._list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self._list.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(p)
                item.setData(Qt.ItemDataRole.UserRole, p)
                self._list.addItem(item)
                existing.add(p)
        self._update_buttons()

    def _remove_selected(self) -> None:
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self._update_buttons()

    def _move_up(self) -> None:
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        if not rows or rows[0] == 0:
            return
        self._list.clearSelection()
        for row in rows:
            item = self._list.takeItem(row)
            self._list.insertItem(row - 1, item)
            item.setSelected(True)
            self._list.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
        self._update_buttons()

    def _move_down(self) -> None:
        rows = sorted((self._list.row(i) for i in self._list.selectedItems()), reverse=True)
        if not rows or rows[0] == self._list.count() - 1:
            return
        self._list.clearSelection()
        for row in rows:
            item = self._list.takeItem(row)
            self._list.insertItem(row + 1, item)
            item.setSelected(True)
            self._list.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
        self._update_buttons()

    def _update_buttons(self) -> None:
        n = self._list.count()
        selected = self._list.selectedItems()
        self._ok_btn.setEnabled(n >= 2)
        self._remove_btn.setEnabled(bool(selected))
        rows = [self._list.row(i) for i in selected]
        self._up_btn.setEnabled(bool(rows) and min(rows) > 0)
        self._down_btn.setEnabled(bool(rows) and max(rows) < n - 1)

    def get_media_paths(self) -> list[Path]:
        return [
            Path(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
        ]

    # Back-compat alias.
    get_video_paths = get_media_paths
