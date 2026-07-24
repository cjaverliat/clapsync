"""A persistent indeterminate progress dialog for silent startup stages.

Startup has stretches where the process is busy but nothing is on screen: the
CUDA/torch warm-up before the file picker, and probing inputs plus loading the
decode stack after selection. Each can take several seconds, so the window looks
hung. ``startup_progress`` puts up a busy dialog with a per-stage status line
across such a stretch.

Deliberately light — imports only PySide6 — so it can wrap the very torch/
framepipe imports it reports on without pulling them in first.

The dialog is indeterminate (range 0..0, marquee) and has no cancel button:
these stages run synchronously on the main thread and can't be interrupted
mid-call. Because they block the main thread, the marquee freezes during a
stage; the status text (updated before each stage, painted via processEvents) is
what tells the user what is happening.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication


@contextmanager
def startup_progress(
    label: str = "Preparing…", parent=None
) -> Iterator[Callable[[str], None]]:
    """Show a busy dialog for the duration of the ``with`` block.

    Yields a ``status(text)`` callable that updates the dialog's label and
    repaints. The dialog is dismissed on exit (including on exception).
    """
    dialog = QProgressDialog(label, "", 0, 0, parent)  # min==max==0 → marquee
    dialog.setWindowTitle("clapsync")
    dialog.setCancelButton(None)  # startup stages aren't interruptible
    dialog.setMinimumWidth(420)
    dialog.setMinimumDuration(0)  # show immediately, don't wait the default 4s
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setValue(0)
    dialog.show()
    QApplication.processEvents()

    def status(text: str) -> None:
        dialog.setLabelText(text)
        QApplication.processEvents()

    try:
        yield status
    finally:
        dialog.close()
        QApplication.processEvents()
