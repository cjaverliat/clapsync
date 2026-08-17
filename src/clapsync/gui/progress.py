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

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, TypeVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication

_T = TypeVar("_T")


def _busy_dialog(label: str, parent) -> QProgressDialog:
    """Build an indeterminate (marquee), cancel-less, modal busy dialog."""
    dialog = QProgressDialog(label, "", 0, 0, parent)  # min==max==0 → marquee
    dialog.setWindowTitle("clapsync")
    dialog.setCancelButton(None)  # startup stages aren't interruptible
    dialog.setMinimumWidth(420)
    dialog.setMinimumDuration(0)  # show immediately, don't wait the default 4s
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setValue(0)
    return dialog


@contextmanager
def startup_progress(
    label: str = "Preparing…", parent=None
) -> Iterator[Callable[[str], None]]:
    """Show a busy dialog for the duration of the ``with`` block.

    Yields a ``status(text)`` callable that updates the dialog's label and
    repaints. The dialog is dismissed on exit (including on exception).
    """
    dialog = _busy_dialog(label, parent)
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


def run_blocking_with_progress(
    work: Callable[[], _T], label: str = "Working…", parent=None
) -> _T:
    """Run ``work()`` on a background thread behind an animated busy dialog.

    Unlike ``startup_progress`` (whose stages block the main thread and freeze
    the marquee), the blocking call runs off the main thread while the main
    thread pumps events here, so the marquee keeps animating and the window
    stays responsive. Use for main-thread-safe work that would otherwise freeze
    the window — e.g. probing inputs.

    ``work`` must not touch Qt objects: it runs off the GUI thread.

    Returns:
        Whatever ``work()`` returns.

    Raises:
        Whatever ``work()`` raises.
    """
    dialog = _busy_dialog(label, parent)
    dialog.show()
    QApplication.processEvents()

    outcome: dict = {}

    def _runner() -> None:
        try:
            outcome["value"] = work()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    while thread.is_alive():
        QApplication.processEvents()
        thread.join(0.02)  # brief wait so the loop doesn't spin the CPU hot

    dialog.close()
    QApplication.processEvents()

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
