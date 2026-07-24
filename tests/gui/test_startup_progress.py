"""startup_progress shows an indeterminate, uncancellable dialog and dismisses it."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QProgressDialog

from clapsync.gui.progress import startup_progress


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["test"])


def _dialogs(qapp):
    return [w for w in qapp.topLevelWidgets() if isinstance(w, QProgressDialog)]


def test_dialog_is_indeterminate_and_updates_label(qapp):
    with startup_progress("Init…") as status:
        live = [d for d in _dialogs(qapp) if d.isVisible()]
        assert live, "a progress dialog should be visible inside the block"
        dialog = live[0]
        # min == max == 0 is Qt's indeterminate (marquee) mode.
        assert dialog.minimum() == 0 and dialog.maximum() == 0
        status("Next stage…")
        assert dialog.labelText() == "Next stage…"


def test_dialog_dismissed_on_exit(qapp):
    with startup_progress("Init…"):
        pass
    qapp.processEvents()
    assert not any(d.isVisible() for d in _dialogs(qapp))


def test_dialog_dismissed_on_exception(qapp):
    with pytest.raises(RuntimeError):
        with startup_progress("Init…"):
            raise RuntimeError("boom")
    qapp.processEvents()
    assert not any(d.isVisible() for d in _dialogs(qapp))
