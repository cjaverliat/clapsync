"""The GUI bpy marshaller runs a callable on the main thread from any worker.

bpy (Blender API) aborts the process with an access violation if touched off the
main thread. make_main_thread_executor lets worker threads submit bpy work that
executes on the main thread while the main thread pumps events.
"""
from __future__ import annotations

import threading

import pytest
from PySide6.QtWidgets import QApplication

from clapsync.gui.bpy_bridge import make_main_thread_executor


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["test"])


def _run_in_worker(qapp, fn):
    """Run ``fn`` on a worker thread while the main thread pumps events."""
    box = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — capture to assert on it
            box["error"] = exc

    t = threading.Thread(target=work)
    t.start()
    while t.is_alive():
        qapp.processEvents()
    t.join()
    return box


def test_executor_runs_callable_on_main_thread(qapp):
    ex = make_main_thread_executor(qapp)
    main_tid = threading.get_ident()

    box = _run_in_worker(qapp, lambda: ex(lambda: threading.get_ident()))

    assert box.get("value") == main_tid


def test_executor_runs_inline_when_already_on_main(qapp):
    ex = make_main_thread_executor(qapp)
    assert ex(lambda: threading.get_ident()) == threading.get_ident()


def test_executor_propagates_exception_to_caller(qapp):
    ex = make_main_thread_executor(qapp)

    def boom():
        raise ValueError("boom")

    box = _run_in_worker(qapp, lambda: ex(boom))

    assert isinstance(box.get("error"), ValueError)
