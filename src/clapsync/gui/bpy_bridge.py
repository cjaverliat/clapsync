"""Marshal bpy (Blender API) calls onto the Qt main thread.

bpy is main-thread-only: any ``bpy.ops.*`` call from a worker thread aborts the
process with a Windows access violation (0xC0000005) that bypasses
``sys.excepthook`` — no traceback, no crash dialog, the process just exits. The
GUI runs sync-offset computation and export on QThreads (to keep the UI
responsive), and both touch fbx files, so those bpy calls must be hopped back to
the main thread.

``app.fbx`` submits all its bpy work through a swappable executor;
``make_main_thread_executor`` builds the GUI's replacement. It runs inline when
already on the main thread and otherwise blocks the worker until the main thread
— which is pumping events under ``workers._run_with_progress`` — runs the work.

Single-consumer only: bpy work is serialised (one worker at a time), so the
runner reuses one slot for the pending callable/result. Do not drive two
concurrent workers through the same executor.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Slot

_T = TypeVar("_T")


class _MainThreadRunner(QObject):
    """Runs a stored callable when its ``_invoke`` slot fires on the main thread."""

    _fn: Callable[[], object]
    _result: object
    _error: BaseException | None

    @Slot()
    def _invoke(self) -> None:
        try:
            self._result = self._fn()
            self._error = None
        except BaseException as exc:  # noqa: BLE001 — marshalled back to caller
            self._result = None
            self._error = exc


def make_main_thread_executor(app) -> Callable[[Callable[[], _T]], _T]:
    """Build an executor that runs bpy work on ``app``'s main thread.

    Args:
        app: The QApplication (or any QObject) whose ``thread()`` is the GUI
            main thread.

    Returns:
        A callable ``executor(work)`` that returns ``work()``'s value, running
        it inline when called from the main thread and otherwise via a
        blocking-queued invocation on the main thread. Exceptions raised by
        ``work`` propagate to the caller.
    """
    runner = _MainThreadRunner()
    main_thread = app.thread()
    runner.moveToThread(main_thread)

    def executor(work: Callable[[], _T]) -> _T:
        if QThread.currentThread() is main_thread:
            return work()
        runner._fn = work
        QMetaObject.invokeMethod(
            runner, "_invoke", Qt.ConnectionType.BlockingQueuedConnection
        )
        if runner._error is not None:
            raise runner._error
        return runner._result  # type: ignore[return-value]

    return executor
