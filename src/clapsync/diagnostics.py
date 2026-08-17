"""Logging setup and crash diagnostics shared by the CLI and GUI entry points.

The core (log directory, handler wiring, ``sys.excepthook``) is stdlib-only and
imports nothing heavy, so it can run at the very top of ``main`` — before torch,
av or PySide6 load — and still capture an import-time crash. The startup
context block and Qt message handler pull those heavier deps lazily, each
guarded, so a missing optional package never breaks logging itself.
"""
from __future__ import annotations

import importlib.metadata
import logging
import logging.handlers
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_LOG_FILE = "clapsync.log"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def log_dir() -> Path:
    """Per-user directory for rotating log files.

    Mirrors the decode cache root (``%LOCALAPPDATA%\\clapsync`` on Windows,
    ``~/.cache/clapsync`` elsewhere) so every piece of app state lives under one
    place a user can find and attach to a bug report.
    """
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    directory = Path(base) / "clapsync" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(verbose: bool = False) -> Path:
    """Route logging to stderr and a rotating file; return the log file path.

    The file handler is the only record that survives the installed GUI, which
    launches with a hidden console whose stderr is discarded — so it always
    records DEBUG regardless of ``verbose``. The console handler mirrors it for
    dev runs. Idempotent: repeated calls reconfigure rather than stack handlers.

    Args:
        verbose: Emit DEBUG (not just INFO) to the console.
    """
    log_file = log_dir() / _LOG_FILE

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Drop handlers from a previous call (or logging.basicConfig) so a second
    # invocation doesn't double every line.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    file = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file.setLevel(logging.DEBUG)
    file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file)

    return log_file


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _ffmpeg_report() -> str:
    """Locate ffmpeg and read its version banner (first line), or say it's absent."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg: NOT FOUND on PATH"
    try:
        out = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        banner = out.stdout.splitlines()[0] if out.stdout else "(no output)"
    except (OSError, subprocess.SubprocessError) as exc:
        banner = f"(version probe failed: {exc})"
    return f"ffmpeg: {ffmpeg}\n  {banner}"


def _gpu_report() -> str:
    """Describe the CUDA device torch sees and the decoder that choice selects."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — torch import itself can explode
        return f"torch: import failed ({exc})"
    lines = [f"torch: {torch.__version__}"]
    try:
        if torch.cuda.is_available():
            lines.append(f"cuda: available, device={torch.cuda.get_device_name(0)}")
        else:
            lines.append("cuda: not available (CPU decode/encode)")
    except Exception as exc:  # noqa: BLE001 — a broken driver can raise here
        lines.append(f"cuda: probe failed ({exc})")
    override = os.environ.get("CONDA_OVERRIDE_CUDA")
    if override:
        lines.append(f"CONDA_OVERRIDE_CUDA={override}")
    return "\n".join(lines)


def log_startup_context() -> None:
    """Log versions, GPU and ffmpeg once at startup.

    Crash reports are overwhelmingly environment-specific (driver, codec,
    ffmpeg build). Capturing this up front means the log file alone is enough to
    reconstruct the machine, without asking the user to re-run anything. Call
    early, before the first dialog, so a crash during selection is still
    diagnosable; log the chosen inputs separately with ``log_inputs``.
    """
    logger.info("clapsync %s", _package_version("clapsync"))
    logger.info(
        "python %s | %s", sys.version.split()[0], platform.platform()
    )
    for name in ("av", "PySide6", "torchaudio", "qtawesome"):
        logger.info("%s: %s", name, _package_version(name))
    logger.info("%s", _gpu_report())
    logger.info("%s", _ffmpeg_report())


def log_inputs(video_paths: list[Path], media: list | None = None) -> None:
    """Probe and log each selected input (resolution/fps/duration/audio).

    A "wrong output" or decode-failure report is far easier to reconstruct with
    the exact source properties on record. A single bad input is logged and
    skipped rather than aborting.

    Args:
        media: Optional pre-probed MediaInfo (one per path, same order). When
            given it is logged directly instead of re-probing — the GUI probes
            once and reuses the result. When None each input is probed here
            (CLI path).
    """
    logger.info("inputs (%d):", len(video_paths))
    for i, path in enumerate(video_paths):
        try:
            if media is not None:
                info = media[i]
            else:
                from clapsync.app.media import probe

                info = probe(Path(path))
            logger.info(
                "  %s | kind=%s dur=%.3fs %sx%s @%sfps audio=%s",
                info.path.name,
                info.kind,
                info.duration,
                info.width,
                info.height,
                f"{info.fps:.3f}" if info.fps else "?",
                info.has_audio,
            )
        except Exception as exc:  # noqa: BLE001 — a bad input must not abort logging
            logger.warning("  %s | probe failed: %s", Path(path).name, exc)


def install_excepthook(on_exception: Callable[[BaseException], None] | None = None) -> None:
    """Log any uncaught exception before the interpreter dies.

    Without this, an unhandled exception on the main thread of the installed
    GUI prints a traceback to a discarded stderr and vanishes. KeyboardInterrupt
    is passed through untouched so Ctrl+C still exits cleanly.

    Args:
        on_exception: Optional hook (e.g. show a dialog) run after logging.
    """
    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        if on_exception is not None:
            try:
                on_exception(exc)
            except Exception:  # noqa: BLE001 — the handler must never re-raise
                logger.exception("on_exception handler failed")

    sys.excepthook = hook


def install_qt_message_handler() -> None:
    """Route Qt's own warnings/errors into the Python log.

    Qt logs (failed paints, missing plugins, layout warnings) otherwise print
    straight to the discarded console. Mapping them into the logger folds them
    into the same file as everything else.
    """
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    level_for = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    qt_logger = logging.getLogger("qt")

    def handler(mode, context, message) -> None:
        qt_logger.log(level_for.get(mode, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)
