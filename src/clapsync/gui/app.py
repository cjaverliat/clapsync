from __future__ import annotations

import argparse
import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from clapsync import diagnostics
from clapsync.gui.media_selection_dialog import MediaSelectionDialog
from clapsync.gui.progress import run_blocking_with_progress, startup_progress

logger = logging.getLogger(__name__)


def _init_audio_devices() -> None:
    """Force Qt's audio-endpoint enumeration once, up front.

    The first ``QMediaDevices`` query enumerates every system audio endpoint and
    Qt caches the result process-wide. On some Windows machines that takes ~12 s,
    and it *always* blocks the GUI thread: Qt's Windows audio backend marshals
    the call into the main thread, so running it on a worker thread freezes the
    window just the same (measured: a single ``processEvents`` call blocked for
    12.5 s). Since the stall cannot be hidden, it is paid here — once, at
    startup, behind a dialog that says what is happening — rather than later
    while the editor builds its ``AudioEngine``.

    Failures are ignored: this is only a warm-up, and AudioEngine reports a
    missing device itself.
    """
    try:
        from PySide6.QtMultimedia import QMediaDevices

        QMediaDevices.defaultAudioOutput()
    except Exception:  # noqa: BLE001 — warm-up only, never fatal
        logger.debug("audio device initialization failed", exc_info=True)


def _report_crash(exc: BaseException) -> None:
    """Show a dialog pointing at the log file after an uncaught exception."""
    log_file = diagnostics.log_dir() / "clapsync.log"
    QMessageBox.critical(
        None,
        "clapsync crashed",
        f"An unexpected error occurred:\n\n{exc}\n\n"
        f"A full report was saved to:\n{log_file}",
    )


def _decide_use_proxies(video_paths: list, forced: bool) -> bool:
    """Return whether to generate preview proxies for this session.

    ``--use_proxies`` forces them on; otherwise proxies are offered (via a
    prompt) only when a source is tall enough to actually need one — ≤1080p
    already plays in real time.

    Args:
        video_paths: Selected input files.
        forced: True if ``--use_proxies`` was passed.
    """
    if forced:
        logger.info("proxies: forced on via --use_proxies")
        return True
    from clapsync.app.decode import source_needs_preview_proxy
    from clapsync.app.media import probe
    if not any(source_needs_preview_proxy(probe(p)) for p in video_paths):
        logger.info("proxies: skipped, no source exceeds 1080p")
        return False
    answer = _prompt_use_proxies()
    logger.info("proxies: high-res source detected, user chose %s", answer)
    return answer


def _prompt_use_proxies() -> bool:
    """Ask whether to generate preview proxies; returns True for Yes.

    Shown only when a high-resolution source is detected. Yes is the default
    (Enter/Escape both select it) and the recommended choice.
    """
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Preview proxies")
    box.setText(
        "Some of your videos are higher than 1080p, which can stutter during "
        "preview playback."
    )
    box.setInformativeText(
        "Generate lightweight 480p preview proxies for smooth playback in the "
        "editor?\n\nThis is a one-time transcode. Export always uses the "
        "full-resolution originals."
    )
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.Yes)
    return box.exec() == QMessageBox.StandardButton.Yes


def _resolve_marker_choices(video_paths: list) -> dict:
    """Resolve clapperboard markers for c3d inputs before offset computation.

    Runs on the main thread so the sync worker never has to raise a dialog.
    Files whose markers auto-classify by name are skipped; ambiguous ones
    prompt a MarkerSelectionDialog. Cancelling leaves a file unresolved — the
    bridge then leaves it at offset 0 with a warning.

    Returns:
        {input_index: (top_indices, bottom_indices)} for manually picked files.
    """
    from clapsync.app.mocap import load_c3d
    from clapsync.core.clap import classify_clap_markers
    from clapsync.gui.marker_selection_dialog import MarkerSelectionDialog

    choices: dict = {}
    for idx, path in enumerate(video_paths):
        if path.suffix.lower() != ".c3d":
            continue
        try:
            data = load_c3d(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s for marker selection: %s",
                           path.name, exc)
            continue
        if classify_clap_markers(data.labels) is not None:
            continue
        dialog = MarkerSelectionDialog(path.name, data.labels)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            choices[idx] = dialog.get_selection()
    return choices


def main() -> None:
    parser = argparse.ArgumentParser(description="clapsync — multi-camera audio sync tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--use_proxies",
        action="store_true",
        help="Force preview proxies without prompting (480p/30fps per source). "
        "By default clapsync offers proxies only when a source exceeds 1080p.",
    )
    args, qt_args = parser.parse_known_args()

    log_file = diagnostics.configure_logging(verbose=args.verbose)
    diagnostics.install_excepthook(on_exception=_report_crash)

    # Create the QApplication before the environment warm-up so a window can
    # exist during it (and so a crash there can raise a dialog).
    app = QApplication([sys.argv[0], *qt_args])
    diagnostics.install_qt_message_handler()

    # Allow CTRL+C to quit the Qt event loop.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Qt blocks the GIL; a periodic no-op timer lets Python check for signals.
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)

    # log_startup_context imports torch and probes CUDA (several seconds). Run it
    # off the main thread so the busy dialog keeps animating instead of freezing;
    # it only logs (no Qt objects), so it is safe off-thread.
    run_blocking_with_progress(
        diagnostics.log_startup_context, "Checking environment…"
    )
    # Audio-endpoint enumeration cannot be moved off the GUI thread (see
    # _init_audio_devices), so it runs here with the status text painted first:
    # the marquee stops, but the user knows what the wait is for.
    with startup_progress() as status:
        status("Initializing audio devices (first run can take a while)…")
        _init_audio_devices()
    logger.info("logging to %s", log_file)

    sel = MediaSelectionDialog()
    if sel.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    video_paths = sel.get_media_paths()

    # Probe every input off the main thread so the busy dialog keeps animating
    # instead of freezing while the inputs load. The result is reused for logging
    # and offset computation (no worker re-probe).
    def _probe_inputs() -> list:
        from clapsync.app.media import probe
        return [probe(p) for p in video_paths]

    try:
        media = run_blocking_with_progress(_probe_inputs, "Probing inputs…")
    except Exception as exc:  # noqa: BLE001 — surface a bad input, don't crash
        QMessageBox.critical(
            None, "Error", f"Could not read an input file:\n{exc}"
        )
        sys.exit(1)

    # Loading the decode stack (torch/torchaudio/av) is a set of heavy
    # imports kept out of module scope so earlier dialogs paint instantly. Run
    # the actual module execution off the main thread so the busy dialog keeps
    # animating; the module-scope re-import below is then a cached no-op. These
    # modules only define classes at import — no QObject is built off-thread.
    def _load_engine() -> None:
        import clapsync.core  # noqa: F401
        import clapsync.gui.sync_editor  # noqa: F401
        import clapsync.gui.workers  # noqa: F401

    run_blocking_with_progress(_load_engine, "Loading sync engine…")
    from clapsync.core import LOW_CONFIDENCE
    from clapsync.gui.sync_editor import SyncEditorWindow
    from clapsync.gui.workers import (
        compute_offsets_with_progress,
        prepare_proxies_with_progress,
    )
    diagnostics.log_inputs(video_paths, media=media)

    marker_choices = _resolve_marker_choices(video_paths)
    alignment = compute_offsets_with_progress(
        video_paths, marker_choices=marker_choices, media=media
    )
    if alignment is None:
        sys.exit(0)
    offsets = alignment.offsets

    if alignment.warnings:
        lines = "\n".join(f"• {w}" for w in alignment.warnings)
        scores = "\n".join(
            f"• {p.name}: confidence {c:.1f}"
            for p, c in zip(video_paths, alignment.confidence)
            if c < LOW_CONFIDENCE
        )
        QMessageBox.warning(
            None,
            "Low sync confidence",
            f"Automatic sync may be wrong for some tracks:\n\n{lines}\n\n"
            f"Scores (below {LOW_CONFIDENCE:g} needs manual verification):\n{scores}",
        )

    use_proxies = _decide_use_proxies(video_paths, args.use_proxies)

    # Transcode preview proxies up front behind a progress bar, so the timeline
    # opens onto warm frames instead of a black "Loading…" while the decode
    # thread transcodes 5.3K footage.
    if use_proxies and not prepare_proxies_with_progress(video_paths):
        sys.exit(0)

    low = [c < LOW_CONFIDENCE for c in alignment.confidence]
    window = SyncEditorWindow(
        video_paths=video_paths, offsets=offsets, use_proxies=use_proxies,
        low_confidence=low,
    )
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
