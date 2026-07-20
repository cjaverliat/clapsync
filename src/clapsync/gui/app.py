from __future__ import annotations

import argparse
import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from clapsync.gui.video_selection_dialog import VideoSelectionDialog

logger = logging.getLogger(__name__)


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
        return True
    from clapsync.app.decode import source_needs_preview_proxy
    from clapsync.app.media import probe
    if not any(source_needs_preview_proxy(probe(p)) for p in video_paths):
        return False
    return _prompt_use_proxies()


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

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = QApplication([sys.argv[0], *qt_args])

    # Allow CTRL+C to quit the Qt event loop.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Qt blocks the GIL; a periodic no-op timer lets Python check for signals.
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)

    sel = VideoSelectionDialog()
    if sel.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    video_paths = sel.get_video_paths()

    # Deferred: importing these pulls torch/torchaudio/framepipe (~3s + CUDA
    # init). Keeping them out of module scope lets the selection dialog above
    # paint instantly instead of after the whole decode stack loads.
    from clapsync.gui.sync_editor import SyncEditorWindow
    from clapsync.gui.workers import (
        compute_offsets_with_progress,
        prepare_proxies_with_progress,
    )

    offsets = compute_offsets_with_progress(video_paths)
    if offsets is None:
        sys.exit(0)

    use_proxies = _decide_use_proxies(video_paths, args.use_proxies)

    # Transcode preview proxies up front behind a progress bar, so the timeline
    # opens onto warm frames instead of a black "Loading…" while the decode
    # thread transcodes 5.3K footage.
    if use_proxies and not prepare_proxies_with_progress(video_paths):
        sys.exit(0)

    window = SyncEditorWindow(
        video_paths=video_paths, offsets=offsets, use_proxies=use_proxies
    )
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
