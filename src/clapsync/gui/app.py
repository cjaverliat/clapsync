from __future__ import annotations

import argparse
import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from clapsync.gui.sync_editor import SyncEditorWindow
from clapsync.gui.video_selection_dialog import VideoSelectionDialog
from clapsync.gui.workers import (
    compute_offsets_with_progress,
    prepare_proxies_with_progress,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="clapsync — multi-camera audio sync tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--use_proxies",
        action="store_true",
        help="Transcode a 480p/30fps proxy per source for preview playback "
        "(faster on high-resolution mosaics; adds a one-time transcode).",
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

    offsets = compute_offsets_with_progress(video_paths)
    if offsets is None:
        sys.exit(0)

    # With --use_proxies, transcode preview proxies up front behind a progress
    # bar, so the timeline opens onto warm frames instead of a black "Loading…"
    # while the decode thread transcodes 5.3K footage.
    if args.use_proxies and not prepare_proxies_with_progress(video_paths):
        sys.exit(0)

    window = SyncEditorWindow(
        video_paths=video_paths, offsets=offsets, use_proxies=args.use_proxies
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
