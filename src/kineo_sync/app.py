from __future__ import annotations

import argparse
import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog

from kineo_sync.gui.video_selection_dialog import VideoSelectionDialog
from kineo_sync.offset_worker import compute_offsets_with_progress
from kineo_sync.sync_editor import SyncEditorWindow

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kineo audio sync tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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

    window = SyncEditorWindow(video_paths=video_paths, offsets=offsets)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
