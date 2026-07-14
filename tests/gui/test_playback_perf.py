"""End-to-end playback performance of the mosaic preview, headless.

Drives the real ``SyncEditorWindow`` through the offscreen Qt platform: the
worker thread decodes on the GPU, emits ``frames_ready``, and the widgets paint
``QPixmap``s — the full pipeline, no Qt server needed. Display is decoupled from
decode and emitted on a steady 30 fps grid, so this checks that playback
*delivers* 30 fps for 6 tracks at 1080p and 4K (motion may repeat frames under
a decode shortfall; the display rate must hold).

Inputs are generated with ffmpeg (testsrc2, real motion/detail — a solid color
would decode unrealistically fast), so no video is checked into the repo. One
clip per resolution is reused across the 6 tracks: 6 independent decoders, 6
NVDEC sessions, same content.

Marked ``slow`` and gated on CUDA + ffmpeg: it is the GPU playback path, opt in
with ``pytest -m slow -s``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

# Must precede any PySide6 import so Qt selects the headless backend.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU playback path requires CUDA"
    ),
]

TRACKS = 6
CLIP_S = 8.0       # generated clip length
WARMUP_S = 1.5     # let prefetchers prime before timing
MEASURE_S = 4.0    # playback window actually timed
MIN_FPS = 24.0     # 0.8 x the 30 fps target: catches lag/regressions

# (id, width, height, gate). Both gated: the display rate is decoupled from
# decode, so playback holds 30 fps at 4K too (motion drops under a decode
# shortfall, but the delivered frame rate must not).
RESOLUTIONS = [("1080p", 1920, 1080, True), ("4K", 3840, 2160, True)]


def _generate(path: Path, w: int, h: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={w}x{h}:rate=30:duration={CLIP_S}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(path)],
        check=True, capture_output=True,
    )


def _pump(qapp, seconds: float) -> None:
    """Run the Qt event loop (delivering queued frames) for ``seconds``."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.001)


def _wait_for_first_frame(qapp, count, timeout: float = 60.0) -> None:
    """Pump until the worker has opened the decoders and delivered a frame."""
    end = time.perf_counter() + timeout
    while count[0] == 0 and time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.005)
    if count[0] == 0:
        pytest.fail("no frame delivered — decoders never opened")


@pytest.mark.parametrize(
    "label,w,h,gate", RESOLUTIONS, ids=[r[0] for r in RESOLUTIONS]
)
def test_playback_sustains_30fps(tmp_path, label, w, h, gate) -> None:
    from PySide6.QtWidgets import QApplication

    from clapsync.gui.sync_editor import SyncEditorWindow
    from clapsync.gui.video_player import _DEVICE, _FPS

    clip = tmp_path / f"src_{label}.mp4"
    _generate(clip, w, h)
    paths = [clip] * TRACKS

    qapp = QApplication.instance() or QApplication([])
    win = SyncEditorWindow(paths, [0.0] * TRACKS)
    win.resize(1600, 900)
    win.show()

    count = [0]
    win._group_worker.frames_ready.connect(lambda *_: count.__setitem__(0, count[0] + 1))

    try:
        _wait_for_first_frame(qapp, count)

        win._on_play_pause()          # start playback
        _pump(qapp, WARMUP_S)         # prime prefetchers, not timed
        count[0] = 0
        t0 = time.perf_counter()
        _pump(qapp, MEASURE_S)
        elapsed = time.perf_counter() - t0
        delivered = count[0]
        win._on_play_pause()          # pause
    finally:
        win.close()

    fps = delivered / elapsed
    ok = fps >= MIN_FPS
    tag = ("OK" if ok else "LAGS") if gate else "informational"
    print(
        f"\n[{label}] {TRACKS}x {w}x{h} on {_DEVICE}: "
        f"{delivered} frames in {elapsed:.2f}s = {fps:.1f} fps "
        f"(target {_FPS:g}) -> {tag}"
    )
    assert delivered > 0, "playback delivered no frames"
    if gate:
        assert ok, (
            f"{label}: sustained {fps:.1f} fps < {MIN_FPS:g} - playback lags "
            f"for {TRACKS} tracks at {w}x{h}"
        )
