"""Audible preview: mixes unmuted tracks through QAudioSink on a wall clock."""
from __future__ import annotations

import logging

import numpy as np
import torch
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal

logger = logging.getLogger(__name__)

_RATE = 48000


def mix_block(
    tracks: list[np.ndarray],
    offsets: list[float],
    muted: list[bool],
    start_s: float,
    n: int,
    rate: int,
) -> np.ndarray:
    """Mean-mix n samples of the unmuted tracks starting at global start_s.

    Each track's local index is (global - offset). Out-of-range regions
    contribute silence. Returns (n,) float32 in [-1, 1]; all-muted -> zeros.
    """
    out = np.zeros(n, dtype=np.float32)
    active = 0
    for wave, off, is_muted in zip(tracks, offsets, muted):
        if is_muted:
            continue
        active += 1
        local0 = int(round((start_s - off) * rate))
        seg = np.zeros(n, dtype=np.float32)
        src_lo = max(0, local0)
        src_hi = min(wave.shape[0], local0 + n)
        if src_hi > src_lo:
            dst_lo = src_lo - local0
            seg[dst_lo:dst_lo + (src_hi - src_lo)] = wave[src_lo:src_hi]
        out += seg
    if active > 0:
        out /= active
    return out


class _MixDevice(QIODevice):
    def __init__(self, engine: "AudioEngine") -> None:
        super().__init__()
        self._engine = engine

    def readData(self, maxlen: int) -> bytes:
        # maxlen is bytes; int16 mono => 2 bytes/sample.
        n = max(0, maxlen // 2)
        if n == 0:
            return b""
        block = self._engine._pull(n)  # float32 (n,)
        i16 = np.clip(block, -1.0, 1.0)
        i16 = (i16 * 32767.0).astype(np.int16)
        return i16.tobytes()

    def writeData(self, data) -> int:  # never written to
        return 0

    def bytesAvailable(self) -> int:
        return 1 << 20  # always claim data available (continuous stream)

    def isSequential(self) -> bool:
        return True


class AudioEngine(QObject):
    position_changed = Signal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tracks: list[np.ndarray] = []
        self._offsets: list[float] = []
        self._muted: list[bool] = []
        self._cursor = 0            # samples from time 0 on the shared timeline
        self.enabled = True
        self._sink = None
        self._dev = None
        try:
            from PySide6.QtMultimedia import (
                QAudioFormat, QAudioSink, QMediaDevices,
            )
            out = QMediaDevices.defaultAudioOutput()
            if out.isNull():
                raise RuntimeError("no default audio output")
            fmt = QAudioFormat()
            fmt.setSampleRate(_RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self._sink = QAudioSink(out, fmt)
            self._dev = _MixDevice(self)
            self._dev.open(QIODevice.OpenModeFlag.ReadOnly)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio preview disabled (%s)", exc)
            self.enabled = False
        # Emit position on a light timer while playing.
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(
            lambda: self.position_changed.emit(self.position_s())
        )

    def set_tracks(self, waveforms: list[torch.Tensor], offsets: list[float]) -> None:
        # Mono float32 numpy at _RATE (waveforms already decoded at 48k).
        self._tracks = [
            w.reshape(-1).to(torch.float32).cpu().numpy() for w in waveforms
        ]
        self._offsets = list(offsets)
        if not self._muted or len(self._muted) != len(self._tracks):
            self._muted = [False] * len(self._tracks)

    def set_muted(self, muted: list[bool]) -> None:
        self._muted = list(muted)

    def set_offsets(self, offsets: list[float]) -> None:
        self._offsets = list(offsets)

    def _pull(self, n: int) -> np.ndarray:
        block = mix_block(
            self._tracks, self._offsets, self._muted,
            self._cursor / _RATE, n, _RATE,
        )
        self._cursor += n
        return block

    def position_s(self) -> float:
        return self._cursor / _RATE

    def seek(self, global_s: float) -> None:
        # Just move the cursor. A running sink picks it up on its next pull
        # (a few ms later) — do NOT stop/start the sink here: restarting the
        # audio device is expensive and, if called per video frame by the
        # editor's drift corrector, tanks playback. play() owns starting.
        self._cursor = max(0, int(round(global_s * _RATE)))

    def play(self) -> None:
        if not self.enabled or self._sink is None:
            return
        self._sink.start(self._dev)
        self._timer.start()

    def pause(self) -> None:
        if not self.enabled or self._sink is None:
            return
        self._sink.stop()
        self._timer.stop()
