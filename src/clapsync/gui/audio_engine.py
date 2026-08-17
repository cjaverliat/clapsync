"""Audible preview: mixes unmuted tracks through QAudioSink on a wall clock."""
from __future__ import annotations

import logging

import numpy as np
import torch
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal

logger = logging.getLogger(__name__)

_RATE = 48000
# QAudioSink pull buffer. processedUSecs() counts samples *pulled into this
# buffer*, not samples the speaker has emitted, so the visual clock built on it
# leads the audible sound by roughly the buffer's fill. Qt's default on some
# backends is ~500 ms (e.g. USB headsets), which puts the picture/c3d a quarter
# second ahead of the clap. A small buffer bounds that lead (and shrinks the
# coarse update step, smoothing the clock) while staying large enough that the
# cheap numpy mix never underruns.
_BUFFER_MS = 100


def played_position_s(origin_s: float, processed_us: float, base_us: float = 0.0) -> float:
    """Content seconds actually audible, from the sink's played-time counter.

    origin_s is the content time at which the current sink.start() began;
    processed_us is QAudioSink.processedUSecs() (microseconds actually pushed to
    the device since that start), base_us the counter value at the last in-play
    re-anchor. This tracks what the speaker is emitting — unlike the pull cursor,
    which leads by the whole ring buffer and primes by a random amount per run.
    """
    return origin_s + (processed_us - base_us) / 1_000_000.0


def audible_position_s(
    origin_s: float,
    processed_us: float,
    buffered_bytes: int,
    byte_rate: int,
    base_us: float = 0.0,
) -> float:
    """Content seconds actually audible, from measured sink buffer occupancy.

    processed_us counts samples pulled *into* the sink; buffered_bytes
    (bufferSize - bytesFree) are the pulled bytes not yet emitted. Subtracting
    them turns the pull-cursor time into what the speaker is emitting now, so the
    video (which chases this clock) matches the sound rather than leading it by
    the buffer. Floored at origin: right after start() the buffer fills before
    any sound comes out, so the raw value would briefly read before origin.
    """
    pulled = origin_s + (processed_us - base_us) / 1_000_000.0
    buffered_s = buffered_bytes / byte_rate if byte_rate else 0.0
    return max(origin_s, pulled - buffered_s)


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
        self._playing = False
        # Anchors for the played-time clock: content seconds where the current
        # sink.start() began, and the processedUSecs baseline for in-play seeks.
        self._origin_s = 0.0
        self._base_us = 0.0
        # Last computed played position (seconds), published for the video worker
        # to chase on the decode thread — a plain float read is GIL-safe, so it
        # never touches the QAudioSink from another thread.
        self.clock_s = 0.0
        self.enabled = True
        self._sink = None
        self._dev = None
        # Format bytes-per-second, to convert queued sink bytes to seconds in
        # the audible-position clock. Set once the format is built below.
        self._byte_rate = 0
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
            self._byte_rate = fmt.bytesForDuration(1_000_000)  # bytes per second
            self._sink = QAudioSink(out, fmt)
            # 2 bytes/sample, mono: bound the pull buffer so the played-clock
            # doesn't lead the speaker by the backend's (large) default fill.
            self._sink.setBufferSize(int(_RATE * 2 * _BUFFER_MS / 1000))
            self._dev = _MixDevice(self)
            self._dev.open(QIODevice.OpenModeFlag.ReadOnly)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio preview disabled (%s)", exc)
            self.enabled = False
        # Refresh the clock and emit position on a light timer while playing.
        # ~33 ms (≈30 Hz) matches the video display grid: fine enough for the
        # worker (which chases clock_s) and the c3d preview to track the sound,
        # without flooding the GUI with preview repaints. Sync accuracy comes
        # from the clock being *correct*, not from a high tick rate.
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self.clock_s = self.position_s()
        self.position_changed.emit(self.clock_s)

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
        """Where the sound actually is, in shared-timeline seconds.

        While playing this is the sink's *played* time, not the pull cursor:
        the cursor leads by the whole ring buffer (primed by a random amount
        each run), which is exactly what made the clap drift out of sync with
        the c3d frame between playbacks.
        """
        if self._playing and self._sink is not None:
            buffered = self._sink.bufferSize() - self._sink.bytesFree()
            return audible_position_s(
                self._origin_s, self._sink.processedUSecs(),
                buffered, self._byte_rate, self._base_us,
            )
        return self._cursor / _RATE

    def seek(self, global_s: float) -> None:
        # Move the pull cursor. A running sink picks it up on its next pull;
        # do NOT stop/start the sink (a device restart is expensive). If we are
        # playing, re-anchor the played-time clock so position_s stays correct
        # (processedUSecs keeps counting across a cursor move — it only resets
        # on start()).
        self._cursor = max(0, int(round(global_s * _RATE)))
        if self._playing and self._sink is not None:
            self._origin_s = self._cursor / _RATE
            self._base_us = self._sink.processedUSecs()
            self.clock_s = self._origin_s

    def play(self) -> None:
        if not self.enabled or self._sink is None:
            return
        # processedUSecs resets to 0 on start(), so anchor the clock at the
        # cursor's content time with a zero baseline.
        self._origin_s = self._cursor / _RATE
        self._base_us = 0.0
        self.clock_s = self._origin_s
        self._playing = True
        self._sink.start(self._dev)
        self._timer.start()

    def pause(self) -> None:
        if not self.enabled or self._sink is None:
            return
        # Capture the played position before stopping so the cursor no longer
        # holds the buffer-lead — resume then starts from what was audible.
        self._cursor = max(0, int(round(self.position_s() * _RATE)))
        self._playing = False
        self._sink.stop()
        self._timer.stop()
