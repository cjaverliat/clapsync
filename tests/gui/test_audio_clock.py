"""The audio clock must report *played* time, not the pull-cursor lead.

Regression for clap/c3d desync: position_s() used to return the sample cursor,
which leads the audible output by the whole sink ring buffer (primed by a random
amount each run) — so the c3d marker and the clap sound never lined up twice.
"""
import numpy as np

from clapsync.gui.audio_engine import AudioEngine, played_position_s


class FakeSink:
    """Minimal stand-in for QAudioSink's played-time counter."""

    def __init__(self) -> None:
        self.us = 0.0
        self.started = False

    def processedUSecs(self) -> float:
        return self.us

    def start(self, _dev) -> None:
        self.started = True
        self.us = 0.0  # QAudioSink resets processedUSecs on start()

    def stop(self) -> None:
        self.started = False


class _NoTimer:
    """Stub for the engine's publish timer (no QApplication in this test)."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _engine_with(sink: FakeSink) -> AudioEngine:
    """An AudioEngine wired to a fake sink, no real audio device."""
    eng = AudioEngine.__new__(AudioEngine)
    eng._cursor = 0
    eng._playing = False
    eng._origin_s = 0.0
    eng._base_us = 0.0
    eng.clock_s = 0.0
    eng.enabled = True
    eng._sink = sink
    eng._dev = object()
    eng._timer = _NoTimer()
    return eng


def test_played_position_helper():
    assert played_position_s(2.0, 500_000) == 2.5
    # in-play re-anchor: only microseconds past the baseline count
    assert played_position_s(2.0, 1_500_000, base_us=1_000_000) == 2.5


def test_position_tracks_played_time_not_pull_cursor():
    sink = FakeSink()
    eng = _engine_with(sink)
    eng._cursor = int(3.0 * 48000)  # seek to 3.0s
    eng.play()
    # Sink has started but no audio has been played out yet.
    assert eng.position_s() == 3.0
    # After 200 ms of real playback, position advances by exactly that much —
    # regardless of how far the pull cursor has run ahead filling the buffer.
    sink.us = 200_000
    eng._cursor = int(3.5 * 48000)  # cursor pulled ahead (buffer prime)
    assert abs(eng.position_s() - 3.2) < 1e-9


def test_pause_captures_played_position():
    sink = FakeSink()
    eng = _engine_with(sink)
    eng._cursor = int(3.0 * 48000)
    eng.play()
    sink.us = 400_000  # 0.4 s played
    eng._cursor = int(4.0 * 48000)  # cursor way ahead from buffering
    eng.pause()
    # Resume must start from what was audible (3.4 s), not the buffered cursor.
    assert abs(eng.position_s() - 3.4) < 1e-6


def test_seek_reanchors_clock_while_playing():
    sink = FakeSink()
    eng = _engine_with(sink)
    eng.play()
    sink.us = 100_000
    eng.seek(10.0)
    # Right after the seek the clock reads the seek target...
    assert abs(eng.position_s() - 10.0) < 1e-9
    # ...and advances from there as more audio plays.
    sink.us = 350_000  # 250 ms past the seek baseline
    assert abs(eng.position_s() - 10.25) < 1e-9
