"""Hybrid preview transport: one sample-exact audio mix + Qt video players.

Independent QMediaPlayers cannot phase-lock their *audio* — each audio pipeline
has its own output latency, so multiple mic/audio sources played through
separate players echo (~150 ms apart, uncorrectable by seeking because the
latency is downstream of setPosition). So audio is played as a single
sample-exact mix (AudioEngine): every source is summed at its exact sample
offset into one stream through one QAudioSink — one pipeline, one latency, no
inter-source echo.

Video stays on Qt QMediaPlayers (muted, video-only): they sync to each other to
~2 ms and decode in real time. They are the *slaves* here — recurrently
re-synced to the mix's audible clock, so the picture tracks the sound. Video vs
audio offset is a single fixed buffer, well within lip-sync tolerance, unlike
the audible audio-audio echo the mix removes.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path

import torch
from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

from clapsync.app.decode import ensure_preview_proxy, load_audio
from clapsync.app.media import MediaInfo, is_av
from clapsync.gui.audio_engine import AudioEngine
from clapsync.gui.progress import run_blocking_with_progress

logger = logging.getLogger(__name__)


def _decode_mix_waveforms(infos: list[MediaInfo]) -> list[torch.Tensor]:
    """Decode each a/v track to a 48 kHz waveform for the mix (silent otherwise).

    Pure decode (no Qt), so it can run off the GUI thread behind a busy dialog.
    """
    waveforms: list[torch.Tensor] = []
    for info in infos:
        if not is_av(info.kind):
            waveforms.append(torch.zeros(1, 1))  # mocap is silent
            continue
        try:
            wave, _rate = load_audio(info.path, target_rate=48000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("no audio for %s: %s", info.path.name, exc)
            wave = torch.zeros(1, 1)
        waveforms.append(wave)
    return waveforms


# Video-follows-audio herding. Video free-runs at rate 1.0 and is hard-seeked to
# the mix's audible clock when its averaged drift exceeds _RESYNC_S. Averaging
# beats QMediaPlayer.position()'s ±50 ms jitter; the threshold is comfortably
# within lip-sync tolerance (video may lead/lag audio ~40-80 ms imperceptibly),
# so recurrent seeks keep the picture locked to the sound without visible churn.
_HERD_MS = 50
_HERD_WINDOW = 10           # samples averaged (~0.5 s) to kill position() jitter
_RESYNC_S = 0.040           # re-sync video when averaged drift exceeds ~40 ms


def local_position_s(global_s: float, offset_s: float) -> float:
    """A track's local media time for a shared-timeline time (floored at 0)."""
    return max(0.0, global_s - offset_s)


def herd_action(avg_drift_s: float) -> tuple[str, float]:
    """Decide whether to re-sync a video player from its averaged drift.

    ``avg_drift_s`` is mean(video_local - target_local) over the herd window.
      ("hold", 0.0)  — within threshold, leave it running
      ("seek", 0.0)  — drift too large, hard-seek the video to the target
    """
    if abs(avg_drift_s) > _RESYNC_S:
        return ("seek", 0.0)
    return ("hold", 0.0)


class _VideoTrack:
    """One video track's muted, video-only player and its display widget."""

    def __init__(self, source: Path) -> None:
        self.player = QMediaPlayer()
        self.widget = QVideoWidget()
        self.player.setVideoOutput(self.widget)
        # No audio output: the mix owns all sound. Video is picture-only.
        self.player.setAudioOutput(None)
        self.player.setSource(QUrl.fromLocalFile(str(source)))
        self.drift: collections.deque[float] = collections.deque(
            maxlen=_HERD_WINDOW
        )


class QtPreviewController(QObject):
    """Sample-exact audio mix as master clock; Qt video players slaved to it.

    Signals:
        position_changed(float): shared-timeline seconds (the mix's clock).
        eof(): playback reached the end of the media.
    """

    position_changed = Signal(float)
    eof = Signal()

    def __init__(
        self,
        infos: list[MediaInfo],
        offsets: list[float],
        reference_idx: int,
        use_proxies: bool,
        muted: list[bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._infos = infos
        self._offsets = list(offsets)
        self._playing = False
        self._global_pos = 0.0
        self._muted = list(muted) if muted is not None else [False] * len(infos)

        # Audio: one sample-exact mix of every a/v source, the master clock.
        # Decoding every track to 48 kHz is the slow part of opening the editor,
        # so run it behind an animated busy dialog (off the GUI thread) instead
        # of freezing the window while it loads.
        self._audio = AudioEngine(self)
        waveforms = run_blocking_with_progress(
            lambda: _decode_mix_waveforms(infos), "Loading audio mix…",
        )
        self._audio.set_tracks(waveforms, self._offsets)
        self._audio.set_muted(self._muted)
        self._audio.position_changed.connect(self._on_audio_clock)

        # Video: muted, video-only Qt players, one per video track.
        self._videos: dict[int, _VideoTrack] = {}
        for i, info in enumerate(infos):
            if info.kind != "video":
                continue
            src = ensure_preview_proxy(info.path) if use_proxies else info.path
            track = _VideoTrack(src)
            track.player.mediaStatusChanged.connect(
                lambda st, idx=i: self._on_video_status(idx, st)
            )
            self._videos[i] = track

        # Recurrently re-sync video to the audible clock while playing.
        self._herd_timer = QTimer(self)
        self._herd_timer.setInterval(_HERD_MS)
        self._herd_timer.timeout.connect(self._herd)
        self._herd_ticks = 0

    # ── Widgets ──────────────────────────────────────────────────────────────
    def video_widget(self, track_idx: int) -> QVideoWidget | None:
        track = self._videos.get(track_idx)
        return track.widget if track else None

    # ── Transport ────────────────────────────────────────────────────────────
    def global_position(self) -> float:
        """Current shared-timeline position — the mix's audible clock."""
        return self._audio.position_s() if self._audio.enabled else self._global_pos

    def seek(self, global_s: float) -> None:
        """Move the mix and every video player to ``global_s`` on the timeline."""
        self._global_pos = global_s
        self._audio.seek(global_s)
        for i, track in self._videos.items():
            local = local_position_s(global_s, self._offsets[i])
            track.player.setPosition(int(round(local * 1000)))
            track.drift.clear()
        self.position_changed.emit(global_s)

    def play(self) -> None:
        self._playing = True
        self._audio.play()
        for track in self._videos.values():
            track.drift.clear()
            track.player.play()
        self._herd_timer.start()

    def pause(self) -> None:
        self._playing = False
        self._herd_timer.stop()
        self._audio.pause()
        for track in self._videos.values():
            track.player.pause()

    def set_offsets(self, offsets: list[float]) -> None:
        self._offsets = list(offsets)
        self._audio.set_offsets(offsets)
        if not self._playing:
            self.seek(self._global_pos)

    def set_muted(self, track_idx: int, muted: bool) -> None:
        """Mute/unmute a track in the audio mix (video carries no sound)."""
        if 0 <= track_idx < len(self._muted):
            self._muted[track_idx] = muted
        self._audio.set_muted(self._muted)

    def stop(self) -> None:
        self.pause()
        for track in self._videos.values():
            track.player.setSource(QUrl())

    # ── Internal ─────────────────────────────────────────────────────────────
    def _on_audio_clock(self, clock_s: float) -> None:
        # The mix's 33 Hz clock drives the playhead / c3d preview.
        self._global_pos = clock_s
        self.position_changed.emit(clock_s)

    def _herd(self) -> None:
        if not self._audio.enabled:
            return
        global_s = self._audio.position_s()
        self._herd_ticks += 1
        log_now = self._herd_ticks % (_HERD_WINDOW * 4) == 0
        for i, track in self._videos.items():
            target = local_position_s(global_s, self._offsets[i])
            track.drift.append(track.player.position() / 1000.0 - target)
            if len(track.drift) < _HERD_WINDOW:
                continue
            avg = sum(track.drift) / len(track.drift)
            if log_now:
                logger.debug(
                    "herd video %d: avg_drift=%+.1fms", i, avg * 1000
                )
            if herd_action(avg)[0] == "seek":
                track.player.setPosition(int(round(target * 1000)))
                track.drift.clear()

    def _on_video_status(
        self, idx: int, status: QMediaPlayer.MediaStatus
    ) -> None:
        track = self._videos.get(idx)
        if track is None:
            return
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            # setSource is async; land the pending playhead once seekable.
            local = local_position_s(self._global_pos, self._offsets[idx])
            track.player.setPosition(int(round(local * 1000)))
