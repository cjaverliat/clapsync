# Qt-Native Preview — Design

Date: 2026-07-24
Status: Proposed
Supersedes: `2026-07-24-measured-audible-clock-design.md` (the custom clock it
fixes is deleted here, so that fix is moot).

## Motivation

Preview A/V is incoherent because audio and video are **two hand-synced
pipelines**: a custom `AudioEngine` (torch waveforms → mixed → `QAudioSink`)
and framepipe video decode, locked by a hand-rolled played-clock. Every attempt
to fix the clock fights the audio subsystem's hidden buffer latency.

A single `QMediaPlayer` keeps its own audio+video coherent internally
(hardware-timestamped) — for free. Let Qt own playback: one master player
provides audible A/V; all other tracks slave to its `position()`. The whole
hand-sync clock disappears.

## Scope

Replaces the **preview playback path only**. The autosync solver (clap
detection via torch `load_audio`) is unchanged — offsets are computed before
the editor opens and passed in; the editor never needs torch audio for preview.

## Architecture

New module `gui/qt_preview.py` — `QtPreviewController(QObject)`:

- Owns one `QMediaPlayer` per **audible** track (`is_av`: video and audio-only
  mp3/wav). c3d/fbx have no player.
- **Master** = the locked reference track's player (the reference is always an
  a/v track, never c3d — see commit 93ec1a6). Master has a `QAudioOutput`
  (audible); it free-runs and defines the timeline.
- **Slaves** = every other player, audio muted (`setAudioOutput(None)`),
  chasing the master.
- Video is shown via `QVideoWidget` per cell (Qt composites on the GPU, no
  per-frame Python). Replaces `VideoPlayerWidget`'s numpy/`_FrameLabel` feed.

### Timeline / offset math

Global timeline seconds ↔ each player's local media seconds:

```
local_i = global - offset_i          # player i's media position
global  = master_local + offset_master
```

- Play: seek each player to `local_i = global - offset_i`, then `play()`.
- The master's `position()` (ms) → `global = master.position()/1000 +
  offset_master` drives the playhead, c3d previews, and time label via a render
  `QTimer` (replaces `_on_audio_tick`).

### Herding controller (`QTimer`, ~50 ms)

Master free-runs. For each slave, `target_local = global - offset_i`,
`drift = slave.position()/1000 - target_local`:

- `|drift| > 120 ms` → hard `setPosition(target_local)` (a real jump; also used
  after any seek).
- `20 ms < |drift| ≤ 120 ms` → nudge `setPlaybackRate(0.96 / 1.04)` toward
  lock.
- else → `setPlaybackRate(1.0)`.

Spike measured ~67 ms worst-case drift with a cruder version; this is
acceptable for preview and well below the old intermittent offset. Tightening
(finer steps / PI) is a later refinement, not required.

### Frame-accurate scrub / step (paused)

Spike proved `setPosition` is frame-deterministic (returns `floor(t·fps)`),
correctable by requesting `(frame + 0.5)/fps`. Paused step/seek sets every
player's position from the global time; master defines the frame, slaves match.

### Proxies

Unchanged. Qt HW-decode sustains only ~2 full-res 5.3K streams (spike), but 9+
480p proxies trivially. The existing `ensure_preview_proxy` path feeds proxy
paths to the players exactly as it did to framepipe.

## Deletions

- `gui/audio_engine.py` (`AudioEngine`, `mix_block`, `played_position_s`,
  `_MixDevice`) — preview no longer mixes audio. **Kept:** nothing; the module
  is removed. `load_audio` (in `app/decode.py`) stays for the solver.
- `gui/video_player.py` `VideoGroupWorker` and the framepipe import
  (`NvVideoDecoder`, `VideoGroupDecoder`, `TrackSpec`, `GroupFrame`,
  `PyAvVideoDecoder`). `VideoPlayerWidget` is replaced by a `QVideoWidget` cell.
- `gui/playback_clock.py` (`slew_clock`) if unused after the switch.
- `sync_editor` wiring: `_init_audio`, `_load_all_videos` (framepipe branch),
  `_on_frames_ready`, `_on_loading_changed` (worker), `_on_audio_tick`,
  `_stop_group_worker`, `_group_worker`/`_seek_gen` plumbing → replaced by
  `QtPreviewController` calls.

## sync_editor rewiring

- Build `QtPreviewController` with `(infos, offsets, use_proxies, reference)`.
- Cells: video tracks → `QVideoWidget` cell. Audio-only tracks have no cell
  (waveform lane only), as today — their player is audible only if it is the
  reference (master), else a muted slave.
- Controls map 1:1: `_on_play_pause` → `controller.play()/pause()`;
  `_on_playhead_seek`/`_step_frame` → `controller.seek(global_s)`;
  `_on_offsets_changed` → `controller.set_offsets(offsets)`; render timer polls
  `controller.global_position()` to drive playhead + `_update_mocap_previews`.
- EOF: `master.mediaStatusChanged == EndOfMedia` → `controller.eof` signal →
  existing loop/stop logic.

## Audio-only tracks & the master

- Reference is an a/v track → master audible with its own audio+video.
- If the reference is an audio-only track (e.g. a wav) → master is that player
  (audible, no video). Videos are all slaves chasing it. Works: audio-only
  master still exposes `position()`.
- Non-reference audio-only tracks: muted slaves (no audible output, no cell).
  You hear one clean track — the reference. (Multi-audio compare is a possible
  later toggle, out of scope.)

## Testing

Pure/unit:
- Offset math helpers (`local_for(global, offset)`,
  `global_from_master(master_local, offset_master)`) — pure functions, unit
  tested (round-trip, negative offsets, clamps).
- Herding decision (`herd_action(drift_ms) -> ("seek"|"rate"|"hold", value)`) —
  pure, table-tested at the 20/120 ms boundaries.

In-app (the actual bug — must be observed):
- Two GoPros: play → the two pictures and the audible track are coherent, and
  stay coherent across repeated play/stop (the intermittent offset is gone).
- Scrub/step lands frame-accurately; c3d preview tracks the playhead.

## Success criteria

1. Preview audio (reference track) and all video are coherent, repeatably.
2. Frame-step/scrub stays frame-accurate; c3d preview follows.
3. `AudioEngine`/framepipe removed from the preview path; app runs without them.
4. Offset-math and herding unit tests pass; existing non-preview tests pass.

## Risks

- **Herding drift** (~67 ms). Mitigation: hard-seek threshold caps it; tighten
  later if visible.
- **QVideoWidget in a grid** (N native windows) — validate it composites/scales
  in the mosaic; fallback is `QGraphicsVideoItem` or `QVideoSink`+paint.
- **Audio-only-as-master** rarely exercised — ensure `position()` drives the
  clock even with no video.
