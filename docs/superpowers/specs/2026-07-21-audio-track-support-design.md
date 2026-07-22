# Audio Track Support — Design

**Date:** 2026-07-21
**Status:** Approved (pending final spec review)

## Goal

Let users sync any mix of video and audio files: video+audio, audio+audio,
video+video. The backend already treats every track as "a thing with an audio
stream" (`MediaInfo.kind`, audio-based sync, audio-only export path, CLI); all
gaps are in the GUI. Audio tracks get the same timeline affordances as video:
offset drag, trim, export. The export dialog gains audio parameters and
per-track selection. Preview becomes audible with per-track mute.

## Non-goals

- CLI flags for the new audio export parameters.
- Stereo preview (preview mixes to mono; export keeps source channels).
- Waveform rendering inside the timeline bars (mosaic cells only).
- Niche/unsupported-by-bundled-ffmpeg formats (opus, aiff, wma, ogg).
  Supported set: wav, mp3, flac, m4a (+ `.aac` accepted as input).

## Architecture stance

The backend stays audio-agnostic: lists of `MediaInfo` + offsets in, files
out. No backend code learns about UI track types. Track selection for export
is a GUI-side filter of those lists — the export API is unchanged apart from
two new optional `ExportSettings` fields.

## Components

### 1. Media selection dialog

`VideoSelectionDialog` → `MediaSelectionDialog`
(`gui/video_selection_dialog.py` → `gui/media_selection_dialog.py`).
File filter gains audio extensions: `*.wav *.mp3 *.flac *.m4a *.aac`.
Labels say "media" instead of "video". Only consumer is `gui/app.py`.

### 2. Preview mosaic

- Video track → `VideoPlayerWidget` (unchanged).
- Audio track → new `WaveformWidget`: static waveform painted from the
  16 kHz mono waveform already disk-cached by the sync phase
  (`load_audio(target_rate=16000)` — cache hit, free), plus a moving playhead
  line driven by the editor's position updates. Peaks are precomputed per
  pixel bucket on resize (numpy min/max), so paints are cheap.
- Bug fix: `sync_editor` debug log formats `info.width`/`info.height` with
  `%d` — crashes on `None` for audio tracks. Log dimensions only for video.

### 3. Video worker mapping

`VideoGroupWorker` receives only the video tracks. The editor keeps a
`video_slot → track_index` map and routes `frames_ready` frames to the right
mosaic cells. Sessions with zero video tracks start no group worker.

### 4. Audio playback engine (new: `gui/audio_engine.py`)

- Loads each track's mono waveform at 48 kHz via
  `load_audio(target_rate=48000)` (disk-cached like the sync waveforms).
  Stored in RAM as int16 (~345 MB/h/track); converted to float per pushed
  block. Long-session memory limit documented.
- On play/seek/mute-change: slices each unmuted track at the current global
  position (offset-shifted), mixes as the mean (no clipping), pushes to
  `QAudioSink` (PySide6 QtMultimedia). Plan step 1 is a spike verifying
  QtMultimedia imports in the pixi env; fallback is a `sounddevice` dep.
- Follows the same play/pause/seek commands as the video worker. Runs on a
  parallel wall-clock schedule, re-locked on every seek/pause.
- When all tracks are muted the engine pushes silence instead of stopping,
  so it remains a valid playback clock.
- Emits `position_changed` derived from sink progress (samples pushed minus
  sink buffer remaining).

### 4b. Clock hierarchy and drift re-sync

- Video frames flowing → video position is master. On each `frames_ready`
  update the editor compares video position vs the engine's played position;
  if |drift| > 100 ms the engine re-seeks its stream to the video position.
- Video worker EOF with playhead < trim_end (videos shorter than the audio
  window) → the editor switches its clock source to the engine's
  `position_changed` until trim_end / loop / next seek. This closes the hole
  where a video-only worker's EOF would cut playback while audio tracks
  extend further.
- Zero-video sessions → engine is the clock from the start.
- Seek transient: audio restarts at the target immediately while the video
  seek is still in flight; the drift corrector snaps audio back once frames
  arrive. Accepted.
- Mute toggle mid-playback restarts the stream (possible small click).
  Accepted for v1.

### 5. Timeline

`TrackState` gains `muted: bool = False`. Each track row gets a small
speaker toggle; a `mute_changed` signal feeds the audio engine. Offset drag
and trim are already track-kind-agnostic — audio tracks get both for free.

### 6. Export dialog

- Track checkbox list, all checked by default. Unchecked tracks are filtered
  out of the `media`/`offsets` lists passed to `export_media` — no backend
  change for selection. OK button disabled when zero tracks are selected.
- Resolution/fps rows are computed over the selected **video** tracks only
  and recomputed when checkboxes change; hidden when no video track is
  selected (also fixes the `None * None` crash on audio-only sessions).
- New audio section (applies to audio-only outputs; video outputs keep aac):
  - Format: `Same as source` (default) / wav / flac / mp3 / m4a
  - Sample rate: `Native` (default) / 44100 / 48000
  - Bitrate: 128/192/256/320 kbps, default 320; enabled only for lossy
    formats (mp3, m4a)
- `get_export_params` returns a small dataclass (the tuple has outgrown
  itself).

### 7. Export backend (small)

- `ExportSettings` gains `audio_sample_rate: int | None = None` and
  `audio_bitrate: int | None = None` (bps).
- Audio-only path: output stream created at the target rate (the existing
  `AudioResampler` converts); `astream.bit_rate` set for lossy codecs.
  The video path ignores both fields.
- `_AUDIO_CODEC_FOR_SUFFIX` gains `.aac → aac`.
  "Same as source" with an unmapped suffix falls back to wav output instead
  of muxing aac into an arbitrary container.

## Error handling

- Probe failures per file: existing warning path in `SyncEditorWindow`
  unchanged.
- QAudioSink open failure (no output device): engine disables itself with a
  log warning; preview stays silent, everything else works.
- Export failures stay per-track (`ExportResult.error`), unchanged.

## Testing

- Backend (pytest, existing `tests/app`): audio export with sample-rate and
  bitrate settings round-trip; codec map fallback for unmapped suffixes;
  export of a filtered (selected) subset.
- GUI logic: mix math (offset slicing, mean mix, mute) and waveform peak
  downsampling extracted as pure functions with unit tests
  (`tests/gui`).
- Qt widgets (mute button, waveform paint, dialog layout): manual
  verification.

## Risks / accepted tradeoffs

| Risk | Resolution |
| --- | --- |
| QtMultimedia missing in pixi env | Spike first; fallback `sounddevice` |
| RAM for playback waveforms | int16 storage; documented limit |
| Mono preview | Accepted; export unaffected |
| Mute click | Accepted v1 |
| Seek transient desync | One-shot drift correction |
