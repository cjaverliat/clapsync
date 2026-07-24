# Measured Audible Clock — Design

Date: 2026-07-24
Status: Superseded by `2026-07-24-qt-native-preview-design.md` — preview moves
to Qt-native playback, deleting the custom clock this fixes.
Scope: Layer 1 of the preview A/V-sync work. Independent of, and shippable
before, the framepipe→Qt mosaic swap (Layer 2, separate spec).

## Problem

In the sync-editor preview, video and the audible sound intermittently drift
apart: "sometimes it's well synced, sometimes it's off." The offset is not a
gap that grows over time (that would be true drift) — it is a roughly fixed
lead whose *size varies from one playback to the next*.

## Root cause (confirmed)

`AudioEngine.position_s()` builds the master playhead clock from
`QAudioSink.processedUSecs()`:

```
played = origin_s + (processedUSecs - base_us) / 1e6
```

`processedUSecs()` counts samples **pulled into the sink's ring buffer**, not
samples the speaker has emitted. So this clock leads the audible sound by
whatever is currently sitting in the buffer. The video decode chases this
clock (`video_player.VideoGroupWorker` → `audio.clock_s`), so the picture is
shown ~one buffer *ahead* of the sound.

The buffer primes by a variable amount at each `QAudioSink.start()` (Windows
WASAPI), so the lead differs run-to-run — exactly the "sometimes synced,
sometimes not" symptom.

A spike (`QAudioSink`, 100 ms buffer, click track) measured the audible
position as `processed − buffered` every 150 ms and confirmed:

- the true audible position sits a steady 60–100 ms **behind** `processed`
  (the lead == the buffer fill), and
- `processed − buffered` is smooth within a run (±5 ms), because the buffered
  term is read live rather than guessed.

## The fix

Compute the clock from the **measured** buffer occupancy instead of guessing
it. QAudioSink exposes `bufferSize()` and `bytesFree()`, so the bytes still
waiting to be played are known exactly at every sample:

```
buffered_bytes   = bufferSize() - bytesFree()
buffered_seconds = buffered_bytes / byte_rate           # byte_rate from format
audible          = pulled_seconds - buffered_seconds
```

Because `buffered_seconds` is *measured*, the per-run prime variance is
subtracted rather than modeled. No latency constant, no calibration step.

### Pure function

Add a pure, unit-testable helper beside `played_position_s` in
`gui/audio_engine.py`:

```python
def audible_position_s(
    origin_s: float,
    processed_us: float,
    buffered_bytes: int,
    byte_rate: int,
    base_us: float = 0.0,
) -> float:
    """Content seconds actually audible, from measured buffer occupancy.

    origin_s is the content time at which the current sink.start() began;
    processed_us is QAudioSink.processedUSecs() since that start; buffered_bytes
    is (bufferSize - bytesFree), the bytes still queued in the sink; byte_rate
    is the format's bytes-per-second. Subtracting the queued bytes converts the
    pull-cursor time into the time the speaker is actually emitting, which —
    unlike the raw cursor — does not lead by the (variably primed) buffer.
    """
    pulled = origin_s + (processed_us - base_us) / 1_000_000.0
    buffered_s = buffered_bytes / byte_rate if byte_rate else 0.0
    return max(origin_s, pulled - buffered_s)
```

`max(origin_s, …)` floors the clock at the play/seek origin: right after
`start()` the buffer fills before any sound is emitted, so the raw
`pulled − buffered` briefly reads *before* origin. Flooring holds the playhead
(and the video) at the origin frame until real sound begins — this matches the
existing "holds at the start until audio begins to move" behaviour the video
worker already expects, and prevents the video seeking backward before origin.

### Wiring

`AudioEngine.position_s()` (the `self._playing` branch) calls the new helper:

```python
buffered = self._sink.bufferSize() - self._sink.bytesFree()
return audible_position_s(
    self._origin_s, self._sink.processedUSecs(),
    buffered, self._byte_rate, self._base_us,
)
```

`self._byte_rate` is derived once from the format in `__init__`
(`fmt.bytesForDuration(1_000_000)`, or
`sampleRate * bytesPerSample * channelCount`), stored so the hot path does no
per-call format lookups.

The `seek()` re-anchor path is unchanged: it still sets `_origin_s` and
`_base_us` from the cursor. The buffered term is read live in `position_s`, so
a mid-play seek self-corrects on the next tick with no extra bookkeeping.

## What does NOT change

- The pull/mix path (`mix_block`, `_pull`, `_cursor`) — audio content and its
  sample-exact offsets are correct already; only the *reported clock* is wrong.
- The video worker, slew clock, and c3d preview — they keep chasing
  `audio.clock_s`; that value is simply now truthful, so they lock correctly.
- The `_BUFFER_MS = 100` sink buffer — with the lead now measured out, the
  buffer size only affects latency headroom, not sync. Left as-is.
- `played_position_s` — kept (still referenced by tests / paused path uses
  `_cursor`); the playing path now routes through `audible_position_s`.

## Residual latency

Beyond the QAudioSink ring buffer there is a fixed device/OS latency (DAC,
system mixer) that `bytesFree()` cannot see. It is *constant per device*, so it
does not cause the run-to-run variance this fixes, and is typically small
(single-digit to low-tens of ms). We deliberately do **not** add a calibration
constant now (YAGNI). If, after shipping, a consistent residual lead remains
audible, a single `AUDIO_HW_LATENCY_S` constant subtracted in the helper is the
follow-up — but only if measurement shows it is needed.

## Testing

Unit (pure, no Qt):

- `audible_position_s` subtracts buffered seconds: e.g. origin 0, processed
  1_000_000 µs, buffered 9600 bytes at 96000 byte/s → 1.0 − 0.1 = 0.9.
- Floor at origin: processed small, buffered large → returns origin, never
  below.
- `byte_rate == 0` guard → no ZeroDivisionError, buffered treated as 0.
- base_us offset (in-play seek anchor) is honoured.
- Monotonic-ish: increasing processed with steady buffered increases the
  result.

Verification in the real app (the `verify` skill):

- Play a mosaic clip with an audible clap; confirm the pictured clap frame and
  the heard clap coincide, and that this holds across repeated play/stop cycles
  (the previous variance is gone). This is the actual reported bug — must be
  observed fixed, not just unit-green.

## Success criteria

1. New pure `audible_position_s` unit tests pass.
2. Existing audio-engine / sync tests still pass.
3. In-app: A/V stays synced across repeated playbacks; the intermittent offset
   no longer appears.
