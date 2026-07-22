# Clap-linking UI — design addendum

**Date:** 2026-07-23
**Branch:** `feat/c3d-mocap-sync`
**Builds on:** `2026-07-22-c3d-mocap-sync-design.md`

## Why

Auto clap detection cannot always be right. Real captures have multiple
clap-like snaps in the c3d (a positioning close vs the real clap; frozen
boot/tail padding) and multiple clap sounds across takes/scenes. No acoustic or
kinematic signal disambiguates them — it needs a human. So the offset must be
set by a human-chosen **association** between one c3d clap movement and one clap
sound, with auto detection reduced to a *proposal* that can be overridden.

## Model

A c3d's offset is defined by a **clap link**: one movement point (local c3d
time) paired with one sound point (shared-timeline time):

    offset_c3d = sound_time - movement_time

Both sides expose **candidates** (auto-detected) plus **custom points** the user
can place anywhere. Auto pre-selects the best pair; the user can re-pair or add
custom points. Setting a link freezes the offset until changed.

## Scope / visibility

- The clap-linking UI appears **only when at least one c3d is loaded**. Pure
  A/V projects show nothing new — MFCC sync as before.
- No scrub-on-hover; the user re-links, then replays the timeline to verify
  (existing playback + previews).

## UI (integrated in the sync editor, one screen)

- **Sound-clap row**: a slim row above the track lanes with a flag per detected
  sound clap (auto proposals) plus any custom points. Click a flag → select it
  as the active sound; playing/scrubbing is done via the normal timeline.
- **c3d movement flags**: on each c3d lane, a flag per detected movement snap
  (auto) plus custom points. Click → select as the active movement.
- **Active link**: the selected movement and sound are highlighted; the c3d lane
  is positioned so the movement flag sits under the sound flag (that *is* the
  offset). A small label shows `c3d <t> ↔ sound <t> → offset <t>` with
  `reset to auto`.
- **Custom points**: `Add clap here` drops a sound point at the playhead;
  `Set c3d clap` drops a movement point at the c3d frame under the playhead.
  Both buttons visible only with a c3d present.
- Re-pair: click a movement flag, click a sound flag → link, c3d slides into
  place, frozen. Verify by replaying the timeline.

## Detection changes

- `detect_clap_motions(top, bottom, rate) -> list[ClapCandidate]`: return the
  ranked movement snaps (each a peak closing velocity above the sharpness gate,
  deduped), not just the best. Frozen-padding regions (constant gap) are still
  searched but the snaps found there rank naturally by velocity; the user picks.
- Sound candidates: the existing per-cluster results (already available).
- Auto link = best sound cluster paired with the highest-velocity movement snap.
  It is only a default.

### Auto-proposal quality (planned refinement)

Reject geometrically-invalid clapperboard motion so the auto proposal is not
fooled by tracking garbage:
- The bottom-arm markers should be **coplanar** (they sit on a flat arm). If
  they do not form a plane at a candidate frame, the tracking is bad — drop it.
- The top-arm markers should be **parallel/coplanar** to the bottom-arm plane
  (both arms are flat). If not, drop it.
- **Debounce**: if most of the last N seconds around a candidate were unreliable
  (bad plane / large residuals), discard the candidate.
This needs the raw per-marker positions (not just centroids) and runs as a
filter/weight over `detect_clap_motions`. It is a proposal-quality improvement;
the manual clap link remains the guarantee.

## Trim change

- The default (common) trim range is computed over **audio/video tracks only** —
  the c3d is excluded because re-linking moves it. Export still applies the
  chosen trim to every track (the c3d is trimmed/padded to it). The user adjusts
  the trim afterward as needed.
- Applies to: the timeline's initial trim (`SyncTrimTimelineWidget.set_tracks`),
  the editor's `_refit_trim`, and `sync_and_trim`'s default range.

## Removed / replaced

- The earlier always-on clickable timeline clap markers (re-anchor by clicking a
  single sound flag using the auto movement time) are replaced by this
  link model, which also lets the user choose the movement and add custom
  points — fixing the boot-vs-real-clap ambiguity the single-marker flow could
  not.

## Staging

1. Core: `detect_clap_motions` (multi), A/V-only trim, expose candidates.
2. GUI: sound-clap row + c3d movement flags + pairing + custom points, gated on
   c3d presence.
3. Verify on the real 4-mic + c3d capture: link c3d 9.4s ↔ sound 54.3s →
   offset 44.8s; confirm preview animation shows the real clap aligned.
