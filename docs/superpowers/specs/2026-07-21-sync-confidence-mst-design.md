# Sync Confidence + MST Solver — Design

**Date:** 2026-07-21
**Status:** Approved (pending final spec review)

## Goal

Detect and survive bad sync inputs. Two deliverables, one mechanism:

1. **Per-track confidence score** so callers can warn when a track is likely
   mis-synced (e.g. a completely unrelated video).
2. **Robust offset solver**: replace the star topology (every track vs one
   reference) with a consistency-weighted minimum-spanning-tree solve over
   the full pairwise graph, per Wonner et al. (Google), including a top-K
   alternate-peak repair pass.

## Background / references

- BBC `audio-offset-finder`: identical pipeline (MFCC cross-correlation),
  ships a "standard score" (z of the correlation peak); documented
  thresholds: >10 likely correct, <5 unlikely.
- Google, *Temporal Synchronization of Multiple Audio Signals* (ICASSP'14):
  pairwise offsets are unreliable one at a time; build the full pairwise
  graph, weight edges by triangle (cycle) consistency
  `z_ijk = |x_ij + x_jk + x_ki|`, solve with an MST (preferred over belief
  propagation — same robustness, less complexity). Resilient to ~20%
  outlier edges.

## Core changes (`core/offsets.py`)

### Pairwise: `find_offset` → rich result

`find_offset` computes the full correlation curve and currently discards
everything but the argmax. New return type:

```python
@dataclass(frozen=True)
class PairAlignment:
    peaks: list[tuple[float, float]]  # (offset_s, score), best first, K<=5
    # peaks[0][0] is the primary offset; peaks[0][1] its confidence.
```

- **Score** = robust standard score of a peak against the curve:
  `(peak − median(corr)) / (1.4826 · MAD(corr))`. Same scale as the BBC
  standard score on clean noise (the 1.4826 factor equates MAD to σ for
  Gaussians), but median/MAD ignores sidelobes (music beats, reverb), so
  correct-but-peaky content is not falsely down-scored.
- **Top-K peaks** (K=5): local maxima of the curve, greedily selected by
  height with a ±0.5 s exclusion window around each accepted peak so lobe
  shoulders are not reported as separate candidates. Each refined with the
  existing parabolic interpolation.
- Cost: argpartition + a few comparisons over an array already in memory.

Existing single-float behavior is kept for external callers via
`peaks[0][0]`; internal callers migrate to `PairAlignment`.

### Multi-track: `align_waveforms` → MST solver

```python
@dataclass(frozen=True)
class Alignment:
    offsets: list[float]          # per track; offsets[reference] == 0.0
    confidence: list[float]       # per track; reference gets +inf
    warnings: list[str]           # human-readable, per flagged track
```

Algorithm:

1. **Pairwise pass**: `find_offset` for all N(N−1)/2 pairs (MFCCs are
   computed once per track and reused — only the FFT cross-correlations
   multiply; they are cheap relative to decode/MFCC).
2. **z-gate**: drop edges whose best peak scores below `EDGE_MIN_SCORE = 5`
   (BBC's "unlikely correct" line). A track left with no edges is
   **isolated**: offset falls back to 0.0, confidence 0.0, warning emitted
   ("no reliable sync found — likely unrelated audio").
3. **Consistency weights**: for each surviving edge,
   `e_ij = Σ_k |x_ij + x_jk + x_ki|` over all triangles with surviving
   edges. An outlier edge wrong by Δ collects ≈ (N−2)·Δ; correct edges
   touching it collect ≈ Δ; the rest ≈ 0 — so the MST routes around liars
   for N≥4.
4. **Alternate-peak repair**: for each high-residual triangle, try
   substituting each member edge's #2..#K peak; if one substitution
   collapses the triangle residual to near zero (< `REPAIR_TOL`, a few hop
   lengths), adopt that peak for the edge and recompute its weight. This
   localizes and fixes the liar even at N=3 (where single-peak consistency
   can detect but not attribute), and recovers repeated-content edges whose
   correct offset is their secondary peak.
5. **MST** (Prim's) over consistency weights; offsets chained along the
   tree, re-based so `offsets[reference_index] == 0.0` (reference semantics
   unchanged).
6. **Per-track confidence**: score of the tree edge that attached the track,
   discounted by that edge's mean triangle residual (converted to the same
   scale). Reference track: +inf. Tracks attached through a repaired or
   low-score edge, or any isolated track, get a warning string.

Degenerate cases:

- **N=2**: one pair, no triangles — solver reduces to today's behavior plus
  the score. Confidence = the pair's peak score.
- **N=3 with one bad edge and no usable alternate peak**: detection without
  attribution — all three tracks warned ("offsets mutually inconsistent —
  verify manually"); MST tie-break by peak score.
- **Consistent collective error** (two tracks matched to the same wrong
  repeat): undetectable by any method; out of scope.

Thresholds as module constants: `EDGE_MIN_SCORE = 5.0`,
`LOW_CONFIDENCE = 10.0`, `REPAIR_TOL` in seconds (derived from hop length).
Calibrated against BBC's documented values; validated by tests below.

Complexity: O(N²) correlations, O(N³) triangle sums — negligible for the
tool's realistic range (≤ ~15 tracks). No cap implemented (YAGNI); the CLI
and GUI are the only entry points and multicam shoots at this scale are the
use case.

## API ripple (all in-repo)

- `align_waveforms(...) -> Alignment` (was `list[float]`).
- `offsets_from_media(...)` / `compute_sync_offsets(...) -> Alignment`
  (was `list[float]`). Callers unpack `.offsets` where they need plain
  lists (export, timeline).
- `sync_and_trim` uses `.offsets`; confidence ignored there beyond logging
  warnings.

## Surfacing

### GUI

- `OffsetWorker` emits the full `Alignment`.
- After sync completes: if any track's confidence < `LOW_CONFIDENCE`, one
  message box listing the flagged tracks ("Low sync confidence — verify
  track X manually").
- Timeline: `TrackState` gains `low_confidence: bool`; flagged rows draw a
  small ⚠ marker, tooltip shows the numeric score. The marker reflects the
  initial sync and is not cleared by manual offset drags.

### CLI

- `sync`: per-track line becomes
  `cam_b.mp4  offset=+1.2345s  confidence=42.1` with a trailing
  `LOW — verify manually` marker under threshold (isolated tracks:
  `confidence=0.0`).
- `synctrim`: warnings printed to stderr before export proceeds.

## Testing

Synthetic fixtures (noise bursts + tones, generated in-test):

1. Shifted copies of the same signal → offsets recovered, confidence ≫ 10.
2. One unrelated-noise track among related ones → isolated, confidence 0,
   warning; other tracks unaffected.
3. Repeated-content trap: signal with an identical repeated segment,
   query clip overlapping one occurrence, constructed so the pairwise peak
   picks the wrong occurrence → MST + repair recovers the correct offset
   (N=3 and N=4 variants).
4. N=2 unrelated pair → low confidence returned (no exception).
5. Regression: existing alignment tests keep passing via `.offsets`.
6. Property: solver output for a clean N-track set equals star output
   within one hop.

## Risks / tradeoffs

| Risk | Resolution |
| --- | --- |
| O(N²) correlations slow huge sessions | Fine ≤ ~15 tracks; revisit if use changes |
| Threshold miscalibration | BBC-derived defaults + synthetic-fixture tests; constants in one place |
| Score grows slowly with length (√(2 ln N)) | Direction = false confidence, margin in threshold 10; documented |
| Repair adopts a coincidentally-consistent peak | REPAIR_TOL tight (few hops); requires near-exact triangle closure |

## Interaction with audio-track-support spec (same date)

Independent features; confidence applies identically to audio-only tracks
(same pipeline). Timeline ⚠ marker and mute button share the track-row
header area — coordinate placement during implementation.
