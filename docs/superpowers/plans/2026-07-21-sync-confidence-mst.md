# Sync Confidence + MST Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-track sync confidence scores plus a consistency-weighted MST solver that survives outlier pairwise matches (spec: `docs/superpowers/specs/2026-07-21-sync-confidence-mst-design.md`).

**Architecture:** `core/offsets.py` keeps all pairwise correlation work and gains top-K peak extraction with a robust standard score. A new pure-math module `core/solver.py` turns the pairwise peak matrix into global offsets via z-gating, triangle-consistency weights, alternate-peak repair, and Prim's MST. `align_waveforms` orchestrates the two and now returns an `Alignment` (offsets + confidence + warnings) that flows through `app.sync` to the CLI and GUI.

**Tech Stack:** Python 3, numpy, torch/torchaudio (existing), pytest via pixi env, PySide6 (GUI task only).

## Global Constraints

- Thresholds (from spec, single source of truth in `core/solver.py`): `EDGE_MIN_SCORE = 5.0`, `LOW_CONFIDENCE = 10.0`, `REPAIR_TOL = 0.05` (seconds).
- Robust score formula: `(peak − median(corr)) / (1.4826 · MAD(corr))`.
- Top-K peaks: `K = 5`, exclusion window `0.5` s.
- Offset sign convention (existing): `find_offset(i_as_ref, j_as_query)` is positive when j leads i; equivalently `x_ij = o_j − o_i` on the shared timeline. Triangle residual `|x_ij + x_jk + x_ki| ≈ 0` for correct edges.
- `find_offset` keeps returning a bare `float` (public API + tests pin it); rich results come from the new `find_offset_peaks`.
- Test runner: `pixi run python -m pytest <path> -v` from the repo root.
- Commit after every task; work on a feature branch `feat/sync-confidence` created from the current branch at the start of Task 1.

---

### Task 1: Robust score + top-K peak extraction (`find_offset_peaks`)

**Files:**
- Modify: `src/clapsync/core/offsets.py`
- Test: `tests/core/test_offsets.py`

**Interfaces:**
- Consumes: existing `_compute_mfcc`, `_mfcc_cross_correlate`, `_parabolic_peak`, `_to_mono_f64`.
- Produces:
  - `@dataclass(frozen=True) PairAlignment` with `peaks: list[tuple[float, float]]` — `(offset_seconds, score)`, best first, at most 5 entries.
  - `find_offset_peaks(ref_waveform, ref_rate, waveform, rate, *, refine="parabolic", n_mfcc=13, n_fft=2048, hop_duration=0.005, win_duration=0.04, n_mels=128, mel_scale="htk") -> PairAlignment`
  - `_peak_candidates(corr: np.ndarray, sub_len: int, hop_length: int, rate: int, refine: Refine) -> list[tuple[float, float]]` (module-private, reused by Task 5)
  - `find_offset(...)` unchanged signature, now implemented as `find_offset_peaks(...).peaks[0][0]`.

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feat/sync-confidence
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/core/test_offsets.py`:

```python
from clapsync.core.offsets import PairAlignment, find_offset_peaks


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_find_offset_peaks_matched_content_scores_high():
    # Same noise content shifted by 0.25 s: the peak must tower over the curve.
    sr = 16000
    base = _noise(sr * 4, seed=1)
    ref = torch.from_numpy(base).unsqueeze(0)
    sub = torch.from_numpy(np.roll(base, int(0.25 * sr))).unsqueeze(0)
    result = find_offset_peaks(ref, sr, sub, sr)
    assert isinstance(result, PairAlignment)
    offset, score = result.peaks[0]
    assert abs(offset - (-0.25)) < 0.02
    assert score > 20.0


def test_find_offset_peaks_unrelated_content_scores_low():
    sr = 16000
    ref = torch.from_numpy(_noise(sr * 4, seed=2)).unsqueeze(0)
    sub = torch.from_numpy(_noise(sr * 4, seed=3)).unsqueeze(0)
    matched = find_offset_peaks(ref, sr, ref.clone(), sr).peaks[0][1]
    unrelated = find_offset_peaks(ref, sr, sub, sr).peaks[0][1]
    assert unrelated < matched / 4
    assert unrelated < 10.0


def test_find_offset_peaks_exclusion_window_separates_candidates():
    # A repeated identical burst yields two peaks at least 0.5 s apart.
    sr = 16000
    burst = _noise(sr // 2, seed=4)
    timeline = _noise(sr * 6, seed=5) * 0.05
    timeline[sr : sr + len(burst)] += burst
    timeline[4 * sr : 4 * sr + len(burst)] += burst
    ref = torch.from_numpy(timeline).unsqueeze(0)
    sub = torch.from_numpy(burst).unsqueeze(0)
    result = find_offset_peaks(ref, sr, sub, sr)
    assert len(result.peaks) >= 2
    offsets = [p[0] for p in result.peaks[:2]]
    assert abs(offsets[0] - offsets[1]) >= 0.5
    # Scores are sorted best-first.
    scores = [p[1] for p in result.peaks]
    assert scores == sorted(scores, reverse=True)


def test_find_offset_still_returns_float():
    sr = 16000
    base = _noise(sr * 2, seed=6)
    ref = torch.from_numpy(base).unsqueeze(0)
    lag = find_offset(ref, sr, ref.clone(), sr)
    assert isinstance(lag, float)
    assert abs(lag) < 0.02
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pixi run python -m pytest tests/core/test_offsets.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'PairAlignment'`; the four pre-existing tests PASS.

- [ ] **Step 4: Implement `PairAlignment`, `_peak_candidates`, `find_offset_peaks`**

In `src/clapsync/core/offsets.py`, add after the imports:

```python
from dataclasses import dataclass

# Peak-candidate extraction. K and the exclusion window bound how many
# distinct correlation lobes are reported per pair; the robust score scale
# matches the BBC audio-offset-finder "standard score" (1.4826*MAD == sigma
# for Gaussian data), so its published 5/10 thresholds apply.
_MAX_PEAKS = 5
_EXCLUSION_S = 0.5


@dataclass(frozen=True)
class PairAlignment:
    """Ranked correlation-peak candidates for one track pair.

    peaks: (offset_seconds, score) tuples, best first. peaks[0][0] is the
    primary offset estimate; peaks[0][1] its robust standard score.
    """

    peaks: list[tuple[float, float]]
```

Add after `_parabolic_peak`:

```python
def _peak_candidates(
    corr: np.ndarray,
    sub_len: int,
    hop_length: int,
    rate: int,
    refine: Refine,
) -> list[tuple[float, float]]:
    """Extract up to _MAX_PEAKS distinct peaks with robust standard scores.

    Score = (peak - median) / (1.4826 * MAD): the BBC-style standard score
    made robust to sidelobes (music beats, reverb) by using median/MAD
    instead of mean/std. Candidates are greedily taken in height order with
    an exclusion window so shoulders of one lobe are not reported twice.
    """
    med = float(np.median(corr))
    mad = float(np.median(np.abs(corr - med)))
    scale = 1.4826 * mad + 1e-12
    exclusion = max(1, round(_EXCLUSION_S * rate / hop_length))

    order = np.argsort(corr)[::-1]
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= _MAX_PEAKS:
            break
        if all(abs(int(idx) - c) >= exclusion for c in chosen):
            chosen.append(int(idx))

    peaks: list[tuple[float, float]] = []
    for idx in chosen:
        pos = _parabolic_peak(corr, idx) if refine == "parabolic" else float(idx)
        lag_hops = pos - (sub_len - 1)
        offset = lag_hops * hop_length / rate
        score = (float(corr[idx]) - med) / scale
        peaks.append((offset, score))
    return peaks
```

Refactor `find_offset` into a thin wrapper and move its body into
`find_offset_peaks` (same parameters and docstring content; only the tail
changes — replace the argmax/lag block with):

```python
    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_sub)
    return PairAlignment(
        _peak_candidates(corr, mfcc_sub.shape[1], hop_length, ref_rate, refine)
    )
```

and:

```python
def find_offset(
    ref_waveform: torch.Tensor,
    ref_rate: int,
    waveform: torch.Tensor,
    rate: int,
    *,
    refine: Refine = "parabolic",
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
) -> float:
    """Primary temporal offset between two waveforms (see find_offset_peaks)."""
    return find_offset_peaks(
        ref_waveform, ref_rate, waveform, rate,
        refine=refine, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_duration=hop_duration, win_duration=win_duration,
        n_mels=n_mels, mel_scale=mel_scale,
    ).peaks[0][0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run python -m pytest tests/core/test_offsets.py -v`
Expected: all PASS. If `test_find_offset_peaks_unrelated_content_scores_low` sits near the 10.0 bound, lengthen the noise tracks to `sr * 6` rather than loosening the assertion.

- [ ] **Step 6: Commit**

```bash
git add src/clapsync/core/offsets.py tests/core/test_offsets.py
git commit -m "feat(core): top-K correlation peaks with robust standard scores"
```

---

### Task 2: Solver module — gating, isolation, `Alignment`

**Files:**
- Create: `src/clapsync/core/solver.py`
- Test: `tests/core/test_solver.py`

**Interfaces:**
- Consumes: nothing from other tasks (pure math; operates on plain peak lists).
- Produces:
  - Constants `EDGE_MIN_SCORE = 5.0`, `LOW_CONFIDENCE = 10.0`, `REPAIR_TOL = 0.05`.
  - `@dataclass(frozen=True) Alignment` with `offsets: list[float]`, `confidence: list[float]`, `warnings: list[str]`.
  - `solve_offsets(n: int, pairs: dict[tuple[int, int], list[tuple[float, float]]], reference_index: int = 0) -> Alignment` — `pairs[(i, j)]` (always `i < j`) is the peak list `(offset_s, score)` best-first, with `offset = o_j − o_i`.
- Note: this task implements gating, MST over score tie-breaks, chaining, isolation, and confidence; consistency weights arrive in Task 3 and repair in Task 4 (the functions are structured so those tasks slot in).

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_solver.py`:

```python
import math

from clapsync.core.solver import (
    EDGE_MIN_SCORE,
    LOW_CONFIDENCE,
    Alignment,
    solve_offsets,
)


def _pair(offset: float, score: float = 50.0) -> list[tuple[float, float]]:
    return [(offset, score)]


def test_two_tracks_clean_pair():
    result = solve_offsets(2, {(0, 1): _pair(0.5)})
    assert isinstance(result, Alignment)
    assert result.offsets == [0.0, 0.5]
    assert math.isinf(result.confidence[0])
    assert result.confidence[1] == 50.0
    assert result.warnings == []


def test_two_tracks_gated_edge_isolates_track():
    result = solve_offsets(2, {(0, 1): _pair(0.5, score=EDGE_MIN_SCORE - 1)})
    assert result.offsets == [0.0, 0.0]
    assert result.confidence[1] == 0.0
    assert any("unrelated" in w for w in result.warnings)


def test_low_score_edge_warns_but_uses_offset():
    result = solve_offsets(2, {(0, 1): _pair(0.5, score=7.0)})
    assert result.offsets == [0.0, 0.5]
    assert result.confidence[1] == 7.0
    assert any("low sync confidence" in w for w in result.warnings)


def test_three_tracks_chain_through_strongest_edges():
    # 0-1 and 1-2 strong; 0-2 gated out. Offsets must chain: o2 = x01 + x12.
    pairs = {
        (0, 1): _pair(1.0),
        (1, 2): _pair(0.5),
        (0, 2): _pair(9.9, score=2.0),  # gated
    }
    result = solve_offsets(3, pairs)
    assert result.offsets[1] == 1.0
    assert abs(result.offsets[2] - 1.5) < 1e-9


def test_nonzero_reference_index_rebases():
    result = solve_offsets(2, {(0, 1): _pair(0.5)}, reference_index=1)
    assert result.offsets == [-0.5, 0.0]
    assert math.isinf(result.confidence[1])


def test_single_track():
    result = solve_offsets(1, {})
    assert result.offsets == [0.0]
    assert math.isinf(result.confidence[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run python -m pytest tests/core/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clapsync.core.solver'`.

- [ ] **Step 3: Implement the solver skeleton**

Create `src/clapsync/core/solver.py`:

```python
"""Consistency-weighted MST solve over pairwise alignment candidates.

Implements the multi-signal synchronization approach of "Temporal
Synchronization of Multiple Audio Signals" (Google, ICASSP 2014): pairwise
offset hypotheses form a graph; edges are gated by peak score, weighted by
triangle (cycle) consistency, repaired via alternate correlation peaks, and
solved with a minimum spanning tree rooted at the reference track.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

# Score thresholds follow the BBC audio-offset-finder calibration for the
# same MFCC cross-correlation pipeline: a standard score below 5 is unlikely
# to be a real match; below 10 deserves a manual check.
EDGE_MIN_SCORE = 5.0
LOW_CONFIDENCE = 10.0
# Triangle closure tolerance. Correct edges close to sub-hop precision
# (~5 ms); outliers miss by whole repeats (seconds).
REPAIR_TOL = 0.05

Peaks = list[tuple[float, float]]  # (offset_seconds, score), best first


@dataclass(frozen=True)
class Alignment:
    """Solved per-track offsets with confidence and human-readable warnings."""

    offsets: list[float]
    confidence: list[float]
    warnings: list[str]


@dataclass
class _Edge:
    i: int
    j: int
    offset: float  # o_j - o_i, seconds
    score: float
    peaks: Peaks   # gated candidates, best first (offset/score mirror [0])


def _gated_edges(
    pairs: dict[tuple[int, int], Peaks],
) -> dict[tuple[int, int], _Edge]:
    edges: dict[tuple[int, int], _Edge] = {}
    for (i, j), peaks in pairs.items():
        usable = [(off, s) for off, s in peaks if s >= EDGE_MIN_SCORE]
        if usable:
            off, s = usable[0]
            edges[(i, j)] = _Edge(i, j, off, s, usable)
    return edges


def _signed_offset(edges, a: int, b: int) -> float | None:
    """x_ab = o_b - o_a from whichever orientation the edge is stored in."""
    if (a, b) in edges:
        return edges[(a, b)].offset
    if (b, a) in edges:
        return -edges[(b, a)].offset
    return None


def _residual(edges, i: int, j: int, k: int) -> float | None:
    """|x_ij + x_jk + x_ki| for one triangle; None if any edge is missing."""
    xij = _signed_offset(edges, i, j)
    xjk = _signed_offset(edges, j, k)
    xki = _signed_offset(edges, k, i)
    if xij is None or xjk is None or xki is None:
        return None
    return abs(xij + xjk + xki)


def _consistency_weights(
    n: int, edges: dict[tuple[int, int], _Edge]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Sum of triangle residuals per edge (Google eq. 5-6), plus counts."""
    weights = {key: 0.0 for key in edges}
    counts = {key: 0 for key in edges}
    for i, j, k in itertools.combinations(range(n), 3):
        res = _residual(edges, i, j, k)
        if res is None:
            continue
        for a, b in ((i, j), (j, k), (i, k)):
            key = (a, b) if (a, b) in edges else (b, a)
            weights[key] += res
            counts[key] += 1
    return weights, counts


def solve_offsets(
    n: int,
    pairs: dict[tuple[int, int], Peaks],
    reference_index: int = 0,
) -> Alignment:
    """Solve global offsets from pairwise peak candidates.

    Args:
        n: Number of tracks.
        pairs: (i, j) with i < j -> peak candidates, offset = o_j - o_i.
        reference_index: Track pinned to offset 0.0.

    Returns:
        Alignment with offsets[reference_index] == 0.0, per-track
        confidence (reference gets +inf, isolated tracks 0.0), and warnings.
    """
    offsets = [0.0] * n
    confidence = [0.0] * n
    confidence[reference_index] = math.inf
    warnings: list[str] = []
    if n == 1:
        return Alignment(offsets, confidence, warnings)

    edges = _gated_edges(pairs)
    weights, counts = _consistency_weights(n, edges)

    adjacency: dict[int, list[tuple[int, tuple[int, int]]]] = {
        i: [] for i in range(n)
    }
    for (i, j) in edges:
        adjacency[i].append((j, (i, j)))
        adjacency[j].append((i, (i, j)))

    # Prim's MST from the reference; consistency weight first, higher peak
    # score as tie-break (relevant when no triangles exist, e.g. n == 2).
    visited = {reference_index}
    attached_by: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, float, int, int, tuple[int, int]]] = []

    def push(node: int) -> None:
        for other, key in adjacency[node]:
            if other not in visited:
                heapq.heappush(
                    heap, (weights[key], -edges[key].score, other, node, key)
                )

    push(reference_index)
    while heap:
        _, _, node, parent, key = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        x = _signed_offset(edges, parent, node)
        offsets[node] = offsets[parent] + x
        attached_by[node] = key
        push(node)

    for t in range(n):
        if t == reference_index:
            continue
        if t not in visited:
            warnings.append(
                f"track {t}: no reliable sync found — likely unrelated audio"
            )
            continue
        key = attached_by[t]
        conf = edges[key].score
        mean_res = weights[key] / counts[key] if counts[key] else 0.0
        if mean_res > REPAIR_TOL:
            conf = min(conf, LOW_CONFIDENCE - 0.1)
            warnings.append(
                f"track {t}: offsets mutually inconsistent — verify manually"
            )
        elif conf < LOW_CONFIDENCE:
            warnings.append(
                f"track {t}: low sync confidence — verify manually"
            )
        confidence[t] = conf

    return Alignment(offsets, confidence, warnings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run python -m pytest tests/core/test_solver.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/solver.py tests/core/test_solver.py
git commit -m "feat(core): gated MST offset solver with per-track confidence"
```

---

### Task 3: Consistency weights route around outlier edges

**Files:**
- Modify: `src/clapsync/core/solver.py` (no code change expected — this task pins the behavior with tests; fix the solver if they fail)
- Test: `tests/core/test_solver.py`

**Interfaces:**
- Consumes: `solve_offsets` from Task 2.
- Produces: verified outlier-routing behavior later tasks and the spec rely on.

- [ ] **Step 1: Write the tests**

Append to `tests/core/test_solver.py`:

```python
def _clean_pairs(offsets: list[float], score: float = 50.0):
    """All-pairs peak lists consistent with the given global offsets."""
    n = len(offsets)
    return {
        (i, j): [(offsets[j] - offsets[i], score)]
        for i in range(n)
        for j in range(i + 1, n)
    }


def test_outlier_edge_routed_around_at_n4():
    true = [0.0, 1.0, 2.0, 3.0]
    pairs = _clean_pairs(true)
    # Edge 1-2 lies by 180 s with a *high* score (repeated-content trap).
    pairs[(1, 2)] = [(true[2] - true[1] + 180.0, 80.0)]
    result = solve_offsets(4, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6
    # The lie never contaminates offsets; consistency weight kept it off the tree.


def test_n3_single_bad_edge_detected_not_silent():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    pairs[(1, 2)] = [(true[2] - true[1] + 180.0, 80.0)]  # no alternate peak
    result = solve_offsets(3, pairs)
    # One triangle, all edges share the blame: every non-reference track
    # must carry an inconsistency warning.
    assert sum("inconsistent" in w for w in result.warnings) >= 1


def test_clean_n4_no_warnings_high_confidence():
    result = solve_offsets(4, _clean_pairs([0.0, -0.5, 1.25, 3.0]))
    assert result.warnings == []
    assert all(c >= LOW_CONFIDENCE for i, c in enumerate(result.confidence) if i != 0)
```

- [ ] **Step 2: Run the tests**

Run: `pixi run python -m pytest tests/core/test_solver.py -v`
Expected: all PASS with the Task 2 implementation (weights already exist). If `test_outlier_edge_routed_around_at_n4` fails, the bug is in `_consistency_weights` or the heap ordering — fix there, do not loosen the test.

- [ ] **Step 3: Commit**

```bash
git add tests/core/test_solver.py
git commit -m "test(core): pin outlier-edge routing and N=3 detection"
```

---

### Task 4: Alternate-peak repair

**Files:**
- Modify: `src/clapsync/core/solver.py`
- Test: `tests/core/test_solver.py`

**Interfaces:**
- Consumes: `_Edge`, `_residual`, `_gated_edges` from Task 2.
- Produces: `_repair_pass(n: int, edges: dict[tuple[int, int], _Edge]) -> None` (mutates edges in place), called from `solve_offsets` between gating and weight computation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_solver.py`:

```python
def test_repair_adopts_alternate_peak_at_n3():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    # Edge 1-2's best peak is the wrong repeat (+180 s) but the correct
    # offset survives as its second peak.
    pairs[(1, 2)] = [
        (true[2] - true[1] + 180.0, 80.0),
        (true[2] - true[1], 60.0),
    ]
    result = solve_offsets(3, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6
    assert not any("inconsistent" in w for w in result.warnings)


def test_repair_adopts_alternate_peak_at_n4():
    true = [0.0, 1.0, 2.0, 3.0]
    pairs = _clean_pairs(true)
    pairs[(0, 3)] = [
        (true[3] - true[0] - 42.0, 70.0),
        (true[3] - true[0], 55.0),
    ]
    result = solve_offsets(4, pairs)
    for got, want in zip(result.offsets, true):
        assert abs(got - want) < 1e-6


def test_repair_skips_when_no_alternate_closes_triangle():
    true = [0.0, 1.0, 2.0]
    pairs = _clean_pairs(true)
    pairs[(1, 2)] = [
        (true[2] - true[1] + 180.0, 80.0),
        (true[2] - true[1] + 90.0, 60.0),  # also wrong
    ]
    result = solve_offsets(3, pairs)
    assert any("inconsistent" in w for w in result.warnings)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pixi run python -m pytest tests/core/test_solver.py -v`
Expected: the two `repair_adopts` tests FAIL (offsets off by the lie at n=3; n=4 passes offsets via routing but this test additionally passes only once repair exists — if it passes already, routing covered it; keep it as a regression pin). `test_repair_skips_when_no_alternate_closes_triangle` may already pass.

- [ ] **Step 3: Implement `_repair_pass`**

Add to `src/clapsync/core/solver.py` after `_consistency_weights`:

```python
def _repair_pass(n: int, edges: dict[tuple[int, int], _Edge]) -> None:
    """Close high-residual triangles by adopting alternate peaks in place.

    For each triangle whose residual exceeds REPAIR_TOL, every gated
    alternate peak of each member edge is tried; the substitution leaving
    the smallest residual under REPAIR_TOL wins (higher alternate score
    breaks ties). Sweeps repeat until nothing is adopted; the sweep count
    is capped to guard against pathological ping-pong between triangles.
    """
    for _ in range(max(1, len(edges))):
        adopted = False
        for i, j, k in itertools.combinations(range(n), 3):
            res = _residual(edges, i, j, k)
            if res is None or res <= REPAIR_TOL:
                continue
            best: tuple[float, float, tuple[int, int], float, float] | None = None
            for a, b in ((i, j), (j, k), (i, k)):
                key = (a, b) if (a, b) in edges else (b, a)
                if key not in edges:
                    continue
                edge = edges[key]
                saved = (edge.offset, edge.score)
                for off, s in edge.peaks[1:]:
                    edge.offset, edge.score = off, s
                    r = _residual(edges, i, j, k)
                    if r is not None and r <= REPAIR_TOL:
                        cand = (r, -s, key, off, s)
                        if best is None or cand < best:
                            best = cand
                edge.offset, edge.score = saved
            if best is not None:
                _, _, key, off, s = best
                edges[key].offset = off
                edges[key].score = s
                adopted = True
        if not adopted:
            return
```

Wire it into `solve_offsets` — immediately after `edges = _gated_edges(pairs)`:

```python
    _repair_pass(n, edges)
```

- [ ] **Step 4: Run all solver tests**

Run: `pixi run python -m pytest tests/core/test_solver.py -v`
Expected: all PASS, including Tasks 2-3 regressions.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/solver.py tests/core/test_solver.py
git commit -m "feat(core): alternate-peak triangle repair in offset solver"
```

---

### Task 5: `align_waveforms` returns `Alignment`; pairwise orchestration

**Files:**
- Modify: `src/clapsync/core/offsets.py`, `src/clapsync/core/__init__.py`
- Test: `tests/core/test_align.py`, `tests/core/test_public_api.py`

**Interfaces:**
- Consumes: `_peak_candidates`, `_compute_mfcc`, `_mfcc_cross_correlate` (Task 1); `solve_offsets`, `Alignment` (Tasks 2-4).
- Produces: `align_waveforms(waveforms, rates, *, refine="parabolic", reference_index=0, progress=None) -> Alignment` (was `list[float]`). `clapsync.core` re-exports `Alignment`, `PairAlignment`, `LOW_CONFIDENCE`.

- [ ] **Step 1: Update and extend the tests**

Replace `tests/core/test_align.py` with:

```python
import numpy as np
import torch

from clapsync.core.offsets import align_waveforms
from clapsync.core.solver import LOW_CONFIDENCE, Alignment


def _click_track(n: int, click_at: int) -> torch.Tensor:
    x = np.zeros(n, dtype=np.float32)
    x[click_at] = 1.0
    return torch.from_numpy(x).unsqueeze(0)


def _noise_track(n: int, seed: int) -> torch.Tensor:
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def test_align_waveforms_reference_zero_and_signed_offsets():
    sr = 48000
    n = sr * 2
    ref = _click_track(n, click_at=sr)                       # 1.000 s
    later = _click_track(n, click_at=sr + int(0.20 * sr))    # +200 ms (delayed)
    earlier = _click_track(n, click_at=sr - int(0.10 * sr))  # -100 ms (leads)

    result = align_waveforms([ref, later, earlier], [sr, sr, sr])
    assert isinstance(result, Alignment)
    offsets = result.offsets
    assert offsets[0] == 0.0
    assert abs(offsets[1] - (-0.20)) < 0.02   # delayed -> negative
    assert abs(offsets[2] - (0.10)) < 0.02    # leads -> positive


def test_align_waveforms_progress_reaches_one():
    sr = 48000
    n = sr
    tracks = [_click_track(n, click_at=n // 2) for _ in range(3)]
    seen = []
    align_waveforms(tracks, [sr, sr, sr], progress=seen.append)
    assert seen and seen[-1] == 1.0


def test_related_tracks_have_high_confidence():
    sr = 16000
    base = np.random.default_rng(7).standard_normal(sr * 4).astype(np.float32)
    a = torch.from_numpy(base).unsqueeze(0)
    b = torch.from_numpy(np.roll(base, int(0.3 * sr))).unsqueeze(0)
    c = torch.from_numpy(np.roll(base, int(-0.2 * sr))).unsqueeze(0)
    result = align_waveforms([a, b, c], [sr, sr, sr])
    assert all(conf > LOW_CONFIDENCE for conf in result.confidence[1:])
    assert result.warnings == []


def test_unrelated_track_isolated_with_warning():
    sr = 16000
    base = np.random.default_rng(8).standard_normal(sr * 6).astype(np.float32)
    a = torch.from_numpy(base).unsqueeze(0)
    b = torch.from_numpy(np.roll(base, int(0.3 * sr))).unsqueeze(0)
    stranger = _noise_track(sr * 6, seed=9)
    result = align_waveforms([a, b, stranger], [sr, sr, sr])
    assert result.confidence[2] < LOW_CONFIDENCE
    assert any("track 2" in w for w in result.warnings)
    # The related pair is unaffected by the stranger.
    assert abs(result.offsets[1] - (-0.3)) < 0.02


def test_repeated_content_trap_repaired():
    """A clip matching the louder wrong repeat is fixed via triangle repair.

    ref carries a burst at 1 s and a LOUDER copy at 4 s, so the clip's best
    correlation peak is the wrong occurrence; witness carries the burst once
    (attenuated repeat), giving unambiguous edges that expose the lie. The
    clip's correct offset survives as its second peak and must be adopted.
    """
    sr = 16000
    rng = np.random.default_rng(10)
    burst = rng.standard_normal(sr).astype(np.float32)
    bed_ref = rng.standard_normal(sr * 6).astype(np.float32) * 0.05
    bed_wit = rng.standard_normal(sr * 6).astype(np.float32) * 0.05

    ref_arr = bed_ref.copy()
    ref_arr[sr : 2 * sr] += burst
    ref_arr[4 * sr : 5 * sr] += 1.5 * burst          # louder wrong repeat

    wit_arr = bed_wit.copy()
    wit_arr[sr : 2 * sr] += burst                     # correct occurrence only

    clip = torch.from_numpy(burst).unsqueeze(0)       # 1 s clip of the burst
    ref = torch.from_numpy(ref_arr).unsqueeze(0)
    wit = torch.from_numpy(wit_arr).unsqueeze(0)

    result = align_waveforms([ref, wit, clip], [sr, sr, sr])
    # Correct clip offset: its content is at t=1s in ref -> o_clip = +1.0
    # (shared = local + offset; clip local 0 aligns with shared 1.0).
    assert abs(result.offsets[2] - 1.0) < 0.05
```

- [ ] **Step 2: Run tests to verify the new behavior fails**

Run: `pixi run python -m pytest tests/core/test_align.py -v`
Expected: FAIL — `align_waveforms` still returns `list[float]` (`AttributeError: 'list' object has no attribute 'offsets'` or import error on `solver`).

- [ ] **Step 3: Reimplement `align_waveforms`**

In `src/clapsync/core/offsets.py`, add `from clapsync.core.solver import Alignment, solve_offsets` to the imports and replace `align_waveforms`:

```python
def align_waveforms(
    waveforms: list[torch.Tensor],
    rates: list[int],
    *,
    refine: Refine = "parabolic",
    reference_index: int = 0,
    progress: Callable[[float], None] | None = None,
) -> Alignment:
    """Align tracks via an all-pairs MFCC correlation graph and MST solve.

    Every pair is correlated (MFCCs are computed once per track); edges are
    score-gated, triangle-consistency weighted, repaired from alternate
    peaks, and solved with an MST rooted at the reference (see
    clapsync.core.solver).

    Args:
        waveforms: Per-track audio tensors.
        rates: Per-track sample rate in Hz (parallel to waveforms).
        refine: Peak refinement ("parabolic" or "none").
        reference_index: Track whose timeline is the origin (offset 0.0).
        progress: Optional 0..1 callback (spans the pairwise correlations).

    Returns:
        Alignment; offsets[reference_index] == 0.0, confidence per track
        (+inf for the reference, 0.0 for isolated tracks), warnings for
        tracks needing manual verification.
    """
    n = len(waveforms)
    if n == 1:
        if progress is not None:
            progress(1.0)
        return Alignment([0.0], [float("inf")], [])

    rate = rates[reference_index]
    hop_length = int(rate * 0.005)
    win_length = int(rate * 0.04)

    monos = []
    for wave, r in zip(waveforms, rates):
        if r != rate:
            wave = AF.resample(wave, orig_freq=r, new_freq=rate)
        monos.append(_to_mono_f64(wave))
    mfccs = [
        _compute_mfcc(m, rate, 13, 2048, hop_length, win_length, 128, "htk")
        for m in monos
    ]

    pairs: dict[tuple[int, int], list[tuple[float, float]]] = {}
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            corr = _mfcc_cross_correlate(mfccs[i], mfccs[j])
            pairs[(i, j)] = _peak_candidates(
                corr, mfccs[j].shape[1], hop_length, rate, refine
            )
            done += 1
            if progress is not None:
                progress(done / total)

    return solve_offsets(n, pairs, reference_index)
```

Update `src/clapsync/core/__init__.py`:

```python
"""clapsync pure core: MFCC audio sync and time-range math (no I/O)."""
from clapsync.core.offsets import (
    PairAlignment,
    Refine,
    align_waveforms,
    find_offset,
    find_offset_peaks,
)
from clapsync.core.solver import LOW_CONFIDENCE, Alignment
from clapsync.core.timerange import (
    TimeRange,
    common_time_range,
    full_time_range,
)

__all__ = [
    "align_waveforms",
    "find_offset",
    "find_offset_peaks",
    "Alignment",
    "PairAlignment",
    "LOW_CONFIDENCE",
    "TimeRange",
    "common_time_range",
    "full_time_range",
    "Refine",
]
```

Extend `tests/core/test_public_api.py`'s import block with `Alignment, find_offset_peaks` from `clapsync.core`.

- [ ] **Step 4: Run the core suite**

Run: `pixi run python -m pytest tests/core -v`
Expected: all PASS. `test_repeated_content_trap_repaired` is the fragile one — if the wrong peak does not dominate, raise the loud repeat's gain (1.5 → 2.0); if the correct second peak gets gated, lengthen the burst.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/core/offsets.py src/clapsync/core/__init__.py tests/core
git commit -m "feat(core): align_waveforms solves the full pairwise graph, returns Alignment"
```

---

### Task 6: App layer propagates `Alignment`

**Files:**
- Modify: `src/clapsync/app/sync.py`, `src/clapsync/app/export.py`
- Test: `tests/app/test_sync.py`

**Interfaces:**
- Consumes: `Alignment` from `clapsync.core.solver`; `align_waveforms` (Task 5).
- Produces: `offsets_from_media(...) -> Alignment`, `compute_sync_offsets(...) -> Alignment` (were `list[float]`). `sync_and_trim` unchanged externally; it unpacks `.offsets` and logs `.warnings`.

- [ ] **Step 1: Update the tests**

In `tests/app/test_sync.py`:
- add `from clapsync.core.solver import Alignment` to the imports;
- in `test_reference_offset_is_zero_and_others_signed`, replace `fake_align` and the assertions:

```python
    def fake_align(waveforms, rates, *, refine="parabolic", reference_index=0, progress=None):
        return Alignment([0.0, 0.5, -0.3], [float("inf"), 42.0, 37.0], [])
```

```python
    alignment = compute_sync_offsets(paths)
    offsets = alignment.offsets
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 0.5) < 1e-9
    assert abs(offsets[2] + 0.3) < 1e-9
    assert alignment.confidence[1] == 42.0
```

- in `test_progress_reaches_one`, replace `fake_align`'s return with `Alignment([0.0, 0.1], [float("inf"), 50.0], [])` (keep its `progress(1.0)` call).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run python -m pytest tests/app/test_sync.py -v`
Expected: FAIL — `compute_sync_offsets` still returns a list (`AttributeError: 'list' object has no attribute 'offsets'`).

- [ ] **Step 3: Update the app layer**

`src/clapsync/app/sync.py`:
- import `Alignment` from `clapsync.core.solver`;
- change both return annotations `-> list[float]` to `-> Alignment` and the docstring return sections to "Alignment (offsets, per-track confidence, warnings); offsets[reference_index] == 0.0";
- the cancelled/early-return paths return `Alignment([0.0] * n, [0.0] * n, [])` instead of `[0.0] * n`;
- the final `return align_waveforms(...)` lines are already correct (it now returns `Alignment`).

`src/clapsync/app/export.py`, in `sync_and_trim`, replace:

```python
    offsets = offsets_from_media(
        media,
        reference_index=reference_index,
        refine=refine,
        progress=sync_progress,
        is_cancelled=is_cancelled,
    )
```

with:

```python
    alignment = offsets_from_media(
        media,
        reference_index=reference_index,
        refine=refine,
        progress=sync_progress,
        is_cancelled=is_cancelled,
    )
    for warning in alignment.warnings:
        logger.warning("sync: %s", warning)
    offsets = alignment.offsets
```

- [ ] **Step 4: Run the app suite**

Run: `pixi run python -m pytest tests/app -v -m "not slow"`
Expected: all PASS (export/media/decode tests untouched by the type change).

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/app/sync.py src/clapsync/app/export.py tests/app/test_sync.py
git commit -m "feat(app): propagate Alignment (confidence + warnings) through sync"
```

---

### Task 7: CLI prints confidence

**Files:**
- Modify: `src/clapsync/cli.py`
- Test: `tests/app/test_cli.py`

**Interfaces:**
- Consumes: `compute_sync_offsets -> Alignment` (Task 6), `LOW_CONFIDENCE` from `clapsync.core`.
- Produces: `sync` output line format `"{name}\toffset={off:+.4f}s\tconfidence={score:.1f}"` with optional trailing `"  LOW — verify manually"`; reference prints `confidence=ref`.

- [ ] **Step 1: Update the test**

In `tests/app/test_cli.py::test_cli_sync_prints_offsets`, add after the existing assertion:

```python
    assert "confidence" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run python -m pytest tests/app/test_cli.py::test_cli_sync_prints_offsets -v`
Expected: FAIL on the new assertion (slow test; generates two tiny videos).

- [ ] **Step 3: Update the CLI**

In `src/clapsync/cli.py`: add `import math` and `from clapsync.core import LOW_CONFIDENCE`; in `_cmd_sync` replace the offsets computation and print loop:

```python
    alignment = compute_sync_offsets(
        args.inputs,
        reference_index=args.reference,
        refine=args.refine,
        progress=lambda f: print(
            f"\rsync {f * 100:3.0f}%", end="", file=sys.stderr
        ),
    )
    offsets = alignment.offsets
    print(file=sys.stderr)
    durations = [probe(p).duration for p in args.inputs]
    common = common_time_range(durations, offsets)
    full = full_time_range(durations, offsets)
    for path, off, conf in zip(args.inputs, offsets, alignment.confidence):
        conf_str = "ref" if math.isinf(conf) else f"{conf:.1f}"
        low = "  LOW — verify manually" if conf < LOW_CONFIDENCE else ""
        print(f"{path.name}\toffset={off:+.4f}s\tconfidence={conf_str}{low}")
```

(The `common range` / `full range` prints stay as they are.)

In `_cmd_synctrim`, no change — `sync_and_trim` already logs warnings (Task 6); configure logging is out of scope.

- [ ] **Step 4: Run the CLI tests**

Run: `pixi run python -m pytest tests/app/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clapsync/cli.py tests/app/test_cli.py
git commit -m "feat(cli): print per-track sync confidence with LOW marker"
```

---

### Task 8: GUI surfaces warnings and marks low-confidence tracks

**Files:**
- Modify: `src/clapsync/gui/workers.py`, `src/clapsync/gui/app.py`, `src/clapsync/gui/sync_editor.py`

**Interfaces:**
- Consumes: `Alignment`, `LOW_CONFIDENCE` from `clapsync.core`; `compute_sync_offsets -> Alignment` (Task 6).
- Produces: `compute_offsets_with_progress(paths, parent=None) -> Alignment | None`; `SyncEditorWindow(..., low_confidence: list[bool] | None = None)`; flagged tracks get a `"⚠ "` label prefix on the timeline (numeric scores live in the post-sync warning dialog — a deliberate simplification of the spec's per-row tooltip).

- [ ] **Step 1: Update `workers.py`**

`OffsetWorker.finished` is already `Signal(list)`; change it to `Signal(object)` and emit the `Alignment` directly:

```python
class OffsetWorker(QObject):
    progress_value = Signal(int)  # 0..1000
    status = Signal(str)  # human-readable current stage
    finished = Signal(object)  # Alignment
    failed = Signal(str)
```

```python
    @Slot()
    def run(self) -> None:
        try:
            alignment = compute_sync_offsets(
                self._paths,
                progress=lambda f: self.progress_value.emit(int(f * 1000)),
                status=lambda s: self.status.emit(s),
            )
            self.finished.emit(alignment)
        except Exception as exc:  # noqa: BLE001
            logger.exception("offset worker failed")
            self.failed.emit(str(exc))
```

Update `compute_offsets_with_progress`'s return annotation to `Alignment | None` and its docstring ("Computed alignment, or None…"); add `from clapsync.core import Alignment` to the imports. `_run_with_progress` already passes payloads through untyped.

- [ ] **Step 2: Update `gui/app.py`**

Replace the offsets block in `main`:

```python
    alignment = compute_offsets_with_progress(video_paths)
    if alignment is None:
        sys.exit(0)
    offsets = alignment.offsets

    if alignment.warnings:
        from clapsync.core import LOW_CONFIDENCE  # noqa: PLC0415 — with the other gui imports at top

        lines = "\n".join(f"• {w}" for w in alignment.warnings)
        scores = "\n".join(
            f"• {p.name}: confidence {c:.1f}"
            for p, c in zip(video_paths, alignment.confidence)
            if c < LOW_CONFIDENCE
        )
        QMessageBox.warning(
            None,
            "Low sync confidence",
            f"Automatic sync may be wrong for some tracks:\n\n{lines}\n\n"
            f"Scores (below {LOW_CONFIDENCE:g} needs manual verification):\n{scores}",
        )
```

(Put the `LOW_CONFIDENCE` and `QMessageBox` imports at the top of the file with the other imports; shown inline here for locality.) Then pass flags to the editor:

```python
    low = [c < LOW_CONFIDENCE for c in alignment.confidence]
    window = SyncEditorWindow(
        video_paths=video_paths, offsets=offsets, use_proxies=args.use_proxies,
        low_confidence=low,
    )
```

- [ ] **Step 3: Update `sync_editor.py`**

Constructor gains `low_confidence: list[bool] | None = None` (after `use_proxies`), stored as `self._low_confidence = list(low_confidence or [])`. In `_init_timeline`, prefix flagged labels:

```python
        tracks = [
            TrackState(
                index=i,
                label=(
                    "⚠ " if i < len(self._low_confidence) and self._low_confidence[i] else ""
                ) + info.path.stem,
                offset_s=self._offsets[i],
                duration_s=info.duration,
                locked=i == 0,
            )
            for i, info in enumerate(self._video_infos)
        ]
```

Known pre-existing caveat (do not fix here): if a probe fails, `_video_infos` shrinks while `offsets`/`low_confidence` keep original indices — the flags follow the same positional convention offsets already use.

- [ ] **Step 4: Manual smoke test**

Run: `pixi run python -m clapsync.gui.app` (or the project's GUI entry point per `pyproject.toml`), select 2+ known-related videos → no warning dialog, no ⚠ prefixes. Then select related videos plus one unrelated clip → warning dialog lists the unrelated track with confidence 0.0; its timeline row shows the ⚠ prefix; offsets of the related tracks unchanged.

- [ ] **Step 5: Run the full non-slow suite**

Run: `pixi run python -m pytest -m "not slow"`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clapsync/gui/workers.py src/clapsync/gui/app.py src/clapsync/gui/sync_editor.py
git commit -m "feat(gui): warn on low sync confidence, mark flagged timeline rows"
```

---

### Task 9: Full-suite verification + real-footage acceptance

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite including slow tests**

Run: `pixi run python -m pytest`
Expected: all PASS (slow CLI/export tests included).

- [ ] **Step 2: Real-footage CLI check (needs user's files)**

Ask the user for 2-3 of their real multicam files, then:

Run: `pixi run clapsync-cli sync <file1> <file2> [file3]`
Expected: offsets match the pre-change behavior (compare against a `git stash` run or known-good values), every confidence well above 10, no LOW markers. Then repeat with one deliberately unrelated file added — expect a LOW/isolated warning for it and unchanged offsets for the related pair.

- [ ] **Step 3: Real-footage GUI check**

Launch the GUI with the same files; confirm the warning dialog and ⚠ marker for the unrelated file, absence of both for the clean set.

- [ ] **Step 4: Merge decision**

Use superpowers:finishing-a-development-branch (options: merge to the base branch, PR, or hold).
```

## Self-review notes

- Spec coverage: PairAlignment/top-K/robust score → Task 1; gating/isolation/Alignment → Task 2; consistency+MST routing → Tasks 2-3; repair → Task 4; align_waveforms orchestration + integration fixtures (spec tests 1-6) → Task 5; app propagation + sync_and_trim warnings → Task 6; CLI → Task 7; GUI dialog + row marker → Task 8 (row tooltip simplified to label prefix + dialog scores, noted); real-video validation → Task 9.
- Deviation from spec recorded: per-row tooltip replaced by "⚠" label prefix + numeric scores in the warning dialog (custom-painted timeline has no per-row hover tooltips; adding hover machinery is not worth it for v1).
- Type consistency: `Alignment(offsets, confidence, warnings)` used identically in Tasks 2, 5, 6, 7, 8; `pairs[(i, j)]` with `i < j`, `offset = o_j − o_i` everywhere; `solve_offsets(n, pairs, reference_index)` signature consistent.
