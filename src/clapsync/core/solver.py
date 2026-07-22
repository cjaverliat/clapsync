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
    if n == 0:
        return Alignment([], [], [])
    offsets = [0.0] * n
    confidence = [0.0] * n
    confidence[reference_index] = math.inf
    warnings: list[str] = []
    if n == 1:
        return Alignment(offsets, confidence, warnings)

    edges = _gated_edges(pairs)
    _repair_pass(n, edges)
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
