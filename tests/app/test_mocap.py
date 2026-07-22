"""Round-trip and parsing tests for the c3d I/O layer."""
from __future__ import annotations

import warnings
from pathlib import Path

import c3d
import numpy as np
import pytest

from clapsync.app.mocap import MocapData, load_c3d, write_c3d


def _write_raw_c3d(
    path: Path,
    labels: list[str],
    points: np.ndarray,
    valid: np.ndarray,
    point_rate: float = 100.0,
) -> None:
    """Write a c3d directly via py-c3d (not clapsync) for a known fixture."""
    writer = c3d.Writer(point_rate=point_rate, analog_rate=0.0)
    writer.set_point_labels(labels)
    residual = np.where(valid, 0.0, -1.0).astype(np.float32)
    frames = []
    for f in range(points.shape[0]):
        row = np.zeros((points.shape[1], 5), dtype=np.float32)
        row[:, :3] = points[f]
        row[:, 3] = residual[f]
        frames.append((row, np.zeros((0, 0), dtype=np.float32)))
    writer.add_frames(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "wb") as handle:
            writer.write(handle)


def test_load_c3d_reads_points_labels_and_rate(tmp_path: Path):
    labels = ["clapperboard_top", "clapperboard_down", "other"]
    points = np.zeros((10, 3, 3), dtype=np.float32)
    points[:, 0, 2] = np.arange(10)  # top marker moves in z
    valid = np.ones((10, 3), dtype=bool)
    path = tmp_path / "in.c3d"
    _write_raw_c3d(path, labels, points, valid, point_rate=120.0)

    data = load_c3d(path)
    assert data.labels == labels
    assert data.point_rate == 120.0
    assert data.n_frames == 10
    assert data.n_markers == 3
    assert np.allclose(data.points[:, 0, 2], np.arange(10), atol=1e-3)
    assert data.valid.all()
    assert abs(data.duration - 10 / 120.0) < 1e-6


def test_load_c3d_marks_invalid_samples(tmp_path: Path):
    labels = ["a", "b"]
    points = np.ones((5, 2, 3), dtype=np.float32)
    valid = np.ones((5, 2), dtype=bool)
    valid[2, 1] = False  # occlude one sample
    path = tmp_path / "occ.c3d"
    _write_raw_c3d(path, labels, points, valid)

    data = load_c3d(path)
    assert not data.valid[2, 1]
    assert data.valid.sum() == 9


def test_write_c3d_round_trips_points(tmp_path: Path):
    labels = ["m0", "m1"]
    points = np.random.default_rng(0).standard_normal((8, 2, 3)).astype(np.float32)
    valid = np.ones((8, 2), dtype=bool)
    valid[3, 0] = False
    src = tmp_path / "src.c3d"
    _write_raw_c3d(src, labels, points, valid)

    data = load_c3d(src)
    out = tmp_path / "out.c3d"
    write_c3d(out, data)
    reloaded = load_c3d(out)

    assert reloaded.labels == labels
    assert reloaded.point_rate == data.point_rate
    assert reloaded.n_frames == data.n_frames
    # Invalid samples stay invalid; valid coordinates survive the round trip.
    assert not reloaded.valid[3, 0]
    keep = reloaded.valid & data.valid
    assert np.allclose(reloaded.points[keep], data.points[keep], atol=1e-2)


def test_load_c3d_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_c3d(tmp_path / "nope.c3d")
