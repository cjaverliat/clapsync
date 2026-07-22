"""Load and write c3d motion-capture files behind a plain data model.

I/O only — parsing and serialization via py-c3d. All analysis (clap detection,
marker classification) lives in ``clapsync.core.clap`` and stays pure.

A c3d point row is ``[x, y, z, residual, camera]``; a residual below zero marks
an occluded/invalid sample. ``read_frames`` bundles each point frame with its
analog sub-block, so trimming at point-frame granularity keeps analog data in
lockstep automatically — no separate analog-rate bookkeeping.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import c3d
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MocapData:
    """Marker trajectories and analog data from one c3d file.

    Attributes:
        point_rate: Marker sampling rate (Hz).
        first_frame: c3d FIRST_FRAME (1-based frame index of the first sample).
        labels: Marker labels; order matches ``points``/``valid`` axis 1.
        points: (n_frames, n_markers, 3) float32 world coordinates.
        valid: (n_frames, n_markers) bool; False where the sample is occluded.
        analog: (n_frames, n_channels, samples_per_frame) float32. Empty
            (n_frames, 0, 0) when the file has no analog data.
        analog_rate: Analog sampling rate (Hz), 0.0 when absent.
        source_path: File the data was read from, so ``write_c3d`` can inherit
            its parameter groups (units, labels, calibration). None if unknown.
    """

    point_rate: float
    first_frame: int
    labels: list[str]
    points: np.ndarray
    valid: np.ndarray
    analog: np.ndarray
    analog_rate: float
    source_path: Path | None = None

    @property
    def n_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_markers(self) -> int:
        return int(self.points.shape[1])

    @property
    def duration(self) -> float:
        """Trajectory length in seconds."""
        return self.n_frames / self.point_rate if self.point_rate else 0.0


def probe_c3d(path: Path) -> tuple[float, int]:
    """Read only the c3d header: (point_rate, n_frames). Cheap, no frame data.

    Args:
        path: Source .c3d path.

    Returns:
        (point_rate_hz, frame_count).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the header cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with open(path, "rb") as handle, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = c3d.Reader(handle)
            return float(reader.point_rate), int(reader.frame_count)
    except (OSError, FileNotFoundError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot read c3d header {path}: {exc}") from exc


def load_c3d(path: Path) -> MocapData:
    """Read a c3d file into a MocapData.

    Args:
        path: Source .c3d path.

    Returns:
        Populated MocapData.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be parsed as c3d.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        with open(path, "rb") as handle, warnings.catch_warnings():
            # py-c3d warns on files with no analog data; that is not an error.
            warnings.simplefilter("ignore")
            reader = c3d.Reader(handle)
            labels = [label.strip() for label in reader.point_labels]
            point_frames: list[np.ndarray] = []
            valid_frames: list[np.ndarray] = []
            analog_frames: list[np.ndarray] = []
            for _idx, points, analog in reader.read_frames():
                point_frames.append(points[:, :3].astype(np.float32))
                valid_frames.append(points[:, 3] >= 0.0)
                analog_frames.append(np.asarray(analog, dtype=np.float32))
            point_rate = float(reader.point_rate)
            first_frame = int(reader.first_frame)
            analog_rate = float(reader.analog_rate)
    except (OSError, FileNotFoundError):
        raise
    except Exception as exc:  # noqa: BLE001 — surface any parse failure uniformly
        raise ValueError(f"cannot read c3d file {path}: {exc}") from exc

    if not point_frames:
        raise ValueError(f"c3d file has no frames: {path}")

    points = np.stack(point_frames)
    valid = np.stack(valid_frames)
    analog = _stack_analog(analog_frames)
    return MocapData(
        point_rate=point_rate,
        first_frame=first_frame,
        labels=labels,
        points=points,
        valid=valid,
        analog=analog,
        analog_rate=analog_rate,
        source_path=path,
    )


def write_c3d(path: Path, data: MocapData) -> None:
    """Write MocapData to a c3d file, inheriting the source file's parameters.

    Parameter groups (units, marker labels, calibration) are copied from
    ``data.source_path`` when set, so the output stays interoperable with the
    tools that produced the input. Invalid samples are written with residual
    −1 so downstream readers do not treat padded frames as real markers.

    EVENT parameters are dropped: their absolute times would be wrong after a
    trim and rebasing them reliably is out of scope. A warning is logged when
    events are discarded.

    Args:
        path: Destination .c3d path.
        data: Marker/analog data to serialize.
    """
    path = Path(path)
    writer = _writer_for(data)

    residual = np.where(data.valid, 0.0, -1.0).astype(np.float32)
    frames = []
    for f in range(data.n_frames):
        row = np.zeros((data.n_markers, 5), dtype=np.float32)
        row[:, :3] = data.points[f]
        row[:, 3] = residual[f]
        frames.append((row, data.analog[f]))
    writer.add_frames(frames)
    writer.set_start_frame(data.first_frame)

    with open(path, "wb") as handle:
        writer.write(handle)


def _stack_analog(analog_frames: list[np.ndarray]) -> np.ndarray:
    """Stack per-frame analog blocks to (n_frames, n_channels, samples).

    Files without analog data yield 1-D empty blocks; normalize those to an
    (n_frames, 0, 0) array so padding math is uniform.
    """
    n = len(analog_frames)
    first = analog_frames[0]
    if first.ndim != 2 or first.size == 0:
        return np.zeros((n, 0, 0), dtype=np.float32)
    return np.stack(analog_frames)


def _writer_for(data: MocapData) -> c3d.Writer:
    """Build a Writer that inherits source parameters when possible."""
    if data.source_path is not None:
        try:
            with open(data.source_path, "rb") as handle, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = c3d.Reader(handle)
                writer = reader.to_writer("copy_metadata")
            if "EVENT" in writer.group_keys():
                logger.warning(
                    "dropping EVENT parameters from %s — event times are not "
                    "rebased on trim", data.source_path,
                )
                writer.remove_group("EVENT")
            return writer
        except Exception as exc:  # noqa: BLE001 — fall back to a minimal writer
            logger.warning(
                "could not inherit c3d parameters from %s (%s) — writing "
                "minimal header", data.source_path, exc,
            )

    writer = c3d.Writer(
        point_rate=data.point_rate, analog_rate=data.analog_rate,
    )
    writer.set_point_labels(data.labels)
    return writer
