"""End-to-end: sync a c3d to an audio clap and export it trimmed.

Exercises the clapperboard bridge (audio clap -> shared anchor -> c3d offset)
and the mocap export branch, confirming the offset sign and value against a
known injected delay.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import c3d
import numpy as np
import pytest

from clapsync.app.export import ExportSettings, export_media
from clapsync.app.media import probe
from clapsync.app.mocap import load_c3d
from clapsync.app.sync import align_media
from clapsync.core.timerange import full_time_range

AUDIO_CLAP_S = 1.0
MOTION_CLAP_S = 0.4
POINT_RATE = 100.0


def _write_clap_wav(path: Path, sample_rate: int = 16000, seconds: float = 2.0):
    rng = np.random.default_rng(0)
    n = int(sample_rate * seconds)
    x = rng.standard_normal(n) * 1e-3
    at = int(AUDIO_CLAP_S * sample_rate)
    length = int(0.008 * sample_rate)
    x[at : at + length] += rng.standard_normal(length)
    pcm = (x / np.max(np.abs(x)) * 30000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _write_clap_c3d(path: Path, seconds: float = 2.0):
    n = int(seconds * POINT_RATE)
    snap_end = int(MOTION_CLAP_S * POINT_RATE)
    snap_start = snap_end - 10
    writer = c3d.Writer(point_rate=POINT_RATE, analog_rate=0.0)
    writer.set_point_labels(["clapperboard_top", "clapperboard_down", "handle"])
    frames = []
    for f in range(n):
        if f < snap_start:
            frac = 1.0
        elif f < snap_end:
            frac = 1.0 - (f - snap_start) / (snap_end - snap_start)
        else:
            frac = 0.0
        row = np.zeros((3, 5), dtype=np.float32)
        row[0, :3] = (0.0, 0.0, 10.0 * frac)    # top arm
        row[1, :3] = (0.0, 0.0, -10.0 * frac)   # bottom arm
        row[2, :3] = (50.0, 0.0, 0.0)           # unrelated marker
        frames.append((row, np.zeros((0, 0), dtype=np.float32)))
    writer.add_frames(frames)
    with open(path, "wb") as handle:
        writer.write(handle)


@pytest.mark.slow
def test_c3d_syncs_to_audio_clap_and_exports(tmp_path: Path):
    wav = tmp_path / "cam.wav"
    c3d_path = tmp_path / "clap.c3d"
    _write_clap_wav(wav)
    _write_clap_c3d(c3d_path)

    media = [probe(wav), probe(c3d_path)]
    assert media[1].kind == "mocap"

    alignment = align_media(media, target_rate=16000)

    # offset = T_sound - T_motion = 1.0 - 0.4 = 0.6, per shared = local + offset.
    assert abs(alignment.offsets[1] - (AUDIO_CLAP_S - MOTION_CLAP_S)) < 0.05

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    durations = [m.duration for m in media]
    trim = full_time_range(durations, alignment.offsets)
    settings = ExportSettings(trim=trim, output_dir=out_dir)
    results = export_media(media, alignment.offsets, settings)

    assert all(r.ok for r in results), [r.error for r in results]
    exported = out_dir / "clap_synced.c3d"
    assert exported.exists()

    data = load_c3d(exported)
    assert data.n_frames == round(trim.duration * POINT_RATE)
    assert data.labels[:2] == ["clapperboard_top", "clapperboard_down"]


@pytest.mark.slow
def test_reference_never_c3d(tmp_path: Path):
    wav = tmp_path / "cam.wav"
    c3d_path = tmp_path / "clap.c3d"
    _write_clap_wav(wav)
    _write_clap_c3d(c3d_path)

    # c3d first in the list must not become the sync reference, and that
    # fallback must be silent (no sync-quality warning).
    media = [probe(c3d_path), probe(wav)]
    alignment = align_media(media, reference_index=0, target_rate=16000)
    assert math.isinf(alignment.confidence[1])   # the wav is the reference
    assert not math.isinf(alignment.confidence[0])  # not the c3d
    assert not any("reference" in w for w in alignment.warnings)
    assert abs(alignment.offsets[0] - (AUDIO_CLAP_S - MOTION_CLAP_S)) < 0.05


@pytest.mark.slow
def test_c3d_unsynced_when_no_clap_sound(tmp_path: Path):
    # Audio with no clap -> no anchor -> c3d left at offset 0 with a warning.
    rng = np.random.default_rng(1)
    sr, n = 16000, int(2.0 * 16000)
    pcm = (rng.standard_normal(n) * 500).astype("<i2")
    wav = tmp_path / "silent.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    c3d_path = tmp_path / "clap.c3d"
    _write_clap_c3d(c3d_path)

    media = [probe(wav), probe(c3d_path)]
    alignment = align_media(media, target_rate=16000)
    assert alignment.offsets[1] == 0.0
    assert any("motion capture left unsynced" in w for w in alignment.warnings)
