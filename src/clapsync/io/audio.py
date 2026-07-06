from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def _get_audio_sample_rate(file_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-select_streams", "a:0",
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}:\n{result.stderr}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream found in {file_path}")
    return int(streams[0].get("sample_rate", 44100))


def load_audio_from_video(
    file_path: Path | str,
    duration_s: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Load audio from a video (or audio) file using ffmpeg subprocess.
    Returns a mono float32 tensor of shape [1, num_samples] and the sample rate.

    If *duration_s* and *progress_callback* are both provided, ffmpeg runs in
    streaming mode and the callback is invoked with values in [0, 1] as audio
    is decoded.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    sample_rate = _get_audio_sample_rate(file_path)

    base_cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", str(file_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
    ]

    use_progress = (
        progress_callback is not None
        and duration_s is not None
        and duration_s > 0
    )

    if use_progress:
        # Audio data → stdout; progress key=value lines → stderr via -progress pipe:2
        cmd = base_cmd + ["-progress", "pipe:2", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def _read_progress() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                line = raw.decode(errors="replace").strip()
                if line.startswith("out_time_ms="):
                    try:
                        t = float(line.split("=", 1)[1]) / 1_000_000.0
                        progress_callback(min(1.0, t / duration_s))  # type: ignore[arg-type]
                    except (ValueError, IndexError):
                        pass

        progress_thread = threading.Thread(target=_read_progress, daemon=True)
        progress_thread.start()

        assert proc.stdout is not None
        audio_bytes = proc.stdout.read()
        proc.wait()
        progress_thread.join(timeout=1.0)
        rc = proc.returncode
    else:
        result = subprocess.run(base_cmd + ["pipe:1"], capture_output=True)
        audio_bytes = result.stdout
        rc = result.returncode
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed for {file_path}:\n"
                + result.stderr.decode(errors="replace")
            )

    if use_progress and rc != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed for {file_path}")

    if not audio_bytes:
        raise RuntimeError(f"No audio data decoded from {file_path}")

    samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)  # [1, T]

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    return waveform, sample_rate
