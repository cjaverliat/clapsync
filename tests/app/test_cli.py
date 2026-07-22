"""Integration tests for the clapsync headless CLI."""
from __future__ import annotations

import pytest

from clapsync.cli import main


@pytest.mark.slow
def test_cli_sync_prints_offsets(av_video, capsys):
    a, *_ = av_video(seconds=1.0, name="a.mp4")
    b, *_ = av_video(seconds=1.0, name="b.mp4")
    rc = main(["sync", str(a), str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "offset" in out.lower()
    assert "confidence" in out.lower()


@pytest.mark.slow
def test_cli_synctrim_writes_files(av_video, tmp_path):
    a, *_ = av_video(seconds=1.0, name="a.mp4")
    b, *_ = av_video(seconds=1.0, name="b.mp4")
    out = tmp_path / "out"
    rc = main(["synctrim", str(a), str(b), "-o", str(out)])
    assert rc == 0
    assert list(out.glob("*_synced.mp4"))


def test_cli_no_command_returns_error(capsys):
    rc = main([])
    assert rc == 2
