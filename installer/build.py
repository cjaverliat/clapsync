"""Builds the clapsync Windows installer.

Stages: build clapsync + framepipe wheels into installer/wheels/, fetch the
pinned pixi.exe into installer/vendor/, relock installer/pixi.toml, then
compile the Inno Setup script into outputs/ (compile stage added with the
.iss file).

Run from the dev environment: pixi run build-installer
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PIXI_VERSION = "0.59.0"
PIXI_URL = (
    "https://github.com/prefix-dev/pixi/releases/download/"
    f"v{PIXI_VERSION}/pixi-x86_64-pc-windows-msvc.zip"
)
ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer"
WHEELS = INSTALLER / "wheels"
VENDOR = INSTALLER / "vendor"
FRAMEPIPE = ROOT.parent / "framepipe"


def run(cmd: list) -> None:
    """Runs a command, echoing it first; exits nonzero on failure."""
    print("+", *map(str, cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def build_wheels() -> None:
    """Builds clapsync and framepipe wheels into a clean wheels/ dir."""
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True)
    for project in (ROOT, FRAMEPIPE):
        if not project.exists():
            sys.exit(f"missing checkout: {project}")
        run([sys.executable, "-m", "pip", "wheel", "--no-deps",
             "--wheel-dir", WHEELS, project])


def fetch_pixi() -> None:
    """Downloads the pinned pixi.exe into vendor/ (no-op if already there)."""
    exe = VENDOR / "pixi.exe"
    if exe.exists():
        proc = subprocess.run([str(exe), "--version"], capture_output=True,
                              text=True)
        if proc.stdout.strip() == f"pixi {PIXI_VERSION}":
            return
    VENDOR.mkdir(parents=True, exist_ok=True)
    archive = VENDOR / "pixi.zip"
    print("+ download", PIXI_URL, flush=True)
    urllib.request.urlretrieve(PIXI_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract("pixi.exe", VENDOR)
    archive.unlink()


def relock() -> None:
    """Relocks the dist manifest with the vendored pixi."""
    run([VENDOR / "pixi.exe", "lock", "--manifest-path",
         INSTALLER / "pixi.toml"])


def main() -> None:
    build_wheels()
    fetch_pixi()
    relock()


if __name__ == "__main__":
    main()
