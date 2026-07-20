"""Builds the clapsync Windows installer.

Stages: build clapsync + framepipe wheels into installer/wheels/, fetch the
pinned pixi.exe into installer/vendor/, relock installer/pixi.toml, then
compile the Inno Setup script into outputs/ (compile stage added with the
.iss file).

Run from the dev environment: pixi run build-installer
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile

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


def run(cmd: list[str | Path]) -> None:
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


def write_env_packages() -> None:
    """Writes env_packages.txt: the locked package set for install progress.

    `pixi list --locked --json` resolves the lock without a materialized
    env, so this runs on the build machine. The installer ships the result
    and uses it to show a determinate "N of M packages" during setup. One
    line per package: "kind|name|size_bytes" (size may be empty for local
    wheels). The line count is the package total.
    """
    print("+ pixi list --locked --json", flush=True)
    proc = subprocess.run(
        [str(VENDOR / "pixi.exe"), "list", "--locked", "--json",
         "--manifest-path", str(INSTALLER / "pixi.toml")],
        check=True, capture_output=True, text=True)
    packages = json.loads(proc.stdout)
    lines = [f"{p['kind']}|{p['name']}|{p.get('size_bytes') or ''}"
             for p in packages]
    (INSTALLER / "env_packages.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote env_packages.txt ({len(lines)} packages)", flush=True)


def app_version() -> str:
    """Reads the clapsync version from the root pyproject."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def compile_installer() -> None:
    """Compiles clapsync.iss with ISCC into outputs/."""
    candidates = [
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path.home() / "AppData/Local/Programs/Inno Setup 6/ISCC.exe",
    ]
    which = shutil.which("ISCC.exe")
    if which:
        candidates.append(Path(which))
    iscc = next((p for p in candidates if p.exists()), None)
    if iscc is None:
        sys.exit("ISCC.exe not found - install Inno Setup 6: "
                 "winget install JRSoftware.InnoSetup")
    run([iscc, f"/DAppVersion={app_version()}", INSTALLER / "clapsync.iss"])


def main() -> None:
    build_wheels()
    fetch_pixi()
    relock()
    write_env_packages()
    compile_installer()


if __name__ == "__main__":
    main()
