# Windows Installer (online bootstrap via pixi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `outputs/clapsync-setup-<version>.exe` — a small Inno Setup
installer that installs clapsync per-user and materializes the locked pixi
environment (torch cu130, PySide6, ffmpeg, framepipe, PyAV, triton-windows)
on the target machine at install time.

**Architecture:** A standalone `installer/` pixi workspace (win-64 only, own
lock) resolves clapsync + framepipe from locally built wheels via find-links.
`installer/build.py` builds the wheels, fetches a pinned `pixi.exe`, relocks,
and compiles the Inno script. On the target, an Inno `[Code]` step runs
`setup_env.cmd` → `pixi install --locked`; a Start Menu shortcut runs
`launcher.vbs` → hidden `pixi run --frozen clapsync-gui`.

**Tech Stack:** pixi 0.59.0, Inno Setup 6 (ISCC), pip wheel, VBScript, cmd.

Spec: `docs/superpowers/specs/2026-07-17-windows-installer-design.md`

## Global Constraints

- Pins copied from the dev manifest — keep identical in `installer/pixi.toml`:
  `torch`/`torchaudio` `>=2.11.0,<2.12.0` from index
  `https://download.pytorch.org/whl/cu130`; `PySide6 >=6.7.0,<6.8`;
  `triton-windows ==3.5.0.post21`; `python >=3.10,<3.13`; `av >=18`.
- Bundled pixi version: **0.59.0** (matches the dev machine).
- Installer: `PrivilegesRequired=lowest`, install dir
  `{localappdata}\clapsync`, refuse install below 12 GB free disk.
- Env-setup step environment: `CONDA_OVERRIDE_CUDA=13.0`,
  `PIXI_CACHE_DIR=<install dir>\cache`.
- Dist env excludes: nuitka, libpython-static, pixi-pycharm, pytest, pip,
  pandoc/weasyprint, conda cuda-version/cudnn (torch cu130 wheels bundle
  their own CUDA DLLs on Windows).
- Python style: Google style guide, 80-col lines, type-annotated public
  functions (applies to `installer/build.py`).

**Approved deviations from the spec** (decided while planning — do NOT
"fix" back):

1. Dist manifest is a native `installer/pixi.toml`, not
   `installer/pyproject.toml` — it is an environment manifest, not a Python
   package; a pyproject manifest would need a fake `[project]` stub. All
   target paths use `app\pixi.toml` accordingly.
2. `clapsync = "*"` / `framepipe = "*"` instead of `==<version>` — find-links
   only sees the freshly built wheels and the committed lock pins exact
   versions; an exact pin in the manifest would need editing every release.
3. New shipped file `setup_env.cmd` — wraps the env vars around
   `pixi install` (Inno's `Exec` can't set env vars) and doubles as the
   user-facing retry entry point after a failed install.

---

### Task 1: GUI entry point + pin-sync comment (dev `pyproject.toml`)

**Files:**
- Modify: `pyproject.toml` (root)

**Interfaces:**
- Produces: console-script-free GUI exe `clapsync-gui` (entry
  `clapsync.gui.app:main`) — Task 2's manifest and Task 4's launcher invoke
  it by this exact name.

- [ ] **Step 1: Add `[project.gui-scripts]`**

In root `pyproject.toml`, directly after the `[project.scripts]` block:

```toml
[project.gui-scripts]
clapsync-gui = "clapsync.gui.app:main"
```

- [ ] **Step 2: Add the pin-sync comment**

Extend the comment above `[tool.pixi.pypi-dependencies]` (the block that
starts "Abstract runtime deps for consumers.") with one line:

```toml
# Torch/PySide6/triton pins are duplicated in installer/pixi.toml (shipped
# Windows env) — keep them in sync when bumping.
```

- [ ] **Step 3: Reinstall + verify the exe exists**

Run: `pixi install`
Then: `ls .pixi/envs/default/Scripts/clapsync-gui.exe`
Expected: file exists. If missing (pixi may not re-install an unchanged
editable dep), force it:
`pixi run python -m pip install -e . --no-deps --force-reinstall`, re-check.

- [ ] **Step 4: Import smoke**

Run: `pixi run python -c "from clapsync.gui.app import main; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "feat(build): add clapsync-gui windowed entry point"
```

(Include `pixi.lock` only if `pixi install` changed it.)

---

### Task 2: Dist manifest `installer/pixi.toml` + gitignore

**Files:**
- Create: `installer/pixi.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: entry point name `clapsync-gui` (Task 1).
- Produces: manifest at `installer/pixi.toml` with find-links dir `wheels/`
  — Task 3 locks it; Task 5 ships it to `{app}\app\pixi.toml`.

- [ ] **Step 1: Write `installer/pixi.toml`**

```toml
# Shipping environment for the Windows installer.
#
# Locked by installer/build.py (which first builds clapsync + framepipe
# wheels into wheels/ — the find-links source); installed on the target by
# setup_env.cmd via `pixi install --locked`.
#
# Version pins for torch/torchaudio/PySide6/triton MUST track
# [tool.pixi.pypi-dependencies] in ../pyproject.toml.

[workspace]
name = "clapsync-dist"
channels = ["conda-forge"]
platforms = ["win-64"]

[dependencies]
python = ">=3.10,<3.13"
ffmpeg = "*"

[pypi-options]
find-links = [{ path = "wheels" }]

[pypi-dependencies]
# clapsync and framepipe resolve to the locally built wheels in wheels/;
# the committed pixi.lock pins their exact versions.
clapsync = "*"
framepipe = "*"
torch = { version = ">=2.11.0,<2.12.0", index = "https://download.pytorch.org/whl/cu130" }
torchaudio = { version = ">=2.11.0,<2.12.0", index = "https://download.pytorch.org/whl/cu130" }
av = ">=18"
PySide6 = ">=6.7.0,<6.8"
triton-windows = "==3.5.0.post21"
```

- [ ] **Step 2: Gitignore the build outputs**

In `.gitignore`, under the `# build` group, add:

```
installer/wheels/
installer/vendor/
```

- [ ] **Step 3: Verify the manifest parses**

Run: `pixi info --manifest-path installer/pixi.toml`
Expected: exit 0, output shows workspace `clapsync-dist`, platform
`win-64`. (Locking comes in Task 3 — it needs the wheels to exist.)

- [ ] **Step 4: Commit**

```bash
git add installer/pixi.toml .gitignore
git commit -m "feat(installer): add win-64 dist manifest"
```

---

### Task 3: `installer/build.py` — wheels, vendored pixi, lock

**Files:**
- Create: `installer/build.py`
- Create (generated, committed): `installer/pixi.lock`

**Interfaces:**
- Consumes: `installer/pixi.toml` (Task 2); sibling checkout
  `../framepipe`.
- Produces: `installer/wheels/*.whl` (2 wheels), `installer/vendor/pixi.exe`
  (0.59.0), committed `installer/pixi.lock`. Functions `app_version() -> str`
  and `run(cmd)` — Task 5 extends this file with `compile_installer()`.

- [ ] **Step 1: Write `installer/build.py`**

```python
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
```

- [ ] **Step 2: Run it**

Run: `pixi run python installer/build.py`
Expected: two `pip wheel` runs succeed; pixi download message; `pixi lock`
writes `installer/pixi.lock`. (Downloads several GB of wheel metadata /
packages for solving — takes minutes.)

- [ ] **Step 3: Verify outputs**

Run: `ls installer/wheels/` → exactly two `.whl` files (clapsync-0.2.0…,
framepipe-…).
Run: `installer/vendor/pixi.exe --version` → `pixi 0.59.0`.
Run: `grep -c "download.pytorch.org/whl/cu130" installer/pixi.lock` → ≥ 1.
Run: `grep -c "clapsync" installer/pixi.lock` → ≥ 1, and the same for
`framepipe`, `pyside6`, `triton-windows`, `ffmpeg`.

- [ ] **Step 4: Commit**

```bash
git add installer/build.py installer/pixi.lock
git commit -m "feat(installer): build script for wheels, vendored pixi, lock"
```

---

### Task 4: Shipped runtime files — `launcher.vbs`, `setup_env.cmd`, icon

**Files:**
- Create: `installer/launcher.vbs`
- Create: `installer/setup_env.cmd`
- Create: `installer/clapsync.ico` (generated binary asset, committed)

**Interfaces:**
- Consumes: entry point `clapsync-gui` (Task 1); target layout
  `{app}\pixi.exe`, `{app}\app\pixi.toml`, `{app}\cache` (spec).
- Produces: the three files Task 5's `[Files]` section ships verbatim.
  `setup_env.cmd` exit code = pixi's exit code (Inno checks it).

- [ ] **Step 1: Write `installer/launcher.vbs`**

Both files live in `{app}` on the target, so `%~dp0` / script-dir prefixing
resolves everything without hardcoded paths:

```vbscript
' Starts the clapsync GUI with a hidden console. Lives next to pixi.exe
' in the install dir; the Start Menu shortcut points here via wscript.exe.
Set shell = CreateObject("WScript.Shell")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
shell.Run """" & appDir & "pixi.exe"" run --frozen --manifest-path """ _
    & appDir & "app\pixi.toml"" clapsync-gui", 0, False
```

- [ ] **Step 2: Write `installer/setup_env.cmd`**

```cmd
@echo off
rem Materializes the clapsync environment. Run by the installer; rerun
rem manually if the install-time environment setup failed.
set "CONDA_OVERRIDE_CUDA=13.0"
set "PIXI_CACHE_DIR=%~dp0cache"
"%~dp0pixi.exe" install --locked --manifest-path "%~dp0app\pixi.toml"
exit /b %ERRORLEVEL%
```

- [ ] **Step 3: Generate `installer/clapsync.ico`**

Write this one-off script to the scratchpad (NOT the repo) as
`make_icon.py`, run it, keep only the `.ico`:

```python
"""One-off: renders the clap emoji to installer/clapsync.ico."""
import sys

from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter
from PySide6.QtCore import Qt

app = QGuiApplication(sys.argv)
img = QImage(256, 256, QImage.Format_ARGB32)
img.fill(Qt.transparent)
p = QPainter(img)
p.setRenderHint(QPainter.Antialiasing)
p.setBrush(QColor(30, 30, 46))
p.setPen(Qt.NoPen)
p.drawRoundedRect(0, 0, 256, 256, 48, 48)
p.setFont(QFont("Segoe UI Emoji", 120))
p.setPen(QColor(255, 255, 255))
p.drawText(img.rect(), Qt.AlignCenter, "\U0001F44F")
p.end()
ok = img.save(r"C:\Users\javerlia\PycharmProjects\clapsync\installer\clapsync.ico")
print("saved" if ok else "FAILED")
```

Run: `pixi run python <scratchpad>\make_icon.py`
Expected: `saved`, and `installer/clapsync.ico` exists (a few KB).
If the emoji renders as a hollow box, replace the two font/pen lines with
`p.setFont(QFont("Segoe UI", 96, QFont.Bold))` and draw the text `"CS"`
instead — a plain monogram beats a broken glyph.

- [ ] **Step 4: Syntax-check the cmd wrapper**

Run (PowerShell, from repo root):
`cmd /c installer\setup_env.cmd; "exit: $LASTEXITCODE"`
Expected: pixi runs and FAILS fast (exit ≠ 0) complaining about
`app\pixi.toml` not existing — that path only exists in the installed
layout. A clean parse + that specific error = pass. An error about `set`
syntax or `%~dp0` = fail.

- [ ] **Step 5: Commit**

```bash
git add installer/launcher.vbs installer/setup_env.cmd installer/clapsync.ico
git commit -m "feat(installer): launcher, env-setup wrapper, icon"
```

---

### Task 5: Inno Setup script + compile stage + `build-installer` task

**Files:**
- Create: `installer/clapsync.iss`
- Modify: `installer/build.py` (add compile stage)
- Modify: `pyproject.toml` (root — add pixi task)

**Interfaces:**
- Consumes: everything Tasks 2–4 placed in `installer/`; `run()` and module
  constants from `installer/build.py`.
- Produces: `outputs/clapsync-setup-<version>.exe`; pixi task
  `build-installer`.

- [ ] **Step 0: Prerequisite — Inno Setup 6 on the build machine**

Run: `winget install JRSoftware.InnoSetup --silent` (skip if
`C:\Program Files (x86)\Inno Setup 6\ISCC.exe` or
`%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe` already exists — winget
installs per-user to the latter under `PrivilegesRequired=lowest`-style
defaults; both paths are probed).

- [ ] **Step 1: Write `installer/clapsync.iss`**

```ini
; clapsync online-bootstrap installer.
; Compiled by installer/build.py: ISCC /DAppVersion=<ver> clapsync.iss
; Installs per-user, then runs setup_env.cmd (pixi install --locked) to
; materialize the Python/CUDA environment (~5 GB download).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{A3E1F6D2-7C48-4B9E-8D15-6F0B2A9C3E71}
AppName=clapsync
AppVersion={#AppVersion}
AppPublisher=Charles Javerliat
DefaultDirName={localappdata}\clapsync
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\outputs
OutputBaseFilename=clapsync-setup-{#AppVersion}
SetupIconFile=clapsync.ico
UninstallDisplayIcon={app}\clapsync.ico
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "vendor\pixi.exe"; DestDir: "{app}"
Source: "launcher.vbs"; DestDir: "{app}"
Source: "setup_env.cmd"; DestDir: "{app}"
Source: "clapsync.ico"; DestDir: "{app}"
Source: "pixi.toml"; DestDir: "{app}\app"
Source: "pixi.lock"; DestDir: "{app}\app"
Source: "wheels\*"; DestDir: "{app}\app\wheels"

[Icons]
Name: "{userprograms}\clapsync"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\launcher.vbs"""; IconFilename: "{app}\clapsync.ico"
Name: "{userdesktop}\clapsync"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\launcher.vbs"""; IconFilename: "{app}\clapsync.ico"; Tasks: desktopicon

[UninstallDelete]
; The pixi env (~10 GB) and cache are created post-install, so Inno doesn't
; track them — delete the whole tree explicitly.
Type: filesandordirs; Name: "{app}"

[Code]
function InitializeSetup(): Boolean;
var
  FreeBytes, TotalBytes: Int64;
begin
  Result := True;
  if GetSpaceOnDisk64(ExpandConstant('{localappdata}'), FreeBytes,
      TotalBytes) then
    if FreeBytes < Int64(12) * 1024 * 1024 * 1024 then
    begin
      SuppressibleMsgBox('clapsync needs about 12 GB of free disk space '
        + 'for its Python/CUDA environment. Free up space and retry.',
        mbError, MB_OK, IDOK);
      Result := False;
    end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
      WizardForm.StatusLabel.Caption :=
        'Downloading the Python/CUDA environment (~5 GB, 10-20 min)...';
    if not Exec(ExpandConstant('{cmd}'),
        '/C ""' + ExpandConstant('{app}') + '\setup_env.cmd""', '',
        SW_SHOW, ewWaitUntilTerminated, ResultCode)
       or (ResultCode <> 0) then
    begin
      SuppressibleMsgBox('Environment setup failed (exit code '
        + IntToStr(ResultCode) + '). Check your internet connection, then '
        + 'rerun ' + ExpandConstant('{app}') + '\setup_env.cmd.',
        mbError, MB_OK, IDOK);
      { Files are already copied at ssPostInstall — Abort marks the setup
        as failed (nonzero exit) but does not roll them back. Rerunning
        the installer or setup_env.cmd repairs the install. }
      Abort;
    end;
  end;
end;
```

- [ ] **Step 2: Add the compile stage to `installer/build.py`**

Add `tomllib` to the stdlib imports (alphabetical order), then append after
`relock()`:

```python
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
```

Also add `import tomllib` to the module-top stdlib imports (alphabetical
order: between `sys` and `urllib.request`). Extend `main()`:

```python
def main() -> None:
    build_wheels()
    fetch_pixi()
    relock()
    compile_installer()
```

- [ ] **Step 3: Add the pixi task**

In root `pyproject.toml`, after the `[tool.pixi.tasks.build-clapsync]`
block:

```toml
[tool.pixi.tasks.build-installer]
cmd = "python installer/build.py"
description = "Build the Windows online-bootstrap installer"
```

- [ ] **Step 4: Full build**

Run: `pixi run build-installer`
Expected: wheels rebuilt, pixi fetch skipped (already vendored), relock
no-op or quick, ISCC compiles with 0 warnings-as-errors, and
`outputs/clapsync-setup-0.2.0.exe` exists. Report its size (expect roughly
40–80 MB — wheels + pixi.exe, lzma2).

- [ ] **Step 5: Commit**

```bash
git add installer/clapsync.iss installer/build.py pyproject.toml
git commit -m "feat(installer): inno setup script + build-installer task"
```

---

### Task 6: README section + clean-box verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the built `outputs/clapsync-setup-<version>.exe` (Task 5).

- [ ] **Step 1: Add a README section**

After the existing "## Build a standalone binary" section:

```markdown
## Build the Windows installer

Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`) and a sibling `../framepipe`
checkout.

```bash
pixi run build-installer
```

Produces `outputs/clapsync-setup-<version>.exe` — a small online installer.
It installs per-user to `%LOCALAPPDATA%\clapsync`, then downloads the locked
Python/CUDA environment (~5 GB) at install time, so internet is required
during install. No NVIDIA GPU is needed to install; without one, clapsync
runs on the CPU (slower sync and export).
```

(Note: nested code fence — use a 4-backtick outer fence or indent when
editing, and verify the rendered README afterwards.)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the windows installer build"
```

- [ ] **Step 3: Clean-box verification (spec success criteria)**

Best-effort automated on this machine; the true clean-box pass needs
Windows Sandbox or a VM (no Python/git/pixi). Attempt Windows Sandbox:
`WindowsSandbox.exe` with the installer exe shared. If Sandbox is
unavailable, STOP and hand the user this checklist instead of faking it:

1. `clapsync-setup-<version>.exe /VERYSILENT` → exit code 0.
2. `%LOCALAPPDATA%\clapsync\app\.pixi\envs\default\Scripts\clapsync-gui.exe`
   exists.
3. `%LOCALAPPDATA%\clapsync\pixi.exe run --frozen --manifest-path
   %LOCALAPPDATA%\clapsync\app\pixi.toml clapsync-cli sync <clipA> <clipB>`
   → exit 0, prints per-clip offsets (proves torch, PyAV, framepipe, ffmpeg
   DLLs resolve).
4. Start Menu shortcut "clapsync" opens the GUI with no console window.
5. Uninstall via Add/Remove Programs → `%LOCALAPPDATA%\clapsync` fully
   gone.

Record pass/fail per item in the final report.
