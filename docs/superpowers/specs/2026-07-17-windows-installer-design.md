# clapsync — Windows installer (online bootstrap via pixi)

Date: 2026-07-17
Status: Approved design (pending spec review)

## Goal

Ship clapsync to Windows end users as a normal installer: a small Inno Setup
executable (~50 MB) that installs per-user, then materializes the exact locked
pixi environment (python, torch cu130, PySide6, ffmpeg, framepipe, PyAV,
triton-windows) on the target machine by running `pixi install --locked` as a
post-install step. The user gets Start Menu / desktop shortcuts, an Add/Remove
Programs entry, and a clean uninstall. No Python, git, conda, or pixi required
on the target beforehand.

This is the "camp 3" pattern used by heavy-ML desktop apps (ComfyUI Desktop,
InvokeAI): nobody ships a 6 GB frozen torch binary; the installer bootstraps
the environment from a lockfile instead. It replaces the Nuitka standalone
route for end-user distribution.

## Non-goals

- Offline installer (pixi-pack self-extractor). Online-only for v1; internet
  is required at install time.
- Auto-update. Updating = run a newer installer over the old install.
- Code signing. V1 installers are unsigned (SmartScreen will warn).
- Linux/macOS packaging.
- CI automation of the installer build. V1 is a local pixi task; CI can come
  later.

## Decisions (with reasons)

| Decision | Choice | Why |
|---|---|---|
| Distribution | Online bootstrap | Torch cu130 + PySide6 + ffmpeg ≈ 5–6 GB. Small installer to hand out; pixi cache makes updates cheap. Offline pixi-pack reserved for a future hard requirement. |
| Env creation | At install time (Inno `[Run]`) | Failures surface during install, not first launch. No bootstrap UI in the app. App launches instantly ever after. |
| Shipping manifest | Separate `installer/pyproject.toml` | Dev manifest bakes editable `../framepipe` + dev tools (nuitka, pytest, pixi-pycharm, libpython-static, pandoc) into the default feature. A second minimal manifest is simpler than `no-default-feature` gymnastics. |
| clapsync/framepipe delivery | Wheels bundled in the installer, resolved via local find-links | No git or `../framepipe` on the target. Non-editable installs. |
| Install scope | Per-user, `%LOCALAPPDATA%\clapsync`, `PrivilegesRequired=lowest` | No admin prompt; the env is written and updated under the user profile where it belongs. |
| Pixi cache | `PIXI_CACHE_DIR` inside the install dir | Uninstall deletes one tree and the machine is clean. Cost: no cross-app cache sharing (acceptable). |
| GPU-less machines | Install succeeds, app degrades | `CONDA_OVERRIDE_CUDA=13.0` at install time makes the solve pass without an NVIDIA driver; at runtime torch falls back to CPU and export already falls back NVENC→CPU. |

## Architecture

### Repo additions

```
installer/
  pyproject.toml      # dist pixi workspace (win-64 only), own pixi.lock
  pixi.lock           # locked dist environment (committed)
  clapsync.iss        # Inno Setup script
  launcher.vbs        # hidden-console launcher (shipped verbatim)
  clapsync.ico        # shortcut/installer icon (new asset)
  wheels/             # build output: clapsync + framepipe wheels (gitignored)
  vendor/             # build output: pinned pixi.exe (gitignored)
```

### Layout on the target machine

```
%LOCALAPPDATA%\clapsync\
  pixi.exe                    # bundled, version-pinned
  launcher.vbs
  unins000.exe                # Inno uninstaller
  cache\                      # PIXI_CACHE_DIR (created at install)
  app\
    pyproject.toml            # the dist manifest
    pixi.lock
    wheels\
      clapsync-<ver>-py3-none-any.whl
      framepipe-<ver>-...whl
    .pixi\envs\default\       # created by `pixi install` post-install step
```

## Components

### 1. Dist manifest — `installer/pyproject.toml`

A standalone pixi workspace, independent of the dev manifest:

- `[tool.pixi.workspace]`: `channels = ["conda-forge"]`,
  `platforms = ["win-64"]`.
- Conda deps: `python >=3.10,<3.13`, `ffmpeg`.
- Pypi deps: `torch` / `torchaudio` `>=2.11.0,<2.12.0` from the
  `https://download.pytorch.org/whl/cu130` index, `PySide6 >=6.7,<6.8`,
  `av >=18`, `triton-windows ==3.5.0.post21`, `clapsync ==<version>`,
  `framepipe` (both resolved from the local `wheels/` directory via
  `[tool.pixi.pypi-options]` `find-links`).
- Excluded on purpose: nuitka, libpython-static, pixi-pycharm, pytest,
  pandoc/weasyprint, pip, setuptools pin — dev-only.
- The cuda130 conda packages (`cuda-version`, `cudnn`) are NOT included: on
  Windows the torch cu130 wheels bundle their own CUDA/cuDNN DLLs; the conda
  CUDA stack is a dev-environment concern.

Pin duplication: torch/torchaudio/PySide6/triton version pins exist in both
the dev `pyproject.toml` and this manifest. Each file carries a comment
pointing at the other; keeping them in sync is a release-checklist item.

`installer/pixi.lock` is committed so the installer build is reproducible and
the target install is `--locked`.

### 2. Wheels

`clapsync` and `framepipe` enter the dist env as ordinary wheels built at
installer-build time (`python -m build --wheel`), dropped into
`installer/wheels/`, and resolved via find-links. framepipe is built from the
sibling checkout `../framepipe` (same source the dev env uses today).

### 3. GUI entry point — change to the main `pyproject.toml`

Add:

```toml
[project.gui-scripts]
clapsync-gui = "clapsync.gui.app:main"
```

`gui-scripts` generates a pythonw-based `clapsync-gui.exe` (Windows GUI
subsystem — no console). The existing `[project.scripts] clapsync` console
entry stays for terminal use. Today the GUI drags a console window behind it
even in dev; this fixes that everywhere.

### 4. Launcher — `launcher.vbs`

Shortcuts cannot target `clapsync-gui.exe` directly: framepipe binds the
ffmpeg shared libraries from the env's `Library\bin`, which requires pixi's
activation (PATH). The shortcut therefore runs a 3-line VBScript that starts
pixi with a hidden window:

```vbscript
Set shell = CreateObject("WScript.Shell")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
shell.Run """" & appDir & "pixi.exe"" run --frozen --manifest-path """ & appDir & "app\pyproject.toml"" clapsync-gui", 0, False
```

`--manifest-path` makes the launcher working-directory independent;
`--frozen` forbids re-solving at launch. If antivirus heuristics ever flag
the .vbs, the fallback is a tiny compiled GUI-subsystem launcher exe — out of
scope for v1.

### 5. Inno Setup script — `installer/clapsync.iss`

- `PrivilegesRequired=lowest`, `DefaultDirName={localappdata}\clapsync`.
- `[Files]`: `vendor/pixi.exe`, `launcher.vbs`, `clapsync.ico`,
  `app/pyproject.toml` (from `installer/pyproject.toml`), `app/pixi.lock`,
  `app/wheels/*`.
- Disk-space pre-check: refuse to install with less than 12 GB free
  (env + cache + payload headroom).
- `[Run]` (waituntilterminated, console visible so pixi progress is honest):

  ```
  {app}\pixi.exe install --locked --manifest-path {app}\app\pyproject.toml
  ```

  with `CONDA_OVERRIDE_CUDA=13.0` and `PIXI_CACHE_DIR={app}\cache` set for
  the step. Nonzero exit aborts the install with a message pointing at the
  pixi log.
- `[Icons]`: Start Menu (always) + desktop (optional checkbox) shortcut →
  `wscript.exe "{app}\launcher.vbs"`, icon `{app}\clapsync.ico`.
- `[UninstallDelete]`: the entire `{app}` tree (env + cache included —
  10+ GB — the uninstaller must remove all of it, not just tracked files).

### 6. Build pipeline — pixi task `build-installer` (dev manifest)

Steps, in order:

1. Build wheels: `python -m build --wheel` for clapsync and `../framepipe`
   → `installer/wheels/` (directory cleaned first so stale versions can't
   leak into the lock).
2. Download the pinned pixi release (version constant in the task) into
   `installer/vendor/pixi.exe` (skipped if already present at that version).
3. Relock: `pixi lock --manifest-path installer/pyproject.toml` — picks up
   the freshly built wheels.
4. Compile: `ISCC.exe installer\clapsync.iss` →
   `outputs/clapsync-setup-<version>.exe`.

Inno Setup 6 is a build-machine prerequisite (`winget install
JRSoftware.InnoSetup`); it is not on conda-forge. The task fails with a clear
message if `ISCC.exe` is absent.

## Failure modes

- **No internet / blocked proxy at install** → the pixi step fails, the
  installer aborts with the log path. Rerunning the installer retries.
- **Insufficient disk** → refused up front by the 12 GB pre-check.
- **No NVIDIA GPU / old driver** → install succeeds (CUDA override); app
  runs on CPU, export uses the CPU encoder fallback. Documented in the
  README, not blocked.
- **SmartScreen** → unsigned v1 shows the "unrecognized app" warning; users
  click through via More info → Run anyway. Signing is future work.
- **Pixi version drift** → impossible at install time: the bundled pixi.exe
  is version-pinned and the env is `--locked` against a committed lockfile.

## Testing / success criteria

On a clean Windows box (Windows Sandbox or VM; no Python, git, conda, pixi):

1. Run `clapsync-setup-<version>.exe /VERYSILENT` → exit code 0.
2. `%LOCALAPPDATA%\clapsync\app\.pixi\envs\default\` exists;
   `...\Scripts\clapsync-gui.exe` exists.
3. `pixi.exe run --frozen --manifest-path ...app\pyproject.toml clapsync-cli
   sync <two test clips>` → exit 0, prints offsets (proves torch, PyAV,
   framepipe, ffmpeg DLLs all resolve).
4. GUI smoke: `pixi.exe run --frozen ... python -c "import
   clapsync.gui.app"` → exit 0. Manual: double-click the Start Menu shortcut
   → GUI opens, no console window.
5. Uninstall from Add/Remove Programs → `%LOCALAPPDATA%\clapsync` fully
   removed.

Interactive install (non-silent) is verified manually once per release:
progress console appears during the pixi step, shortcuts created, icon
correct.
