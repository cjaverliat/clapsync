@echo off
rem Materializes the clapsync environment. Run by the installer; rerun
rem manually if the install-time environment setup failed.
rem The installer matches this exact title to remove the close button.
title clapsync environment setup
set "CONDA_OVERRIDE_CUDA=13.0"
set "PIXI_CACHE_DIR=%~dp0cache"
rem Guard the console against click-to-freeze (QuickEdit) while pixi runs.
rem pixi re-enables QuickEdit on startup, so a background loop re-clears it
rem for as long as the flag file exists (best effort; classic conhost only).
set "QE_FLAG=%~dp0.console_guard"
echo running > "%QE_FLAG%"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0console_guard.ps1" "%QE_FLAG%" >nul 2>&1
"%~dp0pixi.exe" install --locked --manifest-path "%~dp0app\pixi.toml"
set "RC=%ERRORLEVEL%"
del "%QE_FLAG%" >nul 2>&1
rem Signal completion + exit code to the installer (it polls this file
rem while pixi runs so its window stays responsive). Written last, so its
rem absence means "still running or aborted".
> "%~dp0.setup_result" echo %RC%
exit /b %RC%
