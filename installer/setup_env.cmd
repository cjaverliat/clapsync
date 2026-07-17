@echo off
rem Materializes the clapsync environment. Run by the installer; rerun
rem manually if the install-time environment setup failed.
set "CONDA_OVERRIDE_CUDA=13.0"
set "PIXI_CACHE_DIR=%~dp0cache"
"%~dp0pixi.exe" install --locked --manifest-path "%~dp0app\pixi.toml"
exit /b %ERRORLEVEL%
