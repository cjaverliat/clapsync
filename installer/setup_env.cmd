@echo off
rem Materializes the clapsync environment. Run by the installer; rerun
rem manually if the install-time environment setup failed.
rem The installer matches this exact title to remove the close button.
title clapsync environment setup
rem Disable QuickEdit (a stray click freezes console output) and Ctrl+C
rem for this console. Best effort - the install works without it.
powershell -NoProfile -Command "$d='[DllImport(\"kernel32.dll\")]public static extern IntPtr GetStdHandle(int h);[DllImport(\"kernel32.dll\")]public static extern bool GetConsoleMode(IntPtr h,out uint m);[DllImport(\"kernel32.dll\")]public static extern bool SetConsoleMode(IntPtr h,uint m);';$k=Add-Type -MemberDefinition $d -Name K -Namespace W -PassThru;$h=$k::GetStdHandle(-10);$m=0;[void]$k::GetConsoleMode($h,[ref]$m);[void]$k::SetConsoleMode($h,($m -bor 0x80) -band (-bnot 0x41))" >nul 2>&1
set "CONDA_OVERRIDE_CUDA=13.0"
set "PIXI_CACHE_DIR=%~dp0cache"
"%~dp0pixi.exe" install --locked --manifest-path "%~dp0app\pixi.toml"
exit /b %ERRORLEVEL%
