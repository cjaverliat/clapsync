' Starts the clapsync GUI with a hidden console. Lives next to pixi.exe
' in the install dir; the Start Menu shortcut points here via wscript.exe.
' Env vars mirror setup_env.cmd so pixi's implicit env repair (after a
' failed install) uses the app-local cache and solves on GPU-less machines.
Set shell = CreateObject("WScript.Shell")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
Set env = shell.Environment("PROCESS")
env("PIXI_CACHE_DIR") = appDir & "cache"
env("CONDA_OVERRIDE_CUDA") = "13.0"
shell.Run """" & appDir & "pixi.exe"" run --frozen --manifest-path """ _
    & appDir & "app\pixi.toml"" clapsync-gui", 0, False
