' Starts the clapsync GUI with a hidden console. Lives next to pixi.exe
' in the install dir; the Start Menu shortcut points here via wscript.exe.
Set shell = CreateObject("WScript.Shell")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
shell.Run """" & appDir & "pixi.exe"" run --frozen --manifest-path """ _
    & appDir & "app\pixi.toml"" clapsync-gui", 0, False
