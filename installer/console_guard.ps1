# Keeps QuickEdit disabled on the setup console while the flag file exists.
#
# pixi resets the console input mode (re-enabling QuickEdit) when it starts
# rendering progress, so a stray click would otherwise freeze its output. We
# can't disable QuickEdit once up front — pixi turns it back on — so this
# runs alongside pixi and re-clears it on a short loop. setup_env.cmd creates
# the flag before pixi and deletes it after, bounding this process's life.
param([Parameter(Mandatory = $true)][string]$Flag)

$sig = @'
[DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int h);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
'@
$k = Add-Type -MemberDefinition $sig -Name ConGuard -Namespace ClapSync -PassThru
$stdin = $k::GetStdHandle(-10)   # STD_INPUT_HANDLE

# ENABLE_EXTENDED_FLAGS (0x80) must be set to change QuickEdit; clear
# QUICK_EDIT (0x40) and PROCESSED_INPUT (0x01, i.e. Ctrl+C).
while (Test-Path -LiteralPath $Flag) {
  $mode = 0
  if ($k::GetConsoleMode($stdin, [ref]$mode) -and (($mode -band 0x40) -ne 0)) {
    [void]$k::SetConsoleMode($stdin, ($mode -bor 0x80) -band (-bnot 0x41))
  }
  Start-Sleep -Milliseconds 100
}
