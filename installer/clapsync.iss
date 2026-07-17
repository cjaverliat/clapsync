; clapsync online-bootstrap installer.
; Compiled by installer/build.py: ISCC /DAppVersion=<ver> clapsync.iss
; Installs per-user, then runs setup_env.cmd (pixi install --locked) to
; materialize the Python/CUDA environment (~2 GB download, ~8.4 GB on disk).
; pixi runs in its own visible console (a real tty, so it renders its own
; per-package progress); the console is guarded: QuickEdit and Ctrl+C are
; disabled by setup_env.cmd, and the close button is removed below.

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
const
  { setup_env.cmd sets this exact console title; the timer below finds the
    window by it to strip the close button (which also disables Alt+F4). }
  EnvConsoleTitle = 'clapsync environment setup';
  SC_CLOSE_ = $F060;
  MF_BYCOMMAND_ = 0;

var
  EnvProgressPage: TOutputProgressWizardPage;
  EnvStartTick: LongWord;
  EnvTimer: LongWord;
  EnvConsoleGuarded: Boolean;

function SetTimer(Wnd: LongWord; IdEvent, Elapse: LongWord;
  TimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(Wnd: LongWord; IdEvent: LongWord): LongWord;
  external 'KillTimer@user32.dll stdcall';
function GetTickCount: LongWord;
  external 'GetTickCount@kernel32.dll stdcall';
function FindWindowByTitle(ClassName, WindowName: String): LongWord;
  external 'FindWindowW@user32.dll stdcall';
function GetSystemMenu(Wnd: LongWord; Revert: LongWord): LongWord;
  external 'GetSystemMenu@user32.dll stdcall';
function DeleteMenu(Menu: LongWord; Position, Flags: LongWord): LongWord;
  external 'DeleteMenu@user32.dll stdcall';

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

procedure InitializeWizard();
begin
  EnvProgressPage := CreateOutputProgressPage('Setting up the environment',
    'pixi shows its progress in the console window next to this one.');
end;

procedure EnvTimerProc(Wnd: LongWord; Msg: LongWord; IdEvent: LongWord;
  TickCount: LongWord);
var
  Secs: LongWord;
  ConsoleWnd: LongWord;
begin
  { The console appears shortly after Exec starts and only then gets its
    title (set by setup_env.cmd) — so the guard is applied from here,
    first tick that finds it. Classic conhost only; under Windows
    Terminal the lookup misses and the guard is skipped. }
  if not EnvConsoleGuarded then
  begin
    ConsoleWnd := FindWindowByTitle('ConsoleWindowClass', EnvConsoleTitle);
    if ConsoleWnd <> 0 then
    begin
      DeleteMenu(GetSystemMenu(ConsoleWnd, 0), SC_CLOSE_, MF_BYCOMMAND_);
      EnvConsoleGuarded := True;
    end;
  end;
  Secs := (GetTickCount - EnvStartTick) div 1000;
  EnvProgressPage.SetText(
    'pixi is installing the Python/CUDA environment (~2 GB download).',
    Format('Elapsed %d:%.2d — progress shows in the console window.', [
      Secs div 60, Secs mod 60]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ExecOk: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
    begin
      EnvStartTick := GetTickCount;
      EnvConsoleGuarded := False;
      EnvProgressPage.SetText(
        'pixi is installing the Python/CUDA environment (~2 GB download).',
        'Starting pixi...');
      EnvProgressPage.Show;
      { 1 Hz refresh via Wnd=0 timer: fires through the message pump,
        which Inno keeps running during Exec waits. }
      EnvTimer := SetTimer(0, 0, 1000, CreateCallback(@EnvTimerProc));
    end;
    try
      { Visible console: a real tty, so pixi renders its own per-package
        progress there (piping the output would silence it). setup_env.cmd
        disables QuickEdit (click-freeze) and Ctrl+C; the timer removes
        the close button. }
      ExecOk := Exec(ExpandConstant('{cmd}'),
          '/C ""' + ExpandConstant('{app}') + '\setup_env.cmd""', '',
          SW_SHOW, ewWaitUntilTerminated, ResultCode);
    finally
      if EnvTimer <> 0 then
      begin
        KillTimer(0, EnvTimer);
        EnvTimer := 0;
      end;
      if not WizardSilent then
        EnvProgressPage.Hide;
    end;
    if (not ExecOk) or (ResultCode <> 0) then
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
