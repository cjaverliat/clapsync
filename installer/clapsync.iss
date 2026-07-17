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
var
  EnvProgressPage: TOutputProgressWizardPage;
  EnvStartTick: LongWord;
  EnvTimer: LongWord;
  LastPixiLine: String;

function SetTimer(Wnd: LongWord; IdEvent, Elapse: LongWord;
  TimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(Wnd: LongWord; IdEvent: LongWord): LongWord;
  external 'KillTimer@user32.dll stdcall';
function GetTickCount: LongWord;
  external 'GetTickCount@kernel32.dll stdcall';

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
    'clapsync is downloading its Python/CUDA environment.');
end;

procedure OnPixiLine(const S: String; const Error, FirstLine: Boolean);
begin
  { pixi is near-silent on a non-tty pipe (progress bars are tty-only):
    it prints nothing during download/link and one line at the end. Keep
    whatever does arrive for the log and the timer caption below. }
  Log('pixi| ' + S);
  if S <> '' then
    LastPixiLine := S;
end;

procedure EnvTimerProc(Wnd: LongWord; Msg: LongWord; IdEvent: LongWord;
  TickCount: LongWord);
var
  Secs: LongWord;
begin
  Secs := (GetTickCount - EnvStartTick) div 1000;
  EnvProgressPage.SetText(
    'Downloading the Python/CUDA environment (~5 GB, 10-20 min)...',
    Format('Elapsed %d:%.2d — %s', [Secs div 60, Secs mod 60,
      LastPixiLine]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ExecOk: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    LastPixiLine := 'pixi is downloading and linking packages';
    if not WizardSilent then
    begin
      EnvStartTick := GetTickCount;
      EnvProgressPage.SetText(
        'Downloading the Python/CUDA environment (~5 GB, 10-20 min)...',
        LastPixiLine);
      EnvProgressPage.ProgressBar.Style := npbstMarquee;
      EnvProgressPage.ProgressBar.Visible := True;
      EnvProgressPage.Show;
      { 1 Hz elapsed-time refresh: pixi gives a pipe no progress feed, so
        the page proves liveness with a clock instead. Wnd=0 timers fire
        via the message pump, which Inno keeps running during Exec waits. }
      EnvTimer := SetTimer(0, 0, 1000, CreateCallback(@EnvTimerProc));
    end;
    try
      { SW_HIDE: no console window exists, so the environment setup cannot
        be closed by mistake; output goes to the setup log instead. }
      ExecOk := ExecAndLogOutput(ExpandConstant('{cmd}'),
          '/C ""' + ExpandConstant('{app}') + '\setup_env.cmd""', '',
          SW_HIDE, ewWaitUntilTerminated, ResultCode, @OnPixiLine);
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
