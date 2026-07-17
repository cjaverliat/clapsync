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
const
  { Expected bytes written during env setup (unpacked caches + env), in MB.
    Calibrated against a real cold install measured at 8.38 GiB on disk
    (2026-07-17, lock for 0.2.0). Re-measure %LOCALAPPDATA%\clapsync when
    the lock changes materially. }
  EnvTotalMB = 8581;

var
  EnvProgressPage: TOutputProgressWizardPage;
  EnvMemo: TNewMemo;
  EnvStartTick: LongWord;
  EnvTimer: LongWord;
  EnvFreeAtStart: Int64;
  EnvSiteDir, EnvCacheDir: String;
  EnvBaseSite, EnvBaseCache: TFileTime;

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
  { Console-style box under the bar collecting pixi's (sparse) output. }
  EnvMemo := TNewMemo.Create(WizardForm);
  EnvMemo.Parent := EnvProgressPage.Surface;
  EnvMemo.Left := 0;
  EnvMemo.Top := EnvProgressPage.ProgressBar.Top
    + EnvProgressPage.ProgressBar.Height + ScaleY(16);
  EnvMemo.Width := EnvProgressPage.SurfaceWidth;
  EnvMemo.Height := ScaleY(110);
  EnvMemo.ReadOnly := True;
  EnvMemo.ScrollBars := ssVertical;
  EnvMemo.Font.Name := 'Consolas';
end;

procedure OnPixiLine(const S: String; const Error, FirstLine: Boolean);
begin
  { pixi is near-silent on a non-tty pipe (progress bars are tty-only):
    a warning or two mid-run and one line at the end. Collect them as
    history in the memo so a lone warning doesn't read as a hang. }
  Log('pixi| ' + S);
  if (S <> '') and (EnvMemo <> nil) then
    EnvMemo.Lines.Add(S);
end;

function NewerFT(const A, B: TFileTime): Boolean;
begin
  Result := (A.dwHighDateTime > B.dwHighDateTime)
    or ((A.dwHighDateTime = B.dwHighDateTime)
        and (A.dwLowDateTime > B.dwLowDateTime));
end;

{ Name of the most recently created subdirectory, with its timestamp. }
function NewestSubdir(const Dir: String; var Newest: TFileTime): String;
var
  FR: TFindRec;
begin
  Result := '';
  Newest.dwLowDateTime := 0;
  Newest.dwHighDateTime := 0;
  if FindFirst(Dir + '\*', FR) then
  try
    repeat
      if (FR.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0)
         and (FR.Name <> '.') and (FR.Name <> '..')
         and NewerFT(FR.CreationTime, Newest) then
      begin
        Newest := FR.CreationTime;
        Result := FR.Name;
      end;
    until not FindNext(FR);
  finally
    FindClose(FR);
  end;
end;

procedure EnvTimerProc(Wnd: LongWord; Msg: LongWord; IdEvent: LongWord;
  TickCount: LongWord);
var
  FreeNow, TotalDisk: Int64;
  UsedMB: LongInt;
  Secs: LongWord;
  Name, Activity: String;
  FT: TFileTime;
begin
  { pixi reports no progress to a pipe, but its work is visible on disk:
    free-space delta drives the bar, and package dirs appearing in the
    env / package cache name what is being installed right now. Only
    dirs created after this run started count (update runs start with a
    populated env). }
  Secs := (GetTickCount - EnvStartTick) div 1000;
  UsedMB := 0;
  if (EnvFreeAtStart > 0)
     and GetSpaceOnDisk64(ExpandConstant('{localappdata}'), FreeNow,
       TotalDisk)
     and (EnvFreeAtStart > FreeNow) then
    UsedMB := LongInt((EnvFreeAtStart - FreeNow) div 1048576);
  if UsedMB > (EnvTotalMB * 99) div 100 then
    UsedMB := (EnvTotalMB * 99) div 100;
  EnvProgressPage.SetProgress(UsedMB, EnvTotalMB);
  Activity := 'downloading packages';
  Name := NewestSubdir(EnvSiteDir, FT);
  if (Name <> '') and NewerFT(FT, EnvBaseSite) then
    Activity := 'installing ' + Name
  else
  begin
    Name := NewestSubdir(EnvCacheDir, FT);
    if (Name <> '') and NewerFT(FT, EnvBaseCache) then
      Activity := 'extracting ' + Name;
  end;
  EnvProgressPage.SetText(
    'Downloading the Python/CUDA environment (~2 GB download)...',
    Format('Elapsed %d:%.2d — %s', [
      Secs div 60, Secs mod 60, Activity]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ExecOk: Boolean;
  FreeTotal: Int64;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
    begin
      EnvStartTick := GetTickCount;
      EnvFreeAtStart := 0;
      if not GetSpaceOnDisk64(ExpandConstant('{localappdata}'),
          EnvFreeAtStart, FreeTotal) then
        EnvFreeAtStart := 0;
      EnvSiteDir := ExpandConstant('{app}')
        + '\app\.pixi\envs\default\Lib\site-packages';
      EnvCacheDir := ExpandConstant('{app}') + '\cache\pkgs';
      { Baseline newest-dir times so update runs ignore what an earlier
        install already created. }
      NewestSubdir(EnvSiteDir, EnvBaseSite);
      NewestSubdir(EnvCacheDir, EnvBaseCache);
      EnvMemo.Lines.Add('> pixi install --locked  (output below; pixi '
        + 'prints little while it downloads)');
      EnvProgressPage.SetProgress(0, EnvTotalMB);
      EnvProgressPage.SetText(
        'Downloading the Python/CUDA environment (~2 GB download)...',
        'Starting pixi...');
      EnvProgressPage.Show;
      { 1 Hz refresh via Wnd=0 timer: fires through the message pump,
        which Inno keeps running during Exec waits. }
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
