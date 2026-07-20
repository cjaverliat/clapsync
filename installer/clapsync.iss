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
Source: "console_guard.ps1"; DestDir: "{app}"
Source: "clapsync.ico"; DestDir: "{app}"
Source: "pixi.toml"; DestDir: "{app}\app"
Source: "pixi.lock"; DestDir: "{app}\app"
Source: "env_packages.txt"; DestDir: "{app}\app"
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
  { Disk footprint of the finished env, in MB. Calibrated against a real
    cold install measured at 8.38 GiB (2026-07-17, lock for 0.2.0). Drives
    the smooth bar; re-measure %LOCALAPPDATA%\clapsync when the lock changes
    materially. }
  EnvTotalMB = 8581;
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
  EnvPkgTotal: Integer;         { package count from env_packages.txt }
  EnvFreeAtStart: Int64;
  EnvCondaMeta, EnvSitePkgs, EnvCacheDir: String;

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
    'clapsync is downloading and installing its Python/CUDA packages.');
end;

{ Count of filesystem entries matching Mask (e.g. dir\*.json), excluding
  . and .. — each installed conda package leaves one conda-meta\*.json and
  each pypi package one site-packages\*.dist-info, so this counts progress
  without matching names. }
function CountMatches(const Mask: String): Integer;
var
  FR: TFindRec;
begin
  Result := 0;
  if FindFirst(Mask, FR) then
  try
    repeat
      if (FR.Name <> '.') and (FR.Name <> '..') then
        Result := Result + 1;
    until not FindNext(FR);
  finally
    FindClose(FR);
  end;
end;

{ Newest-created subdirectory of Dir, with its name shortened to the part
  before the first '-' (e.g. torch-2.11.0+cu130.dist-info -> torch). }
function NewestPkgName(const Dir: String): String;
var
  FR: TFindRec;
  Newest: TFileTime;
  Name: String;
  Dash: Integer;
begin
  Result := '';
  Newest.dwLowDateTime := 0; Newest.dwHighDateTime := 0;
  if FindFirst(Dir + '\*', FR) then
  try
    repeat
      if (FR.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0)
         and (FR.Name <> '.') and (FR.Name <> '..')
         and ((FR.CreationTime.dwHighDateTime > Newest.dwHighDateTime)
              or ((FR.CreationTime.dwHighDateTime = Newest.dwHighDateTime)
                  and (FR.CreationTime.dwLowDateTime > Newest.dwLowDateTime))) then
      begin
        Newest := FR.CreationTime;
        Name := FR.Name;
      end;
    until not FindNext(FR);
  finally
    FindClose(FR);
  end;
  Dash := Pos('-', Name);
  if Dash > 1 then
    Result := Copy(Name, 1, Dash - 1)
  else
    Result := Name;
end;

procedure EnvTimerProc(Wnd: LongWord; Msg: LongWord; IdEvent: LongWord;
  TickCount: LongWord);
var
  Secs: LongWord;
  ConsoleWnd: LongWord;
  FreeNow, TotalDisk: Int64;
  UsedMB, Done: Integer;
  PkgName, DoneText: String;
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

  { Smooth bar from bytes written to disk — moves continuously even during
    a single large download (torch), which a package-completion bar can't. }
  UsedMB := 0;
  if (EnvFreeAtStart > 0)
     and GetSpaceOnDisk64(ExpandConstant('{localappdata}'), FreeNow,
       TotalDisk)
     and (EnvFreeAtStart > FreeNow) then
    UsedMB := Integer((EnvFreeAtStart - FreeNow) div 1048576);
  if UsedMB > (EnvTotalMB * 99) div 100 then
    UsedMB := (EnvTotalMB * 99) div 100;
  EnvProgressPage.SetProgress(UsedMB, EnvTotalMB);

  { Determinate structure from the package manifest + install markers. }
  Done := CountMatches(EnvCondaMeta + '\*.json')
        + CountMatches(EnvSitePkgs + '\*.dist-info');
  if EnvPkgTotal > 0 then
    DoneText := Format('Package %d of %d', [Done, EnvPkgTotal])
  else
    DoneText := Format('%d packages installed', [Done]);

  PkgName := NewestPkgName(EnvSitePkgs);
  if PkgName = '' then PkgName := NewestPkgName(EnvCacheDir);
  if PkgName = '' then PkgName := 'downloading packages';

  EnvProgressPage.SetText(
    Format('%s  —  %s', [DoneText, PkgName]),
    Format('Elapsed %d:%.2d', [Secs div 60, Secs mod 60]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ExecOk: Boolean;
  Lines: TArrayOfString;
  EnvRoot: String;
  FreeTotal: Int64;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
    begin
      EnvStartTick := GetTickCount;
      EnvConsoleGuarded := False;
      EnvRoot := ExpandConstant('{app}') + '\app\.pixi\envs\default';
      EnvCondaMeta := EnvRoot + '\conda-meta';
      EnvSitePkgs := EnvRoot + '\Lib\site-packages';
      EnvCacheDir := ExpandConstant('{app}') + '\cache\pkgs';
      EnvPkgTotal := 0;
      if LoadStringsFromFile(ExpandConstant('{app}') + '\app\env_packages.txt',
          Lines) then
        EnvPkgTotal := GetArrayLength(Lines);
      EnvFreeAtStart := 0;
      if not GetSpaceOnDisk64(ExpandConstant('{localappdata}'),
          EnvFreeAtStart, FreeTotal) then
        EnvFreeAtStart := 0;
      EnvProgressPage.SetProgress(0, EnvTotalMB);
      EnvProgressPage.SetText('Starting pixi...', '');
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
