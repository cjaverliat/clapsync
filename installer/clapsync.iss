; clapsync online-bootstrap installer.
; Compiled by installer/build.py: ISCC /DAppVersion=<ver> clapsync.iss
; Installs per-user, then runs setup_env.cmd (pixi install --locked) to
; materialize the Python/CUDA environment (~2 GB download, ~8.4 GB on disk).
; pixi runs in its own visible console and renders its real per-package
; progress there. The console is closable and the wizard's standard Cancel
; button stops it too; setup_env.cmd still disables click-to-freeze
; (QuickEdit).

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
Source: "wheels\*"; DestDir: "{app}\app\wheels"

[Icons]
Name: "{userprograms}\clapsync"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\launcher.vbs"""; IconFilename: "{app}\clapsync.ico"
Name: "{userdesktop}\clapsync"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\launcher.vbs"""; IconFilename: "{app}\clapsync.ico"; Tasks: desktopicon

[UninstallDelete]
; The pixi env (~8.4 GB) and cache are created post-install, so Inno doesn't
; track them — delete the whole tree explicitly.
Type: filesandordirs; Name: "{app}"

[Code]
const
  { setup_env.cmd sets this exact console title; Cancel finds the window by
    it to close it. }
  EnvConsoleTitle = 'clapsync environment setup';
  WM_CLOSE_ = $0010;
  PM_REMOVE = 1;

type
  TMsg = record
    hwnd: LongWord;
    message: LongWord;
    wParam: LongWord;
    lParam: LongWord;
    time: LongWord;
    pt_x: LongInt;
    pt_y: LongInt;
  end;

var
  EnvRunning: Boolean;          { true while the env-setup process runs }
  EnvCancelRequested: Boolean;

function GetTickCount: LongWord;
  external 'GetTickCount@kernel32.dll stdcall';
function FindWindowByTitle(ClassName, WindowName: String): LongWord;
  external 'FindWindowW@user32.dll stdcall';
function PostMessageW(Wnd, Msg, WParam, LParam: LongWord): Boolean;
  external 'PostMessageW@user32.dll stdcall';
function PeekMessageW(var lpMsg: TMsg; Wnd, FilterMin, FilterMax,
  RemoveMsg: LongWord): Boolean;
  external 'PeekMessageW@user32.dll stdcall';
function TranslateMessage(const lpMsg: TMsg): Boolean;
  external 'TranslateMessage@user32.dll stdcall';
function DispatchMessageW(const lpMsg: TMsg): LongWord;
  external 'DispatchMessageW@user32.dll stdcall';

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

{ Fires when the user clicks the wizard's standard Cancel button (or closes
  the window). During env setup, close the console instead of tearing down
  the wizard: pixi (the console's child) dies, the poll loop below sees the
  console vanish, and CurStepChanged reports the cancellation. }
procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
var
  Wnd: LongWord;
begin
  if EnvRunning then
  begin
    EnvCancelRequested := True;
    Wnd := FindWindowByTitle('ConsoleWindowClass', EnvConsoleTitle);
    if Wnd <> 0 then
      PostMessageW(Wnd, WM_CLOSE_, 0, 0);
    Cancel := False;   { we drive the abort ourselves, below }
    Confirm := False;  { no second "Exit Setup?" prompt }
  end;
end;

{ Dispatch any pending window messages so the wizard stays responsive (the
  Cancel button, repaint, marquee) while we wait for pixi. Inno's own loop
  is blocked inside CurStepChanged, so this is the only pump running. }
procedure PumpMessages();
var
  Msg: TMsg;
begin
  while PeekMessageW(Msg, 0, 0, 0, PM_REMOVE) do
  begin
    TranslateMessage(Msg);
    DispatchMessageW(Msg);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  StartTick, Secs, LastSec: LongWord;
  ConsoleWnd: LongWord;
  ResultFile, Params: String;
  ResultLine: AnsiString;
  ResultCode: Integer;
  Launched, ConsoleSeen, Done, Ok: Boolean;
begin
  if CurStep <> ssPostInstall then
    Exit;

  EnvCancelRequested := False;
  ResultFile := ExpandConstant('{app}') + '\.setup_result';
  DeleteFile(ResultFile);
  Params := '/C ""' + ExpandConstant('{app}') + '\setup_env.cmd""';

  if not WizardSilent then
  begin
    { Marquee on the standard install page = honest "working" animation,
      not a (wrong) percentage. }
    WizardForm.ProgressGauge.Style := npbstMarquee;
    WizardForm.StatusLabel.Caption :=
      'Installing the Python/CUDA environment (~2 GB download)...';
  end;

  { Launch pixi NON-blocking so the wizard message loop stays alive; a
    blocking Exec would freeze the whole UI (no Cancel) for the entire
    install. setup_env.cmd writes its exit code to .setup_result when it
    finishes. Visible console: a real tty, so pixi renders its own
    per-package progress; the window is closable and QuickEdit is disabled
    (no click-to-freeze). }
  EnvRunning := True;
  Launched := Exec(ExpandConstant('{cmd}'), Params, '', SW_SHOW, ewNoWait,
    ResultCode);

  StartTick := GetTickCount;
  LastSec := 999999;
  ConsoleSeen := False;
  Done := False;
  Ok := False;
  ResultCode := 1;

  if not Launched then
    Done := True
  else
    repeat
      PumpMessages();
      Sleep(50);

      if (not WizardSilent) then
      begin
        Secs := (GetTickCount - StartTick) div 1000;
        if Secs <> LastSec then
        begin
          WizardForm.FilenameLabel.Caption := Format('Elapsed %d:%.2d — '
            + 'live progress in the console window.', [Secs div 60,
            Secs mod 60]);
          LastSec := Secs;
        end;
      end;

      if FileExists(ResultFile) then
      begin
        { Normal completion: setup_env.cmd wrote its exit code. }
        if LoadStringFromFile(ResultFile, ResultLine)
           and (Trim(ResultLine) <> '') then
        begin
          ResultCode := StrToIntDef(Trim(ResultLine), 1);
          Ok := True;
          Done := True;
        end;
      end
      else
      begin
        { No result yet. If the console has appeared and then vanished
          without writing one, the user closed it (or Cancel did) — abort. }
        ConsoleWnd := FindWindowByTitle('ConsoleWindowClass',
          EnvConsoleTitle);
        if ConsoleWnd <> 0 then
          ConsoleSeen := True
        else if ConsoleSeen then
          Done := True
        else if (GetTickCount - StartTick) > 20000 then
          { console never appeared in 20 s — treat as a failed launch
            rather than looping forever. }
          Done := True;
      end;
    until Done;

  EnvRunning := False;
  DeleteFile(ResultFile);
  if not WizardSilent then
    WizardForm.ProgressGauge.Style := npbstNormal;

  if EnvCancelRequested then
  begin
    SuppressibleMsgBox('Installation cancelled. Re-run the installer to '
      + 'finish setting up clapsync.', mbInformation, MB_OK, IDOK);
    Abort;
  end
  else if (not Ok) or (ResultCode <> 0) then
  begin
    SuppressibleMsgBox('Environment setup did not complete (exit code '
      + IntToStr(ResultCode) + '). Check your internet connection, then '
      + 'rerun ' + ExpandConstant('{app}') + '\setup_env.cmd.',
      mbError, MB_OK, IDOK);
    { Files are already copied at ssPostInstall — Abort marks the setup as
      failed but does not roll them back. Rerunning the installer or
      setup_env.cmd repairs the install. }
    Abort;
  end;
end;
