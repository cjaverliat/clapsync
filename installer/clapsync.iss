; clapsync online-bootstrap installer.
; Compiled by installer/build.py: ISCC /DAppVersion=<ver> clapsync.iss
; Installs per-user, then runs setup_env.cmd (pixi install --locked) to
; materialize the Python/CUDA environment (~2 GB download, ~8.4 GB on disk).
; pixi runs in its own visible console and renders its real per-package
; progress there. The console is closable and a Cancel button on the wizard
; stops it too; setup_env.cmd still disables click-to-freeze (QuickEdit).

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
  { setup_env.cmd sets this exact console title; the Cancel button finds the
    window by it to close it. }
  EnvConsoleTitle = 'clapsync environment setup';
  WM_CLOSE_ = $0010;

var
  EnvPage: TOutputProgressWizardPage;
  EnvCancelButton: TNewButton;
  EnvStartTick: LongWord;
  EnvTimer: LongWord;
  EnvCancelRequested: Boolean;

function SetTimer(Wnd: LongWord; IdEvent, Elapse: LongWord;
  TimerFunc: LongWord): LongWord;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(Wnd: LongWord; IdEvent: LongWord): LongWord;
  external 'KillTimer@user32.dll stdcall';
function GetTickCount: LongWord;
  external 'GetTickCount@kernel32.dll stdcall';
function FindWindowByTitle(ClassName, WindowName: String): LongWord;
  external 'FindWindowW@user32.dll stdcall';
function PostMessageW(Wnd, Msg, WParam, LParam: LongWord): Boolean;
  external 'PostMessageW@user32.dll stdcall';

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

procedure EnvCancelClick(Sender: TObject);
var
  Wnd: LongWord;
begin
  EnvCancelRequested := True;
  EnvCancelButton.Enabled := False;
  EnvCancelButton.Caption := 'Cancelling...';
  { Close the setup console; pixi (its child) dies and the blocking Exec
    below returns. Same effect as the user closing the console window. }
  Wnd := FindWindowByTitle('ConsoleWindowClass', EnvConsoleTitle);
  if Wnd <> 0 then
    PostMessageW(Wnd, WM_CLOSE_, 0, 0);
end;

procedure InitializeWizard();
begin
  EnvPage := CreateOutputProgressPage('Setting up the environment',
    'clapsync is installing its Python/CUDA packages. pixi shows live '
    + 'progress in the console window.');
  { Marquee = honest "working" animation, not a (wrong) percentage. }
  EnvPage.ProgressBar.Style := npbstMarquee;
  EnvCancelButton := TNewButton.Create(WizardForm);
  EnvCancelButton.Parent := EnvPage.Surface;
  EnvCancelButton.Width := ScaleX(150);
  EnvCancelButton.Height := ScaleY(25);
  EnvCancelButton.Top := EnvPage.ProgressBar.Top
    + EnvPage.ProgressBar.Height + ScaleY(18);
  EnvCancelButton.Left := 0;
  EnvCancelButton.Caption := 'Cancel installation';
  EnvCancelButton.OnClick := @EnvCancelClick;
end;

procedure EnvTimerProc(Wnd: LongWord; Msg: LongWord; IdEvent: LongWord;
  TickCount: LongWord);
var
  Secs: LongWord;
begin
  Secs := (GetTickCount - EnvStartTick) div 1000;
  EnvPage.SetText(
    'Installing the Python/CUDA environment (~2 GB download).',
    Format('Elapsed %d:%.2d — live progress in the console window.', [
      Secs div 60, Secs mod 60]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ExecOk: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    EnvCancelRequested := False;
    if not WizardSilent then
    begin
      EnvStartTick := GetTickCount;
      EnvCancelButton.Enabled := True;
      EnvCancelButton.Caption := 'Cancel installation';
      EnvPage.SetText('Starting pixi...', '');
      EnvPage.Show;
      { 1 Hz elapsed-clock refresh via a Wnd=0 timer, which fires through
        the message pump Inno keeps running during the Exec wait — the same
        pump that delivers the Cancel button click. }
      EnvTimer := SetTimer(0, 0, 1000, CreateCallback(@EnvTimerProc));
    end;
    try
      { Visible console: a real tty, so pixi renders its own per-package
        progress there. The window is closable (no close-button lock) so
        the user can abort; setup_env.cmd disables QuickEdit so a stray
        click doesn't freeze the output. }
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
        EnvPage.Hide;
    end;
    if EnvCancelRequested then
    begin
      SuppressibleMsgBox('Installation cancelled. Re-run the installer to '
        + 'finish setting up clapsync.', mbInformation, MB_OK, IDOK);
      Abort;
    end
    else if (not ExecOk) or (ResultCode <> 0) then
    begin
      SuppressibleMsgBox('Environment setup did not complete (exit code '
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
