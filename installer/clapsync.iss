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

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    if not WizardSilent then
      WizardForm.StatusLabel.Caption :=
        'Downloading the Python/CUDA environment (~5 GB, 10-20 min)...';
    if not Exec(ExpandConstant('{cmd}'),
        '/C ""' + ExpandConstant('{app}') + '\setup_env.cmd""', '',
        SW_SHOW, ewWaitUntilTerminated, ResultCode)
       or (ResultCode <> 0) then
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
