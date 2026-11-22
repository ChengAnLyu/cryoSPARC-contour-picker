#ifndef AppVersion
  #define AppVersion "3.0.0"
#endif
#ifndef AppSourceDir
  #define AppSourceDir "..\..\dist\SARU_Picker_crYOLO"
#endif
#ifndef AppOutputDir
  #define AppOutputDir "..\..\artifacts\cryolo"
#endif

[Setup]
AppId={{FB137682-60A8-4BC1-AC91-518AE16F0E8E}
AppName=SARU Picker
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}
AppPublisher=SARU Picker
AppVerName=SARU Picker {#AppVersion}
DefaultDirName={localappdata}\Programs\SARU Picker crYOLO
DefaultGroupName=SARU Picker 3.0
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#AppOutputDir}
OutputBaseFilename=SARU-Picker-Setup-{#AppVersion}-win64
SetupIconFile=..\..\assets\app-icon.ico
UninstallDisplayIcon={app}\_internal\assets\app-icon.ico
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "Install-crYOLO-Runtime.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup-runtime.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "runtime-requirements-windows.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "runtime_health.py"; DestDir: "{app}"; Flags: ignoreversion
; Private learned weights are supplied beside Setup. Verified public SAM ViT-B
; is already bundled and does not need a separate sidecar.
Source: "{src}\cryolo-capsids.h5"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\cryolo-config.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\capsid-profile.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.ico"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.png"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\SARU Picker 3.0"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"
Name: "{group}\Quick start"; Filename: "{app}\_internal\QUICK_START.txt"
Name: "{autodesktop}\SARU Picker 3.0"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"; Tasks: desktopicon

[InstallDelete]
; Remove only shortcuts created by the prior edition owned by this same AppId.
Type: files; Name: "{userprograms}\SARU Picker crYOLO\SARU Picker crYOLO.lnk"
Type: files; Name: "{userprograms}\SARU Picker crYOLO\Install crYOLO runtime.lnk"
Type: files; Name: "{userprograms}\SARU Picker crYOLO\Quick start.lnk"
Type: files; Name: "{userdesktop}\SARU Picker crYOLO.lnk"

[Run]
Filename: "{app}\SARU Picker crYOLO.exe"; Description: "Launch SARU Picker 3.0"; Flags: nowait postinstall skipifsilent; Check: RuntimeIsReady

[Code]
var
  RuntimeReady: Boolean;
  RuntimeError: String;

function RuntimeIsReady: Boolean;
begin
  Result := RuntimeReady;
end;

function GetCustomSetupExitCode: Integer;
begin
  if RuntimeReady then Result := 0 else Result := 1;
end;

procedure RuntimeSetupFailed(MessageText: String);
begin
  RuntimeError := MessageText;
  Log(RuntimeError);
  if not WizardSilent then MsgBox(RuntimeError, mbError, MB_OK);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and not RuntimeReady then
  begin
    WizardForm.FinishedHeadingLabel.Caption := 'crYOLO setup needs attention';
    WizardForm.FinishedLabel.Caption := RuntimeError + #13#10#13#10 +
      'SARU Picker was installed, but it will not launch automatically. Open SARU Picker 3.0 from Start, then Advanced > Repair crYOLO runtime.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ExitCode: Integer;
  ProgramPath, Parameters: String;
begin
  if CurStep = ssPostInstall then
  begin
    RuntimeReady := False;
    WizardForm.StatusLabel.Caption := 'Preparing the crYOLO detection profile...';
    ProgramPath := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
    Parameters := '-NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\setup-runtime.ps1') + '"';
    if not Exec(ProgramPath, Parameters, ExpandConstant('{app}'), SW_SHOWMINIMIZED,
      ewWaitUntilTerminated, ExitCode) then
    begin
      RuntimeSetupFailed('The crYOLO runtime setup could not start. Open SARU Picker 3.0 from Start, then Advanced > Repair crYOLO runtime.');
      Exit;
    end;
    if ExitCode <> 0 then
    begin
      RuntimeSetupFailed('The crYOLO runtime is not ready. Open SARU Picker 3.0 from Start, then Advanced > Repair crYOLO runtime. Details: ' +
        ExpandConstant('{localappdata}\SARU_Picker_crYOLO\runtime-install-report.json'));
      Exit;
    end;
    RuntimeReady := True;
  end;
end;
