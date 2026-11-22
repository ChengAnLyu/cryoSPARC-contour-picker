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
; Private learned weights are supplied beside Setup. Verified public SAM ViT-B
; is already bundled and does not need a separate sidecar.
Source: "{src}\cryolo-capsids.onnx"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\cryolo-config.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\capsid-profile.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.ico"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.png"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\SARU Picker 3.0"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"
Name: "{group}\Quick start"; Filename: "{app}\_internal\QUICK_START.txt"
Name: "{autodesktop}\SARU Picker 3.0"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"; Tasks: desktopicon

[InstallDelete]
; Remove only obsolete helper files installed by this same edition.
; The old private runtime environment and user data remain untouched.
Type: files; Name: "{app}\setup-runtime.ps1"
Type: files; Name: "{app}\Install-crYOLO-Runtime.cmd"
Type: files; Name: "{app}\runtime-requirements-windows.txt"
Type: files; Name: "{app}\runtime_health.py"
Type: files; Name: "{app}\python.3.7.9.nupkg"
; Remove only shortcuts created by the prior edition owned by this same AppId.
Type: files; Name: "{userprograms}\SARU Picker crYOLO\SARU Picker crYOLO.lnk"
Type: files; Name: "{userprograms}\SARU Picker crYOLO\Install crYOLO runtime.lnk"
Type: files; Name: "{userprograms}\SARU Picker crYOLO\Quick start.lnk"
Type: files; Name: "{userdesktop}\SARU Picker crYOLO.lnk"

[Run]
Filename: "{app}\SARU Picker crYOLO.exe"; Description: "Launch SARU Picker 3.0"; Flags: nowait postinstall skipifsilent
