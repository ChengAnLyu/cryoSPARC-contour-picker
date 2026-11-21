#ifndef AppVersion
  #define AppVersion "2.9.0-cryolo.1"
#endif
#ifndef AppSourceDir
  #define AppSourceDir "..\..\dist\SARU_Picker_crYOLO"
#endif
#ifndef AppOutputDir
  #define AppOutputDir "..\..\artifacts\cryolo"
#endif

[Setup]
AppId={{FB137682-60A8-4BC1-AC91-518AE16F0E8E}
AppName=SARU Picker crYOLO
AppVersion={#AppVersion}
AppPublisher=SARU Picker
AppVerName=SARU Picker crYOLO {#AppVersion}
DefaultDirName={localappdata}\Programs\SARU Picker crYOLO
DefaultGroupName=SARU Picker crYOLO
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#AppOutputDir}
OutputBaseFilename=SARU-Picker-crYOLO-Setup-{#AppVersion}-win64
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
; These user's sidecars are supplied privately beside Setup, not in the public build.
Source: "{src}\cryolo-capsids.h5"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\cryolo-config.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\capsid-profile.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.ico"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.png"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\SARU Picker crYOLO"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"
Name: "{group}\Install crYOLO runtime"; Filename: "{app}\Install-crYOLO-Runtime.cmd"; WorkingDir: "{app}"
Name: "{group}\Quick start"; Filename: "{app}\_internal\QUICK_START.txt"
Name: "{autodesktop}\SARU Picker crYOLO"; Filename: "{app}\SARU Picker crYOLO.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.crYOLO"; Tasks: desktopicon

[Run]
Filename: "{app}\SARU Picker crYOLO.exe"; Description: "Launch SARU Picker crYOLO"; Flags: nowait postinstall skipifsilent
