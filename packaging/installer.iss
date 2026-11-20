; Inno Setup 6.3+ / 7. Build through packaging/build.ps1.
#ifndef AppVersion
  #define AppVersion "2.7.0"
#endif
#ifndef AppSourceDir
  #define AppSourceDir "..\dist\SARU_Picker"
#endif
#ifndef AppOutputDir
  #define AppOutputDir "..\artifacts"
#endif

[Setup]
AppId={{A7D0C866-7ED3-4EB2-8A49-6C6B13E84876}
AppName=SARU Picker
AppVersion={#AppVersion}
AppPublisher=SARU Picker
AppVerName=SARU Picker {#AppVersion}
DefaultDirName={localappdata}\Programs\SARU Picker
DefaultGroupName=SARU Picker
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
OutputDir={#AppOutputDir}
OutputBaseFilename=SARU-Picker-Setup-{#AppVersion}-win64
SetupIconFile=..\assets\app-icon.ico
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
; Optional sidecars remain beside Setup and are never part of the public build.
Source: "{src}\app-icon.png"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist
Source: "{src}\app-icon.ico"; DestDir: "{app}\_internal\assets"; Flags: external ignoreversion skipifsourcedoesntexist
; The learned capsid profile stays private and is read as bounded JSON data.
Source: "{src}\capsid-profile.json"; DestDir: "{app}\_internal\models"; Flags: external ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\SARU Picker"; Filename: "{app}\SARU Picker.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.Desktop"
Name: "{group}\Quick start"; Filename: "{app}\_internal\QUICK_START.txt"
Name: "{autodesktop}\SARU Picker"; Filename: "{app}\SARU Picker.exe"; WorkingDir: "{app}"; IconFilename: "{app}\_internal\assets\app-icon.ico"; AppUserModelID: "SARU.Picker.Desktop"; Tasks: desktopicon

[Run]
Filename: "{app}\SARU Icon Helper.exe"; Parameters: "--target ""{app}\SARU Picker.exe"" --icon ""{app}\_internal\assets\app-icon.ico"" --report ""{localappdata}\SARU_Picker\icon-install.json"""; Flags: runhidden waituntilterminated
Filename: "{app}\SARU Picker.exe"; Description: "Launch SARU Picker"; Flags: nowait postinstall skipifsilent
