; Inno Setup 6.3+ / 7. Build through packaging/build.ps1.
#ifndef AppVersion
  #define AppVersion "2.1.0"
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
UninstallDisplayIcon={app}\SARU Picker.exe
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

[Icons]
Name: "{group}\SARU Picker"; Filename: "{app}\SARU Picker.exe"; WorkingDir: "{app}"
Name: "{group}\Quick start"; Filename: "{app}\_internal\QUICK_START.txt"
Name: "{autodesktop}\SARU Picker"; Filename: "{app}\SARU Picker.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\SARU Picker.exe"; Description: "Launch SARU Picker"; Flags: nowait postinstall skipifsilent
