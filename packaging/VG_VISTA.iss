; VG VISTA installer definition.  Compile through build_vg_vista.ps1 so the
; application and installer versions remain synchronized.

#ifndef AppVersion
  #define AppVersion "1.6.1"
#endif

#ifndef SourceDir
  #define SourceDir "..\build\vg-vista\VG VISTA.dist"
#endif

#define AppName "VG VISTA"
#define AppPublisher "Venus Glass"
#define AppExeName "vg_vista.exe"

[Setup]
AppId={{4B6D49A4-5B73-4D05-A05E-BBB098E55483}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=Copyright (c) 2026 Venus Glass
DefaultDirName={autopf}\Venus Glass\VG VISTA
DefaultGroupName=Venus Glass
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=VG-VISTA-Setup-{#AppVersion}
SetupIconFile=..\hmi_app\qml\assets\vg-vista.ico
LicenseFile=..\LICENSE
UninstallDisplayIcon={app}\{#AppExeName}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription=VG VISTA Industrial Vision and Inspection Installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}.0
VersionInfoVersion={#AppVersion}.0
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Inspection history is deliberately stored outside Program Files so it
; survives upgrades and remains writable for the logged-in operator.
Name: "{localappdata}\Venus Glass\VG VISTA\data"

[Icons]
Name: "{autoprograms}\Venus Glass\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; AppUserModelID: "VenusGlass.VGVista"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; AppUserModelID: "VenusGlass.VGVista"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
