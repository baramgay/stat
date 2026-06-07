; NuriStat 설치 마법사 스크립트 (Inno Setup 6)
; PyInstaller onedir 산출물(dist\NuriStat)을 설치파일로 패키징한다.
; 빌드: ISCC.exe NuriStat.iss  → Output\NuriStat_Setup_v<버전>.exe

#define MyAppName "NuriStat"
#define MyAppNameKo "누리스탯"
#define MyAppVersion "3.4.3"
#define MyAppPublisher "경남빅데이터센터"
#define MyAppExeName "NuriStat.exe"

[Setup]
; 고정 AppId (버전 업그레이드 시 동일 유지 — 자동 업그레이드 인식)
AppId={{9F2C4E10-7A3B-4D6E-9C1F-2B5A8E0D4C77}
AppName={#MyAppNameKo} {#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppNameKo} {#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppNameKo}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=NuriStat_Setup_v{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppNameKo} {#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
; 일반 사용자도 설치 가능하도록 (관리자 권한 불필요 시 lowest)
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
; PyInstaller 산출물 전체 (Python 인터프리터·라이브러리 포함 — 별도 런타임 불필요)
Source: "dist\NuriStat\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppNameKo}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppNameKo}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppNameKo}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppNameKo}}"; Flags: nowait postinstall skipifsilent
