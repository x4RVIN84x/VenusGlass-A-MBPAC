[CmdletBinding()]
param(
    [switch]$CreateInstaller,
    [switch]$SkipIconGeneration,
    [string]$InnoSetupCompiler = "",
    [ValidateSet("Auto", "MSVC", "MinGW64")]
    [string]$Compiler = "MSVC"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$EntryPoint = Join-Path $RepositoryRoot "vg_vista.py"
$QmlDirectory = Join-Path $RepositoryRoot "hmi_app\qml"
$RecipesDirectory = Join-Path $RepositoryRoot "recipes"
$IconPath = Join-Path $QmlDirectory "assets\vg-vista.ico"
$BuildDirectory = Join-Path $RepositoryRoot "build\vg-vista"
$InstallerScript = Join-Path $PSScriptRoot "VG_VISTA.iss"

function Require-File([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not found: $Path"
    }
}

function Find-InnoSetupCompiler([string]$PreferredPath) {
    $candidates = @()
    if ($PreferredPath) {
        $candidates += $PreferredPath
    }

    if (${env:ProgramFiles(x86)}) {
        $candidates += (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe")
    }
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    }

    return $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}

Require-File $Python "Project virtual-environment Python"
Require-File $EntryPoint "VG VISTA entry point"
Require-File $InstallerScript "Inno Setup installer definition"
if (-not (Test-Path -LiteralPath $QmlDirectory -PathType Container)) {
    throw "VG VISTA QML directory was not found: $QmlDirectory"
}
if (-not (Test-Path -LiteralPath $RecipesDirectory -PathType Container)) {
    throw "Recipe directory was not found: $RecipesDirectory"
}

if (-not $SkipIconGeneration) {
    & $Python (Join-Path $PSScriptRoot "create_windows_icon.py") `
        --source (Join-Path $QmlDirectory "assets\venus-glass-logo.png") `
        --destination $IconPath
    if ($LASTEXITCODE -ne 0) {
        throw "VG VISTA icon generation failed."
    }
}
Require-File $IconPath "VG VISTA Windows icon"

$Version = (& $Python -c "from hmi_app.vista_metadata import PRODUCT_VERSION; print(PRODUCT_VERSION)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $Version) {
    throw "Could not read PRODUCT_VERSION from hmi_app.vista_metadata."
}
if ($Version -notmatch '^\d+\.\d+\.\d+$') {
    throw "PRODUCT_VERSION must use major.minor.patch so Windows version metadata can be created."
}
$FileVersion = "$Version.0"

$PythonBitness = (& $Python -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($LASTEXITCODE -ne 0 -or $PythonBitness -ne "64") {
    throw "VG VISTA's installer targets 64-bit Windows; use a 64-bit Python virtual environment."
}

New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null

$nuitkaArguments = @(
    "-m",
    "nuitka",
    "--standalone",
    "--enable-plugin=pyside6",
    "--assume-yes-for-downloads",
    # Nuitka's PySide6 plugin supplies its sensible platform plugins; QML
    # additionally packages QtQuick and QtQuick.Controls import modules.
    "--include-qt-plugins=sensible,qml",
    "--include-data-dir=$QmlDirectory=hmi_app/qml",
    "--include-data-dir=$RecipesDirectory=recipes",
    "--output-dir=$BuildDirectory",
    # Leave the physical filename at Nuitka's stable script-derived default
    # (vg_vista.exe). The installed product and all shortcuts remain branded
    # "VG VISTA" through the Windows version metadata and Inno Setup.
    "--windows-console-mode=disable",
    "--windows-icon-from-ico=$IconPath",
    "--company-name=Venus Glass",
    "--product-name=VG VISTA",
    "--file-description=VG VISTA Industrial Vision and Inspection",
    "--file-version=$FileVersion",
    "--product-version=$FileVersion",
    "--copyright=Copyright (c) 2026 Venus Glass",
    "--nofollow-import-to=tests",
    $EntryPoint
)

if ($Compiler -eq "MSVC") {
    # Release builds normally use the Microsoft toolchain.  Keep the explicit
    # option so a build agent never silently changes compiler family.
    $nuitkaArguments += "--msvc=latest"
} elseif ($Compiler -eq "MinGW64") {
    $nuitkaArguments += "--mingw64"
}

Write-Host "Building VG VISTA $Version with Nuitka..." -ForegroundColor Cyan
& $Python @nuitkaArguments
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed."
}

$distribution = Get-ChildItem -LiteralPath $BuildDirectory -Directory -Filter "*.dist" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $distribution) {
    throw "Nuitka completed but no standalone .dist directory was found in $BuildDirectory."
}

Write-Host "Standalone application: $($distribution.FullName)" -ForegroundColor Green

if ($CreateInstaller) {
    $iscc = Find-InnoSetupCompiler $InnoSetupCompiler
    if (-not $iscc) {
        throw "Inno Setup 6 was not found. Install it on the build workstation or pass -InnoSetupCompiler <path-to-ISCC.exe>."
    }

    Write-Host "Creating signed-ready installer with Inno Setup..." -ForegroundColor Cyan
    & $iscc "/DSourceDir=$($distribution.FullName)" "/DAppVersion=$Version" $InstallerScript
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup compilation failed."
    }

    Write-Host "Installer output: $(Join-Path $RepositoryRoot 'dist\installer')" -ForegroundColor Green
}
