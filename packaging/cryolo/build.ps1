#requires -Version 5.1
[CmdletBinding()]
param([string]$PythonExe = "python", [string]$IsccPath = "", [switch]$TestInstaller)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne 'Windows_NT') { throw "Build this Windows application on Windows." }
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$BuildDirectory = Join-Path $ProjectRoot "build\cryolo"
$ArtifactDirectory = Join-Path $ProjectRoot "artifacts\cryolo"
$Venv = Join-Path $ProjectRoot ".build-venv-cryolo"
$BuildPython = Join-Path $Venv "Scripts\python.exe"
$Version = "2.9.0-cryolo.1"
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Program" }
}
function Invoke-Diagnostics {
    param([string]$Executable, [string]$Report)
    $Process = Start-Process -FilePath $Executable -ArgumentList "--diagnostics --report `"$Report`"" -PassThru
    if (-not $Process.WaitForExit(120000)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "GUI diagnostics timed out."
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $Report)) { throw "GUI diagnostics failed." }
    $Data = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    if ($Data.status -ne "ok" -or $Data.backend -ne "cryolo") { throw "Wrong or failed backend in diagnostics." }
}
Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $BuildDirectory, $ArtifactDirectory | Out-Null
    Invoke-Checked $PythonExe @("-c", "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8")
    if (-not (Test-Path -LiteralPath $BuildPython)) { Invoke-Checked $PythonExe @("-m", "venv", $Venv) }
    Invoke-Checked $BuildPython @("-m", "pip", "install", "pip==26.3", "setuptools==80.9.0", "wheel==0.45.1")
    Invoke-Checked $BuildPython @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements-build.txt"))
    Invoke-Checked $BuildPython @("-m", "pip", "check")
    Invoke-Checked $BuildPython @("-m", "pytest", "-q")
    Invoke-Checked $BuildPython @("packaging\cryolo\verify_gui.py", $ArtifactDirectory)
    Invoke-Checked $BuildPython @("packaging\cryolo\collect_notices.py", (Join-Path $BuildDirectory "third-party-notices"))
    Invoke-Checked $BuildPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", "dist", "--workpath", (Join-Path $BuildDirectory "pyinstaller"), "packaging\cryolo\saru_cryolo.spec")
    $Distribution = Join-Path $ProjectRoot "dist\SARU_Picker_crYOLO"
    $Executable = Join-Path $Distribution "SARU Picker crYOLO.exe"
    if (@(Get-ChildItem -LiteralPath $Distribution -Recurse -File | Where-Object { $_.Extension -in @('.h5','.pth','.pt') }).Count -ne 0) {
        throw "Public GUI bundle unexpectedly contains private or third-party model weights."
    }
    foreach ($Library in @('torch','tensorflow','keras','cryolo','segment_anything')) {
        if (Test-Path -LiteralPath (Join-Path $Distribution "_internal\$Library")) { throw "Unexpected model runtime bundled: $Library" }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'setup-runtime.ps1'),(Join-Path $PSScriptRoot 'runtime-requirements-windows.txt'),(Join-Path $PSScriptRoot 'Install-crYOLO-Runtime.cmd') -Destination $Distribution -Force
    Invoke-Diagnostics $Executable (Join-Path $ArtifactDirectory 'frozen-diagnostics.json')
    Compress-Archive -Path $Distribution -DestinationPath (Join-Path $ArtifactDirectory "SARU-Picker-crYOLO-GUI-$Version-win64.zip") -Force
    if ($IsccPath) {
        Invoke-Checked $IsccPath @("/DAppVersion=$Version", "/DAppSourceDir=$Distribution", "/DAppOutputDir=$ArtifactDirectory", (Join-Path $PSScriptRoot "installer.iss"))
        $Installer = Join-Path $ArtifactDirectory "SARU-Picker-crYOLO-Setup-$Version-win64.exe"
        if ($TestInstaller) {
            $InstallDirectory = Join-Path $env:RUNNER_TEMP 'SARU crYOLO Installed Test'
            if (Test-Path -LiteralPath $InstallDirectory) { throw "Installer test requires a new directory." }
            $Process = Start-Process -FilePath $Installer -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=`"$InstallDirectory`"" -Wait -PassThru
            if ($Process.ExitCode -ne 0) { throw "Installer exited $($Process.ExitCode)." }
            Invoke-Diagnostics (Join-Path $InstallDirectory 'SARU Picker crYOLO.exe') (Join-Path $ArtifactDirectory 'installed-diagnostics.json')
            $Uninstaller = Start-Process -FilePath (Join-Path $InstallDirectory 'unins000.exe') -ArgumentList '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -Wait -PassThru
            if ($Uninstaller.ExitCode -ne 0) { throw "Uninstaller exited $($Uninstaller.ExitCode)." }
        }
    }
    $Hashes = @(Get-ChildItem -LiteralPath $ArtifactDirectory -File | Where-Object { $_.Extension -in @('.exe','.zip') } | ForEach-Object {
        $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$Hash  $($_.Name)"
    })
    $Hashes | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'SHA256SUMS.txt') -Encoding ASCII
    @{ status = 'ok'; version = $Version; backend = 'cryolo'; private_weights_bundled = $false;
       cryolo_runtime_bundled = $false; model_accuracy_tested = $false;
       note = 'GUI/native interface/installer validation only. Separately installed crYOLO and private model are required for detection.' } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'build-manifest.json') -Encoding UTF8
} finally { Pop-Location }
