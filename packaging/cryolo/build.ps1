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
$Version = "3.0.0"
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
    if ($Data.status -ne "ok" -or $Data.backend -ne "cryolo" -or $Data.version -ne $Version) {
        throw "Diagnostics must report crYOLO $Version by default."
    }
    $FileInfo = (Get-Item -LiteralPath $Executable).VersionInfo
    if ($FileInfo.FileVersion -ne $Version -or $FileInfo.ProductVersion -ne $Version -or
        $FileInfo.FileMajorPart -ne 3 -or $FileInfo.FileMinorPart -ne 0 -or $FileInfo.FileBuildPart -ne 0 -or
        $FileInfo.ProductName -ne 'SARU Picker crYOLO') {
        throw "Executable Windows metadata does not identify SARU Picker crYOLO $Version."
    }
    @{status='ok'; version=$Version; file_version=$FileInfo.FileVersion;
      product_version=$FileInfo.ProductVersion; product_name=$FileInfo.ProductName;
      original_filename=$FileInfo.OriginalFilename; default_backend=$Data.backend} |
      ConvertTo-Json | Set-Content -LiteralPath ($Report -replace '\.json$', '-pe-version.json') -Encoding UTF8
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
            # Exercise automatic external-sidecar installation using clearly
            # synthetic fixtures in a temporary directory. No fixture becomes
            # part of a user-facing artifact and no model inference is claimed.
            $SidecarStage = Join-Path $env:RUNNER_TEMP 'SARU crYOLO Sidecar Test'
            if (Test-Path -LiteralPath $SidecarStage) { throw "Sidecar test requires a new staging directory." }
            New-Item -ItemType Directory -Path $SidecarStage | Out-Null
            $StagedInstaller = Join-Path $SidecarStage (Split-Path $Installer -Leaf)
            Copy-Item -LiteralPath $Installer -Destination $StagedInstaller
            'Synthetic installer-copy fixture; this is not a trained model.' |
                Set-Content -LiteralPath (Join-Path $SidecarStage 'cryolo-capsids.h5') -Encoding ASCII
            '{"fixture":"synthetic installer-copy test; not an inference config"}' |
                Set-Content -LiteralPath (Join-Path $SidecarStage 'cryolo-config.json') -Encoding ASCII
            '{"fixture":"synthetic installer-copy test; not an appearance profile"}' |
                Set-Content -LiteralPath (Join-Path $SidecarStage 'capsid-profile.json') -Encoding ASCII
            Copy-Item -LiteralPath (Join-Path $ProjectRoot 'assets\app-icon.ico'),(Join-Path $ProjectRoot 'assets\app-icon.png') -Destination $SidecarStage
            $Process = Start-Process -FilePath $StagedInstaller -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=`"$InstallDirectory`"" -Wait -PassThru
            if ($Process.ExitCode -ne 0) { throw "Installer exited $($Process.ExitCode)." }
            Invoke-Diagnostics (Join-Path $InstallDirectory 'SARU Picker crYOLO.exe') (Join-Path $ArtifactDirectory 'installed-diagnostics.json')
            $CopiedSidecars = @()
            foreach ($Sidecar in @('cryolo-capsids.h5','cryolo-config.json','capsid-profile.json','app-icon.ico','app-icon.png')) {
                $Subdirectory = if ($Sidecar -like 'app-icon.*') { '_internal\assets' } else { '_internal\models' }
                $InstalledSidecar = Join-Path (Join-Path $InstallDirectory $Subdirectory) $Sidecar
                if (-not (Test-Path -LiteralPath $InstalledSidecar)) { throw "Installer did not copy sidecar: $Sidecar" }
                $ExpectedHash = (Get-FileHash -LiteralPath (Join-Path $SidecarStage $Sidecar) -Algorithm SHA256).Hash
                $InstalledHash = (Get-FileHash -LiteralPath $InstalledSidecar -Algorithm SHA256).Hash
                if ($ExpectedHash -ne $InstalledHash) { throw "Installed sidecar bytes differ: $Sidecar" }
                $CopiedSidecars += $Sidecar
            }
            @{status='ok'; version=$Version; copied_sidecars=$CopiedSidecars;
              fixture_scope='Synthetic model/config/profile fixtures and public icon assets; copy integrity only, no trained-model inference.'} |
              ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'installed-sidecar-copy.json') -Encoding UTF8
            $Uninstaller = Start-Process -FilePath (Join-Path $InstallDirectory 'unins000.exe') -ArgumentList '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -Wait -PassThru
            if ($Uninstaller.ExitCode -ne 0) { throw "Uninstaller exited $($Uninstaller.ExitCode)." }
            Remove-Item -LiteralPath $SidecarStage -Recurse -Force
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
