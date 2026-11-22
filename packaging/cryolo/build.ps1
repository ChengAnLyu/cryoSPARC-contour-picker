#requires -Version 5.1
[CmdletBinding()]
param([string]$PythonExe = "python", [string]$Checkpoint = "", [string]$IsccPath = "", [switch]$TestInstaller)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne 'Windows_NT') { throw "Build this Windows application on Windows." }
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$BuildDirectory = Join-Path $ProjectRoot "build\cryolo"
$ArtifactDirectory = Join-Path $ProjectRoot "artifacts\cryolo"
$Venv = Join-Path $ProjectRoot ".build-venv-cryolo"
$BuildPython = Join-Path $Venv "Scripts\python.exe"
$Version = "3.0.0"
$Manifest = Get-Content (Join-Path $ProjectRoot 'packaging\model-manifest.json') -Raw | ConvertFrom-Json
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Program" }
}
function Assert-Checkpoint {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
        (Get-Item -LiteralPath $Path).Length -ne [long]$Manifest.size_bytes -or
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Manifest.sha256) {
        throw "SAM ViT-B checkpoint size or checksum mismatch: $Path"
    }
}
function Invoke-Frozen {
    param([string]$Executable, [string]$Arguments, [string]$Report, [int]$TimeoutSeconds = 600)
    if (Test-Path -LiteralPath $Report) { Remove-Item -LiteralPath $Report }
    $Process = Start-Process -FilePath $Executable -ArgumentList "$Arguments --report `"$Report`"" -PassThru
    if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "Frozen application check timed out: $Report"
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $Report)) { throw "Application check failed: $Report" }
    $Data = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    if ($Data.status -ne 'ok' -or $Data.version -ne $Version) { throw "Application check did not report $Version success: $Report" }
    return $Data
}
function Invoke-Diagnostics {
    param([string]$Executable, [string]$Report)
    $Data = Invoke-Frozen $Executable '--diagnostics' $Report
    if ($Data.backend -ne "sam") { throw "The default detector must remain SAM capsid." }
    $FileInfo = (Get-Item -LiteralPath $Executable).VersionInfo
    if ($FileInfo.FileVersion -ne $Version -or $FileInfo.ProductVersion -ne $Version -or
        $FileInfo.FileMajorPart -ne 3 -or $FileInfo.FileMinorPart -ne 0 -or $FileInfo.FileBuildPart -ne 0 -or
        $FileInfo.ProductName -ne 'SARU Picker') {
        throw "Executable Windows metadata does not identify SARU Picker $Version."
    }
    @{status='ok'; version=$Version; file_version=$FileInfo.FileVersion;
      product_version=$FileInfo.ProductVersion; product_name=$FileInfo.ProductName;
      original_filename=$FileInfo.OriginalFilename; default_backend=$Data.backend} |
      ConvertTo-Json | Set-Content -LiteralPath ($Report -replace '\.json$', '-pe-version.json') -Encoding UTF8
}
function Invoke-SamSmoke {
    param([string]$Executable, [string]$Report)
    $Result = Invoke-Frozen $Executable '--smoke-test --automatic-profile capsid' $Report
    foreach ($Name in @('image_processing','torchvision_native','windows_runtime','sam_cpu_inference','capsid_profile')) {
        if (@($Result.checks | Where-Object { $_.name -eq $Name -and $_.status -eq 'ok' }).Count -ne 1) {
            throw "Required SAM/native check failed: $Name"
        }
    }
}
Push-Location $ProjectRoot
$PreviousModel = $env:SARU_MODEL_FILE
try {
    New-Item -ItemType Directory -Force -Path $BuildDirectory, $ArtifactDirectory | Out-Null
    Invoke-Checked $PythonExe @('-c', "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8")
    if (-not (Test-Path -LiteralPath $BuildPython)) { Invoke-Checked $PythonExe @('-m', 'venv', $Venv) }
    Invoke-Checked $BuildPython @('-m', 'pip', 'install', 'pip==26.3', 'setuptools==80.9.0', 'wheel==0.45.1')
    Invoke-Checked $BuildPython @('-m', 'pip', 'install', 'torch==2.5.1+cpu', 'torchvision==0.20.1+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
    Invoke-Checked $BuildPython @('-m', 'pip', 'install', '--no-build-isolation', '-r', (Join-Path $PSScriptRoot 'requirements-build.txt'))
    Invoke-Checked $BuildPython @('-m', 'pip', 'check')
    Invoke-Checked $BuildPython @('-c', "import torch,torchvision; assert torch.__version__=='2.5.1+cpu'; assert torchvision.__version__=='0.20.1+cpu'; assert torch.version.cuda is None")
    Invoke-Checked $BuildPython @('-m', 'pytest', '-q')
    Invoke-Checked $BuildPython @('packaging\cryolo\verify_gui.py', $ArtifactDirectory)
    if ($Checkpoint) {
        $ModelFile = (Resolve-Path -LiteralPath $Checkpoint).Path
    } else {
        $ModelCache = Join-Path $ProjectRoot '.build-cache\models'
        New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null
        $ModelFile = Join-Path $ModelCache $Manifest.filename
        if (-not (Test-Path -LiteralPath $ModelFile)) {
            $Partial = "$ModelFile.partial"
            try {
                Invoke-WebRequest -Uri $Manifest.url -OutFile $Partial -UseBasicParsing
                Assert-Checkpoint $Partial
                Move-Item -LiteralPath $Partial -Destination $ModelFile -Force
            } finally { if (Test-Path -LiteralPath $Partial) { Remove-Item -LiteralPath $Partial } }
        }
    }
    Assert-Checkpoint $ModelFile
    $env:SARU_MODEL_FILE = $ModelFile
    # Bundle the official private Python package; no installed Python or
    # machine-wide installer registration is needed on the user's computer.
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot '.build-cache') | Out-Null
    $PythonPackage = Join-Path $ProjectRoot '.build-cache\python.3.7.9.nupkg'
    if (-not (Test-Path -LiteralPath $PythonPackage)) {
        Invoke-WebRequest -Uri 'https://api.nuget.org/v3-flatcontainer/python/3.7.9/python.3.7.9.nupkg' -OutFile $PythonPackage -UseBasicParsing
    }
    if ((Get-Item -LiteralPath $PythonPackage).Length -ne 14758084 -or
        (Get-FileHash -LiteralPath $PythonPackage).Hash.ToLowerInvariant() -ne '029e1ea5beeb142cd0f2da4e75c45f49bd7e0c7b48eb5bb6c6947f2d1b665c5e') {
        throw 'The official private Python 3.7.9 package failed verification.'
    }
    Invoke-Checked $BuildPython @('packaging\cryolo\collect_notices.py', (Join-Path $BuildDirectory 'third-party-notices'))
    Invoke-Checked $BuildPython @('-c', "import pathlib,sys,zipfile; pathlib.Path(sys.argv[2]).write_bytes(zipfile.ZipFile(sys.argv[1]).read('tools/LICENSE.txt'))", $PythonPackage, (Join-Path $BuildDirectory 'third-party-notices\Python37-LICENSE.txt'))
    Invoke-Checked $BuildPython @('-m', 'PyInstaller', '--noconfirm', '--clean', '--distpath', 'dist', '--workpath', (Join-Path $BuildDirectory 'pyinstaller'), 'packaging\cryolo\saru_cryolo.spec')
    $Distribution = Join-Path $ProjectRoot 'dist\SARU_Picker_crYOLO'
    $Executable = Join-Path $Distribution 'SARU Picker crYOLO.exe'
    Assert-Checkpoint (Join-Path $Distribution "_internal\models\$($Manifest.filename)")
    foreach ($File in @(Get-ChildItem -LiteralPath $Distribution -Recurse -File | Where-Object { $_.Extension -in @('.h5','.pth','.pt') })) {
        if ($File.Name -ne $Manifest.filename) { throw "Public bundle contains unexpected model weights: $($File.Name)" }
    }
    foreach ($Library in @('tensorflow','keras','cryolo')) {
        if (Test-Path -LiteralPath (Join-Path $Distribution "_internal\$Library")) { throw "External runtime was bundled: $Library" }
    }
    foreach ($Runtime in @('MSVCP140.dll','VCRUNTIME140.dll','VCRUNTIME140_1.dll','torch\lib\fbgemm.dll','torch\lib\asmjit.dll','torch\lib\libiomp5md.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Distribution "_internal\$Runtime"))) { throw "Required native library is absent: $Runtime" }
    }
    # Check exactly the canonical DLLs that runtime setup will copy. Keeping
    # this evidence before Setup exposes signature failures even if setup fails.
    $VCFiles = @()
    $VCFailures = @()
    foreach ($Name in @('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','msvcp140_atomic_wait.dll','msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll','concrt140.dll')) {
        $Path = Join-Path $Distribution "_internal\$Name"
        if (-not (Test-Path -LiteralPath $Path)) { continue }
        $Signature = Get-AuthenticodeSignature -LiteralPath $Path
        $Subject = if ($Signature.SignerCertificate) { $Signature.SignerCertificate.Subject } else { '' }
        $Bytes = [IO.File]::ReadAllBytes($Path)
        $Offset = [BitConverter]::ToInt32($Bytes, 0x3c)
        $Machine = [BitConverter]::ToUInt16($Bytes, $Offset + 4)
        $VCFiles += @{name=$Name; signature_status=[string]$Signature.Status; signature_message=$Signature.StatusMessage; signer=$Subject; machine=$Machine; sha256=(Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()}
        if ($Signature.Status -ne 'Valid' -or $Subject -notmatch 'Microsoft Corporation' -or $Machine -ne 0x8664) {
            $VCFailures += "$Name : $($Signature.Status) : $($Signature.StatusMessage) : $Subject : machine=$Machine"
        }
    }
    $VCJson = @{status=$(if ($VCFailures.Count) { 'error' } else { 'ok' }); files=$VCFiles} | ConvertTo-Json -Depth 5
    $VCJson | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'bundled-vc-runtime.json') -Encoding UTF8
    Write-Host $VCJson
    if ($VCFailures.Count) { throw "Bundled VC runtime verification failed: $($VCFailures -join '; ')" }
    foreach ($File in @('setup-runtime.ps1','runtime_health.py','runtime-requirements-windows.txt','Install-crYOLO-Runtime.cmd')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $File) -Destination $Distribution -Force
    }
    Copy-Item -LiteralPath $PythonPackage -Destination $Distribution -Force
    Invoke-Diagnostics $Executable (Join-Path $ArtifactDirectory 'frozen-diagnostics.json')
    Invoke-SamSmoke $Executable (Join-Path $ArtifactDirectory 'frozen-sam-smoke.json')
    $Refinement = Invoke-Frozen $Executable '--cryolo-refinement-smoke' (Join-Path $ArtifactDirectory 'frozen-cryolo-refinement.json') 300
    if (-not $Refinement.frozen -or @($Refinement.checks | Where-Object { $_.status -eq 'ok' }).Count -ne 4) { throw 'Frozen positive/dense refinement checks were incomplete.' }
    Compress-Archive -Path $Distribution -DestinationPath (Join-Path $ArtifactDirectory "SARU-Picker-GUI-$Version-win64.zip") -Force
    if ($IsccPath) {
        Invoke-Checked $IsccPath @("/DAppVersion=$Version", "/DAppSourceDir=$Distribution", "/DAppOutputDir=$ArtifactDirectory", (Join-Path $PSScriptRoot 'installer.iss'))
        $Installer = Join-Path $ArtifactDirectory "SARU-Picker-Setup-$Version-win64.exe"
        if ($TestInstaller) {
            $UninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{FB137682-60A8-4BC1-AC91-518AE16F0E8E}_is1'
            if ((Test-Path "HKCU:\$UninstallKey") -or (Test-Path "HKLM:\$UninstallKey")) { throw 'Installer test requires a disposable machine without this app installed.' }
            $InstallDirectory = Join-Path $env:RUNNER_TEMP 'SARU Installed Test'
            $SidecarStage = Join-Path $env:RUNNER_TEMP 'SARU Sidecar Test'
            if ((Test-Path -LiteralPath $InstallDirectory) -or (Test-Path -LiteralPath $SidecarStage)) { throw 'Installer test requires new directories.' }
            New-Item -ItemType Directory -Path $SidecarStage | Out-Null
            $StagedInstaller = Join-Path $SidecarStage (Split-Path $Installer -Leaf)
            Copy-Item -LiteralPath $Installer -Destination $StagedInstaller
            'Synthetic copy fixture; this is not a trained model.' | Set-Content -LiteralPath (Join-Path $SidecarStage 'cryolo-capsids.h5') -Encoding ASCII
            '{"fixture":"copy check only"}' | Set-Content -LiteralPath (Join-Path $SidecarStage 'cryolo-config.json') -Encoding ASCII
            Invoke-Checked $BuildPython @('packaging\verify_capsid_profile_installation.py', 'make-fixture', '--directory', $SidecarStage, '--report', (Join-Path $ArtifactDirectory 'capsid-profile-fixture.json'))
            Copy-Item -LiteralPath (Join-Path $ProjectRoot 'assets\app-icon.ico'),(Join-Path $ProjectRoot 'assets\app-icon.png') -Destination $SidecarStage
            $InstallLog = Join-Path $ArtifactDirectory 'installer-smoke.log'
            $Process = Start-Process -FilePath $StagedInstaller -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=`"$InstallDirectory`" /LOG=`"$InstallLog`"" -Wait -PassThru
            $RuntimeReport = Join-Path $env:LOCALAPPDATA 'SARU_Picker_crYOLO\runtime-install-report.json'
            if (Test-Path -LiteralPath $RuntimeReport) {
                Copy-Item -LiteralPath $RuntimeReport -Destination (Join-Path $ArtifactDirectory 'installed-runtime-setup.json')
                Write-Host (Get-Content -LiteralPath $RuntimeReport -Raw)
            }
            if ($Process.ExitCode -ne 0) {
                Write-Host (Get-Content -LiteralPath $InstallLog -Tail 50 | Out-String)
                throw "Installer exited $($Process.ExitCode); inspect installed-runtime-setup.json and installer-smoke.log."
            }
            $RuntimeResult = Get-Content -LiteralPath $RuntimeReport -Raw | ConvertFrom-Json
            if ($RuntimeResult.status -ne 'ok') { throw 'Setup did not finish a healthy crYOLO runtime.' }
            $RuntimePython = Join-Path $env:LOCALAPPDATA 'SARU_Picker_crYOLO\cryolo-env\Scripts\python.exe'
            if (-not (Test-Path -LiteralPath $RuntimePython)) { throw 'Automatic private runtime is absent.' }
            $InstalledExe = Join-Path $InstallDirectory 'SARU Picker crYOLO.exe'
            Invoke-Diagnostics $InstalledExe (Join-Path $ArtifactDirectory 'installed-diagnostics.json')
            Invoke-SamSmoke $InstalledExe (Join-Path $ArtifactDirectory 'installed-sam-smoke.json')
            $Refinement = Invoke-Frozen $InstalledExe '--cryolo-refinement-smoke' (Join-Path $ArtifactDirectory 'installed-cryolo-refinement.json') 300
            if (-not $Refinement.frozen -or @($Refinement.checks | Where-Object { $_.status -eq 'ok' }).Count -ne 4) { throw 'Installed positive/dense refinement checks were incomplete.' }
            Invoke-Checked $BuildPython @('packaging\verify_capsid_profile_installation.py', 'verify', '--install-directory', $InstallDirectory, '--fixture-directory', $SidecarStage, '--runtime-report', (Join-Path $ArtifactDirectory 'installed-sam-smoke.json'), '--report', (Join-Path $ArtifactDirectory 'installed-capsid-profile.json'))
            $CopiedSidecars = @()
            foreach ($Sidecar in @('cryolo-capsids.h5','cryolo-config.json','capsid-profile.json','app-icon.ico','app-icon.png')) {
                $Subdirectory = if ($Sidecar -like 'app-icon.*') { '_internal\assets' } else { '_internal\models' }
                $InstalledSidecar = Join-Path (Join-Path $InstallDirectory $Subdirectory) $Sidecar
                if (-not (Test-Path -LiteralPath $InstalledSidecar)) { throw "Installer did not copy sidecar: $Sidecar" }
                if ((Get-FileHash -LiteralPath (Join-Path $SidecarStage $Sidecar)).Hash -ne (Get-FileHash -LiteralPath $InstalledSidecar).Hash) { throw "Installed sidecar differs: $Sidecar" }
                $CopiedSidecars += $Sidecar
            }
            @{status='ok'; version=$Version; copied_sidecars=$CopiedSidecars; fixture_scope='Synthetic private-sidecar copy checks and constant appearance-model inference; no trained detector accuracy.'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'installed-sidecar-copy.json') -Encoding UTF8
            # Crucial boundary check: the frozen, torch-bearing executable must
            # spawn the isolated TensorFlow process without inheriting its DLL path.
            $Synthetic = Join-Path $env:RUNNER_TEMP 'SARU Synthetic crYOLO'
            try {
                Invoke-Checked $RuntimePython @('packaging\cryolo\verify_runtime.py', $Synthetic, '--keep-fixture')
                Copy-Item -LiteralPath (Join-Path $SidecarStage 'capsid-profile.json') -Destination $Synthetic
                $Arguments = "--smoke-test --automatic-profile cryolo --checkpoint `"$(Join-Path $Synthetic 'synthetic-untrained.h5')`" --cryolo-config `"$(Join-Path $Synthetic 'synthetic-config.json')`" --cryolo-threshold 0.99"
                $Bridge = Invoke-Frozen $InstalledExe $Arguments (Join-Path $ArtifactDirectory 'installed-frozen-cryolo-smoke.json')
                if ($Bridge.backend -ne 'cryolo' -or @($Bridge.checks | Where-Object { $_.name -eq 'cryolo_inference' -and $_.status -eq 'ok' }).Count -ne 1) { throw 'Frozen GUI did not execute real crYOLO inference.' }
                Copy-Item -LiteralPath (Join-Path $Synthetic 'runtime-inference.json') -Destination (Join-Path $ArtifactDirectory 'installed-native-cryolo-smoke.json')
            } finally { if (Test-Path -LiteralPath $Synthetic) { Remove-Item -LiteralPath $Synthetic -Recurse -Force } }
            $Uninstaller = Start-Process -FilePath (Join-Path $InstallDirectory 'unins000.exe') -ArgumentList '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART' -Wait -PassThru
            if ($Uninstaller.ExitCode -ne 0) { throw "Uninstaller exited $($Uninstaller.ExitCode)." }
            Remove-Item -LiteralPath $SidecarStage -Recurse -Force
        }
    }
    @(Get-ChildItem -LiteralPath $ArtifactDirectory -File | Where-Object { $_.Extension -in @('.exe','.zip') } | ForEach-Object {
        "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)"
    }) | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'SHA256SUMS.txt') -Encoding ASCII
    @{status='ok'; version=$Version; default_profile='capsid'; profiles=@('capsid','capsid-learned','generic','crYOLO'); sam_runtime_and_vit_b_bundled=$true; private_python_package_bundled=$true; private_python_version='3.7.9'; private_weights_bundled=$false; cryolo_runtime_bundled=$false; cryolo_runtime_automatically_installed=$true; frozen_sam_inference_tested=$true; frozen_positive_dense_refinement_tested=$true; installed_positive_dense_refinement_tested=[bool]($TestInstaller -and $IsccPath); installed_cryolo_inference_tested=[bool]($TestInstaller -and $IsccPath); model_accuracy_tested=$false; note='Synthetic runtime checks only; installed crYOLO boundary verification requires -TestInstaller. No private trained-model accuracy claim.'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ArtifactDirectory 'build-manifest.json') -Encoding UTF8
    Invoke-Checked $BuildPython @('packaging\cryolo\prepare_downloads.py', $ArtifactDirectory, '--version', $Version)
} finally { $env:SARU_MODEL_FILE = $PreviousModel; Pop-Location }
