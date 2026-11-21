#requires -Version 5.1
<#
.SYNOPSIS
Automatically prepare the private stock crYOLO CPU runtime for this user.
.DESCRIPTION
Healthy runtimes are reused offline. New installations extract the official PSF
NuGet Python package without registry, PATH, launcher or association changes. Only the app's
private environment is repaired. crYOLO itself remains unmodified upstream code.
#>
[CmdletBinding()]
param(
    [string]$Python37 = "", # Hidden command-line override; never requested by GUI.
    [string]$RuntimeDirectory = (Join-Path $env:LOCALAPPDATA "SARU_Picker_crYOLO\cryolo-env"),
    [string]$Report = "",
    [switch]$ForcePrivatePython,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($env:OS -ne "Windows_NT") { throw "This setup is for Windows x64." }
if ($ForcePrivatePython -and $Python37) { throw "ForcePrivatePython cannot be combined with Python37." }

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Program" }
}
function Read-Health {
    if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) { return $null }
    $ProbeOutput = Join-Path $ParentDirectory "runtime-health.json"
    $ProbeError = Join-Path $ParentDirectory "runtime-health.log"
    $Arguments = '"' + (Join-Path $PSScriptRoot "runtime_health.py") + '"'
    try {
        $Probe = Start-Process -FilePath $RuntimePython -ArgumentList $Arguments -PassThru -Wait -NoNewWindow -RedirectStandardOutput $ProbeOutput -RedirectStandardError $ProbeError
        if ($Probe.ExitCode -ne 0) { return $null }
    } catch { return $null }
    try {
        $Health = Get-Content -LiteralPath $ProbeOutput -Raw | ConvertFrom-Json
        if ($Health.status -eq "ok") { return $Health }
    } catch { return $null }
    return $null
}
function Assert-NoReparsePoint {
    param([string]$Path)
    if ((Test-Path -LiteralPath $Path) -and ((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "The private runtime location cannot be a link or junction: $Path"
    }
}
function Save-Report {
    $InstallReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Report -Encoding UTF8
}
function Ensure-BackendWeights {
    $Backend = Join-Path $RuntimeDirectory "Lib\site-packages\cryolo\full_yolo_backend.h5"
    $Expected = "0eb044b4018be4b8c530389843d9650569751439fdab2880f3020e65ff2ff358"
    if ((Test-Path -LiteralPath $Backend) -and ((Get-FileHash -LiteralPath $Backend -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected)) { return }
    $Temporary = "$Backend.download"
    Invoke-WebRequest -Uri "https://ndownloader.figshare.com/files/16406843" -OutFile $Temporary -UseBasicParsing
    if ((Get-FileHash -LiteralPath $Temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
        Remove-Item -LiteralPath $Temporary -Force
        throw "The upstream crYOLO initialization weights failed their checksum."
    }
    Move-Item -LiteralPath $Temporary -Destination $Backend -Force
}

function Copy-PrivateVCRuntime {
    param([string]$Destination)
    # The combined GUI carries licensed Microsoft redistributable DLLs. Copy only
    # canonical signed x64 VC14 runtime files into this app's private Python.
    $Allowed = @('msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
        'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
        'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll')
    foreach ($Name in $Allowed) {
        foreach ($Directory in @($PSScriptRoot, (Join-Path $PSScriptRoot '_internal'))) {
            $Source = Join-Path $Directory $Name
            if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { continue }
            $Signature = Get-AuthenticodeSignature -LiteralPath $Source
            if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
                throw "The bundled Microsoft runtime file has an invalid signature: $Name"
            }
            $Bytes = [IO.File]::ReadAllBytes($Source)
            $PE = [BitConverter]::ToInt32($Bytes, 0x3c)
            if ([BitConverter]::ToUInt16($Bytes, $PE + 4) -ne 0x8664) { throw "Expected an x64 Microsoft runtime file: $Name" }
            $Target = Join-Path $Destination $Name
            $SourceVersion = (Get-Item -LiteralPath $Source).VersionInfo
            $SourceVersion = [version](('{0}.{1}.{2}.{3}' -f $SourceVersion.FileMajorPart, $SourceVersion.FileMinorPart, $SourceVersion.FileBuildPart, $SourceVersion.FilePrivatePart))
            $TargetVersion = [version]'0.0.0.0'
            if (Test-Path -LiteralPath $Target) {
                $Info = (Get-Item -LiteralPath $Target).VersionInfo
                $TargetVersion = [version](('{0}.{1}.{2}.{3}' -f $Info.FileMajorPart, $Info.FileMinorPart, $Info.FileBuildPart, $Info.FilePrivatePart))
            }
            if (-not (Test-Path -LiteralPath $Target) -or $SourceVersion -gt $TargetVersion) {
                Copy-Item -LiteralPath $Source -Destination $Target -Force
            }
            $InstallReport.vc_runtime_files += @{ name = $Name; sha256 = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant() }
            break
        }
    }
}

$RuntimeDirectory = [IO.Path]::GetFullPath($RuntimeDirectory)
$ParentDirectory = Split-Path $RuntimeDirectory -Parent
$RuntimePython = Join-Path $RuntimeDirectory "Scripts\python.exe"
$OwnerMarker = Join-Path $RuntimeDirectory "saru-cryolo-runtime.json"
$Started = [DateTime]::UtcNow.ToString("o")
if (-not $Report) { $Report = Join-Path $ParentDirectory "runtime-install-report.json" }
$Report = [IO.Path]::GetFullPath($Report)
Assert-NoReparsePoint $ParentDirectory
Assert-NoReparsePoint $RuntimeDirectory
New-Item -ItemType Directory -Force -Path $ParentDirectory | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Report -Parent) | Out-Null
$InstallReport = @{ status = "running"; stage = "health-check"; started_utc = $Started;
    python = $RuntimePython; reused = $false; vc_runtime_files = @(); private_python_requested = [bool]$ForcePrivatePython;
    cryolo_version = "1.9.9"; tensorflow_version = "1.15.4";
    scope = "private CPU runtime and upstream initialization weights; no trained capsid model" }
$Lock = $null
$OldEnvironment = @{}
foreach ($Name in @('CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS', 'PYTHONNOUSERSITE', 'PYTHONPATH', 'PYTHONHOME', 'PIP_CONFIG_FILE')) {
    $OldEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}
try {
    # An interrupted process releases the OS lock; first-use and installer serialize.
    $Deadline = [DateTime]::UtcNow.AddMinutes(45)
    while (-not $Lock) {
        try { $Lock = [IO.File]::Open((Join-Path $ParentDirectory "runtime-setup.lock"), 'OpenOrCreate', 'ReadWrite', 'None') }
        catch [IO.IOException] {
            if ([DateTime]::UtcNow -gt $Deadline) { throw "Another crYOLO setup is still running." }
            Start-Sleep -Milliseconds 500
        }
    }
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:TF_CPP_MIN_LOG_LEVEL = "2"
    $env:OMP_NUM_THREADS = "4"
    $env:PYTHONNOUSERSITE = "1"
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PIP_CONFIG_FILE = "NUL"
    $Health = Read-Health
    if ($Health) {
        $InstallReport.status = "ok"
        $InstallReport.stage = "ready"
        $InstallReport.reused = $true
        $InstallReport.health = $Health
        $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
        Save-Report
        Write-Host "crYOLO runtime is ready. Reusing the existing installation."
        exit 0
    }
    if ($CheckOnly) { throw "The private crYOLO runtime is not ready." }
    if (Test-Path -LiteralPath $RuntimeDirectory) {
        $Owned = $false
        if (Test-Path -LiteralPath $OwnerMarker) {
            try { $Owned = (Get-Content -LiteralPath $OwnerMarker -Raw | ConvertFrom-Json).owner -eq 'SARU Picker crYOLO' } catch { $Owned = $false }
        }
        if (-not $Owned -and @(Get-ChildItem -LiteralPath $RuntimeDirectory -Force).Count -gt 0) {
            throw "An unrelated installation occupies the private crYOLO runtime folder. It was left unchanged: $RuntimeDirectory"
        }
    }
    foreach ($Child in @('Scripts', 'Lib', 'Include')) { Assert-NoReparsePoint (Join-Path $RuntimeDirectory $Child) }
    New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null
    # Ownership is recorded before venv creation so interrupted setup can be repaired.
    @{ owner = "SARU Picker crYOLO"; created_utc = $Started; schema = 1 } | ConvertTo-Json | Set-Content -LiteralPath $OwnerMarker -Encoding UTF8
    Write-Host "Preparing crYOLO automatically. This may take several minutes."
    Write-Host "Upstream license: https://cryolo-doc.readthedocs.io/en/stable/other/license.html"
    $InstallReport.stage = "python"
    Save-Report
    $BaseDirectory = $null
    if (-not $Python37) {
        $BaseDirectory = Join-Path $ParentDirectory "python37"
        Assert-NoReparsePoint $BaseDirectory
        $Python37 = Join-Path $BaseDirectory "python.exe"
        $BaseHealthy = $false
        if (Test-Path -LiteralPath $Python37 -PathType Leaf) {
            try {
                & $Python37 -c "import sys,struct,venv,ensurepip; assert sys.version_info[:2] == (3,7) and struct.calcsize('P') == 8" 2>$null
                $BaseHealthy = $LASTEXITCODE -eq 0
            } catch { $BaseHealthy = $false }
        }
        if (-not $BaseHealthy) {
            $BaseMarker = Join-Path $BaseDirectory "saru-private-python.json"
            if ((Test-Path -LiteralPath $BaseDirectory) -and -not (Test-Path -LiteralPath $BaseMarker) -and @(Get-ChildItem -LiteralPath $BaseDirectory -Force).Count -gt 0) {
                throw "The private Python folder is occupied by an unrecognized incomplete installation: $BaseDirectory"
            }
            New-Item -ItemType Directory -Force -Path $BaseDirectory | Out-Null
            @{ owner = "SARU Picker crYOLO"; created_utc = $Started } | ConvertTo-Json | Set-Content -LiteralPath $BaseMarker -Encoding UTF8
            $DownloadDirectory = Join-Path $ParentDirectory "downloads"
            New-Item -ItemType Directory -Force -Path $DownloadDirectory | Out-Null
            $Package = Join-Path $DownloadDirectory "python.3.7.9.zip"
            $PythonUrl = "https://api.nuget.org/v3-flatcontainer/python/3.7.9/python.3.7.9.nupkg"
            $PythonHash = "029e1ea5beeb142cd0f2da4e75c45f49bd7e0c7b48eb5bb6c6947f2d1b665c5e"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $BundledPackage = Join-Path $PSScriptRoot 'python.3.7.9.nupkg'
            if (Test-Path -LiteralPath $BundledPackage -PathType Leaf) {
                if ((Get-FileHash -LiteralPath $BundledPackage -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PythonHash) {
                    throw "The bundled Python package failed its checksum. Reinstall SARU Picker."
                }
                Copy-Item -LiteralPath $BundledPackage -Destination $Package -Force
                $InstallReport.python_package_source = 'bundled'
            } elseif (-not (Test-Path -LiteralPath $Package) -or (Get-FileHash -LiteralPath $Package -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PythonHash) {
                Invoke-WebRequest -Uri $PythonUrl -OutFile $Package -UseBasicParsing
                $InstallReport.python_package_source = 'upstream-download'
            } else {
                $InstallReport.python_package_source = 'verified-cache'
            }
            if ((Get-FileHash -LiteralPath $Package -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PythonHash) {
                throw "The official Python package failed its checksum."
            }
            $InstallReport.python_package_url = $PythonUrl
            $InstallReport.python_package_sha256 = $PythonHash
            $ExtractDirectory = Join-Path $DownloadDirectory "python.3.7.9"
            Expand-Archive -LiteralPath $Package -DestinationPath $ExtractDirectory -Force
            Get-ChildItem -LiteralPath (Join-Path $ExtractDirectory "tools") -Force | Copy-Item -Destination $BaseDirectory -Recurse -Force
        }
    }
    Invoke-Checked $Python37 @('-c', "import sys,struct; assert sys.version_info[:2] == (3,7) and struct.calcsize('P') == 8, 'Python 3.7 x64 is required'")
    $InstallReport.base_python = $Python37
    # venv repairs its own missing files in place; no recursive removal of any folder.
    Invoke-Checked $Python37 @('-m', 'venv', $RuntimeDirectory)
    Copy-PrivateVCRuntime (Join-Path $RuntimeDirectory 'Scripts')
    if ($BaseDirectory) { Copy-PrivateVCRuntime $BaseDirectory }
    $InstallReport.stage = "packages"
    Save-Report
    Invoke-Checked $RuntimePython @('-m', 'pip', '--isolated', 'install', 'pip==24.0', 'setuptools==65.6.3', 'wheel==0.42.0')
    Invoke-Checked $RuntimePython @('-m', 'pip', '--isolated', 'install', '--force-reinstall', '-r', (Join-Path $PSScriptRoot 'runtime-requirements-windows.txt'))
    Invoke-Checked $RuntimePython @('-m', 'pip', '--isolated', 'install', '--force-reinstall', '--no-deps', '--no-build-isolation', 'cryolo==1.9.9', 'GooeyDev==1.0.8b5', 'lineenhancer==1.0.9.dev2', 'cryoloBM==1.6.1')
    $InstallReport.stage = "initialization-weights"
    Save-Report
    Ensure-BackendWeights
    $InstallReport.stage = "verification"
    Save-Report
    $Health = Read-Health
    if (-not $Health) { throw "The crYOLO health check failed. See $(Join-Path $ParentDirectory 'runtime-health.log') and runtime-health.json." }
    $InstallReport.status = "ok"
    $InstallReport.stage = "ready"
    $InstallReport.health = $Health
    $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
    Save-Report
    Write-Host "crYOLO is ready. SARU Picker detects the private runtime automatically."
    exit 0
} catch {
    $InstallReport.status = "error"
    $InstallReport.error = $_.Exception.Message
    $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
    Save-Report
    Write-Error -Message $InstallReport.error -ErrorAction Continue
    exit 1
} finally {
    foreach ($Name in $OldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($Name, $OldEnvironment[$Name], 'Process') }
    if ($Lock) { $Lock.Dispose() }
}
