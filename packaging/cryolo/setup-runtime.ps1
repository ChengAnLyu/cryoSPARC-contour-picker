#requires -Version 5.1
<#
.SYNOPSIS
Install a separate stock crYOLO CPU command-line runtime for this user.
.DESCRIPTION
Downloads packages from PyPI. It does not redistribute or modify crYOLO.
When Python 3.7 x64 is unavailable, downloads the signed official Python 3.7.9
installer to a private application directory without changing PATH or file associations.
The legacy interpreter is used only by crYOLO; the GUI uses its own modern runtime.
#>
[CmdletBinding()]
param(
    [string]$Python37 = "",
    [string]$RuntimeDirectory = (Join-Path $env:LOCALAPPDATA "SARU_Picker_crYOLO\cryolo-env"),
    [string]$Report = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "This setup is for Windows x64." }
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Program" }
}
$RuntimeDirectory = [IO.Path]::GetFullPath($RuntimeDirectory)
$RuntimePython = Join-Path $RuntimeDirectory "Scripts\python.exe"
$OwnerMarker = Join-Path $RuntimeDirectory "saru-cryolo-runtime.json"
if ((Test-Path -LiteralPath $RuntimeDirectory) -and -not (Test-Path -LiteralPath $OwnerMarker)) {
    throw "The chosen directory already exists and is not an SARU crYOLO runtime. Choose a new directory."
}
$Started = [DateTime]::UtcNow.ToString("o")
if (-not $Report) { $Report = Join-Path (Split-Path $RuntimeDirectory -Parent) "runtime-install-report.json" }
New-Item -ItemType Directory -Force -Path (Split-Path $Report -Parent) | Out-Null
$InstallReport = @{ status = "running"; started_utc = $Started; python = $RuntimePython;
    cryolo_version = "1.9.9"; tensorflow_version = "1.15.4";
    scope = "isolated CPU CLI runtime; no model or capsid-accuracy validation" }
try {
    Write-Host "crYOLO is supplied by Max Planck Society for non-commercial academic/research use."
    Write-Host "License: https://cryolo-doc.readthedocs.io/en/stable/other/license.html"
    if (-not (Test-Path -LiteralPath $RuntimePython)) {
        if (-not $Python37) {
            $Launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($Launcher) {
                $Candidate = & $Launcher.Source -3.7 -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $Candidate) { $Python37 = [string]($Candidate | Select-Object -Last 1) }
            }
        }
        if (-not $Python37) {
            $BaseDirectory = Join-Path (Split-Path $RuntimeDirectory -Parent) "python37"
            $Python37 = Join-Path $BaseDirectory "python.exe"
            if (-not (Test-Path -LiteralPath $Python37)) {
                $DownloadDirectory = Join-Path (Split-Path $RuntimeDirectory -Parent) "downloads"
                New-Item -ItemType Directory -Force -Path $DownloadDirectory | Out-Null
                $Installer = Join-Path $DownloadDirectory "python-3.7.9-amd64.exe"
                [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
                Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe" -OutFile $Installer -UseBasicParsing
                $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
                if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                    throw "The Python installer does not have a valid Python Software Foundation signature."
                }
                $InstallReport.python_installer_sha256 = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
                $Arguments = "/quiet InstallAllUsers=0 TargetDir=`"$BaseDirectory`" Include_launcher=0 InstallLauncherAllUsers=0 Include_test=0 Include_doc=0 Include_tcltk=0 Include_pip=1 PrependPath=0 AssociateFiles=0 Shortcuts=0"
                $Process = Start-Process -FilePath $Installer -ArgumentList $Arguments -Wait -PassThru
                if ($Process.ExitCode -notin @(0, 3010) -or -not (Test-Path -LiteralPath $Python37)) {
                    throw "The private Python 3.7 installation failed: $($Process.ExitCode)."
                }
            }
        }
        Invoke-Checked $Python37 @("-c", "import sys,struct; assert sys.version_info[:2] == (3,7) and struct.calcsize('P') == 8, 'Python 3.7 x64 is required'")
        Invoke-Checked $Python37 @("-m", "venv", $RuntimeDirectory)
        @{ owner = "SARU Picker crYOLO"; created_utc = $Started } | ConvertTo-Json | Set-Content -LiteralPath $OwnerMarker -Encoding UTF8
    }
    Invoke-Checked $RuntimePython @("-m", "pip", "install", "pip==24.0", "setuptools==65.6.3", "wheel==0.42.0")
    Invoke-Checked $RuntimePython @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "runtime-requirements-windows.txt"))
    # Stock packages only. GUI-only optional dependencies are unnecessary for
    # this subprocess integration and are deliberately not installed.
    Invoke-Checked $RuntimePython @("-m", "pip", "install", "--no-deps", "--no-build-isolation", "cryolo==1.9.9", "GooeyDev==1.0.8b5", "lineenhancer==1.0.9.dev2", "cryoloBM==1.6.1")
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:TF_CPP_MIN_LOG_LEVEL = "2"
    $env:OMP_NUM_THREADS = "4"
    Invoke-Checked $RuntimePython @("-c", "import tensorflow as tf,keras,cryolo; from cryolo.frontend import YOLO; assert tf.__version__=='1.15.4'; print('TensorFlow',tf.__version__,'Keras',keras.__version__,'crYOLO import passed')")
    Invoke-Checked $RuntimePython @("-m", "cryolo.predict", "--help")
    Invoke-Checked $RuntimePython @("-m", "cryolo.train", "--help")
    $InstallReport.status = "ok"
    $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
    $InstallReport | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Report -Encoding UTF8
    Write-Host "Runtime ready: $RuntimePython"
    Write-Host "Launch SARU Picker crYOLO. The default location is detected automatically."
} catch {
    $InstallReport.status = "error"
    $InstallReport.error = $_.Exception.Message
    $InstallReport | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Report -Encoding UTF8
    throw
}
