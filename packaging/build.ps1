#requires -Version 5.1
<#
.SYNOPSIS
Build and verify the Windows x64 CPU application and optional Inno installer.
.DESCRIPTION
Run on Windows with Python 3.12 x64 and Git. The end user needs neither.
The default build downloads verified SAM weights if no checkpoint is supplied.
-TestInstaller is intended for a fresh disposable Windows build machine; it
refuses to run if this user's SARU Picker is already installed.
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$Checkpoint = "",
    [string]$IsccPath = "",
    [switch]$SkipInstaller,
    [switch]$TestInstaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "This build must run on Windows. PyInstaller cannot cross-compile Windows binaries."
}
if ($SkipInstaller -and $TestInstaller) {
    throw "-TestInstaller cannot be combined with -SkipInstaller."
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDirectory = Join-Path $ProjectRoot "build"
$ArtifactDirectory = Join-Path $ProjectRoot "artifacts"
$VenvDirectory = Join-Path $ProjectRoot ".build-venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$AppVersion = "2.3.0"
$Manifest = Get-Content (Join-Path $PSScriptRoot "model-manifest.json") -Raw | ConvertFrom-Json

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}

function Assert-Checkpoint {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Model file does not exist: $Path"
    }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Manifest.size_bytes) {
        throw "Model size is incorrect: $Path"
    }
    $Digest = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Digest -ne $Manifest.sha256) {
        throw "Model SHA-256 is incorrect: $Path"
    }
}

function Invoke-SmokeTest {
    param([string]$Executable, [string]$Report)
    if (Test-Path -LiteralPath $Report) { Remove-Item -LiteralPath $Report }
    # Start-Process joins ArgumentList elements without adding quotes. Supply
    # one deliberately quoted argument string for paths containing spaces.
    $Process = Start-Process -FilePath $Executable -ArgumentList "--smoke-test --report `"$Report`"" -PassThru
    if (-not $Process.WaitForExit(600000)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "Frozen application smoke test timed out after 10 minutes."
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0) {
        throw "Frozen application smoke test failed (exit $($Process.ExitCode)); inspect $Report and the application logs."
    }
    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "The windowed executable produced no smoke-test report: $Report"
    }
    $Result = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    if ($Result.status -ne "ok" -or $Result.mode -ne "smoke-test") {
        throw "Smoke-test report does not indicate success: $Report"
    }
    foreach ($Required in @("image_processing", "torchvision_native", "windows_runtime", "sam_cpu_inference")) {
        $Passed = @($Result.checks | Where-Object { $_.name -eq $Required -and $_.status -eq "ok" })
        if ($Passed.Count -ne 1) {
            throw "Smoke test did not pass required check ${Required}: $Report"
        }
    }
    Write-Host "Passed executable smoke test: $Report"
}

function Assert-BundledLibraries {
    param([string]$Distribution)
    # A hosted build machine already has the VC redistributable. Its global
    # DLLs must not conceal a bundle that would fail on a fresh user machine.
    foreach ($Runtime in @("MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Distribution "_internal\$Runtime") -PathType Leaf)) {
            throw "Required app-local Microsoft runtime is absent: $Runtime"
        }
    }
    foreach ($Runtime in @("fbgemm.dll", "asmjit.dll", "libiomp5md.dll")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Distribution "_internal\torch\lib\$Runtime") -PathType Leaf)) {
            throw "Required bundled torch runtime is absent: $Runtime"
        }
    }
    Assert-Checkpoint (Join-Path $Distribution "_internal\models\$($Manifest.filename)")
    if (-not (Test-Path -LiteralPath (Join-Path $Distribution "SARU Icon Helper.exe") -PathType Leaf)) {
        throw "The independent icon installation helper is missing from the distribution."
    }
}

function Invoke-IconHelper {
    param([string]$Helper, [string]$Target, [string]$Icon, [string]$Report)
    if (Test-Path -LiteralPath $Report) { Remove-Item -LiteralPath $Report }
    $Arguments = "--target `"$Target`" --icon `"$Icon`" --report `"$Report`""
    $Process = Start-Process -FilePath $Helper -ArgumentList $Arguments -PassThru
    if (-not $Process.WaitForExit(120000)) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "The icon installation helper timed out."
    }
    $Process.Refresh()
    if ($Process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "The icon installation helper failed or produced no report: $Report"
    }
    $Result = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    $ExpectedHash = (Get-FileHash -LiteralPath $Icon -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Result.status -ne "ok" -or -not $Result.overlay_preserved -or $Result.source_ico_sha256 -ne $ExpectedHash) {
        throw "The icon installation helper report failed its image/archive checks: $Report"
    }
}

Push-Location $ProjectRoot
$PreviousModel = $env:SARU_MODEL_FILE
$IconSidecarTested = $false
try {
    New-Item -ItemType Directory -Path $BuildDirectory, $ArtifactDirectory -Force | Out-Null
    Invoke-Checked $PythonExe @("-c", "import struct,sys; assert sys.platform == 'win32' and sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Python 3.12 x64 is required'")
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Invoke-Checked $PythonExe @("-m", "venv", $VenvDirectory)
    }
    Invoke-Checked $VenvPython @("-c", "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Remove .build-venv and rebuild with Python 3.12 x64'")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "pip==26.3", "setuptools==80.9.0", "wheel==0.45.1")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch==2.5.1+cpu", "torchvision==0.20.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu")
    Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "-r", (Join-Path $PSScriptRoot "requirements-build.txt"))
    Invoke-Checked $VenvPython @("-m", "pip", "check")
    Invoke-Checked $VenvPython @("-c", "import torch,torchvision; assert torch.__version__ == '2.5.1+cpu'; assert torchvision.__version__ == '0.20.1+cpu'; assert torch.version.cuda is None")
    Invoke-Checked $VenvPython @("build_standalone.py")
    Invoke-Checked $VenvPython @("-m", "unittest", "discover", "-s", "tests", "-v")
    # Catch native overlay/focus/layout defects before bundling the large model.
    Invoke-Checked $VenvPython @("-c", "from pathlib import Path; from app import setup_logging, _smoke_windows_runtime, write_json_report; result=_smoke_windows_runtime(setup_logging(), screenshot_dir=Path('artifacts/ui')); write_json_report(Path('artifacts/source-ui-smoke.json'), {'status':'ok', 'mode':'ui-smoke', 'details':result})")

    if ($Checkpoint) {
        $ModelFile = (Resolve-Path -LiteralPath $Checkpoint).Path
        Assert-Checkpoint $ModelFile
    } else {
        $ModelCache = Join-Path $ProjectRoot ".build-cache\models"
        New-Item -ItemType Directory -Path $ModelCache -Force | Out-Null
        $ModelFile = Join-Path $ModelCache $Manifest.filename
        if (-not (Test-Path -LiteralPath $ModelFile)) {
            $Partial = "$ModelFile.partial"
            try {
                Write-Host "Downloading original SAM ViT-B weights from the official source..."
                Invoke-WebRequest -Uri $Manifest.url -OutFile $Partial -UseBasicParsing
                Assert-Checkpoint $Partial
                Move-Item -LiteralPath $Partial -Destination $ModelFile -Force
            } finally {
                if (Test-Path -LiteralPath $Partial) { Remove-Item -LiteralPath $Partial }
            }
        }
        Assert-Checkpoint $ModelFile
    }
    $env:SARU_MODEL_FILE = $ModelFile
    Invoke-Checked $VenvPython @("packaging\collect_licenses.py", "--output", (Join-Path $BuildDirectory "third-party-notices"))
    Invoke-Checked $VenvPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", (Join-Path $ProjectRoot "dist"), "--workpath", (Join-Path $BuildDirectory "pyinstaller"), "packaging\saru_picker.spec")

    $Distribution = Join-Path $ProjectRoot "dist\SARU_Picker"
    $Executable = Join-Path $Distribution "SARU Picker.exe"
    # Keep the tiny stdlib helper independent of the application's large
    # onedir bundle. Its own onefile runtime cannot replace base_library.zip
    # or any DLL in the main application's _internal directory.
    $IconHelperBuild = Join-Path $BuildDirectory "icon-helper"
    Invoke-Checked $VenvPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed", "--noupx", "--name", "SARU Icon Helper", "--icon", (Join-Path $ProjectRoot "assets\app-icon.ico"), "--distpath", (Join-Path $IconHelperBuild "dist"), "--workpath", (Join-Path $IconHelperBuild "work"), "--specpath", (Join-Path $IconHelperBuild "spec"), "packaging\install_icon.py")
    $IconHelper = Join-Path $Distribution "SARU Icon Helper.exe"
    Copy-Item -LiteralPath (Join-Path $IconHelperBuild "dist\SARU Icon Helper.exe") -Destination $IconHelper -Force
    Assert-BundledLibraries $Distribution
    # Test the real onefile helper and then execute its patched application.
    # This catches accidental loss of the appended PyInstaller archive even
    # when the Windows resource update itself reports success.
    Invoke-IconHelper $IconHelper $Executable (Join-Path $Distribution "_internal\assets\app-icon.ico") (Join-Path $ArtifactDirectory "frozen-icon-install.json")
    Invoke-SmokeTest $Executable (Join-Path $ArtifactDirectory "frozen-smoke-test.json")

    if (-not $SkipInstaller) {
        if (-not $IsccPath) {
            $Candidates = @(
                (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7\ISCC.exe"),
                (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
                (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe"),
                (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
            )
            $IsccPath = $Candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        }
        if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
            throw "Install Inno Setup 6.3+ / 7 or pass -IsccPath. Use -SkipInstaller for the verified portable directory only."
        }
        Invoke-Checked $IsccPath @("/DAppVersion=$AppVersion", "/DAppSourceDir=$Distribution", "/DAppOutputDir=$ArtifactDirectory", (Join-Path $PSScriptRoot "installer.iss"))
        $Installer = Join-Path $ArtifactDirectory "SARU-Picker-Setup-$AppVersion-win64.exe"
        if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
            throw "Inno Setup returned without producing the expected installer."
        }
        if ($TestInstaller) {
            # The same app ID is used for upgrades. Protect an existing user's
            # installation from a test installer replacing its uninstall entry.
            $UninstallKey = "Software\Microsoft\Windows\CurrentVersion\Uninstall\{A7D0C866-7ED3-4EB2-8A49-6C6B13E84876}_is1"
            if ((Test-Path "HKCU:\$UninstallKey") -or (Test-Path "HKLM:\$UninstallKey")) {
                throw "Installer integration test requires a machine without SARU Picker already installed."
            }
            $InstallDirectory = Join-Path ([IO.Path]::GetTempPath()) ("SARU Picker Smoke " + [guid]::NewGuid().ToString("N"))
            $InstallLog = Join-Path $ArtifactDirectory "installer-smoke.log"
            # Exercise the sidecar override using generated, nonprivate test
            # artwork. The public installer remains paired with generic art;
            # no user's image is needed in the repository or on the runner.
            $IconFixtureDirectory = Join-Path $BuildDirectory ("icon-installer-test-" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType Directory -Path $IconFixtureDirectory -Force | Out-Null
            $FixtureInstaller = Join-Path $IconFixtureDirectory (Split-Path -Leaf $Installer)
            Copy-Item -LiteralPath $Installer -Destination $FixtureInstaller
            Invoke-Checked $VenvPython @("packaging\verify_icon_installation.py", "make-fixture", "--directory", $IconFixtureDirectory, "--baseline-directory", (Join-Path $ProjectRoot "assets"), "--report", (Join-Path $ArtifactDirectory "icon-fixture.json"))
            $InstallIconReport = Join-Path $env:LOCALAPPDATA "SARU_Picker\icon-install.json"
            if (Test-Path -LiteralPath $InstallIconReport) { Remove-Item -LiteralPath $InstallIconReport }
            $Installed = $false
            try {
                $InstallArguments = "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=`"$InstallDirectory`" /LOG=`"$InstallLog`""
                $InstallProcess = Start-Process -FilePath $FixtureInstaller -ArgumentList $InstallArguments -Wait -PassThru
                if ($InstallProcess.ExitCode -ne 0) { throw "Installer failed (exit $($InstallProcess.ExitCode))." }
                $Installed = $true
                Assert-BundledLibraries $InstallDirectory
                Invoke-SmokeTest (Join-Path $InstallDirectory "SARU Picker.exe") (Join-Path $ArtifactDirectory "installed-smoke-test.json")
                Invoke-Checked $VenvPython @("packaging\verify_icon_installation.py", "verify", "--install-directory", $InstallDirectory, "--fixture-directory", $IconFixtureDirectory, "--helper-report", $InstallIconReport, "--runtime-report", (Join-Path $ArtifactDirectory "installed-smoke-test.json"), "--report", (Join-Path $ArtifactDirectory "icon-install-verification.json"))
                $IconSidecarTested = $true
            } finally {
                if (Test-Path -LiteralPath $InstallIconReport -PathType Leaf) {
                    Copy-Item -LiteralPath $InstallIconReport -Destination (Join-Path $ArtifactDirectory "icon-helper-install.json") -Force
                }
                $Uninstaller = Join-Path $InstallDirectory "unins000.exe"
                if (Test-Path -LiteralPath $Uninstaller) {
                    $UninstallProcess = Start-Process -FilePath $Uninstaller -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART" -Wait -PassThru
                    if ($UninstallProcess.ExitCode -ne 0) { throw "Test installation could not be uninstalled (exit $($UninstallProcess.ExitCode))." }
                    if (Test-Path -LiteralPath (Join-Path $InstallDirectory "SARU Picker.exe")) { throw "Uninstaller left the application executable behind." }
                } elseif ($Installed) {
                    throw "Installed application has no uninstaller."
                }
                if (Test-Path -LiteralPath $IconFixtureDirectory) {
                    Remove-Item -LiteralPath $IconFixtureDirectory -Recurse -Force
                }
            }
        }
    }

    $DependencyList = & $VenvPython -m pip list --format=json
    if ($LASTEXITCODE -ne 0) { throw "Could not record installed dependency versions." }
    $BuildMetadata = [ordered]@{
        application = "SARU Picker"; version = $AppVersion; platform = "windows-x64"; edition = "cpu"
        built_at_utc = [DateTime]::UtcNow.ToString("o")
        source_commit = $env:GITHUB_SHA
        model = $Manifest
        dependencies = @($DependencyList | ConvertFrom-Json)
        installer_tested = [bool]$TestInstaller
        icon_sidecar_installation_tested = $IconSidecarTested
        icon_helper_bytes = (Get-Item -LiteralPath $IconHelper).Length
    }
    $BuildMetadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $ArtifactDirectory "build-manifest.json") -Encoding UTF8
    $Checksums = @(Get-ChildItem -LiteralPath $ArtifactDirectory -Filter "*.exe" | ForEach-Object {
        "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
    })
    if ($Checksums.Count) { $Checksums | Set-Content -LiteralPath (Join-Path $ArtifactDirectory "SHA256SUMS.txt") -Encoding ASCII }
    Write-Host "Verified Windows distribution: $Distribution"
    if (-not $SkipInstaller) { Write-Host "Installer and validation reports: $ArtifactDirectory" }
} finally {
    $env:SARU_MODEL_FILE = $PreviousModel
    Pop-Location
}
