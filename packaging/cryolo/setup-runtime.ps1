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
    [string]$RuntimeDirectory = "",
    [string]$Report = "",
    [switch]$ForcePrivatePython,
    [switch]$CheckOnly,
    [switch]$Repair,
    [ValidateRange(1, 3600)][int]$LockTimeoutSeconds = 10,
    [ValidateRange(1, 3600)][int]$CommandTimeoutSeconds = 480,
    [ValidateRange(1, 7200)][int]$SetupTimeoutSeconds = 900
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($env:OS -ne "Windows_NT") { throw "This setup is for Windows x64." }
if ($ForcePrivatePython -and $Python37) { throw "ForcePrivatePython cannot be combined with Python37." }

function Assert-SetupTime {
    if ([DateTime]::UtcNow -ge $script:SetupDeadline) {
        throw "crYOLO setup exceeded its $SetupTimeoutSeconds-second limit during $($InstallReport.stage). See $ParentDirectory for command logs."
    }
}
function Stop-SetupChild {
    param([Diagnostics.Process]$Process)
    if (-not $Process -or $Process.HasExited) { return }
    # Only the child tree created by this setup is eligible for termination.
    $TaskKill = if ($env:SystemRoot) { Join-Path $env:SystemRoot 'System32\taskkill.exe' } else { '' }
    if ($TaskKill -and (Test-Path -LiteralPath $TaskKill)) {
        $Killer = $null
        try {
            $Killer = Start-Process -FilePath $TaskKill -ArgumentList @('/PID', $Process.Id, '/T', '/F') -PassThru -NoNewWindow
            if (-not $Killer.WaitForExit(5000)) { $Killer.Kill() }
        } catch {
            Write-Warning 'Process-tree termination failed; stopping the direct setup child.'
        } finally {
            if ($Killer) { $Killer.Dispose() }
        }
    }
    if (-not $Process.HasExited) { $Process.Kill() }
    if (-not $Process.WaitForExit(5000)) { throw 'The setup child process could not be stopped.' }
}
function ConvertTo-NativeArgument {
    param([string]$Value)
    # Windows CRT quoting: double backslashes before quotes and before the close.
    $Escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $Escaped = [regex]::Replace($Escaped, '(\\+)$', '$1$1')
    return '"' + $Escaped + '"'
}
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments, [int]$TimeoutSeconds = $CommandTimeoutSeconds)
    Assert-SetupTime
    $script:CommandSequence++
    $Prefix = Join-Path $ParentDirectory ('runtime-command-{0:00}' -f $script:CommandSequence)
    $OutputLog = "$Prefix.out.log"
    $ErrorLog = "$Prefix.err.log"
    $ArgumentLine = ($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    $Child = $null
    try {
        $Child = Start-Process -FilePath $Program -ArgumentList $ArgumentLine -PassThru -NoNewWindow -RedirectStandardOutput $OutputLog -RedirectStandardError $ErrorLog
        $null = $Child.Handle
        $CommandStarted = [DateTime]::UtcNow
        $CommandDeadline = $CommandStarted.AddSeconds($TimeoutSeconds)
        $InstallReport.command = @{ program = $Program; pid = $Child.Id; stdout_log = $OutputLog; stderr_log = $ErrorLog }
        Save-Report
        $NextProgress = [DateTime]::MinValue
        while (-not $Child.WaitForExit(250)) {
            Assert-SetupTime
            if ([DateTime]::UtcNow -ge $CommandDeadline) { throw "The $($InstallReport.stage) command exceeded $TimeoutSeconds seconds. See $OutputLog and $ErrorLog." }
            if ([DateTime]::UtcNow -ge $NextProgress) {
                $InstallReport.command.elapsed_seconds = [int]([DateTime]::UtcNow - $CommandStarted).TotalSeconds
                Save-Report
                Write-Host "crYOLO $($InstallReport.stage): $($InstallReport.command.elapsed_seconds)s elapsed. Log: $OutputLog"
                $NextProgress = [DateTime]::UtcNow.AddSeconds(5)
            }
        }
        $ExitCode = $Child.ExitCode
        if ($null -eq $ExitCode -or $ExitCode -ne 0) { throw "Command failed with exit code ${ExitCode}: $Program. See $ErrorLog." }
        Assert-SetupTime
    } finally {
        if ($Child) {
            try { Stop-SetupChild $Child } finally { $Child.Dispose() }
        }
    }
}
function Get-BoundedDownload {
    param([string]$Uri, [string]$Destination, [int]$TimeoutSeconds = 300)
    Assert-SetupTime
    $Client = New-Object Net.WebClient
    try {
        $Task = $Client.DownloadFileTaskAsync([Uri]$Uri, $Destination)
        $StartedDownload = [DateTime]::UtcNow
        $DownloadDeadline = $StartedDownload.AddSeconds($TimeoutSeconds)
        $NextProgress = [DateTime]::MinValue
        while (-not $Task.IsCompleted) {
            Assert-SetupTime
            if ([DateTime]::UtcNow -ge $DownloadDeadline) { throw "Download exceeded $TimeoutSeconds seconds: $Uri" }
            if ([DateTime]::UtcNow -ge $NextProgress) {
                $Downloaded = 0
                if (Test-Path -LiteralPath $Destination) { $Downloaded = (Get-Item -LiteralPath $Destination).Length }
                $InstallReport.download = @{ url = $Uri; bytes = $Downloaded; elapsed_seconds = [int]([DateTime]::UtcNow - $StartedDownload).TotalSeconds }
                Save-Report
                Write-Host "crYOLO $($InstallReport.stage): downloaded $Downloaded bytes."
                $NextProgress = [DateTime]::UtcNow.AddSeconds(5)
            }
            Start-Sleep -Milliseconds 250
        }
        $Task.GetAwaiter().GetResult()
        $InstallReport.download = @{ url = $Uri; bytes = (Get-Item -LiteralPath $Destination).Length; completed_utc = [DateTime]::UtcNow.ToString('o') }
        Save-Report
        Assert-SetupTime
    } finally {
        $Client.CancelAsync()
        $Client.Dispose()
    }
}
function Read-Health {
    $script:HealthFailure = $null
    if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) { return $null }
    $ProbeOutput = Join-Path $ParentDirectory "runtime-health.json"
    $ProbeError = Join-Path $ParentDirectory "runtime-health.log"
    $ProbeProgress = Join-Path $ParentDirectory "runtime-health-progress.json"
    $Arguments = '"' + (Join-Path $PSScriptRoot "runtime_health.py") + '" --progress-report "' + $ProbeProgress + '"'
    try {
        Assert-SetupTime
        $Probe = Start-Process -FilePath $RuntimePython -ArgumentList $Arguments -PassThru -NoNewWindow -RedirectStandardOutput $ProbeOutput -RedirectStandardError $ProbeError
        # Keep the native process handle while it is alive. Windows PowerShell
        # Start-Process can otherwise lose ExitCode after a short-lived child exits.
        # Process.ExitCode requires a valid local process handle (Microsoft .NET).
        $null = $Probe.Handle
        $HealthWait = [int][Math]::Min(120000, [Math]::Max(1, ($script:SetupDeadline - [DateTime]::UtcNow).TotalMilliseconds))
        if (-not $Probe.WaitForExit($HealthWait)) {
            $Probe.Kill()
            if (-not $Probe.WaitForExit(5000)) { throw 'The runtime health process did not stop after cancellation.' }
            $HealthSeconds = [int][Math]::Ceiling($HealthWait / 1000.0)
            $script:HealthFailure = @{ status = 'timeout'; error = "The crYOLO runtime health check exceeded $HealthSeconds seconds."; log = $ProbeError }
            $script:HealthFailure | ConvertTo-Json | Set-Content -LiteralPath $ProbeOutput -Encoding UTF8
            return $null
        }
        $ExitCode = $Probe.ExitCode
        if ($null -eq $ExitCode) { throw 'The runtime health process did not provide an exit code.' }
        $Health = $null
        try { $Health = Get-Content -LiteralPath $ProbeOutput -Raw | ConvertFrom-Json } catch { $Health = $null }
        $HealthStatus = $null
        $HealthError = $null
        if ($Health -and $Health.PSObject.Properties['status']) { $HealthStatus = $Health.status }
        if ($Health -and $Health.PSObject.Properties['error']) { $HealthError = $Health.error }
        if ($ExitCode -eq 0 -and $HealthStatus -eq 'ok') { return $Health }
        if ($HealthError) {
            $script:HealthFailure = @{ status = 'error'; returncode = $ExitCode; error = $HealthError; log = $ProbeError }
        } else {
            $script:HealthFailure = @{ status = 'crashed'; returncode = $ExitCode; error = "The crYOLO runtime check stopped without a report (exit $ExitCode)."; log = $ProbeError }
            $script:HealthFailure | ConvertTo-Json | Set-Content -LiteralPath $ProbeOutput -Encoding UTF8
        }
    } catch {
        $script:HealthFailure = @{ status = 'error'; error = $_.Exception.Message; log = $ProbeError }
    }
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
    Get-BoundedDownload "https://ndownloader.figshare.com/files/16406843" $Temporary
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

$Lock = $null
$OwnLock = $false
$script:CommandSequence = 0
$script:SetupDeadline = [DateTime]::UtcNow.AddSeconds($SetupTimeoutSeconds)
$InstallReport = $null
$OldEnvironment = @{}
foreach ($Name in @('CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS', 'PYTHONNOUSERSITE', 'PYTHONPATH', 'PYTHONHOME', 'PIP_CONFIG_FILE', 'PSModulePath')) {
    $OldEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}
try {
    # Inno/Python launched from PowerShell 7 inherits its incompatible module path.
    # Resolve this host's built-ins explicitly; do not change user/machine settings.
    # https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_psmodulepath
    $HostModuleRoot = [IO.Path]::Combine($PSHOME, 'Modules')
    $env:PSModulePath = $HostModuleRoot
    foreach ($Module in @('Microsoft.PowerShell.Utility', 'Microsoft.PowerShell.Security', 'Microsoft.PowerShell.Archive')) {
        Import-Module ([IO.Path]::Combine($HostModuleRoot, $Module, "$Module.psd1")) -ErrorAction Stop
    }
    if (-not $RuntimeDirectory) {
        $RuntimeDirectory = [IO.Path]::Combine($env:LOCALAPPDATA, 'SARU_Picker_crYOLO', 'cryolo-env')
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
        python = $RuntimePython; reused = $false; repair_requested = [bool]$Repair; vc_runtime_files = @(); private_python_requested = [bool]$ForcePrivatePython;
        cryolo_version = "1.9.9"; tensorflow_version = "1.15.4";
        scope = "private CPU runtime and upstream initialization weights; no trained capsid model" }
    $InstallReport.powershell_version = $PSVersionTable.PSVersion.ToString()
    $InstallReport.powershell_module_root = $HostModuleRoot
    # A second installer/repair must not queue for most of an hour.
    $LockOwner = Join-Path $ParentDirectory 'runtime-setup-owner.json'
    $InstallReport.stage = 'waiting for setup lock'
    $Deadline = [DateTime]::UtcNow.AddSeconds($LockTimeoutSeconds)
    while (-not $Lock) {
        try { $Lock = [IO.File]::Open((Join-Path $ParentDirectory "runtime-setup.lock"), 'OpenOrCreate', 'ReadWrite', 'None') }
        catch [IO.IOException] {
            if (($_.Exception.HResult -band 0xffff) -notin @(11, 32, 33)) { throw }
            if ([DateTime]::UtcNow -ge $Deadline) {
                $Owner = ''
                if (Test-Path -LiteralPath $LockOwner) { $Owner = Get-Content -LiteralPath $LockOwner -Raw }
                throw "Another crYOLO setup is still running. Finish or cancel it before retrying. Owner: $Owner"
            }
            Write-Host 'Waiting for the other crYOLO installer or repair to finish...'
            Start-Sleep -Milliseconds 500
        }
    }
    $OwnLock = $true
    @{ pid = $PID; started_utc = $Started; runtime_directory = $RuntimeDirectory } | ConvertTo-Json | Set-Content -LiteralPath $LockOwner -Encoding UTF8
    $InstallReport.stage = 'health-check'
    Save-Report
    Assert-SetupTime
    $env:CUDA_VISIBLE_DEVICES = ""
    $env:TF_CPP_MIN_LOG_LEVEL = "2"
    $env:OMP_NUM_THREADS = "4"
    $env:PYTHONNOUSERSITE = "1"
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PIP_CONFIG_FILE = "NUL"
    $Health = $null
    if (-not $Repair) {
        $Health = Read-Health
        if ($script:HealthFailure -and $script:HealthFailure.status -in @('timeout', 'crashed')) {
            $InstallReport.health = $script:HealthFailure
            throw "$($script:HealthFailure.error) Open Advanced and choose Repair crYOLO runtime. Diagnostic log: $($script:HealthFailure.log)"
        }
    }
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
                Invoke-Checked $Python37 @('-c', "import sys,struct,venv,ensurepip; assert sys.version_info[:2] == (3,7) and struct.calcsize('P') == 8") -TimeoutSeconds 30
                $BaseHealthy = $true
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
                Get-BoundedDownload $PythonUrl $Package -TimeoutSeconds 120
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
    Invoke-Checked $Python37 @('-m', 'venv', $RuntimeDirectory) -TimeoutSeconds 180
    Copy-PrivateVCRuntime (Join-Path $RuntimeDirectory 'Scripts')
    if ($BaseDirectory) { Copy-PrivateVCRuntime $BaseDirectory }
    $InstallReport.stage = "packages"
    Save-Report
    $PipLog = Join-Path $ParentDirectory 'runtime-pip.log'
    Set-Content -LiteralPath $PipLog -Value '' -Encoding UTF8
    $PipArguments = @('-m', 'pip', '--isolated', '--no-input', '--disable-pip-version-check', '--timeout', '15', '--retries', '1', '--log', $PipLog)
    Invoke-Checked $RuntimePython ($PipArguments + @('install', 'pip==24.0', 'setuptools==65.6.3', 'wheel==0.42.0'))
    Invoke-Checked $RuntimePython ($PipArguments + @('install', '--force-reinstall', '-r', (Join-Path $PSScriptRoot 'runtime-requirements-windows.txt')))
    Invoke-Checked $RuntimePython ($PipArguments + @('install', '--force-reinstall', '--no-deps', '--no-build-isolation', 'cryolo==1.9.9', 'GooeyDev==1.0.8b5', 'lineenhancer==1.0.9.dev2', 'cryoloBM==1.6.1'))
    $InstallReport.stage = "initialization-weights"
    Save-Report
    Ensure-BackendWeights
    $InstallReport.stage = "verification"
    Save-Report
    $Health = Read-Health
    if (-not $Health) {
        $InstallReport.health = $script:HealthFailure
        throw "The crYOLO health check failed: $($script:HealthFailure.error) See $(Join-Path $ParentDirectory 'runtime-health.log') and runtime-health.json."
    }
    $InstallReport.status = "ok"
    $InstallReport.stage = "ready"
    $InstallReport.health = $Health
    $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
    Save-Report
    Write-Host "crYOLO is ready. SARU Picker detects the private runtime automatically."
    exit 0
} catch {
    $FailureMessage = $_.Exception.Message
    if ($InstallReport) {
        $InstallReport.status = "error"
        $InstallReport.error = $FailureMessage
        $InstallReport.completed_utc = [DateTime]::UtcNow.ToString("o")
        Save-Report
    }
    Write-Error -Message $FailureMessage -ErrorAction Continue
    exit 1
} finally {
    foreach ($Name in $OldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($Name, $OldEnvironment[$Name], 'Process') }
    if ($OwnLock -and (Test-Path -LiteralPath $LockOwner)) { Remove-Item -LiteralPath $LockOwner -Force }
    if ($Lock) { $Lock.Dispose() }
}
