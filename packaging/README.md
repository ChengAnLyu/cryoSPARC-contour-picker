# Windows installer build

This directory turns the source into an installed Windows x64 application. The
installer includes the Python interpreter, Tcl/Tk, CPU PyTorch/torchvision,
desktop automation libraries, and the original SAM ViT-B weights. End users
install `SARU-Picker-Setup-2.5.0-win64.exe` and open **SARU Picker** from the Start
menu or desktop shortcut. The installed application does not download models.

The source archive is not itself an installer. A successful Windows workflow
run is required to produce and verify the executable; Linux syntax and source
tests do not establish that the frozen Windows application works.

## Build with GitHub Actions

Place this project's files at the repository root. The workflow at
`.github/workflows/build-windows.yml` runs on pushes to `packaging/**` branches
when application, test, or packaging inputs change. It also supports manual
`workflow_dispatch`. GitHub normally exposes the manual Run workflow button
once the workflow is present on the default branch.

The job builds on Windows Server 2022 with Python 3.12 x64, verifies the downloaded
model and Inno Setup compiler checksums, and runs both the frozen application
and the installed application. It tests installation to a path with spaces and
then uninstalls the test copy. An installer artifact is uploaded only after all
of those steps pass. The workflow creates no GitHub Release and publishes no
website. Download the `SARU-Picker-Windows-CPU-*` artifact from a successful
Actions run and extract its ZIP to obtain the setup executable and reports.

The workflow uses commit-pinned official GitHub actions, Inno Setup 7.1.1, and
the dependencies in `requirements-build.txt`. `build-manifest.json` records the
complete resolved dependency list and source commit for the artifact.

## Build locally on Windows

Install Python 3.12 **x64**, Git, and Inno Setup 6.3 or newer. Then run PowerShell
in the project root:

```powershell
.\packaging\build.ps1 -PythonExe (Get-Command python).Source
```

If script execution is blocked by local policy, run the script using the policy
approved for your computer. There is no need to change machine-wide policy.

The build creates its own `.build-venv` and installs CPU-only PyTorch 2.5.1 with
torchvision 0.20.1. If a different Python version created `.build-venv`, remove
that build environment and rerun with Python 3.12. Model weights are downloaded
once to `.build-cache/models` and verified on every build. To use an already
downloaded official checkpoint:

```powershell
.\packaging\build.ps1 -Checkpoint 'C:\Models\sam_vit_b_01ec64.pth'
```

The accepted model must be exactly 375,042,383 bytes with SHA-256:

```text
ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912
```

Both the build script and the PyInstaller specification enforce this checksum.
No installer is built without the model. The official model and original SAM
code use the Apache 2.0 license; collected dependency licenses and model
attribution are included in the installed `_internal/licenses` directory.

Use `-IsccPath 'C:\Path\To\ISCC.exe'` for a compiler outside its normal install
location. Use `-SkipInstaller` to build and test just `dist/SARU_Picker`.
Keep that entire directory together: its executable requires `_internal`.
`-TestInstaller` adds install, full inference smoke test, and uninstall checks.
Use that option on a disposable Windows machine; it refuses to replace an
existing SARU Picker installation's uninstall entry.

## What is checked

The build regenerates `main.py` from the tested source modules and runs the source
unit suite before freezing. PyInstaller's official torch and torchvision hooks
retain the native libraries and Python sources needed by TorchScript. Explicit
Windows hidden imports retain pynput backends, PyAutoGUI, Tk and Pillow's Tk
bridge. The resulting one-directory application includes three app-local
Microsoft runtime DLLs and torch's OpenMP dependencies; missing DLLs fail the
build even if the build machine has a global runtime installation.

The executable's `--smoke-test --report <absolute-path>` command checks image
processing, torchvision's native NMS operation, the compact panel, native screen-overlay hit testing, Windows
desktop imports, and a real CPU SAM inference using a synthetic image. It sends
no operating-system mouse clicks or keyboard input. It also checks the actual
Windows small/large window icon handles and three consecutive semi-auto contour
selections through native Tk events and the real worker with an in-memory
desktop adapter. The second selection includes late pick-marker painting to
exercise the repaired alignment guard. Navigation fixtures also exercise the
detect/pick/Down cycle and semi-auto advancement while picking is disabled.
Both the exit code and the individual JSON
check results must pass. Its purpose is deployment validation, not measurement
of scientific recognition accuracy. The installed application is tested with
the same command.

Version 2.5.0 retains **Next micrograph** in semi-auto mode
and **Limit (0=all)** / **Load (s)** in auto mode. The default auto limit is 0
for continuous operation; positive limits include the current image and empty
images. Load is a maximum timeout, default 10 seconds, rather than a fixed
post-navigation delay. A new stable image starts the next detection. Empty
images are skipped, and unchanged/repeated images, loading failures and lost
target focus stop navigation without repeated Down-key retries. Semi-auto
navigation performs fresh detection without picking, including while Disable
Picking is checked.

Manual, semi-auto and auto now share a 0.01-second default click delay. A 0.00
setting adds no inter-click delay, while still yielding between clicks and
retaining per-click cancellation, focus/position checks and the mouse fail-safe.
Settings are available through **Advanced…** in model modes, **F9** while the
SARU panel has focus, and right-clicking the panel > **Picking speed…**. This
preserves the compact manual layout. The previous temporary native annotation
surface is recreated before a new manual trace so its old pixels cannot flash
when drawing resumes; the panel remains in place.

No added delay does not imply unlimited browser throughput. Actual accepted
picks depend on the target workstation and browser; inference and loading time
are separate. The version 2.4 benchmark in `VALIDATION.md` remains historical
and is not a measurement of the new 2.5 default. Recognition weights and the
private-icon packaging path are unchanged.

The public geometric fallback lives in `assets/app-icon.png` and a
multi-resolution `assets/app-icon.ico`; `generate_fallback_icon.py` reproduces
it. A private distribution can place custom `app-icon.png` and `app-icon.ico`
beside setup in the delivered ZIP. Inno copies those optional sidecars over the
installed artwork and runs `SARU Icon Helper.exe` to update the installed main
EXE. The helper preserves the appended PyInstaller archive and other resources.
Personal artwork need not be committed to the public repository. Extract the
entire custom-icon ZIP before running setup. The setup EXE itself retains the
fallback icon. The native window default and shortcut AppUserModelID are shared
by source and installed launches. The windowed launcher keeps `native_fault.log`
open while the app runs so supported native exceptions can leave Python thread
tracebacks.

`--diagnostics --report <absolute-path>` records metadata without running full
inference. A windowed executable has no normal console; use a report file. For
example:

```powershell
$app = "$env:LOCALAPPDATA\Programs\SARU Picker\SARU Picker.exe"
$report = "$env:TEMP\saru diagnostics.json"
Start-Process -FilePath $app -ArgumentList "--diagnostics --report `"$report`"" -Wait
Get-Content $report
```

## Installer behavior and release limits

The installer targets Windows 10/11 x64 and installs for the current user at
`%LOCALAPPDATA%\Programs\SARU Picker` without administrator privileges. It creates
a Start menu entry, enables a desktop shortcut by default, and registers a normal
uninstaller. The app writes state and logs in `%LOCALAPPDATA%\SARU_Picker`, outside
the installation folder. Uninstalling does not intentionally erase user exports
or saved application state. ARM64 and GPU acceleration are not packaged here.

This CPU installer favors deployment simplicity. It still needs the original
model's memory and computation; packaging does not improve recognition accuracy
or make CPU inference instantaneous. The checkpoint alone is about 358 MiB, so
the setup file will be a substantial download. Keep the installed directory on
a local drive with room for the runtime and model.

The workflow produces an unsigned installer. Authenticode signing requires the
project owner's signing certificate or trusted signing account and should be
added before broad public distribution. Do not claim a signature or Windows
acceptance test until those steps have actually been completed.

Build outputs:

- `dist/SARU_Picker/`: full portable application directory.
- `artifacts/SARU-Picker-Setup-2.5.0-win64.exe`: Windows installer.
- `artifacts/frozen-smoke-test.json`: frozen application validation.
- `artifacts/installed-smoke-test.json`: installation validation when requested.
- `artifacts/build-manifest.json`: versions, source commit, and model checksum.
- `artifacts/SHA256SUMS.txt`: installer integrity checksum.

Review these reports and exercise normal capture, mask review, hotkeys, and
live-click alignment on the intended workstation before a production release.
