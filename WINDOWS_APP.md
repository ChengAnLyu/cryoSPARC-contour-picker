# SARU Picker for Windows

The Windows CPU edition packages the application, Python, its dependencies, and
the original SAM ViT-B checkpoint into a normal per-user installer. End users do
not need to install Python, PyCharm, pip packages, or model files separately.

## Install and use

The installer produced by the Windows build is named
`SARU-Picker-Setup-2.0.0-win64.exe`. Run it, complete the installation, and open
**SARU Picker** from the Start menu. A desktop shortcut is optional. Windows
Settings can uninstall it later.

The application runs locally after installation. Select the micrograph region,
draw a contour or use a box/point prompt, review the selected masks and sampling
points, and start playback. Escape stops; F8 pauses or resumes. Keep cryoSPARC's
image zoom and position fixed while using a capture.

This first installer is for Windows x64 and CPU inference. It includes the
same SAM ViT-B model used in the source version. Packaging improves setup and
launching; it does not train the model or change the recognition results.

## Model and application data

The bundled checkpoint is `sam_vit_b_01ec64.pth`, 375,042,383 bytes. The build
verifies its SHA-256 before packaging:

`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

The application finds the included model automatically, including when launched
from a Start-menu shortcut with a different working directory. Logs and other
writable application data use `%LOCALAPPDATA%\SARU_Picker`; the install directory
does not need to be writable during normal operation.

## Build checks

The build is configured to:

1. Install the pinned CPU runtime and package dependencies on Windows.
2. Verify the model's file size and checksum.
3. Run the regression suite and generate the standalone source file.
4. Freeze a windowed, one-directory application with PyInstaller.
5. Check bundled native DLLs and run the frozen application's smoke test.
6. Build the installer, install it into a test directory, run the installed
   smoke test, and uninstall the test copy.
7. Retain the installer, SHA-256 hashes, and validation reports as build artifacts.

The smoke test checks image processing, a native torchvision operation,
Windows GUI/runtime dependencies, and real SAM inference on a synthetic image.
It does not send mouse or keyboard events to cryoSPARC and does not measure
biological recognition accuracy.

## For maintainers

The build entry point is `packaging/build.ps1`, and the hosted Windows build is
defined in `.github/workflows/build-windows.yml`. The model is downloaded from
Meta's published checkpoint URL during the build, or supplied locally. It is
not stored in Git.

Build on Windows. PyInstaller's output depends on the host operating system;
a Linux build is not a Windows executable. The installed files use one-directory
packaging so the large model does not need to be unpacked on every launch.

See [PyInstaller's platform requirements](https://pyinstaller.org/en/stable/index.html)
and [Inno Setup's per-user installation setting](https://jrsoftware.org/is6help/topic_setup_privilegesrequired.htm).

The build must complete and its reports must be checked before describing a
particular installer as built or tested. Source-level and Linux smoke tests
alone do not validate the Windows executable.
