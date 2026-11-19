# SARU Picker 2.1 for Windows

Install **SARU-Picker-Setup-2.1.0-win64.exe**, then open **SARU Picker** from the Start menu or desktop shortcut. This x64 CPU edition includes Python, its libraries, and the original SAM ViT-B checkpoint. No separate Python or model setup is needed.

## Work directly over cryoSPARC

The normal application is the original compact floating control panel. cryoSPARC keeps its own layout and remains the live image viewer.

| Mode | Screen interaction |
| --- | --- |
| manual | Uncheck Disable Picking, hold left mouse to trace a red contour, then release to fill it with picks. No ROI or model step. |
| semi-auto | Select screen area, Detect, then click inside an intended green proposal to pick it. Box prompt and Next alternative support guided selection. |
| auto | Select screen area, choose frame count and wait, enable picking, and Start auto. All proposals are accepted for the bounded run. |

Escape stops and disables picking. Space pauses/resumes an active auto run; F8 is an additional pause key. Keep cryoSPARC on the primary monitor and keep its micrograph zoom and position fixed during recognition and playback.

The earlier 2.0 default snapshot editor is now optional, available with --review. Screenshots in the normal workflow are internal model input; they do not replace the live screen.

## Installed data

The build verifies and includes sam_vit_b_01ec64.pth, 375,042,383 bytes, SHA-256:

`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

The app finds this checkpoint independently of its working directory. Logs use %LOCALAPPDATA%\SARU_Picker. Installation is per user, with an uninstaller registered in Windows Settings. This test edition is unsigned.

## Build and verification

The build is defined by packaging/build.ps1 and .github/workflows/build-windows.yml. It downloads verified model weights from Meta's published URL, keeps those large weights out of Git, and collects dependency notices.

Before packaging, it runs source regressions and native Windows UI checks. The executable and installed copy then run image-processing, native torchvision, compact-panel, transparent-overlay, input-surface, and synthetic SAM inference checks. A successful run uploads the installer, checksums and reports as an Actions artifact.

Native UI checks create their own synthetic target window. They verify that annotation pixels are mouse-transparent, that the ROI input catcher receives interior selection, and that hiding the surfaces restores the target. They do not send mouse/keyboard events or test biological accuracy.

The installer uses a normal application folder so its model does not need to be unpacked each launch. It is built on Windows because [PyInstaller distributions are platform-specific](https://pyinstaller.org/en/stable/operating-mode.html).

Packaging improves deployment. The recognition model remains original SAM and its automatic proposals still need evaluation on representative micrographs. Live capture, hotkeys and click alignment must also be checked on the user's workstation.

See [the main guide](README.md) for operation and [maintainer instructions](packaging/README.md) for building.
