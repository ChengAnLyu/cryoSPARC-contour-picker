# SARU Picker 2.3 for Windows

Install **SARU-Picker-Setup-2.3.0-win64.exe**, then open **SARU Picker** from the Start menu or desktop shortcut. This x64 CPU edition includes Python, its libraries, and the original SAM ViT-B checkpoint. No separate Python or model setup is needed. It upgrades the existing SARU Picker installation.

When the download includes a custom icon, first **extract the whole ZIP** so
`app-icon.png` and `app-icon.ico` are beside setup. The installer applies them
to the installed application, its EXE and its shortcuts. The setup EXE itself
uses the public geometric icon. Setup also works without the optional sidecars,
using that fallback for the installed app.

## Work directly over cryoSPARC

The normal application is the original compact floating control panel. cryoSPARC keeps its own layout and remains the live image viewer. The panel stays visible during drawing, detection and picking, and sizes itself to fit its controls and Windows text scaling. The terminate button has a reserved footer. Version 2.3 preserves these UI fixes and the original manual picking path.

| Mode | Screen interaction |
| --- | --- |
| manual | Uncheck Disable Picking, hold left mouse to trace a red contour, then release to fill it with picks. No ROI or model step. |
| semi-auto | Select screen area, Detect, then click and release inside one intended green contour to fill only that contour. Box prompt and Next alternative support guided selection. |
| auto | Select screen area, optionally Detect first, choose frame count and wait, enable picking, and Start auto. It fills every displayed contour one after another before advancing frames. |

Escape stops and disables picking. Space pauses/resumes an active auto run; F8 is an additional pause key. Keep cryoSPARC on the primary monitor and keep its micrograph zoom and position fixed during recognition and playback.

The earlier 2.0 default snapshot editor is now optional, available with --review. Screenshots are internal model input; manual mode takes no screenshots. Keep the control panel outside the selected model screen area. It remains visible throughout detection and picking.

## Installed data

The build verifies and includes sam_vit_b_01ec64.pth, 375,042,383 bytes, SHA-256:

`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

The app finds this checkpoint independently of its working directory. Logs use %LOCALAPPDATA%\SARU_Picker. Installation is per user, with an uninstaller registered in Windows Settings. This test edition is unsigned.

## Build and verification

The build is defined by packaging/build.ps1 and .github/workflows/build-windows.yml. It downloads verified model weights from Meta's published URL, keeps those large weights out of Git, and collects dependency notices.

Before packaging, it runs source regressions and native Windows UI checks. The executable and installed copy then run image-processing, native torchvision, compact-panel, transparent-overlay, input-surface, and synthetic SAM inference checks. A successful run uploads the installer, checksums and reports as an Actions artifact.

Native UI checks create their own synthetic target window. They verify that annotation pixels are mouse-transparent, that the ROI input catcher receives interior selection, and that hiding the surfaces restores the target. Three consecutive semi-auto selections use real Tk canvas events and the background worker with an in-memory target; delayed pick markers are introduced before selection two. The tests also verify the small and large native icon handles. They send no operating-system input to cryoSPARC and do not test biological accuracy.

The installer uses a normal application folder so its model does not need to be unpacked each launch. It is built on Windows because [PyInstaller distributions are platform-specific](https://pyinstaller.org/en/stable/operating-mode.html).

The default capsid profile adds a conservative texture-based interior filter and polygon fitting to original SAM. It is not a trained capsid classifier; faint targets, fragments, and ambiguous material still need review. The fitted polygon mask drives both outline display and picking. Advanced settings retain a generic SAM profile for comparison. Both screen model modes use the manual spacing convention per contour and avoid crowding earlier picks. Live capture, hotkeys and click alignment must also be checked on the user's workstation.

Version 2.3 records completed picks before restoring the next selectable contour and excludes their marker neighborhoods from scene-change checks. It also releases completed worker callbacks on the GUI thread, keeps polling alive after recoverable errors, and cancels timers during shutdown. Terminate requests cancellation immediately; if a native model calculation is still running, the panel shows a stopping message until it can close safely. Custom artwork is used for the installed application, shortcuts and uninstall entry.

For an unexpected exit, preserve `%LOCALAPPDATA%\SARU_Picker\saru_picker.log`, `startup.log` and `native_fault.log` before relaunching repeatedly. The last file can record native exceptions that an ordinary Python error handler cannot catch; it is normally empty.

See [the main guide](README.md) for operation and [maintainer instructions](packaging/README.md) for building.
