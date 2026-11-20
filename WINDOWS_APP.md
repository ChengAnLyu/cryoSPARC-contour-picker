# SARU Picker 2.6.1 for Windows

Install **SARU-Picker-Setup-2.6.1-win64.exe**, then open **SARU Picker** from the Start menu or desktop shortcut. This x64 CPU edition includes Python, its libraries, and the original SAM ViT-B checkpoint. No separate Python or model setup is needed. It upgrades the existing SARU Picker installation.

When the download includes a custom icon, first **extract the whole ZIP** so
`app-icon.png` and `app-icon.ico` are beside setup. The installer applies them
to the installed application, its EXE and its shortcuts. The setup EXE itself
uses the public geometric icon. Setup also works without the optional sidecars,
using that fallback for the installed app.

## Version 2.6.1 update

Both capsid detection modes add a shared carbon-boundary check. It uses a candidate’s relationship to a much longer image boundary instead of rejecting all straight sides or all material on one side of a line. The existing learned profile, manual interaction, compact panel and picking controls are retained.

## Version 2.6 update

Semi-auto contour presentations now use a fresh temporary annotation window after each completed selection. The persistent panel and original manual behavior remain. An optional `capsid-profile.json` supplied beside Setup is installed next to the SAM checkpoint and makes the experimental learned capsid appearance profile available through **Advanced → Detection profile → capsid-learned**. Extract the entire ZIP and retain that file and both custom-icon files beside Setup during installation. See [the capsid profile guide](CAPSID_PROFILE.md).

## Work directly over cryoSPARC

The normal application is the original compact floating control panel. cryoSPARC keeps its own layout and remains the live image viewer. The panel stays visible during drawing, detection and picking, and sizes itself to fit its controls and Windows text scaling. The terminate button has a reserved footer. Version 2.6 preserves the original manual gesture and spacing while retaining shared faster click pacing across all modes. A new manual stroke clears the previous outline before its new trace appears, with the panel retained.

| Mode | Screen interaction |
| --- | --- |
| manual | Uncheck Disable Picking, hold left mouse to trace a red contour, then release to fill it with picks. No ROI or model step. |
| semi-auto | Select screen area, Detect, then click and release inside one intended green contour to fill only that contour. Box prompt supports guided selection. Next micrograph sends Down, waits for a new stable image, and detects fresh contours without picking them. |
| auto | Select screen area, optionally Detect first, enable picking, and Start auto. It fills every contour, sends Down, waits for a new stable image, detects, and repeats. Limit (0=all) defaults to continuous operation. |

**Next micrograph** replaces Next alternative and works even with **Disable Picking** checked, so navigation and detection can be used for review. The selected screen area is retained. A new image receives a fresh set of contours and pick history.

In auto mode, **Limit (0=all)** accepts 0–1000. Zero continues until stopped or until another micrograph cannot be verified; a positive number includes the current image and images with no detections. Empty images are skipped. **Load (s)** is the maximum time to wait for a changed, stable image, default 10 seconds; the app continues sooner when loading completes. A navigation attempt sends Down once. An unchanged/end-of-list view, a recently repeated image, a loading timeout, or loss of target focus stops the cycle.

Manual, semi-auto and auto share a **0.01-second default click delay**. Set **0.00** for no added inter-click delay, or increase the value if the browser misses picks. Open settings using **Advanced…** in model modes, **F9** while the SARU panel has focus, or a right-click on the panel followed by **Picking speed…**. The manual panel layout stays compact.

Zero delay still yields between clicks, and per-click cancellation, focus/position checks and the mouse fail-safe remain active. Actual accepted-pick throughput depends on Windows and cryoSPARC; this setting does not speed up model inference or image loading. See [version 2.5 validation scope](VALIDATION.md#version-250-shared-picking-speed-and-manual-outline-cleanup). The earlier 2.4 synthetic timing results are retained as historical evidence.

Escape stops and disables picking. Space pauses/resumes an active auto run; F8 is an additional pause key. Keep cryoSPARC on the primary monitor and keep its micrograph zoom and position fixed during recognition and playback.

The earlier 2.0 default snapshot editor is now optional, available with --review. Screenshots are internal model input; manual mode takes no screenshots. Keep the control panel outside the selected model screen area. It remains visible throughout detection and picking.

## Installed data

The build verifies and includes sam_vit_b_01ec64.pth, 375,042,383 bytes, SHA-256:

`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

The app finds this checkpoint independently of its working directory. Logs use %LOCALAPPDATA%\SARU_Picker. Installation is per user, with an uninstaller registered in Windows Settings. This test edition is unsigned.

## Build and verification

The build is defined by packaging/build.ps1 and .github/workflows/build-windows.yml. It downloads verified model weights from Meta's published URL, keeps those large weights out of Git, and collects dependency notices.

Before packaging, it runs source regressions and native Windows UI checks. The executable and installed copy then run image-processing, native torchvision, compact-panel, transparent-overlay, input-surface, and synthetic SAM inference checks. A successful run uploads the installer, checksums and reports as an Actions artifact.

Native UI checks create their own synthetic target window. They verify that annotation pixels are mouse-transparent, that the ROI input catcher receives interior selection, and that hiding the surfaces restores the target. Repeated semi-auto selections use real Tk canvas events and the background worker with an in-memory target; delayed pick markers are introduced before selection two. Navigation fixtures exercise the detect/pick/Down cycle and semi-auto navigation while picking is disabled. The tests also verify the small and large native icon handles. They send no operating-system input to cryoSPARC and do not test biological accuracy.

The installer uses a normal application folder so its model does not need to be unpacked each launch. It is built on Windows because [PyInstaller distributions are platform-specific](https://pyinstaller.org/en/stable/operating-mode.html).

The default **capsid** setting retains the earlier texture-based interior filter. The optional **capsid-learned** setting requires its installed sidecar and enables the experimental appearance model. Its development results include both recovered targets and missed or unwanted regions, so inspect it in semi-auto mode first. Both paths use polygon fitting after selecting image-supported regions. Faint targets, fragments and ambiguous material still need review, and the learned scores are not calibrated capsid probabilities. The fitted polygon mask drives both outline display and picking. Advanced settings retain a generic SAM profile for comparison. Both screen model modes use the manual spacing convention per contour and avoid crowding earlier picks. Live capture, hotkeys and click alignment must also be checked on the user's workstation.

The repairs introduced in version 2.3 remain: completed picks are recorded before restoring the next selectable contour, and their marker neighborhoods are excluded from scene-change checks. Completed worker callbacks are released on the GUI thread, polling remains alive after recoverable errors, and timers are cancelled during shutdown. Terminate requests cancellation immediately; if a native model calculation is still running, the panel shows a stopping message until it can close safely. Custom artwork is used for the installed application, shortcuts and uninstall entry.

For an unexpected exit, preserve `%LOCALAPPDATA%\SARU_Picker\saru_picker.log`, `startup.log` and `native_fault.log` before relaunching repeatedly. The last file can record native exceptions that an ordinary Python error handler cannot catch; it is normally empty.

See [the main guide](README.md) for operation and [maintainer instructions](packaging/README.md) for building.
