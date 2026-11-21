# SARU Picker 3.0.0

SARU Picker works directly over the cryoSPARC screen. cryoSPARC remains the image viewer, with its existing layout and controls. A compact, always-on-top **SARU PICKER** panel selects manual, semi-auto, or auto mode.

## Windows application

Version **3.0.0** integrates the trained **crYOLO 1.9.9** detector as the default recognition model. It carries over the latest 2.9.0 screen controls, manual drawing, contour refinement, grid sampling, carbon checks, 3,500-screen-pixel automatic area minimum, click pacing and micrograph navigation. Both `main.py` and `main_cryolo.py` now start the crYOLO configuration with the learned capsid appearance profile.

Extract the complete private release ZIP and run **SARU-Picker-crYOLO-Setup-3.0.0-win64.exe**. Keep `cryolo-capsids.h5`, `cryolo-config.json`, `capsid-profile.json`, `app-icon.ico` and `app-icon.png` beside Setup; it installs them automatically. The public build artifacts contain the GUI only. Training data and weights are supplied privately.

Run **Start → SARU Picker crYOLO → Install crYOLO runtime** once if the runtime is not already installed. This obtains stock crYOLO in an isolated CPU environment. Existing users of the earlier crYOLO edition retain its installation, runtime and settings locations. The SAM 2.9.0 installation remains separate. Launch **SARU Picker crYOLO** and select the screen area as before. See [crYOLO model and training details](CRYOLO.md) and [3.0.0 Windows build instructions](packaging/cryolo/README.md).

The model is the previously completed 160-example, 20-epoch fit, selected at epoch 18. Version 3.0.0 integrates that exact checkpoint; it does not claim another training run or improved accuracy from the version change. At confidence 0.3, its in-sample contour evaluation covered at least half of 131/146 confirmed marked cores and produced 16 unwanted contours across 43 confirmed blank examples. These partial annotations do not establish exhaustive recall or exact biological boundaries.

## Manual mode — original interaction

1. Keep cryoSPARC's picking view on the primary Windows monitor.
2. Choose **manual** and uncheck **Disable Picking**.
3. Hold the left mouse button and draw a contour directly over the micrograph. A red outline follows the pointer.
4. Release the button. The app fills the contour with picks in cryoSPARC and becomes ready for the next contour.

Manual mode requires no screen-area selection, model, image import, mask acceptance, or separate pick button. The original panel controls remain: the three mode buttons, **Particle spacing** slider (5–25 screen pixels, default 13), **Disable Picking**, status, and **terminate**.

Manual pick placement preserves the original contour bounding-box origin, integer hexagonal row spacing, alternating half-spacing offset, polygon boundary rule and zero inset. It now uses the same click-delay setting as semi-auto and auto, defaulting to **0.01 seconds**. Starting a new stroke clears the previous outline before displaying the new trace, while the control panel stays visible. The implementation uses a worker and queued drawing events to keep the panel responsive. Generated clicks are excluded from the contour listener. Manual mode takes no screenshots and starts picking directly on release; each stroke selects its own target window. The panel's size fits the requested controls at the current Windows text scaling, including a reserved footer for **terminate**.

## Semi-auto mode

1. Choose **semi-auto**.
2. Use **Select screen area** and drag around the micrograph on the actual screen. Keep browser toolbars and other controls outside this area.
3. Choose **Detect**. The model reads a clean screenshot internally and draws green proposal outlines over cryoSPARC.
4. Uncheck **Disable Picking** when ready. Click inside an intended green region. After the mouse button is released, only that contour is filled with picks directly in cryoSPARC. The panel stays visible.
5. Select another proposal, or run **Detect** again on the current view.
6. Choose **Next micrograph** to send one **Down** key to the verified cryoSPARC window. The app waits for a different, stable image and detects fresh contours in the same screen area. It does not pick them until you click a contour.

**Next micrograph** replaces the former Next alternative button. Navigation and detection also work while **Disable Picking** is checked; the new contours remain available for review without generating picks. If the view does not change or is still loading at the timeout, the operation stops rather than sending repeated Down keys.

Use **Select by box** to select crYOLO detections within a dragged screen region. Box/point prompts select existing detections; they cannot make the detector recover a missed object. Manual drawing remains available. No separate image editor is needed.

The model uses the selected screen area, replacing the original hardcoded MICRO_REGION. Mask-interior selection respects holes and disconnected components. Each contour uses its own bounding-box origin, integer row spacing, alternating offset, and zero inset, matching the manual spacing convention. All three modes share the **0.01-second default click delay**. Bitmap membership determines the detected boundary. Later contours retain their own lattice and omit sites too close to previously planned or actual picks; overlapping regions do not receive nearly coincident picks.

Completed picks are recorded before the next contour becomes selectable. Scene checks exclude small neighborhoods around those known picks, so delayed cryoSPARC marker painting does not make the second contour fail a full-image comparison. The remaining image is still checked for movement. A real alignment error stops picking and keeps the panel available; select/detect again after changing the image or zoom.

## Auto mode

Choose **auto**, select the screen area, uncheck **Disable Picking**, and choose **Start auto**. If Detect has already displayed contours, Start auto uses that exact ordered set after checking that the image still matches. Otherwise it detects first. It finishes contour 1, then contour 2, and so on, with per-contour progress. After finishing a micrograph, it sends one **Down** key, waits for a different stable image, detects its contours, and repeats the picking/navigation cycle.

- **Limit (0=all)** defaults to **0**, which continues until stopped or until another micrograph cannot be verified. Set **1** to process only the current micrograph, or **2–1000** for a bounded run. A positive limit includes the current micrograph and images with no detected contours.
- **Load (s)** defaults to **10 seconds** and is the maximum loading timeout, not a fixed delay after each Down key. Detection starts once the changed image is stable. Increase this value for a slower-loading cryoSPARC session.
- An image with **no detected contours** is counted as processed and skipped. Auto continues to the next image while its limit allows.

Auto mode picks every displayed contour. The default **capsid-learned** setting traces contours inside crYOLO detections using the supplied appearance profile. Faint cores and fragments can be missed, and unwanted material can survive. Check representative images in semi-auto mode before an automatic run. Re-detecting the same scene retains the recorded pick positions; a verified new frame clears them.

The app stops if navigation leaves the same image visible, returns to a recently seen image, or fails to reach a stable new image before the timeout. It also stops on lost target focus, unexpected movement during detection/picking, other errors, or insufficient stable background to verify remaining masks. It sends no repeated Down-key retries. At the end of a list, an unchanged view stops the run. Image-change checks are heuristics, not image registration. Keep the image zoom and position fixed during recognition and picking.

## Screen controls and behavior

- **Escape** disables picking and cancels active or pending work.
- **Space** pauses/resumes an active auto run. Its press/release pair is consumed only during that run so it cannot scroll the browser underneath the picks.
- **F8** is an additional pause/resume key.
- **F9**, while the SARU panel has focus, opens settings in any mode. You can also right-click the panel and choose **Picking speed…**.
- **Disable Picking** removes the input-catching layer so cryoSPARC receives ordinary interaction again.
- **terminate** closes the application and its listeners.

Visible annotations are passive. A separate, nearly transparent input surface catches ROI selection, box prompts, and semi-auto mask clicks. Only the temporary annotation/input layers are hidden before screenshots and generated clicks. The control panel remains visible, interactive, and above these layers. Keep it outside the selected micrograph area; an overlapping panel produces an explicit message instead of disappearing. The underlying screen is never replaced by a required snapshot editor.

Live control currently supports the primary Windows monitor. Moving the micrograph or changing zoom requires a fresh detection. Up to 10,000 generated clicks are allowed per operation. Cancellation is checked between clicks; an active native model forward pass must finish before its cancelled result can be discarded.

## Picking speed and model settings

The click delay and particle spacing retain the 2.9.0 behavior. Recognition latency is separate from click playback speed. A new capture starts crYOLO in its isolated environment; model startup contributes to latency. Cached predictions are reused for an identical capture. No hardware-matched speed improvement over SAM is claimed.

**Advanced** provides the crYOLO weights, matching configuration, runtime Python, confidence threshold and contour settings. Default confidence is **0.3**. Lowering it can recover more marked regions but also creates more unwanted detections. The detector supplies bounding boxes; existing image-based capsid processing refines their interiors, checks carbon boundaries and filters final contour size. These contours are approximate picking regions.

Detection confidence is separate from mask IoU. Changing recognition settings retires displayed proposals and requires fresh detection, while retaining actual pick history. Missing weights, a broken runtime or failed inference produces an error; the application does not silently substitute SAM.

## Running from source

Use Python 3.12 x64 with Tcl/Tk for the GUI:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --checkpoint C:\models\cryolo-capsids.h5 --cryolo-config C:\models\cryolo-config.json --cryolo-python C:\path\to\cryolo-env\Scripts\python.exe
```

Keep `capsid-profile.json` beside the H5 weights. The Windows runtime installer is `packaging/cryolo/Install-crYOLO-Runtime.cmd`; Advanced can also select an existing compatible crYOLO Python. The modern GUI and stock Python 3.7 model runtime remain isolated.

Legacy SAM is available from source with explicit `--backend sam`, its `.pth` checkpoint, and separately installed PyTorch/torchvision and `requirements-sam.txt`. The 3.0.0 crYOLO Windows bundle does not include the SAM runtime or weights.

## Optional offline tools

`python main.py --review` opens the separate snapshot editor. `--image my_micrograph.png` opens an image there for offline work. These are optional; ordinary launching uses the screen overlay.

Headless segmentation remains available, with no desktop input:

```powershell
python main.py --segment-image my_micrograph.png --checkpoint cryolo-capsids.h5 --cryolo-python C:\path\to\cryolo-env\Scripts\python.exe --device cpu --output result
```

Without a box or point, headless recognition uses automatic crYOLO detection and the learned contour profile. Box/point arguments select matching detections. CLI and optional offline review exports retain their explicit spacing/inset controls; live screen modes use the per-contour manual convention described above.

Exports are bitmap masks and capture/screen pixel coordinates, not calibrated micrograph coordinates or direct cryoSPARC .cs imports.

## Validation and diagnostics

```powershell
python -m pytest -q
python main.py --diagnostics --report diagnostics.json
python main.py --smoke-test --cryolo-python C:\path\to\cryolo-env\Scripts\python.exe --report smoke.json
```

Diagnostics checks metadata. The smoke test performs real inference and requires the trained weights, configuration and runtime. Windows build checks exercise native controls and overlays at four text scales, the crYOLO Advanced panel, frozen version/backend metadata, installer sidecar copying and uninstall. The separate stock-runtime job executes synthetic network inference; private biological accuracy comes from the supplied model evaluation record.

Logs remain in `%LOCALAPPDATA%\SARU_Picker_crYOLO` on Windows and `~/.saru_picker_cryolo` elsewhere. The installed launcher also records startup and native-fault information there. Historical SAM packaging files remain in `packaging/`; use `packaging/cryolo/build.ps1` for the 3.0.0 release.
