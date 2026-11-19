# SARU Picker 2.3

SARU Picker works directly over the cryoSPARC screen. cryoSPARC remains the image viewer, with its existing layout and controls. A compact, always-on-top **SARU PICKER** panel selects manual, semi-auto, or auto mode.

## Windows application

The CPU installer is named **SARU-Picker-Setup-2.3.0-win64.exe**. It includes Python, the desktop libraries, and the original SAM ViT-B model. After installation, launch **SARU Picker** from the Start menu. See [Windows installation](WINDOWS_APP.md) and [maintainer build instructions](packaging/README.md).

Version 2.3.0 preserves the stable floating panel and original manual interaction. It fits polygonal boundaries to the **capsid interior** detections, repairs repeated semi-auto picking and worker cleanup, and supports a private custom application icon. Auto completes one contour before the next. The optional snapshot editor remains available with `--review`.

For a custom-icon download, **extract the complete ZIP before running setup**. Keep `app-icon.png` and `app-icon.ico` beside the setup EXE during installation. Setup copies this artwork into the installed application and updates its EXE and shortcut icons. Personal artwork is distributed with the private download rather than committed to the public source tree. Running setup without these optional files uses the repository's geometric fallback icon.

## Manual mode — original interaction

1. Keep cryoSPARC's picking view on the primary Windows monitor.
2. Choose **manual** and uncheck **Disable Picking**.
3. Hold the left mouse button and draw a contour directly over the micrograph. A red outline follows the pointer.
4. Release the button. The app fills the contour with picks in cryoSPARC and becomes ready for the next contour.

Manual mode requires no screen-area selection, model, image import, mask acceptance, or separate pick button. The original panel controls remain: the three mode buttons, **Particle spacing** slider (5–25 screen pixels, default 13), **Disable Picking**, status, and **terminate**.

Manual pick placement preserves the original contour bounding-box origin, integer hexagonal row spacing, alternating half-spacing offset, polygon boundary rule, zero inset, and approximately 0.10-second click cadence. The implementation now uses a worker and queued drawing events to keep the panel responsive. Generated clicks are excluded from the contour listener. Manual mode takes no screenshots and starts picking directly on release; each stroke selects its own target window. The panel stays visible and its size fits the requested controls at the current Windows text scaling, including a reserved footer for **terminate**.

## Semi-auto mode

1. Choose **semi-auto**.
2. Use **Select screen area** and drag around the micrograph on the actual screen. Keep browser toolbars and other controls outside this area.
3. Choose **Detect**. The model reads a clean screenshot internally and draws green proposal outlines over cryoSPARC.
4. Uncheck **Disable Picking** when ready. Click inside an intended green region. After the mouse button is released, only that contour is filled with picks directly in cryoSPARC. The panel stays visible.
5. Select another proposal, or run **Detect** again on the current view.

Use **Box prompt** to drag a tight box around an intended region directly on screen. **Next alternative** cycles the model's alternatives; click the displayed region to pick it. Prompting does not require a separate image editor.

The model uses the selected screen area, replacing the original hardcoded MICRO_REGION. Mask-interior selection respects holes and disconnected components. Each contour uses its own bounding-box origin, integer row spacing, alternating offset, zero inset, and a default 0.10-second click interval, matching the manual spacing convention. Bitmap membership determines the detected boundary. Later contours retain their own lattice and omit sites too close to previously planned or actual picks; overlapping regions do not receive nearly coincident picks.

Completed picks are recorded before the next contour becomes selectable. Scene checks exclude small neighborhoods around those known picks, so delayed cryoSPARC marker painting does not make the second contour fail a full-image comparison. The remaining image is still checked for movement. A real alignment error stops picking and keeps the panel available; select/detect again after changing the image or zoom.

## Auto mode

Choose **auto**, select the screen area, set **Frames** and **Wait (s)**, uncheck **Disable Picking**, and choose **Start auto**. If Detect has already displayed contours, Start auto uses that exact ordered set after checking that the image still matches. Otherwise it detects first. It finishes contour 1, then contour 2, and so on, with per-contour progress. One frame is the default. For additional frames, the app finishes all current contours, sends **Down** to the captured cryoSPARC window, waits, and checks that a new stable image appeared.

Auto mode picks every displayed contour. The default capsid profile is an image-based filter on SAM proposals, **not a newly trained or validated capsid classifier**. It can miss faint cores and fragments and can still include unwanted material. Check representative images in semi-auto mode before an automatic run. Re-detecting the same scene retains the recorded pick positions; a verified new frame clears them.

The app stops on errors, empty proposals, changed or repeated images, lost target focus, or insufficient stable background to verify remaining masks. Image-change checks are heuristics, not image registration. Keep the image zoom and position fixed during recognition and picking.

## Screen controls and behavior

- **Escape** disables picking and cancels active or pending work.
- **Space** pauses/resumes an active auto run. Its press/release pair is consumed only during that run so it cannot scroll the browser underneath the picks.
- **F8** is an additional pause/resume key.
- **Disable Picking** removes the input-catching layer so cryoSPARC receives ordinary interaction again.
- **terminate** closes the application and its listeners.

Visible annotations are passive. A separate, nearly transparent input surface catches ROI selection, box prompts, and semi-auto mask clicks. Only the temporary annotation/input layers are hidden before screenshots and generated clicks. The control panel remains visible, interactive, and above these layers. Keep it outside the selected micrograph area; an overlapping panel produces an explicit message instead of disappearing. The underlying screen is never replaced by a required snapshot editor.

Live control currently supports the primary Windows monitor. Moving the micrograph or changing zoom requires a fresh detection. Up to 10,000 generated clicks are allowed per operation. Cancellation is checked between clicks; an active native model forward pass must finish before its cancelled result can be discarded.

## Model settings

**Advanced…** in model modes exposes checkpoint selection, model type, CPU/CUDA device selection, detection profile, automatic proposal grid density, and model-mode click delay. The installer supplies the checkpoint automatically and contains CPU PyTorch. A source installation can use a compatible CUDA PyTorch build.

The default **capsid** profile starts with permissive SAM proposals, then compares granular interior texture with the nearby background and the rest of the selected image. It refines supported interiors, fits polygonal boundaries, and resolves duplicate regions after those content checks. The polygon fit simplifies noisy edges while bounding shape changes, retaining separate objects, substantial holes, and partially visible image-edge regions. The fitted mask is used for the displayed outline, click selection, and pick placement. Broad background, membranes without dense interiors, and small unsupported fragments are therefore less likely to survive solely because they received a high SAM quality score. These are explicit image heuristics, not learned biological labels. **generic** and box-prompt alternatives retain their original SAM masks.

Modified core masks do not inherit the original proposal's predicted IoU or stability: those numbers do not describe the new boundary. Headless exports retain the original proposal evidence separately. The profile uses the existing checkpoint and does not require another model download or a second image encoding. Model, prompt embeddings, and identical-image automatic results are reused where possible. See the [recognition roadmap](RECOGNITION_ROADMAP.md) for the supervised-model plan and evaluation requirements.

## Running from source

Use Python 3.12 x64 with Tcl/Tk and Git. Create a virtual environment and install:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-sam.txt
.\.venv\Scripts\python.exe main.py
```

The source distribution does not include the large checkpoint. Put sam_vit_b_01ec64.pth beside main.py or select it in Advanced. The [official SAM source](https://github.com/facebookresearch/segment-anything) is pinned by requirements-sam.txt.

## Optional offline tools

`python main.py --review` opens the separate snapshot editor. `--image my_micrograph.png` opens an image there for offline work. These are optional; ordinary launching uses the screen overlay.

Headless segmentation remains available, with no desktop input:

```powershell
python main.py --segment-image my_micrograph.png --checkpoint sam_vit_b_01ec64.pth --device cpu --box 195 120 370 265 --output result
```

Without a box or point, headless segmentation uses the capsid profile. Add `--automatic-profile generic` for the earlier proposal pipeline. CLI and optional offline review exports retain their explicit spacing/inset controls; the live screen modes use the per-contour manual convention described above.

Exports are bitmap masks and capture/screen pixel coordinates, not calibrated micrograph coordinates or direct cryoSPARC .cs imports.

## Validation and diagnostics

```powershell
python -m unittest discover -s tests
python main.py --diagnostics --report diagnostics.json
```

The Windows build tests complete control/button layout at four simulated Windows Tk font scales, persistent panel visibility, native overlay hit testing and stacking, application icons, three consecutive semi-auto selections through the real Tk surfaces and worker, image processing, compiled torchvision operations, and real synthetic SAM inference. The repeated-contour test includes delayed pick-marker painting. It then installs, tests, and uninstalls the package. These checks use synthetic windows and an in-memory click target and send no clicks or keys to cryoSPARC. Live picking alignment still needs a check on the intended workstation.

Logs are in %LOCALAPPDATA%\SARU_Picker on Windows and ~/.saru_picker elsewhere. The installed launcher additionally records native-fault tracebacks in `native_fault.log` when Python can capture them. If the process exits unexpectedly, preserve that file together with `saru_picker.log` and `startup.log`. A handled scene-change stop is recorded in the ordinary application log. The build and dependency manifest records the exact installer inputs.
