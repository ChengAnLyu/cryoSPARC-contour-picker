# SARU Picker 2.1

SARU Picker works directly over the cryoSPARC screen. cryoSPARC remains the image viewer, with its existing layout and controls. A compact, always-on-top **SARU PICKER** panel selects manual, semi-auto, or auto mode.

## Windows application

The CPU installer is named **SARU-Picker-Setup-2.1.1-win64.exe**. It includes Python, the desktop libraries, and the original SAM ViT-B model. After installation, launch **SARU Picker** from the Start menu. See [Windows installation](WINDOWS_APP.md) and [maintainer build instructions](packaging/README.md).

Version 2.1.1 fixes clipped controls, keeps the panel visible throughout screen operations, and restores immediate manual drawing without screenshot validation. The optional snapshot editor remains available with `--review`.

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
4. Uncheck **Disable Picking** when ready. Click inside an intended green region. After the mouse button is released, the overlay hides and the app places the picks directly into cryoSPARC.
5. Select another proposal, or run **Detect** again on the current view.

Use **Box prompt** to drag a tight box around an intended region directly on screen. **Next alternative** cycles the model's alternatives; click the displayed region to pick it. Prompting does not require a separate image editor.

The model uses the selected screen area, replacing the original hardcoded MICRO_REGION. Mask-interior selection respects holes and disconnected components. Model modes sample the union of accepted masks with a two-pixel inset and avoid repeating grid positions already picked in the same capture.

## Auto mode

Choose **auto**, select the screen area, set **Frames** and **Wait (s)**, uncheck **Disable Picking**, and choose **Start auto**. One frame is the default. For additional frames, the app picks the current proposals, sends **Down** to the captured cryoSPARC window, waits, and checks that a new stable image appeared.

Auto mode accepts all model proposals. Original SAM has not been trained here to recognize the intended inner capsid region. Check representative images in semi-auto mode before using an automatic run.

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

**Advanced…** in model modes exposes checkpoint selection, model type, CPU/CUDA device selection, automatic proposal grid density, and model-mode click delay. The installer supplies the checkpoint automatically and contains CPU PyTorch. A source installation can use a compatible CUDA PyTorch build.

The model and image embeddings are reused where possible. Packaging and the restored screen workflow do not train the recognition model or establish biological recognition accuracy.

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

Exports are bitmap masks and capture/screen pixel coordinates, not calibrated micrograph coordinates or direct cryoSPARC .cs imports.

## Validation and diagnostics

```powershell
python -m unittest discover -s tests
python main.py --diagnostics --report diagnostics.json
```

The Windows build tests complete control/button layout at four simulated Windows Tk font scales, persistent panel visibility, native overlay hit testing and stacking, image processing, compiled torchvision operations, and real synthetic SAM inference. It then installs, tests, and uninstalls the package. These checks use synthetic windows and images and send no clicks or keys to cryoSPARC. Live picking alignment still needs a check on the intended workstation.

Logs are in %LOCALAPPDATA%\SARU_Picker on Windows and ~/.saru_picker elsewhere. The build and dependency manifest records the exact installer inputs.
