# SARU Picker 2.9.0

SARU Picker works directly over the cryoSPARC screen. cryoSPARC remains the image viewer, with its existing layout and controls. A compact, always-on-top **SARU PICKER** panel selects manual, semi-auto, or auto mode.

## Windows application

The CPU installer is named **SARU-Picker-Setup-2.9.0-win64.exe**. It includes Python, the desktop libraries, and the original SAM ViT-B model. After installation, launch **SARU Picker** from the Start menu. See [Windows installation](WINDOWS_APP.md) and [maintainer build instructions](packaging/README.md).

Version 2.9.0 supports the private appearance profile trained with all 160 supplied examples. Examples 121–160 add confirmed large-capsid picks; the 43 previously confirmed blank images remain negatives. Small markings and unmarked regions in nonblank images remain unknown. Select **Advanced → Detection profile → capsid-learned** to use the updated profile. Learned-mode geometry calculations now run only over the relevant local image regions, avoiding repeated full-image work while preserving the existing prompt and contour rules. The full image is still scored, SAM is encoded once, and automatic prompting retains its 24-prompt limit. The private guide reports measured coverage and runtime comparisons. Manual picking, the floating panel, click pacing, carbon checks, the 3,500-pixel automatic area floor and SAM weights retain their existing behavior. Full-image search does not guarantee exhaustive capsid detection or exact biological boundaries.

Version 2.8.0 adds explicit confirmed-blank supervision to appearance-profile training. The private update combines 120 examples, uses confirmed large-capsid markings as positives, and treats the 43 confirmed blank images as negative examples. Unmarked regions in other images remain unknown. Select **Advanced → Detection profile → capsid-learned** to use the updated private profile. The learned detector also adds tight prompts for strong cores missed inside merged support, and checks final contour size before overlap suppression so a rejected small fragment cannot hide a usable larger contour. The installed profile, training audit and measured internal comparison are supplied in the private release bundle. The on-screen workflow, manual picking, picking speed, carbon checks, 3,500-pixel automatic size floor and SAM checkpoint retain their existing behavior.

Version 2.7.2 corrects semi-auto handling of overlapping proposals that have no new pick positions. After a contour finishes, remaining proposals are checked independently against actual completed picks. Fully covered proposals are removed before the selectable overlay returns. A zero-pick selection explains whether the contour is already covered or has no grid sites at the current spacing, and the final status says when no selectable contours remain. Pick history, duplicate-spacing protection, manual controls and model weights are retained. The Windows regression fixture now exercises eight fresh contours plus a covered, zero-pick selection after the fifth contour.

Version 2.7.1 excludes small automatically detected capsid regions by default. Both **capsid** and **capsid-learned** require each final connected region to contain at least **3,500 native screen pixels**. Change **Advanced → Minimum capsid area (screen px²)** to adjust this for your zoom; **0** restores all sizes. The cutoff counts actual foreground pixels, excludes holes, and never combines disconnected small fragments to pass the limit. It is applied after contour refinement, so surviving contours keep their original boundaries. Empty detection results stay empty. Manual contours, explicit Box prompts, and generic SAM proposals retain their existing behavior. Changing recognition settings retires old displayed contours and requires Detect again while retaining actual pick history. This release keeps the 2.7.0 learned weights; no retraining is claimed.

Version 2.7.0 supports the updated private appearance profile trained with examples 1–80. The training utility records both source decks, holds every screenshot out during development evaluation, and supports conservative positive interiors from small marked groups as well as balanced sampling across examples. Recognition now derives compact prompt regions when an attached extension would otherwise prevent SAM from outlining a capsid; the stricter final shape and carbon checks still apply. Large-capsid coverage and contour boundaries have priority in the private model comparison. The delivered model and its remaining limitations are documented in the private guide. The screen controls, manual gesture, picking speed, and neural-network architecture are retained.

Version 2.6.1 introduced the shared carbon-boundary check used by both capsid detection modes. It uses long image-context edges to identify support-boundary artifacts while retaining compact neighboring objects. That release retained the earlier learned weights; the 2.7.0 private package supplies a newly trained appearance profile. Neither update retrains SAM or assumes that every region on carbon is unwanted. Manual drawing, generic SAM proposals and explicit box prompts retain their original behavior.

Version 2.6.0 replaces the temporary annotation window between semi-auto contour selections so a completed contour is not remapped from an old painted buffer. It also supports an optional [learned capsid appearance profile](CAPSID_PROFILE.md). The compact panel, original manual gesture and spacing, polygonal regions, **Next micrograph**, continuous auto cycle and private custom icons are retained. All three picking modes retain the shared **0.01-second default** and **0.00-second option with no added inter-click delay**. Auto completes one contour before the next. The optional snapshot editor remains available with `--review`.

For a custom-icon download, **extract the complete ZIP before running setup**. Keep `app-icon.png` and `app-icon.ico` beside the setup EXE during installation. Setup copies this artwork into the installed application and updates its EXE and shortcut icons. Personal artwork is distributed with the private download rather than committed to the public source tree. Running setup without these optional files uses the repository's geometric fallback icon.

If the download includes **capsid-profile.json**, keep it beside setup as well. Installation places this learned appearance profile next to the bundled SAM model; **Advanced → Detection profile → capsid-learned** selects the experimental learned detector. The default **capsid** detector uses its existing appearance heuristic and the same new size cutoff; installing the sidecar does not select the learned mode. The public installer without that optional profile retains the earlier texture-based detector. See [profile installation and validation limits](CAPSID_PROFILE.md).

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

Use **Box prompt** to drag a tight box around an intended region directly on screen. Click the displayed proposal to pick it, or draw a new box to refine the prompt. Prompting does not require a separate image editor.

The model uses the selected screen area, replacing the original hardcoded MICRO_REGION. Mask-interior selection respects holes and disconnected components. Each contour uses its own bounding-box origin, integer row spacing, alternating offset, and zero inset, matching the manual spacing convention. All three modes share the **0.01-second default click delay**. Bitmap membership determines the detected boundary. Later contours retain their own lattice and omit sites too close to previously planned or actual picks; overlapping regions do not receive nearly coincident picks.

Completed picks are recorded before the next contour becomes selectable. Scene checks exclude small neighborhoods around those known picks, so delayed cryoSPARC marker painting does not make the second contour fail a full-image comparison. The remaining image is still checked for movement. A real alignment error stops picking and keeps the panel available; select/detect again after changing the image or zoom.

## Auto mode

Choose **auto**, select the screen area, uncheck **Disable Picking**, and choose **Start auto**. If Detect has already displayed contours, Start auto uses that exact ordered set after checking that the image still matches. Otherwise it detects first. It finishes contour 1, then contour 2, and so on, with per-contour progress. After finishing a micrograph, it sends one **Down** key, waits for a different stable image, detects its contours, and repeats the picking/navigation cycle.

- **Limit (0=all)** defaults to **0**, which continues until stopped or until another micrograph cannot be verified. Set **1** to process only the current micrograph, or **2–1000** for a bounded run. A positive limit includes the current micrograph and images with no detected contours.
- **Load (s)** defaults to **10 seconds** and is the maximum loading timeout, not a fixed delay after each Down key. Detection starts once the changed image is stable. Increase this value for a slower-loading cryoSPARC session.
- An image with **no detected contours** is counted as processed and skipped. Auto continues to the next image while its limit allows.

Auto mode picks every displayed contour. The default capsid setting uses the earlier texture heuristic; **capsid-learned** explicitly selects the optional experimental appearance model. Neither path guarantees biological classification: faint cores and fragments can be missed, and unwanted material can survive. Check representative images in semi-auto mode before an automatic run. Re-detecting the same scene retains the recorded pick positions; a verified new frame clears them.

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

The **click delay** setting applies to manual, semi-auto and auto picking. Its default is **0.01 seconds**. Set it to **0.00** for the fastest available pacing with no added inter-click delay; increase it if the cryoSPARC browser does not register every pick. Open settings from **Advanced…** in model modes, press **F9** while the SARU panel has focus, or right-click the panel and choose **Picking speed…**. These access points preserve the compact manual layout.

Zero delay still yields execution between clicks and retains per-click cancellation, target-focus checks, position checks and the mouse fail-safe. It does not remove browser or operating-system processing time, accelerate detection, or guarantee a particular number of accepted picks per second. Use the lowest delay that registers reliably on the intended workstation. See [version 2.5 validation scope](VALIDATION.md#version-250-shared-picking-speed-and-manual-outline-cleanup); the recorded 2.4 benchmark remains historical.

Settings also expose checkpoint selection, model type, CPU/CUDA device selection, detection profile and automatic proposal grid density. The installer supplies the checkpoint automatically and contains CPU PyTorch. A source installation can use a compatible CUDA PyTorch build.

The default **capsid** setting starts with permissive SAM proposals, then compares granular interior texture with the nearby background and the rest of the selected image. It refines supported interiors, fits polygonal boundaries, and resolves duplicate regions after those content checks. The polygon fit simplifies noisy edges while bounding shape changes, retaining separate objects, substantial holes, and partially visible image-edge regions. The fitted mask is used for the displayed outline, click selection, and pick placement. Broad background, membranes without dense interiors, and small unsupported fragments are therefore less likely to survive solely because they received a high SAM quality score. The fallback consists of explicit image heuristics. The opt-in **capsid-learned** setting uses the experimental trained appearance model described in [CAPSID_PROFILE.md](CAPSID_PROFILE.md), guides SAM with image-supported box-and-point prompts, and refines polygonal cores. It requires its sidecar and limits input to 4,194,304 pixels. Development results are mixed, so it does not replace the default detector or make the output a validated biological label. **generic** and box-prompt alternatives retain their original SAM masks.

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

Without a box or point, headless segmentation uses the capsid profile. Add `--automatic-profile generic` for the earlier proposal pipeline, or `--automatic-profile capsid-learned` for the experimental trained profile. CLI and optional offline review exports retain their explicit spacing/inset controls; the live screen modes use the per-contour manual convention described above.

Exports are bitmap masks and capture/screen pixel coordinates, not calibrated micrograph coordinates or direct cryoSPARC .cs imports.

## Validation and diagnostics

```powershell
python -m unittest discover -s tests
python main.py --diagnostics --report diagnostics.json
```

The Windows build tests complete control/button layout at four simulated Windows Tk font scales, persistent panel visibility, native overlay hit testing and stacking, application icons, repeated semi-auto selection, and the micrograph navigation cycle through real Tk surfaces and the worker. It also checks image processing, compiled torchvision operations, and real synthetic SAM inference. The repeated-contour test includes delayed pick-marker painting. It then installs, tests, and uninstalls the package. These checks use synthetic windows and an in-memory click target and send no clicks or keys to cryoSPARC. Live picking alignment and browser navigation still need observation on the intended workstation.

Logs are in %LOCALAPPDATA%\SARU_Picker on Windows and ~/.saru_picker elsewhere. The installed launcher additionally records native-fault tracebacks in `native_fault.log` when Python can capture them. If the process exits unexpectedly, preserve that file together with `saru_picker.log` and `startup.log`. A handled scene-change stop is recorded in the ordinary application log. The build and dependency manifest records the exact installer inputs.
