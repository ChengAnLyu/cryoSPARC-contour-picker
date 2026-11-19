# SARU Picker 2

Review segmentation masks on a captured image, then place a hexagonal grid of picks in cryoSPARC.

## Windows desktop application

The Windows CPU build creates **`SARU-Picker-Setup-2.0.0-win64.exe`** with Python, dependencies, and the SAM ViT-B model included. Once that build has passed its installation checks, users can install it and open **SARU Picker** from the Start menu without setting up Python or model files. See [Windows application instructions](WINDOWS_APP.md) and [build instructions](packaging/README.md). The build workflow uploads a verified installer as a GitHub Actions artifact; it does not publish a release automatically.

The source application is also available as **`main.py`**: put it alongside `sam_vit_b_01ec64.pth`, or select that checkpoint using Browse. Modular source and regression tests are included for development.

The workflow now uses a **snapshot preview**. Manual drawing and SAM prompts happen in that preview; they do not send mouse clicks to cryoSPARC. Actual picking is a separate action after mask review. Original SAM generates segmentation proposals; it has not been trained here to classify the intended inner capsid region.

## Running from Python source

Use 64-bit Python **3.12** with Tcl/Tk support. The application syntax requires Python 3.10 or later. Windows desktop playback and CUDA execution still require validation on the user's computer; installer smoke tests exercise the interface and synthetic inference without controlling cryoSPARC.

In a Windows terminal opened in the extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For **CPU inference**, the torch/torchvision combination exercised in offline validation is:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, install the matching CUDA torch/torchvision pair from the [official PyTorch selector](https://pytorch.org/get-started/locally/) instead of the CPU command. Keep both packages from the same installation source. Official paired CPU commands are also listed in [PyTorch's version instructions](https://pytorch.org/get-started/previous-versions/).

Install original SAM and launch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-sam.txt
.\.venv\Scripts\python.exe main.py
```

Git must be available on PATH for the pinned SAM requirement. It uses the [official original SAM repository](https://github.com/facebookresearch/segment-anything), fixed at commit `dca509fe793f601edb92606367a655c15ac00fdf`. The weights are not included in this code bundle; reuse the supplied `sam_vit_b_01ec64.pth`. Choose model type `vit_b`. SAM2 or micro-sam2 checkpoints require different loading code.

In an existing environment, `python main.py` starts the standalone file. Developers can instead run `python app.py` from this folder. Manual drawing and offline review do not require loading SAM. If Tkinter is unavailable, install Python with Tcl/Tk support; `python -m tkinter` should open its demonstration window.

## Capture, segment, and review

1. Open cryoSPARC's picking view on the **primary Windows monitor**. Set its image position and zoom.
2. Click **Select screen region (3 s)**. Activate cryoSPARC during the countdown, then drag around the micrograph on the captured desktop. Recapture uses the same region. An image opened with **Open image** supports offline analysis only. Use a display-ready 8-bit export; raw 16-bit and floating-point images are rejected instead of silently clipping their intensity values.
3. Choose **Manual contour** to trace a region in the preview, or **Box around target** to draw a tight box around an intended inner region. Add **Foreground point (+)** or **Background point (−)** prompts as needed; right-click also adds a negative point.
4. Click **Segment prompt**. Review the yellow preview, use **Next alternative** if necessary, and click **Accept mask** for the intended region. Repeat for other regions.
5. **Generate automatic proposals** offers unclassified masks. Select or deselect them in the table or with the selection tool. Green means selected; yellow means a proposal. A high predicted IoU estimates mask quality, not confidence that it is a capsid.

The worker reuses the loaded model and caches the image embedding for repeated prompts. Device `auto` chooses CUDA when available, otherwise CPU. The default automatic proposal grid is 8 per side; increasing it may recover missed objects but adds work.

## Sampling and live playback

Spacing defaults to **13 captured-image pixels**, with a **2-pixel boundary inset**. Resizing the preview does not change that spacing. The grid samples the union of selected bitmap masks, preserving holes and avoiding duplicate points from overlapping masks. Coordinates are screen/capture pixels, without physical pixel-size calibration.

Review the point count, enable **live desktop clicks**, and choose **Pick selected masks**. Live playback requires a fresh screen capture and working global hotkeys. Opened image files cannot send desktop clicks. The application hides its preview, focuses the captured window, checks for image drift, and verifies target focus during playback. Keep cryoSPARC's image position and zoom fixed: drift checks are a heuristic, not image registration.

Click delay defaults to **0.08 seconds**. PyAutoGUI's additional default pause is disabled so this setting controls timing; its corner fail-safe remains enabled. Each operation is limited to **10,000 grid points**. Increase spacing or reduce the selection if the limit is exceeded. The preview draws at most the first 3,000 points but reports the full count.

**F8 pauses/resumes; Escape stops.** Cancellation is checked between clicks and during waits. An active native model forward pass cannot be interrupted immediately; its result is discarded after cancellation. Errors stop the run rather than silently advancing. Review and save the resulting picks using cryoSPARC.

## Optional automatic batch

After checking proposals on representative images, enable **Accept all automatic proposals**, choose a finite frame count and load delay, then select **Run bounded auto batch**. This explicitly accepts the model's proposals without reviewing each frame. The current next-image key is **Down**. The run stops on errors, no candidates, unstable loading, or a detected unchanged/recently repeated image. It is not evidence of reliable target recognition.

## Exports and offline command line

**Export image, masks and points** writes an aligned clean image, union mask, separate masks, `points.csv`, and `review.json`. Selection metadata records user review, not expert-verified ground truth. CSV columns are capture pixels and, for screen captures, corresponding screen pixels. These are not calibrated micrograph coordinates or direct cryoSPARC `.cs` imports. Export accepted masks before playback, which marks them as already picked.

Example headless segmentation of an 8-bit micrograph export, using a box in input-image pixel coordinates:

```powershell
python main.py --segment-image my_micrograph.png --checkpoint sam_vit_b_01ec64.pth --device cpu --box 195 120 370 265 --output example_result
```

Add repeatable point prompts such as `--point 290 190 1` or `--point 185 230 0`. Coordinates refer to the input image. CLI output uses the highest predicted-IoU prompted alternative, or all automatic proposals when no prompts are provided. It is explicitly **unreviewed** and sends no desktop events.

## Diagnostics and development

```powershell
python main.py --diagnostics
python -m unittest discover -s tests
```

Run tests from this folder with the environment's Python. Logs rotate in `%LOCALAPPDATA%\SARU_Picker\saru_picker.log` on Windows and `~/.saru_picker/saru_picker.log` elsewhere (`~` is the user's home directory). Include diagnostics and the relevant log when reporting a failure. The tested core dependency versions were NumPy 2.3.5, Pillow 13.0.0, and SciPy 1.17.0; the requirement ranges are compatibility bounds, not a claim that every combination was exercised. Improving automatic target recognition requires representative expert labels and evaluation on held-out micrographs.
