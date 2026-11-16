# cryoSPARC Contour Picker

An interactive Python helper for contour-based particle picking alongside the cryoSPARC manual-picking workflow. Trace a capsid contour on screen, then populate the enclosed displayed area with a dense, regularly spaced set of automated mouse clicks. Adjust `click_interval` to control the sampling density.

**Windows is the preferred environment.** PyCharm can be used to edit and run the script; a Python interpreter is required.

## How it works

The script creates a transparent, always-on-top drawing overlay. Holding the left mouse button records a freehand contour and displays it in red. Releasing the button triggers a point-in-polygon calculation that selects a square grid of screen positions inside the contour, then sends a mouse click at each position. The polygon is implicitly closed between the final and first recorded points.

The supplied implementation operates on the live desktop. Its coordinate list consists of temporary **2D screen positions**, and the receiving picking application handles recording, saving, and exporting the resulting picks. It does not read cryoSPARC output files, write `.cs`, STAR, or CSV coordinate files, or reconstruct a 3D capsid surface. In a cryoSPARC workflow, prepare the relevant image and picking view before using this helper.

## Repository files

| File | Purpose |
| --- | --- |
| `particle_picker_ChengAn.py` | Runnable Python copy of the supplied script. |
| `particle_picker_ChengAn.txt` | Original supplied source, preserved unchanged. |
| `INTRODUCTION.md` | Brief overview of the intended workflow. |
| `requirements.txt` | Third-party Python dependencies. |
| `README.md` | Setup, operation, and parameter guidance. |

The `.py` and `.txt` versions contain identical source bytes. Edit and run the `.py` file for local use.

## Installation

1. Install Python 3 for Windows with Tcl/Tk support. PyCharm is an optional development environment; select the Python interpreter you intend to use for this project.
2. Download this repository and open a terminal in its folder. In PyCharm, use the project's terminal with the selected interpreter.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   The equivalent direct command is:

   ```bash
   python -m pip install pyautogui pynput
   ```

4. Confirm Tkinter is available:

   ```bash
   python -m tkinter
   ```

   A small demonstration window should open. Close it before running the picker. If Tkinter is missing, modify or reinstall your Python distribution with Tcl/Tk support. Tkinter is supplied through the Python installation rather than installed with `pip install tkinter`; see the [Python Tkinter documentation](https://docs.python.org/3/library/tkinter.html).

The source imports `pyautogui`, `pynput.mouse`, `tkinter`, and `time`. Only the first two require pip installation for a typical Windows Python setup. NumPy is not used by this version.

## Usage

1. Open the intended cryoSPARC manual-picking view and position the capsid image at the zoom level you will use. Keep the image position and zoom fixed while drawing and clicking.
2. Set `click_interval` in `particle_picker_ChengAn.py` if needed, then run the entire file:

   ```bash
   python particle_picker_ChengAn.py
   ```

   Alternatively, run that file from PyCharm. This version starts at the top level; it has no separate `main()` function.
3. When the transparent overlay appears, hold the **left mouse button** and trace the contour around the displayed region to fill. Bring the trace back toward its starting point to define the intended closed outline.
4. Release the button. With more than two recorded contour points, the script generates the interior grid and clicks its positions automatically. The red contour clears after the click loop finishes.
5. Review the resulting picks in cryoSPARC and use cryoSPARC's own controls to save or export them.

Begin with a small region to check that the target application receives clicks at the expected positions. The listener is global and the script sends real desktop mouse events, so keep the intended picking view in place during operation.

## Adjusting picking density

Edit this line near the top of the script before launching it:

```python
click_interval = 7  # density of picking
```

`click_interval` is the grid spacing along **both axes, in screen pixels**. Use a positive integer.

| Example value | Effect at the same screen zoom |
| --- | --- |
| `4` | Denser sampling, with more clicks. |
| `7` | Supplied default spacing. |
| `14` | Coarser sampling, with fewer clicks. |

For a fixed enclosed screen area, halving the interval gives approximately four times as many samples; the exact count depends on the contour and grid boundary. These values are screen-pixel distances, so changing the displayed image zoom changes the spacing represented in the underlying image. The script does not perform pixel-size calibration.

Click timing is separate from grid spacing. `time.sleep(0.01)` adds a delay after each click, and PyAutoGUI may add its own pause. `click_interval` does not specify a time interval.

## Stopping and practical limits

Stop the Python process when finished, for example with PyCharm's Stop control. This version has no dedicated Escape, pause, or undo shortcut.

PyAutoGUI's default fail-safe is left enabled. Moving the pointer to a corner of the primary monitor can raise `FailSafeException` during a PyAutoGUI call; see the [PyAutoGUI fail-safe documentation](https://pyautogui.readthedocs.io/en/latest/index.html#fail-safes). That exception is not a complete shutdown routine for this script, so terminate the Python process if the overlay remains.

The transparent-color Tk window setup is intended for Windows. Display scaling and multiple-monitor arrangements can affect coordinate alignment. Large or very densely sampled contours can take time to finish because sampling and clicking run inside the mouse callback; the [pynput documentation](https://pynput.readthedocs.io/en/latest/mouse.html#the-mouse-listener-thread) describes the limitations of blocking work in these callbacks.

## Validation

The published `.py` file is an unchanged copy of the supplied `.txt` source and passes a Python syntax check. The desktop interaction has not been executed or validated against a live cryoSPARC session as part of this upload.
