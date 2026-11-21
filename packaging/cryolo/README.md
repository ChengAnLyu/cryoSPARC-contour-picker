# SARU Picker crYOLO 3.0.0 for Windows

crYOLO is the default detector in SARU Picker 3.0.0. The GUI retains the screen
capture, manual contours, overlays, sampling, cancellation and navigation from
the existing picker. It uses an independent adapter to stock crYOLO 1.9.9.

The installer keeps the earlier crYOLO application GUID, executable name,
`%LOCALAPPDATA%\Programs\SARU Picker crYOLO` installation folder and
`%LOCALAPPDATA%\SARU_Picker_crYOLO` settings/runtime folder. Installing 3.0.0
therefore upgrades that edition without replacing the separate SAM 2.9.0 app.
Existing custom model paths in Advanced remain user choices.

## Install the delivered private release

1. Extract the entire release ZIP and run
   `SARU-Picker-crYOLO-Setup-3.0.0-win64.exe` with its sidecars beside it.
2. If a compatible crYOLO runtime is not already installed, double-click
   `Install-crYOLO-Runtime.cmd` in the installed folder or use
   **Start → SARU Picker crYOLO → Install crYOLO runtime**. An existing compatible
   interpreter can instead be selected in Advanced.
3. Launch **SARU Picker crYOLO**. The interface identifies version 3.0.0 and
   defaults to crYOLO with the learned contour profile. Select the screen area
   and inspect detections in semi-auto mode at the normal zoom. The default
   minimum contour area remains 3,500 screen pixels.

Setup copies these exact external filenames automatically. The private model
and appearance profile are supplied with the user's release, never committed
to the public repository or included in public CI artifacts.

| Sidecar beside Setup | Installed location under the application folder |
| --- | --- |
| `cryolo-capsids.h5` | `_internal/models/cryolo-capsids.h5` |
| `cryolo-config.json` | `_internal/models/cryolo-config.json` |
| `capsid-profile.json` | `_internal/models/capsid-profile.json` |
| `app-icon.ico` | `_internal/assets/app-icon.ico` |
| `app-icon.png` | `_internal/assets/app-icon.png` |

Missing sidecars do not stop the generic installer, but detection requires the
weights and matching config. The default `capsid-learned` contour profile also
requires `capsid-profile.json`. Keep all sidecars beside Setup before installing.
When selecting existing weights/config in Advanced, keep `capsid-profile.json`
beside the chosen H5 weights.

For the portable GUI ZIP, extract `SARU_Picker_crYOLO` and copy the sidecars into
the same relative locations in the table. Run the runtime-install command if
needed and then `SARU Picker crYOLO.exe`. Do not run the EXE inside a ZIP.

## Separate runtime

`setup-runtime.ps1` uses an existing Python 3.7 x64 interpreter when supplied or
available through the Windows Python launcher. Otherwise it downloads the
signed official Python 3.7.9 installer, validates its Python Software Foundation
signature and installs privately without changing PATH or file associations.
The crYOLO venv is
`%LOCALAPPDATA%\SARU_Picker_crYOLO\cryolo-env\Scripts\python.exe`.

The setup obtains unmodified crYOLO 1.9.9, TensorFlow 1.15.4, Keras 2.3.1 and
compatible command-line dependencies from upstream. Optional crYOLO desktop
GUI dependencies are omitted because the SARU GUI uses its command-line
interface. The application does not redistribute crYOLO software or general
pretrained weights. The upstream
[research license](https://cryolo-doc.readthedocs.io/en/stable/other/license.html)
applies. A successful runtime-install report covers imports and CLI access;
model inference is tested separately.

## Build and verification

Run `packaging/cryolo/build.ps1` on Windows with Python 3.12 x64. The dedicated
`packaging/cryolo/windows-version.txt` supplies both fixed Windows version
`3.0.0.0` and display file/product version `3.0.0`. The installer declares
version 3.0.0. The older files under `packaging/` are historical SAM-build
scaffolding; this release uses the `packaging/cryolo/` build path.

The [Windows workflow](../../.github/workflows/build-cryolo-windows.yml) runs
regression tests and native shared controls, then opens the actual crYOLO
Advanced panel. It checks that source defaults, frozen diagnostics, installed
diagnostics and executable metadata all identify version 3.0.0 with crYOLO as
the default. It also verifies automatic sidecar copying with synthetic model,
config and profile fixtures in a temporary staging directory. Those fixtures
are never delivered as models or included in build artifacts.

The separate native-runtime job installs stock crYOLO on Windows and exercises
PhosaurusNet and the prediction CLI using synthetic noise and an untrained
detection head. The synthetic detector is deleted after the check. This checks
runtime operation, not capsid recognition accuracy.

Generated artifacts use these names:

- `SARU-Picker-crYOLO-Setup-3.0.0-win64.exe`
- `SARU-Picker-crYOLO-GUI-3.0.0-win64.zip`
- `build-manifest.json`, `SHA256SUMS.txt`, source/frozen/installed reports,
  executable-version reports and `installed-sidecar-copy.json`

The delivered release record must identify its actual completed CI run, commit
and artifact hashes. A prior 2.9.0 build record is not evidence that a 3.0.0
binary passed these checks. Public builds contain no private weights or training
images. These checks do not establish trained-model accuracy or new training
completion.

## Model training provenance

The reproducible full-fit entry point is `training/train_cryolo.py` (Python
3.11 or newer), which starts `training/train_cryolo_native.py` inside the crYOLO
environment. The independent driver uses the unchanged upstream architecture,
loss, augmented target generator and checkpoint writer. It explicitly averages
every equal-size monitoring batch's loss with metric state reset between
batches, avoiding the observed last-batch validation reporting in the legacy
Keras path.

The full-fit recipe uses all 160 full images, 146 approximate large-region
boxes, fixed `[320, 320]` inputs, standard normalization, no additional low-pass
filter, batch size four, learning rate `1e-4`, two fine-tuning layers and
object/background loss weights of 100/1. These are recipe settings, not a
statement that a new run finished. The application version change to 3.0.0
alone does not retrain the detector. Refer to the supplied model's training
record for its actual completed epochs, selected checkpoint, input/config/source
hashes and package versions. Full-fit monitoring is in-sample. Direct
`cryolo_train.py` commands do not reproduce the driver's monitoring or checkpoint
selection.
