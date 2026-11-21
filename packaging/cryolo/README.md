# SARU Picker 3.0.0 for Windows

One application provides four detection profiles. The default remains SAM
`capsid`, preserving the previous picker's screen capture, manual contours,
overlays, sampling, cancellation and navigation.

| Detection profile | Backend | Automatic contour processing |
| --- | --- | --- |
| `capsid` | SAM ViT-B | Existing capsid filtering |
| `capsid-learned` | SAM ViT-B | Supplied learned appearance profile |
| `generic` | SAM ViT-B | Generic SAM masks |
| `crYOLO` | Stock crYOLO 1.9.9 | Supplied learned appearance profile within detected regions |

## Install the delivered release

Extract the entire release ZIP and run `SARU-Picker-Setup-3.0.0-win64.exe`.
Leave the supplied model/config/profile and icon sidecars beside Setup.
Setup copies the sidecars, installs and verifies the private crYOLO runtime
when needed, then enables launch. First installation requires internet access
and can take several minutes. There is no separate Python selection or runtime
installation step. A failure stops launch and explains how to retry Setup.

SAM ViT-B and its CPU runtime are bundled. Launch **SARU Picker 3.0**, choose
one of the four profiles in Advanced and inspect detections in semi-auto mode
at normal screen zoom. SAM and crYOLO retain separate model paths.

| Private sidecar beside Setup | Installed location under the app folder |
| --- | --- |
| `cryolo-capsids.h5` | `_internal/models/cryolo-capsids.h5` |
| `cryolo-config.json` | `_internal/models/cryolo-config.json` |
| `capsid-profile.json` | `_internal/models/capsid-profile.json` |
| `app-icon.ico` | `_internal/assets/app-icon.ico` |
| `app-icon.png` | `_internal/assets/app-icon.png` |

The public generic installer omits the private H5 and appearance profile.
The delivered private release supplies these sidecars. `capsid-learned` and
`crYOLO` require the matching appearance profile; crYOLO additionally requires
its H5 and JSON config. Public CI fixtures are never delivered as trained models.

For the portable ZIP, extract the complete `SARU_Picker_crYOLO` folder, place
sidecars at the table's relative paths and run `SARU Picker crYOLO.exe`.
The crYOLO profile prepares its runtime automatically when first needed. The
EXE and `_internal` directory must stay together.

## Upgrade identity and runtime

The existing crYOLO application GUID, internal executable name and
`%LOCALAPPDATA%\Programs\SARU Picker crYOLO` folder are retained so this
installer upgrades the previous 3.0 edition. Display names are neutral
**SARU Picker 3.0**. Only that edition's obsolete shortcuts are replaced;
the separate legacy SAM 2.9 installation remains independent.
Detector preferences live at `%LOCALAPPDATA%\SARU_Picker\detector-settings.json`.

The official Python Software Foundation Python 3.7.9 package is bundled,
verified by size and SHA-256, with its license in `Python37-LICENSE.txt`.
Setup extracts that private interpreter without invoking a global Python
installer, then obtains unmodified crYOLO 1.9.9, TensorFlow 1.15.4, Keras 2.3.1
and compatible command-line dependencies. Source installs can download the
same verified Python package automatically when it is not bundled.
It leaves global Python, PATH and file associations unchanged. The runtime is
`%LOCALAPPDATA%\SARU_Picker_crYOLO\cryolo-env\Scripts\python.exe`.
A healthy existing managed runtime can be reused without network access.
The install report is `runtime-install-report.json` in that parent folder.
The application can repair a missing managed runtime when detection starts.

crYOLO software and general pretrained weights are obtained from upstream,
not redistributed in this bundle. Its
[research license](https://cryolo-doc.readthedocs.io/en/stable/other/license.html)
applies. Optional upstream desktop GUI dependencies are omitted because SARU
uses the command-line interface.

## Build and verification

Run `packaging/cryolo/build.ps1` on Windows with Python 3.12 x64 and Git. The
build downloads the official SAM ViT-B checkpoint and verifies its pinned size
and SHA-256 before including it. `packaging/cryolo/windows-version.txt` supplies
fixed Windows version `3.0.0.0` and display version `3.0.0`; the installer also
declares 3.0.0. The older files under `packaging/` are historical SAM build
scaffolding; this release uses `packaging/cryolo/`.

The [Windows workflow](../../.github/workflows/build-cryolo-windows.yml) checks:

- Regression tests and actual Windows shared controls, including all four
  selector values, backend routing and independent checkpoint paths.
- Frozen and installed version metadata, default SAM profile, native libraries
  and genuine SAM CPU inference.
- Automatic private-runtime setup during installation, exact private-sidecar
  copying and numerical inference using a synthetic appearance-model fixture.
- Actual frozen EXE → isolated crYOLO process inference, catching Windows DLL
  search-path conflicts between the bundled torch runtime and TensorFlow.
- A separate clean private Python bootstrap with no supplied Python interpreter,
  followed by stock PhosaurusNet and CLI inference on synthetic data.

Generated synthetic H5 weights remain in temporary directories and are removed.
These checks establish runtime operation, not capsid accuracy. No private
training examples or trained weights are committed or uploaded by public CI.

Artifacts are `SARU-Picker-Setup-3.0.0-win64.exe`,
`SARU-Picker-GUI-3.0.0-win64.zip`, `SHA256SUMS.txt`, `build-manifest.json`
and source/frozen/installed verification reports. The delivered release record
must identify the actual completed workflow run, commit and hashes; a previous
crYOLO-only 3.0 binary is not evidence for this four-profile build.

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
