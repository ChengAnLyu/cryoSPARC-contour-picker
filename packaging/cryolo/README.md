# SARU Picker 3.0.0 for Windows — offline inference

One application provides four detection profiles. The default remains SAM
`capsid`. Screen capture, manual contours, overlays, sampling, cancellation and
navigation remain available.

| Detection profile | Inference engine | Automatic contour processing |
| --- | --- | --- |
| `capsid` | Bundled SAM ViT-B | Existing capsid filtering |
| `capsid-learned` | Bundled SAM ViT-B | Supplied learned appearance profile |
| `generic` | Bundled SAM ViT-B | Generic SAM masks |
| `crYOLO` | Bundled ONNX Runtime CPU, evaluating the exported trained detector | Supplied learned appearance profile within detected regions |

## Install the delivered release

Extract the delivered installer and model files into the same folder. Keep
`cryolo-capsids.onnx`, `cryolo-config.json`, `capsid-profile.json` and the supplied
icons beside `SARU-Picker-Setup-3.0.0-win64.exe`, then run Setup. If the delivery
uses two ZIPs, extract both into the same destination and open their shared
`SARU-Picker-3.0.0` folder.

Setup copies the application and private sidecars. Installation and recognition
require no network access, runtime download, Python selection or repair step.
Launch **SARU Picker 3.0**, choose a profile in Advanced, and inspect detections
in semi-auto mode at normal screen zoom. Each detector keeps its own model path.

| Private sidecar beside Setup | Installed location under the app folder |
| --- | --- |
| `cryolo-capsids.onnx` | `_internal/models/cryolo-capsids.onnx` |
| `cryolo-config.json` | `_internal/models/cryolo-config.json` |
| `capsid-profile.json` | `_internal/models/capsid-profile.json` |
| `app-icon.ico` | `_internal/assets/app-icon.ico` |
| `app-icon.png` | `_internal/assets/app-icon.png` |

The ONNX model embeds its tensor layout, preprocessing, learned anchors and
source checkpoint hash. The unchanged training JSON accompanies it for
provenance; inference reads the embedded metadata. The public generic installer
omits private models and the learned appearance profile. The private delivery
supplies them. Public CI fixtures are never delivered as trained models.

For the portable ZIP, extract the complete `SARU_Picker_crYOLO` folder, place
sidecars at the paths above and run `SARU Picker crYOLO.exe`. The EXE and
`_internal` directory must remain together. This also works offline.

## Upgrade and inference behavior

Version remains **3.0.0**. The existing application GUID, internal executable
name and `%LOCALAPPDATA%\Programs\SARU Picker crYOLO` folder are retained so
Setup upgrades the previous 3.0 edition. The visible name is **SARU Picker 3.0**.
Setup removes only this edition's obsolete bootstrap helper files and shortcuts.
It leaves the old private Python environment, global Python, PATH and the
separate legacy SAM 2.9 installation untouched. Detector preferences remain at
`%LOCALAPPDATA%\SARU_Picker\detector-settings.json`.

The crYOLO profile now evaluates an export of the trained network with bundled
ONNX Runtime 1.23.2 on CPU. A short-lived worker uses the same bundled EXE and
supports cancellation; no external interpreter, TensorFlow or stock crYOLO
installation is required. The model export does not retrain or change the
learned weights. Release-specific numerical comparison and contour results
belong in the supplied model record.

Known default H5 model selections migrate to the verified matching ONNX export.
Custom H5 files require a matching exported model; they do not trigger a runtime
installation. The previous **Repair crYOLO runtime** control is removed.
Prediction errors retain diagnostics at
`%LOCALAPPDATA%\SARU_Picker\cryolo-prediction.log`.

ONNX Runtime and its third-party notices are bundled under their respective
licenses. Stock crYOLO software and its general pretrained weights are not
redistributed. Training dependencies retain their upstream licenses; this
runtime change makes no claim to broaden those permissions.

## Auto picking marker redraws

The 22 November auto-picking update disables old-reference screenshot comparisons
before, during and after auto click playback, and immediately before Down.
cryoSPARC can paint markers without interrupting the contour sequence.
Semi-auto alignment checks, recognition alignment, focus/window/geometry checks,
pause and cancellation remain. Navigation still requires a different stable
frame after Down. Existing models are unchanged and can be reused.
The native micrograph-cycle fixture exercises marker redraws through the real
controller and worker with synthetic click/key sinks.

## Build and verification

Run `packaging/cryolo/build.ps1` on Windows with Python 3.12 x64 and Git. Build
preparation downloads pinned dependencies and verifies the official SAM ViT-B
checkpoint's size and SHA-256. Network access is needed to prepare a build,
not to install or use the completed application. `-TestInstaller` requires a
disposable Windows host with permission to configure temporary firewall rules.

`packaging/cryolo/windows-version.txt` sets Windows version `3.0.0.0` and display
version `3.0.0`; Setup also declares 3.0.0. Older files directly under
`packaging/` are historical SAM build scaffolding. This release uses
`packaging/cryolo/`. Legacy runtime setup scripts remain source-only developer
utilities and are excluded from the distribution.

The [Windows workflow](../../.github/workflows/build-cryolo-windows.yml) requires:

- Regression tests and real Windows controls, including all four selector
  values, backend routing, independent model paths and DPI/work-area checks.
- Frozen and installed version metadata, native libraries, and real SAM CPU
  inference.
- A generated tiny ONNX Conv graph that produces a positive detection through
  the production worker and decoder, followed by learned contour refinement.
  Source, frozen and installed execution must all succeed.
- Separate positive/dense refinement, blank-image and cancellation checks.
- An installation with outbound firewall rules for the installer and app,
  followed by actual inference with sockets and external-runtime launches
  forbidden by a verification guard. No legacy environment or setup report
  may be created. Installation is limited to five minutes.
- Exact private-sidecar copying, appearance-profile inference and removal of
  obsolete owned runtime helpers during upgrade.

Synthetic models and images remain temporary and are removed. These checks
establish packaging operation, not biological accuracy. Private training images,
H5 checkpoints and exported models are never committed or uploaded by public CI.

The installer artifact is separate from diagnostics and optional portable ZIP
parts so each download stays within the connector size limit. Outputs include
`SARU-Picker-Setup-3.0.0-win64.exe`, `SARU-Picker-GUI-3.0.0-win64.zip`,
`SHA256SUMS.txt`, `download-manifest.json` and `build-manifest.json`.
`source-onnx-smoke.json`, `frozen-onnx-smoke.json`,
`installed-frozen-onnx-smoke.json` and `installed-offline-setup.json` record the
new runtime checks; existing SAM, GUI, refinement, sidecar and PE version reports
remain required. The release record must identify the completed workflow run,
commit and hashes. An earlier external-runtime build is not evidence for this
update.

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
