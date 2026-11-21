# Parallel crYOLO Windows package

The GUI uses the SARU 2.9.0 controls and an independent adapter to stock crYOLO
1.9.9. It installs into `%LOCALAPPDATA%\Programs\SARU Picker crYOLO`, with a
separate application GUID and `%LOCALAPPDATA%\SARU_Picker_crYOLO` settings/log
folder. It does not replace the SAM edition.

## Install the delivered private release

1. Extract the entire release ZIP and run
   `SARU-Picker-crYOLO-Setup-2.9.0-cryolo.1-win64.exe` with its sidecars beside it.
2. Double-click `Install-crYOLO-Runtime.cmd` in the installed folder, or use
   **Start → SARU Picker crYOLO → Install crYOLO runtime**. A working compatible
   crYOLO interpreter can instead be selected in Advanced.
3. Launch **SARU Picker crYOLO**, select the screen area and inspect detections
   in semi-auto mode at the normal zoom. The default learned contour profile
   and 3,500-screen-pixel minimum remain active.

The installer reads these exact external filenames. They are private release
inputs and are absent from public CI artifacts.

| Sidecar beside Setup | Installed location under the application folder |
| --- | --- |
| `cryolo-capsids.h5` | `_internal/models/cryolo-capsids.h5` |
| `cryolo-config.json` | `_internal/models/cryolo-config.json` |
| `capsid-profile.json` | `_internal/models/capsid-profile.json` |
| `app-icon.ico` | `_internal/assets/app-icon.ico` |
| `app-icon.png` | `_internal/assets/app-icon.png` |

Missing sidecars do not stop the generic installer, but detection requires the
weights and matching config. The default `capsid-learned` contour profile also
requires `capsid-profile.json`. Place all sidecars beside Setup before installing.
When selecting existing weights/config in Advanced, keep `capsid-profile.json`
beside the chosen H5 weights.

For the portable GUI ZIP, extract `SARU_Picker_crYOLO` and copy the sidecars into
the same relative locations in the table. Run its runtime-install command once
and then `SARU Picker crYOLO.exe`. Do not try to run the EXE from inside a ZIP.

## Separate runtime

`setup-runtime.ps1` uses an existing Python 3.7 x64 interpreter when supplied or
available through the Windows Python launcher. Otherwise it downloads the
signed official Python 3.7.9 installer, validates its Python Software Foundation
signature and installs privately without changing PATH or file associations.
The crYOLO venv is
`%LOCALAPPDATA%\SARU_Picker_crYOLO\cryolo-env\Scripts\python.exe`.

The setup obtains unmodified crYOLO 1.9.9, TensorFlow 1.15.4, Keras 2.3.1 and
compatible command-line dependencies from upstream. Optional crYOLO desktop
GUI dependencies are omitted because the SARU GUI uses only its command-line
interface. The application does not redistribute crYOLO software or general
pretrained weights. The upstream
[research license](https://cryolo-doc.readthedocs.io/en/stable/other/license.html)
applies. A successful runtime-install report covers imports and CLI access;
model inference is tested separately.

## Verified build 3

The [completed CI run](https://github.com/ChengAnLyu/cryoSPARC-contour-picker/actions/runs/44942803369)
used source commit `f7f7f31dae0e2f24d1b370f39cfeb42fe632e04c` on branch
`parallel/cryolo-2.9.0-20261121`.

| Build artifact | SHA-256 |
| --- | --- |
| `SARU-Picker-crYOLO-Setup-2.9.0-cryolo.1-win64.exe` | `7fc1f634883e163c6cd9cd31185f0e4b6ba9e62c18ad8c5753bc1dd1c8f90b47` |
| `SARU-Picker-crYOLO-GUI-2.9.0-cryolo.1-win64.zip` | `51d3ae3b53c4d775c0e8e9d54d63b8c7f644ed2e4c745f24656de7aa0b73b3ec` |

The GUI job runs the regression suite, native shared controls and the actual
crYOLO Advanced panel, then builds and installs the separate app and checks
its diagnostics. The native-runtime job installs stock crYOLO under Windows
Python 3.7.9/TF 1.15.4 and runs PhosaurusNet and its prediction CLI on synthetic
noise. That synthetic detector has an untrained detection head; it is removed
and is not supplied as the user's model. The runtime check produced a CBOX
output successfully.

Public artifacts contain no private weights, training images or learned
appearance profile. Installed GUI diagnostics therefore report absent private
weights by design. These checks do not establish trained-model accuracy or
training completion. Model hashes, completed epochs and evaluation scope
belong to the separate private training/evaluation record.

## Training provenance is separate from the GUI build

The reproducible full-fit entry point is `training/train_cryolo.py` (Python
3.11 or newer), which starts `training/train_cryolo_native.py` inside the crYOLO environment. The
independent driver uses the unchanged upstream architecture, loss, augmented
target generator and checkpoint writer. It explicitly averages every equal-size
monitoring batch's loss with metric state reset between batches; this avoids
the observed last-batch validation reporting in the legacy Keras path.

The full-fit recipe uses all 160 full images, 146 approximate large-region
boxes, fixed `[320, 320]` inputs, standard normalization, no additional low-pass
filter, batch size four, learning rate `1e-4`, two fine-tuning layers and
object/background loss weights of 100/1. These are recipe settings, not a
statement that any run finished. The run record identifies actual completion,
selected checkpoint, input/config/source hashes and package versions. Full-fit
monitoring is in-sample. Direct `cryolo_train.py` commands do not reproduce this
driver's monitoring or checkpoint selection.

Later training-driver or documentation changes do not change the already
verified build-3 executable bytes. Cite its build commit separately from the
training-driver commit and the selected private model hash.
