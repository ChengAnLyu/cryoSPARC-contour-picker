# SARU Picker 3.0.0 — fourth detection profile

The corrected version 3.0.0 adds real crYOLO 1.9.9 as the fourth detection profile
alongside the existing SAM-based capsid, capsid-learned and generic choices. It keeps
the 2.9.0 screen capture, manual contours, selection overlays, grid sampling,
click delay, cancellation and micrograph navigation. All four detection profiles
share the same application and unified settings.

crYOLO predicts boxes, not segmentation masks. The existing capsid appearance
and texture processing traces contours inside its detections; carbon rejection
and the 3,500-screen-pixel minimum remain active. The supplied appearance
profile is the unchanged 160-example profile from SARU 2.9.0. No SAM checkpoint
or SAM inference is used by the crYOLO backend.

Box/point prompts select crYOLO detections. They cannot make crYOLO segment an
object it missed; manual drawing remains available. Object confidence is kept
separate from mask IoU and is not presented as a calibrated capsid probability.

## Run version 3.0.0

Install the current GUI requirements in Python 3.12:

```sh
python -m pip install -r requirements.txt
python main.py --automatic-profile cryolo --checkpoint /path/to/cryolo-capsids.h5 \
  --cryolo-config /path/to/cryolo-config.json \
  --cryolo-python /path/to/cryolo-environment/bin/python
```

Place `capsid-profile.json` beside the H5 weights. Configure model paths and
confidence under Advanced. The managed Windows runtime is found automatically;
`--cryolo-python` remains an optional source-only override. Unified detector
settings remember separate SAM and crYOLO paths. `main.py` defaults to the original SAM capsid profile; select `cryolo` in
Advanced → Detection profile to use the trained detector. `main_cryolo.py`
remains a compatibility shortcut. All four choices are available in one GUI.

The Windows release includes one combined installer, delivered in an installer
ZIP and a separate trained-model ZIP to fit the download size limit. Extract both
archives into the same folder. Keep its private model,
config, profile and icon sidecars beside Setup when installing. Setup includes the verified private Python 3.7 package and automatically
acquires and verifies stock crYOLO in its isolated CPU environment. A healthy existing runtime
is reused. No Python executable picker is required. First-time setup requires
internet access. The modern
GUI and legacy model environment run in separate processes. Setup does not
replace the SAM installation or alter the system Python PATH.

The crYOLO authors provide the software under their own
[research license](https://cryolo-doc.readthedocs.io/en/stable/other/license.html).
This repository contains an independent adapter, not a redistributed copy of
crYOLO or its general pretrained model. Obtain those from the official sources.

## Training labels and reproducibility

The user explicitly chose standard full-image crYOLO training on 21 November
2026 despite incomplete labels. The export retains all 160 clean images and
146 bounding boxes derived from confirmed large picking extents. Its 75 empty
annotation files comprise 43 confirmed blanks and 32 other images with no
confirmed large box. Those 32 images are **not confirmed blank ground truth**;
standard crYOLO nevertheless trains their unlabelled cells as background. The
same background treatment applies to unmarked areas of positive images.

This intentionally differs from the earlier positive/unlabelled appearance
training. Annotated picking footprints are approximate, and their boxes do not
establish exact capsid boundaries or exhaustive particle counts. Annotated
screenshots are never used as model inputs.

```sh
python training/prepare_cryolo_data.py \
  --dataset /path/to/dataset/manifest.json \
  --folds /path/to/folds.json --output /path/to/new-cryolo-export \
  --accept-incomplete-background --input-size 320 --epochs 20
```

To run the full fit, use the Python 3.11-or-newer wrapper, supplying the separately
obtained official initialization files and the Python executable from the
crYOLO environment:

```sh
python training/train_cryolo.py \
  --export-directory /path/to/new-cryolo-export \
  --cryolo-python /path/to/cryolo-environment/bin/python \
  --pretrained-weights /path/to/gmodel_phosnet_202005_N63_c17.h5 \
  --backend-weights /path/to/full_yolo_backend.h5 \
  --output /path/to/new-training-run \
  --input-size 320 --epochs 20 --cpu-threads 4 \
  --fine-tune-layers 2 --object-scale 100 --no-object-scale 1
```

The wrapper runs `training/train_cryolo_native.py` in the legacy environment.
That independent driver calls the unchanged crYOLO `YOLO`, `custom_loss`,
`PatchwiseBatchGenerator` and `ExtendedModelCheckpoint` APIs. It neither
patches nor vendors the crYOLO package. Images are resized as complete frames
to an explicit `[320, 320]` input with standard normalization, one patch and no
added low-pass filter. The default full-fit recipe uses batch size four,
learning rate `1e-4`, two fine-tuning layers and stock augmentation.

Checkpoint selection uses the arithmetic mean of every equal-size monitoring
batch's `test_on_batch(..., reset_metrics=True)` loss. This avoids the observed
legacy Keras validation-reporting problem in which the reported value reflected
the last monitoring batch. Calling `cryolo_train.py` directly does not reproduce
this driver's monitoring or checkpoint-selection procedure. The driver records
its source hash, package versions, input hashes, actual completed epochs and
selected checkpoint; a requested epoch count is not evidence of completion.

The exporter preserves all ten original grouped folds. Creating fold directories
does not mean that ten models were trained. The final `full` configuration uses
all 160 examples for training and those same images for monitoring; its metrics
are in-sample. The training wrapper audits the stock generator to verify that
all 146 exported positive boxes survive target assignment in full-frame mode. Each config uses paths relative to its own directory. See the
delivered training record for actual commands, initialization hashes, epochs
and completed evaluation scope.

`training/evaluate_cryolo.py` reads real CBOX predictions and runs the production
contour refinement. It measures coverage of six-pixel-eroded reference cores
and unwanted contours on explicitly confirmed blanks. It does not infer
precision or exhaustive recall from incomplete annotations. Reusing the
all-160 appearance profile also prevents claiming that the complete contour
pipeline is held out merely because a detector fold was held out.

## Inference coordinates and checks

Clean captures are converted to grayscale PNG without resizing by the adapter.
crYOLO performs the configured resizing/normalization. BOX/CBOX coordinates use
a bottom-left origin; the adapter converts them back to the screen's top-left
origin. Estimated object dimensions are read separately from extraction-box
dimensions. Missing or malformed output is an error; an existing empty CBOX is
a valid no-detection result. Cancellation/timeout terminates the external process.

```sh
python main_cryolo.py --diagnostics
python main_cryolo.py --smoke-test --cryolo-python /path/to/cryolo/python
python main_cryolo.py --segment-image clean-image.png \
  --cryolo-python /path/to/cryolo/python --output review-output
python -m pytest -q
```

Diagnostics reports metadata only. The smoke test requires installed weights,
configuration and a working crYOLO runtime, and performs real inference. Public
Windows build checks cover packaging and interface behavior; they do not measure
biological detection accuracy on private examples.


## Version 3.0.0 integration

This release integrates the completed 160-example, 20-epoch checkpoint selected
at epoch 18. Its SHA-256 is
`a883be0bf39f0a93b0e2c36441c062fac5ab3672c044a7cf0703dc01853d3761`.
The exact H5 weights, fixed 320×320 inference configuration and all-160
appearance profile are carried forward. No additional model training is claimed.

The Windows installer is `SARU-Picker-Setup-3.0.0-win64.exe`.
Application diagnostics, the EXE's Windows version metadata, the installer and
release manifest identify version 3.0.0. The four-profile selector keeps SAM and
crYOLO configuration separate. The managed crYOLO runtime path is retained. Private model sidecars are installed automatically from
beside Setup. See [the packaging notes](packaging/cryolo/README.md) for the
build procedure. The private release record identifies the actual successful
3.0.0 build commit, artifact hashes and trained-model integration checks.
