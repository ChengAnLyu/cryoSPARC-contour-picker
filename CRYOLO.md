# SARU Picker 3.0.0 — fourth detection profile

The corrected version 3.0.0 runs the trained crYOLO 1.9.9 network as the fourth detection profile
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

Install the GUI requirements in Python 3.12, including ONNX Runtime CPU:

```sh
python -m pip install -r requirements.txt
python main.py --automatic-profile cryolo --checkpoint /path/to/cryolo-capsids.onnx
```

Place `capsid-profile.json` beside the ONNX model. Configure model paths and
confidence under Advanced. The model embeds its fixed input size, normalization,
learned anchors and source H5 hash. No separate runtime or JSON configuration
is required for ONNX inference. Unified settings remember separate SAM and
crYOLO paths. The default remains SAM `capsid`; `main_cryolo.py` is a compatibility
shortcut selecting crYOLO.

The Windows release has an installer ZIP and a trained-model ZIP. Download the
updated versions of both, extract them into the same folder, and keep the ONNX,
profile and icon sidecars beside Setup. Installation and prediction work offline.
The application bundles ONNX Runtime CPU and starts inference in a cancellable
child process. It does not install Python 3.7, TensorFlow or crYOLO, and has no
runtime Repair step. Existing 3.0.0 installation identity and settings are retained.

An old saved standard `cryolo-capsids.h5` path can resolve to its sibling ONNX
export only after the worker verifies its source identity. Custom H5 files are
not silently substituted. An expert can run an H5 model with an explicitly
supplied `--cryolo-python`; that path uses the separately installed stock runtime.

The crYOLO authors provide their software under their own
[research license](https://cryolo-doc.readthedocs.io/en/stable/other/license.html).
This repository contains an independent adapter and decoder, not a redistributed
copy of the crYOLO program or its general pretrained checkpoint. Training and
export tools call separately obtained, unchanged upstream APIs.

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

The ONNX worker uses full-frame Pillow grayscale/bilinear resizing to 320×320
and the trained STANDARD normalization. Its independent single-anchor YOLO
decoder uses the learned anchor metadata from the checkpoint, confidence
filtering and NMS, then passes detected regions to the unchanged contour pipeline.
Legacy BOX/CBOX inputs remain supported with bottom-left to top-left conversion.
A valid empty detection list means no objects; malformed output is an error.
Cancellation and timeouts terminate the owned prediction process.

```sh
python main_cryolo.py --diagnostics
python main_cryolo.py --smoke-test --checkpoint /path/to/cryolo-capsids.onnx
python main_cryolo.py --segment-image clean-image.png \
  --checkpoint /path/to/cryolo-capsids.onnx --output review-output
python -m pytest -q
```

Diagnostics reports metadata only. The smoke test performs real inference.
Public Windows fixtures validate offline packaging and the complete prediction
pipeline without private training images or weights. Private format-equivalence
checks are separate from the original biological accuracy evaluation.

To export a compatible checkpoint, run `training/export_cryolo_onnx.py` in an
isolated Python 3.7 environment containing stock crYOLO 1.9.9, TensorFlow 1.15.4,
Keras 2.3.1, tf2onnx 1.14.0 and onnx 1.12.0:

```sh
python training/export_cryolo_onnx.py --weights cryolo-capsids.h5 \
  --config cryolo-config.json --output cryolo-capsids.onnx
```

The export embeds its inference contract and writes a provenance report. This
is a build-time conversion, not training or an installation step. Numerical,
decoder and contour equivalence must be checked before distributing a new export.

## Version 3.0.0 integration

This release integrates the completed 160-example, 20-epoch checkpoint selected
at epoch 18. Its SHA-256 is
`a883be0bf39f0a93b0e2c36441c062fac5ab3672c044a7cf0703dc01853d3761`.
The fixed 320×320 network is exported from those H5 weights; the all-160
appearance profile is carried forward unchanged. No additional model training is claimed.

The Windows installer is `SARU-Picker-Setup-3.0.0-win64.exe`.
Application diagnostics, the EXE's Windows version metadata, the installer and
release manifest identify version 3.0.0. The four-profile selector keeps SAM and
crYOLO configuration separate. The CPU inference runtime is bundled. Private model sidecars are installed automatically from
beside Setup. See [the packaging notes](packaging/cryolo/README.md) for the
build procedure. The private release record identifies the actual successful
3.0.0 build commit, artifact hashes and trained-model integration checks.
