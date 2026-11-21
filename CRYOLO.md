# SARU Picker crYOLO — parallel 2.9.0 edition

This edition replaces SAM proposals with the real crYOLO 1.9.9 detector. It keeps
the 2.9.0 screen capture, manual contours, selection overlays, grid sampling,
click delay, cancellation and micrograph navigation. It has a separate launcher,
Windows installation and settings directory.

crYOLO predicts boxes, not segmentation masks. The existing capsid appearance
and texture processing traces contours inside its detections; carbon rejection
and the 3,500-screen-pixel minimum remain active. The supplied appearance
profile is the unchanged 160-example profile from SARU 2.9.0. No SAM checkpoint
or SAM inference is used by the crYOLO backend.

Box/point prompts select crYOLO detections. They cannot make crYOLO segment an
object it missed; manual drawing remains available. Object confidence is kept
separate from mask IoU and is not presented as a calibrated capsid probability.

## Run the parallel edition

Install the current GUI requirements in Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python main_cryolo.py --checkpoint /path/to/cryolo-capsids.h5 \
  --cryolo-config /path/to/cryolo-config.json \
  --cryolo-python /path/to/cryolo-environment/bin/python
```

Place `capsid-profile.json` beside the H5 weights. Configure the model paths,
runtime and confidence threshold under Advanced. Settings are remembered in
the separate crYOLO user directory. `main.py` continues to default to SAM;
`main_cryolo.py` defaults to crYOLO with the learned capsid contour profile.

The Windows release includes a separate GUI installer. Keep its private model,
config, profile and icon sidecars beside Setup when installing. Run the included
`setup-runtime.ps1` once to acquire stock crYOLO and its isolated Python 3.7 CPU
environment, or choose an existing compatible crYOLO interpreter. The modern
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

The exporter preserves all ten original grouped folds. Creating fold directories
does not mean that ten models were trained. The final `full` configuration uses
all 160 examples for training and those same images for monitoring; its metrics
are in-sample. Each config uses paths relative to its own directory. See the
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
