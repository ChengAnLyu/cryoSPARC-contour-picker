"""Export approved standard full-image crYOLO supervision from SARU examples.

The user explicitly accepted crYOLO's standard background treatment on
2026-11-21: all unlabelled cells (including unknown-only images) contribute as
background. This differs from the earlier positive/unlabelled appearance loss.
Only the clean screenshot is a model input. Marked companions are never used.

Run from the repository with --dataset manifest.json --folds folds.json
--output NEW_DIRECTORY --accept-incomplete-background. Run each native crYOLO
training command with its config's containing directory as the working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from training.train_capsid_profile import (
    source_deck_records, validate_folds, validate_training_supervision,
)


def sha_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Dataset asset is missing or outside the dataset: {relative}")
    return path


def reference_boxes(regions, valid, region_ids):
    """Return weak bounding boxes in native top-left (x, y, width, height)."""
    regions, valid = np.asarray(regions), np.asarray(valid, dtype=bool)
    if regions.ndim != 2 or regions.shape != valid.shape:
        raise ValueError("Region and valid masks must be equally sized 2D arrays.")
    if len(set(region_ids)) != len(region_ids):
        raise ValueError("Positive region IDs must be unique.")
    result = []
    for rid in region_ids:
        yy, xx = np.nonzero((regions == rid) & valid)
        if not len(xx):
            raise ValueError(f"Confirmed region {rid} is absent from valid pixels.")
        x, y = int(xx.min()), int(yy.min())
        result.append({"region_id": int(rid), "xywh": [x, y, int(xx.max()) + 1 - x,
                                                       int(yy.max()) + 1 - y]})
    return result


def box_file_text(boxes, image_height):
    """crYOLO/EMAN BOX origin is bottom-left; PNG arrays remain top-left."""
    rows = []
    for record in boxes:
        x, y, width, height = record["xywh"]
        if min(x, y) < 0 or min(width, height) < 1 or y + height > image_height:
            raise ValueError("Box is invalid or outside the image.")
        rows.append(f"{x} {image_height - y - height} {width} {height}\n")
    # An intentionally zero-byte file is required for every image with no boxes.
    return "".join(rows)


def native_config(*, train_images, train_annotations, valid_images, valid_annotations,
                  anchors, input_size=320, epochs=20, batch_size=4, train_times=1,
                  pretrained_weights="", max_boxes=32):
    if type(input_size) is not int or input_size < 64 or input_size % 32:
        raise ValueError("input_size must be a positive multiple of 32, at least 64.")
    for key, value in (("epochs", epochs), ("batch_size", batch_size), ("train_times", train_times)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    return {
        "model": {"architecture": "PhosaurusNet", "input_size": [input_size, input_size],
                  "anchors": list(anchors), "max_box_per_image": max_boxes,
                  "norm": "STANDARD"},
        "train": {"train_image_folder": train_images, "train_annot_folder": train_annotations,
                  "train_times": train_times, "pretrained_weights": str(pretrained_weights),
                  "batch_size": batch_size, "learning_rate": 1e-4, "nb_epoch": epochs,
                  "warmup_epochs": 0, "object_scale": 5.0, "no_object_scale": 1.0,
                  "coord_scale": 1.0, "class_scale": 1.0,
                  "saved_weights_name": "capsid-cryolo.h5", "debug": False},
        "valid": {"valid_image_folder": valid_images, "valid_annot_folder": valid_annotations,
                  "valid_times": 1},
        "other": {"log_path": "logs"},
    }


def _copy_or_link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def export_dataset(dataset, folds_path, output, *, accept_incomplete_background=False,
                   input_size=320, epochs=20, batch_size=4, train_times=1,
                   pretrained_weights="", folds_to_export=None):
    """Write a new deterministic export and exact source/split provenance."""
    if not accept_incomplete_background:
        raise ValueError("Standard crYOLO makes unlabelled cells background; explicit acceptance is required.")
    dataset, folds_path, output = Path(dataset).resolve(), Path(folds_path).resolve(), Path(output).resolve()
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    fold_record = json.loads(folds_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    case_map = {case["case_id"]: case for case in cases}
    if len(case_map) != len(cases) or any(type(key) is not int for key in case_map):
        raise ValueError("Case IDs must be unique integers.")
    provenance = source_deck_records(manifest)
    folds = fold_record["folds"]
    validate_folds(case_map, folds)
    if fold_record.get("manifest_sha256") not in (None, sha_file(dataset)):
        raise ValueError("Fold manifest hash does not match the supplied dataset.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; refusing to overwrite an export.")
    root = dataset.parent
    # Validate every input and confirmation before creating output files.
    loaded = {}
    for cid, case in sorted(case_map.items()):
        supervision = validate_training_supervision(case)
        image_path = _asset(root, case["images"]["input"]["path"])
        input_info = case["images"]["input"]
        if input_info.get("png_sha256") and sha_file(image_path) != input_info["png_sha256"]:
            raise ValueError(f"Clean input hash mismatch for case {cid}.")
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        expected = input_info.get("decoded_rgb_sha256")
        if expected and hashlib.sha256(rgb.tobytes()).hexdigest() != expected:
            raise ValueError(f"Decoded clean input hash mismatch for case {cid}.")
        region_path = _asset(root, case["valid_image_pixels"]["positive_regions_path"])
        valid_path = _asset(root, case["valid_image_pixels"]["path"])
        regions = np.asarray(Image.open(region_path))
        valid = np.asarray(Image.open(valid_path)) > 0
        if regions.shape != rgb.shape[:2] or valid.shape != rgb.shape[:2]:
            raise ValueError(f"Image/mask shape mismatch for case {cid}.")
        boxes = reference_boxes(regions, valid, supervision["confirmed_positive_region_ids"])
        if supervision["confirmed_blank"] and np.any(regions):
            raise ValueError(f"Confirmed blank case {cid} contains annotation pixels.")
        loaded[cid] = (image_path, region_path, valid_path, rgb, boxes, supervision)
    selected_folds = set(f["fold"] for f in folds) if folds_to_export is None else set(folds_to_export)
    if not selected_folds <= {f["fold"] for f in folds}:
        raise ValueError("Requested fold does not exist.")
    all_boxes = [box for row in loaded.values() for box in row[4]]
    if not all_boxes:
        raise ValueError("At least one confirmed positive reference is required.")
    anchors = [int(round(float(np.median([box["xywh"][i] for box in all_boxes])))) for i in (2, 3)]
    max_boxes = max(32, max(len(row[4]) for row in loaded.values()))
    base_options = dict(input_size=input_size, epochs=epochs, batch_size=batch_size,
                        train_times=train_times, pretrained_weights=pretrained_weights, max_boxes=max_boxes)
    # Check options even when exporting no held-out folds.
    native_config(train_images="images", train_annotations="annotations", valid_images="images",
                  valid_annotations="annotations", anchors=anchors, **base_options)
    (output / "full" / "images").mkdir(parents=True, exist_ok=True)
    (output / "full" / "annotations").mkdir(parents=True, exist_ok=True)
    audit_cases = []
    for cid, row in loaded.items():
        image_path, region_path, valid_path, rgb, boxes, supervision = row
        stem = f"case_{cid:03d}"
        image_out = output / "full" / "images" / (stem + ".png")
        annot_out = output / "full" / "annotations" / (stem + ".box")
        Image.fromarray(rgb).convert("L").save(image_out)
        annot_out.write_text(box_file_text(boxes, rgb.shape[0]), encoding="ascii")
        status = ("confirmed_blank" if supervision["confirmed_blank"] else
                  "confirmed_large_positive" if boxes else "unknown_only_treated_as_background_by_user_choice")
        audit_cases.append({
            "case_id": cid, "evaluation_group": case_map[cid]["evaluation_group"],
            "supervision_status": status, "source_input": case_map[cid]["images"]["input"]["path"],
            "source_input_sha256": sha_file(image_path), "region_mask_sha256": sha_file(region_path),
            "valid_mask_sha256": sha_file(valid_path), "export_image": f"full/images/{stem}.png",
            "export_image_sha256": sha_file(image_out), "export_annotation_sha256": sha_file(annot_out),
            "source_shape_hw": list(rgb.shape[:2]), "boxes_top_left_xywh": boxes,
            "transform": "Export uses PIL RGB to L only; native full-frame fixed-size resizing occurs during crYOLO training/prediction",
            "native_model_resize": {"width": input_size, "height": input_size,
                "mode": "FIXED_SIZE", "preserves_entire_frame": True,
                "preserves_aspect_ratio": rgb.shape[0] == rgb.shape[1]},
            "annotation_transform": "BOX_y = image_height - top_left_y - box_height",
            "marked_regions_are_exact_boundaries": False,
        })
    _write_json(output / "full" / "config.json", native_config(
        train_images="images", train_annotations="annotations", valid_images="images",
        valid_annotations="annotations", anchors=anchors, **base_options))
    split_audits = []
    for fold in folds:
        if fold["fold"] not in selected_folds:
            continue
        split_dir = output / f"fold_{fold['fold']:02d}"
        for split, key in (("train", "train_cases"), ("valid", "test_cases")):
            for cid in fold[key]:
                for kind, suffix in (("images", ".png"), ("annotations", ".box")):
                    name = f"case_{cid:03d}{suffix}"
                    _copy_or_link(output / "full" / kind / name, split_dir / split / kind / name)
        # Fit anchor sizes only on the training side of each held-out split.
        train_boxes = [box for cid in fold["train_cases"] for box in loaded[cid][4]]
        if not train_boxes:
            raise ValueError(f"Fold {fold['fold']} has no training positives.")
        fold_anchors = [int(round(float(np.median([box["xywh"][i] for box in train_boxes]))))
                        for i in (2, 3)]
        _write_json(split_dir / "config.json", native_config(
            train_images="train/images", train_annotations="train/annotations", valid_images="valid/images",
            valid_annotations="valid/annotations", anchors=fold_anchors, **base_options))
        split_audits.append({"fold": fold["fold"], "train_cases": fold["train_cases"],
                             "validation_cases": fold["test_cases"], "anchors_from_training_only": fold_anchors,
                             "validation_scope": "source-group held out; weak and incomplete box labels"})
    confirmed_blank_ids = [cid for cid, row in loaded.items() if row[5]["confirmed_blank"]]
    unknown_ids = [cid for cid, row in loaded.items() if not row[5]["confirmed_blank"] and not row[4]]
    audit = {
        "format": "SARU-crYOLO-standard-full-image-export-v1", "status": "data_export_only_not_trained",
        "dataset_sha256": sha_file(dataset), "folds_sha256": sha_file(folds_path), "source_decks": provenance,
        "user_choice": {"date": "2026-11-21", "instruction": "Use standard full-image crYOLO training despite incomplete labels",
                        "accepted_by_caller": True},
        "case_count": len(loaded), "confirmed_large_box_count": len(all_boxes),
        "confirmed_large_case_count": sum(bool(row[4]) for row in loaded.values()),
        "confirmed_blank_case_ids": confirmed_blank_ids, "unknown_only_case_ids": unknown_ids,
        "empty_annotation_count": sum(not row[4] for row in loaded.values()),
        "all_cases_used_for_full_fit_gradients": sorted(loaded),
        "background_semantics": f"Stock crYOLO treats all unlabelled cells as background. The {len(unknown_ids)} unknown-only cases are not newly biologically confirmed blanks.",
        "full_fit_validation_scope": "Same images and labels as training; in-sample monitoring only, never held-out performance.",
        "fold_evaluation_scope": "Grouped development splits; incomplete labels cannot establish exhaustive recall or exact boundaries.",
        "no_filter": True, "native_input_size": [input_size, input_size],
        "native_resize_policy": "Explicit two-element input_size selects crYOLO FIXED_SIZE, resizing the entire image and boxes. Non-square source images are warped to square; no mixed-aspect random crop mode.",
        "full_fit_anchors": anchors, "cases": audit_cases, "folds": split_audits,
        "notes": ["Training and prediction use grayscale clean screenshots. No marked companion image is a model input.",
                  "No synthetic composites were made; the user selected standard full-image crYOLO training.",
                  "Explicit [width,height] model input size selects whole-frame FIXED_SIZE resizing in stock crYOLO, avoiding mixed-aspect random crops.",
                  "Non-square frames are resized to a square model input with matching box transforms; aspect ratio is not preserved.",
                  "Weak boxes enclose valid marked extents; they need not enclose an entire biological capsid.",
                  "Annotation masks do not train a segmentation head. Existing SARU boundary refinement remains separate.",
                  "Run crYOLO with the directory containing config.json as the current working directory."],
    }
    _write_json(output / "export-audit.json", audit)
    _write_json(output / "original-folds.json", fold_record)
    (output / "README.txt").write_text(
        "SARU crYOLO standard full-image data export\n\n"
        f"{len(loaded)} images; {len(all_boxes)} confirmed weak large-region boxes; "
        f"{len(confirmed_blank_ids)} confirmed blanks; {len(unknown_ids)} unknown-only images.\n"
        "The user accepted stock crYOLO treating all unlabelled cells as background.\n"
        "This changes the earlier positive/unlabelled learning semantics.\n"
        "No trained checkpoint is created by this exporter.\n"
        f"Native input_size [{input_size},{input_size}] selects whole-frame FIXED_SIZE resizing.\n"
        "Non-square source frames are warped to square with matching box transforms.\n"
        "Exported PNG dimensions remain unchanged; resizing occurs inside crYOLO.\n\n"
        "Run training from full/ or fold_NN/, e.g. cryolo_train.py -c config.json -w 0.\n"
        "full/config.json uses all images for gradients and the same images for\n"
        "in-sample loss monitoring. Its validation loss is NOT held-out evidence.\n"
        "fold_NN/config.json preserves the original source-group split.\n"
        "Paths in configs are relative to the config's directory.\n"
        "Weights, epochs, training logs and evaluation must be reported separately.\n",
        encoding="utf-8")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-incomplete-background", action="store_true")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-times", type=int, default=1)
    parser.add_argument("--pretrained-weights", default="")
    parser.add_argument("--fold", type=int, action="append", dest="fold_ids",
                        help="Export only these fold IDs (repeatable); default exports all folds.")
    args = parser.parse_args(argv)
    audit = export_dataset(args.dataset, args.folds, args.output,
        accept_incomplete_background=args.accept_incomplete_background, input_size=args.input_size,
        epochs=args.epochs, batch_size=args.batch_size, train_times=args.train_times,
        pretrained_weights=args.pretrained_weights, folds_to_export=args.fold_ids)
    print(json.dumps({key: audit[key] for key in ("status", "case_count", "confirmed_large_box_count",
                                                "confirmed_large_case_count", "empty_annotation_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
