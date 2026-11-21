"""Real-data preparation contracts: label meaning, geometry and held-out groups."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from training.prepare_cryolo_data import box_file_text, export_dataset, reference_boxes, native_config


class CryoloAnnotationTests(unittest.TestCase):
    def test_bottom_left_box_round_trip_and_valid_clipping(self):
        regions = np.zeros((60, 90), np.uint16)
        regions[8:20, 23:41] = 4
        valid = np.ones(regions.shape, bool)
        valid[:, :25] = False
        boxes = reference_boxes(regions, valid, [4])
        self.assertEqual(boxes, [{"region_id": 4, "xywh": [25, 8, 16, 12]}])
        x, by, w, h = map(int, box_file_text(boxes, 60).split())
        self.assertEqual([x, 60 - by - h, w, h], boxes[0]["xywh"])
        self.assertEqual(box_file_text([], 60), "")

    def test_native_fixed_size_mode_uses_two_element_input_not_random_crop_scalar(self):
        config = native_config(train_images="train", train_annotations="labels",
            valid_images="valid", valid_annotations="valid_labels", anchors=[50, 60], input_size=512)
        self.assertEqual(config["model"]["input_size"], [512, 512])

    def test_missing_region_never_becomes_an_empty_negative(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            reference_boxes(np.zeros((12, 14), int), np.ones((12, 14), bool), [1])


class CryoloExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data"
        self.dataset.mkdir()
        self.cases = []
        for cid in range(1, 5):
            folder = self.dataset / str(cid)
            folder.mkdir()
            image = np.arange(80 * 96, dtype=np.uint8).reshape(80, 96)
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            Image.fromarray(rgb).save(folder / "clean.png")
            regions = np.zeros(image.shape, np.uint16)
            positive = cid in (1, 3)
            if positive:
                regions[10:40, 20:60] = 1
            Image.fromarray(regions).save(folder / "regions.png")
            Image.fromarray(np.full(image.shape, 255, np.uint8)).save(folder / "valid.png")
            self.cases.append({
                "case_id": cid, "evaluation_group": f"group_{cid}",
                "images": {"input": {"path": f"{cid}/clean.png",
                    "png_sha256": hashlib.sha256((folder / "clean.png").read_bytes()).hexdigest(),
                    "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}},
                "annotations": {"regions": [{"region_id": 1}] if positive else []},
                "valid_image_pixels": {"path": f"{cid}/valid.png", "positive_regions_path": f"{cid}/regions.png"},
                "training_supervision": {"confirmed_blank": cid == 2,
                    "confirmed_positive_region_ids": [1] if positive else [],
                    "positive_scope": "large_capsids", "unmarked_regions": "unknown",
                    "confirmation": "User confirmed marked large regions and explicit blank."},
            })
        self.manifest = self.dataset / "manifest.json"
        self.folds = self.root / "folds.json"
        self.write_manifest()
        self.folds.write_text(json.dumps({"folds": [
            {"fold": 0, "train_cases": [1, 2], "test_cases": [3, 4]},
            {"fold": 1, "train_cases": [3, 4], "test_cases": [1, 2]},
        ]}))

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"cases": self.cases, "source_decks": [
            {"uploaded_name": "examples.pptx", "sha256": "a" * 64, "case_ids": [1, 2, 3, 4]}]}))

    def test_explicit_standard_background_acceptance_is_required(self):
        with self.assertRaisesRegex(ValueError, "explicit acceptance"):
            export_dataset(self.manifest, self.folds, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_all_cases_full_fit_and_split_separation_are_preserved(self):
        out = self.root / "out"
        audit = export_dataset(self.manifest, self.folds, out, accept_incomplete_background=True)
        self.assertEqual(audit["case_count"], 4)
        self.assertEqual(audit["confirmed_blank_case_ids"], [2])
        self.assertEqual(audit["unknown_only_case_ids"], [4])
        self.assertEqual(audit["empty_annotation_count"], 2)
        self.assertEqual(audit["all_cases_used_for_full_fit_gradients"], [1, 2, 3, 4])
        self.assertEqual((out / "full/annotations/case_004.box").stat().st_size, 0)
        config = json.loads((out / "full/config.json").read_text())
        self.assertEqual(config["train"]["train_image_folder"], config["valid"]["valid_image_folder"])
        self.assertEqual(config["model"]["input_size"], [320, 320])
        self.assertEqual(audit["native_input_size"], [320, 320])
        self.assertTrue(all(case["native_model_resize"]["preserves_entire_frame"] for case in audit["cases"]))
        self.assertFalse(audit["cases"][0]["native_model_resize"]["preserves_aspect_ratio"])
        self.assertIn("in-sample", audit["full_fit_validation_scope"])
        for fold in audit["folds"]:
            self.assertFalse(set(fold["train_cases"]) & set(fold["validation_cases"]))
        self.assertEqual(sorted(p.stem for p in (out / "fold_00/train/images").iterdir()),
                         ["case_001", "case_002"])
        self.assertEqual(sorted(p.stem for p in (out / "fold_00/valid/images").iterdir()),
                         ["case_003", "case_004"])
        self.assertEqual(Image.open(out / "full/images/case_001.png").mode, "L")

    def test_tampered_input_rejected_before_export(self):
        (self.dataset / "1/clean.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            export_dataset(self.manifest, self.folds, self.root / "out", accept_incomplete_background=True)
        self.assertFalse((self.root / "out").exists())

    def test_unknown_case_is_not_relabeled_confirmed_blank(self):
        audit = export_dataset(self.manifest, self.folds, self.root / "out", accept_incomplete_background=True)
        unknown = next(case for case in audit["cases"] if case["case_id"] == 4)
        self.assertEqual(unknown["supervision_status"], "unknown_only_treated_as_background_by_user_choice")

    def test_overlapping_source_group_is_rejected(self):
        self.cases[2]["evaluation_group"] = self.cases[0]["evaluation_group"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "groups cross"):
            export_dataset(self.manifest, self.folds, self.root / "out", accept_incomplete_background=True)


if __name__ == "__main__":
    unittest.main()
