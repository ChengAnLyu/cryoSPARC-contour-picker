"""Offline regression tests: no desktop automation, model download, or GPU."""
import contextlib
import importlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recognition import (
    Cancelled, RecognitionConfig, RecognitionError, SamRecognizer,
    filter_candidates, make_candidate, polygon_mask, sample_hex_grid,
    union_masks, validate_rgb,
)


class BitmapSamplingTests(unittest.TestCase):
    def test_holes_and_disconnected_islands_are_preserved(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[5:70, 5:75] = True
        mask[20:60, 20:60] = False
        mask[80:95, 100:115] = True
        points = sample_hex_grid(mask, 5)
        self.assertGreater(len(points), 100)
        self.assertTrue(mask[points[:, 1], points[:, 0]].all())
        self.assertFalse(((points[:, 0] >= 20) & (points[:, 0] < 60)
                          & (points[:, 1] >= 20) & (points[:, 1] < 60)).any())
        self.assertTrue(((points[:, 0] >= 100) & (points[:, 1] >= 80)).any())

    def test_union_prevents_duplicate_picks(self):
        first = np.zeros((80, 80), dtype=bool)
        second = first.copy()
        first[5:60, 5:50] = True
        second[30:75, 25:75] = True
        union = union_masks([make_candidate(first), make_candidate(second)])
        points = sample_hex_grid(union, 4)
        expected = set(map(tuple, sample_hex_grid(first, 4))) | set(map(tuple, sample_hex_grid(second, 4)))
        self.assertEqual(set(map(tuple, points)), expected)
        self.assertEqual(len(points), len(expected))

    def test_lattice_phase_does_not_follow_candidate_bbox(self):
        all_true = np.ones((65, 77), dtype=bool)
        small = np.zeros_like(all_true)
        small[13:51, 19:70] = True
        full_points = sample_hex_grid(all_true, 7, origin=(2, -3))
        expected = full_points[small[full_points[:, 1], full_points[:, 0]]]
        np.testing.assert_array_equal(sample_hex_grid(small, 7, origin=(2, -3)), expected)

    def test_inset_respects_image_edges_and_internal_holes(self):
        mask = np.ones((60, 60), dtype=bool)
        mask[20:40, 20:40] = False
        points = sample_hex_grid(mask, 2, inset=3)
        self.assertTrue((points >= 3).all())
        self.assertTrue((points <= 56).all())
        self.assertFalse(((points[:, 0] >= 17) & (points[:, 0] <= 42)
                          & (points[:, 1] >= 20) & (points[:, 1] <= 39)).any())

    def test_point_limit_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(RecognitionError, "More than 20"):
            sample_hex_grid(np.ones((100, 100), dtype=bool), 3, max_points=20)

    def test_empty_output_has_xy_shape(self):
        points = sample_hex_grid(np.zeros((10, 10), dtype=bool), 3)
        self.assertEqual(points.shape, (0, 2))
        self.assertEqual(points.dtype, np.int32)

    def test_cancelled_sampling_does_not_return_partial_result(self):
        calls = [0]
        def stop():
            calls[0] += 1
            return calls[0] > 5
        with self.assertRaises(Cancelled):
            sample_hex_grid(np.ones((300, 300), dtype=bool), 3, cancel=stop)

    def test_invalid_spacing_cannot_create_infinite_loop(self):
        for spacing in (0, -3, 1.5, True):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                sample_hex_grid(np.ones((10, 10), dtype=bool), spacing)

    def test_xy_order_matches_non_square_manual_polygon(self):
        mask = polygon_mask([(21, 3), (38, 3), (38, 12), (21, 12)], (20, 60))
        self.assertTrue(mask[8, 30])
        self.assertFalse(mask[15, 10])
        points = sample_hex_grid(mask, 3)
        self.assertTrue(((points[:, 0] >= 21) & (points[:, 0] <= 38)).all())
        self.assertTrue(((points[:, 1] >= 3) & (points[:, 1] <= 12)).all())

    def test_union_rejects_coordinate_frame_mismatch(self):
        with self.assertRaises(ValueError):
            union_masks([np.zeros((20, 30), bool), np.zeros((30, 20), bool)])

    def test_degenerate_manual_line_is_not_a_pick_region(self):
        with self.assertRaisesRegex(ValueError, "straight line"):
            polygon_mask([(1, 1), (2, 2), (3, 3)], (20, 20))


class CandidateTests(unittest.TestCase):
    def test_candidate_owns_immutable_mask_and_stable_id(self):
        mask = np.zeros((20, 30), bool)
        mask[2:8, 7:16] = True
        first = make_candidate(mask, predicted_iou=.82)
        second = make_candidate(mask, predicted_iou=.91, source="automatic proposal")
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.bbox, (7, 2, 9, 6))
        self.assertEqual(first.area, 54)
        mask[:] = False
        self.assertEqual(first.mask.sum(), 54)
        with self.assertRaises(ValueError):
            first.mask[3, 8] = False

    def test_nested_mask_preference_is_explicit(self):
        outer = np.zeros((100, 100), bool)
        inner = outer.copy()
        outer[10:80, 10:80] = True
        inner[25:65, 25:65] = True
        a = make_candidate(outer, predicted_iou=.95)
        b = make_candidate(inner, predicted_iou=.85)
        config = RecognitionConfig("unused", max_area_ratio=.8)
        self.assertEqual([c.id for c in filter_candidates([a, b], config)], [b.id])
        quality_first = RecognitionConfig("unused", max_area_ratio=.8, prefer_smaller=False)
        self.assertEqual([c.id for c in filter_candidates([a, b], quality_first)], [a.id])

    def test_disjoint_targets_survive_and_frame_is_filtered(self):
        frame = np.ones((100, 100), bool)
        first = np.zeros_like(frame)
        second = first.copy()
        first[5:30, 5:30] = True
        second[55:90, 55:90] = True
        result = filter_candidates([make_candidate(frame), make_candidate(second), make_candidate(first)],
                                   RecognitionConfig("unused"))
        self.assertEqual(len(result), 2)
        self.assertEqual([c.area for c in result], [625, 1225])

    def test_optional_border_rejection_keeps_cropped_targets_by_default(self):
        mask = np.zeros((100, 100), bool)
        mask[10:40, :20] = True
        candidate = make_candidate(mask)
        self.assertEqual(len(filter_candidates([candidate], RecognitionConfig("unused"))), 1)
        self.assertEqual(filter_candidates([candidate], RecognitionConfig("unused", exclude_border=True)), [])


class FakePredictor:
    def __init__(self, model=None):
        self.set_calls = 0
        self.predict_calls = 0
        self.last_kwargs = None

    def set_image(self, rgb, image_format="RGB"):
        self.set_calls += 1
        self.shape = rgb.shape[:2]

    def predict(self, **kwargs):
        self.last_kwargs = kwargs
        self.predict_calls += 1
        masks = np.zeros((3, *self.shape), bool)
        masks[0, 2:8, 2:8] = True
        masks[1, 1:10, 1:10] = True
        masks[2, 4:6, 4:6] = True
        return masks, np.array([.9, .8, .7]), None


def fake_loaded_recognizer(profile="generic"):
    recognizer = SamRecognizer(RecognitionConfig("unused", automatic_profile=profile))
    recognizer._torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    recognizer._predictor = FakePredictor()
    return recognizer


class PromptTests(unittest.TestCase):
    def test_embedding_cached_across_prompt_edits_and_invalidated_by_pixels(self):
        recognizer = fake_loaded_recognizer()
        image = np.zeros((20, 30, 3), np.uint8)
        recognizer.prompt(image, points=[(4, 5)])
        result = recognizer.prompt(image.copy(), points=[(4, 5), (1, 1)], labels=[1, 0])
        self.assertEqual(recognizer._predictor.set_calls, 1)
        self.assertEqual(recognizer._predictor.predict_calls, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual([c.predicted_iou for c in result], [.9, .8, .7])
        image[0, 0, 0] = 1
        recognizer.prompt(image, box=(1, 1, 10, 10))
        self.assertEqual(recognizer._predictor.set_calls, 2)

    def test_negative_points_and_box_reach_predictor_in_xy_coordinates(self):
        recognizer = fake_loaded_recognizer()
        recognizer.prompt(np.zeros((20, 30, 3), np.uint8),
                          box=(2, 3, 18, 15), points=[(5, 8), (15, 3)], labels=[1, 0])
        kwargs = recognizer._predictor.last_kwargs
        np.testing.assert_array_equal(kwargs["point_coords"], [[5, 8], [15, 3]])
        np.testing.assert_array_equal(kwargs["point_labels"], [1, 0])
        np.testing.assert_array_equal(kwargs["box"], [2, 3, 18, 15])

    def test_cancel_after_encoder_never_publishes_embedding_or_result(self):
        recognizer = fake_loaded_recognizer()
        stopped = [False]
        original = recognizer._predictor.set_image
        def set_then_cancel(*args, **kwargs):
            original(*args, **kwargs)
            stopped[0] = True
        recognizer._predictor.set_image = set_then_cancel
        with self.assertRaises(Cancelled):
            recognizer.prompt(np.zeros((20, 30, 3), np.uint8), points=[(3, 3)], cancel=lambda: stopped[0])
        self.assertIsNone(recognizer._predictor_image_key)
        self.assertEqual(recognizer._predictor.predict_calls, 0)

    def test_prompt_coordinate_validation_precedes_model_loading(self):
        recognizer = SamRecognizer(RecognitionConfig("missing.pth"))
        image = np.zeros((20, 30, 3), np.uint8)
        bad = [dict(points=[(30, 5)]), dict(points=[(5, 20)]),
               dict(points=[(3, 4)], labels=[0]), dict(box=(8, 3, 2, 10)),
               dict(points=[(3, 4)], labels=[1, 0]), dict(labels=[1]),
               dict(points=5), dict(points=[])]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                recognizer.prompt(image, **kwargs)

    def test_model_output_shape_mismatch_fails_before_candidate_creation(self):
        recognizer = fake_loaded_recognizer()
        recognizer._predictor.predict = lambda **kwargs: (np.ones((1, 4, 4), bool), [.8], None)
        with self.assertRaisesRegex(RecognitionError, "dimensions"):
            recognizer.prompt(np.zeros((20, 30, 3), np.uint8), points=[(3, 3)])


class ModelLoadingTests(unittest.TestCase):
    def test_automatic_profile_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "automatic_profile"):
            RecognitionConfig("unused", automatic_profile="unknown")

    def test_module_import_does_not_import_gui_or_torch(self):
        code = ("import sys; import recognition; "
                "assert not {'torch', 'tkinter', 'pyautogui', 'pynput'} & set(sys.modules)")
        process = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_loader_uses_weights_only_cpu_mapping_and_matching_registry(self):
        calls = {}
        class Model:
            def load_state_dict(self, state, strict):
                calls["strict"] = strict
            def eval(self):
                calls["eval"] = True
            def to(self, device):
                calls["device"] = device
        class Generator:
            def __init__(self, **kwargs):
                calls["generator"] = kwargs
            def _process_batch(self, *args, **kwargs):
                return "processed"
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.set_num_threads = lambda n: calls.update(threads=n)
        torch.load = lambda path, **kwargs: calls.update(load=kwargs) or {}
        sam = types.ModuleType("segment_anything")
        sam.sam_model_registry = {"vit_b": lambda checkpoint: calls.update(registry_checkpoint=checkpoint) or Model()}
        sam.SamPredictor = FakePredictor
        sam.SamAutomaticMaskGenerator = Generator
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "sam_vit_b_01ec64.pth"
            checkpoint.touch()
            with patch.dict(sys.modules, {"torch": torch, "segment_anything": sam}):
                recognizer = SamRecognizer(RecognitionConfig(str(checkpoint), cpu_threads=2))
                recognizer._ensure_model(None)
        self.assertEqual(calls["load"], {"map_location": "cpu", "weights_only": True})
        self.assertIsNone(calls["registry_checkpoint"])
        self.assertTrue(calls["strict"])
        self.assertTrue(calls["eval"])
        self.assertEqual(calls["device"], "cpu")
        self.assertEqual(calls["generator"]["points_per_side"], 8)
        self.assertEqual(calls["generator"]["points_per_batch"], 8)
        self.assertEqual(calls["generator"]["min_mask_region_area"], 0)
        self.assertEqual(calls["generator"]["pred_iou_thresh"], .50)
        self.assertEqual(calls["generator"]["stability_score_thresh"], .75)
        recognizer._active_cancel = lambda: True
        with self.assertRaises(Cancelled):
            recognizer._generator._process_batch(None)

    def test_checkpoint_filename_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "sam_vit_h_example.pth"
            checkpoint.touch()
            recognizer = SamRecognizer(RecognitionConfig(str(checkpoint), model_type="vit_b"))
            with self.assertRaisesRegex(RecognitionError, "indicates vit_h"):
                recognizer._ensure_model(None)

    def test_automatic_identical_image_result_cached_without_second_inference(self):
        recognizer = fake_loaded_recognizer()
        calls = [0]
        def generate(rgb):
            calls[0] += 1
            mask = np.zeros(rgb.shape[:2], bool)
            mask[4:12, 4:12] = True
            return [{"segmentation": mask, "predicted_iou": .9, "stability_score": .95}]
        recognizer._generator = types.SimpleNamespace(generate=generate)
        image = np.zeros((30, 30, 3), np.uint8)
        first = recognizer.automatic(image)
        second = recognizer.automatic(image.copy())
        self.assertEqual(calls[0], 1)
        self.assertIsNot(first, second)
        self.assertEqual(first[0].id, second[0].id)
        with self.assertRaises(Cancelled):
            recognizer.automatic(image, cancel=lambda: True)

    def test_capsid_content_filter_sees_inner_and_outer_before_nested_suppression(self):
        recognizer = fake_loaded_recognizer("capsid")
        outer = np.zeros((60, 80), bool)
        outer[8:52, 10:70] = True
        inner = np.zeros_like(outer)
        inner[16:44, 24:56] = True
        raw = [{"segmentation": mask, "predicted_iou": score, "stability_score": .8}
               for mask, score in ((outer, .99), (inner, .86))]
        recognizer._generator = types.SimpleNamespace(generate=lambda image: raw)
        observations = []
        def choose_core(rgb, proposals, **kwargs):
            observations.append(proposals)
            kwargs["check_cancel"]()
            return [{"segmentation": inner, "predicted_iou": None,
                     "stability_score": None, "source": "capsid texture core (review)",
                     "capsid_evidence": {"proposal_predicted_iou": .86,
                                         "is_calibrated_probability": False}}]
        image = np.full((60, 80, 3), 125, np.uint8)
        with patch("recognition.filter_capsid_proposals", side_effect=choose_core):
            first = recognizer.automatic(image)
            second = recognizer.automatic(image.copy())
        self.assertEqual(len(observations), 1)
        self.assertIs(observations[0], raw)
        np.testing.assert_array_equal(first[0].mask, inner)
        self.assertIsNone(first[0].predicted_iou)
        self.assertIsNone(first[0].stability_score)
        self.assertEqual(second[0].id, first[0].id)
        self.assertEqual(recognizer.last_detection["sam_proposals"], 2)
        self.assertEqual(recognizer.last_detection["selected_contours"], 1)
        self.assertFalse(recognizer.last_detection["capsid_evidence"][first[0].id]["is_calibrated_probability"])

    def test_capsid_filter_cancellation_does_not_publish_or_cache_partial_results(self):
        recognizer = fake_loaded_recognizer("capsid")
        mask = np.zeros((30, 30), bool)
        mask[4:12, 4:12] = True
        recognizer._generator = types.SimpleNamespace(generate=lambda rgb: [{"segmentation": mask}])
        with patch("recognition.filter_capsid_proposals", side_effect=Cancelled("stop")):
            with self.assertRaises(Cancelled):
                recognizer.automatic(np.full((30, 30, 3), 125, np.uint8))
        self.assertIsNone(recognizer._automatic_result)
        self.assertIsNone(recognizer._automatic_image_key)
        self.assertIsNone(recognizer.last_detection)
        self.assertIsNone(recognizer._active_cancel)

    def test_bad_raw_mask_shape_is_rejected_before_capsid_processing(self):
        recognizer = fake_loaded_recognizer("capsid")
        recognizer._generator = types.SimpleNamespace(generate=lambda rgb: [{"segmentation": np.ones((5, 5), bool)}])
        with patch("recognition.filter_capsid_proposals") as process:
            with self.assertRaisesRegex(RecognitionError, "dimensions"):
                recognizer.automatic(np.zeros((30, 30, 3), np.uint8))
            process.assert_not_called()

    def test_rgb_validation_requires_explicit_uint8_rgb_shape(self):
        for image in (np.zeros((20, 20), np.uint8), np.zeros((20, 20, 4), np.uint8),
                      np.zeros((20, 20, 3), np.float32)):
            with self.assertRaises(ValueError):
                validate_rgb(image)
        original = np.zeros((20, 30, 3), np.uint8)[:, ::2]
        self.assertFalse(original.flags.c_contiguous)
        self.assertTrue(validate_rgb(original).flags.c_contiguous)




class LearnedAutomaticProfileTests(unittest.TestCase):
    def _recognizer(self, directory):
        import json
        from test_capsid_prior import fixture
        from test_capsid import synthetic_core
        profile = fixture()
        profile['layers'][0]['weight'][0][4] = 2.
        profile['layers'][0]['bias'][0] = -.5
        profile['layers'][1]['weight'][0][0] = 2.
        profile['layers'][2]['weight'][0][0] = 3.
        profile['layers'][2]['bias'][0] = 0.
        path = Path(directory) / 'capsid-profile.json'
        path.write_text(json.dumps(profile))
        rgb, core, proposals = synthetic_core()
        recognizer = SamRecognizer(RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile="capsid-learned"))
        recognizer._torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        class Predictor(FakePredictor):
            def predict(self, **kwargs):
                self.last_kwargs = kwargs
                self.predict_calls += 1
                return np.stack([proposals[0]['segmentation']]), np.array([.9]), None
        recognizer._predictor = Predictor()
        recognizer._generator = types.SimpleNamespace(generate=lambda rgb: self.fail('Learned path used the uniform grid'))
        return recognizer, rgb, core, path

    def test_actual_profile_scoring_guides_sam_and_uses_same_bitmap_for_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, core, path = self._recognizer(directory)
            result = recognizer.automatic(rgb)
            self.assertEqual(len(result), 1)
            self.assertGreater(np.count_nonzero(result[0].mask & core), .7 * core.sum())
            self.assertEqual(recognizer._predictor.set_calls, 1)
            self.assertGreater(recognizer._predictor.predict_calls, 0)
            info = recognizer.last_detection['appearance_model']
            self.assertEqual(info['status'], 'loaded')
            self.assertTrue(info['inference_executed'])
            self.assertFalse(info['fallback_used'])
            self.assertEqual(info['scored_pixels'], rgb.shape[0] * rgb.shape[1])
            self.assertEqual(Path(info['path']), path.resolve())
            self.assertTrue(result[0].source.startswith('learned'))
            first_calls = recognizer._predictor.predict_calls
            again = recognizer.automatic(rgb.copy())
            self.assertEqual(again[0].id, result[0].id)
            self.assertEqual(recognizer._predictor.predict_calls, first_calls)

    def test_cancelled_guided_predictor_cannot_publish_masks_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, _, _ = self._recognizer(directory)
            original = recognizer._predictor.predict
            stopped = False
            def predict(**kwargs):
                nonlocal stopped
                value = original(**kwargs)
                stopped = True
                return value
            recognizer._predictor.predict = predict
            with self.assertRaises(Cancelled):
                recognizer.automatic(rgb, cancel=lambda: stopped)
            self.assertIsNone(recognizer._automatic_result)
            self.assertIsNone(recognizer.last_detection)
            self.assertIsNone(recognizer._active_cancel)

    def test_corrupt_profile_fails_before_loading_sam(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, _, path = self._recognizer(directory)
            path.write_text('{not json')
            with patch.object(recognizer, '_ensure_model') as ensure:
                with self.assertRaisesRegex(RecognitionError, 'capsid appearance profile'):
                    recognizer.automatic(rgb)
                ensure.assert_not_called()

    def test_default_capsid_mode_keeps_existing_filter_despite_installed_profile(self):
        from test_capsid import synthetic_core
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, _, _ = self._recognizer(directory)
            recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'))
            _, _, proposals = synthetic_core()
            recognizer._generator = types.SimpleNamespace(generate=lambda rgb: proposals)
            with patch('recognition.score_image', side_effect=AssertionError('Default mode invoked learned profile')):
                self.assertEqual(len(recognizer.automatic(rgb)), 1)
            self.assertEqual(recognizer.last_detection['appearance_model']['status'], 'disabled')

    def test_explicit_learned_mode_missing_sidecar_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer = SamRecognizer(RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile='capsid-learned'))
            with patch.object(recognizer, '_ensure_model') as ensure:
                with self.assertRaisesRegex(RecognitionError, 'Select capsid for the earlier texture detector'):
                    recognizer.automatic(np.zeros((30, 30, 3), np.uint8))
                ensure.assert_not_called()

    def test_generic_mode_does_not_load_or_apply_an_installed_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, _, _ = self._recognizer(directory)
            recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile='generic')
            mask = np.zeros(rgb.shape[:2], bool)
            mask[15:60, 30:90] = True
            recognizer._generator = types.SimpleNamespace(generate=lambda rgb: [{'segmentation': mask}])
            with patch('recognition.score_image', side_effect=AssertionError('Generic mode invoked learned profile')):
                self.assertEqual(len(recognizer.automatic(rgb)), 1)
            self.assertEqual(recognizer.last_detection['appearance_model']['status'], 'disabled')
            self.assertFalse(recognizer._appearance_profile_loaded)

    def test_shared_carbon_guard_adjudicates_refined_contours_and_cache_in_both_modes(self):
        from test_capsid import synthetic_core
        for mode in ('capsid', 'capsid-learned'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                recognizer, rgb, _, _ = self._recognizer(directory)
                recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile=mode)
                _, _, parents = synthetic_core()
                recognizer._generator = types.SimpleNamespace(generate=lambda image: parents)
                first = np.zeros(rgb.shape[:2], bool)
                second = np.zeros_like(first)
                first[25:65, 40:90] = True
                second[70:110, 100:140] = True
                refined = [{'segmentation': first}, {'segmentation': second}]
                context = object()
                observed = []
                def adjudicate(actual_context, records, **kwargs):
                    self.assertIs(actual_context, context)
                    self.assertIs(records, refined)
                    kwargs['check_cancel']()
                    observed.append(records)
                    return [records[1]], {'input_count': 2, 'retained_count': 1,
                                          'rejected_count': 1, 'rejected_indices': [0]}
                refiner = ('recognition.filter_capsid_proposals' if mode == 'capsid'
                           else 'recognition.filter_capsid_with_profile')
                with patch(refiner, return_value=refined), \
                     patch('recognition.detect_carbon_edges', return_value=context) as detect, \
                     patch('recognition.guard_proposals', side_effect=adjudicate):
                    result = recognizer.automatic(rgb)
                    cached = recognizer.automatic(rgb.copy())
                self.assertEqual(detect.call_count, 1)
                self.assertEqual(len(observed), 1)
                self.assertEqual(len(result), 1)
                np.testing.assert_array_equal(result[0].mask, second)
                self.assertFalse(result[0].mask.flags.writeable)
                self.assertEqual(result[0].id, cached[0].id)
                self.assertTrue(recognizer.last_detection['carbon_guard']['enabled'])
                self.assertEqual(recognizer.last_detection['carbon_guard']['rejected_count'], 1)

    def test_carbon_cancellation_never_commits_partial_detection(self):
        from test_capsid import synthetic_core
        for mode in ('capsid', 'capsid-learned'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                recognizer, rgb, _, _ = self._recognizer(directory)
                recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile=mode)
                _, _, parents = synthetic_core()
                recognizer._generator = types.SimpleNamespace(generate=lambda image: parents)
                with patch('recognition.guard_proposals', side_effect=Cancelled('stop carbon pass')):
                    with self.assertRaisesRegex(Cancelled, 'stop carbon pass'):
                        recognizer.automatic(rgb)
                self.assertIsNone(recognizer._automatic_result)
                self.assertIsNone(recognizer._automatic_image_key)
                self.assertIsNone(recognizer.last_detection)
                self.assertIsNone(recognizer._active_cancel)

    def test_generic_and_explicit_prompts_bypass_automatic_carbon_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            recognizer, rgb, _, _ = self._recognizer(directory)
            recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile='generic')
            mask = np.zeros(rgb.shape[:2], bool)
            mask[15:60, 30:90] = True
            recognizer._generator = types.SimpleNamespace(generate=lambda image: [{'segmentation': mask}])
            with patch('recognition.detect_carbon_edges', side_effect=AssertionError('automatic guard invoked')):
                self.assertEqual(len(recognizer.automatic(rgb)), 1)
            self.assertFalse(recognizer.last_detection['carbon_guard']['enabled'])
        for mode in ('generic', 'capsid', 'capsid-learned'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                recognizer, rgb, _, _ = self._recognizer(directory)
                recognizer.config = RecognitionConfig(str(Path(directory) / 'sam.pth'), automatic_profile=mode)
                with patch('recognition.detect_carbon_edges', side_effect=AssertionError('automatic guard invoked')):
                    self.assertTrue(recognizer.prompt(rgb, box=(30, 20, 99, 77)))


if __name__ == "__main__":
    unittest.main()
