"""Schema, cancellation, resource, and structural invariants for learned profiles."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capsid_prior import (FEATURE_NAMES, PROFILE_FORMAT, appearance_features,
                          load_profile, resolve_profile_path, score_features, score_image)
from capsid import capsid_guided_prompts, filter_capsid_with_profile, _profile_pool_offer
from test_capsid import synthetic_core


def fixture():
    return {'format': PROFILE_FORMAT, 'feature_names': list(FEATURE_NAMES),
            'mean': [0.] * 23, 'scale': [1.] * 23,
            'layers': [{'weight': np.zeros(shape).tolist(),
                        'bias': ([1.] if shape[0] == 1 else [0.] * shape[0])}
                       for shape in ((32, 23), (16, 32), (1, 16))]}


class CapsidProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'capsid-profile.json'
        self.path.write_text(json.dumps(fixture()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_numeric_profile_inference_and_immutable_parameters(self):
        profile = load_profile(self.path)
        image = np.full((32, 48, 3), 128, np.uint8)
        scores = score_image(image, profile)
        np.testing.assert_allclose(scores, 1 / (1 + np.exp(-1)), rtol=1e-6)
        self.assertEqual(scores.shape, (32, 48))
        self.assertFalse(profile.mean.flags.writeable)
        self.assertFalse(profile.layers[0][0].flags.writeable)
        self.assertFalse(profile.describe()['score_is_calibrated_probability'])
        self.assertEqual(len(profile.sha256), 64)

    def test_coloured_screen_annotations_and_white_ui_are_not_features(self):
        profile = load_profile(self.path)
        image = np.full((64, 80, 3), 128, np.uint8)
        image[:, :10] = 255
        image[20:35, 30:45] = [0, 255, 0]
        scores = score_image(image, profile)
        self.assertFalse(scores[:, :7].any())
        self.assertFalse(scores[20:35, 30:45].any())
        self.assertGreater(scores[50, 60], .7)

    def test_wrong_schema_nonfinite_and_wrong_numeric_types_are_rejected(self):
        corruptions = [lambda p: p.update(format='other'),
                       lambda p: p.update(feature_names=list(reversed(FEATURE_NAMES))),
                       lambda p: p['mean'].__setitem__(0, float('nan')),
                       lambda p: p['scale'].__setitem__(0, 0),
                       lambda p: p['mean'].__setitem__(0, '0'),
                       lambda p: p['layers'][0].update(weight=[[1.]])]
        for corrupt in corruptions:
            with self.subTest(corruption=corrupt):
                record = fixture()
                corrupt(record)
                self.path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_profile(self.path)

    def test_oversized_profile_rejected_before_json_decoding(self):
        self.path.write_bytes(b' ' * 262145)
        with self.assertRaisesRegex(ValueError, 'size limit'):
            load_profile(self.path)

    def test_sidecar_resolution_is_exact_and_explicit_missing_is_an_error(self):
        checkpoint = Path(self.tmp.name) / 'sam.pth'
        self.assertEqual(resolve_profile_path(checkpoint), self.path)
        missing = Path(self.tmp.name) / 'absent.json'
        with self.assertRaisesRegex(ValueError, 'not found'):
            resolve_profile_path(checkpoint, missing)
        self.path.unlink()
        self.assertIsNone(resolve_profile_path(checkpoint))

    def test_four_megapixel_guard_precedes_large_feature_allocation(self):
        large = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (2049, 2049, 3))
        with patch('capsid_prior.np.zeros', side_effect=AssertionError('unexpected allocation')):
            with self.assertRaisesRegex(ValueError, 'smaller micrograph area'):
                appearance_features(large)

    def test_cancel_during_feature_pass_and_between_neural_chunks(self):
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise InterruptedError('stopped')
        with self.assertRaisesRegex(InterruptedError, 'stopped'):
            appearance_features(np.full((32, 48, 3), 128, np.uint8), cancel)
        calls = 0
        with self.assertRaisesRegex(InterruptedError, 'stopped'):
            score_features(np.zeros((1, 32769, 23), np.float32), load_profile(self.path), cancel)


class GuidedCapsidTests(unittest.TestCase):
    def test_guided_box_and_click_point_come_from_dense_core_not_whole_envelope(self):
        rgb, core, proposals = synthetic_core()
        scores = np.where(core, .95, 0).astype(np.float32)
        prompts = capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 1)
        x, y = prompts[0]['point']
        self.assertTrue(core[y, x])
        self.assertLess(prompts[0]['box'][2] - prompts[0]['box'][0], 130)
        outputs = filter_capsid_with_profile(rgb, proposals, scores)
        self.assertEqual(len(outputs), 1)
        selected = outputs[0]['segmentation']
        self.assertGreater(np.count_nonzero(selected & core), .75 * core.sum())
        self.assertFalse(np.any(selected & ~proposals[0]['segmentation']))
        self.assertIsNone(outputs[0]['predicted_iou'])
        self.assertFalse(outputs[0]['capsid_evidence']['is_calibrated_probability'])

    def test_long_diagonal_strip_and_thin_ring_do_not_become_capsid_boxes(self):
        yy, xx = np.mgrid[:240, :420]
        core = ((xx - 65) / 35) ** 2 + ((yy - 125) / 45) ** 2 <= 1
        strip = (np.abs(yy - 1.9 * (xx - 145)) < 7) & (xx > 145) & (xx < 250)
        radius = np.hypot(xx - 340, yy - 125)
        ring = (radius > 40) & (radius < 46)
        scores = np.where(core | strip | ring, .99, 0).astype(np.float32)
        prompts = capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 1)
        x, y = prompts[0]['point']
        self.assertTrue(core[y, x])

    def test_substantial_capsid_truncated_at_image_edge_remains_eligible(self):
        yy, xx = np.mgrid[:220, :220]
        core = ((xx - 4) / 50) ** 2 + ((yy - 110) / 65) ** 2 < 1
        scores = np.where(core, .95, 0).astype(np.float32)
        prompts = capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['box'][0], 0)

    def test_refinement_preserves_original_holes_and_coloured_exclusions(self):
        rgb, core, proposals = synthetic_core()
        proposals[0]['segmentation'][88:103, 75:90] = False
        rgb[104:112, 92:102] = [0, 255, 0]
        scores = np.where(core, .95, 0).astype(np.float32)
        outputs = filter_capsid_with_profile(rgb, proposals[:1], scores)
        self.assertTrue(outputs)
        for item in outputs:
            self.assertFalse(item['segmentation'][88:103, 75:90].any())
            self.assertFalse(item['segmentation'][104:112, 92:102].any())

    def test_candidate_pool_stays_bounded_and_retains_highest_stable_ranks(self):
        import weakref
        pool, references = [], []
        expected = []
        for serial in range(300):
            mask = np.full((8, 8), serial, np.int32)
            references.append(weakref.ref(mask))
            record = {'segmentation': mask, 'capsid_score': float(serial % 97)}
            expected.append((float(serial % 97), -serial))
            _profile_pool_offer(pool, record, serial)
            self.assertLessEqual(len(pool), 64)
            self.assertLessEqual(sum(ref() is not None for ref in references), 65)
        self.assertEqual(sorted((entry[0], entry[1]) for entry in pool), sorted(expected, reverse=True)[:64][::-1])

    def test_wrong_coordinate_shape_and_zero_budget(self):
        rgb, core, proposals = synthetic_core()
        with self.assertRaisesRegex(ValueError, 'matching image dimensions'):
            filter_capsid_with_profile(rgb, proposals, np.zeros((20, 30)))
        scores = core.astype(np.float32)
        self.assertEqual(capsid_guided_prompts(scores, max_prompts=0), [])
        self.assertEqual(filter_capsid_with_profile(rgb, proposals, scores, max_candidates=0), [])

    def test_cancel_during_guided_proposal_work(self):
        _, core, _ = synthetic_core()
        def stop():
            raise InterruptedError('stopped')
        with self.assertRaisesRegex(InterruptedError, 'stopped'):
            capsid_guided_prompts(core.astype(np.float32), check_cancel=stop)


if __name__ == '__main__':
    unittest.main()
