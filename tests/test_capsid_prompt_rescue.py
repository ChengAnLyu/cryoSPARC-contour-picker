"""Regression contracts for high-score prompts and final-size-aware selection."""
from pathlib import Path
from dataclasses import replace
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capsid
from test_recognition import fake_loaded_recognizer


class HighScorePromptTests(unittest.TestCase):
    def nested(self):
        scores = np.zeros((300, 300), np.float32)
        scores[60:240, 60:240] = .65
        scores[115:185, 115:185] = .98
        return scores

    def test_tighter_core_inside_large_box_gets_a_supplemental_prompt(self):
        scores = self.nested()
        prompts = capsid.capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]['seed_threshold'], .5)
        self.assertEqual(prompts[0]['box'], [38, 38, 262, 262])
        self.assertEqual(prompts[0]['point'], [149, 149])
        self.assertEqual(prompts[1]['seed_threshold'], .9)
        self.assertLess(prompts[1]['support_area'], prompts[0]['support_area'] / 3)
        x, y = prompts[1]['point']
        self.assertGreaterEqual(float(scores[y, x]), .9)

    def test_oversized_merged_support_does_not_hide_a_strong_compact_core(self):
        scores = np.full((300, 300), .65, np.float32)
        scores[110:190, 110:190] = .98
        prompts = capsid.capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['seed_threshold'], .9)
        self.assertLess(prompts[0]['box'][2] - prompts[0]['box'][0], 110)

    def test_identical_high_and_low_support_is_not_prompted_twice(self):
        scores = np.zeros((240, 240), np.float32)
        scores[80:160, 80:160] = .98
        prompts = capsid.capsid_guided_prompts(scores)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['seed_threshold'], .5)

    def test_high_score_strip_and_thin_ring_still_fail_shape_checks(self):
        yy, xx = np.mgrid[:300, :500]
        strip = (np.abs(yy - 1.7 * (xx - 40)) < 6) & (xx > 40) & (xx < 190)
        radius = np.hypot(xx - 370, yy - 150)
        ring = (radius > 45) & (radius < 50)
        scores = np.where(strip | ring, .99, 0).astype(np.float32)
        self.assertEqual(capsid.capsid_guided_prompts(scores), [])

    def test_original_prompt_has_priority_when_budget_is_full(self):
        prompts = capsid.capsid_guided_prompts(self.nested(), max_prompts=1)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]['seed_threshold'], .5)
        self.assertEqual(prompts[0]['box'], [38, 38, 262, 262])
        self.assertEqual(capsid.capsid_guided_prompts(self.nested(), max_prompts=0), [])

    def test_many_candidates_remain_bounded_and_do_not_modify_scores(self):
        scores = np.zeros((600, 600), np.float32)
        yy, xx = np.mgrid[:600, :600]
        for y in range(50, 600, 100):
            for x in range(50, 600, 100):
                scores[(xx - x) ** 2 + (yy - y) ** 2 < 25 ** 2] = .98
        before = scores.copy()
        prompts = capsid.capsid_guided_prompts(scores, max_prompts=100)
        self.assertEqual(len(prompts), 24)
        self.assertTrue(all(p['seed_threshold'] == .5 for p in prompts))
        np.testing.assert_array_equal(scores, before)

    def test_cancellation_is_honored_at_supplemental_work(self):
        calls = 0
        def count():
            nonlocal calls
            calls += 1
        capsid._capsid_guided_prompts_at_threshold(self.nested(), check_cancel=count)
        primary_calls = calls
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            if calls > primary_calls:
                raise InterruptedError('cancelled supplemental work')
        with self.assertRaisesRegex(InterruptedError, 'supplemental'):
            capsid.capsid_guided_prompts(self.nested(), check_cancel=stop)

    def test_extra_prompt_does_not_relax_final_shape_acceptance(self):
        support = np.zeros((240, 240), bool)
        support[40:100, 40:100] = True
        support[90:110, 70:200] = True
        scores = np.where(support, .65, 0).astype(np.float32)
        scores[40:100, 40:100] = .98
        self.assertTrue(capsid.capsid_guided_prompts(scores))
        rng = np.random.default_rng(15)
        gray = np.clip(128 + rng.normal(0, 3, support.shape)
                       + rng.normal(0, 26, support.shape) * support, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        self.assertEqual(capsid.filter_capsid_with_profile(
            rgb, [{'segmentation': support}], scores, min_capsid_area_px=3500), [])


class PostPolygonEligibilityTests(unittest.TestCase):
    def fitted(self, before, after, minimum=3500):
        record = {'segmentation': before, '_allowed': np.ones_like(before),
                  'area': int(before.sum()), 'bbox': capsid._capsid_bbox(before),
                  'capsid_score': 1., 'capsid_evidence': {},
                  'predicted_iou': None, 'stability_score': None, 'source': 'test'}
        with patch.object(capsid, 'polygonal_capsid_mask', return_value=(after, {'fixture': True})):
            return list(capsid._profile_fitted_eligible_records(record, minimum))

    def test_pre_polygon_3449_post_polygon_3650_is_eligible(self):
        before = np.zeros((180, 180), bool)
        before[60:119, 60:119] = True
        before[60, 60:92] = False
        after = np.zeros_like(before)
        after[55:105, 55:128] = True
        self.assertEqual(int(before.sum()), 3449)
        self.assertEqual(int(after.sum()), 3650)
        actual = self.fitted(before, after)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['area'], 3650)
        np.testing.assert_array_equal(actual[0]['segmentation'], after)

    def test_large_pre_polygon_mask_cannot_pass_with_small_final_foreground(self):
        before = np.zeros((180, 180), bool)
        before[40:140, 40:140] = True
        after = np.zeros_like(before)
        after[60:100, 60:110] = True
        self.assertEqual(self.fitted(before, after), [])

    def test_disconnected_subthreshold_pieces_are_not_pooled(self):
        mask = np.zeros((180, 180), bool)
        mask[10:50, 10:60] = True
        mask[100:140, 100:150] = True
        self.assertEqual(int(mask.sum()), 4000)
        self.assertEqual(self.fitted(mask, mask), [])

    def test_only_eligible_component_participates_in_overlap_selection(self):
        mask = np.zeros((180, 180), bool)
        mask[10:70, 10:70] = True
        mask[110:150, 110:160] = True
        result = self.fitted(mask, mask)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['area'], 3600)
        self.assertFalse(result[0]['segmentation'][110:150, 110:160].any())

    def test_minimum_counts_actual_foreground_excluding_holes(self):
        mask = np.zeros((180, 180), bool)
        mask[30:80, 30:100] = True
        self.assertEqual(self.fitted(mask, mask)[0]['area'], 3500)
        mask[50, 50] = False
        self.assertEqual(self.fitted(mask, mask), [])

    def test_diagonal_contact_is_not_a_four_connected_component(self):
        mask = np.zeros((160, 160), bool)
        mask[10:55, 10:55] = True
        mask[55:100, 55:100] = True
        self.assertEqual(int(mask.sum()), 4050)
        self.assertEqual(self.fitted(mask, mask), [])

    def test_cancel_before_polygon_fit_does_not_publish_partial_work(self):
        mask = np.zeros((80, 80), bool)
        mask[10:70, 10:70] = True
        def stop():
            raise InterruptedError('stopped before fitting')
        with patch.object(capsid, 'polygonal_capsid_mask', side_effect=AssertionError('unexpected fitting')):
            with self.assertRaisesRegex(InterruptedError, 'stopped'):
                list(capsid._profile_fitted_eligible_records({'segmentation': mask}, 3500, stop))

    def test_invalid_minimum_is_rejected(self):
        rgb = np.full((80, 80, 3), 120, np.uint8)
        scores = np.zeros((80, 80), np.float32)
        for value in (True, -1, 3500., float('nan'), '3500'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                capsid.filter_capsid_with_profile(rgb, [], scores, min_capsid_area_px=value)

    def test_rejected_small_winner_cannot_suppress_usable_larger_alternative(self):
        shape = (180, 240)
        outer = np.zeros(shape, bool)
        outer[60:110, 50:170] = True
        inner = np.zeros(shape, bool)
        inner[62:108, 85:131] = True
        scores = np.where(outer, .6, 0).astype(np.float32)
        scores[inner] = .99
        texture = np.where(outer, 5., 1.).astype(np.float32)
        rgb = np.full((*shape, 3), 120, np.uint8)
        raw = [{'segmentation': inner}, {'segmentation': outer}]
        with patch.object(capsid, '_capsid_texture', return_value=(texture, np.ones(shape, bool))):
            default = capsid.filter_capsid_with_profile(rgb, raw, scores, max_candidates=1)
            zero = capsid.filter_capsid_with_profile(rgb, raw, scores, max_candidates=1, min_capsid_area_px=0)
            eligible = capsid.filter_capsid_with_profile(rgb, raw, scores, max_candidates=1, min_capsid_area_px=3500)
        self.assertEqual(len(default), 1)
        self.assertEqual(len(zero), 1)
        self.assertLess(default[0]['area'], 3500)
        np.testing.assert_array_equal(default[0]['segmentation'], zero[0]['segmentation'])
        self.assertEqual(default[0]['capsid_evidence'], zero[0]['capsid_evidence'])
        self.assertEqual(len(eligible), 1)
        self.assertGreaterEqual(eligible[0]['area'], 3500)
        self.assertGreater(eligible[0]['area'], default[0]['area'] * 2)


class LearnedAreaHandoffTests(unittest.TestCase):
    def test_configured_floor_reaches_learned_refiner_only_and_final_guard_remains(self):
        for mode in ('capsid-learned', 'capsid'):
            with self.subTest(mode=mode):
                recognizer = fake_loaded_recognizer(mode)
                recognizer.config = replace(recognizer.config, min_capsid_area_px=3700)
                mask = np.zeros((160, 240), bool)
                mask[30:100, 50:120] = True
                records = [{'segmentation': mask}]
                recognizer._generator = types.SimpleNamespace(generate=lambda rgb: records)
                profile = types.SimpleNamespace(describe=lambda: {'status': 'loaded'}) if mode == 'capsid-learned' else None
                with patch.object(recognizer, '_get_appearance_profile', return_value=profile), \
                     patch('recognition.score_image', return_value=np.ones(mask.shape, np.float32)) as scoring, \
                     patch('recognition.capsid_guided_prompts', return_value=[{'box': [40, 20, 130, 110], 'point': [80, 60]}]), \
                     patch('recognition.filter_capsid_with_profile', return_value=records) as learned, \
                     patch('recognition.filter_capsid_proposals', return_value=records) as heuristic, \
                     patch('recognition.detect_carbon_edges', return_value=object()), \
                     patch('recognition.guard_proposals', return_value=(records, {'enabled': True})):
                    result = recognizer.automatic(np.full((160, 240, 3), 125, np.uint8))
                self.assertEqual([candidate.area for candidate in result], [4900])
                self.assertEqual(recognizer.last_detection['capsid_area_filter']['minimum_area_px'], 3700)
                if mode == 'capsid-learned':
                    self.assertEqual(learned.call_args.kwargs['min_capsid_area_px'], 3700)
                    heuristic.assert_not_called()
                    scoring.assert_called_once()
                else:
                    learned.assert_not_called()
                    scoring.assert_not_called()
                    self.assertNotIn('min_capsid_area_px', heuristic.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
