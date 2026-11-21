"""Training-label, sampling and optional PyTorch integration regressions."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from training.train_capsid_profile import (
    main, pool_draw_summary, sample_negative_rows, sample_positive_rows,
    sample_training_indices, sha_file, source_deck_records, training_loss,
    training_masks, training_positive_mask, validate_folds,
    validate_training_options, validate_training_supervision,
)


class TrainingPositivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.regions = np.zeros((80, 100), np.uint16)
        self.regions[5:35, 5:35] = 1
        self.regions[50:63, 65:78] = 2
        self.valid = np.ones(self.regions.shape, bool)
        self.annotations = [{'region_id': 1, 'substantial_cluster': True},
                            {'region_id': 2, 'substantial_cluster': False}]

    def test_all_marked_keeps_small_positive_center_without_labeling_unmarked_pixels(self):
        earlier, _ = training_positive_mask(self.regions, self.annotations, self.valid)
        updated, ids = training_positive_mask(self.regions, self.annotations, self.valid, 'all-marked')
        self.assertEqual(ids, [1, 2])
        self.assertFalse(earlier[56, 71])
        self.assertTrue(updated[56, 71])
        self.assertFalse(np.any(updated & (self.regions == 0)))
        self.assertTrue(np.all(updated[earlier]))
        self.assertFalse(updated[50, 65])

    def test_valid_field_still_limits_positive_samples(self):
        self.valid[50:63, 65:78] = False
        updated, _ = training_positive_mask(self.regions, self.annotations, self.valid, 'all-marked')
        self.assertFalse(np.any(updated & ~self.valid))
        self.assertFalse(updated[56, 71])

    def test_unannotated_case_has_no_positive_samples(self):
        updated, ids = training_positive_mask(self.regions, [], self.valid, 'all-marked')
        self.assertFalse(updated.any())
        self.assertEqual(ids, [])


class TrainingSamplingTests(unittest.TestCase):
    def test_case_balance_gives_sparse_case_equal_probability_and_skips_empty_case(self):
        rows = sample_positive_rows(np.random.default_rng(270), [5, 5000, 0], 100_000, 'case-balanced')
        sparse_fraction = float(np.mean(rows < 5))
        self.assertGreater(sparse_fraction, .49)
        self.assertLess(sparse_fraction, .51)
        self.assertGreaterEqual(rows.min(), 0)
        self.assertLess(rows.max(), 5005)

    def test_legacy_pixel_draws_remain_identical(self):
        actual = sample_positive_rows(np.random.default_rng(270), [5, 5000, 0], 256, 'pixel')
        expected = np.random.default_rng(270).integers(5005, size=256)
        np.testing.assert_array_equal(actual, expected)

    def test_empty_or_negative_pools_are_rejected(self):
        for lengths in ([0, 0], [-1, 2]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                sample_positive_rows(np.random.default_rng(270), lengths, 256, 'case-balanced')


class TrainingSourceProvenanceTests(unittest.TestCase):
    def test_both_decks_are_recorded_without_local_directories(self):
        data = {'cases': [{'case_id': 1}, {'case_id': 41}], 'source_decks': [
            {'path': '/private/data/first.pptx', 'sha256': 'a' * 64, 'case_ids': [1]},
            {'path': '/private/data/second.pptx', 'sha256': 'b' * 64, 'case_ids': [41]}]}
        records = source_deck_records(data)
        self.assertEqual([item['name'] for item in records], ['first.pptx', 'second.pptx'])
        self.assertEqual([item['sha256'] for item in records], ['a' * 64, 'b' * 64])
        self.assertFalse(any('path' in item for item in records))

    def test_missing_or_duplicate_case_assignment_is_rejected(self):
        for ids in ([1], [1, 41, 41]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                source_deck_records({'cases': [{'case_id': 1}, {'case_id': 41}],
                                     'source_decks': [{'sha256': 'a' * 64, 'case_ids': ids}]})

    def test_uploaded_names_are_preserved_when_manifest_has_no_local_path(self):
        records = source_deck_records({'cases': [{'case_id': 41}], 'source_decks': [
            {'uploaded_name': 'second_deck.pptx', 'sha256': 'a' * 64, 'case_ids': [41]}]})
        self.assertEqual(records[0]['name'], 'second_deck.pptx')


class ConfirmedTrainingSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.regions = np.zeros((80, 100), np.uint16)
        self.regions[5:35, 5:35] = 1
        self.regions[50:70, 65:85] = 2
        self.valid = np.ones(self.regions.shape, bool)
        self.annotations = [{'region_id': 1, 'substantial_cluster': True},
                            {'region_id': 2, 'substantial_cluster': False}]
        self.case = {'annotations': {'regions': self.annotations}, 'training_supervision': {
            'confirmed_blank': False, 'confirmed_positive_region_ids': [1],
            'positive_scope': 'large_capsids', 'confirmation': {'source': 'explicit synthetic test instruction'},
            'unmarked_regions': 'unknown'}}

    def test_only_explicit_large_ids_are_positive_and_other_pixels_stay_unknown(self):
        supervision = validate_training_supervision(self.case)
        p, u, n, ids = training_masks(self.regions, self.annotations, self.valid,
                                     'confirmed-large', supervision)
        self.assertEqual(ids, [1])
        self.assertTrue(p[20, 20])
        self.assertFalse(p[60, 75])
        self.assertFalse(p[self.regions == 0].any())
        np.testing.assert_array_equal(u, self.valid)
        self.assertFalse(n.any())

    def test_confirmed_blank_has_negative_valid_pixels_and_no_positive_or_unlabeled_rows(self):
        self.case['annotations']['regions'] = []
        self.case['training_supervision'].update(confirmed_blank=True, confirmed_positive_region_ids=[])
        self.valid[:, :10] = False
        p, u, n, ids = training_masks(np.zeros_like(self.regions), [], self.valid, 'confirmed-large',
                                     validate_training_supervision(self.case))
        self.assertEqual(ids, [])
        self.assertFalse(p.any())
        self.assertFalse(u.any())
        np.testing.assert_array_equal(n, self.valid)

    def test_unmarked_case_without_confirmation_is_unknown_in_legacy_mode(self):
        p, u, n, ids = training_masks(np.zeros_like(self.regions), [], self.valid)
        self.assertEqual(ids, [])
        self.assertFalse(p.any() or n.any())
        np.testing.assert_array_equal(u, self.valid)

    def test_missing_conflicting_or_ambiguous_confirmation_is_rejected(self):
        invalid = []
        missing = copy.deepcopy(self.case)
        missing.pop('training_supervision')
        invalid.append(missing)
        for change in ({'confirmed_blank': True}, {'confirmed_blank': 'true'},
                       {'confirmed_positive_region_ids': [1, 1]},
                       {'confirmed_positive_region_ids': [3]},
                       {'confirmed_positive_region_ids': [True]},
                       {'confirmation': ''}, {'confirmation': {'bad': float('nan')}},
                       {'positive_scope': 'all_particles'}, {'unmarked_regions': 'negative'}):
            case = copy.deepcopy(self.case)
            case['training_supervision'].update(change)
            invalid.append(case)
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_training_supervision(case)

    def test_confirmed_region_must_exist_in_the_bitmap_and_valid_roi_limits_positives(self):
        supervision = validate_training_supervision(self.case)
        with self.assertRaisesRegex(ValueError, 'absent'):
            training_masks(np.zeros_like(self.regions), self.annotations, self.valid,
                           'confirmed-large', supervision)
        self.valid[:25] = False
        p, _, _, _ = training_masks(self.regions, self.annotations, self.valid,
                                    'confirmed-large', supervision)
        self.assertFalse(p[~self.valid].any())

    def test_supervised_legacy_policy_cannot_add_unconfirmed_small_regions(self):
        with self.assertRaisesRegex(ValueError, 'unconfirmed'):
            training_masks(self.regions, self.annotations, self.valid, 'all-marked',
                           validate_training_supervision(self.case))


class ConfirmedNegativeSamplingTests(unittest.TestCase):
    def test_legacy_pu_indices_and_rng_state_are_unchanged_without_negatives(self):
        positive = np.zeros((100, 100), bool)
        positive[10:80, 10:80] = True
        valid = np.ones_like(positive)
        actual_rng, expected_rng = np.random.default_rng(280), np.random.default_rng(280)
        p, u, n = sample_training_indices(actual_rng, positive, valid, np.zeros_like(valid))
        expected_p = expected_rng.choice(np.flatnonzero(positive), 2500, replace=False)
        expected_u = expected_rng.choice(np.flatnonzero(valid), 5000, replace=False)
        np.testing.assert_array_equal(p, expected_p)
        np.testing.assert_array_equal(u, expected_u)
        self.assertEqual(n.size, 0)
        np.testing.assert_array_equal(actual_rng.integers(1000, size=20), expected_rng.integers(1000, size=20))

    def test_negative_rows_are_case_balanced_and_skip_empty_cases(self):
        rows = sample_negative_rows(np.random.default_rng(280), [0, 5, 5000, 0], 100_000)
        self.assertGreater(float(np.mean(rows < 5)), .49)
        self.assertLess(float(np.mean(rows < 5)), .51)
        self.assertGreaterEqual(rows.min(), 0)
        self.assertLess(rows.max(), 5005)

    def test_negative_indices_and_actual_draw_summary_keep_empty_case_offsets(self):
        valid = np.ones((100, 100), bool)
        valid[:10] = False
        p, u, n = sample_training_indices(np.random.default_rng(280), np.zeros_like(valid),
                                          np.zeros_like(valid), valid)
        self.assertEqual((len(p), len(u), len(n)), (0, 0, 5000))
        self.assertTrue(valid.ravel()[n].all())
        summary, offsets = pool_draw_summary([2, 4, 6], [0, 3, 0], np.array([2, 0, 4]), 'negative')
        np.testing.assert_array_equal(offsets, [0, 0, 3, 3])
        self.assertEqual(summary['2']['negative_draws'], 0)
        self.assertEqual(summary['4'], {'available_negative_rows': 3, 'negative_draws': 6,
                                       'unique_negative_rows_drawn': 2, 'max_reuse_of_one_row': 4})
        self.assertEqual(summary['6']['negative_draws'], 0)

    def test_invalid_prior_or_negative_weight_is_rejected(self):
        for prior in (0, 1, float('nan'), float('inf'), True):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                validate_training_options(prior, .15)
        for weight in (-.01, float('nan'), float('inf'), True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_training_options(.15, weight)
        validate_training_options(.15, 0)
        validate_training_options(.15, .3)


class ConfirmedFoldIsolationTests(unittest.TestCase):
    def setUp(self):
        self.cases = {1: {'evaluation_group': 'same-image'}, 2: {'evaluation_group': 'same-image'},
                      3: {'evaluation_group': 'other-image'}, 4: {'evaluation_group': 'other-image'}}

    def test_positive_and_blank_crops_of_one_group_cannot_cross_the_split(self):
        validate_folds(self.cases, [{'train_cases': [1, 2], 'test_cases': [3, 4]},
                                   {'train_cases': [3, 4], 'test_cases': [1, 2]}])
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            validate_folds(self.cases, [{'train_cases': [1, 3], 'test_cases': [2, 4]},
                                       {'train_cases': [2, 4], 'test_cases': [1, 3]}])

    def test_duplicate_cases_or_incomplete_heldout_coverage_are_rejected(self):
        for folds in ([{'train_cases': [1, 1, 2], 'test_cases': [3, 4]}],
                      [{'train_cases': [1, 2], 'test_cases': [3, 4]}] * 2):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                validate_folds(self.cases, folds)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch training extra is not installed')
class ConfirmedNegativeObjectiveTests(unittest.TestCase):
    def test_zero_negative_weight_preserves_legacy_loss_and_gradients_exactly(self):
        import torch
        from torch.nn.functional import softplus
        for p_values, u_values in (([.2, 1.3], [-1., .5]), ([8., 8.], [-8., -8.])):
            p = torch.tensor(p_values, requires_grad=True)
            u = torch.tensor(u_values, requires_grad=True)
            actual, _ = training_loss(p, u)
            actual_grads = torch.autograd.grad(actual, (p, u))
            expected = .15 * softplus(-p).mean() + torch.relu(softplus(u).mean() - .15 * softplus(p).mean())
            expected_grads = torch.autograd.grad(expected, (p, u))
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads)))

    def test_negative_supervision_pushes_a_false_positive_score_down(self):
        import torch
        p = torch.tensor([1.], requires_grad=True)
        u = torch.tensor([-1.], requires_grad=True)
        n = torch.tensor([2.], requires_grad=True)
        loss, components = training_loss(p, u, n, negative_weight=.15)
        p_grad, _, n_grad = torch.autograd.grad(loss, (p, u, n))
        self.assertLess(float(p_grad), 0)
        self.assertGreater(float(n_grad), 0)
        updated = n.detach() - .1 * n_grad
        self.assertLess(float(torch.sigmoid(updated)), float(torch.sigmoid(n.detach())))
        self.assertGreater(float(components['weighted_negative_risk'].detach()), 0)
        with self.assertRaisesRegex(ValueError, 'confirmed-negative'):
            training_loss(p, u, negative_weight=.15)

    def test_cli_training_binds_separate_pun_pools_and_uses_training_only_normalizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for cid in range(1, 5):
                blank = cid % 2 == 0
                gray = np.random.default_rng(cid).integers(30, 225, (96, 96), dtype=np.uint8)
                Image.fromarray(np.repeat(gray[..., None], 3, axis=2)).save(root / f'image-{cid}.png')
                Image.fromarray(np.full(gray.shape, 255, np.uint8)).save(root / f'valid-{cid}.png')
                regions = np.zeros(gray.shape, np.uint16)
                if not blank:
                    regions[20:76, 20:76] = 1
                Image.fromarray(regions).save(root / f'regions-{cid}.png')
                cases.append({'case_id': cid, 'evaluation_group': cid,
                              'images': {'input': {'path': f'image-{cid}.png'}},
                              'valid_image_pixels': {'path': f'valid-{cid}.png',
                                                     'positive_regions_path': f'regions-{cid}.png'},
                              'annotations': {'regions': [] if blank else [{'region_id': 1, 'substantial_cluster': True}]},
                              'training_supervision': {'confirmed_blank': blank,
                                  'confirmed_positive_region_ids': [] if blank else [1],
                                  'positive_scope': 'large_capsids', 'unmarked_regions': 'unknown',
                                  'confirmation': 'Synthetic test fixture, not user data'}})
            manifest = {'cases': cases, 'source_decks': [{'uploaded_name': 'synthetic-fixture.pptx',
                        'sha256': 'a' * 64, 'case_ids': [1, 2, 3, 4]}]}
            folds = [{'train_cases': [1, 2], 'test_cases': [3, 4]},
                     {'train_cases': [3, 4], 'test_cases': [1, 2]}]
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'folds.json').write_text(json.dumps({'folds': folds}))
            output = root / 'trained'
            argv = ['train', '--dataset', str(root / 'manifest.json'), '--folds', str(root / 'folds.json'),
                    '--output', str(output), '--steps', '2', '--positive-policy', 'confirmed-large',
                    '--positive-sampling', 'case-balanced', '--negative-weight', '.15']
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            for cid in range(1, 5):
                with np.load(output / 'samples' / f'case_{cid:02d}.npz') as sample:
                    self.assertEqual(len(sample['n']) == 0, cid % 2 != 0)
                    self.assertEqual(len(sample['u']) == 0, cid % 2 == 0)
                    self.assertEqual(len(sample['p']) == 0, cid % 2 == 0)
                    self.assertEqual(len(sample['n']), len(sample['n_indices']))
            for index, fold in enumerate(folds):
                profile_path = output / f'prior-fold-{index}.json'
                model = json.loads(profile_path.read_text())
                training = model['training']
                self.assertEqual(training['case_ids'], fold['train_cases'])
                train_u = []
                for cid in fold['train_cases']:
                    with np.load(output / 'samples' / f'case_{cid:02d}.npz') as sample:
                        train_u.append(sample['u'])
                train_u = np.concatenate(train_u)
                np.testing.assert_array_equal(np.asarray(model['mean'], np.float32), train_u.mean(0))
                np.testing.assert_array_equal(np.asarray(model['scale'], np.float32), np.maximum(train_u.std(0), .05))
                self.assertEqual(training['training_supervision']['confirmed_blank_case_ids'], [fold['train_cases'][1]])
                self.assertEqual(set(training['negative_draw_summary']), {str(c) for c in fold['train_cases']})
                with np.load(output / f'prior-fold-{index}-draws.npz') as draws:
                    self.assertEqual(int(draws['negative_row_draws'].sum()), 512)
                    self.assertEqual(int(draws['positive_row_draws'].sum()), 512)
                    self.assertEqual(int(draws['unlabeled_row_draws'].sum()), 2048)
                    np.testing.assert_array_equal(draws['train_case_ids'], fold['train_cases'])
                self.assertEqual(training['sampling_draws_sha256'], sha_file(output / f'prior-fold-{index}-draws.npz'))
                self.assertTrue(all(row['weighted_negative_risk'] > 0 for row in training['loss_trace']))
                for cid in fold['test_cases']:
                    metadata = json.loads((output / 'scores' / f'case_{cid:02d}.json').read_text())
                    self.assertEqual(metadata['model_sha256'], sha_file(profile_path))


if __name__ == '__main__':
    unittest.main()
