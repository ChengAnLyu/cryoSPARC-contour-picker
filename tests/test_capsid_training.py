"""Training-label and sampling regressions independent of PyTorch."""
import unittest

import numpy as np

from training.train_capsid_profile import (
    sample_positive_rows, source_deck_records, training_positive_mask,
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


if __name__ == '__main__':
    unittest.main()
