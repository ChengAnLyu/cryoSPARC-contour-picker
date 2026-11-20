"""Synthetic invariants for the conservative carbon-edge veto, not accuracy tests."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carbon_support import (
    CarbonContext, carbon_candidate_evidence, detect_carbon_edges, guard_proposals,
)


def film_and_nearby_core():
    """Long film edge, a false sliver along it, and a compact adjacent target.

    The compact target intentionally approaches the film edge.  The false
    sliver is shorter than the visible edge, so global continuation is known.
    """
    h, w = 320, 480
    yy, xx = np.mgrid[:h, :w]
    distance = xx - .35 * yy - 150
    rng = np.random.default_rng(261)
    gray = 125 + rng.normal(0, 4, (h, w))
    gray[distance < 0] -= 35
    gray[np.abs(distance) < 1.7] = 25
    sliver = ((distance >= -20) & (distance <= -2) &
              (yy >= 95) & (yy <= 220))
    core = (np.abs(xx - 235) < 30) & (np.abs(yy - 160) < 30)
    gray[core] += rng.normal(0, 25, int(core.sum()))
    rgb = np.repeat(np.uint8(np.clip(gray, 0, 255))[..., None], 3, axis=2)
    return rgb, sliver, core


class CarbonSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb, cls.sliver, cls.core = film_and_nearby_core()
        cls.context = detect_carbon_edges(cls.rgb)

    def test_long_edge_rejects_a_sliver_but_keeps_adjacent_compact_core(self):
        self.assertGreaterEqual(len(self.context.segments), 1)
        sliver = carbon_candidate_evidence(self.sliver, self.context)
        core = carbon_candidate_evidence(self.core, self.context)
        self.assertTrue(sliver['reject'])
        self.assertFalse(core['reject'])
        # The target is actually near the edge; a passing test cannot be
        # explained by accidentally putting it far from the film.
        self.assertTrue(core.get('matches'))
        self.assertGreater(max(m['boundary_contact_fraction'] for m in core['matches']), .15)

    def test_kept_records_masks_and_screenshot_are_not_modified_or_reordered(self):
        rgb_before = self.rgb.copy()
        first = {'segmentation': self.core.copy(), 'id': 'first'}
        rejected = {'segmentation': self.sliver.copy(), 'id': 'film'}
        last = {'segmentation': np.roll(self.core, 130, axis=1), 'id': 'last'}
        masks_before = [record['segmentation'].copy() for record in (first, rejected, last)]
        kept, summary = guard_proposals(self.context, [first, rejected, last])
        self.assertEqual(summary['rejected_indices'], [1])
        self.assertEqual(summary['accepted_indices'], [0, 2])
        self.assertIs(kept[0], first)
        self.assertIs(kept[1], last)
        np.testing.assert_array_equal(self.rgb, rgb_before)
        for record, before in zip((first, rejected, last), masks_before):
            np.testing.assert_array_equal(record['segmentation'], before)

    def test_internal_holes_do_not_dilute_external_boundary_evidence(self):
        perforated = self.sliver.copy()
        yy, xx = np.mgrid[:perforated.shape[0], :perforated.shape[1]]
        distance = xx - .35 * yy - 150
        holes = ((distance > -15) & (distance < -8) &
                 (yy % 9 < 3) & (yy > 104) & (yy < 210))
        perforated[holes] = False
        before = perforated.copy()
        self.assertTrue(carbon_candidate_evidence(perforated, self.context)['reject'])
        np.testing.assert_array_equal(perforated, before)
        self.assertGreater(np.count_nonzero(self.sliver & ~perforated), 0)

    def test_compact_object_on_carbon_side_is_kept_without_blanket_side_exclusion(self):
        yy, xx = np.mgrid[:self.core.shape[0], :self.core.shape[1]]
        compact = ((xx - 100) / 25) ** 2 + ((yy - 150) / 28) ** 2 < 1
        self.assertFalse(carbon_candidate_evidence(compact, self.context)['reject'])

    def test_broad_parent_does_not_veto_its_independent_compact_child(self):
        yy, xx = np.mgrid[:self.core.shape[0], :self.core.shape[1]]
        broad_parent = xx - .35 * yy - 150 < 0
        # A large region may contain a real core. This conservative guard does
        # not infer the child's identity from that parent or its dark side.
        self.assertFalse(carbon_candidate_evidence(broad_parent, self.context)['reject'])
        self.assertFalse(carbon_candidate_evidence(self.core, self.context)['reject'])

    def test_short_polygon_facets_do_not_become_a_global_carbon_boundary(self):
        image = Image.new('L', (480, 320), 125)
        vertices = [(155, 140), (200, 95), (250, 115),
                    (280, 165), (235, 205), (175, 195)]
        ImageDraw.Draw(image).polygon(vertices, fill=125, outline=30, width=3)
        mask_image = Image.new('1', image.size)
        ImageDraw.Draw(mask_image).polygon(vertices, fill=1)
        rgb = np.repeat(np.asarray(image)[..., None], 3, axis=2)
        context = detect_carbon_edges(rgb)
        self.assertFalse(carbon_candidate_evidence(np.asarray(mask_image, bool), context)['reject'])
        self.assertEqual(len(context.segments), 0)

    def test_isolated_internal_long_facet_requires_full_field_context(self):
        image = Image.new('L', (640, 640), 125)
        # Longer than the minimum Hough length, but wholly inside the field.
        # It might be a membrane facet, so line length alone is insufficient.
        ImageDraw.Draw(image).line([(150, 170), (420, 470)], fill=25, width=4)
        rgb = np.repeat(np.asarray(image)[..., None], 3, axis=2)
        self.assertEqual(len(detect_carbon_edges(rgb).segments), 0)

    def test_white_screenshot_margin_is_not_a_carbon_edge(self):
        rng = np.random.default_rng(5)
        gray = np.full((320, 480), 255., dtype=float)
        gray[:, 45:435] = 125 + rng.normal(0, 4, (320, 390))
        rgb = np.repeat(np.uint8(np.clip(gray, 0, 255))[..., None], 3, axis=2)
        self.assertEqual(len(detect_carbon_edges(rgb).segments), 0)

    def test_curved_membrane_without_a_long_straight_span_fails_open(self):
        yy, xx = np.mgrid[:480, :480]
        radius = np.hypot(xx - 240, yy - 240)
        gray = np.full(radius.shape, 125, np.uint8)
        gray[np.abs(radius - 160) < 2] = 20
        rgb = np.repeat(gray[..., None], 3, axis=2)
        context = detect_carbon_edges(rgb)
        self.assertEqual(len(context.segments), 0)
        self.assertFalse(carbon_candidate_evidence(radius < 160, context)['reject'])

    def test_blank_screens_and_tiny_inputs_return_no_edges(self):
        for shape in [(320, 480, 3), (5, 5, 3)]:
            for level in (0, 125, 255):
                with self.subTest(shape=shape, level=level):
                    context = detect_carbon_edges(np.full(shape, level, np.uint8))
                    self.assertEqual(len(context.segments), 0)

    def test_noise_is_not_a_coherent_edge(self):
        rng = np.random.default_rng(20)
        gray = np.uint8(np.clip(125 + rng.normal(0, 35, (400, 600)), 0, 255))
        rgb = np.repeat(gray[..., None], 3, axis=2)
        self.assertEqual(len(detect_carbon_edges(rgb).segments), 0)

    def test_large_screenshot_uses_bounded_context_and_correct_mask_coordinates(self):
        # Nearest expansion preserves the known synthetic scene; detection
        # resizes its own working data and must return source coordinates.
        rgb = np.repeat(np.repeat(self.rgb, 4, axis=0), 4, axis=1)
        sliver = np.repeat(np.repeat(self.sliver, 4, axis=0), 4, axis=1)
        core = np.repeat(np.repeat(self.core, 4, axis=0), 4, axis=1)
        context = detect_carbon_edges(rgb)
        self.assertLessEqual(max(context.work_shape), 640)
        self.assertEqual(context.image_shape, rgb.shape[:2])
        self.assertTrue(carbon_candidate_evidence(sliver, context)['reject'])
        self.assertFalse(carbon_candidate_evidence(core, context)['reject'])
        self.assertTrue(all(segment.length > 500 for segment in context.segments))

    def test_rotation_keeps_diagonal_association_in_source_coordinates(self):
        for turns in (1, 2, 3):
            with self.subTest(turns=turns):
                rgb = np.ascontiguousarray(np.rot90(self.rgb, turns))
                context = detect_carbon_edges(rgb)
                self.assertTrue(carbon_candidate_evidence(np.rot90(self.sliver, turns), context)['reject'])
                self.assertFalse(carbon_candidate_evidence(np.rot90(self.core, turns), context)['reject'])

    def test_wrong_mask_size_and_non_rgb_input_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'different size'):
            carbon_candidate_evidence(np.zeros((10, 20), bool), self.context)
        with self.assertRaisesRegex(ValueError, 'RGB uint8'):
            detect_carbon_edges(np.zeros((30, 40), np.uint8))
        with self.assertRaisesRegex(ValueError, 'RGB uint8'):
            detect_carbon_edges(np.zeros((30, 40, 3), np.float32))

    def test_cancellation_propagates_during_context_and_candidate_work(self):
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            if calls >= 6:
                raise InterruptedError('cancelled carbon guard')
        with self.assertRaisesRegex(InterruptedError, 'cancelled carbon guard'):
            detect_carbon_edges(self.rgb, check_cancel=stop)
        def stop_immediately():
            raise InterruptedError('cancelled contour')
        with self.assertRaisesRegex(InterruptedError, 'cancelled contour'):
            guard_proposals(self.context, [{'segmentation': self.core}], stop_immediately)

    def test_no_edge_context_preserves_every_proposal_and_checks_shapes(self):
        context = CarbonContext(self.core.shape, self.core.shape, ())
        proposal = {'segmentation': self.sliver}
        kept, summary = guard_proposals(context, [proposal])
        self.assertIs(kept[0], proposal)
        self.assertEqual(summary['rejected_count'], 0)
        with self.assertRaisesRegex(ValueError, 'different size'):
            guard_proposals(context, [{'segmentation': np.ones((2, 2), bool)}])

    def test_small_valid_island_inside_ui_fails_open_without_crashing(self):
        rgb = np.full((640, 640, 3), 255, np.uint8)
        noise = np.uint8(np.clip(120 + np.random.default_rng(22).normal(0, 30, (70, 70)), 0, 255))
        rgb[280:350, 280:350] = noise[..., None]
        self.assertEqual(len(detect_carbon_edges(rgb).segments), 0)


if __name__ == '__main__':
    unittest.main()
