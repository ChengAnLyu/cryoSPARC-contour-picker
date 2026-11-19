"""Geometry and consumer contracts for polygonal capsid proposals."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capsid import (polygonal_capsid_mask, _capsid_boundary_rings,
                    _capsid_raster_polygon, _capsid_simple_polygon)
from execution import CaptureGeometry
from manual_sampling import mask_contour_points
from recognition import make_candidate
from screen_app import screen_mask_at
from screen_overlay import _mask_image, TRANSPARENT_RGB


def polygon(shape, points):
    image = Image.new("1", (shape[1], shape[0]))
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


class CapsidPolygonTests(unittest.TestCase):
    def test_polygon_edges_replace_noisy_boundary_without_large_area_change(self):
        shape = (160, 180)
        original = polygon(shape, [(44, 25), (112, 18), (150, 62),
                                   (139, 120), (77, 141), (24, 96)])
        # Small boundary notches/spikes imitate threshold variation.
        original = original.copy()
        original[24:27, 54:58] = False
        original[72:76, 145:151] = True
        fitted, details = polygonal_capsid_mask(original)
        component = details["components"][0]
        self.assertTrue(component["applied"])
        self.assertLessEqual(component["vertices"], 12)
        self.assertGreaterEqual(np.count_nonzero(fitted & original), .90 * original.sum())
        self.assertLessEqual(np.count_nonzero(fitted & ~original), .10 * original.sum())
        self.assertEqual(details["polygon_area"], fitted.sum())
        self.assertFalse(np.array_equal(fitted, original))

    def test_deep_concavity_is_preserved_instead_of_filling_a_convex_hull(self):
        original = np.zeros((110, 120), dtype=bool)
        original[10:100, 10:110] = True
        original[10:75, 45:75] = False
        fitted, details = polygonal_capsid_mask(original)
        self.assertTrue(details["components"][0]["applied"])
        self.assertFalse(fitted[10:70, 48:72].any())
        self.assertTrue(fitted[80:95, 20:100].all())

    def test_near_convex_core_gets_simple_polygon_with_bounded_growth(self):
        original = np.zeros((120, 130), dtype=bool)
        original[15:105, 20:110] = True
        original[15:28, 50:64] = False  # A shallow threshold notch.
        fitted, details = polygonal_capsid_mask(original)
        component = details["components"][0]
        self.assertEqual(component["strategy"], "near-convex polygon")
        self.assertEqual(component["vertices"], 4)
        self.assertTrue(fitted[15:28, 50:64].all())
        self.assertLessEqual(np.count_nonzero(fitted & ~original), .25 * original.sum())
        self.assertLessEqual(component["maximum_outward_growth_pixels"], 24)

    def test_polygon_must_not_bridge_a_long_unsupported_notch(self):
        original = np.zeros((200, 200), dtype=bool)
        original[10:190, 10:190] = True
        original[10:80, 60:140] = False
        fitted, details = polygonal_capsid_mask(original)
        # Even though this notch occupies less than a quarter of the rectangle,
        # a hull would grow far beyond the permitted support band.
        self.assertEqual(details["components"][0]["strategy"], "concave polygon")
        self.assertFalse(fitted[10:70, 65:135].any())

    def test_original_holes_and_forbidden_pixels_stay_unpickable(self):
        original = polygon((120, 140), [(30, 10), (105, 12), (125, 65),
                                        (93, 105), (24, 95), (10, 43)])
        original = original.copy()
        original[48:65, 47:68] = False
        allowed = np.ones_like(original)
        allowed[72:78, 86:92] = False
        fitted, details = polygonal_capsid_mask(original, allowed_mask=allowed)
        self.assertFalse(fitted[48:65, 47:68].any())
        self.assertFalse(fitted[72:78, 86:92].any())
        self.assertFalse((fitted & ~allowed).any())
        self.assertTrue(details["all_internal_exclusions_preserved"])

    def test_only_tiny_texture_pinholes_are_filled_within_hard_constraints(self):
        original = np.zeros((120, 120), dtype=bool)
        original[10:110, 10:110] = True
        original[30:33, 30:33] = False  # Nine-pixel texture pinhole.
        original[50:53, 50:53] = False  # Same size, but forbidden by SAM/UI.
        original[70:85, 70:85] = False  # Larger dark-region exclusion.
        allowed = np.ones_like(original)
        allowed[50:53, 50:53] = False
        fitted, details = polygonal_capsid_mask(original, allowed_mask=allowed)
        self.assertTrue(fitted[30:33, 30:33].all())
        self.assertFalse(fitted[50:53, 50:53].any())
        self.assertFalse(fitted[70:85, 70:85].any())
        self.assertEqual(details["components"][0]["filled_pinhole_pixels"], 9)
        self.assertTrue(details["hard_exclusions_preserved"])

    def test_pinhole_cleanup_has_total_area_budget(self):
        original = np.zeros((110, 110), dtype=bool)
        original[5:105, 5:105] = True
        for y in range(12, 100, 6):
            for x in range(12, 100, 6):
                original[y, x] = False
        fitted, details = polygonal_capsid_mask(original)
        filled_count = details["components"][0]["filled_pinhole_pixels"]
        self.assertGreater(filled_count, 0)
        self.assertLessEqual(filled_count, .02 * original.sum())
        self.assertLess(filled_count, np.count_nonzero(~original[5:105, 5:105]))

    def test_disconnected_targets_and_diagonal_contacts_are_not_joined(self):
        original = np.zeros((100, 140), dtype=bool)
        original[10:45, 10:50] = True
        original[45:85, 50:95] = True  # Diagonal touch only.
        original[12:85, 97:130] = True  # One-column gap.
        fitted, details = polygonal_capsid_mask(original)
        self.assertEqual(ndi.label(fitted)[1], 3)
        self.assertEqual(len(details["components"]), 3)
        self.assertFalse(fitted[:, 95:97].any())

    def test_frame_truncated_polygon_keeps_the_straight_frame_edge(self):
        original = polygon((90, 120), [(0, 0), (60, 0), (78, 34),
                                       (60, 75), (0, 68)])
        fitted, _ = polygonal_capsid_mask(original)
        self.assertTrue(fitted[1:60, 0].all())
        self.assertTrue(fitted[0, 1:55].all())
        self.assertFalse(fitted[:, 85:].any())

    def test_exact_cell_boundaries_rasterize_without_half_pixel_shift(self):
        for shape, bounds in (((20, 30), (slice(3, 15), slice(7, 21))),
                              ((20, 30), (slice(0, 20), slice(0, 30)))):
            original = np.zeros(shape, dtype=bool)
            original[bounds] = True
            rings = _capsid_boundary_rings(original)
            self.assertEqual(len(rings), 1)
            self.assertEqual(len(rings[0]), 4)
            np.testing.assert_array_equal(_capsid_raster_polygon(rings[0], shape), original)
            fitted, _ = polygonal_capsid_mask(original)
            np.testing.assert_array_equal(fitted, original)

    def test_empty_single_pixel_and_thin_masks_do_not_crash(self):
        for region in (None, (slice(6, 7), slice(4, 5)),
                       (slice(0, 20), slice(5, 6)),
                       (slice(3, 4), slice(0, 30))):
            original = np.zeros((20, 30), dtype=bool)
            if region is not None:
                original[region] = True
            fitted, _ = polygonal_capsid_mask(original)
            np.testing.assert_array_equal(fitted, original)

    def test_self_crossing_polygon_is_rejected(self):
        self.assertFalse(_capsid_simple_polygon(np.array([(0, 0), (10, 10),
                                                          (0, 10), (10, 0)])))
        self.assertTrue(_capsid_simple_polygon(np.array([(0, 0), (10, 0),
                                                         (10, 10), (0, 10)])))

    def test_mask_and_constraint_validation(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            polygonal_capsid_mask(np.zeros((0, 4), bool))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            polygonal_capsid_mask(np.zeros((4, 5), bool), allowed_mask=np.ones((5, 4), bool))

    def test_cancellation_interrupts_fitting(self):
        original = np.ones((200, 200), bool)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise InterruptedError("cancel polygon fit")
        with self.assertRaisesRegex(InterruptedError, "cancel polygon"):
            polygonal_capsid_mask(original, check_cancel=cancel)

    def test_display_hit_testing_and_sampling_share_the_fitted_mask(self):
        original = polygon((120, 140), [(30, 10), (105, 12), (125, 65),
                                        (93, 105), (24, 95), (10, 43)])
        fitted, _ = polygonal_capsid_mask(original)
        candidate = make_candidate(fitted, source="capsid polygon (review)")
        geometry = CaptureGeometry(200, 300, 140, 120)
        outline = np.asarray(_mask_image([candidate], geometry))
        painted = np.any(outline != np.asarray(TRANSPARENT_RGB), axis=2)
        self.assertTrue(painted.any())
        self.assertFalse((painted & ~fitted).any())
        for x, y in mask_contour_points(candidate.mask, 13):
            self.assertTrue(fitted[y, x])
            self.assertEqual(screen_mask_at([candidate], geometry, x + 200, y + 300), candidate)
        self.assertIsNone(screen_mask_at([candidate], geometry, 200, 300))


if __name__ == "__main__":
    unittest.main()
