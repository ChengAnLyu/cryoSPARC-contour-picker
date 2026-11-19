"""Manual picking retains the original grid and polygon-boundary behavior."""
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution import CancelledOperation
from manual_sampling import legacy_contour_points, mask_contour_points


def original_picks(points, spacing):
    """Independent literal baseline from the user's original manual script."""
    def original_contains(x, y, pts):
        n = len(pts)
        inside = False
        p1x, p1y = pts[0]
        for i in range(1, n + 1):
            p2x, p2y = pts[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if (p2y - p1y) != 0:
                    x_intersect = p1x + (y - p1y) * (p2x - p1x) / (p2y - p1y)
                    if x <= x_intersect:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    x_min = int(min(p[0] for p in points))
    y_min = int(min(p[1] for p in points))
    x_max = int(max(p[0] for p in points))
    y_max = int(max(p[1] for p in points))
    result = []
    row_height = int(spacing * (3 ** 0.5) / 2)
    row = 0
    y = y_min
    while y <= y_max:
        x_offset = spacing // 2 if row % 2 else 0
        x = x_min + x_offset
        while x <= x_max:
            if original_contains(x, y, points):
                result.append((x, y))
            x += spacing
        y += row_height
        row += 1
    return np.asarray(result, dtype=np.int32).reshape((-1, 2))


class OriginalGridTests(unittest.TestCase):
    def test_fixed_contours_match_original_points_and_order(self):
        contours = [
            [(0, 0), (105, 0), (105, 80), (0, 80)],
            [(101, 307), (203, 310), (211, 389), (112, 403), (95, 350)],
            [(10, 10), (70, 10), (70, 30), (30, 30), (30, 70), (10, 70)],
            [(-70, -25), (-2, -20), (-10, 40), (-90, 21)],
            [(-32.8, -21.7), (-2.3, -18.4), (-8.2, 19.6), (-30.1, 8.1)],
            [(3.25, 4.75), (38.8, 7.2), (23.4, 38.6), (4.8, 22.3)],
            [(0, 0), (60, 50), (0, 50), (60, 0)],  # zero signed area, nonempty lobes
            [(4, 9), (4, 9), (48, 9), (48, 55), (4, 55), (4, 9)],
        ]
        for points in contours:
            for spacing in (5, 6, 13, 25):
                with self.subTest(points=points, spacing=spacing):
                    actual = legacy_contour_points(points, spacing)
                    np.testing.assert_array_equal(actual, original_picks(points, spacing))
                    self.assertEqual(actual.dtype, np.int32)

    def test_original_asymmetric_boundary_rule_has_no_inset(self):
        actual = legacy_contour_points([(0, 0), (20, 0), (20, 20), (0, 20)], 5)
        self.assertFalse((actual[:, 0] == 0).any())
        self.assertFalse((actual[:, 1] == 0).any())
        self.assertTrue((actual[:, 0] == 20).any())
        self.assertTrue((actual[:, 1] == 20).any())
        np.testing.assert_array_equal(actual[:4], [[2, 4], [7, 4], [12, 4], [17, 4]])

    def test_manual_grid_moves_with_contour_bounding_box(self):
        polygon = np.array([(10, 20), (80, 20), (80, 90), (10, 90)])
        displacement = np.array([7, 5])
        first = legacy_contour_points(polygon, 13)
        shifted = legacy_contour_points(polygon + displacement, 13)
        np.testing.assert_array_equal(shifted, first + displacement)

    def test_tiny_valid_polygon_returns_empty_xy_array(self):
        actual = legacy_contour_points([(0, 0), (1, 0), (0, 1)], 13)
        self.assertEqual(actual.shape, (0, 2))
        self.assertEqual(actual.dtype, np.int32)

    def test_numpy_integer_controls_are_supported(self):
        contour = [(5, 5), (40, 5), (40, 40), (5, 40)]
        np.testing.assert_array_equal(
            legacy_contour_points(contour, np.int64(13), np.int32(100)),
            original_picks(contour, 13))


class ManualValidationTests(unittest.TestCase):
    contour = [(0, 0), (500, 0), (500, 400), (0, 400)]

    def test_invalid_contours_are_rejected(self):
        invalid = [None, [], [(1, 2)], [(1, 2), (3, 4)],
                   [1, 2, 3], [(0, 0), (3, 0), (3,)],
                   [(0, 0), (1, 1), (2, 2)],
                   [(3, 3), (3, 3), (3, 3)],
                   [(0, 0), (math.inf, 0), (0, 3)],
                   [(0, 0), (3, 0), (0, math.nan)],
                   [(0, 0), (3 + 1j, 0), (0, 3)],
                   [(False, False), (True, False), (False, True)],
                   [("0", "0"), ("3", "0"), ("0", "3")],
                   [(0, 0), (2**31, 0), (0, 3)]]
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(ValueError):
                legacy_contour_points(points, 13)

    def test_invalid_spacing_cannot_create_an_infinite_loop(self):
        for spacing in (0, 1, -2, 1.5, 13.0, True, np.bool_(True), math.inf, 2**100):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                legacy_contour_points(self.contour, spacing)

    def test_invalid_limits_are_rejected(self):
        for limit in (0, -1, 2.5, True, np.bool_(False), None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                legacy_contour_points(self.contour, 13, max_points=limit)

    def test_point_limit_rejects_instead_of_returning_partial_picks(self):
        with self.assertRaisesRegex(ValueError, "More than 20"):
            legacy_contour_points(self.contour, 5, max_points=20)

    def test_exact_limit_is_allowed(self):
        expected = original_picks(self.contour, 13)
        np.testing.assert_array_equal(
            legacy_contour_points(self.contour, 13, max_points=len(expected)), expected)
        with self.assertRaises(ValueError):
            legacy_contour_points(self.contour, 13, max_points=len(expected) - 1)

    def test_extreme_bounding_box_is_rejected_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "too many sampling checks"):
            legacy_contour_points([(0, 0), (1_000_000, 0), (1, 1_000_000)], 13)
        # A narrow box could evade a total-grid-size check while requiring many rows.
        with self.assertRaisesRegex(ValueError, "too many sampling checks"):
            legacy_contour_points([(0, 0), (0.1, 0), (0.1, 1_000_000)], 13)

    def test_excessive_vertex_work_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too many vertices"):
            legacy_contour_points([(0, 0), (10, 0), (0, 10)] * 20_000, 13)
        with self.assertRaisesRegex(ValueError, "too many sampling checks"):
            legacy_contour_points([(0, 0), (10, 0), (0, 20_000)] * 2_000, 5)

    def test_cancellation_before_or_during_work_returns_no_partial_result(self):
        with self.assertRaises(CancelledOperation):
            legacy_contour_points(self.contour, 13, cancel=lambda: True)
        calls = [0]
        def cancel_after_rows():
            calls[0] += 1
            return calls[0] > 4
        with self.assertRaises(CancelledOperation):
            legacy_contour_points(self.contour, 13, cancel=cancel_after_rows)
        self.assertGreater(calls[0], 4)

    def test_cancel_argument_is_validated(self):
        with self.assertRaisesRegex(ValueError, "cancel"):
            legacy_contour_points(self.contour, 13, cancel=True)

    def test_import_needs_no_model_or_desktop_packages(self):
        result = subprocess.run(
            [sys.executable, "-c", "import manual_sampling,sys; "
             "assert not any(name in sys.modules for name in "
             "('torch','segment_anything','cv2','tkinter','pyautogui','pynput'))"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class MaskContourGridTests(unittest.TestCase):
    def test_mask_owns_its_bbox_phase_and_has_no_boundary_inset(self):
        mask = np.zeros((25, 40), dtype=bool)
        mask[7:18, 11:29] = True
        expected = [[11, 7], [16, 7], [21, 7], [26, 7],
                    [13, 11], [18, 11], [23, 11], [28, 11],
                    [11, 15], [16, 15], [21, 15], [26, 15]]
        actual = mask_contour_points(mask, 5)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.int32)
        self.assertEqual(np.count_nonzero(mask), 11 * 18)

    def test_odd_spacing_uses_manual_integer_row_height_and_offset(self):
        mask = np.zeros((35, 40), dtype=bool)
        mask[5:29, 3:34] = True
        expected = [[3, 5], [16, 5], [29, 5],
                    [9, 16], [22, 16],
                    [3, 27], [16, 27], [29, 27]]
        np.testing.assert_array_equal(mask_contour_points(mask, 13), expected)

    def test_holes_and_empty_space_between_components_are_not_filled(self):
        mask = np.zeros((50, 70), dtype=bool)
        mask[3:30, 5:32] = True
        mask[10:22, 10:25] = False
        mask[35:48, 48:68] = True
        actual = mask_contour_points(mask, 5)
        self.assertTrue(np.all(mask[actual[:, 1], actual[:, 0]]))
        self.assertTrue(np.any(actual[:, 0] >= 48))
        self.assertFalse(np.any((actual[:, 0] >= 32) & (actual[:, 0] < 48)))
        self.assertFalse(np.any((actual[:, 0] >= 10) & (actual[:, 0] < 25)
                               & (actual[:, 1] >= 10) & (actual[:, 1] < 22)))

    def test_moving_the_mask_moves_the_whole_lattice(self):
        mask = np.zeros((40, 45), dtype=bool)
        mask[4:30, 7:35] = True
        shifted = np.pad(mask, ((9, 0), (6, 0)))
        np.testing.assert_array_equal(mask_contour_points(shifted, 13),
                                      mask_contour_points(mask, 13) + [6, 9])

    def test_empty_mask_has_no_points_and_binary_integer_masks_are_supported(self):
        self.assertEqual(mask_contour_points(np.zeros((20, 30), bool), 5).shape, (0, 2))
        mask = np.zeros((25, 30), np.uint8)
        mask[2:24, 7:28] = 1
        np.testing.assert_array_equal(mask_contour_points(mask, np.int64(5)),
                                      mask_contour_points(mask.astype(bool), 5))

    def test_bad_masks_and_grid_settings_are_rejected(self):
        for mask in (None, [], [0, 1], np.zeros((0, 5)), np.zeros((2, 3, 3)),
                     [[0, 2]], [[np.nan, 1]], [["0", "1"]]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                mask_contour_points(mask, 5)
        for spacing in (0, 1, -2, 5.0, True, np.bool_(False), math.inf):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                mask_contour_points(np.ones((10, 10), bool), spacing)
        for limit in (0, -1, 3.5, True, None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                mask_contour_points(np.ones((10, 10), bool), 5, max_points=limit)

    def test_limit_is_complete_or_error_and_exact_limit_is_allowed(self):
        mask = np.ones((30, 40), bool)
        expected = mask_contour_points(mask, 5)
        np.testing.assert_array_equal(mask_contour_points(mask, 5, max_points=len(expected)), expected)
        with self.assertRaisesRegex(ValueError, "More than"):
            mask_contour_points(mask, 5, max_points=len(expected)-1)

    def test_mask_sampling_cancels_before_returning_any_positions(self):
        mask = np.ones((100, 200), bool)
        with self.assertRaises(CancelledOperation):
            mask_contour_points(mask, 5, cancel=lambda: True)
        calls = [0]
        def cancel_in_rows():
            calls[0] += 1
            return calls[0] > 5
        with self.assertRaises(CancelledOperation):
            mask_contour_points(mask, 5, cancel=cancel_in_rows)
        with self.assertRaisesRegex(ValueError, "cancel"):
            mask_contour_points(mask, 5, cancel=True)


if __name__ == "__main__":
    unittest.main()
