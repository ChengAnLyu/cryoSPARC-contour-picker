"""Synthetic invariants for the untrained capsid filter, not accuracy claims."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capsid import filter_capsid_proposals


def synthetic_core():
    """Dense polygon inside a quiet envelope, alongside a membrane-only oval."""
    rng = np.random.default_rng(48)
    shape = (192, 256)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    envelope = ((xx - 82) / 66) ** 2 + ((yy - 96) / 72) ** 2 <= 1
    negative = ((xx - 204) / 26) ** 2 + ((yy - 80) / 32) ** 2 <= 1
    canvas = Image.new("1", (shape[1], shape[0]))
    ImageDraw.Draw(canvas).polygon([(55, 70), (95, 60), (120, 87),
                                    (108, 127), (70, 132), (45, 101)], fill=1)
    core = np.asarray(canvas, dtype=bool)
    gray = 125 + rng.normal(0, 5, shape)
    gray[core] = 125 + rng.normal(0, 45, np.count_nonzero(core))
    # The membrane has a coherent sharp edge, without a dense interior.
    radius = ((xx - 204) / 26) ** 2 + ((yy - 80) / 32) ** 2
    gray[(radius > .88) & (radius <= 1)] = 30
    rgb = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    proposals = [{"segmentation": mask, "predicted_iou": .98, "stability_score": .97}
                 for mask in (envelope, negative, np.ones(shape, bool))]
    return rgb, core, proposals


class CapsidFilterTests(unittest.TestCase):
    def test_dense_interior_survives_while_empty_membrane_and_whole_frame_do_not(self):
        rgb, core, proposals = synthetic_core()
        outputs = filter_capsid_proposals(rgb, proposals)
        self.assertEqual(len(outputs), 1)
        selected = outputs[0]["segmentation"]
        self.assertGreater(np.count_nonzero(selected & core), .75 * np.count_nonzero(core))
        # Unlike the user's sparse circles, this synthetic polygon has a known
        # complete area: selecting most of its empty envelope is a real error.
        self.assertGreater(np.count_nonzero(selected & core), .70 * np.count_nonzero(selected))
        self.assertFalse(np.any(selected & proposals[1]["segmentation"]))
        self.assertFalse(np.any(selected & ~proposals[0]["segmentation"]))

    def test_refined_mask_does_not_inherit_original_sam_quality_estimates(self):
        rgb, _, proposals = synthetic_core()
        output = filter_capsid_proposals(rgb, proposals)[0]
        self.assertIsNone(output["predicted_iou"])
        self.assertIsNone(output["stability_score"])
        self.assertEqual(output["capsid_evidence"]["proposal_predicted_iou"], .98)
        self.assertFalse(output["capsid_evidence"]["is_calibrated_probability"])

    def test_original_mask_holes_and_coloured_screen_marks_are_excluded(self):
        rgb, _, proposals = synthetic_core()
        proposals[0]["segmentation"][88:103, 75:90] = False
        rgb[104:112, 92:102] = [0, 255, 0]
        outputs = filter_capsid_proposals(rgb, proposals)
        self.assertTrue(outputs)
        selected = outputs[0]["segmentation"]
        self.assertFalse(selected[88:103, 75:90].any())
        self.assertFalse(selected[104:112, 92:102].any())

    def test_constant_and_blank_screens_have_no_dense_cores(self):
        mask = np.zeros((90, 120), bool)
        mask[20:70, 30:90] = True
        for gray in (0, 125, 255):
            with self.subTest(gray=gray):
                rgb = np.full((90, 120, 3), gray, np.uint8)
                self.assertEqual(filter_capsid_proposals(rgb, [{"segmentation": mask}]), [])

    def test_wrong_shape_fails_instead_of_moving_coordinates(self):
        with self.assertRaisesRegex(ValueError, "different size"):
            filter_capsid_proposals(np.zeros((90, 120, 3), np.uint8),
                                   [{"segmentation": np.ones((120, 90), bool)}])

    def test_cancellation_propagates_before_any_result(self):
        rgb, _, proposals = synthetic_core()
        calls = 0
        def stop():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise InterruptedError("cancelled")
        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            filter_capsid_proposals(rgb, proposals, check_cancel=stop)

    def test_duplicate_proposals_do_not_duplicate_a_core(self):
        rgb, _, proposals = synthetic_core()
        outputs = filter_capsid_proposals(rgb, [proposals[0], proposals[0]])
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
