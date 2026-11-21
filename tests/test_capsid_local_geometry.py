"""Compare cropped geometry against independent full-image calculations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capsid


class LocalPromptGeometryTests(unittest.TestCase):
    def test_prompt_points_match_full_image_distances_at_every_frame_edge(self):
        yy, xx = np.mgrid[:260, :420]
        centers = [(170, 130), (0, 130), (419, 130), (170, 0), (170, 259),
                   (0, 0), (419, 259), (0, 259), (419, 0)]
        for x, y in centers:
            with self.subTest(center=(x, y)):
                mask = ((xx - x) / 50) ** 2 + ((yy - y) / 65) ** 2 < 1
                # A hole and asymmetric scores exercise nearest-zero distances
                # and row-major tie handling beyond a symmetric filled ellipse.
                mask[(xx - x - 12) ** 2 + (yy - y + 4) ** 2 < 6 ** 2] = False
                scores = np.where(mask, .8 + (xx % 19) / 100., 0).astype(np.float32)
                distances = ndi.distance_transform_edt(mask)
                cy, cx = np.unravel_index(np.argmax(distances * scores), mask.shape)
                with patch.object(capsid, '_profile_components', return_value=iter([(mask, {})])), \
                     patch.object(capsid.ndi, 'distance_transform_edt', wraps=ndi.distance_transform_edt) as edt:
                    actual = capsid._capsid_guided_prompts_at_threshold(scores)
                self.assertEqual(actual[0]['point'], [cx, cy])
                self.assertEqual(actual[0]['support_area'], int(mask.sum()))
                self.assertLess(edt.call_args.args[0].size, mask.size / 2)

    def test_local_annulus_has_the_same_texture_evidence_as_full_image_dilation(self):
        shape = (360, 520)
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        rng = np.random.default_rng(290)
        for x, y in [(240, 180), (0, 180), (519, 180), (240, 0), (240, 359)]:
            with self.subTest(center=(x, y)):
                core = ((xx - x) / 65) ** 2 + ((yy - y) / 80) ** 2 < 1
                parent = ndi.binary_dilation(core, iterations=8)
                scores = np.where(core, .97, 0).astype(np.float32)
                texture = rng.uniform(1, 3, shape).astype(np.float32)
                texture[core] += 5
                valid = np.ones(shape, bool)
                valid[150:155, 100:180] = False
                rgb = np.full((*shape, 3), 128, np.uint8)
                with patch.object(capsid, '_capsid_texture', return_value=(texture, valid)), \
                     patch.object(capsid, 'polygonal_capsid_mask', side_effect=lambda mask, **kw: (mask, {})):
                    records = capsid.filter_capsid_with_profile(rgb, [{'segmentation': parent}], scores)
                self.assertEqual(len(records), 1)
                mask = records[0]['segmentation']
                full_annulus = (ndi.binary_dilation(mask, iterations=20)
                                & ~ndi.binary_dilation(mask, iterations=3) & valid)
                expected = float(np.median(texture[mask])) / float(np.median(texture[full_annulus]))
                self.assertEqual(records[0]['capsid_evidence']['local_texture_ratio'], expected)

    def test_overlap_decisions_match_full_bitmap_intersections(self):
        rng = np.random.default_rng(291)
        shape = (300, 600)
        for _ in range(30):
            masks = []
            for shift in rng.integers(0, 50, size=2):
                mask = np.zeros(shape, bool)
                mask[90:180, 170 + shift:270 + shift] = True
                mask[120:140, 200 + shift:210 + shift] = False
                masks.append(mask)
            records = [{'segmentation': mask, 'area': int(mask.sum()),
                        'bbox': capsid._capsid_bbox(mask)} for mask in masks]
            expected = np.count_nonzero(masks[0] & masks[1]) / min(r['area'] for r in records) >= .8
            self.assertEqual(capsid._profile_masks_overlap(*records), expected)


if __name__ == '__main__':
    unittest.main()
