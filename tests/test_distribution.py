"""Check the actual single-file artifact and raw image boundary handling."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from app import load_rgb_file
from build_standalone import build


class DistributionTests(unittest.TestCase):
    def test_standalone_import_and_bitmap_sampling_work_without_gui_or_sam(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build(Path(directory) / "main.py")
            code = """
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location('picker_distribution', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'pyautogui' not in sys.modules
assert 'tkinter' not in sys.modules
mask=np.ones((20,30), dtype=bool)
mask[5:15,10:20]=False
points=module.sample_hex_grid(mask, 3)
assert len(points)>0
assert mask[points[:,1],points[:,0]].all()
assert module.CaptureGeometry(40,50,30,20).local_to_screen(4,6)==(44,56)
blank=np.full((64,96,3),125,np.uint8)
proposal=np.zeros((64,96),bool)
proposal[20:44,24:72]=True
assert module.filter_capsid_proposals(blank,[{'segmentation':proposal}])==[]
diagnostic=module._smoke_image_processing()
assert diagnostic['cross_contour_duplicate_guard']
assert diagnostic['capsid_uniform_scene_rejected']
"""
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_16bit_input_is_not_silently_saturated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.tif"
            Image.fromarray(np.array([[0,1000],[2000,4000]], dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "16-bit"):
                load_rgb_file(path)

    def test_greyscale_8bit_input_preserves_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            source=np.array([[0,90],[180,255]], dtype=np.uint8)
            Image.fromarray(source).save(path)
            actual=load_rgb_file(path)
            np.testing.assert_array_equal(actual, np.repeat(source[...,None],3,axis=2))


if __name__ == "__main__":
    unittest.main()
