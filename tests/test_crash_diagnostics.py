"""A native fault log must remain writable for the entire frozen launch."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_PATH = Path(__file__).resolve().parents[1] / "packaging" / "launcher.py"
_SPEC = importlib.util.spec_from_file_location("crash_test_launcher", _PATH)
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


class NativeFaultLogTests(unittest.TestCase):
    def test_native_fault_stream_survives_import_and_application_lifetime(self):
        seen = []
        def enable(*, file, all_threads):
            self.assertTrue(all_threads)
            self.assertFalse(file.closed)
            seen.append(file)
        def main():
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].closed)
            seen[0].write("synthetic native trace\n")
            seen[0].flush()
            return 0
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(launcher.os.environ, {"LOCALAPPDATA": directory}), \
             patch.dict(sys.modules, {"main": SimpleNamespace(main=main)}), \
             patch.object(launcher.faulthandler, "is_enabled", return_value=False), \
             patch.object(launcher.faulthandler, "enable", side_effect=enable), \
             patch.object(launcher.faulthandler, "disable") as disable:
            self.assertEqual(launcher.launch(), 0)
            self.assertIn("synthetic native trace", (Path(directory)/"SARU_Picker"/"native_fault.log").read_text())
        disable.assert_called_once()
        self.assertTrue(seen[0].closed)

    def test_existing_fault_handler_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(launcher.os.environ, {"LOCALAPPDATA": directory}), \
             patch.dict(sys.modules, {"main": SimpleNamespace(main=lambda: 0)}), \
             patch.object(launcher.faulthandler, "is_enabled", return_value=True), \
             patch.object(launcher.faulthandler, "enable") as enable, \
             patch.object(launcher.faulthandler, "disable") as disable:
            self.assertEqual(launcher.launch(), 0)
        enable.assert_not_called()
        disable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
