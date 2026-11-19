"""Pre-import launcher behavior used by the real Windows packaging checks.

These tests do not claim to run Windows native libraries. They exercise the
small stdlib launcher before application imports can succeed or fail.
"""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


_LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "packaging" / "launcher.py"
_SPEC = importlib.util.spec_from_file_location("saru_packaging_launcher", _LAUNCHER_PATH)
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)


class LauncherContractTests(unittest.TestCase):
    def test_import_failure_reports_every_headless_mode_without_a_modal_dialog(self):
        for mode in ("--smoke-test", "--diagnostics", "--segment-image"):
            for equals_form in (False, True):
                with self.subTest(mode=mode, equals_form=equals_form), \
                     tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    report = base / "report folder" / "result.json"
                    report_args = ([f"--report={report}"] if equals_form
                                   else ["--report", str(report)])
                    argv = ["SARU Picker.exe", mode]
                    if mode == "--segment-image":
                        argv.append(str(base / "input image.png"))
                    argv.extend(report_args)
                    message_box = Mock()
                    windows = SimpleNamespace(user32=SimpleNamespace(MessageBoxW=message_box))
                    # A None module entry forces the static import to fail
                    # before app.main and its own report handler can run.
                    with patch.dict(sys.modules, {"main": None}), \
                         patch.object(sys, "argv", argv), \
                         patch.object(sys, "platform", "win32"), \
                         patch.object(sys, "stdout", io.StringIO()), \
                         patch.object(sys, "stderr", io.StringIO()), \
                         patch.dict(launcher.os.environ, {"LOCALAPPDATA": str(base)}), \
                         patch.object(launcher.ctypes, "windll", windows, create=True):
                        result = launcher.launch()
                    self.assertEqual(result, 1)
                    message_box.assert_not_called()
                    self.assertTrue(report.is_file(), "Import failures must leave the requested JSON report.")
                    payload = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(payload["status"], "error")
                    self.assertEqual(payload["stage"], "startup")

    def test_windowed_launch_supplies_streams_before_importing_the_application(self):
        observed = {}
        real_import = __import__

        def import_dependency(name, *args, **kwargs):
            if name != "main":
                return real_import(name, *args, **kwargs)
            observed["stdout"] = sys.stdout
            observed["stderr"] = sys.stderr
            # Native libraries and progress helpers may call flush or fileno.
            sys.stdout.flush()
            sys.stderr.flush()
            self.assertGreaterEqual(sys.stdout.fileno(), 0)
            self.assertGreaterEqual(sys.stderr.fileno(), 0)
            return SimpleNamespace(main=lambda: 0)

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(launcher.os.environ, {"LOCALAPPDATA": directory}), \
             patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
             patch("builtins.__import__", side_effect=import_dependency):
            try:
                self.assertEqual(launcher.launch(), 0)
            finally:
                for stream in {observed.get("stdout"), observed.get("stderr")} - {None}:
                    stream.close()
        self.assertIn("stdout", observed)
        self.assertIn("stderr", observed)


if __name__ == "__main__":
    unittest.main()
