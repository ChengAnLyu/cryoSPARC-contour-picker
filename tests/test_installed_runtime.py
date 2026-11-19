"""Installed-path, windowed launch, and noninteractive build-check regressions."""

import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import app


class RuntimePathTests(unittest.TestCase):
    def test_frozen_model_is_found_inside_bundle_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = base / "installed" / "_internal"
            model = bundle / "models" / app.DEFAULT_CHECKPOINT
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            unrelated = base / "other"
            unrelated.mkdir()
            (unrelated / app.DEFAULT_CHECKPOINT).write_bytes(b"unrelated")
            previous = Path.cwd()
            try:
                os.chdir(unrelated)
                with patch.object(sys, "frozen", True, create=True), \
                     patch.object(sys, "_MEIPASS", str(bundle), create=True), \
                     patch.object(sys, "executable", str(base / "installed" / "SARU Picker.exe")):
                    self.assertEqual(app.resolve_checkpoint().resolve(), model.resolve())
            finally:
                os.chdir(previous)

    def test_explicit_missing_checkpoint_is_not_replaced_by_bundled_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / "models" / app.DEFAULT_CHECKPOINT
            model.parent.mkdir()
            model.write_bytes(b"bundled")
            explicit = base / "requested-model.pth"
            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", str(base), create=True):
                self.assertEqual(app.resolve_checkpoint(explicit), explicit.resolve())
                self.assertFalse(app.resolve_checkpoint(explicit).exists())

    def test_source_and_executable_sidecar_model_locations_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / app.DEFAULT_CHECKPOINT
            model.write_bytes(b"sidecar")
            # Windows temp directories may use an 8.3 alias such as RUNNER~1.
            # Compare canonical locations while still requiring the exact file.
            with patch.object(sys, "frozen", False, create=True), \
                 patch.object(app, "__file__", str(base / "app.py")):
                self.assertEqual(app.resolve_checkpoint().resolve(), model.resolve())
            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", str(base / "_internal"), create=True), \
                 patch.object(sys, "executable", str(base / "SARU Picker.exe")):
                self.assertEqual(app.resolve_checkpoint().resolve(), model.resolve())

    def test_windows_state_uses_local_app_data_instead_of_install_folder(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sys, "platform", "win32"), \
             patch.dict(os.environ, {"LOCALAPPDATA": directory}), \
             patch.object(sys, "_MEIPASS", "/read-only-install/_internal", create=True):
            self.assertEqual(app.user_state_directory(), Path(directory) / "SARU_Picker")

    def test_relative_local_app_data_does_not_write_into_current_directory(self):
        with patch.object(sys, "platform", "win32"), \
             patch.dict(os.environ, {"LOCALAPPDATA": "relative-folder"}):
            result = app.user_state_directory()
            self.assertTrue(result.is_absolute())
            self.assertEqual(result, Path.home() / "AppData" / "Local" / "SARU_Picker")


class ReportAndConsoleTests(unittest.TestCase):
    def test_no_console_streams_are_valid_temporarily_and_restored(self):
        with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            with app.windowed_console_streams():
                output = sys.stdout
                error = sys.stderr
                self.assertGreaterEqual(output.fileno(), 0)
                print("windowed output")
                print("windowed error", file=error)
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)
            self.assertTrue(output.closed)
            self.assertTrue(error.closed)

    def test_report_serialization_failure_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            app.write_json_report(path, {"status": "ok"})
            with self.assertRaises(TypeError):
                app.write_json_report(path, {"invalid": object()})
            self.assertEqual(json.loads(path.read_text()), {"status": "ok"})
            self.assertEqual(list(Path(directory).glob(".saru-report-*")), [])

    def test_logging_works_without_console_even_when_other_handlers_exist(self):
        root = logging.getLogger()
        previous_handlers, previous_level = list(root.handlers), root.level
        for handler in previous_handlers:
            root.removeHandler(handler)
        root.addHandler(logging.NullHandler())
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(sys, "stderr", None):
                log_path = app.setup_logging(Path(directory), include_console=False)
                app.APP_LOG.info("installed-runtime-log-check")
                for handler in root.handlers:
                    handler.flush()
                self.assertIn("installed-runtime-log-check", log_path.read_text())
                self.assertFalse(any(getattr(h, "_saru_console", False) for h in root.handlers))
                for handler in list(root.handlers):
                    root.removeHandler(handler)
                    handler.close()
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.close()
            for handler in previous_handlers:
                root.addHandler(handler)
            root.setLevel(previous_level)


class SmokeAndLaunchTests(unittest.TestCase):
    def test_diagnostics_do_not_load_model_and_work_without_console(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
                 patch.object(app, "setup_logging", return_value=Path(directory) / "app.log"), \
                 patch.object(app, "SamRecognizer") as recognizer:
                result = app.main(["--diagnostics", "--report", str(report)])
            self.assertEqual(result, 0)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["mode"], "diagnostics")
            recognizer.assert_not_called()

    def test_smoke_failure_returns_nonzero_and_json_without_error_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "smoke.json"
            with patch.object(app, "setup_logging", return_value=Path(directory) / "app.log"), \
                 patch.object(app, "run_smoke_test", return_value={"status": "error", "error": "missing DLL"}), \
                 patch.object(app, "show_startup_error") as dialog, \
                 patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
                result = app.main(["--smoke-test", "--report", str(report)])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(report.read_text())["error"], "missing DLL")
            dialog.assert_not_called()

    def test_unexpected_headless_failure_also_creates_report_without_dialog(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "smoke.json"
            with patch.object(app, "setup_logging", return_value=Path(directory) / "app.log"), \
                 patch.object(app, "run_smoke_test", side_effect=RuntimeError("loader failed")), \
                 patch.object(app, "show_startup_error") as dialog, \
                 patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
                 self.assertLogs("saru_picker", level="ERROR"):
                result = app.main(["--smoke-test", "--report", str(report)])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(report.read_text())["error_type"], "RuntimeError")
            dialog.assert_not_called()

    def test_native_operation_failure_stops_before_model_check(self):
        with patch.object(sys, "platform", "linux"), \
             patch.object(app, "_smoke_image_processing", return_value={}), \
             patch.object(app, "_smoke_torchvision", side_effect=RuntimeError("torchvision::nms unavailable")), \
             patch.object(app, "_smoke_sam") as model, \
             self.assertLogs("saru_picker", level="ERROR"):
            result = app.run_smoke_test(Path("model.pth"), Path("app.log"))
        self.assertEqual(result["status"], "error")
        self.assertIn("torchvision_native", result["error"])
        model.assert_not_called()

    def test_windows_smoke_requires_tk_desktop_and_model_checks(self):
        with patch.object(sys, "platform", "win32"), \
             patch.object(app, "_smoke_image_processing", return_value={}), \
             patch.object(app, "_smoke_torchvision", return_value={}), \
             patch.object(app, "_smoke_windows_runtime", return_value={}) as windows, \
             patch.object(app, "_smoke_sam", return_value={}) as model:
            result = app.run_smoke_test(Path("model.pth"), Path("app.log"))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(all(item["status"] == "ok" for item in result["checks"]))
        windows.assert_called_once_with(Path("app.log"))
        model.assert_called_once_with(Path("model.pth"))
        self.assertNotIn("Metadata inspection only", result["note"])

    def test_offline_constructor_does_not_construct_or_start_global_hooks(self):
        root = Mock()
        args = SimpleNamespace(checkpoint=None, device="cpu", model_type="vit_b", image=None)
        def minimal_ui(picker):
            picker.status = SimpleNamespace(set=Mock())
        with patch.object(app.PickerApp, "_build_ui", minimal_ui), \
             patch.object(app, "TaskRunner") as runner, \
             patch.object(app, "DesktopAdapter"), \
             patch.object(app, "GlobalHotkeys") as hooks:
            picker = app.PickerApp(root, args, Path("app.log"), start_hotkeys=False)
            self.assertIsNone(picker.hotkeys)
            hooks.assert_not_called()
            picker.close()
        runner.return_value.shutdown.assert_called_once_with(wait=False)
        root.destroy.assert_called_once()

    def test_normal_constructor_retains_global_hotkey_start(self):
        root = Mock()
        args = SimpleNamespace(checkpoint=None, device="cpu", model_type="vit_b", image=None)
        def minimal_ui(picker):
            picker.status = SimpleNamespace(set=Mock())
        with patch.object(app.PickerApp, "_build_ui", minimal_ui), \
             patch.object(app, "TaskRunner"), \
             patch.object(app, "DesktopAdapter"), \
             patch.object(app, "GlobalHotkeys") as hooks:
            picker = app.PickerApp(root, args, Path("app.log"))
            hooks.return_value.start.assert_called_once()
            picker.close()
            hooks.return_value.stop.assert_called_once()

    def test_missing_model_never_constructs_recognizer(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "SamRecognizer") as model:
            with self.assertRaises(FileNotFoundError):
                app._smoke_sam(Path(directory) / "missing.pth")
            model.assert_not_called()

    def test_fatal_gui_startup_uses_existing_root_and_destroys_it(self):
        root = Mock()
        tkinter = SimpleNamespace(Tk=Mock(return_value=root))
        with patch.dict(sys.modules, {"tkinter": tkinter}), \
             patch.object(app, "setup_logging", return_value=Path("app.log")), \
             patch.object(app, "enable_dpi_awareness"), \
             patch.object(app, "ScreenPickerApp", side_effect=RuntimeError("initialization failed")), \
             patch.object(app, "show_startup_error") as dialog, \
             self.assertLogs("saru_picker", level="ERROR"):
            result = app.main([])
        self.assertEqual(result, 1)
        tkinter.Tk.assert_called_once()
        root.destroy.assert_called_once()
        self.assertIs(dialog.call_args.kwargs["parent"], root)

    def test_smoke_and_diagnostics_cannot_be_combined_to_skip_model_check(self):
        with patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                app.build_parser().parse_args(["--smoke-test", "--diagnostics"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
