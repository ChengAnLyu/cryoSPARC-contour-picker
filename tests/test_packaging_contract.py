"""Pre-import launcher behavior used by the real Windows packaging checks.

These tests do not claim to run Windows native libraries. They exercise the
small stdlib launcher before application imports can succeed or fail.
"""

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, call, patch

from PIL import Image

import application_icon


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


class ApplicationIconContractTests(unittest.TestCase):
    def test_public_artwork_is_the_generic_fallback_with_windows_sizes(self):
        ico, png = application_icon.application_icon_paths()
        self.assertEqual(hashlib.sha256(png.read_bytes()).hexdigest(),
                         "d824339e4c81ac13e967e82ed8f776b7a6fb4d74a8311dc0e81da95d6a9812e2")
        with Image.open(png) as original:
            self.assertEqual(original.size, (256, 256))
        with Image.open(ico) as icons:
            sizes = {(n, n) for n in (16, 24, 32, 48, 64, 128, 256)}
            self.assertEqual(icons.ico.sizes(), sizes)
            for size in sizes:
                frame = icons.ico.getimage(size).convert("RGBA")
                self.assertEqual(frame.size, size)
                self.assertIsNotNone(frame.getbbox())

    def test_installer_sidecars_are_optional_external_files_and_applied_before_launch(self):
        installer = (_LAUNCHER_PATH.parent / "installer.iss").read_text(encoding="utf-8")
        files = installer.split("[Files]", 1)[1].split("[Icons]", 1)[0]
        bundled_position = files.index('Source: "{#AppSourceDir}\\*"')
        for name in ("app-icon.png", "app-icon.ico"):
            source = 'Source: "{src}\\' + name + '"'
            line = next(line for line in files.splitlines() if line.startswith(source))
            self.assertGreater(files.index(line), bundled_position)
            self.assertIn('DestDir: "{app}\\_internal\\assets"', line)
            flags = line.split("Flags:", 1)[1].strip().split()
            self.assertIn("external", flags, "Private sidecars must not be embedded at build time.")
            self.assertIn("skipifsourcedoesntexist", flags)
            self.assertIn("ignoreversion", flags)
        run = installer.split("[Run]", 1)[1]
        commands = [line for line in run.splitlines() if line.startswith("Filename:")]
        self.assertIn('Filename: "{app}\\SARU Icon Helper.exe"', commands[0])
        self.assertIn('--icon ""{app}\\_internal\\assets\\app-icon.ico""', commands[0])
        self.assertIn('--report ""{localappdata}\\SARU_Picker\\icon-install.json""', commands[0])
        flags = commands[0].split("Flags:", 1)[1].strip().split()
        self.assertEqual(set(flags), {"runhidden", "waituntilterminated"})
        self.assertIn('Filename: "{app}\\SARU Picker.exe"', commands[1])
        icon_path = "{app}\\_internal\\assets\\app-icon.ico"
        self.assertIn("UninstallDisplayIcon=" + icon_path, installer)
        shortcuts = installer.split("[Icons]", 1)[1].split("[Run]", 1)[0]
        self.assertEqual(shortcuts.count('IconFilename: "' + icon_path + '"'), 2)

    def test_frozen_windows_uses_bundled_ico_without_creating_a_tk_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            assets = bundle / "assets"
            assets.mkdir()
            ico = assets / "app-icon.ico"
            ico.write_bytes(b"bundled icon placeholder")
            root = SimpleNamespace(iconbitmap=Mock(), iconphoto=Mock())
            photo_factory = Mock(side_effect=AssertionError("No Tk image should be needed."))
            with patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "_MEIPASS", str(bundle), create=True), \
                 patch.object(sys, "platform", "win32"), \
                 patch.dict(sys.modules, {"tkinter": SimpleNamespace(PhotoImage=photo_factory)}):
                self.assertEqual(application_icon.application_icon_paths()[0], ico)
                self.assertTrue(application_icon.apply_application_icon(root))
            self.assertEqual(root.iconbitmap.call_args_list,
                             [call(str(ico)), call(default=str(ico))])
            self.assertEqual(root._saru_application_icon_path, str(ico))
            root.iconphoto.assert_not_called()
            photo_factory.assert_not_called()

    def test_native_class_icons_are_accepted_after_custom_artwork_was_applied(self):
        ico, _ = application_icon.application_icon_paths()
        root = SimpleNamespace(winfo_id=lambda: 111,
                               _saru_application_icon_path=str(ico))
        api = SimpleNamespace(GetAncestor=Mock(return_value=222),
                              SendMessageW=Mock(return_value=0),
                              GetClassLongPtrW=Mock(side_effect=lambda hwnd, which: 2**40-which))
        with patch.object(sys, "platform", "win32"), \
             patch.object(application_icon.ctypes, "WinDLL", return_value=api, create=True):
            self.assertEqual(application_icon.windows_application_icon_status(root),
                             {"small_icon": True, "large_icon": True})
        self.assertEqual(api.GetClassLongPtrW.call_args_list, [call(222, -34), call(222, -14)])
        self.assertEqual(api.GetClassLongPtrW.restype, application_icon.ctypes.c_size_t)

    def test_native_class_icons_do_not_validate_an_unconfigured_tk_window(self):
        root = SimpleNamespace(winfo_id=lambda: 111)
        api = SimpleNamespace(GetAncestor=Mock(return_value=222),
                              SendMessageW=Mock(return_value=0),
                              GetClassLongPtrW=Mock(return_value=555))
        with patch.object(sys, "platform", "win32"), \
             patch.object(application_icon.ctypes, "WinDLL", return_value=api, create=True):
            self.assertEqual(application_icon.windows_application_icon_status(root),
                             {"small_icon": False, "large_icon": False})

    def test_native_explicit_icons_do_not_need_class_fallback(self):
        ico, _ = application_icon.application_icon_paths()
        root = SimpleNamespace(winfo_id=lambda: 111,
                               _saru_application_icon_path=str(ico))
        api = SimpleNamespace(GetAncestor=Mock(return_value=222),
                              SendMessageW=Mock(return_value=2**40),
                              GetClassLongPtrW=Mock(side_effect=AssertionError("Unexpected class lookup.")))
        with patch.object(sys, "platform", "win32"), \
             patch.object(application_icon.ctypes, "WinDLL", return_value=api, create=True):
            self.assertEqual(application_icon.windows_application_icon_status(root),
                             {"small_icon": True, "large_icon": True})
        api.GetClassLongPtrW.assert_not_called()

    def test_png_fallback_releases_tk_image_on_the_calling_thread(self):
        references = []
        calls = []

        class Photo:
            pass

        def photo_factory(**kwargs):
            photo = Photo()
            references.append(weakref.ref(photo))
            return photo

        root = SimpleNamespace(iconphoto=lambda default, photo: calls.append(default))
        with patch.object(sys, "platform", "linux"), \
             patch.dict(sys.modules, {"tkinter": SimpleNamespace(PhotoImage=photo_factory)}):
            self.assertTrue(application_icon.apply_application_icon(root))
        self.assertEqual(calls, [True])
        self.assertEqual(len(references), 1)
        self.assertIsNone(references[0](), "Window icon setup must not retain a Tk image for worker GC.")

    def test_standalone_without_optional_artwork_still_starts(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(application_icon, "__file__", str(Path(directory) / "main.py")), \
             patch.object(sys, "frozen", False, create=True), \
             self.assertLogs(application_icon.APPLICATION_ICON_LOG, level="WARNING"):
            self.assertFalse(application_icon.apply_application_icon(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
