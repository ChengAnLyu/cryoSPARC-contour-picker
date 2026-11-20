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


def layout_widget(parent=None, *, x=0, y=0, width=100, height=30,
                  requested=None, visible=True, manager="pack"):
    """Measured widget fixture; requested size can exceed its allocated space."""
    widget = Mock()
    widget.master = parent
    widget._children = []
    if parent is not None:
        parent._children.append(widget)
    widget.winfo_children.side_effect = lambda: list(widget._children)
    widget.winfo_width.return_value = width
    widget.winfo_height.return_value = height
    widget.winfo_reqwidth.return_value = width if requested is None else requested[0]
    widget.winfo_reqheight.return_value = height if requested is None else requested[1]
    widget.winfo_rootx.return_value = x
    widget.winfo_rooty.return_value = y
    widget.winfo_viewable.return_value = visible
    widget.winfo_ismapped.return_value = visible
    widget.winfo_geometry.return_value = f"{width}x{height}{x:+d}{y:+d}"
    widget.winfo_manager.return_value = manager
    widget.winfo_class.return_value = "Button"
    return widget


class PanelLayoutValidationTests(unittest.TestCase):
    def test_native_map_and_resize_events_are_dispatched_before_measurements(self):
        root = layout_widget(width=500, height=500)
        root.state.return_value = "normal"
        root.attributes.return_value = True
        root.tk.getboolean.side_effect = bool
        footer = layout_widget(root, x=0, y=440, width=500, height=60, visible=False)
        button = layout_widget(footer, x=200, y=450, width=100, height=1,
                               requested=(100, 30), visible=False)
        status = layout_widget(root, x=10, y=300, width=480, height=80)
        calls = []
        def dispatch():
            calls.append("update")
            for widget in (footer, button):
                widget.winfo_viewable.return_value = True
                widget.winfo_ismapped.return_value = True
            if "fit" in calls:
                button.winfo_height.return_value = 30
        root.update.side_effect = dispatch
        picker = SimpleNamespace(root=root, fit_panel=Mock(side_effect=lambda: calls.append("fit")),
                                 terminate_button=button, status_label=status, spacing_slider=Mock())
        # Idle updates do not model the pending native map/resize messages.
        root.update_idletasks()
        self.assertFalse(footer.winfo_viewable())
        with patch.object(app, "_smoke_terminate_text", return_value={}), \
             patch.object(app, "_smoke_scale_label", return_value={}):
            result = app._smoke_panel_layout(picker, "manual", "default")
        self.assertEqual(calls, ["update", "fit", "update"])
        self.assertTrue(result["all_managed_controls_fit"])
        root.withdraw.assert_not_called()
        root.deiconify.assert_not_called()
        root.lift.assert_not_called()
        self.assertTrue(all(not call.args for call in root.state.call_args_list))

    def test_mapping_failure_includes_actionable_native_geometry(self):
        root = layout_widget(x=30, y=40, width=420, height=320)
        root.state.return_value = "normal"
        footer = layout_widget(root, x=30, y=331, width=420, height=1,
                               requested=(420, 49), visible=False)
        with self.assertRaisesRegex(RuntimeError, "not visible") as error:
            app._smoke_widget_layout(root, footer, "manual/default Frame")
        details = json.loads(str(error.exception).split("layout=", 1)[1])
        self.assertEqual(details["state"], "normal")
        self.assertEqual(details["panel"]["actual"], [420, 320])
        self.assertEqual(details["widget"]["actual"], [420, 1])
        self.assertEqual(details["widget"]["requested"], [420, 49])
        self.assertFalse(details["widget"]["mapped"])
        self.assertFalse(details["widget"]["viewable"])
        self.assertEqual(details["widget"]["geometry"], "420x1+30+331")

    def test_shortened_terminate_fails_even_when_bottom_edge_fits_window(self):
        root = layout_widget(width=420, height=320)
        footer = layout_widget(root, x=0, y=291, width=420, height=29)
        button = layout_widget(footer, x=155, y=311, width=110, height=9,
                               requested=(110, 29))
        # This is the old smoke check that passed the user's clipped layout.
        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(),
                             root.winfo_rooty() + root.winfo_height())
        with self.assertRaisesRegex(RuntimeError, "clipped below its requested size"):
            app._smoke_widget_layout(root, button, "manual Terminate")

    def test_full_size_button_must_also_fit_every_ancestor(self):
        root = layout_widget(width=420, height=320)
        footer = layout_widget(root, x=0, y=290, width=420, height=30)
        button = layout_widget(footer, x=155, y=285, width=110, height=29)
        with self.assertRaisesRegex(RuntimeError, "extends outside its parent"):
            app._smoke_widget_layout(root, button, "Terminate")

    def test_fully_packed_button_reports_actual_and_requested_click_area(self):
        root = layout_widget(x=-900, y=100, width=630, height=550)
        footer = layout_widget(root, x=-900, y=585, width=630, height=65)
        button = layout_widget(footer, x=-670, y=595, width=170, height=45,
                               requested=(160, 40))
        result = app._smoke_widget_layout(root, button, "Terminate")
        self.assertEqual(result["actual"], [170, 45])
        self.assertEqual(result["requested"], [160, 40])
        self.assertEqual(result["panel_rect"], [230, 495, 170, 45])
        self.assertTrue(result["full_requested_size"])

    def test_text_line_and_baseline_must_fit_button_padding(self):
        root = layout_widget()
        root.winfo_pixels.side_effect = int
        button = layout_widget(root, width=120, height=34)
        settings = {"font": "Arial 10", "text": "terminate", "borderwidth": 2,
                    "highlightthickness": 1, "padx": 4, "pady": 2}
        button.cget.side_effect = settings.__getitem__
        font = Mock()
        font.measure.return_value = 80
        font.metrics.side_effect = {"ascent": 15, "descent": 4, "linespace": 20}.__getitem__
        with patch.object(app.importlib, "import_module", return_value=SimpleNamespace(Font=Mock(return_value=font))):
            result = app._smoke_terminate_text(root, button)
            self.assertEqual(result["usable_text_size"], [106, 24])
            self.assertEqual(result["centered_first_baseline_from_top"], 22)
            self.assertTrue(result["full_font_line_fits"])
            # A fixed requested widget height alone cannot validate its font.
            button.winfo_height.return_value = 27
            with self.assertRaisesRegex(RuntimeError, "font baseline is clipped"):
                app._smoke_terminate_text(root, button)

    def test_managed_invisible_controls_fail_but_inactive_mode_groups_are_skipped(self):
        root = layout_widget(width=500, height=500)
        root.state.return_value = "normal"
        root.attributes.return_value = True
        root.tk.getboolean.side_effect = bool
        footer = layout_widget(root, x=0, y=440, width=500, height=60)
        button = layout_widget(footer, x=200, y=450, width=100, height=30)
        status = layout_widget(root, x=10, y=300, width=480, height=80)
        inactive = layout_widget(root, manager="", visible=False)
        layout_widget(inactive, height=1, requested=(100, 40), visible=False)
        picker = SimpleNamespace(root=root, fit_panel=Mock(), terminate_button=button,
                                 status_label=status, spacing_slider=Mock())
        with patch.object(app, "_smoke_terminate_text", return_value={}), \
             patch.object(app, "_smoke_scale_label", return_value={"checked": True}) as caption:
            result = app._smoke_panel_layout(picker, "manual", "multiline_status")
            self.assertEqual(result["managed_controls_checked"], 3)
            self.assertEqual(result["spacing_caption"], {"checked": True})
            caption.assert_called_once_with(root, picker.spacing_slider)
            picker.fit_panel.assert_called_once()
            # Pack-managed controls can become unmapped when siblings take all
            # available space. Do not silently omit that failure from coverage.
            layout_widget(root, x=0, y=400, height=1, requested=(100, 30), visible=False)
            with self.assertRaisesRegex(RuntimeError, "not visible"):
                app._smoke_panel_layout(picker, "manual", "multiline_status")

    def test_scale_caption_can_clip_even_when_widget_has_full_requested_size(self):
        root = layout_widget(width=840, height=640)
        root.winfo_pixels.side_effect = int
        slider = layout_widget(root, width=154, height=100, requested=(154, 100))
        settings = {"font": "TkDefaultFont", "label": "Particle spacing",
                    "borderwidth": 1, "highlightthickness": 1}
        slider.cget.side_effect = settings.__getitem__
        font = Mock()
        font.measure.return_value = 175
        self.assertTrue(app._smoke_widget_layout(root, slider, "spacing")["full_requested_size"])
        with patch.object(app.importlib, "import_module", return_value=SimpleNamespace(Font=Mock(return_value=font))):
            with self.assertRaisesRegex(RuntimeError, "Particle spacing caption is clipped"):
                app._smoke_scale_label(root, slider)
            slider.winfo_width.return_value = 304
            result = app._smoke_scale_label(root, slider)
        self.assertEqual(result["text"], "Particle spacing")
        self.assertEqual(result["text_width"], 175)
        self.assertEqual(result["available_width"], 300)
        self.assertTrue(result["full_label_width_fits"])

    def test_withdrawn_or_non_topmost_panel_fails(self):
        root = layout_widget(width=420, height=320)
        root.state.return_value = "withdrawn"
        picker = SimpleNamespace(root=root, fit_panel=Mock())
        with self.assertRaisesRegex(RuntimeError, "not visible"):
            app._smoke_panel_layout(picker, "manual", "default")
        root.state.return_value = "normal"
        root.attributes.return_value = False
        root.tk.getboolean.side_effect = bool
        with self.assertRaisesRegex(RuntimeError, "not kept above"):
            app._smoke_panel_layout(picker, "manual", "default")


class PanelCaptureValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ui" / "panel-125-manual.png"
        self.window = layout_widget(x=-192, y=61, width=509, height=401)
        self.window.state.return_value = "normal"
        self.window.cget.return_value = "SystemButtonFace"
        self.window.winfo_rgb.return_value = (240 * 257,) * 3
        self.grab, self.flush = Mock(), Mock(return_value=0)
        self.grab.grab.side_effect = [self.painted_frame(), self.painted_frame()]
        import ctypes
        patches = [patch.object(app.importlib, "import_module", return_value=self.grab),
                   patch.object(app, "control_panel_bounds", return_value=(-200, 30, 525, 440)),
                   patch.object(app.time, "sleep"),
                   patch.object(ctypes, "windll", SimpleNamespace(
                       dwmapi=SimpleNamespace(DwmFlush=self.flush)), create=True)]
        self.imports, self.bounds, self.sleep, _ = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    @staticmethod
    def painted_frame(*, controls=True):
        # Deliberately dark nonclient pixels must be excluded from validation.
        frame = app.Image.new("RGB", (525, 440), (20, 20, 20))
        draw = app.ImageDraw.Draw(frame)
        draw.rectangle((8, 31, 516, 431), fill=(240, 240, 240))
        if controls:
            draw.rectangle((210, 50, 310, 66), fill=(20, 20, 20))
            for x in (40, 190, 340):
                draw.rectangle((x, 100, x+100, 140), fill=(255, 255, 255), outline=(20, 20, 20))
                draw.rectangle((x+25, 114, x+75, 125), fill=(20, 20, 20))
            draw.rectangle((190, 365, 330, 405), fill=(235, 235, 235), outline=(20, 20, 20))
        return frame

    def test_two_painted_captures_validate_actual_client_without_repair_or_pixel_identity(self):
        second = self.painted_frame()
        # Harmless text/raster variation between frames must not fail the check.
        app.ImageDraw.Draw(second).rectangle((230, 50, 250, 66), fill=(240, 240, 240))
        self.grab.grab.side_effect = [self.painted_frame(), second]
        result = app._smoke_panel_screenshot(self.window, self.path)
        self.imports.assert_called_once_with("PIL.ImageGrab")
        self.assertEqual(self.flush.call_count, 2)
        self.assertEqual(self.grab.grab.call_count, 2)
        for call in self.grab.grab.call_args_list:
            self.assertEqual(call.kwargs, {"bbox": (-200, 30, 325, 470),
                                          "include_layered_windows": True, "all_screens": True})
        self.assertEqual(result["native_outer_bbox"], [-200, 30, 325, 470])
        rendered = result["render_check"]
        self.assertEqual(rendered["client_bbox"], [-192, 61, 317, 462])
        self.assertEqual(rendered["frames_checked"], 2)
        self.assertEqual(rendered["status"], "ok")
        self.assertEqual(rendered["background_rgb"], [240, 240, 240])
        self.assertNotEqual(rendered["frames"][0]["varied_fraction"],
                            rendered["frames"][1]["varied_fraction"])
        for frame in rendered["frames"]:
            self.assertTrue(Path(frame["path"]).is_file())
            self.assertEqual(frame["image_dimensions"], [525, 440])
        with app.Image.open(self.path) as saved:
            self.assertEqual(saved.tobytes(), second.tobytes())
        self.assertAlmostEqual(sum(call.args[0] for call in self.sleep.call_args_list), .1)
        self.window.withdraw.assert_not_called()
        self.window.deiconify.assert_not_called()
        self.window.lift.assert_not_called()
        self.window.geometry.assert_not_called()
        self.window.attributes.assert_not_called()
        self.assertTrue(all(not call.args for call in self.window.state.call_args_list))

    def test_unpainted_first_client_is_retained_and_rejected_without_retry_to_good_frame(self):
        unpainted = app.Image.new("RGB", (525, 440), (0, 0, 0))
        self.grab.grab.side_effect = [unpainted, self.painted_frame()]
        with self.assertRaisesRegex(RuntimeError, "capture 1 does not show a fully painted client"):
            app._smoke_panel_screenshot(self.window, self.path)
        self.assertEqual(self.grab.grab.call_count, 1)
        self.assertFalse(self.path.exists())
        failure = self.path.with_name(self.path.stem + "-frame-1.png")
        with app.Image.open(failure) as saved:
            self.assertEqual(saved.tobytes(), unpainted.tobytes())

    def test_second_capture_with_unpainted_bottom_strip_fails_despite_high_total_background(self):
        partial = self.painted_frame()
        app.ImageDraw.Draw(partial).rectangle((8, 412, 516, 431), fill=(0, 0, 0))
        self.grab.grab.side_effect = [self.painted_frame(), partial]
        with self.assertRaisesRegex(RuntimeError, "capture 2 does not show a fully painted client") as error:
            app._smoke_panel_screenshot(self.window, self.path)
        evidence = json.loads(str(error.exception).split("controls: ", 1)[1])
        self.assertGreater(evidence["background_fraction"], .65)
        self.assertEqual(evidence["edge_background_fraction"]["bottom"], 0)
        with app.Image.open(self.path) as saved:
            self.assertEqual(saved.tobytes(), partial.tobytes())

    def test_uniform_client_background_without_visible_controls_fails(self):
        self.grab.grab.side_effect = [self.painted_frame(controls=False)]
        with self.assertRaisesRegex(RuntimeError, "does not show a fully painted client") as error:
            app._smoke_panel_screenshot(self.window, self.path)
        evidence = json.loads(str(error.exception).split("controls: ", 1)[1])
        self.assertEqual(evidence["background_fraction"], 1)
        self.assertEqual(evidence["varied_fraction"], 0)

    def test_compositor_failure_stops_before_capture_instead_of_claiming_synchronization(self):
        self.flush.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "DwmFlush failed"):
            app._smoke_panel_screenshot(self.window, self.path)
        self.grab.grab.assert_not_called()

    def test_client_rectangle_outside_capture_is_rejected_not_padded_into_valid_image(self):
        self.window.winfo_rootx.return_value = -205
        with self.assertRaisesRegex(RuntimeError, "client rectangle is outside"):
            app._smoke_panel_screenshot(self.window, self.path)


class WindowsScaleCoverageTests(unittest.TestCase):

    def test_native_smoke_constructs_each_scale_before_widgets_and_checks_every_mode(self):
        windows, pickers, constructions, measured = [], [], [], []
        original_scaling = 1.5

        def make_window():
            window = Mock()
            window._scale, window._state = original_scaling, "normal"
            def tk_call(*args):
                if args[:2] == ("tk", "scaling"):
                    if len(args) == 3:
                        window._scale = args[2]
                    return window._scale
                self.assertEqual(args, ("info", "patchlevel"))
                return "8.6.14"
            def state(value=None):
                if value is not None:
                    window._state = value
                return window._state
            window.tk.call.side_effect = tk_call
            window.tk.getboolean.side_effect = bool
            window.state.side_effect = state
            window.withdraw.side_effect = lambda: state("withdrawn")
            window.winfo_viewable.side_effect = lambda: window._state == "normal"
            window.attributes.return_value = True
            windows.append(window)
            return window

        def make_picker(window, args, log_path, checkpoint, *, start_listeners):
            self.assertFalse(start_listeners)
            constructions.append(window._scale)
            picker = SimpleNamespace(root=window, mode="manual", hotkeys=None,
                                     mouse_observer=None, overlay=None, disabled=Mock(),
                                     status=Mock(), fit_panel=Mock(), _hide_overlays=Mock(), close=Mock())
            picker.disabled.get.return_value = True
            picker._status = "Manual drawing is ready."
            picker.status.get.side_effect = lambda: picker._status
            picker.status.set.side_effect = lambda value: setattr(picker, "_status", value)
            def set_mode(mode):
                picker.mode = mode
                picker._status = "Select the screen area, then detect or start auto."
            picker.set_mode = Mock(side_effect=set_mode)
            pickers.append(picker)
            return picker

        def measure(picker, mode, status_case):
            measured.append((picker.root._scale, mode, status_case, picker.status.get()))
            return {"panel_size": [round(420 * picker.root._scale), 500],
                    "terminate": {"full_requested_size": True}}

        modules = {"tkinter": SimpleNamespace(Tk=make_window),
                   "pyautogui": SimpleNamespace(click=Mock()),
                   "pynput.keyboard": SimpleNamespace(Key=SimpleNamespace(esc=object())),
                   "pynput.mouse": SimpleNamespace(Button=SimpleNamespace(left=object()))}
        def verify_overlay(window):
            self.assertEqual(window.state(), "normal")
            self.assertTrue(window.winfo_viewable())
            return {"checked": True}
        with patch.object(app.importlib, "import_module", side_effect=modules.__getitem__), \
             patch.object(app, "enable_dpi_awareness") as dpi, \
             patch.object(app, "resolve_checkpoint", return_value=Path("model.pth")), \
             patch.object(app, "ScreenPickerApp", side_effect=make_picker), \
             patch.object(app, "_smoke_panel_layout", side_effect=measure), \
             patch.object(app, "_smoke_advanced_settings", return_value={"status":"ok"}) as settings, \
             patch.object(app, "windows_application_icon_status", return_value={"small_icon": True, "large_icon": True}), \
             patch.object(app, "_smoke_repeated_screen_picks", return_value={"contours_completed": 3}) as repeated, \
             patch.object(app, "_smoke_micrograph_cycle", return_value={"continuous_auto": {"down_keys": 3}}) as cycle, \
             patch.object(app, "_smoke_manual_strokes", return_value={"strokes_completed": 2}) as manual, \
             patch.object(app, "verify_screen_overlay", side_effect=verify_overlay) as overlay:
            result = app._smoke_windows_runtime(Path("app.log"))
        expected_scales = [96 * percent / 100 / 72 for percent in (100, 125, 150, 200)]
        self.assertEqual(constructions, expected_scales)
        self.assertEqual(settings.call_count, 4)
        self.assertEqual(len(measured), 4 * 3 * 2)
        for scale in expected_scales:
            self.assertEqual({(mode, case) for value, mode, case, _ in measured if value == scale},
                             {(mode, case) for mode in ("manual", "semi-auto", "auto")
                              for case in ("default", "multiline_status")})
        self.assertTrue(all("\n" in status for _, _, case, status in measured
                            if case == "multiline_status"))
        self.assertEqual([item["scale_percent"] for item in result["layout_scale_checks"]],
                         [100, 125, 150, 200])
        self.assertIn("monitor scaling is unchanged", result["layout_scaling_method"])
        for window, picker in zip(windows, pickers):
            self.assertEqual(window._scale, original_scaling)
            self.assertEqual(window.withdraw.call_count, 1)
            self.assertEqual(picker._hide_overlays.call_count, 3)
            picker.close.assert_called_once()
        dpi.assert_called_once()
        overlay.assert_called_once_with(windows[0])
        repeated.assert_called_once_with(pickers[0])
        cycle.assert_called_once_with(pickers[0])
        manual.assert_called_once_with(pickers[0])
        self.assertEqual(result["repeated_semi_auto_contours"]["contours_completed"], 3)
        self.assertEqual(result["micrograph_navigation_cycle"]["continuous_auto"]["down_keys"], 3)
        self.assertEqual(result["consecutive_manual_strokes"]["strokes_completed"], 2)
        modules["pyautogui"].click.assert_not_called()
        self.assertEqual(result["desktop_events_sent"], 0)
        self.assertEqual(result["global_listeners_started"], 0)


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
