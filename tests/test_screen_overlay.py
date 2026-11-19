"""Screen overlay contracts; these fakes never read or control the desktop."""

from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from execution import CaptureGeometry
from screen_overlay import (INPUT_ALPHA, TRANSPARENT_COLOR, TRANSPARENT_RGB,
                            InputSurface, ManualMouseObserver, OverlayError,
                            ScreenOverlay, _OverlayWindowsAPI, _mask_image,
                            _verify_main_control_panel,
                            control_panel_bounds, keep_control_panel_above)


class FakeWindow:
    def __init__(self, root):
        self.root = root
        root.windows.append(self)
        self.options = {}
        self.bindings = {}
        self.calls = []
        self.destroyed = False
        self.handle = 100 + len(root.windows)

    def withdraw(self):
        self.calls.append("withdraw")

    def overrideredirect(self, value):
        self.options["overrideredirect"] = value

    def attributes(self, name, value):
        self.options[name] = value

    def geometry(self, value):
        self.options["geometry"] = value

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def update_idletasks(self):
        self.calls.append("update_idletasks")

    def state(self, value):
        self.calls.append(("state", value))

    def winfo_id(self):
        return self.handle

    def bind(self, event, callback):
        self.bindings[event] = callback

    def destroy(self):
        self.destroyed = True


class FakeCanvas:
    def __init__(self, window, **kwargs):
        self.window = window
        self.options = kwargs
        self.items = []
        self.bindings = {}

    def pack(self, **kwargs):
        self.pack_options = kwargs

    def bind(self, event, callback):
        self.bindings[event] = callback

    def delete(self, tag):
        if tag == "all":
            self.items.clear()
        else:
            self.items = [item for item in self.items if item[2].get("tags") != tag]

    def create_line(self, *coords, **kwargs):
        self.items.append(("line", coords, kwargs))

    def create_rectangle(self, *coords, **kwargs):
        self.items.append(("rectangle", coords, kwargs))

    def create_oval(self, *coords, **kwargs):
        self.items.append(("oval", coords, kwargs))

    def create_image(self, *coords, **kwargs):
        self.items.append(("image", coords, kwargs))


class FakeStyles:
    def __init__(self):
        self.configurations = []
        self.raised = []
        self.panel_rectangle = (600, 450, 180, 120)
        self.hidden = set()
        self.minimized_windows = set()

    def primary_size(self):
        return 800, 600

    def configure(self, handle, *, click_through):
        self.configurations.append((handle, click_through))
        return handle

    def raise_without_activation(self, handle):
        self.raised.append(handle)

    def visible(self, handle):
        return handle not in self.hidden

    def minimized(self, handle):
        return handle in self.minimized_windows

    def bounds(self, handle):
        return self.panel_rectangle


class ScreenSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = SimpleNamespace(windows=[], winfo_id=lambda: 50)
        self.styles = FakeStyles()
        self.options = dict(window_factory=FakeWindow, canvas_factory=FakeCanvas,
                            style_api=self.styles, platform_name="Windows")

    def test_annotation_is_lazy_and_show_is_nonactivating_click_through(self):
        overlay = ScreenOverlay(self.root, **self.options)
        overlay.hide()
        overlay.clear()
        self.assertEqual(self.root.windows, [])
        overlay.show()
        window = overlay.window
        self.assertEqual(window.options["geometry"], "800x600+0+0")
        self.assertEqual(window.options["-transparentcolor"], TRANSPARENT_COLOR)
        self.assertTrue(window.options["-topmost"])
        self.assertIn(("state", "normal"), window.calls)
        self.assertTrue(all(click_through for _, click_through in self.styles.configurations))
        self.assertEqual(self.styles.raised, [window.handle, self.root.winfo_id()])
        overlay.hide()
        self.assertFalse(overlay.visible)
        self.assertEqual(window.calls[-1], "withdraw")
        overlay.close()
        overlay.close()
        self.assertTrue(window.destroyed)

    def test_each_surface_show_restores_the_interactive_panel_above_it(self):
        overlay = ScreenOverlay(self.root, **self.options)
        surface = InputSurface(self.root, CaptureGeometry(0, 0, 800, 600),
                               lambda *xy: None, lambda *xy: None, lambda *xy: None,
                               **self.options)
        surface.show()
        overlay.show()
        overlay.show()
        self.assertEqual(self.styles.raised, [surface.window.handle, 50,
                                              overlay.window.handle, 50,
                                              overlay.window.handle, 50])
        # The panel is raised only; its styles are never changed to passive.
        self.assertNotIn(50, [handle for handle, _ in self.styles.configurations])
        surface.hide()
        overlay.hide()
        self.assertTrue(self.styles.visible(50))

    def test_panel_helpers_use_native_outer_bounds_without_activating_it(self):
        self.assertEqual(control_panel_bounds(self.root, style_api=self.styles,
                                              platform_name="Windows"),
                         (600, 450, 180, 120))
        self.assertTrue(keep_control_panel_above(self.root, style_api=self.styles,
                                                 platform_name="Windows"))
        self.assertEqual(self.styles.raised, [50])
        self.assertEqual(self.styles.configurations, [])
        self.styles.hidden.add(50)
        self.assertFalse(keep_control_panel_above(self.root, style_api=self.styles,
                                                  platform_name="Windows"))
        self.assertEqual(self.styles.raised, [50])

    def test_input_excludes_panel_caption_and_rechecks_after_panel_moves(self):
        events = []
        self.styles.panel_rectangle = (100, 100, 120, 80)
        surface = InputSurface(self.root, CaptureGeometry(0, 0, 800, 600),
                               lambda *xy: events.append(("press", *xy)),
                               lambda *xy: events.append(("move", *xy)),
                               lambda *xy: events.append(("release", *xy)),
                               on_right=lambda *xy: events.append(("right", *xy)),
                               on_cancel=lambda: events.append(("cancel",)),
                               **self.options)
        surface.show()
        event = lambda x, y: SimpleNamespace(x_root=x, y_root=y)
        surface._press(event(110, 105))  # Native caption, not just client area.
        surface._right(event(110, 130))
        self.assertEqual(events, [])
        surface._press(event(50, 50))
        surface._move(event(110, 120))
        surface._release(event(60, 60))
        self.assertEqual(events, [("press", 50, 50), ("cancel",)])
        self.styles.panel_rectangle = (300, 200, 120, 80)
        surface._press(event(110, 120))  # Old panel position is now usable.
        surface._release(event(310, 210))  # Release over its new caption cancels.
        self.assertEqual(events[-2:], [("press", 110, 120), ("cancel",)])

    def test_retired_input_surface_cannot_cancel_or_select_the_next_contour(self):
        events = []
        surface = InputSurface(self.root, CaptureGeometry(0, 0, 300, 300),
                               lambda *xy: events.append("press"),
                               lambda *xy: events.append("move"),
                               lambda *xy: surface.close(),
                               on_cancel=lambda: events.append("cancel"), **self.options)
        surface.show()
        event = SimpleNamespace(x_root=30, y_root=30)
        surface._press(event)
        surface._release(event)
        self.assertTrue(surface.closed)
        self.assertIsNone(surface._release_callback)
        surface._cancel()
        surface._press(event)
        surface._release(event)
        self.assertEqual(events, ["press"])

    def test_mask_pixels_keep_roi_origin_holes_and_disconnected_parts(self):
        geometry = CaptureGeometry(123, 234, 20, 16)
        mask = np.zeros((16, 20), dtype=bool)
        mask[2:13, 3:15] = True
        mask[6:9, 7:11] = False
        mask[14, 18] = True
        original = mask.copy()
        candidate = SimpleNamespace(id="one", mask=mask)
        photo_factory = lambda bitmap, master: SimpleNamespace(pixels=np.asarray(bitmap).copy())
        overlay = ScreenOverlay(self.root, photo_factory=photo_factory, **self.options)
        overlay.draw_masks([candidate], geometry, {"one"})
        item = overlay.canvas.items[-1]
        self.assertEqual(item[1], (123, 234))
        self.assertEqual(item[2]["anchor"], "nw")
        pixels = item[2]["image"].pixels
        self.assertEqual(pixels.shape, (16, 20, 3))
        self.assertEqual(pixels[7, 8].tolist(), list(TRANSPARENT_RGB))
        self.assertNotEqual(pixels[14, 18].tolist(), list(TRANSPARENT_RGB))
        self.assertNotEqual(pixels[6, 6].tolist(), list(TRANSPARENT_RGB))
        np.testing.assert_array_equal(mask, original)
        self.assertFalse(overlay.visible)  # Drawing is not permission to show.

    def test_mask_shape_mismatch_is_rejected(self):
        candidate = SimpleNamespace(id="one", mask=np.ones((8, 9), dtype=bool))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            _mask_image([candidate], CaptureGeometry(30, 40, 9, 7))

    def test_trace_box_region_and_points_use_unscaled_screen_coordinates(self):
        overlay = ScreenOverlay(self.root, **self.options)
        overlay.draw_trace([(110, 205), (121, 217)])
        overlay.draw_region(CaptureGeometry(100, 200, 40, 30))
        overlay.draw_box((104, 206, 130, 222))
        overlay.draw_points([(125, 215)], [1])
        self.assertEqual(overlay.canvas.items[0][1], (110, 205, 121, 217))
        self.assertEqual(overlay.canvas.items[1][1], (100, 200, 139, 229))
        self.assertEqual(overlay.canvas.items[2][1], (104, 206, 130, 222))
        self.assertEqual(overlay.canvas.items[3][1], (121, 211, 129, 219))
        overlay.clear()
        self.assertEqual(overlay.canvas.items, [])

    def test_input_uses_nonzero_alpha_without_color_key_and_root_coordinates(self):
        events = []
        surface = InputSurface(self.root, CaptureGeometry(100, 200, 80, 90),
                               lambda x, y: events.append(("press", x, y)),
                               lambda x, y: events.append(("move", x, y)),
                               lambda x, y: events.append(("release", x, y)),
                               on_right=lambda x, y: events.append(("right", x, y)),
                               **self.options)
        surface.show()
        self.assertEqual(surface.window.options["geometry"], "80x90+100+200")
        self.assertEqual(surface.window.options["-alpha"], INPUT_ALPHA)
        self.assertGreater(INPUT_ALPHA, 0)
        self.assertNotIn("-transparentcolor", surface.window.options)
        self.assertTrue(all(not click_through for _, click_through in self.styles.configurations))
        press = SimpleNamespace(x_root=120, y_root=220, x=20, y=20)
        outside = SimpleNamespace(x_root=195, y_root=310, x=95, y=110)
        surface.canvas.bindings["<ButtonPress-1>"](press)
        surface.canvas.bindings["<B1-Motion>"](outside)
        surface.canvas.bindings["<ButtonRelease-1>"](outside)
        surface.canvas.bindings["<ButtonPress-3>"](press)
        self.assertEqual(events, [("press", 120, 220), ("move", 195, 310),
                                  ("release", 195, 310), ("right", 120, 220)])
        surface.canvas.bindings["<ButtonPress-1>"](outside)
        surface.canvas.bindings["<ButtonRelease-1>"](press)
        self.assertEqual(len(events), 4)

    def test_hiding_input_cancels_an_active_drag(self):
        events = []
        surface = InputSurface(self.root, CaptureGeometry(10, 20, 40, 50),
                               lambda *xy: events.append(("press", *xy)),
                               lambda *xy: events.append(("move", *xy)),
                               lambda *xy: events.append(("release", *xy)),
                               **self.options)
        event = SimpleNamespace(x_root=15, y_root=25)
        surface.show()
        surface._press(event)
        surface.hide()
        surface._release(event)
        surface.show()
        surface._move(event)
        surface._release(event)
        self.assertEqual(events, [("press", 15, 25)])

    def test_off_primary_input_is_rejected_before_creating_a_window(self):
        surface = InputSurface(self.root, CaptureGeometry(790, 20, 40, 50),
                               lambda *xy: None, lambda *xy: None, lambda *xy: None,
                               **self.options)
        with self.assertRaisesRegex(OverlayError, "primary monitor"):
            surface.show()
        self.assertEqual(self.root.windows, [])

    def test_gui_calls_from_worker_fail_before_creating_a_window(self):
        overlay = ScreenOverlay(self.root, **self.options)
        failures = []
        def work():
            try:
                overlay.show()
            except RuntimeError as error:
                failures.append(str(error))
        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        self.assertEqual(len(failures), 1)
        self.assertIn("GUI thread", failures[0])
        self.assertEqual(self.root.windows, [])

    def test_native_style_failure_closes_the_partly_created_overlay(self):
        self.styles.configure = lambda *args, **kwargs: (_ for _ in ()).throw(OverlayError("style error"))
        overlay = ScreenOverlay(self.root, **self.options)
        with self.assertRaisesRegex(OverlayError, "style error"):
            overlay.show()
        self.assertTrue(self.root.windows[0].destroyed)
        self.assertIsNone(overlay.window)

    def test_native_show_failure_withdraws_the_mouse_catcher(self):
        self.styles.raise_without_activation = lambda handle: (_ for _ in ()).throw(OverlayError("show error"))
        surface = InputSurface(self.root, CaptureGeometry(10, 20, 40, 50),
                               lambda *xy: None, lambda *xy: None, lambda *xy: None,
                               **self.options)
        with self.assertRaisesRegex(OverlayError, "show error"):
            surface.show()
        self.assertFalse(surface.visible)
        self.assertEqual(surface.window.calls[-1], "withdraw")


class NativeStyleTests(unittest.TestCase):
    def test_styles_preserve_existing_bits_and_pointer_sized_handles(self):
        handle = 1 << 42
        class Native:
            style = 0x100
            def GetAncestor(self, child, mode):
                return handle
            def GetWindowLongPtrW(self, window, index):
                self.read = (window, index)
                return self.style
            def SetWindowLongPtrW(self, window, index, style):
                self.written = (window, index, style)
                previous, self.style = self.style, style
                return previous
            def SetWindowPos(self, *args):
                self.position = args
                return True
        native = Native()
        api = _OverlayWindowsAPI(native)
        api.configure(15, click_through=True)
        self.assertEqual(native.written[0], handle)
        self.assertTrue(native.style & 0x100)
        self.assertTrue(native.style & api.WS_EX_TRANSPARENT)
        self.assertTrue(native.style & api.WS_EX_NOACTIVATE)
        self.assertTrue(native.style & api.WS_EX_LAYERED)
        api.configure(15, click_through=False)
        self.assertFalse(native.style & api.WS_EX_TRANSPARENT)
        self.assertTrue(native.style & api.WS_EX_NOACTIVATE)
        api.raise_without_activation(15)
        self.assertEqual(native.position[:2], (handle, -1))
        self.assertTrue(native.position[-1] & 0x0010)


class MainPanelVerifierTests(unittest.TestCase):
    def setUp(self):
        button = SimpleNamespace(winfo_class=lambda: "Button", cget=lambda key: "terminate",
                                 winfo_viewable=lambda: True, winfo_rootx=lambda: 130,
                                 winfo_rooty=lambda: 260, winfo_width=lambda: 100,
                                 winfo_height=lambda: 40)
        self.root = SimpleNamespace(update=lambda: None, winfo_id=lambda: 50,
                                    winfo_rootx=lambda: 108, winfo_rooty=lambda: 130,
                                    winfo_width=lambda: 240, winfo_height=lambda: 200,
                                    winfo_children=lambda: [button])
        self.api = FakeStyles()
        self.api.panel_rectangle = (100, 100, 256, 238)
        self.api.root_handle = lambda handle: handle
        self.api.foreground = lambda: 999  # The test must not activate the panel.
        self.api.is_above = lambda first, second: first == 50
        self.api.window_at = lambda x, y: 50

    def test_main_root_check_uses_fullscreen_defaults_and_keeps_panel_accessible(self):
        with patch("screen_overlay.platform.system", return_value="Windows"), \
             patch("screen_overlay.ScreenOverlay") as overlay, \
             patch("screen_overlay.InputSurface") as input_surface:
            result = _verify_main_control_panel(self.root, self.api)
            overlay.assert_called_once_with(self.root, style_api=self.api)
            self.assertEqual(input_surface.call_args.args[:2],
                             (self.root, CaptureGeometry(0, 0, 800, 600)))
            self.assertNotIn("control_panel", input_surface.call_args.kwargs)
            self.assertTrue(result["main_root_above_surfaces"])
            self.assertTrue(result["main_root_button_and_caption_hit_testable"])
            input_surface.return_value.close.assert_called_once()
            overlay.return_value.close.assert_called_once()

    def test_main_root_check_rejects_owned_popup_ordering_and_cleans_up(self):
        self.api.is_above = lambda first, second: False
        with patch("screen_overlay.platform.system", return_value="Windows"), \
             patch("screen_overlay.ScreenOverlay") as overlay, \
             patch("screen_overlay.InputSurface") as input_surface:
            with self.assertRaisesRegex(OverlayError, "covers the main Tk control panel"):
                _verify_main_control_panel(self.root, self.api)
            input_surface.return_value.close.assert_called_once()
            overlay.return_value.close.assert_called_once()


class FakeMouseListener:
    def __init__(self, **callbacks):
        self.callbacks = callbacks
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class ManualObserverTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.listeners = []
        self.buttons = SimpleNamespace(left="LEFT", right="RIGHT")
        def factory(**callbacks):
            listener = FakeMouseListener(**callbacks)
            self.listeners.append(listener)
            return listener
        self.observer = ManualMouseObserver(self.events.append, listener_factory=factory,
                                            button_enum=self.buttons, platform_name="Windows")

    def test_only_physical_left_drag_events_are_queued(self):
        with patch("screen_overlay.importlib.import_module") as imported:
            self.observer.start()
        imported.assert_not_called()
        callbacks = self.listeners[0].callbacks
        callbacks["on_move"](100, 200, False)
        callbacks["on_click"](100, 200, "RIGHT", True, False)
        callbacks["on_click"](100, 200, "LEFT", True, True)
        callbacks["on_click"](100, 200, "LEFT", True, False)
        callbacks["on_move"](110, 210, True)
        callbacks["on_click"](110, 210, "LEFT", False, True)
        callbacks["on_move"](115, 215, False)
        callbacks["on_click"](120, 220, "LEFT", False, False)
        self.assertEqual(self.events, [("press", 100, 200), ("move", 115, 215), ("release", 120, 220)])
        event_filter = callbacks["win32_event_filter"]
        self.assertTrue(event_filter(0, SimpleNamespace(flags=0)))
        self.assertFalse(event_filter(0, SimpleNamespace(flags=1)))
        self.assertFalse(event_filter(0, SimpleNamespace(flags=2)))

    def test_old_listener_callbacks_cannot_escape_stop_and_restart(self):
        self.observer.start()
        old = self.listeners[0]
        old.callbacks["on_click"](1, 2, "LEFT", True)
        self.observer.stop()
        self.observer.stop()
        self.assertTrue(old.stopped)
        self.observer.start()
        self.assertEqual(len(self.listeners), 2)
        old.callbacks["on_click"](3, 4, "LEFT", True)
        old.callbacks["on_move"](5, 6)
        old.callbacks["on_click"](7, 8, "LEFT", False)
        self.assertFalse(old.callbacks["win32_event_filter"](0, SimpleNamespace(flags=0)))
        self.listeners[1].callbacks["on_click"](10, 20, "LEFT", True)
        self.assertEqual(self.events, [("press", 1, 2), ("press", 10, 20)])

    def test_callback_can_run_on_listener_thread_without_gui_calls(self):
        self.observer.start()
        callbacks = self.listeners[0].callbacks
        thread = threading.Thread(target=lambda: (
            callbacks["on_click"](20, 30, "LEFT", True),
            callbacks["on_move"](30, 40),
            callbacks["on_click"](40, 50, "LEFT", False)))
        thread.start()
        thread.join()
        self.assertEqual(self.events, [("press", 20, 30), ("move", 30, 40), ("release", 40, 50)])


if __name__ == "__main__":
    unittest.main()
