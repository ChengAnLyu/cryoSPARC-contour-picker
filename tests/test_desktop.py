"""Desktop regressions with in-memory backends; never touch the real screen."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from desktop import DesktopAdapter, DesktopError, GlobalHotkeys, enable_dpi_awareness
from execution import CaptureGeometry


class FakeBackend:
    def __init__(self):
        self.PAUSE = 0.1
        self.FAILSAFE = False
        self.clicks = []
        self.keys = []
        self.screenshot_regions = []
        self.dimensions = (800, 600)
        self.bad_capture = False

    def size(self):
        return self.dimensions

    def screenshot(self, region=None):
        self.screenshot_regions.append(region)
        dimensions = self.dimensions if region is None else region[2:]
        if self.bad_capture:
            dimensions = (dimensions[0] + 1, dimensions[1])
        return Image.new("RGB", dimensions, (1, 2, 3))

    def click(self, x, y):
        self.clicks.append((x, y))

    def press(self, key):
        self.keys.append(key)


class FakeWindows:
    def __init__(self):
        self.current = 11
        self.windows = {11, 22}
        self.hidden = set()
        self.minimized = set()
        self.allow_focus = True
        self.region_window = lambda x, y: 11

    def root_window(self, handle):
        return handle

    def foreground(self):
        return self.current

    def is_window(self, handle):
        return handle in self.windows

    def is_visible(self, handle):
        return handle not in self.hidden

    def is_minimized(self, handle):
        return handle in self.minimized

    def restore(self, handle):
        self.minimized.remove(handle)

    def focus(self, handle):
        if self.allow_focus:
            self.current = handle
            return True
        return False

    def window_at(self, x, y):
        return self.region_window(x, y)

    def primary_size(self):
        return 800, 600


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.windows = FakeWindows()
        self.adapter = DesktopAdapter(self.backend, self.windows, "Windows")

    def test_no_desktop_import_until_live_operation(self):
        with patch("desktop.importlib.import_module") as loader:
            DesktopAdapter(platform_name="Linux")
            loader.assert_not_called()

    def test_non_windows_allows_no_accidental_live_operations(self):
        adapter = DesktopAdapter(self.backend, self.windows, "Linux")
        self.assertIsNone(adapter.capture_target())
        with self.assertRaises(DesktopError):
            adapter.screenshot()
        self.assertEqual(self.backend.screenshot_regions, [])

    def test_capture_is_rgb_and_region_is_exact(self):
        image = self.adapter.screenshot(CaptureGeometry(10, 20, 50, 30))
        self.assertEqual(image.shape, (30, 50, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(self.backend.screenshot_regions, [(10, 20, 50, 30)])
        self.assertEqual(self.backend.PAUSE, 0)
        self.assertTrue(self.backend.FAILSAFE)

    def test_negative_or_off_primary_region_rejected_before_capture(self):
        for region in [CaptureGeometry(-1, 0, 10, 10), CaptureGeometry(795, 0, 10, 10)]:
            with self.subTest(region=region), self.assertRaises(DesktopError):
                self.adapter.screenshot(region)
        self.assertEqual(self.backend.screenshot_regions, [])

    def test_scaled_backend_and_wrong_capture_sizes_are_rejected(self):
        self.backend.dimensions = (400, 300)
        with self.assertRaises(DesktopError):
            self.adapter.screenshot()
        self.backend.dimensions = (800, 600)
        self.backend.bad_capture = True
        with self.assertRaises(DesktopError):
            self.adapter.screenshot(CaptureGeometry(10, 20, 50, 30))

    def test_window_under_roi_rejects_multiple_apps(self):
        region = CaptureGeometry(10, 20, 50, 30)
        self.assertEqual(self.adapter.window_under_roi(region), 11)
        self.windows.region_window = lambda x, y: 11 if x < 40 else 22
        with self.assertRaises(DesktopError):
            self.adapter.window_under_roi(region)

    def test_click_requires_captured_focused_target(self):
        with self.assertRaises(DesktopError):
            self.adapter.click(30, 40)
        self.adapter.focus_target(self.adapter.capture_target())
        self.adapter.click(30, 40)
        self.assertEqual(self.backend.clicks, [(30, 40)])
        self.windows.current = 22
        with self.assertRaises(DesktopError):
            self.adapter.click(31, 41)
        self.assertEqual(self.backend.clicks, [(30, 40)])

    def test_failed_focus_does_not_allow_clicks(self):
        self.windows.current = 22
        self.windows.allow_focus = False
        with self.assertRaises(DesktopError):
            self.adapter.focus_target(11)
        with self.assertRaises(DesktopError):
            self.adapter.click(30, 40)
        self.assertEqual(self.backend.clicks, [])

    def test_overlay_window_cannot_receive_pick_even_when_target_has_focus(self):
        self.adapter.focus_target(11)
        self.windows.region_window = lambda x, y: 22
        with self.assertRaises(DesktopError):
            self.adapter.click(30, 40)
        self.assertEqual(self.windows.current, 11)
        self.assertEqual(self.backend.clicks, [])

    def test_closed_window_and_off_screen_click_rejected(self):
        self.adapter.focus_target(11)
        with self.assertRaises(DesktopError):
            self.adapter.click(800, 40)
        self.windows.windows.remove(11)
        with self.assertRaises(DesktopError):
            self.adapter.click(30, 40)
        self.assertEqual(self.backend.clicks, [])

    def test_navigation_requires_unchanged_target(self):
        self.adapter.focus_target(11)
        self.adapter.press_next("down")
        self.assertEqual(self.backend.keys, ["down"])
        self.windows.current = 22
        with self.assertRaises(DesktopError):
            self.adapter.press_next("down")
        self.assertEqual(self.backend.keys, ["down"])

    def test_minimized_target_is_restored_before_focus(self):
        self.windows.minimized.add(11)
        self.windows.current = 22
        self.adapter.focus_target(11)
        self.assertNotIn(11, self.windows.minimized)
        self.adapter.ensure_target(11)


class HotkeyTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.listener = None

        def factory(**callbacks):
            self.listener = SimpleNamespace(
                **callbacks, start=lambda: None, stop=lambda: self.events.append("stopped")
            )
            return self.listener

        self.hotkeys = GlobalHotkeys(
            self.events.append,
            listener_factory=factory,
            key_enum=SimpleNamespace(esc="esc", f8="f8"),
            platform_name="Windows",
        )

    def test_repeat_press_is_debounced_and_release_rearms(self):
        self.hotkeys.start()
        self.listener.on_press("f8")
        self.listener.on_press("f8")
        self.listener.on_release("f8")
        self.listener.on_press("f8")
        self.listener.on_press("esc")
        self.listener.on_press("a")
        self.assertEqual(self.events, ["pause", "pause", "stop"])

    def test_stop_disables_further_callbacks(self):
        self.hotkeys.start()
        listener = self.listener
        self.hotkeys.stop()
        listener.on_press("esc")
        self.assertEqual(self.events, ["stopped"])
        self.hotkeys.stop()  # Idempotent cleanup.


class DpiTests(unittest.TestCase):
    def test_non_windows_does_not_touch_injected_api(self):
        self.assertFalse(enable_dpi_awareness("Linux", user32=object()))

    def test_modern_dpi_request(self):
        requests = []
        api = SimpleNamespace(SetProcessDpiAwarenessContext=lambda value: requests.append(value.value) or True)
        self.assertTrue(enable_dpi_awareness("Windows", user32=api))
        self.assertEqual(len(requests), 1)

    def test_legacy_fallback_is_explicit(self):
        api = SimpleNamespace(SetProcessDPIAware=lambda: True)
        with self.assertLogs("desktop", level="WARNING"):
            self.assertTrue(enable_dpi_awareness("Windows", user32=api))

    def test_unverified_dpi_failure_returns_false(self):
        api = SimpleNamespace(SetProcessDPIAware=lambda: False, IsProcessDPIAware=lambda: False)
        with self.assertLogs("desktop", level="ERROR"):
            self.assertFalse(enable_dpi_awareness("Windows", user32=api))


if __name__ == "__main__":
    unittest.main()
