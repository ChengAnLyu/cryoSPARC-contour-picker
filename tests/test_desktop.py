"""Desktop regressions with in-memory backends; never touch the real screen."""

from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from desktop import DesktopAdapter, DesktopError, GlobalHotkeys, enable_dpi_awareness
from execution import CancelledOperation, CancellationToken, CaptureGeometry, play_clicks


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

    def process_id(self, handle):
        return os.getpid() if handle == 22 else os.getpid() + 1


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

    def test_manual_target_is_resolved_at_the_pointer_without_changing_focus(self):
        self.windows.current = 22
        self.assertEqual(self.adapter.target_at(20, 30), 11)
        self.assertEqual(self.windows.current, 22)
        self.assertTrue(self.adapter.is_target_at(20, 30, 11))
        self.windows.region_window = lambda x, y: 22
        self.assertFalse(self.adapter.is_target_at(20, 30, 11))
        self.assertFalse(self.adapter.is_target_at(-1, 30, 11))
        with self.assertRaises(DesktopError):
            self.adapter.target_at(800, 30)

    def test_own_windows_are_identified_without_sending_input(self):
        self.assertTrue(self.adapter.is_own_window(22))
        self.assertFalse(self.adapter.is_own_window(11))
        self.assertEqual(self.backend.clicks, [])
        self.assertEqual(self.backend.keys, [])

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

    def test_bind_target_enables_first_batch_without_changing_native_focus(self):
        with patch.object(self.windows, "focus", side_effect=AssertionError("Must not steal focus")), \
             patch.object(self.windows, "restore", side_effect=AssertionError("Must not restore windows")):
            self.adapter.bind_target(11)
            self.adapter.click(30, 40)
        self.assertEqual(self.adapter._target, 11)
        self.assertEqual(self.windows.current, 11)
        self.assertEqual(self.backend.clicks, [(30, 40)])

    def test_bind_target_replaces_the_previous_batch_target(self):
        self.adapter.focus_target(11)
        self.windows.current = 22
        self.windows.region_window = lambda x, y: 22
        with patch.object(self.windows, "focus", side_effect=AssertionError("Must not steal focus")), \
             patch.object(self.windows, "restore", side_effect=AssertionError("Must not restore windows")):
            self.adapter.bind_target(22)
            self.adapter.click(31, 41)
        self.assertEqual(self.adapter._target, 22)
        self.assertEqual(self.backend.clicks, [(31, 41)])

    def test_failed_bind_clears_stale_target_and_never_focuses_or_restores(self):
        self.adapter.focus_target(11)
        self.windows.current = 22
        with patch.object(self.windows, "focus", side_effect=AssertionError("Must not steal focus")), \
             patch.object(self.windows, "restore", side_effect=AssertionError("Must not restore windows")):
            with self.assertRaisesRegex(DesktopError, "lost focus"):
                self.adapter.bind_target(11)
        self.assertIsNone(self.adapter._target)
        self.assertEqual(self.windows.current, 22)
        with self.assertRaises(DesktopError):
            self.adapter.click(30, 40)
        self.assertEqual(self.backend.clicks, [])

    def test_bind_rejects_a_minimized_target_without_restoring_it(self):
        self.windows.minimized.add(11)
        with patch.object(self.windows, "restore", side_effect=AssertionError("Must not restore windows")):
            with self.assertRaisesRegex(DesktopError, "lost focus"):
                self.adapter.bind_target(11)
        self.assertIn(11, self.windows.minimized)
        self.assertIsNone(self.adapter._target)
        self.assertEqual(self.backend.clicks, [])

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

    def test_fast_playback_stops_before_next_pick_if_focus_changes(self):
        self.adapter.bind_target(11)
        original_click = self.backend.click

        def click_then_lose_focus(x, y):
            original_click(x, y)
            self.windows.current = 22

        self.backend.click = click_then_lose_focus
        with self.assertRaisesRegex(DesktopError, "lost focus"):
            play_clicks([(30, 40), (40, 40), (50, 40)], self.adapter.click,
                        CaptureGeometry(20, 20, 100, 100), CancellationToken(), delay=.03)
        self.assertEqual(self.backend.clicks, [(30, 40)])
        self.assertTrue(self.backend.FAILSAFE)
        self.assertEqual(self.backend.PAUSE, 0)

    def test_fast_playback_does_not_swallow_backend_fail_safe(self):
        class FailSafeError(RuntimeError):
            pass

        self.adapter.bind_target(11)
        original_click = self.backend.click
        failure = FailSafeError("Mouse reached a fail-safe corner")

        def guarded_backend(x, y):
            if self.backend.clicks:
                raise failure
            original_click(x, y)

        self.backend.click = guarded_backend
        with self.assertRaises(FailSafeError) as raised:
            play_clicks([(30, 40), (40, 40), (50, 40)], self.adapter.click,
                        CaptureGeometry(20, 20, 100, 100), CancellationToken(), delay=.03)
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.backend.clicks, [(30, 40)])
        self.assertTrue(self.backend.FAILSAFE)

    def test_fast_playback_cancellation_prevents_remaining_native_calls(self):
        self.adapter.bind_target(11)
        token = CancellationToken()
        progress = []

        def cancel_after_first_pick(done, total):
            progress.append((done, total))
            token.cancel()

        with self.assertRaises(CancelledOperation):
            play_clicks([(30, 40), (40, 40), (50, 40)], self.adapter.click,
                        CaptureGeometry(20, 20, 100, 100), token, delay=.03,
                        progress=cancel_after_first_pick)
        self.assertEqual(self.backend.clicks, [(30, 40)])
        self.assertEqual(progress, [(1, 3)])
        self.assertEqual(self.backend.keys, [])

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

    def test_space_is_consumed_only_during_auto_and_release_is_paired(self):
        suppressed = []
        holder = []
        def factory(**callbacks):
            listener = SimpleNamespace(**callbacks, start=lambda: None, stop=lambda: None,
                                       suppress_event=lambda: suppressed.append(True))
            holder.append(listener)
            return listener
        hotkeys = GlobalHotkeys(self.events.append, listener_factory=factory,
                                key_enum=SimpleNamespace(esc="esc", f8="f8"),
                                platform_name="Windows", space_pause=True)
        hotkeys.start()
        filt = holder[0].win32_event_filter
        space = SimpleNamespace(vkCode=0x20, flags=0)
        self.assertTrue(filt(0x0100, space))
        self.assertTrue(filt(0x0101, space))
        self.assertEqual(self.events, [])
        hotkeys.enable_space_pause(True)
        self.assertFalse(filt(0x0100, space))
        self.assertFalse(filt(0x0100, space))
        self.assertEqual(self.events, ["space_pause"])
        hotkeys.enable_space_pause(False)
        self.assertFalse(filt(0x0101, space))
        self.assertEqual(len(suppressed), 3)
        self.assertTrue(filt(0x0100, space))
        # Starting auto while an ordinary Space press is held must not take
        # ownership of its repeat or release from the underlying browser.
        hotkeys.enable_space_pause(True)
        self.assertTrue(filt(0x0100, space))
        self.assertTrue(filt(0x0101, space))
        self.assertFalse(filt(0x0100, space))
        self.assertFalse(filt(0x0101, space))
        self.assertEqual(self.events, ["space_pause", "space_pause"])
        self.assertTrue(filt(0x0100, SimpleNamespace(vkCode=0x41, flags=0)))
        hotkeys.enable_space_pause(True)
        self.assertTrue(filt(0x0100, SimpleNamespace(vkCode=0x20, flags=0x10)))
        self.assertEqual(self.events, ["space_pause", "space_pause"])
        hotkeys.stop()


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
