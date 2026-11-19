"""Live-screen controller behavior without screenshots, hooks, or real clicks."""
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from execution import CancellationToken, CaptureGeometry, TaskResult
from manual_sampling import legacy_contour_points
from recognition import make_candidate
from screen_app import (ScreenPickerApp, screen_changed_fraction, screen_mask_at, screen_model_points,
                        screen_background_guard, screen_background_changed, screen_region_overlaps_panel)


class Variable:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class Root:
    def __init__(self):
        self.hidden = False
        self.callbacks = {}
        self.next_id = 0
        self.bounds = (0, 0, 100, 80)
    def after(self, delay, callback):
        self.next_id += 1
        self.callbacks[self.next_id] = callback
        return self.next_id
    def after_cancel(self, identifier):
        self.callbacks.pop(identifier, None)
    def run_next(self):
        identifier = next(iter(self.callbacks))
        self.callbacks.pop(identifier)()
    def withdraw(self):
        raise AssertionError("The control panel must never be withdrawn")
    def deiconify(self):
        raise AssertionError("Panel restoration must not request foreground focus")
    def state(self, value):
        assert value == "normal"
        self.hidden = False
    def lift(self):
        pass
    def destroy(self):
        pass
    def winfo_rootx(self):
        return self.bounds[0]
    def winfo_rooty(self):
        return self.bounds[1]
    def winfo_width(self):
        return self.bounds[2]
    def winfo_height(self):
        return self.bounds[3]


class Surface:
    def __init__(self, *args, **kwargs):
        self.visible = False
        self.closed = False
        self.traces = []
        self.masks = []
    def show(self):
        self.visible = True
    def hide(self):
        self.visible = False
    def close(self):
        self.hide()
        self.closed = True
    def clear(self):
        self.traces = []
    def draw_trace(self, points):
        self.traces.append(tuple(points))
    def draw_masks(self, candidates, geometry, selected_ids=(), preview=None):
        self.masks = list(candidates)
    def draw_region(self, geometry):
        pass
    def draw_box(self, box):
        pass


class Runner:
    def __init__(self):
        self.generation = 0
        self.busy = False
        self.jobs = []
        self.results = []
    def submit(self, function, token=None):
        if self.busy:
            raise RuntimeError("busy")
        token = token or CancellationToken(self.generation)
        self.busy = True
        self.jobs.append((function, token))
        return token
    def invalidate(self):
        self.generation += 1
        for _, token in self.jobs:
            token.cancel()
    def finish(self):
        function, token = self.jobs[-1]
        try:
            return function(token)
        finally:
            self.busy = False
    def poll_results(self):
        result, self.results = self.results, []
        if result:
            self.busy = False
        return result
    def shutdown(self, **kwargs):
        self.invalidate()


class Desktop:
    def __init__(self, app):
        self.app = app
        self.rgb = np.full((320, 480, 3), 100, np.uint8)
        self.clicks, self.keys, self.captures = [], [], []
        self.foreground = 42
        self.target = 42
    def size(self):
        return 480, 320
    def target_at(self, x, y):
        return self.target
    def is_own_window(self, target):
        return target == 123
    def is_target_at(self, x, y, target):
        return self.target_at(x, y) == target
    def capture_target(self):
        return self.foreground
    def window_under_roi(self, region):
        return self.target
    def focus_target(self, target):
        self.foreground = target
    def bind_target(self, target):
        self.ensure_target(target)
    def ensure_target(self, target=None):
        if self.foreground != (target or self.target):
            raise RuntimeError("focus changed")
    def _ready_screen(self):
        assert not self.app.root.hidden, "Control panel must remain visible during capture and clicks"
        assert self.app.overlay is None or not self.app.overlay.visible
        assert self.app.input_surface is None or not self.app.input_surface.visible
    def screenshot(self, region=None):
        self._ready_screen()
        self.captures.append(region)
        if region is None:
            return self.rgb.copy()
        return self.rgb[region.top:region.top+region.height, region.left:region.left+region.width].copy()
    def click(self, x, y):
        self._ready_screen()
        self.clicks.append((x, y))
    def press_next(self, key):
        self._ready_screen()
        self.keys.append(key)


def app_state(mode="manual"):
    app = ScreenPickerApp.__new__(ScreenPickerApp)
    app.root, app.runner = Root(), Runner()
    app.tk = SimpleNamespace(TclError=RuntimeError)
    app.events = queue.Queue()
    app.desktop = Desktop(app)
    app.overlay = Surface()
    app.input_surface = app.advanced = None
    app.mouse_observer = Mock()
    app.hotkeys = Mock()
    app._manual_gate = threading.Event()
    app._input_lock = threading.Lock()
    app._emergency_stop = threading.Event()
    app._manual_gate.set()
    app._mouse_epoch = 0
    app.mode, app.mode_var = mode, Variable(mode)
    app.disabled, app.spacing = Variable(False), Variable(13)
    app.status, app.region_label = Variable(""), Variable("")
    app.checkpoint, app.model_type = Variable("supplied-model.pth"), Variable("vit_b")
    app.device, app.grid_side, app.click_delay = Variable("cpu"), Variable(8), Variable(.08)
    app.frames, app.wait_seconds = Variable(1), Variable(5.0)
    app.closed = app.transition = app.paused = False
    app._fit_after = None
    app._fitting = False
    app.scheduled = set()
    app.active_token = None
    app.job_kind = None
    app.manual_window = app.manual_geometry = None
    app.trace, app.drawing, app.released = [], False, False
    app.trace_dirty = False
    app.panel_bounds = (0, 0, 100, 80)
    app.geometry = CaptureGeometry(140, 100, 80, 70) if mode != "manual" else None
    app.target = 42 if mode != "manual" else None
    app.rgb = np.full((70, 80, 3), 100, np.uint8) if mode != "manual" else None
    app.reference = app.rgb.copy() if app.rgb is not None else None
    app.candidates, app.alternatives, app.auto_history = [], [], []
    app.alternative_index, app.frame_id, app.auto_remaining = 0, 1, 0
    app.picked_points = set()
    app.recognizer = app.recognizer_config = None
    app.input_kind = app.input_start = None
    app.log_path = Path("test.log")
    return app


class ScreenGeometryTests(unittest.TestCase):
    def test_visible_panel_overlap_uses_full_rectangle_not_roi_sample_points(self):
        region = CaptureGeometry(100, 100, 300, 200)
        self.assertTrue(screen_region_overlaps_panel(region, (120, 120, 50, 40)))
        self.assertTrue(screen_region_overlaps_panel(region, (380, 280, 80, 80)))
        self.assertFalse(screen_region_overlaps_panel(region, (400, 100, 80, 80)))
        self.assertFalse(screen_region_overlaps_panel(region, None))

    def test_image_drift_does_not_wrap_unsigned_values(self):
        bright = np.full((10, 20, 3), 250, np.uint8)
        dark = np.full((10, 20, 3), 5, np.uint8)
        self.assertEqual(screen_changed_fraction(bright, dark), 1)
        self.assertEqual(screen_changed_fraction(bright, bright), 0)

    def test_hit_test_preserves_holes_and_prefers_small_nested_region(self):
        region = CaptureGeometry(100, 200, 40, 30)
        outer = np.ones((30, 40), bool)
        outer[10:20, 10:20] = False
        small = np.zeros_like(outer)
        small[2:8, 2:8] = True
        a, b = make_candidate(outer), make_candidate(small)
        self.assertIs(screen_mask_at([a, b], region, 104, 204), b)
        self.assertIsNone(screen_mask_at([a, b], region, 115, 215))
        self.assertIsNone(screen_mask_at([a, b], region, 99, 204))

    def test_model_union_is_unique_and_remembers_previously_clicked_points(self):
        region = CaptureGeometry(100, 200, 40, 30)
        a = make_candidate(np.ones((30, 40), bool))
        local, screen = screen_model_points([a, a], region, 5)
        self.assertEqual(len(screen), len({tuple(p) for p in screen}))
        self.assertTrue(all(region.contains(*point) for point in screen))
        kept, _ = screen_model_points([a], region, 5, picked=[tuple(local[0])])
        self.assertEqual(len(kept), len(local)-1)

    def test_background_guard_excludes_marker_region_but_detects_other_scene_changes(self):
        region = CaptureGeometry(0, 0, 100, 100)
        mask = np.zeros((100, 100), bool)
        mask[30:60, 30:60] = True
        guard = screen_background_guard([make_candidate(mask)], region, 13)
        before = np.full((100, 100, 3), 100, np.uint8)
        after = before.copy()
        after[30:60, 30:60] = 250
        self.assertEqual(screen_background_changed(before, after, guard), 0)
        after[:20, :] = 0
        self.assertGreater(screen_background_changed(before, after, guard), .03)
        self.assertIsNone(screen_background_guard([make_candidate(np.ones((100, 100), bool))], region, 13))


class ScreenControllerTests(unittest.TestCase):
    def setUp(self):
        self.sleep = patch.object(CancellationToken, "sleep", lambda token, seconds: token.check_cancelled())
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        for replacement in (
            patch("screen_app.keep_control_panel_above"),
            patch("screen_app.control_panel_bounds", side_effect=lambda root: root.bounds),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_listener_callbacks_only_enqueue_and_cancel_token(self):
        app = app_state()
        app.disabled = SimpleNamespace(get=Mock(side_effect=AssertionError("No Tk on listener")))
        app._mouse_event(("press", 150, 100))
        self.assertEqual(app.events.get_nowait(), ("mouse", 0, "press", 150, 100))
        app.active_token = CancellationToken()
        app._hotkey("stop")
        self.assertTrue(app.active_token.cancelled)
        self.assertEqual(app.events.get_nowait(), ("hotkey", "stop"))

    def test_disabled_or_panel_press_never_starts_a_capture(self):
        app = app_state()
        app._manual_input(0, "press", 50, 50)
        self.assertFalse(app.drawing)
        app.disabled.set(True)
        app._manual_input(0, "press", 160, 100)
        self.assertFalse(app.drawing)
        self.assertEqual(app.desktop.captures, [])

    def test_each_manual_stroke_acquires_its_current_window(self):
        app = app_state()
        app.manual_window = 42
        app.desktop.target = app.desktop.foreground = 99
        app._manual_input(0, "press", 160, 100)
        self.assertTrue(app.drawing)
        self.assertEqual(app.manual_window, 99)
        self.assertFalse(app.root.hidden)

    def test_manual_does_not_acquire_its_own_native_titlebar_as_target(self):
        app = app_state()
        app.desktop.target = app.desktop.foreground = 123
        app._manual_input(0, "press", 150, 100)  # outside client bounds, own native HWND
        self.assertIsNone(app.manual_window)
        self.assertFalse(app.drawing)
        self.assertFalse(app.root.hidden)

    def test_manual_is_independent_of_model_delay_settings(self):
        app = app_state()
        app.click_delay.set("unfinished model setting")
        self.assertEqual(app._live_ready(), .1)

    def test_manual_release_immediately_submits_original_grid_with_panel_visible(self):
        app = app_state()
        polygon = [(150, 110), (210, 110), (210, 155), (150, 155), (150, 110)]
        app._manual_input(0, "press", *polygon[0])
        app._sync_manual()
        self.assertTrue(app._manual_gate.is_set())
        self.assertTrue(app.overlay.visible, "Show the red trace immediately")
        self.assertEqual(app.runner.jobs, [], "Drawing must not start capture or a model job")
        for point in polygon[1:]:
            app._manual_input(0, "move", *point)
        app._manual_input(0, "release", *polygon[-1])
        app._sync_manual()
        self.assertFalse(app._manual_gate.is_set(), "Release must gate later movement")
        self.assertEqual(len(app.runner.jobs), 1, "Release submits picking without a capture delay")
        self.assertEqual(app.root.callbacks, {})
        self.assertIsNone(app.geometry, "Manual mode must not require a selected ROI")
        self.assertIsNone(app.recognizer)
        app.desktop.focus_target = Mock(side_effect=AssertionError("Manual must not change focus"))
        with patch("screen_app.play_clicks", wraps=__import__("execution").play_clicks) as playback:
            result = app.runner.finish()
        self.assertEqual(playback.call_args.kwargs["delay"], .1)
        self.assertEqual(app.desktop.clicks, [tuple(p) for p in legacy_contour_points(polygon, 13)])
        self.assertEqual(app.desktop.captures, [], "Manual mode must never take screenshots")
        app._result("manual picks", result)
        app._sync_manual()
        self.assertTrue(app._manual_gate.is_set())
        self.assertFalse(app.disabled.get())
        self.assertFalse(app.root.hidden)

    def test_manual_does_not_reject_normal_hover_or_pick_feedback(self):
        app = app_state()
        app.manual_window = 42
        app.trace = [(150, 110), (210, 110), (210, 155), (150, 155)]
        app.desktop.rgb[100:180, 140:230] = 0
        app._finish_manual()
        result = app.runner.finish()
        self.assertGreater(result["count"], 0)
        self.assertEqual(app.desktop.captures, [])

    def test_first_press_is_not_lost_while_browser_activation_is_pending(self):
        app = app_state()
        app.desktop.foreground = 123
        app._manual_input(0, "press", 150, 110)
        self.assertTrue(app.drawing)
        app.desktop.foreground = 42
        for point in [(210, 110), (210, 155), (150, 155)]:
            app._manual_input(0, "move", *point)
        app._manual_input(0, "release", 150, 155)
        self.assertGreater(app.runner.finish()["count"], 0)

    def test_manual_playback_stops_if_browser_loses_focus(self):
        app = app_state()
        app.manual_window = 42
        app.trace = [(150, 110), (210, 110), (210, 155), (150, 155)]
        app._finish_manual()
        app.desktop.foreground = 99
        with self.assertRaisesRegex(RuntimeError, "focus changed"):
            app.runner.finish()
        self.assertEqual(app.desktop.clicks, [])
        self.assertFalse(app.root.hidden)

    def test_old_mouse_epoch_and_playback_mouse_callbacks_are_ignored(self):
        app = app_state()
        app._gate_manual()
        app._mouse_event(("press", 150, 100))
        self.assertTrue(app.events.empty())
        app._manual_gate.set()
        app._manual_input(0, "press", 150, 100)
        self.assertFalse(app.drawing)

    def test_listener_keeps_old_epoch_if_stop_and_rearm_happen_before_queue_put(self):
        app = app_state()
        entered, proceed = threading.Event(), threading.Event()
        records = []
        class SlowQueue:
            def put(self, item):
                entered.set()
                if not proceed.wait(2):
                    raise RuntimeError("test timeout")
                records.append(item)
        app.events = SlowQueue()
        thread = threading.Thread(target=app._mouse_event, args=(("press", 150, 100),))
        thread.start()
        self.assertTrue(entered.wait(2))
        app._gate_manual()
        app._sync_manual()
        proceed.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(records[0][1], 0)
        app._manual_input(*records[0][1:])
        self.assertFalse(app.drawing)

    def test_escape_prevents_rearming_or_release_submission_before_queue_is_polled(self):
        app = app_state()
        app._manual_input(0, "press", 150, 110)
        app._hotkey("stop")
        app._sync_manual()
        self.assertFalse(app._manual_gate.is_set())
        app._manual_input(0, "release", 190, 140)
        self.assertEqual(app.runner.jobs, [])

    def test_disable_cancels_drawing_and_removes_catchers_with_panel_visible(self):
        app = app_state()
        app._manual_input(0, "press", 150, 110)
        catcher = app.input_surface = Surface()
        catcher.show()
        app.stop()
        self.assertTrue(app.disabled.get())
        self.assertFalse(app._manual_gate.is_set())
        self.assertEqual(app.root.callbacks, {})
        self.assertTrue(catcher.closed)
        self.assertEqual(app.desktop.captures, [])

    def test_detection_hides_only_temporary_layers_and_keeps_controls_visible(self):
        app = app_state("semi-auto")
        app.overlay.show()
        app.input_surface = Surface()
        app.input_surface.show()
        app.input_kind = "masks"
        model = Mock()
        def recognize(*args, **kwargs):
            self.assertFalse(app.root.hidden, "Panel must remain visible throughout inference")
            return []
        model.automatic.side_effect = recognize
        app._get_recognizer = Mock(return_value=model)
        app.detect()
        app.root.run_next()
        app.checkpoint.get = Mock(side_effect=AssertionError("Tk queried by worker"))
        result = app.runner.finish()
        self.assertEqual(result["rgb"].shape, (70, 80, 3))
        self.assertEqual(len(app.desktop.captures), 2)
        model.automatic.assert_called_once()
        self.assertEqual(app.desktop.clicks, [])

    def test_detection_rejects_panel_inside_roi_without_hiding_it(self):
        app = app_state("semi-auto")
        app.root.bounds = (155, 115, 10, 10)  # misses the ROI center and corners
        app.detect()
        self.assertIn("Move the SARU panel", app.status.get())
        self.assertEqual(app.runner.jobs, [])
        self.assertEqual(app.desktop.captures, [])
        self.assertFalse(app.root.hidden)

    def test_panel_moved_into_roi_during_recognition_invalidates_capture(self):
        app = app_state("semi-auto")
        def recognize(*args, **kwargs):
            app.root.bounds = (155, 115, 10, 10)
            app._remember_panel()
            return []
        app._get_recognizer = Mock(return_value=SimpleNamespace(automatic=recognize))
        app.detect()
        app.root.run_next()
        with self.assertRaisesRegex(ValueError, "Move the SARU panel"):
            app.runner.finish()
        self.assertEqual(len(app.desktop.captures), 1)
        self.assertFalse(app.root.hidden)

    def test_all_temporary_layer_hides_leave_control_panel_visible(self):
        app = app_state()
        app.overlay.show()
        catcher = app.input_surface = Surface()
        catcher.show()
        app._hide_overlays()
        self.assertFalse(app.root.hidden)
        self.assertFalse(app.overlay.visible)
        self.assertTrue(catcher.closed)

    def test_clicking_mask_interior_uses_screen_coordinates_without_a_review_window(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:40, 15:45] = True
        mask[20:25, 20:25] = False
        candidate = make_candidate(mask)
        app.candidates = [candidate]
        app._start_model_picks = Mock()
        app._semi_click(162, 122)  # hole
        app._start_model_picks.assert_not_called()
        app._semi_click(170, 130)
        app._start_model_picks.assert_called_once_with([candidate])

    def test_mask_selection_waits_for_release_and_rejects_drags(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:40, 15:45] = True
        candidate = make_candidate(mask)
        app.candidates = [candidate]
        app.input_kind = "masks"
        app._start_model_picks = Mock()
        app._input_press(170, 130)
        app._start_model_picks.assert_not_called()
        app._input_release(171, 130)
        app._start_model_picks.assert_called_once_with([candidate])
        app._start_model_picks.reset_mock()
        app._input_press(160, 120)
        app._input_move(180, 130)
        app._input_release(180, 130)
        app._start_model_picks.assert_not_called()

    def test_semi_pick_checks_scene_before_any_click_and_never_advances(self):
        app = app_state("semi-auto")
        candidate = make_candidate(np.ones((70, 80), bool))
        app.desktop.rgb[100:170, 140:220] = 0
        app._start_model_picks([candidate])
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "changed before picking"):
            app.runner.finish()
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(app.desktop.keys, [])

    def test_successful_semi_pick_updates_reference_and_removes_only_used_region(self):
        app = app_state("semi-auto")
        left = np.zeros((70, 80), bool)
        right = left.copy()
        left[10:40, 5:30], right[10:40, 45:75] = True, True
        a, b = make_candidate(left), make_candidate(right)
        app.candidates = [a, b]
        app._start_model_picks([a])
        app.root.run_next()
        result = app.runner.finish()
        with patch("screen_app.InputSurface", Surface):
            app._result("model picks", result)
        self.assertEqual([c.id for c in app.candidates], [b.id])
        np.testing.assert_array_equal(app.reference, result["reference"])
        self.assertEqual(app.desktop.keys, [])
        self.assertTrue(app.input_surface.visible)
        self.assertTrue(app.overlay.visible)

    def test_post_pick_background_change_never_becomes_a_new_trusted_reference(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:40, 5:30] = True
        selected = make_candidate(mask)
        other = mask.copy()
        other[:, :] = False
        other[10:40, 45:75] = True
        app.candidates = [selected, make_candidate(other)]
        original_click = app.desktop.click
        def change_frame(x, y):
            original_click(x, y)
            app.desktop.rgb[100:170, 140:220] = 0
        app.desktop.click = change_frame
        app._start_model_picks([selected])
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "changed during picking"):
            app.runner.finish()
        self.assertTrue(np.all(app.reference == 100))
        app._error("The micrograph changed during picking. Detect again.")
        self.assertIsNone(app.reference)
        self.assertEqual(app.candidates, [])
        self.assertEqual(app.desktop.keys, [])

    def test_periodic_scene_check_stops_clicks_after_the_image_changes(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:60, 5:40] = True
        selected = make_candidate(mask)
        original_click = app.desktop.click
        def change_frame(x, y):
            original_click(x, y)
            app.desktop.rgb[100:170, 140:220] = 0
        app.desktop.click = change_frame
        app._start_model_picks([selected])
        app.root.run_next()
        with patch("screen_app.time.monotonic", side_effect=[0.0, 0.1, 1.0, 1.1, 2.0]):
            with self.assertRaisesRegex(RuntimeError, "changed during picking"):
                app.runner.finish()
        self.assertEqual(len(app.desktop.clicks), 1)

    def test_insufficient_background_clears_masks_and_stops_auto_before_next_frame(self):
        app = app_state("auto")
        app.auto_remaining = 2
        app.candidates = [make_candidate(np.ones((70, 80), bool))]
        app._result("model picks", {"count": 10, "ids": set(), "reference": app.reference,
                                     "auto": True, "scene_verified": False})
        self.assertEqual(app.auto_remaining, 0)
        self.assertEqual(app.candidates, [])
        self.assertIsNone(app.reference)
        self.assertTrue(app.disabled.get())
        self.assertEqual(app.desktop.keys, [])

    def test_trace_rendering_is_coalesced_after_recording_all_mouse_points(self):
        app = app_state()
        app.drawing = True
        app.trace = [(150, 110)]
        app.manual_geometry = CaptureGeometry(0, 0, 480, 320)
        app.manual_window = 42
        for x in range(151, 190):
            app.events.put(("mouse", 0, "move", x, 110))
        app.poll()
        self.assertEqual(len(app.trace), 40)
        self.assertEqual(len(app.overlay.traces), 1)

    def test_auto_is_explicit_finite_and_arms_space_only_for_the_run(self):
        app = app_state("auto")
        app._start_detection = Mock()
        self.assertEqual(app.auto_remaining, 0)
        app.start_auto()
        self.assertEqual(app.auto_remaining, 1)
        app._start_detection.assert_called_once_with(auto=True)
        app.hotkeys.enable_space_pause.assert_called_with(True)
        app._result("model picks", {"count": 10, "ids": set(), "reference": app.reference, "auto": True})
        self.assertEqual(app.auto_remaining, 0)
        app.hotkeys.enable_space_pause.assert_called_with(False)
        self.assertEqual(app.desktop.keys, [])

    def test_no_proposals_stop_auto_before_navigation(self):
        app = app_state("auto")
        app.auto_remaining = 2
        with self.assertRaisesRegex(RuntimeError, "No proposals"):
            app._result("recognition", {"rgb": app.rgb, "candidates": [], "auto": True, "prompt": False})
        self.assertEqual(app.desktop.keys, [])
        self.assertEqual(app.desktop.clicks, [])

    def test_next_frame_rejects_unchanged_image(self):
        app = app_state("auto")
        app.auto_remaining = 1
        app.auto_history = [app.reference.copy()]
        app._next_frame()
        with self.assertRaisesRegex(RuntimeError, "did not change"):
            app.runner.finish()
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])

    def test_pause_before_worker_submission_is_retained(self):
        app = app_state("auto")
        app.auto_remaining = 1
        app.transition = True
        app.pause()
        app._submit("test", lambda token: None)
        self.assertTrue(app.active_token.paused)
        app.pause()
        self.assertFalse(app.active_token.paused)

    def test_auto_capture_rechecks_the_image_between_navigation_and_recognition(self):
        app = app_state("auto")
        app._get_recognizer = Mock()
        expected = np.zeros_like(app.rgb)
        app._start_detection(auto=True, expected=expected)
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "between automatic frames"):
            app.runner.finish()
        app._get_recognizer.assert_not_called()

    def test_stop_during_worker_cancels_token_and_preserves_disabled_state(self):
        app = app_state()
        app._submit("test", lambda token: None)
        token = app.active_token
        app.stop()
        self.assertTrue(token.cancelled)
        self.assertTrue(app.disabled.get())
        self.assertFalse(app._manual_gate.is_set())
        app.hotkeys.enable_space_pause.assert_called_with(False)


if __name__ == "__main__":
    unittest.main()
