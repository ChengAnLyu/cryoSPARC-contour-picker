"""Live-screen controller behavior without screenshots, hooks, or real clicks."""
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from execution import CancellationToken, CancelledOperation, CaptureGeometry, TaskResult
from manual_sampling import legacy_contour_points, mask_contour_points
from recognition import RecognitionConfig, make_candidate
from micrograph_navigation import (NoNewMicrographError, MicrographLoadTimeoutError,
                                   wait_for_next_micrograph)
from screen_app import (ScreenPickerApp, screen_changed_fraction, screen_mask_at, screen_model_points, screen_model_batches,
                        screen_background_guard, screen_background_changed, screen_region_overlaps_panel,
                        DEFAULT_CLICK_DELAY)


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
        self.destroyed = False
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
        self.destroyed = True
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
        self.masks = []
    def draw_trace(self, points):
        self.traces.append(tuple(points))
    def begin_trace(self, points):
        self.clear()
        self.draw_trace(points)
    def begin_selection(self):
        self.hide()
        self.clear()
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
        return not self.busy


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
    app._picking_menu = None
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
    app.device, app.grid_side, app.click_delay = Variable("cpu"), Variable(8), Variable(DEFAULT_CLICK_DELAY)
    app.automatic_profile = Variable("capsid")
    app.min_capsid_area = Variable("3500")
    app.frames, app.wait_seconds = Variable(0), Variable(10.0)
    app.closed = app.transition = app.paused = False
    app._fit_after = None
    app._poll_after = app._shutdown_after = None
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
    app.auto_completed = 0
    app.picked_points = set()
    app.recognizer = app.recognizer_config = None
    app.input_kind = app.input_start = None
    app.log_path = Path("test.log")
    return app


class ScreenGeometryTests(unittest.TestCase):
    def test_size_setting_retires_stale_contours_but_preserves_actual_picks(self):
        app = app_state("semi-auto")
        app.min_capsid_area.set("0")
        previous = app._model_config()
        candidate = make_candidate(np.ones((70, 80), bool))
        app.candidates, app.alternatives = [candidate], [candidate]
        app.overlay.draw_masks([candidate], app.geometry)
        app.picked_points.add((12, 18))
        reference = app.reference
        app.min_capsid_area.set("5000")
        self.assertTrue(app._apply_model_settings(previous))
        self.assertEqual(app.candidates, [])
        self.assertEqual(app.alternatives, [])
        self.assertEqual(app.overlay.masks, [])
        self.assertEqual(app.picked_points, {(12, 18)})
        self.assertIs(app.reference, reference)
        self.assertEqual(app._model_config().min_capsid_area_px, 5000)
        self.assertIn("Detect", app.status.get())

    def test_invalid_size_setting_cannot_retire_or_accept_current_contours(self):
        app = app_state("semi-auto")
        previous = app._model_config()
        candidate = make_candidate(np.ones((70, 80), bool))
        app.candidates = [candidate]
        for invalid in ("-1", "nan", "4999.5", ""):
            with self.subTest(invalid=invalid):
                app.min_capsid_area.set(invalid)
                with self.assertRaises(ValueError):
                    app._apply_model_settings(previous)
                self.assertEqual(app.candidates, [candidate])

    def test_previous_pick_coordinates_reject_fractional_nan_and_out_of_bounds(self):
        geometry = CaptureGeometry(100, 100, 80, 70)
        candidate = make_candidate(np.ones((70, 80), bool))
        for point in ((1.5, 2), (float("nan"), 2), (float("inf"), 2), (-1, 2),
                      (80, 2), (1, 70), (True, 2), (1, 2, 3), ("bad", 2)):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "Previously placed picks"):
                    screen_model_batches([candidate], geometry, 13, [point])
                with self.assertRaisesRegex(ValueError, "Previously placed picks"):
                    screen_background_guard([], geometry, 13, [point])

    def test_previous_marker_guard_excludes_only_near_known_picks(self):
        geometry = CaptureGeometry(0, 0, 100, 100)
        guard = screen_background_guard([], geometry, 13, [(20, 20)])
        self.assertFalse(guard[20, 20])
        self.assertFalse(guard[20, 33])
        self.assertTrue(guard[20, 34])
        self.assertTrue(guard[80, 80])
        self.assertTrue(np.all(screen_background_guard([], geometry, 13)))

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

    def test_model_contours_are_unique_and_remember_previously_clicked_points(self):
        region = CaptureGeometry(100, 200, 40, 30)
        a = make_candidate(np.ones((30, 40), bool))
        local, screen = screen_model_points([a, a], region, 5)
        self.assertEqual(len(screen), len({tuple(p) for p in screen}))
        self.assertTrue(all(region.contains(*point) for point in screen))
        kept, _ = screen_model_points([a], region, 5, picked=[tuple(local[0])])
        self.assertEqual(len(kept), len(local)-1)

    def test_each_contour_keeps_its_own_manual_lattice_in_detection_order(self):
        region = CaptureGeometry(100, 200, 80, 60)
        first, second = np.zeros((60, 80), bool), np.zeros((60, 80), bool)
        first[7:35, 41:68] = True
        second[11:39, 5:30] = True
        a, b = make_candidate(first), make_candidate(second)
        batches = screen_model_batches([a, b], region, 13)
        self.assertEqual([batch.candidate_id for batch in batches], [a.id, b.id])
        np.testing.assert_array_equal(batches[0].local, mask_contour_points(first, 13))
        np.testing.assert_array_equal(batches[1].local, mask_contour_points(second, 13))
        np.testing.assert_array_equal(batches[0].screen, batches[0].local + [100, 200])
        np.testing.assert_array_equal(batches[1].screen, batches[1].local + [100, 200])
        np.testing.assert_array_equal(batches[0].local[0], [41, 7])
        np.testing.assert_array_equal(batches[1].local[0], [5, 11])

    def test_overlapping_contours_keep_order_without_repeated_exact_clicks(self):
        region = CaptureGeometry(100, 200, 70, 60)
        first, second = np.zeros((60, 70), bool), np.zeros((60, 70), bool)
        first[5:36, 5:36] = True
        second[5:36, 18:55] = True  # same row phase and x phase modulo spacing
        a, b = make_candidate(first), make_candidate(second)
        remembered = (5, 5)
        batches = screen_model_batches([a, b], region, 13, picked=[remembered])
        first_points = [tuple(point) for point in batches[0].local]
        second_points = [tuple(point) for point in batches[1].local]
        self.assertNotIn(remembered, first_points + second_points)
        self.assertFalse(set(first_points) & set(second_points))
        expected_second = [tuple(point) for point in mask_contour_points(second, 13)
                           if tuple(point) not in set(first_points + [remembered])]
        self.assertEqual(second_points, expected_second)

    def test_different_contour_phases_cannot_make_nearly_coincident_picks(self):
        region = CaptureGeometry(0, 0, 160, 120)
        first, second = np.zeros((120, 160), bool), np.zeros((120, 160), bool)
        first[20:80, 20:60], second[20:80, 34:74] = True, True
        a, b = make_candidate(first), make_candidate(second)
        batches = screen_model_batches([a, b], region, 13)
        original_first = mask_contour_points(first, 13)
        original_second = mask_contour_points(second, 13)
        np.testing.assert_array_equal(batches[0].local, original_first)
        self.assertGreater(len(batches[1].local), 0)
        self.assertLess(len(batches[1].local), len(original_second))
        retained = {tuple(point) for point in batches[1].local}
        self.assertEqual([tuple(point) for point in batches[1].local],
                         [tuple(point) for point in original_second if tuple(point) in retained])
        distances = np.sqrt(np.sum((batches[0].local[:, None, :] - batches[1].local[None, :, :])**2, axis=2))
        self.assertGreaterEqual(float(distances.min()), float(np.hypot(6, 11))-1e-6)
        # Repeating the second contour as a later semi-auto selection applies
        # exactly the same guard to actual positions from the first selection.
        later = screen_model_batches([b], region, 13, picked=map(tuple, original_first))
        np.testing.assert_array_equal(later[0].local, batches[1].local)

    def test_manual_lattice_neighbor_at_equal_distance_is_retained(self):
        region = CaptureGeometry(0, 0, 50, 50)
        mask = np.zeros((50, 50), bool)
        mask[31, 26] = True
        batches = screen_model_batches([make_candidate(mask)], region, 13, picked=[(20, 20)])
        np.testing.assert_array_equal(batches[0].local, [[26, 31]])

    def test_contour_plan_reports_raw_sites_and_actual_history_exclusions(self):
        geometry = CaptureGeometry(0, 0, 160, 120)
        original, shifted = np.zeros((120, 160), bool), np.zeros((120, 160), bool)
        original[10:110, 20:120] = True
        shifted[12:112, 22:122] = True
        original_points = mask_contour_points(original, 20)
        plan = screen_model_batches([make_candidate(shifted)], geometry, 20,
                                    picked=map(tuple, original_points))[0]
        self.assertEqual(plan.raw_count, 30)
        self.assertEqual(plan.blocked_count, 30)
        self.assertEqual(len(plan.local), 0)
        self.assertEqual(len(plan.screen), 0)

    def test_click_limit_applies_after_filtering_already_picked_positions(self):
        region = CaptureGeometry(0, 0, 80, 70)
        mask = np.zeros((70, 80), bool)
        mask[5:55, 5:65] = True
        original = mask_contour_points(mask, 13)
        self.assertGreater(len(original), 2)
        batches = screen_model_batches([make_candidate(mask)], region, 13,
                                       picked=map(tuple, original[:-2]), max_points=2)
        np.testing.assert_array_equal(batches[0].local, original[-2:])

    def test_total_contour_limit_is_validated_before_a_plan_is_returned(self):
        region = CaptureGeometry(0, 0, 80, 40)
        first, second = np.zeros((40, 80), bool), np.zeros((40, 80), bool)
        first[3:35, 3:32], second[3:35, 45:74] = True, True
        a, b = make_candidate(first), make_candidate(second)
        per_contour = len(mask_contour_points(first, 5))
        with self.assertRaisesRegex(ValueError, "across the contours"):
            screen_model_batches([a, b], region, 5, max_points=per_contour)

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

    def test_live_detection_passes_progress_only_to_cryolo_without_worker_tk_access(self):
        for backend in ("sam", "cryolo"):
            with self.subTest(backend=backend):
                app = app_state("semi-auto")
                app._model_config = Mock(return_value=RecognitionConfig(
                    "model.onnx" if backend == "cryolo" else "model.pth", backend=backend))
                def recognize(rgb, **options):
                    self.assertEqual("progress" in options, backend == "cryolo")
                    if backend == "cryolo":
                        options["progress"]("Refining contours")
                    self.assertFalse(options["cancel"]())
                    return []
                app._get_recognizer = Mock(return_value=SimpleNamespace(automatic=recognize))
                app.detect()
                app.root.run_next()
                with patch.object(app.status, "set", side_effect=AssertionError("Worker touched Tk")):
                    result = app.runner.finish()
                self.assertEqual(result["candidates"], [])
                phases = []
                while not app.events.empty():
                    phases.append(app.events.get_nowait()[2])
                self.assertIn("Checking screen alignment", phases)
                self.assertIn("Refining contours" if backend == "cryolo" else "Running SAM", phases)

    def test_recognition_phase_keeps_pause_and_stop_messages_authoritative(self):
        app = app_state("semi-auto")
        with patch("screen_app.time.perf_counter", return_value=100):
            app._submit("recognition", lambda token: None)
        token = app.active_token
        app.events.put(("recognition_phase", token, "Refining contours"))
        with patch("screen_app.time.perf_counter", return_value=109):
            app.poll()
        self.assertIn("Refining contours\n9s elapsed", app.status.get())
        app.pause()
        paused = app.status.get()
        app.events.put(("recognition_phase", token, "Finishing contours"))
        with patch("screen_app.time.perf_counter", return_value=110):
            app.poll()
        self.assertEqual(app.status.get(), paused)
        app.stop()
        stopped = app.status.get()
        app.events.put(("recognition_phase", token, "Late phase"))
        app.poll()
        self.assertEqual(app.status.get(), stopped)
        self.assertIsNone(app._recognition_progress)

    def test_previous_recognition_phase_cannot_replace_next_auto_job_in_same_generation(self):
        app = app_state("auto")
        app._submit("recognition", lambda token: None)
        previous = app.active_token
        for _ in range(4000):
            app.events.put(("mouse", -1, "move", 150, 120))
        app.events.put(("recognition_phase", previous, "Old job phase"))
        app.runner.results.append(TaskResult(previous, value={}))
        app._result = lambda kind, value: app._submit("recognition", lambda token: None)
        app.poll()
        self.assertIsNot(previous, app.active_token)
        self.assertEqual(previous.generation, app.active_token.generation)
        app.poll()
        self.assertNotIn("Old job phase", app.status.get())
        self.assertEqual(app._recognition_progress.phase, "Starting detector")

    def test_failed_recognition_does_not_leave_elapsed_status_running(self):
        app = app_state("semi-auto")
        app._submit("recognition", lambda token: None)
        token = app.active_token
        app.events.put(("recognition_phase", token, "Loading model"))
        app.runner.results.append(TaskResult(token, error=RuntimeError("model runtime failed")))
        with self.assertLogs("saru.screen", level="ERROR"):
            app.poll()
        app.events.put(("recognition_phase", token, "Late phase"))
        app.poll()
        self.assertEqual(app.status.get(), "model runtime failed")
        self.assertIsNone(app._recognition_progress)

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

    def test_manual_press_prepares_clean_trace_before_mapping_overlay(self):
        app = app_state()
        old = ((150, 110), (210, 110), (210, 155), (150, 155))
        app.overlay.traces = [old]
        previous_show = app.overlay.show
        def show_only_clean_trace():
            self.assertEqual(app.overlay.traces, [((180, 130),)])
            previous_show()
        app.overlay.show = show_only_clean_trace
        app._manual_input(0, "press", 180, 130)
        self.assertTrue(app.overlay.visible)
        self.assertEqual(app.trace, [(180, 130)])
        self.assertFalse(app.root.hidden)

    def test_manual_uses_shared_speed_without_reading_model_settings(self):
        app = app_state()
        app.click_delay.set(.005)
        app.grid_side.set("unfinished model setting")
        app.checkpoint.set("")
        self.assertEqual(app._live_ready(), .005)

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
        self.assertEqual(playback.call_args.kwargs["delay"], DEFAULT_CLICK_DELAY)
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

    def test_three_semi_auto_selections_survive_delayed_previous_pick_markers(self):
        app = app_state("semi-auto")
        masks = [np.zeros((70, 80), bool) for _ in range(3)]
        masks[0][12:38, 4:20] = True
        masks[1][17:45, 31:47] = True
        masks[2][10:40, 60:77] = True
        candidates = [make_candidate(mask) for mask in masks]
        app.candidates = list(candidates)
        y_grid, x_grid = np.indices((70, 80))
        observed_clicks = []
        with patch("screen_app.InputSurface", Surface):
            app._present_masks()
            for number, candidate in enumerate(candidates):
                if number:
                    # Browser markers arrive after the previous completion
                    # screenshot, reproducing the old second-contour refusal.
                    for x, y in app.picked_points:
                        marker = (x_grid-x)**2 + (y_grid-y)**2 <= 6**2
                        app.desktop.rgb[100:170, 140:220][marker] = (0, 255, 0)
                    self.assertGreater(screen_changed_fraction(app.reference, app.desktop.rgb[100:170, 140:220]), .03)
                expected = screen_model_batches([candidate], app.geometry, 13, app.picked_points)[0].screen
                x, y = candidate.bbox[:2]
                app._input_press(140+x+1, 100+y+1)
                app._input_release(140+x+1, 100+y+1)
                for _ in range(4):
                    if app.runner.busy:
                        break
                    app.root.run_next()
                self.assertTrue(app.runner.busy)
                result = app.runner.finish()
                app.runner.results.append(TaskResult(app.active_token, value=result))
                app.poll()
                observed_clicks.extend(tuple(point) for point in expected)
                self.assertEqual(app.desktop.clicks, observed_clicks)
                self.assertFalse(app.disabled.get())
                self.assertEqual([item.id for item in app.candidates], [item.id for item in candidates[number+1:]])
                self.assertEqual([item.id for item in app.overlay.masks],
                                 [item.id for item in candidates[number+1:]])
                self.assertEqual(app.overlay.visible, number < len(candidates)-1)
                self.assertFalse(app.root.hidden)
                self.assertFalse(app.root.destroyed)
            self.assertIsNone(app.input_surface)
        self.assertEqual(app.desktop.keys, [])

    def test_eight_semi_selections_retire_exhausted_overlaps_and_continue_after_zero_plan(self):
        app = app_state("semi-auto")
        app.spacing.set(20)
        app.click_delay.set(0)
        app.geometry = CaptureGeometry(726, 421, 696, 499)
        app.desktop.rgb = np.full((1080, 1920, 3), 100, np.uint8)
        app.reference = app.rgb = np.full((499, 696, 3), 100, np.uint8)
        candidates = []
        for y in (60, 300):
            for x in (25, 190, 355, 520):
                mask = np.zeros((499, 696), bool)
                mask[y:y+100, x:x+100] = True
                candidates.append(make_candidate(mask))
        duplicates = []
        for source in (candidates[0], candidates[6]):
            x, y = source.bbox[:2]
            mask = np.zeros((499, 696), bool)
            mask[y+2:y+102, x+2:x+102] = True
            duplicates.append(make_candidate(mask))
        app.candidates = list(candidates) + duplicates
        expected_clicks = []
        yy, xx = np.indices((499, 696))

        def select(candidate):
            self.assertIsNotNone(app.input_surface)
            self.assertTrue(app.input_surface.visible)
            x, y = candidate.bbox[:2]
            app._input_press(726+x+1, 421+y+1)
            app._input_release(726+x+1, 421+y+1)
            for _ in range(10):
                if app.runner.busy:
                    break
                app.root.run_next()
            self.assertTrue(app.runner.busy)
            result = app.runner.finish()
            app.runner.results.append(TaskResult(app.active_token, value=result))
            app.poll()
            self.assertFalse(app.disabled.get())
            self.assertFalse(app.root.hidden)
            self.assertFalse(app.root.destroyed)
            return result

        with patch("screen_app.InputSurface", Surface):
            app._present_masks()
            for number, candidate in enumerate(candidates):
                if number:
                    # The browser can paint older markers after the previous
                    # completion capture, including after the fifth contour.
                    for x, y in app.picked_points:
                        marker = (xx-x)**2 + (yy-y)**2 <= 14**2
                        app.desktop.rgb[421:920, 726:1422][marker] = (0, 255, 0)
                if number == 5:
                    # A fresh Detect can propose an already picked region.
                    # The empty-plan detour must explain itself and restore
                    # selection for all three untouched regions afterwards.
                    app._result("recognition", {"rgb": app.reference.copy(), "auto": False,
                                "prompt": False, "candidates": [duplicates[0], *app.candidates]})
                    before = set(app.picked_points)
                    empty = select(duplicates[0])
                    self.assertEqual(empty["count"], 0)
                    self.assertEqual(empty["planning"], {"spacing": 20, "raw_grid_points": 30,
                                     "blocked_grid_points": 30, "eligible_grid_points": 0, "sent_points": 0})
                    self.assertEqual(app.picked_points, before)
                    self.assertIn("already covered", app.status.get())
                    self.assertIsNotNone(app.input_surface)
                result = select(candidate)
                expected_clicks.extend(tuple(point) for point in
                                       mask_contour_points(candidate.mask, 20) + [726, 421])
                self.assertEqual(result["count"], 30)
                self.assertEqual(result["planning"]["raw_grid_points"], 30)
                self.assertEqual(result["planning"]["blocked_grid_points"], 0)
                self.assertEqual(result["planning"]["sent_points"], 30)
                self.assertEqual(app.desktop.clicks, expected_clicks)
                self.assertEqual(len(app.picked_points), 30 * (number+1))
                remaining_ids = {item.id for item in app.candidates}
                self.assertNotIn(duplicates[0].id, remaining_ids)
                if number < 6:
                    # Both overlapping, unpicked choices must stay available.
                    # Planning one must not reserve points against the other.
                    records = {item["candidate_id"]: item for item in result["remaining_plans"]}
                    for unpicked in (candidates[6], duplicates[1]):
                        self.assertIn(unpicked.id, remaining_ids)
                        self.assertEqual(records[unpicked.id]["eligible_grid_points"], 30)
                else:
                    self.assertNotIn(duplicates[1].id, remaining_ids)
            self.assertEqual(len(app.desktop.clicks), 240)
            self.assertEqual(app.candidates, [])
            self.assertIsNone(app.input_surface)
            self.assertFalse(app.overlay.visible)
            self.assertIn("No selectable contours remain", app.status.get())
            self.assertIn("Next micrograph", app.status.get())
            self.assertNotIn("Click another green", app.status.get())
        self.assertEqual(app.desktop.keys, [])

    def test_spacing_changed_during_playback_does_not_apply_stale_exhaustion_plan(self):
        app = app_state("semi-auto")
        first, shifted = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:45, 10:40] = True
        shifted[16:46, 11:41] = True
        selected, remaining = make_candidate(first), make_candidate(shifted)
        app.candidates = [selected, remaining]
        app._start_model_picks([selected])
        app.root.run_next()
        result = app.runner.finish()
        self.assertEqual(result["exhausted_ids"], {remaining.id})
        app.spacing.set(5)
        app.runner.results.append(TaskResult(app.active_token, value=result))
        with patch("screen_app.InputSurface", Surface):
            app.poll()
        self.assertEqual([item.id for item in app.candidates], [remaining.id])
        self.assertEqual(app.picked_points, set(result["picked"]))
        self.assertTrue(app.input_surface.visible)
        self.assertIn("1 contour ready", app.status.get())

    def test_empty_grid_is_explained_separately_from_an_already_picked_contour(self):
        app = app_state("semi-auto")
        app.spacing.set(20)
        mask = np.zeros((70, 80), bool)
        yy, xx = np.indices((10, 10))
        mask[10:20, 10:20] = xx + yy >= 9
        candidate = make_candidate(mask)
        app.candidates = [candidate]
        app._start_model_picks([candidate])
        app.root.run_next()
        result = app.runner.finish()
        self.assertEqual(result["planning"], {"spacing": 20, "raw_grid_points": 0,
                         "blocked_grid_points": 0, "eligible_grid_points": 0, "sent_points": 0})
        app.runner.results.append(TaskResult(app.active_token, value=result))
        with patch("screen_app.InputSurface", Surface):
            app.poll()
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(app.picked_points, set())
        self.assertIn("No grid sites at spacing 20", app.status.get())
        self.assertIn("No selectable contours remain", app.status.get())
        self.assertNotIn("already covered", app.status.get())

    def test_second_contour_still_rejects_global_image_change_before_clicking(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[15:45, 50:75] = True
        app.picked_points = {(8, 15), (21, 15), (14, 26)}
        app.desktop.rgb[100:170, 140:220] = 0
        app._start_model_picks([make_candidate(mask)])
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "changed before picking"):
            app.runner.finish()
        self.assertEqual(app.desktop.clicks, [])

    def test_completion_handoff_records_final_pick_before_reopening_contours(self):
        app = app_state("semi-auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[10:40, 5:30], second[10:40, 45:70] = True, True
        a, b = make_candidate(first), make_candidate(second)
        app.candidates = [a, b]
        app.job_kind = "model picks"
        final_point = (18, 32)
        result = TaskResult(CancellationToken(app.runner.generation), value={
            "count": 1, "ids": {a.id}, "reference": app.reference.copy(), "auto": False})
        def collect_at_handoff():
            app.events.put(("picked", app.runner.generation, app.frame_id, final_point, 1, 1, 1, 1))
            app.events.put(("contour_done", app.runner.generation, app.frame_id, a.id))
            return [result]
        app.runner.poll_results = collect_at_handoff
        with patch.object(app, "_present_masks", side_effect=lambda: self.assertIn(final_point, app.picked_points)):
            app.poll()
        self.assertEqual(app.picked_points, {final_point})
        self.assertEqual(app.candidates, [b])

    def test_result_snapshot_prevents_lost_picks_when_event_budget_is_full(self):
        app = app_state("semi-auto")
        app.job_kind = "model picks"
        points = ((5, 10), (18, 10), (11, 21))
        for _ in range(4001):
            app.events.put(("mouse", -1, "move", 150, 120))
        app.runner.results.append(TaskResult(CancellationToken(app.runner.generation), value={
            "count": len(points), "ids": set(), "reference": app.reference.copy(),
            "auto": False, "frame_id": app.frame_id, "picked": points}))
        with patch.object(app, "_present_masks", side_effect=lambda: self.assertEqual(app.picked_points, set(points))):
            app.poll()
        self.assertEqual(app.events.qsize(), 1)

    def test_escape_during_completion_handoff_does_not_restore_stale_contours(self):
        app = app_state("semi-auto")
        app.job_kind = "model picks"
        app.runner.results.append(TaskResult(CancellationToken(app.runner.generation), value={
            "count": 1, "ids": set(), "reference": app.reference.copy(), "auto": False}))
        app.events.put(("hotkey", "stop"))
        with patch.object(app, "_present_masks") as present:
            app.poll()
        present.assert_not_called()
        self.assertTrue(app.disabled.get())

    def test_unexpected_event_error_cancels_work_but_keeps_polling(self):
        app = app_state()
        app._poll_after = app.root.after(30, app.poll)
        app.events.put(("mouse", 0, "press", 150, 120))
        with patch.object(app, "_manual_input", side_effect=RuntimeError("broken queued input")):
            with self.assertLogs("saru.screen", level="ERROR"):
                app.poll()
        self.assertTrue(app.disabled.get())
        self.assertFalse(app.root.destroyed)
        self.assertEqual(len(app.root.callbacks), 1)
        self.assertIn(app._poll_after, app.root.callbacks)
        app.root.run_next()
        self.assertEqual(len(app.root.callbacks), 1)

    def test_close_cancels_poll_before_destroying_idle_window(self):
        app = app_state()
        app._poll_after = app.root.after(30, app.poll)
        app.close()
        self.assertEqual(app.root.callbacks, {})
        self.assertTrue(app.root.destroyed)
        self.assertIsNone(app._poll_after)

    def test_close_releases_real_tcl_variable_trace_callbacks(self):
        import gc
        import tkinter as tk
        import weakref

        interpreter = tk.Tcl()  # No display or native windows are created.
        app = app_state()
        app.tk = tk
        app.status = tk.StringVar(master=interpreter, value="ready")
        app.region_label = tk.StringVar(master=interpreter, value="region")
        variables = (app.status, app.region_label)
        app._variable_traces = [(variable, variable.trace_add("write", app._queue_panel_fit))
                                for variable in variables]
        owner = weakref.ref(app)
        self.assertTrue(all(variable.trace_info() for variable in variables))
        app.close()
        self.assertTrue(all(not variable.trace_info() for variable in variables))
        self.assertTrue(all(not variable._tclCommands for variable in variables))
        app.desktop.app = None  # Remove the fake desktop's deliberate owner link.
        del app
        gc.collect()
        # Tcl and its variables intentionally remain alive here; their former
        # registered callbacks must not keep the closed controller alive.
        self.assertIsNone(owner())

    def test_close_waits_for_worker_ownership_before_destroying_tk(self):
        app = app_state("semi-auto")
        app.runner.busy = True
        app._poll_after = app.root.after(30, app.poll)
        app.close()
        self.assertTrue(app.closed)
        self.assertFalse(app.root.destroyed)
        self.assertEqual(len(app.root.callbacks), 1)
        app.runner.busy = False
        app.root.run_next()
        self.assertTrue(app.root.destroyed)
        self.assertEqual(app.root.callbacks, {})

    def test_semi_release_fills_only_clicked_contour_on_its_manual_grid(self):
        app = app_state("semi-auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[10:40, 5:30], second[13:43, 45:75] = True, True
        second[23:33, 52:62] = False
        a, b = make_candidate(first), make_candidate(second)
        app.candidates = [a, b]
        app.input_kind = "masks"
        app.input_surface = Surface()
        app.input_surface.show()
        app._input_press(187, 115)
        self.assertEqual(app.runner.jobs, [])
        app._input_release(187, 115)
        app.root.run_next()
        result = app.runner.finish()
        expected = mask_contour_points(second, 13) + [140, 100]
        self.assertEqual(app.desktop.clicks, [tuple(point) for point in expected])
        self.assertEqual(result["contours"], 1)
        with patch("screen_app.InputSurface", Surface):
            app._result("model picks", result)
        self.assertEqual([candidate.id for candidate in app.candidates], [a.id])
        self.assertEqual([candidate.id for candidate in app.overlay.masks], [a.id])
        self.assertTrue(app.overlay.visible)
        self.assertTrue(app.input_surface.visible)
        self.assertEqual(app.desktop.keys, [])

    def test_auto_finishes_each_contour_before_next_and_advances_only_after_all(self):
        app = app_state("auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:40, 5:25], second[15:40, 45:65] = True, True
        a, b = make_candidate(first), make_candidate(second)
        app.candidates, app.auto_remaining = [a, b], 2
        app._next_frame = Mock()
        expected = [mask_contour_points(mask, 13) + [140, 100] for mask in (first, second)]
        app._start_model_picks(app.candidates, auto=True)
        app.root.run_next()
        with patch("screen_app.play_clicks", wraps=__import__("execution").play_clicks) as playback:
            result = app.runner.finish()
        self.assertEqual(playback.call_count, 2)
        for call, points in zip(playback.call_args_list, expected):
            np.testing.assert_array_equal(call.args[0], points)
        self.assertEqual(app.desktop.clicks, [tuple(point) for points in expected for point in points])
        self.assertEqual(result["count"], sum(map(len, expected)))
        self.assertEqual(result["contours"], 2)
        self.assertEqual(app.desktop.keys, [])
        app._next_frame.assert_not_called()
        app.poll()
        self.assertIn("Contour 2/2", app.status.get())
        self.assertEqual(app.candidates, [])
        self.assertEqual(len(app.picked_points), result["count"])
        app._result("model picks", result)
        app._next_frame.assert_called_once_with()
        self.assertEqual(app.auto_remaining, 1)

    def test_auto_marker_changes_do_not_stop_contours_or_prevent_navigation(self):
        app = app_state("auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:40, 5:25], second[15:40, 45:65] = True, True
        app.candidates, app.auto_remaining = [make_candidate(first), make_candidate(second)], 2
        expected = [tuple(point) for mask in (first, second)
                    for point in mask_contour_points(mask, 13) + [140, 100]]
        # Markers can paint before playback, throughout a contour and between
        # contours. Deliberately change far more than the old 3% threshold.
        app.desktop.rgb[100:170, 140:220] = 50
        click = app.desktop.click
        def paint_markers(x, y):
            click(x, y)
            app.desktop.rgb[100:170, 140:220] = 80 + len(app.desktop.clicks) * 5
        app.desktop.click = paint_markers
        with patch("screen_app.screen_background_changed",
                   side_effect=AssertionError("Auto playback must not compare screenshots")):
            app._start_model_picks(app.candidates, auto=True)
            app.root.run_next()
            result = app.runner.finish()
            self.assertEqual(app.desktop.clicks, expected)
            self.assertEqual(result["contours"], 2)
            self.assertFalse(result["scene_verified"])
            self.assertEqual(result["scene_check"], "skipped-auto")
            app._result("model picks", result)
            self.assertEqual(app.auto_remaining, 1)
            self.assertFalse(app.disabled.get())
            # More delayed drawing before Down must not be compared with the
            # post-pick reference. The freshly captured frame is the baseline.
            app.desktop.rgb[100:170, 140:220] = 200
            before = app.desktop.rgb[100:170, 140:220].copy()
            marker_only = before.copy()
            for x, y in app.picked_points:
                marker_only[max(0, y-2):y+3, max(0, x-2):x+3] = 0
            new = np.full_like(before, 20)
            captures = [marker_only, new, new, new, new]
            capture = app._capture_region
            def capture_after_down(geometry):
                if not app.desktop.keys:
                    return capture(geometry)
                return captures.pop(0) if captures else new.copy()
            app._capture_region = capture_after_down
            app.root.run_next()
            def wait_without_real_sleep(capture, previous, token, **kwargs):
                np.testing.assert_array_equal(previous, before)
                clock = [0.0]
                def sleep(seconds):
                    token.check_cancelled()
                    clock[0] += seconds
                with patch.object(token, "sleep", side_effect=sleep):
                    return wait_for_next_micrograph(capture, previous, token, **kwargs,
                        clock=lambda: clock[0], minimum_wait=.01,
                        settle_seconds=.02, poll_seconds=.01)
            with patch("screen_app.wait_for_next_micrograph", side_effect=wait_without_real_sleep):
                next_result = app.runner.finish()
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertTrue(next_result["advanced"])
        np.testing.assert_array_equal(next_result["rgb"], new)
        self.assertEqual(app.desktop.clicks, expected)
        app._start_detection = Mock()
        app._result("next frame", next_result)
        self.assertEqual(app.picked_points, set())
        self.assertTrue(app._start_detection.call_args.kwargs["auto"])
        np.testing.assert_array_equal(app._start_detection.call_args.kwargs["expected"], new)

    def test_auto_navigation_marker_only_change_does_not_authorize_another_frame(self):
        app = app_state("auto")
        app.auto_remaining = 2
        app.picked_points = {(20, 20)}
        # The last reference can predate a widespread marker redraw, but Down
        # must still produce a different background from its fresh baseline.
        app.desktop.rgb[100:170, 140:220] = 180
        before = app.desktop.rgb[100:170, 140:220].copy()
        marker_only = before.copy()
        marker_only[18:23, 18:23] = 0
        capture = app._capture_region
        app._capture_region = lambda geometry: marker_only.copy() if app.desktop.keys else capture(geometry)
        app._next_frame(auto=True)
        app.root.run_next()
        def wait_without_real_sleep(capture, previous, token, **kwargs):
            clock = [0.0]
            def sleep(seconds):
                token.check_cancelled()
                clock[0] += seconds
            with patch.object(token, "sleep", side_effect=sleep):
                return wait_for_next_micrograph(capture, previous, token, **kwargs,
                    clock=lambda: clock[0], minimum_wait=.01,
                    settle_seconds=.02, poll_seconds=.1)
        with patch("screen_app.wait_for_next_micrograph", side_effect=wait_without_real_sleep):
            result = app.runner.finish()
        self.assertFalse(result["advanced"])
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])
        app._start_detection = Mock()
        app._result("next frame", result)
        app._start_detection.assert_not_called()
        self.assertFalse(app.auto_remaining)
        self.assertTrue(app.disabled.get())

    def test_auto_playback_with_no_unmarked_background_still_completes(self):
        app = app_state("auto")
        app.candidates = [make_candidate(np.ones((70, 80), bool))]
        app.picked_points = {(x, y) for y in range(0, 70, 10) for x in range(0, 80, 10)}
        self.assertIsNone(screen_background_guard((), app.geometry, 13, app.picked_points))
        app.auto_remaining = 1
        app._start_model_picks(app.candidates, auto=True)
        app.root.run_next()
        result = app.runner.finish()
        self.assertEqual(result["count"], 0)
        self.assertFalse(result["scene_verified"])
        app._result("model picks", result)
        self.assertFalse(app.auto_remaining)
        self.assertFalse(app.disabled.get())
        self.assertIn("Auto finished", app.status.get())
        self.assertEqual(app.desktop.clicks, [])
        # Playback does not need unmarked background. Navigation still needs
        # enough of it to distinguish a real next frame from delayed markers.
        app._next_frame(auto=True)
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "Too little unchanged background"):
            app.runner.finish()
        self.assertEqual(app.desktop.keys, [])

    def test_auto_image_changes_do_not_disable_target_focus_checks(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[10:60, 5:40] = True
        app.candidates = [make_candidate(mask)]
        click = app.desktop.click
        def lose_focus(x, y):
            click(x, y)
            app.desktop.rgb[100:170, 140:220] = 0
            app.desktop.foreground = 77
        app.desktop.click = lose_focus
        app._start_model_picks(app.candidates, auto=True)
        app.root.run_next()
        with self.assertRaisesRegex(RuntimeError, "focus changed"):
            app.runner.finish()
        self.assertEqual(len(app.desktop.clicks), 1)
        self.assertEqual(app.desktop.keys, [])

    def test_later_invalid_contour_rejects_all_batches_before_clicking(self):
        app = app_state("auto")
        first = np.zeros((70, 80), bool)
        first[15:40, 5:25] = True
        candidates = [make_candidate(first), make_candidate(np.ones((20, 20), bool))]
        app._start_model_picks(candidates, auto=True)
        app.root.run_next()
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            app.runner.finish()
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(app.desktop.keys, [])

    def test_pause_between_contours_blocks_later_contour_until_resume(self):
        app = app_state("auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:40, 5:25], second[15:40, 45:65] = True, True
        app.candidates = [make_candidate(first), make_candidate(second)]
        expected_first = mask_contour_points(first, 13) + [140, 100]
        paused, result, errors = threading.Event(), [], []
        click = app.desktop.click
        def pause_after_first(x, y):
            click(x, y)
            if len(app.desktop.clicks) == len(expected_first):
                app.active_token.pause()
                paused.set()
        app.desktop.click = pause_after_first
        app._start_model_picks(app.candidates, auto=True)
        app.root.run_next()
        def finish():
            try:
                result.append(app.runner.finish())
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=finish)
        worker.start()
        try:
            self.assertTrue(paused.wait(2))
            self.assertEqual(app.desktop.clicks, [tuple(point) for point in expected_first])
            self.assertTrue(worker.is_alive())
        finally:
            app.active_token.resume()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result[0]["contours"], 2)
        self.assertGreater(result[0]["count"], len(expected_first))

    def test_stop_after_first_contour_records_its_completion_without_picking_next(self):
        app = app_state("auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:40, 5:25], second[15:40, 45:65] = True, True
        a, b = make_candidate(first), make_candidate(second)
        app.candidates, app.auto_remaining = [a, b], 2
        expected_first = mask_contour_points(first, 13)
        click = app.desktop.click
        def stop_after_first(x, y):
            click(x, y)
            if len(app.desktop.clicks) == len(expected_first):
                app._hotkey("stop")
        app.desktop.click = stop_after_first
        app._start_model_picks(app.candidates, auto=True)
        app.root.run_next()
        with self.assertRaises(CancelledOperation):
            app.runner.finish()
        app.poll()
        self.assertEqual(app.desktop.clicks, [tuple(point) for point in expected_first + [140, 100]])
        self.assertEqual(app.picked_points, {tuple(point) for point in expected_first})
        self.assertEqual([candidate.id for candidate in app.candidates], [b.id])
        self.assertTrue(app.disabled.get())
        self.assertEqual(app.auto_remaining, 0)
        self.assertEqual(app.desktop.keys, [])

    def test_partial_contour_stop_remembers_clicks_for_same_frame_retry(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:45, 5:40] = True
        candidate = make_candidate(mask)
        app.candidates = [candidate]
        click = app.desktop.click
        def stop_early(x, y):
            click(x, y)
            if len(app.desktop.clicks) == 2:
                app._hotkey("stop")
        app.desktop.click = stop_early
        app._start_model_picks(app.candidates)
        app.root.run_next()
        with self.assertRaises(CancelledOperation):
            app.runner.finish()
        app.poll()
        self.assertEqual(len(app.picked_points), 2)
        self.assertEqual([item.id for item in app.candidates], [candidate.id])
        remaining = screen_model_batches(app.candidates, app.geometry, 13, picked=app.picked_points)
        planned = {tuple(point) for point in remaining[0].screen}
        self.assertFalse(planned & set(app.desktop.clicks))
        self.assertEqual(len(planned), len(mask_contour_points(mask, 13))-2)

    def test_old_frame_progress_cannot_remove_current_contours_or_record_points(self):
        app = app_state("semi-auto")
        candidate = make_candidate(np.ones((70, 80), bool))
        app.candidates = [candidate]
        app.frame_id = 2
        app.events.put(("picked", app.runner.generation, 1, (5, 5), 1, 2, 1, 1))
        app.events.put(("contour_progress", app.runner.generation, 1, 1, 1, 1, 2))
        app.events.put(("contour_done", app.runner.generation, 1, candidate.id))
        app.poll()
        self.assertEqual(app.picked_points, set())
        self.assertEqual([item.id for item in app.candidates], [candidate.id])
        self.assertEqual(app.status.get(), "")

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

    def test_insufficient_background_still_clears_semi_auto_masks(self):
        app = app_state("semi-auto")
        app.candidates = [make_candidate(np.ones((70, 80), bool))]
        app._result("model picks", {"count": 10, "ids": set(), "reference": app.reference,
                                     "auto": False, "scene_verified": False})
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
        app.frames.set(1)
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

    def test_continuous_auto_keeps_advancing_until_explicitly_stopped(self):
        app = app_state("auto")
        app._start_detection, app._next_frame = Mock(), Mock()
        app.frames.set(0)
        app.start_auto()
        self.assertEqual(app.auto_remaining, -1)
        for _ in range(3):
            app._result("model picks", {"count": 1, "ids": set(),
                                       "reference": app.reference, "auto": True})
        self.assertEqual(app.auto_remaining, -1)
        self.assertEqual(app.auto_completed, 3)
        self.assertEqual(app._next_frame.call_count, 3)
        app.stop()
        self.assertEqual(app.auto_remaining, 0)
        app.hotkeys.enable_space_pause.assert_called_with(False)

    def test_empty_last_micrograph_completes_limit_without_extra_down(self):
        app = app_state("auto")
        app.auto_remaining = 1
        app._next_frame = Mock()
        app._result("recognition", {"rgb": app.rgb, "candidates": [], "auto": True, "prompt": False})
        app._next_frame.assert_not_called()
        self.assertEqual(app.auto_completed, 1)
        self.assertEqual(app.auto_remaining, 0)
        self.assertEqual(app.desktop.keys, [])
        self.assertEqual(app.desktop.clicks, [])
        self.assertIn("Auto finished", app.status.get())

    def test_invalid_auto_limits_never_start_detection_or_navigation(self):
        for limit, wait in ((-1, 10), (1001, 10), (0, .5), (0, float("nan")), (0, 61)):
            with self.subTest(limit=limit, wait=wait):
                app = app_state("auto")
                app.frames.set(limit)
                app.wait_seconds.set(wait)
                app._start_detection = Mock()
                app.start_auto()
                app._start_detection.assert_not_called()
                self.assertFalse(app.auto_remaining)
                self.assertEqual(app.desktop.keys, [])

    def test_semi_next_moves_once_then_detects_without_picking(self):
        app = app_state("semi-auto")
        candidate = make_candidate(np.ones((70, 80), bool))
        app.candidates = [candidate]
        app.input_surface = Surface()
        app.input_kind = "masks"
        app.input_surface.show()
        app.overlay.show()
        app.picked_points = {(10, 15)}
        old_frame = app.frame_id
        new = np.full_like(app.reference, 150)
        app.next_micrograph()
        self.assertEqual(app.candidates, [], "Navigation must retire old contours immediately")
        self.assertIsNone(app.input_surface)
        self.assertFalse(app.overlay.visible)
        self.assertEqual(app.runner.jobs, [], "Allow native layers to hide before capturing")
        app.root.run_next()
        with patch("screen_app.wait_for_next_micrograph", return_value=new) as waiter:
            result = app.runner.finish()
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(waiter.call_args.kwargs["timeout"], 10.0)
        guard = waiter.call_args.kwargs["unchanged_mask"]
        self.assertFalse(guard[15, 10])
        self.assertTrue(guard[60, 70])
        self.assertEqual(waiter.call_args.kwargs["history"], ())
        app._start_detection = Mock()
        app._result("next frame", result)
        self.assertFalse(app._start_detection.call_args.kwargs["auto"])
        np.testing.assert_array_equal(app._start_detection.call_args.kwargs["expected"], new)
        self.assertEqual(app.picked_points, set())
        self.assertGreater(app.frame_id, old_frame)
        self.assertFalse(app.root.hidden)

    def test_explicit_semi_navigation_still_works_with_picking_disabled(self):
        app = app_state("semi-auto")
        app.disabled.set(True)
        app.next_micrograph()
        app.root.run_next()
        with patch("screen_app.wait_for_next_micrograph", return_value=np.full_like(app.reference, 150)):
            result = app.runner.finish()
        app._start_detection = Mock()
        app._result("next frame", result)
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])
        self.assertTrue(app.disabled.get())
        self.assertFalse(app._start_detection.call_args.kwargs["auto"])

    def test_stop_before_navigation_callback_sends_no_down(self):
        app = app_state("semi-auto")
        app.next_micrograph()
        app.stop()
        self.assertFalse(app.root.callbacks)
        self.assertFalse(app.runner.jobs)
        self.assertEqual(app.desktop.keys, [])
        self.assertEqual(app.candidates, [])

    def test_stop_after_down_does_not_restore_old_contours(self):
        app = app_state("semi-auto")
        mask = np.zeros((70, 80), bool)
        mask[10:30, 10:30] = True
        app.candidates = [make_candidate(mask)]
        app.next_micrograph()
        app.root.run_next()
        def interrupted(capture, previous, token, **kwargs):
            app._hotkey("stop")
            token.check_cancelled()
        with patch("screen_app.wait_for_next_micrograph", side_effect=interrupted):
            with self.assertRaises(CancelledOperation):
                app.runner.finish()
        app.poll()
        app.disabled.set(False)
        app.disabled_changed()
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])
        self.assertFalse(app.candidates)
        self.assertFalse(app.overlay.visible)
        self.assertIsNone(app.input_surface)

    def test_navigation_focus_or_occlusion_failure_prevents_down(self):
        for failure in ("focus", "occluded"):
            with self.subTest(failure=failure):
                app = app_state("semi-auto")
                if failure == "focus":
                    app.desktop.focus_target = Mock(side_effect=RuntimeError("focus lost"))
                else:
                    app.desktop.window_under_roi = Mock(return_value=77)
                app.next_micrograph()
                app.root.run_next()
                with self.assertRaises(RuntimeError):
                    app.runner.finish()
                self.assertEqual(app.desktop.keys, [])
                self.assertEqual(app.desktop.clicks, [])

    def test_load_timeout_does_not_retry_down_or_start_recognition(self):
        app = app_state("semi-auto")
        app.next_micrograph()
        app.root.run_next()
        token = app.active_token
        app._start_detection = Mock()
        with patch("screen_app.wait_for_next_micrograph",
                   side_effect=MicrographLoadTimeoutError("Still loading")):
            with self.assertRaises(MicrographLoadTimeoutError) as error:
                app.runner.finish()
        app.runner.results = [TaskResult(token, error=error.exception)]
        app.poll()
        app._start_detection.assert_not_called()
        self.assertEqual(app.desktop.keys, ["down"])
        self.assertEqual(app.desktop.clicks, [])
        self.assertTrue(app.disabled.get())

    def test_redetection_with_delayed_pick_markers_keeps_known_picks(self):
        app = app_state("semi-auto")
        app.picked_points = {(20, 20)}
        frame = app.reference.copy()
        frame[12:29, 12:29] = 250
        self.assertGreater(screen_changed_fraction(app.reference, frame), .02)
        app._present_masks = Mock()
        app._result("recognition", {"rgb": frame, "candidates": [], "auto": False, "prompt": False})
        self.assertEqual(app.picked_points, {(20, 20)})

    def test_old_pick_progress_cannot_repopulate_new_micrograph(self):
        app = app_state("semi-auto")
        previous_frame = app.frame_id
        app._start_detection = Mock()
        app._result("next frame", {"rgb": np.full_like(app.rgb, 150), "auto": False, "advanced": True})
        app.events.put(("picked", app.runner.generation, previous_frame, (10, 15), 1, 1, 1, 1))
        app.poll()
        self.assertEqual(app.picked_points, set())

    def test_all_modes_share_fast_default_and_configurable_maximum_speed(self):
        self.assertEqual(DEFAULT_CLICK_DELAY, .01)
        app = app_state()
        for delay in (DEFAULT_CLICK_DELAY, 0, .005, .05, 2):
            app.click_delay.set(delay)
            for mode in ("manual", "semi-auto", "auto"):
                with self.subTest(delay=delay, mode=mode):
                    app.mode = mode
                    self.assertEqual(app._live_ready(), delay)

    def test_all_modes_reject_invalid_shared_delays_before_playback(self):
        app = app_state()
        for delay in (-.01, 2.01, float("nan"), float("inf"), "unfinished"):
            app.click_delay.set(delay)
            for mode in ("manual", "semi-auto", "auto"):
                with self.subTest(delay=delay, mode=mode):
                    app.mode = mode
                    with self.assertRaises(ValueError):
                        app._live_ready()
        self.assertEqual(app.desktop.clicks, [])

    def test_manual_maximum_speed_keeps_original_grid_and_panel(self):
        app = app_state()
        app.click_delay.set(0)
        app.trace = [(150, 110), (210, 110), (210, 155), (150, 155)]
        polygon = tuple(app.trace)
        app.manual_window = 42
        app._finish_manual()
        with patch("screen_app.play_clicks", wraps=__import__("execution").play_clicks) as playback:
            app.runner.finish()
        self.assertEqual(playback.call_args.kwargs["delay"], 0)
        self.assertEqual(app.desktop.clicks, [tuple(p) for p in legacy_contour_points(polygon, 13)])
        self.assertFalse(app.root.hidden)

    def test_manual_speed_shortcut_opens_shared_settings_without_mode_change(self):
        app = app_state()
        app.open_advanced = Mock()
        self.assertEqual(app._open_picking_speed(), "break")
        app.open_advanced.assert_called_once_with()
        self.assertEqual(app.mode, "manual")
        self.assertEqual(app.root.callbacks, {})
        self.assertFalse(app.root.hidden)

    def test_panel_context_menu_exposes_picking_speed_and_f9(self):
        app = app_state()
        menu = Mock()
        app.tk.Menu = Mock(return_value=menu)
        app.open_advanced = Mock()
        event = SimpleNamespace(x_root=35, y_root=45)
        self.assertEqual(app._show_picking_menu(event), "break")
        menu.add_command.assert_called_once_with(label="Picking speed…", accelerator="F9",
                                                 command=app._open_picking_speed)
        menu.tk_popup.assert_called_once_with(35, 45)
        menu.grab_release.assert_called_once_with()
        menu.add_command.call_args.kwargs["command"]()
        app.open_advanced.assert_called_once_with()

    def test_speed_entry_points_do_not_interrupt_active_work(self):
        for state in ("worker", "transition", "auto", "drawing", "selection", "closed"):
            with self.subTest(state=state):
                app = app_state()
                app.open_advanced, app.stop = Mock(), Mock()
                app.tk.Menu = Mock()
                if state == "worker":
                    app.runner.busy = True
                elif state == "auto":
                    app.auto_remaining = -1
                elif state == "selection":
                    app.input_surface, app.input_kind = Surface(), "region"
                else:
                    setattr(app, state, True)
                self.assertEqual(app._open_picking_speed(), "break")
                self.assertEqual(app._show_picking_menu(SimpleNamespace(x_root=35, y_root=45)), "break")
                app.open_advanced.assert_not_called()
                app.tk.Menu.assert_not_called()
                app.stop.assert_not_called()

    def test_start_auto_reuses_exact_displayed_contours_without_redetecting(self):
        app = app_state("auto")
        first, second = np.zeros((70, 80), bool), np.zeros((70, 80), bool)
        first[15:40, 5:25], second[15:40, 45:65] = True, True
        a, b = make_candidate(first), make_candidate(second)
        app.candidates = [b, a]
        app._start_detection, app._start_model_picks = Mock(), Mock()
        app.start_auto()
        app._start_detection.assert_not_called()
        app._start_model_picks.assert_called_once_with([b, a], auto=True)
        self.assertEqual(len(app.auto_history), 1)
        np.testing.assert_array_equal(app.auto_history[0], app.reference)
        self.assertIsNot(app.auto_history[0], app.reference)

    def test_auto_reused_proposals_allow_marker_redraw_before_clicking(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        app.candidates = [make_candidate(mask)]
        app.desktop.rgb[100:170, 140:220] = 0
        app.start_auto()
        app.root.run_next()
        result = app.runner.finish()
        expected = mask_contour_points(mask, 13) + [140, 100]
        self.assertEqual(app.desktop.clicks, [tuple(point) for point in expected])
        self.assertEqual(result["count"], len(expected))
        self.assertEqual(app.desktop.keys, [])

    def test_auto_reenable_restores_retained_outlines_and_reports_their_count(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        candidate = make_candidate(mask)
        app.candidates = [candidate]
        app.stop()
        self.assertFalse(app.overlay.visible)
        app.disabled.set(False)
        app.disabled_changed()
        self.assertTrue(app.overlay.visible)
        self.assertEqual([item.id for item in app.overlay.masks], [candidate.id])
        self.assertIn("1 detected contour shown", app.status.get())
        self.assertIsNone(app.input_surface, "Auto preview must not intercept clicks")
        self.assertEqual(app.runner.jobs, [])

    def test_auto_reenable_waits_for_cancelled_worker_before_restoring_outlines(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        app.candidates = [make_candidate(mask)]
        app.stop()
        app.runner.busy = True
        app.disabled.set(False)
        app.disabled_changed()
        self.assertFalse(app.overlay.visible)
        self.assertIn("Waiting", app.status.get())
        app.runner.busy = False
        app.root.run_next()
        self.assertTrue(app.overlay.visible)
        self.assertIn("1 detected contour shown", app.status.get())

    def test_same_scene_auto_recognition_retains_actual_picks(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        candidates = [make_candidate(mask)]
        remembered = {(5, 15), (18, 15)}
        app.picked_points = set(remembered)
        app._start_model_picks = Mock()
        app._result("recognition", {"rgb": app.reference.copy(), "auto": True,
                                     "prompt": False, "candidates": candidates})
        self.assertEqual(app.picked_points, remembered)
        app._start_model_picks.assert_called_once_with(candidates, auto=True)

    def test_repeated_auto_on_a_fully_picked_scene_completes_without_clicks(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        candidate = make_candidate(mask)
        app.picked_points = {tuple(point) for point in mask_contour_points(mask, 13)}
        app.auto_remaining = 1
        app._result("recognition", {"rgb": app.reference.copy(), "auto": True,
                                     "prompt": False, "candidates": [candidate]})
        app.root.run_next()
        result = app.runner.finish()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["ids"], {candidate.id})
        self.assertFalse(result["scene_verified"])
        self.assertEqual(result["scene_check"], "skipped-auto")
        app.poll()
        self.assertEqual(app.candidates, [])
        app._result("model picks", result)
        self.assertEqual(app.auto_remaining, 0)
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(app.desktop.keys, [])

    def test_changed_scene_recognition_clears_previous_frame_pick_positions(self):
        app = app_state("auto")
        mask = np.zeros((70, 80), bool)
        mask[15:40, 5:25] = True
        app.picked_points = {(5, 15), (18, 15)}
        app._start_model_picks = Mock()
        app._result("recognition", {"rgb": np.zeros_like(app.reference), "auto": True,
                                     "prompt": False, "candidates": [make_candidate(mask)]})
        self.assertEqual(app.picked_points, set())

    def test_no_cryolo_proposals_direct_review_to_manual_contours(self):
        app = app_state("semi-auto")
        app.args = SimpleNamespace(backend="cryolo")
        app._result("recognition", {"rgb": app.rgb, "candidates": [], "auto": False, "prompt": False})
        self.assertIn("manual mode", app.status.get())
        self.assertIn("only selects detections", app.status.get())
        self.assertNotIn("tight Box prompt", app.status.get())
        self.assertEqual(app.desktop.clicks, [])

    def test_no_proposals_advance_auto_without_sending_any_picks(self):
        app = app_state("auto")
        app.auto_remaining = 2
        app._next_frame = Mock()
        app._result("recognition", {"rgb": app.rgb, "candidates": [], "auto": True, "prompt": False})
        app._next_frame.assert_called_once_with()
        self.assertEqual(app.auto_remaining, 1)
        self.assertEqual(app.auto_completed, 1)
        self.assertEqual(app.desktop.keys, [])
        self.assertEqual(app.desktop.clicks, [])

    def test_next_frame_stops_on_unchanged_image_without_clearing_actual_picks(self):
        app = app_state("auto")
        app.auto_remaining = 1
        app.auto_history = [app.reference.copy()]
        app.picked_points = {(10, 15)}
        app._next_frame()
        app.root.run_next()
        with patch("screen_app.wait_for_next_micrograph", side_effect=NoNewMicrographError("did not change")):
            result = app.runner.finish()
        app._start_detection = Mock()
        app._result("next frame", result)
        app._start_detection.assert_not_called()
        self.assertEqual(app.auto_remaining, 0)
        self.assertTrue(app.disabled.get())
        self.assertEqual(app.picked_points, {(10, 15)})
        self.assertIsNotNone(app.reference)
        self.assertIn("No new micrograph", app.status.get())
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
        with self.assertRaisesRegex(RuntimeError, "after navigation"):
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
