"""Headless app integration regressions with no Tk windows or desktop events."""

import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app import PickerApp, changed_fraction, export_review, image_digest, selected_grid
from execution import CancellationToken, CaptureGeometry, TaskResult, WorkerBusyError
from recognition import RecognitionConfig, make_candidate


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeRoot:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0
        self.hidden = False

    def after(self, milliseconds, callback):
        self.next_id += 1
        identifier = f"after#{self.next_id}"
        self.callbacks[identifier] = callback
        return identifier

    def after_cancel(self, identifier):
        self.callbacks.pop(identifier, None)

    def withdraw(self):
        self.hidden = True

    def deiconify(self):
        self.hidden = False

    def run_next(self):
        identifier = next(iter(self.callbacks))
        self.callbacks.pop(identifier)()


class ProbeRunner:
    def __init__(self):
        self.generation = 0
        self.busy = False
        self.submissions = []
        self.results = []

    def submit(self, function, token=None):
        if self.busy:
            raise WorkerBusyError("Busy")
        token = token or CancellationToken(self.generation)
        self.submissions.append((function, token, token.paused))
        self.busy = True
        return token

    def invalidate(self):
        self.generation += 1
        for _, token, _ in self.submissions:
            token.cancel()

    def poll_results(self):
        results, self.results = self.results, []
        if results:
            self.busy = False
        return results


class FakeDesktop:
    def __init__(self, rgb):
        self.rgb = rgb.copy()
        self.clicks = []
        self.keys = []

    def focus_target(self, target):
        pass

    def ensure_target(self, target=None):
        pass

    def screenshot(self, geometry=None):
        return self.rgb.copy()

    def click(self, x, y):
        self.clicks.append((x, y))

    def press_next(self, key):
        self.keys.append(key)


def headless_app():
    app = PickerApp.__new__(PickerApp)
    app.root, app.runner = FakeRoot(), ProbeRunner()
    app.rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    app.playback_reference = app.rgb.copy()
    app.capture_id = 1
    app.geometry, app.target = CaptureGeometry(100, 200, 20, 20), 11
    app.desktop = FakeDesktop(app.rgb)
    app.live, app.click_delay = Variable(True), Variable(.02)
    app.hotkeys = object()
    app.active_token = None
    app.events = queue.Queue()
    app.scheduled = set()
    app.closed = False
    app.playback_pending = app.playback_active = app.capture_pending = False
    app.batch_remaining = 0
    app.batch_paused = False
    app.batch_seen = []
    app.wait_seconds, app.next_key = Variable(1), Variable("down")
    app.selector = None
    app.picked_points, app.selected, app.picked = set(), {"candidate"}, set()
    app.candidates, app.alternatives = [], []
    app.status, app.source_label = Variable(""), Variable("")
    app.job_kind, app.job_started = "existing", 0
    app.errors = []
    app._error = app.errors.append
    app.render = lambda: None
    app.refresh_table = lambda: None
    app._grid = lambda: np.array([[1, 1], [2, 2]], dtype=np.int32)
    return app


class ImageAndExportTests(unittest.TestCase):
    def test_drift_guard_does_not_wrap_uint8_subtraction(self):
        white = np.full((4, 4, 3), 250, dtype=np.uint8)
        black = np.full((4, 4, 3), 5, dtype=np.uint8)
        self.assertEqual(changed_fraction(white, black), 1)
        self.assertEqual(changed_fraction(white, white), 0)
        self.assertEqual(changed_fraction(white, white[:2]), 1)

    def test_image_identity_includes_shape(self):
        first = np.zeros((2, 6, 3), dtype=np.uint8)
        second = np.zeros((3, 4, 3), dtype=np.uint8)
        self.assertNotEqual(image_digest(first), image_digest(second))

    def test_selection_excludes_unselected_masks_and_deduplicates_overlap(self):
        left = np.zeros((30, 30), bool)
        left[2:20, 2:20] = True
        overlap = np.zeros_like(left)
        overlap[10:25, 10:25] = True
        other = np.zeros_like(left)
        other[25:30, :5] = True
        candidates = [make_candidate(mask, source="manual") for mask in (left, overlap, other)]
        selected = {candidates[0].id, candidates[1].id}
        points = selected_grid(candidates, selected, left.shape, 3, 0, 1000)
        self.assertGreater(len(points), 0)
        self.assertEqual(len(points), len(set(map(tuple, points))))
        self.assertTrue(np.all((left | overlap)[points[:, 1], points[:, 0]]))
        self.assertFalse(np.any(other[points[:, 1], points[:, 0]]))

    def test_export_preserves_pixels_masks_and_uncalibrated_coordinate_labels(self):
        rgb = np.zeros((10, 12, 3), dtype=np.uint8)
        rgb[4, 5] = [80, 90, 100]
        mask = np.zeros(rgb.shape[:2], bool)
        mask[2:7, 3:8] = True
        candidate = make_candidate(mask, source="manual")
        points = np.array([[5, 4]], np.int32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            export_review(path, rgb, [candidate], {candidate.id}, points, None)
            details = json.loads((path / "review.json").read_text())
            self.assertIn("no micrograph or physical calibration", details["coordinate_system"])
            self.assertIn("not expert-verified", details["selection_status"])
            self.assertIsNone(details["capture_region_xywh"])
            np.testing.assert_array_equal(np.asarray(Image.open(path / "image.png")), rgb)
            np.testing.assert_array_equal(np.asarray(Image.open(path / "selected_mask.png")) > 0, mask)
            with (path / "points.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["capture_x_px", "capture_y_px", "screen_x_px", "screen_y_px"])
            self.assertEqual(rows[1], ["5", "4", "", ""])


class AppStateTests(unittest.TestCase):
    def test_stale_timer_cannot_restart_work_after_generation_change(self):
        app = headless_app()
        actions = []
        app._later(10, lambda: actions.append("ran"))
        callback = next(iter(app.root.callbacks.values()))
        app.runner.invalidate()
        callback()
        self.assertEqual(actions, [])

    def test_stop_cancels_playback_countdown_and_old_callback_is_inert(self):
        app = headless_app()
        app.start_playback()
        self.assertTrue(app.playback_pending)
        callback = next(iter(app.root.callbacks.values()))
        app.stop()
        callback()  # Simulate a callback already dequeued by Tk.
        self.assertFalse(app.live.get())
        self.assertFalse(app.playback_pending)
        self.assertEqual(app.runner.submissions, [])
        self.assertEqual(app.desktop.clicks, [])

    def test_busy_load_does_not_read_or_replace_image(self):
        app = headless_app()
        original = app.rgb
        app.runner.busy = True
        with patch("app.Image.open") as opener:
            app.load_image(Path("not-read.png"))
            opener.assert_not_called()
        self.assertIs(app.rgb, original)
        self.assertEqual(len(app.errors), 1)

    def test_no_live_playback_without_global_hotkeys(self):
        app = headless_app()
        app.hotkeys = None
        app.start_playback()
        self.assertEqual(app.runner.submissions, [])
        self.assertFalse(app.root.hidden)
        self.assertIn("hotkeys", str(app.errors[0]))

    def test_worker_error_terminates_batch_without_handling_success(self):
        app = headless_app()
        app.batch_remaining = 4
        app.playback_active = True
        failure = RuntimeError("simulated target focus loss")
        app.runner.results = [TaskResult(CancellationToken(), error=failure)]
        handled = []
        app._handle_result = lambda *args: handled.append(args)
        app.poll()
        self.assertEqual(app.batch_remaining, 0)
        self.assertFalse(app.playback_active)
        self.assertFalse(app.root.hidden)
        self.assertEqual(handled, [])
        self.assertEqual(app.errors, [failure])
        self.assertEqual(app.desktop.keys, [])

    def test_paused_job_is_paused_before_worker_dispatch(self):
        app = headless_app()
        app.batch_paused = True
        app._submit("playback", lambda token: None)
        self.assertTrue(app.runner.submissions[0][2])

    def test_failed_submit_keeps_existing_job_metadata(self):
        app = headless_app()
        app.runner.busy = True
        app.job_kind = "prompt"
        with self.assertRaises(WorkerBusyError):
            app._submit("playback", lambda token: None)
        self.assertEqual(app.job_kind, "prompt")

    def test_invalid_final_point_rejects_whole_batch_before_hiding_ui(self):
        app = headless_app()
        app._grid = lambda: np.array([[1, 1], [20, 2]], np.int32)
        app.start_playback()
        self.assertEqual(app.runner.submissions, [])
        self.assertEqual(app.desktop.clicks, [])
        self.assertEqual(len(app.errors), 1)
        self.assertFalse(app.root.hidden)

    def test_changed_snapshot_stops_before_any_click(self):
        app = headless_app()
        app.desktop.rgb[:] = 255
        app.start_playback()
        app.root.run_next()
        work, token, _ = app.runner.submissions[0]
        with patch.object(token, "sleep", lambda seconds: token.check_cancelled()):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                work(token)
        self.assertEqual(app.desktop.clicks, [])

    def test_playback_captures_point_snapshot_before_worker_starts(self):
        app = headless_app()
        points = np.array([[1, 1], [2, 2]], np.int32)
        app._grid = lambda: points
        app.start_playback()
        points[:] = 10
        app.root.run_next()
        work, token, _ = app.runner.submissions[0]
        with patch.object(token, "sleep", lambda seconds: token.check_cancelled()):
            value = work(token)
        self.assertEqual(value["count"], 2)
        self.assertEqual(app.desktop.clicks, [(101, 201), (102, 202)])

    def test_finished_playback_does_not_pollute_clean_source_image(self):
        app = headless_app()
        clean = app.rgb.copy()
        app.desktop.rgb[5:10, 5:10] = 255  # Simulate visible cryoSPARC pick markers.
        app._handle_result("playback", {"count": 2, "selected": {"candidate"}, "batch": False}, .1)
        np.testing.assert_array_equal(app.rgb, clean)

    def test_late_click_progress_cannot_mark_a_new_image_as_picked(self):
        app = headless_app()
        app.events.put(("clicked", app.runner.generation, (3, 4), 1, 2, app.capture_id))
        app.capture_id += 1  # The new image was loaded before this poll.
        app.poll()
        self.assertEqual(app.picked_points, set())

    def test_delivered_click_is_remembered_after_stop_on_same_image(self):
        app = headless_app()
        app.events.put(("clicked", app.runner.generation, (3, 4), 1, 2, app.capture_id))
        app.stop()
        stopped_status = app.status.get()
        app.poll()
        self.assertEqual(app.picked_points, {(3, 4)})
        self.assertEqual(app.status.get(), stopped_status)

    def test_empty_automatic_result_stops_batch_without_navigation(self):
        app = headless_app()
        app.batch_remaining = 3
        app.job_kind = "batch recognition"
        app.runner.results = [TaskResult(CancellationToken(), value=[])]
        with self.assertLogs("saru_picker", level="ERROR"):
            app.poll()
        self.assertEqual(app.batch_remaining, 0)
        self.assertEqual(app.desktop.keys, [])
        self.assertEqual(app.desktop.clicks, [])
        self.assertIn("No automatic candidates", str(app.errors[0]))

    def test_pick_markers_cannot_masquerade_as_a_new_micrograph(self):
        app = headless_app()
        app.batch_remaining = 2
        app.batch_seen = [app.rgb.copy()]
        app.desktop.rgb[:5] = 255  # New markers on the same micrograph.
        app._handle_result("batch playback", {"count": 2, "selected": {"candidate"}, "batch": True}, .1)
        app.root.run_next()
        work, token, _ = app.runner.submissions[0]
        with patch.object(token, "sleep", lambda seconds: token.check_cancelled()):
            with self.assertRaisesRegex(RuntimeError, "did not change"):
                work(token)
        self.assertEqual(app.desktop.keys, ["down"])

    def test_global_stop_cancels_worker_before_gui_poll(self):
        app = headless_app()
        token = CancellationToken()
        app.active_token = token
        app._hotkey("stop")
        self.assertTrue(token.cancelled)
        self.assertEqual(app.events.get_nowait(), ("hotkey", "stop"))

    def test_delayed_callback_failure_restores_ui_and_stops_batch(self):
        app = headless_app()
        app.batch_remaining = 2
        app.root.withdraw()

        def fail():
            raise ValueError("invalid batch delay")

        app._later(1, fail)
        with self.assertLogs("saru_picker", level="ERROR"):
            app.root.run_next()
        self.assertEqual(app.batch_remaining, 0)
        self.assertFalse(app.root.hidden)
        self.assertEqual(len(app.errors), 1)


class ReviewRecognitionProgressTests(unittest.TestCase):
    def recognition_app(self, backend="cryolo"):
        app = headless_app()
        app.prompt_box = (1, 2, 8, 9)
        app.prompt_points, app.prompt_labels = [(3, 4)], [1]
        config = RecognitionConfig(
            checkpoint="model.onnx" if backend == "cryolo" else "model.pth",
            backend=backend,
            automatic_profile="capsid-learned" if backend == "cryolo" else "capsid",
        )
        app._model_config = Mock(return_value=config)
        recognizer = Mock()
        recognizer.prompt.return_value = []
        recognizer.automatic.return_value = []
        app._get_recognizer = Mock(return_value=recognizer)
        return app, recognizer

    def start_detection(self, app, operation):
        with patch("app.time.perf_counter", return_value=100.0):
            if operation == "prompt":
                app.segment_prompt()
            else:
                app._start_auto_detection(batch=operation == "batch")
        self.assertEqual(app.errors, [])
        return app.runner.submissions[-1][:2]

    def poll_at(self, app, now):
        with patch("app.time.perf_counter", return_value=now):
            app.poll()

    def test_cryolo_prompt_and_automatic_receive_cancellable_progress_callback(self):
        for operation in ("prompt", "automatic", "batch"):
            with self.subTest(operation=operation):
                app, recognizer = self.recognition_app()
                work, token = self.start_detection(app, operation)
                self.assertEqual(work(token), [])
                method = recognizer.prompt if operation == "prompt" else recognizer.automatic
                options = method.call_args.kwargs
                self.assertTrue(callable(options["progress"]))
                self.assertFalse(options["cancel"]())

                before = app.status.get()
                with ThreadPoolExecutor(max_workers=1) as worker:
                    worker.submit(options["progress"], "Refining contours").result(timeout=2)
                self.assertEqual(app.status.get(), before)
                self.poll_at(app, 103.0)
                self.assertEqual(
                    app.status.get(),
                    "Recognition: Refining contours\n3s elapsed. Escape stops.",
                )
                token.cancel()
                self.assertTrue(options["cancel"]())

    def test_sam_prompt_and_automatic_keep_existing_method_signatures(self):
        def sam_prompt(rgb, *, box, points, labels, cancel):
            return []

        def sam_automatic(rgb, *, cancel):
            return []

        for operation in ("prompt", "automatic", "batch"):
            with self.subTest(operation=operation):
                app, recognizer = self.recognition_app(backend="sam")
                recognizer.prompt.side_effect = sam_prompt
                recognizer.automatic.side_effect = sam_automatic
                work, token = self.start_detection(app, operation)
                self.assertEqual(work(token), [])
                method = recognizer.prompt if operation == "prompt" else recognizer.automatic
                self.assertNotIn("progress", method.call_args.kwargs)
                self.poll_at(app, 102.0)
                self.assertEqual(
                    app.status.get(),
                    "Recognition: Running SAM\n2s elapsed. Escape stops.",
                )

    def test_poll_updates_queued_stage_and_elapsed_while_worker_is_busy(self):
        app, recognizer = self.recognition_app()
        work, token = self.start_detection(app, "automatic")
        work(token)
        recognizer.automatic.call_args.kwargs["progress"]("Reading detections")
        self.poll_at(app, 103.0)
        self.assertIn("Reading detections\n3s elapsed", app.status.get())
        self.poll_at(app, 104.0)
        self.assertIn("Reading detections\n4s elapsed", app.status.get())
        self.assertTrue(app.runner.busy)
        self.assertIs(app.active_token, token)

    def test_poll_keeps_paused_status_then_resumes_latest_stage(self):
        app, recognizer = self.recognition_app()
        work, token = self.start_detection(app, "automatic")
        work(token)
        app.pause()
        paused_status = app.status.get()
        recognizer.automatic.call_args.kwargs["progress"]("Refining contours")
        self.poll_at(app, 104.0)
        self.assertEqual(app.status.get(), paused_status)
        self.assertTrue(token.paused)

        app.pause()
        self.poll_at(app, 105.0)
        self.assertFalse(token.paused)
        self.assertIn("Refining contours\n5s elapsed", app.status.get())

    def test_stop_clears_progress_and_late_callback_cannot_restore_it(self):
        app, recognizer = self.recognition_app()
        work, token = self.start_detection(app, "prompt")
        work(token)
        callback = recognizer.prompt.call_args.kwargs["progress"]
        app.stop()
        stopped_status = app.status.get()
        callback("Late detector phase")
        self.poll_at(app, 110.0)

        self.assertTrue(token.cancelled)
        self.assertIsNone(app.active_token)
        self.assertIsNone(app._recognition_progress)
        self.assertEqual(app.status.get(), stopped_status)

    def test_success_clears_progress_before_later_stage_events(self):
        for operation in ("prompt", "automatic"):
            with self.subTest(operation=operation):
                app, recognizer = self.recognition_app()
                work, token = self.start_detection(app, operation)
                value = work(token)
                method = recognizer.prompt if operation == "prompt" else recognizer.automatic
                app.runner.results = [TaskResult(token, value=value)]
                self.poll_at(app, 104.0)
                finished_status = app.status.get()
                self.assertIsNone(app._recognition_progress)
                self.assertIsNone(app.active_token)
                self.assertNotIn("elapsed. Escape stops.", finished_status)

                method.call_args.kwargs["progress"]("Late detector output")
                self.poll_at(app, 110.0)
                self.assertEqual(app.status.get(), finished_status)

    def test_error_and_cancellation_clear_progress(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                app, recognizer = self.recognition_app()
                work, token = self.start_detection(app, "automatic")
                work(token)
                failure = None if cancelled else RuntimeError("Detector runtime failed")
                app.runner.results = [TaskResult(token, error=failure, cancelled=cancelled)]
                self.poll_at(app, 104.0)
                expected = "Cancelled." if cancelled else "Detector runtime failed"
                self.assertEqual(app.status.get(), expected)
                self.assertIsNone(app._recognition_progress)
                self.assertIsNone(app.active_token)
                self.assertEqual(app.errors, [] if cancelled else [failure])

                recognizer.automatic.call_args.kwargs["progress"]("Late detector output")
                self.poll_at(app, 110.0)
                self.assertEqual(app.status.get(), expected)

    def test_completion_can_start_new_job_without_accepting_old_same_generation_phase(self):
        app, _ = self.recognition_app()
        _, previous = self.start_detection(app, "automatic")
        app.runner.results = [TaskResult(previous, value=[])]

        def start_next_job(kind, value, elapsed):
            self.assertIsNone(app._recognition_progress)
            self.assertIsNone(app.active_token)
            app._submit("prompt", lambda token: [])

        app._handle_result = start_next_job
        self.poll_at(app, 105.0)
        current = app.active_token
        self.assertIsNot(current, previous)
        self.assertEqual(current.generation, previous.generation)
        app.events.put(("recognition_phase", current, "Current detector phase"))
        app.events.put(("recognition_phase", previous, "Stale detector phase"))
        self.poll_at(app, 106.0)

        self.assertIs(app._recognition_progress.token, current)
        self.assertEqual(
            app.status.get(),
            "Recognition: Current detector phase\n1s elapsed. Escape stops.",
        )



if __name__ == "__main__":
    unittest.main()
