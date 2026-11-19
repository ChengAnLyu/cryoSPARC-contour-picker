"""Regression tests using fake mouse adapters; no desktop is accessed."""

import threading
import time
import unittest

from execution import (
    CancelledOperation,
    CancellationToken,
    CaptureGeometry,
    TaskRunner,
    WorkerBusyError,
    play_clicks,
)


class GeometryTests(unittest.TestCase):
    def test_scaled_letterboxed_viewer_maps_to_capture_and_screen(self):
        geometry = CaptureGeometry(655, 600, 600, 420)
        local = geometry.viewer_to_local(160, 125, 300, 210, 10, 20)
        self.assertEqual(local, (300, 210))
        self.assertEqual(geometry.local_to_screen(*local), (955, 810))
        self.assertEqual(geometry.screen_to_local(955, 810), local)

    def test_half_open_edges_and_bad_geometry_are_rejected(self):
        geometry = CaptureGeometry(-100, 50, 20, 10)
        self.assertTrue(geometry.contains(-100, 50))
        self.assertFalse(geometry.contains(-80, 50))
        self.assertEqual(geometry.local_to_screen(19.9, 9.9), (-81, 59))
        with self.assertRaises(ValueError):
            geometry.viewer_to_local(20, 0, 20, 10)
        with self.assertRaises(ValueError):
            CaptureGeometry(0, 0, 0, 10)
        with self.assertRaises(ValueError):
            CaptureGeometry(0.5, 0, 20, 10)


class ClickTests(unittest.TestCase):
    def setUp(self):
        self.geometry = CaptureGeometry(100, 200, 20, 20)
        self.token = CancellationToken()
        self.clicks = []

    def play(self, points, **kwargs):
        return play_clicks(
            points,
            kwargs.pop("click_fn", lambda x, y: self.clicks.append((x, y))),
            self.geometry,
            self.token,
            delay=kwargs.pop("delay", 0),
            **kwargs,
        )

    def test_validates_entire_batch_before_any_click(self):
        for points in (
            [(100, 200), (120, 200)],
            [(100, 200), (100.5, 200)],
            [(100, 200), (float("nan"), 200)],
            [(100, 200), (101,)],
        ):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.play(points)
        self.assertEqual(self.clicks, [])

    def test_maximum_rejects_before_first_click(self):
        with self.assertRaises(ValueError):
            self.play(((100 + x, 200) for x in range(3)), max_clicks=2)
        self.assertEqual(self.clicks, [])

    def test_deduplicates_and_copies_mutable_source(self):
        points = [[100, 200], [101, 201], [100, 200]]

        def click(x, y):
            self.clicks.append((x, y))
            points[1][:] = [500, 500]

        count = self.play(points, click_fn=click)
        self.assertEqual(count, 2)
        self.assertEqual(self.clicks, [(100, 200), (101, 201)])

    def test_cancel_after_first_click_stops_remaining_batch(self):
        def click(x, y):
            self.clicks.append((x, y))
            self.token.cancel()

        with self.assertRaises(CancelledOperation):
            self.play([(100, 200), (101, 201)], click_fn=click)
        self.assertEqual(self.clicks, [(100, 200)])

    def test_click_error_stops_remaining_batch_and_propagates(self):
        failure = RuntimeError("simulated fail-safe")

        def click(x, y):
            self.clicks.append((x, y))
            raise failure

        with self.assertRaises(RuntimeError) as context:
            self.play([(100, 200), (101, 201)], click_fn=click)
        self.assertIs(context.exception, failure)
        self.assertEqual(self.clicks, [(100, 200)])

    def test_pause_then_cancel_never_releases_a_click(self):
        self.token.pause()
        outcome = []
        started = threading.Event()

        def work():
            started.set()
            try:
                self.play([(100, 200)])
            except CancelledOperation:
                outcome.append("cancelled")

        worker = threading.Thread(target=work)
        worker.start()
        self.assertTrue(started.wait(1))
        self.assertEqual(self.clicks, [])
        self.token.cancel()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.clicks, [])
        self.assertEqual(outcome, ["cancelled"])

    def test_resume_allows_paused_batch_to_finish(self):
        self.token.pause()
        worker = threading.Thread(target=lambda: self.play([(100, 200)]))
        worker.start()
        self.assertEqual(self.clicks, [])
        self.token.resume()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.clicks, [(100, 200)])

    def test_long_delay_is_interruptible(self):
        first = threading.Event()
        outcome = []

        def click(x, y):
            self.clicks.append((x, y))
            first.set()

        def work():
            try:
                self.play([(100, 200), (101, 201)], click_fn=click, delay=60)
            except CancelledOperation:
                outcome.append("cancelled")

        worker = threading.Thread(target=work)
        worker.start()
        self.assertTrue(first.wait(1))
        self.token.cancel()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.clicks, [(100, 200)])
        self.assertEqual(outcome, ["cancelled"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = TaskRunner()

    def tearDown(self):
        self.runner.shutdown(wait=True, timeout=1)

    def collect(self):
        deadline = time.monotonic() + 2
        results = []
        while self.runner.busy and time.monotonic() < deadline:
            results.extend(self.runner.poll_results())
            if self.runner.busy:
                threading.Event().wait(0.005)
        self.assertFalse(self.runner.busy, "Worker failed to finish in time")
        return results

    def test_one_outstanding_and_same_worker_thread(self):
        caller = threading.get_ident()
        finished = threading.Event()

        def work(token):
            finished.set()
            return threading.get_ident()

        self.runner.submit(work)
        self.assertTrue(finished.wait(1))
        with self.assertRaises(WorkerBusyError):
            self.runner.submit(work)
        first = self.collect()[0]
        self.runner.submit(work)
        second = self.collect()[0]
        self.assertNotEqual(first.value, caller)
        self.assertEqual(first.value, second.value)

    def test_invalidated_result_is_dropped_before_new_job(self):
        entered = threading.Event()
        release = threading.Event()

        def native_work(token):
            entered.set()
            release.wait(1)
            return "stale image result"

        old_token = self.runner.submit(native_work)
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.runner.invalidate(), 1)
        self.assertTrue(old_token.cancelled)
        self.assertTrue(self.runner.busy)
        with self.assertRaises(WorkerBusyError):
            self.runner.submit(lambda token: "new image result")
        release.set()
        self.assertEqual(self.collect(), [])
        new_token = self.runner.submit(lambda token: "new image result")
        result = self.collect()[0]
        self.assertIs(result.token, new_token)
        self.assertEqual(result.token.generation, 1)
        self.assertEqual(result.value, "new image result")

    def test_error_is_reported_and_worker_is_reusable(self):
        failure = RuntimeError("expected model failure")

        def fail(token):
            raise failure

        with self.assertLogs("execution", level="ERROR"):
            self.runner.submit(fail)
            result = self.collect()[0]
        self.assertIs(result.error, failure)
        self.assertIn("expected model failure", result.traceback_text)
        self.runner.submit(lambda token: 12)
        self.assertEqual(self.collect()[0].value, 12)

    def test_cancelled_same_generation_reports_cancellation(self):
        token = CancellationToken(self.runner.generation)
        token.pause()
        self.runner.submit(lambda token: self.fail("Paused work must not execute"), token)
        token.cancel()
        result = self.collect()[0]
        self.assertTrue(result.cancelled)
        self.assertIsNone(result.error)

    def test_closed_runner_rejects_new_work(self):
        self.runner.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            self.runner.submit(lambda token: None)


if __name__ == "__main__":
    unittest.main()
