"""Recognition progress stays on the GUI thread and belongs to one worker job."""
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from execution import CancellationToken
import screen_app


class ThreadBoundStatus:
    """A Tk-like status value that rejects any worker-thread access."""

    def __init__(self, value="Ready"):
        self.thread_id = threading.get_ident()
        self.value = value
        self.writes = []

    def _check_thread(self):
        if threading.get_ident() != self.thread_id:
            raise AssertionError("Recognition worker accessed GUI status")

    def get(self):
        self._check_thread()
        return self.value

    def set(self, value):
        self._check_thread()
        self.value = value
        self.writes.append(value)


def progress_owner():
    token = CancellationToken(7)
    owner = SimpleNamespace(
        runner=SimpleNamespace(generation=7, busy=True),
        active_token=token,
        events=queue.Queue(),
        status=ThreadBoundStatus(),
        closed=False,
    )
    return owner, token


class RecognitionProgressTests(unittest.TestCase):
    def setUp(self):
        self.owner, self.token = progress_owner()

    def begin(self, started=100.0):
        screen_app.begin_recognition_progress(self.owner, self.token, started=started)

    def refresh_at(self, now, *, paused=False):
        with patch.object(screen_app.time, "perf_counter", return_value=now):
            screen_app.refresh_recognition_progress(self.owner, paused=paused)

    def test_begin_tracks_token_and_clock_without_touching_status(self):
        with patch.object(screen_app.time, "perf_counter", return_value=123.5):
            screen_app.begin_recognition_progress(self.owner, self.token)

        progress = self.owner._recognition_progress
        self.assertIs(progress.token, self.token)
        self.assertEqual(progress.started, 123.5)
        self.assertEqual(progress.phase, "Starting detector")
        self.assertEqual(self.owner.status.writes, [])

    def test_explicit_start_time_is_preserved(self):
        with patch.object(screen_app.time, "perf_counter", return_value=999.0):
            self.begin(started=12.25)

        self.assertEqual(self.owner._recognition_progress.started, 12.25)

    def test_worker_callback_only_queues_phase_without_touching_gui(self):
        self.begin()
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(
                screen_app.queue_recognition_progress,
                self.owner, self.token, "Loading detector",
            ).result(timeout=2)

        self.assertEqual(
            self.owner.events.get_nowait(),
            ("recognition_phase", self.token, "Loading detector"),
        )
        self.assertTrue(self.owner.events.empty())
        self.assertEqual(self.owner._recognition_progress.phase, "Starting detector")
        self.assertEqual(self.owner.status.writes, [])

    def test_queued_phase_is_converted_to_text(self):
        screen_app.queue_recognition_progress(self.owner, self.token, 42)
        self.assertEqual(
            self.owner.events.get_nowait(), ("recognition_phase", self.token, "42")
        )

    def test_accepted_phase_is_compact_but_full_detail_is_logged(self):
        self.begin()
        phase = "Loading the crYOLO detector and its trained checkpoint " + "detail " * 20 + "complete detail"
        with self.assertLogs("saru.screen", level="DEBUG") as captured:
            screen_app.accept_recognition_progress(self.owner, self.token, phase)

        compact = self.owner._recognition_progress.phase
        self.assertLessEqual(len(compact), 56)
        self.assertTrue(compact.startswith("Loading"))
        self.assertIn(phase, "\n".join(captured.output))
        self.assertEqual(self.owner.status.writes, [])

    def test_phase_whitespace_does_not_add_status_lines(self):
        self.begin()
        screen_app.accept_recognition_progress(
            self.owner, self.token, "  Loading\n detector\t network  "
        )
        self.assertEqual(self.owner._recognition_progress.phase, "Loading detector network")

    def test_unchanged_phase_refreshes_only_once_per_elapsed_second(self):
        self.begin()
        for now in (100.1, 100.9, 101.0, 101.9, 102.2):
            self.refresh_at(now)

        self.assertEqual(self.owner.status.writes, [
            "Recognition: Starting detector\n0s elapsed. Escape stops.",
            "Recognition: Starting detector\n1s elapsed. Escape stops.",
            "Recognition: Starting detector\n2s elapsed. Escape stops.",
        ])

    def test_phase_change_refreshes_immediately_within_same_second(self):
        self.begin()
        self.refresh_at(100.1)
        screen_app.accept_recognition_progress(self.owner, self.token, "Reading detections")
        self.refresh_at(100.2)
        self.refresh_at(100.8)

        self.assertEqual(len(self.owner.status.writes), 2)
        self.assertEqual(
            self.owner.status.get(),
            "Recognition: Reading detections\n0s elapsed. Escape stops.",
        )

    def test_same_generation_replacement_rejects_previous_job_phase(self):
        self.begin()
        previous = self.token
        replacement = CancellationToken(previous.generation)
        self.owner.active_token = replacement
        screen_app.begin_recognition_progress(self.owner, replacement, started=200.0)

        screen_app.accept_recognition_progress(self.owner, previous, "Old detector completed")
        screen_app.clear_recognition_progress(self.owner, token=previous)
        self.refresh_at(201.1)

        self.assertIs(self.owner._recognition_progress.token, replacement)
        self.assertEqual(self.owner._recognition_progress.phase, "Starting detector")
        self.assertEqual(
            self.owner.status.get(),
            "Recognition: Starting detector\n1s elapsed. Escape stops.",
        )

    def test_active_token_change_without_new_tracking_cannot_refresh_old_job(self):
        self.begin()
        self.owner.active_token = CancellationToken(self.token.generation)
        screen_app.accept_recognition_progress(self.owner, self.token, "Stale detector phase")
        self.refresh_at(104.0)

        self.assertEqual(self.owner.status.writes, [])
        self.assertEqual(self.owner._recognition_progress.phase, "Starting detector")

    def test_generation_change_rejects_phase_and_elapsed_refresh(self):
        self.begin()
        self.owner.runner.generation += 1
        screen_app.accept_recognition_progress(self.owner, self.token, "Previous generation")
        self.refresh_at(104.0)

        self.assertEqual(self.owner.status.writes, [])
        self.assertEqual(self.owner._recognition_progress.phase, "Starting detector")

    def test_immediate_token_cancellation_blocks_phase_before_gui_stop(self):
        self.begin()
        self.token.cancel()
        screen_app.accept_recognition_progress(self.owner, self.token, "Too late")
        self.refresh_at(104.0)

        self.assertEqual(self.owner.runner.generation, self.token.generation)
        self.assertEqual(self.owner._recognition_progress.phase, "Starting detector")
        self.assertEqual(self.owner.status.writes, [])

    def test_paused_status_survives_phase_events_and_resume_shows_latest_phase(self):
        self.begin()
        self.token.pause()
        self.owner.status.set("Paused. Resume to continue.")
        screen_app.accept_recognition_progress(self.owner, self.token, "Refining contours")
        self.refresh_at(105.0, paused=True)

        self.assertEqual(self.owner.status.get(), "Paused. Resume to continue.")
        self.assertEqual(len(self.owner.status.writes), 1)
        self.assertEqual(self.owner._recognition_progress.phase, "Refining contours")

        self.token.resume()
        self.refresh_at(106.0)
        self.assertEqual(
            self.owner.status.get(),
            "Recognition: Refining contours\n6s elapsed. Escape stops.",
        )

    def test_finished_runner_does_not_overwrite_completion_status(self):
        self.begin()
        self.owner.runner.busy = False
        self.owner.status.set("Found 3 contours.")
        self.refresh_at(110.0)

        self.assertEqual(self.owner.status.writes, ["Found 3 contours."])

    def test_closed_owner_does_not_refresh_status(self):
        self.begin()
        self.owner.closed = True
        self.refresh_at(110.0)

        self.assertEqual(self.owner.status.writes, [])

    def test_stop_clearing_prevents_queued_phase_from_restoring_progress(self):
        self.begin()
        screen_app.queue_recognition_progress(self.owner, self.token, "Reading detections")
        self.token.cancel()
        self.owner.runner.generation += 1
        screen_app.clear_recognition_progress(self.owner)
        self.owner.status.set("Stopped.")

        _, token, phase = self.owner.events.get_nowait()
        screen_app.accept_recognition_progress(self.owner, token, phase)
        self.refresh_at(110.0)

        self.assertIsNone(getattr(self.owner, "_recognition_progress", None))
        self.assertEqual(self.owner.status.writes, ["Stopped."])

    def test_matching_completion_clears_progress_before_later_poll(self):
        self.begin()
        screen_app.clear_recognition_progress(self.owner, token=self.token)
        self.owner.status.set("Recognition complete.")
        self.refresh_at(110.0)

        self.assertIsNone(getattr(self.owner, "_recognition_progress", None))
        self.assertEqual(self.owner.status.writes, ["Recognition complete."])

    def test_idle_owner_accept_clear_and_refresh_are_safe_without_tracking(self):
        screen_app.accept_recognition_progress(self.owner, self.token, "Untracked phase")
        screen_app.clear_recognition_progress(self.owner, token=self.token)
        self.refresh_at(110.0)

        self.assertEqual(self.owner.status.writes, [])


if __name__ == "__main__":
    unittest.main()
