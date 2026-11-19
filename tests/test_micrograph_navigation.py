"""Navigation acknowledgement with synthetic images and no desktop input."""

import unittest

import numpy as np

from execution import CancellationToken, CancelledOperation
from micrograph_navigation import (
    MicrographLoadTimeoutError, NavigationError, NoNewMicrographError,
    RepeatedMicrographError, wait_for_next_micrograph,
)


class Clock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now


class Token(CancellationToken):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.waits = []
        self.pause_on_ready = 0.0
        self.pause_on_sleep = 0.0
    def wait_until_ready(self):
        super().wait_until_ready()
        self.clock.now += self.pause_on_ready
        self.pause_on_ready = 0.0
    def sleep(self, seconds):
        self.check_cancelled()
        self.waits.append(seconds)
        self.clock.now += seconds + self.pause_on_sleep
        self.pause_on_sleep = 0.0


class NavigationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.previous = rng.integers(65, 170, (32, 40, 3), dtype=np.uint8)
        self.next_image = rng.integers(65, 170, (32, 40, 3), dtype=np.uint8)
        self.other = rng.integers(65, 170, (32, 40, 3), dtype=np.uint8)
        self.clock = Clock()
        self.token = Token(self.clock)
        self.captures = 0
        self.target_checks = 0

    def run_wait(self, capture=None, **kwargs):
        actual = capture or (lambda: self.next_image)
        def count_capture():
            self.captures += 1
            return actual()
        def check_target():
            self.target_checks += 1
        return wait_for_next_micrograph(
            count_capture, self.previous, self.token, clock=self.clock,
            timeout=kwargs.pop('timeout', 1.5),
            check_target=kwargs.pop('check_target', check_target), **kwargs,
        )

    def test_new_image_requires_settling_then_returns_independent_snapshot(self):
        result = self.run_wait(timeout=1)
        np.testing.assert_array_equal(result, self.next_image)
        self.assertGreaterEqual(self.clock.now, .6)
        self.assertLess(self.clock.now, 1)
        self.assertGreaterEqual(self.captures, 5)
        self.assertEqual(self.target_checks, self.captures + 1)
        self.next_image[:] = 0
        self.assertTrue(np.any(result))

    def test_unchanged_last_micrograph_times_out_without_an_input_retry(self):
        with self.assertRaises(NoNewMicrographError) as context:
            self.run_wait(lambda: self.previous, timeout=1)
        self.assertIn('without sending another Down', str(context.exception))
        self.assertAlmostEqual(self.clock.now, 1)

    def test_delayed_markers_cannot_be_mistaken_for_a_new_micrograph(self):
        marked = self.previous.copy()
        marked[4:14, 8:18] = (0, 255, 0)
        mask = np.ones(self.previous.shape[:2], bool)
        mask[3:15, 7:19] = False
        self.assertGreater(np.mean(np.any(marked != self.previous, axis=2)), .02)
        with self.assertRaises(NoNewMicrographError):
            self.run_wait(lambda: marked, unchanged_mask=mask)

    def test_new_micrograph_after_delayed_old_markers_is_accepted(self):
        marked = self.previous.copy()
        marked[4:14, 8:18] = (0, 255, 0)
        mask = np.ones(self.previous.shape[:2], bool)
        mask[3:15, 7:19] = False
        result = self.run_wait(lambda: marked if self.clock.now < .3 else self.next_image,
                               unchanged_mask=mask)
        np.testing.assert_array_equal(result, self.next_image)
        self.assertGreaterEqual(self.clock.now, .9 - 1e-9)

    def test_stable_recent_frame_is_rejected_including_marker_only_differences(self):
        marked = self.other.copy()
        marked[4:14, 8:18] = (0, 255, 0)
        mask = np.ones(self.previous.shape[:2], bool)
        mask[3:15, 7:19] = False
        with self.assertRaises(RepeatedMicrographError):
            self.run_wait(lambda: marked, unchanged_mask=mask, history=[self.previous, self.other])

    def test_transient_recent_preview_does_not_stop_eventual_new_micrograph(self):
        result = self.run_wait(lambda: self.other if self.clock.now < .3 else self.next_image,
                               history=[self.other])
        np.testing.assert_array_equal(result, self.next_image)

    def test_blank_loading_screen_is_not_returned_even_when_stable(self):
        blank = np.full_like(self.previous, 128)
        # A spinner or short loading message must not make this look textured.
        blank[1:3, 1:5] = 255
        result = self.run_wait(lambda: blank if self.clock.now < .9 else self.next_image,
                               timeout=2)
        np.testing.assert_array_equal(result, self.next_image)
        self.assertGreaterEqual(self.clock.now, 1.5 - 1e-9)

    def test_permanent_blank_loading_screen_is_a_load_timeout(self):
        with self.assertRaises(MicrographLoadTimeoutError):
            self.run_wait(lambda: np.full_like(self.previous, 128), timeout=1)

    def test_ongoing_changes_do_not_pass_adjacent_frame_stability(self):
        state = self.next_image.copy()
        def slowly_change():
            # Each step affects <1%, but cumulative drift exceeds 1% before
            # the settling interval. Comparing only adjacent captures fails.
            index = (self.captures - 1) * 10
            state.reshape(-1, 3)[index:index+10] = 255
            return state
        with self.assertRaises(MicrographLoadTimeoutError):
            self.run_wait(slowly_change, timeout=1.5)

    def test_initial_pause_does_not_consume_timeout_or_settling_time(self):
        self.token.pause_on_ready = 30
        result = self.run_wait(timeout=1)
        np.testing.assert_array_equal(result, self.next_image)
        self.assertGreaterEqual(self.clock.now, 30.6 - 1e-9)
        self.assertLess(self.clock.now, 31)

    def test_pause_inside_sleep_does_not_consume_timeout(self):
        self.token.pause_on_sleep = 30
        result = self.run_wait(timeout=1)
        np.testing.assert_array_equal(result, self.next_image)
        self.assertGreaterEqual(self.clock.now, 30.6 - 1e-9)
        self.assertLess(self.clock.now, 31)

    def test_capture_time_counts_against_timeout(self):
        def slow_capture():
            self.clock.now += .7
            return self.next_image
        with self.assertRaises(MicrographLoadTimeoutError):
            self.run_wait(slow_capture, timeout=1)
        self.assertEqual(self.captures, 2)

    def test_stop_during_capture_prevents_more_captures(self):
        def capture_and_cancel():
            self.token.cancel()
            return self.next_image
        with self.assertRaises(CancelledOperation):
            self.run_wait(capture_and_cancel)
        self.assertEqual(self.captures, 1)
        self.assertEqual(self.token.waits, [])

    def test_already_cancelled_never_captures(self):
        self.token.cancel()
        with self.assertRaises(CancelledOperation):
            self.run_wait()
        self.assertEqual(self.captures, 0)

    def test_target_loss_aborts_before_capture(self):
        with self.assertRaises(NavigationError):
            self.run_wait(check_target=lambda: False)
        self.assertEqual(self.captures, 0)

    def test_target_exception_is_preserved(self):
        failure = RuntimeError('target covered')
        def fail():
            raise failure
        with self.assertRaises(RuntimeError) as context:
            self.run_wait(check_target=fail)
        self.assertIs(context.exception, failure)
        self.assertEqual(self.captures, 0)

    def test_changed_capture_dimensions_abort(self):
        with self.assertRaises(NavigationError):
            self.run_wait(lambda: self.next_image[:, :-1])
        self.assertEqual(self.captures, 1)

    def test_bad_masks_and_settings_fail_before_capture(self):
        for mask in (np.zeros((32, 40), bool), np.ones((32, 40), np.uint8),
                     np.ones((31, 40), bool)):
            with self.subTest(mask_shape=mask.shape, dtype=mask.dtype), self.assertRaises((NavigationError, ValueError)):
                self.run_wait(unchanged_mask=mask)
        for kwargs in ({'timeout': 0}, {'timeout': float('inf')}, {'timeout': True},
                       {'settle_seconds': 2}, {'poll_seconds': 0}, {'minimum_wait': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_wait(**kwargs)
        self.assertEqual(self.captures, 0)

    def test_unsigned_pixels_do_not_wrap_during_change_comparison(self):
        self.previous[:] = 250
        self.next_image[:] = 5
        result = self.run_wait()
        np.testing.assert_array_equal(result, self.next_image)


if __name__ == '__main__':
    unittest.main()
