"""Screen-only navigation acknowledgement, without GUI access or input events.

The caller sends exactly one verified Down key, then invokes this helper from
its worker. An unchanged screen never authorizes another key or another pick.
Recognition must still verify alignment after inference: a stable screenshot
cannot establish the browser's internal loading state.
"""

from __future__ import annotations

import math
from numbers import Real
import time
from typing import Callable, Iterable

import numpy as np

from execution import CancellationToken


class NavigationError(RuntimeError):
    """The next micrograph cannot be verified from the selected screen area."""


class NoNewMicrographError(NavigationError):
    """Down did not produce a distinguishable next micrograph before timeout."""


class MicrographLoadTimeoutError(NavigationError):
    """A change appeared, but no usable, stable micrograph was established."""


class RepeatedMicrographError(NavigationError):
    """A stable capture matches a recently processed micrograph."""


def _navigation_seconds(value: float, name: str, *, zero_allowed: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number of seconds.")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
        raise ValueError(f"{name} must be {'nonnegative' if zero_allowed else 'positive'} and finite.")
    return value


def _navigation_rgb(value, *, expected_shape=None) -> np.ndarray:
    frame = np.asarray(value)
    if (frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0]
            or not frame.shape[1] or frame.dtype != np.uint8):
        raise NavigationError("Navigation requires a nonempty RGB screenshot with 8-bit pixels.")
    if expected_shape is not None and frame.shape != expected_shape:
        raise NavigationError("The selected screen area changed size while moving to the next micrograph.")
    return frame


def _navigation_changed_fraction(before: np.ndarray, after: np.ndarray, mask=None) -> float:
    difference = np.max(np.abs(before.astype(np.int16) - after.astype(np.int16)), axis=2)
    if mask is not None:
        difference = difference[mask]
    return float(np.mean(difference > 12))


def wait_for_next_micrograph(
    capture: Callable[[], np.ndarray],
    previous: np.ndarray,
    token: CancellationToken,
    *,
    unchanged_mask: np.ndarray | None = None,
    history: Iterable[np.ndarray] = (),
    timeout: float = 15.0,
    minimum_wait: float = 0.5,
    settle_seconds: float = 0.6,
    poll_seconds: float = 0.15,
    check_target: Callable[[], object] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> np.ndarray:
    """Wait for one different, nonblank, stable micrograph after one Down key.

    ``unchanged_mask`` selects pixels outside previous pick-marker margins.
    It is used for previous/history comparisons so late marker painting cannot
    masquerade as navigation. At least 64 pixels and 2% of the ROI must remain
    when a mask is supplied. Stability is compared against a fixed whole-image
    anchor, so small accumulating changes cannot pass merely because adjacent
    screenshots are similar. All returned pixels are copied from the capture.

    The default change/stability thresholds are respectively 2% and 1% of
    pixels, using a 12-level RGB tolerance. Obvious blank loading screens are
    rejected when the previous scene was textured. Recent images are checked
    only after settling, allowing transient browser previews to disappear.

    Pause time does not consume the active timeout or settling interval. Stop
    interrupts every wait through ``token``. ``check_target`` runs before each
    capture and before returning; it should raise on obscuration/focus loss.
    Neither this module nor this function sends keys, mouse events, or retries.
    """
    if not callable(capture) or not callable(clock):
        raise TypeError("Capture and clock must be callable.")
    if check_target is not None and not callable(check_target):
        raise TypeError("Target check must be callable.")
    timeout = _navigation_seconds(timeout, "Load timeout")
    minimum_wait = _navigation_seconds(minimum_wait, "Minimum wait", zero_allowed=True)
    settle_seconds = _navigation_seconds(settle_seconds, "Settling time")
    poll_seconds = _navigation_seconds(poll_seconds, "Polling interval")
    if minimum_wait > timeout or settle_seconds > timeout:
        raise ValueError("Load timeout must cover both the minimum wait and settling time.")

    baseline = _navigation_rgb(previous).copy()
    mask = None
    if unchanged_mask is not None:
        mask = np.asarray(unchanged_mask)
        if mask.dtype != np.bool_ or mask.shape != baseline.shape[:2]:
            raise ValueError("The unchanged-background mask must be boolean and match the screenshot.")
        if np.count_nonzero(mask) < max(64, math.ceil(0.02 * mask.size)):
            raise NavigationError("Too little unchanged background to verify the next micrograph. Select a larger screen area.")
        mask = mask.copy()
    recent = tuple(_navigation_rgb(frame, expected_shape=baseline.shape) for frame in history)

    # Bound only the blank-screen heuristic's sampling cost. Scene difference
    # and stability still inspect every applicable physical screen pixel.
    indices = np.flatnonzero(mask) if mask is not None else np.arange(baseline.shape[0] * baseline.shape[1])
    indices = indices[::max(1, math.ceil(len(indices) / 65536))]

    def texture_span(frame):
        values = frame.reshape(-1, 3)[indices].mean(axis=1, dtype=np.float32)
        lower, upper = np.percentile(values, [10, 90])
        return float(upper - lower)

    baseline_is_textured = texture_span(baseline) >= 6.0
    start = clock()
    paused_seconds = 0.0

    def active_elapsed():
        return max(0.0, clock() - start - paused_seconds)

    def ready():
        nonlocal paused_seconds
        before = clock()
        token.wait_until_ready()
        # Normally a few microseconds; any blocking here is the pause gate.
        blocked = max(0.0, clock() - before)
        paused_seconds += blocked
        return blocked

    def pause_aware_sleep(duration):
        nonlocal paused_seconds
        before = clock()
        token.sleep(duration)
        # CancellationToken.sleep gates both before and after its timed wait.
        # Any excess wait is excluded, including a pause arriving mid-sleep.
        excess = max(0.0, clock() - before - duration)
        paused_seconds += excess
        return excess

    def verify_target():
        if check_target is not None and check_target() is False:
            raise NavigationError("The selected micrograph window is no longer visible or focused.")

    anchor = None
    anchor_started = 0.0
    ever_changed = False
    while True:
        if ready() > 0.01:
            anchor = None
        if active_elapsed() >= timeout - 1e-9:
            break
        verify_target()
        current = _navigation_rgb(capture(), expected_shape=baseline.shape).copy()
        if ready() > 0.01:
            # A capture made before a pause must not become the final frame
            # merely because it looked stable before the user paused.
            anchor = None
            continue
        elapsed = active_elapsed()
        if elapsed > timeout:
            break
        if _navigation_changed_fraction(baseline, current, mask) < 0.02:
            anchor = None
        else:
            ever_changed = True
            if baseline_is_textured and texture_span(current) < 2.0:
                anchor = None
            elif anchor is None or _navigation_changed_fraction(anchor, current) > 0.01:
                anchor, anchor_started = current, elapsed
            elif (elapsed >= minimum_wait
                  and elapsed - anchor_started >= settle_seconds - 1e-9):
                if any(_navigation_changed_fraction(old, current, mask) < 0.02 for old in recent):
                    raise RepeatedMicrographError(
                        "A recent micrograph reappeared after Down. Navigation stopped to avoid picking the same image twice."
                    )
                if ready() > 0.01:
                    anchor = None
                    continue
                if active_elapsed() > timeout:
                    break
                verify_target()
                return current
        remaining = timeout - active_elapsed()
        if remaining <= 1e-9:
            break
        if pause_aware_sleep(min(poll_seconds, remaining)) > 0.01:
            anchor = None

    if not ever_changed:
        raise NoNewMicrographError(
            "No new micrograph appeared after Down. The last micrograph may have been reached, "
            "or cryoSPARC did not receive the key. Navigation stopped without sending another Down key."
        )
    raise MicrographLoadTimeoutError(
        f"The next micrograph did not finish loading and settle within {timeout:g} seconds. "
        "Increase the load timeout or wait for cryoSPARC, then try again."
    )
