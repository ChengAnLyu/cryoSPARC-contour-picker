"""Headless, cancellable execution for SARU Picker.

This module never imports Tkinter, pynput, or PyAutoGUI. A GUI may submit work,
poll results on its own event loop, and provide a click adapter when the user
starts playback. In particular, worker callbacks must not access GUI widgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from numbers import Integral, Real
import queue
import threading
import traceback
from typing import Any, Callable, Iterable


EXECUTION_LOGGER = logging.getLogger(__name__)


class CancelledOperation(RuntimeError):
    """A cooperative operation was cancelled by its owner."""


class WorkerBusyError(RuntimeError):
    """A runner already has work running or a result awaiting collection."""


class CancellationToken:
    """Thread-safe cancellation and pause state for exactly one generation.

    Cancellation is cooperative: it cannot interrupt a native model forward
    pass that is already running. It does interrupt pauses and click delays.
    """

    def __init__(self, generation: int = 0) -> None:
        if isinstance(generation, bool) or not isinstance(generation, Integral):
            raise ValueError("Generation must be an integer.")
        self.generation = int(generation)
        self._cancel = threading.Event()
        self._ready = threading.Event()
        self._ready.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return not self._ready.is_set() and not self.cancelled

    def cancel(self) -> None:
        self._cancel.set()
        self._ready.set()  # Wake a paused operation immediately.

    def pause(self) -> None:
        if not self.cancelled:
            self._ready.clear()

    def resume(self) -> None:
        self._ready.set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledOperation("Operation cancelled.")

    def wait_until_ready(self) -> None:
        self.check_cancelled()
        while not self._ready.wait(0.02):
            self.check_cancelled()
        self.check_cancelled()  # Cancellation also sets _ready.

    def sleep(self, seconds: float) -> None:
        """Wait an interruptible delay, then respect any active pause."""
        seconds = _execution_nonnegative_finite(seconds, "Delay")
        self.wait_until_ready()
        if self._cancel.wait(seconds):
            raise CancelledOperation("Operation cancelled.")
        self.wait_until_ready()


def _execution_nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _execution_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _execution_integer(value: int, name: str) -> int:
    numeric = _execution_finite(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer pixel coordinate.")
    return int(numeric)


@dataclass(frozen=True)
class CaptureGeometry:
    """An unscaled capture rectangle in desktop pixel coordinates.

    All rectangles use half-open bounds: left <= x < left + width. Capture
    images must have exactly (width, height) pixels. A negative desktop origin
    is representable here; the chosen desktop adapter must support it too.
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("left", "top", "width", "height"):
            object.__setattr__(self, name, _execution_integer(getattr(self, name), name))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture width and height must both be positive.")

    @property
    def rectangle(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height

    def contains(self, x: float, y: float) -> bool:
        x, y = _execution_finite(x, "x"), _execution_finite(y, "y")
        return (
            self.left <= x < self.left + self.width
            and self.top <= y < self.top + self.height
        )

    def local_to_screen(self, x: float, y: float) -> tuple[int, int]:
        x, y = _execution_finite(x, "x"), _execution_finite(y, "y")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Image point lies outside the capture.")
        return self.left + math.floor(x), self.top + math.floor(y)

    def screen_to_local(self, x: float, y: float) -> tuple[float, float]:
        x, y = _execution_finite(x, "x"), _execution_finite(y, "y")
        if not self.contains(x, y):
            raise ValueError("Desktop point lies outside the capture.")
        return x - self.left, y - self.top

    def viewer_to_local(
        self,
        x: float,
        y: float,
        display_width: float,
        display_height: float,
        offset_x: float = 0,
        offset_y: float = 0,
    ) -> tuple[float, float]:
        """Map viewer coordinates through its displayed image rectangle.

        Offsets account for letterboxing or canvas padding. Points on the
        right/bottom outer boundary are rejected, not silently clamped.
        """
        x, y = _execution_finite(x, "x"), _execution_finite(y, "y")
        display_width = _execution_finite(display_width, "Displayed width")
        display_height = _execution_finite(display_height, "Displayed height")
        offset_x = _execution_finite(offset_x, "Horizontal offset")
        offset_y = _execution_finite(offset_y, "Vertical offset")
        if display_width <= 0 or display_height <= 0:
            raise ValueError("Displayed image dimensions must be positive.")
        dx, dy = x - offset_x, y - offset_y
        if not (0 <= dx < display_width and 0 <= dy < display_height):
            raise ValueError("Viewer point lies outside the displayed image.")
        return dx * self.width / display_width, dy * self.height / display_height


@dataclass(frozen=True)
class TaskResult:
    token: CancellationToken
    value: Any = None
    error: Exception | None = None
    cancelled: bool = False
    traceback_text: str | None = None


class TaskRunner:
    """One daemon worker, one outstanding job, and generation-safe results.

    Busy remains true until ``poll_results`` collects the completed job. This
    bounds pending results and prevents submission loops from accumulating
    model images. ``invalidate`` cancels work and advances the generation;
    stale output is discarded by ``poll_results``. A running native inference
    must return before the worker can accept another model operation.
    """

    def __init__(self, initial_generation: int = 0) -> None:
        self._generation = CancellationToken(initial_generation).generation
        self._lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._results: queue.Queue[TaskResult] = queue.Queue(maxsize=1)
        self._busy = False
        self._closed = False
        self._active_token: CancellationToken | None = None
        # The submitting/polling thread owns the final reference to callbacks.
        # A screen callback may close over Tk objects, whose destructors must
        # never run when the worker starts the next operation.
        self._active_function: Callable | None = None
        self._thread = threading.Thread(
            target=self._worker, name="saru-worker", daemon=True
        )
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def invalidate(self) -> int:
        with self._lock:
            self._generation += 1
            if self._active_token is not None:
                self._active_token.cancel()
            return self._generation

    def submit(
        self,
        fn: Callable[[CancellationToken], Any],
        token: CancellationToken | None = None,
    ) -> CancellationToken:
        if not callable(fn):
            raise TypeError("Worker task must be callable.")
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker has been shut down.")
            if self._busy:
                raise WorkerBusyError("Wait for the current operation to finish.")
            token = token if token is not None else CancellationToken(self._generation)
            if token.generation != self._generation or token.cancelled:
                raise ValueError("Task token is cancelled or belongs to a stale generation.")
            self._active_token = token
            self._active_function = fn
            self._busy = True
            self._jobs.put_nowait((token, fn))
            return token

    def poll_results(self) -> list[TaskResult]:
        """Collect completed work; call this from the GUI's event loop."""
        results = []
        while True:
            with self._lock:
                try:
                    result = self._results.get_nowait()
                except queue.Empty:
                    if self._closed and not self._thread.is_alive():
                        # Shutdown can win before a queued callback starts.
                        # Drain those callback references on this thread too.
                        while True:
                            try:
                                self._jobs.get_nowait()
                            except queue.Empty:
                                break
                        self._active_function = self._active_token = None
                        self._busy = False
                    break
                if self._active_token is result.token:
                    self._active_token = None
                    self._active_function = None
                    self._busy = False
                if result.token.generation == self._generation and not self._closed:
                    results.append(result)
        return results

    def shutdown(self, wait: bool = False, timeout: float = 2.0) -> bool:
        """Stop accepting work and request cancellation; return worker status.

        ``wait=False`` keeps a window close callback responsive. A native
        inference already in progress cannot be forcibly interrupted.
        """
        timeout = _execution_nonnegative_finite(timeout, "Shutdown timeout")
        with self._lock:
            self._closed = True
            if self._active_token is not None:
                self._active_token.cancel()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        stopped = not self._thread.is_alive()
        if stopped:
            self.poll_results()
        return stopped

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            item = self._jobs.get()
            if item is None:
                return
            token, fn = item
            value = None
            try:
                token.wait_until_ready()
                value = fn(token)
                token.check_cancelled()
                result = TaskResult(token=token, value=value)
            except CancelledOperation:
                result = TaskResult(token=token, cancelled=True)
            except Exception as exc:
                trace = traceback.format_exc()
                EXECUTION_LOGGER.exception("Picker operation failed; the operation was stopped.")
                # Keep the useful formatted stack, not live worker frames and
                # their GUI-capturing closures. Exception chains need the same
                # treatment or a later collection can still own those frames.
                pending, visited = [exc], set()
                while pending:
                    failure = pending.pop()
                    if id(failure) in visited:
                        continue
                    visited.add(id(failure))
                    pending.extend(part for part in (failure.__cause__, failure.__context__) if part is not None)
                    failure.__traceback__ = None
                del pending, visited, failure
                result = TaskResult(token=token, error=exc, traceback_text=trace)
            with self._lock:
                self._results.put_nowait(result)
                # poll_results takes this same lock. By the time it can drop
                # the owning references, this thread holds no old callback,
                # return value or exception result while awaiting the next job.
                del item, fn, value, result, token


def play_clicks(
    points: Iterable[tuple[int, int]],
    click_fn: Callable[[int, int], Any],
    geometry: CaptureGeometry,
    token: CancellationToken,
    delay: float = 0.08,
    max_clicks: int = 10_000,
    progress: Callable[[int, int], Any] | None = None,
) -> int:
    """Validate a screen-coordinate batch, then play it with cooperative control.

    Every coordinate is copied and checked before the first click. Duplicate
    coordinates are removed in their original order, to avoid toggling an
    existing pick. Invalid inputs reject the whole batch. Any click exception
    propagates immediately; a caller must not advance to another image after
    such an error. Progress callbacks execute on the caller's worker thread.

    ``delay`` is the wait *between* clicks. The click adapter should disable
    additional automatic pauses while leaving its fail-safe enabled.
    """
    if not callable(click_fn):
        raise TypeError("click_fn must be callable.")
    if not isinstance(geometry, CaptureGeometry):
        raise TypeError("geometry must be a CaptureGeometry.")
    if not isinstance(token, CancellationToken):
        raise TypeError("token must be a CancellationToken.")
    delay = _execution_nonnegative_finite(delay, "Click delay")
    max_clicks = _execution_integer(max_clicks, "Maximum clicks")
    if max_clicks < 1:
        raise ValueError("Maximum clicks must be positive.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable.")

    snapshot = []
    seen = set()
    token.check_cancelled()
    for index, point in enumerate(points):
        if index >= max_clicks:
            raise ValueError(f"Batch exceeds the limit of {max_clicks} input points.")
        token.check_cancelled()
        try:
            x, y = point
        except (TypeError, ValueError) as exc:
            raise ValueError("Each click must contain exactly two coordinates.") from exc
        xy = _execution_integer(x, "Click x"), _execution_integer(y, "Click y")
        if not geometry.contains(*xy):
            raise ValueError(f"Click {xy} lies outside the capture region.")
        if xy not in seen:
            seen.add(xy)
            snapshot.append(xy)
    snapshot = tuple(snapshot)

    completed = 0
    total = len(snapshot)
    for x, y in snapshot:
        token.wait_until_ready()
        click_fn(x, y)
        completed += 1
        if progress is not None:
            progress(completed, total)
        token.check_cancelled()
        if completed < total:
            token.sleep(delay)
    return completed
