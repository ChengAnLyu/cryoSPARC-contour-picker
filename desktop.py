"""Windows desktop adapter; importing this module performs no desktop actions.

Live capture/playback supports the primary monitor. Offline image analysis is
platform independent. Call ``enable_dpi_awareness`` before constructing Tk.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
from numbers import Real
import platform
import threading
import time
from typing import Callable

import numpy as np

from execution import CaptureGeometry


DESKTOP_LOGGER = logging.getLogger(__name__)


class DesktopError(RuntimeError):
    """Capture, focus, or desktop automation could not be verified."""


def enable_dpi_awareness(
    platform_name: str | None = None, *, user32=None, shcore=None
) -> bool:
    """Request per-monitor-v2 coordinates before Tk/PyAutoGUI are imported.

    Older Windows versions fall back to per-monitor or system awareness. The
    latter is adequate for the primary-monitor-only capture contract here.
    Returns False on unsupported platforms or when awareness cannot be set or
    verified; the caller can still offer offline image analysis.
    """
    if (platform_name or platform.system()) != "Windows":
        return False
    try:
        injected = user32 is not None
        user32 = user32 or ctypes.WinDLL("user32", use_last_error=True)
        modern = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if modern is not None:
            if not injected:
                modern.argtypes = [ctypes.c_void_p]
                modern.restype = ctypes.c_bool
            if modern(ctypes.c_void_p(-4)):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                DESKTOP_LOGGER.info("Windows per-monitor-v2 DPI awareness enabled.")
                return True
            context = getattr(user32, "GetThreadDpiAwarenessContext", None)
            awareness = getattr(user32, "GetAwarenessFromDpiAwarenessContext", None)
            if context is not None and awareness is not None:
                if not injected:
                    context.argtypes = []
                    context.restype = ctypes.c_void_p
                    awareness.argtypes = [ctypes.c_void_p]
                    awareness.restype = ctypes.c_int
                if awareness(context()) == 2:
                    DESKTOP_LOGGER.info("Windows per-monitor DPI awareness was already active.")
                    return True
        if shcore is None and not injected:
            try:
                shcore = ctypes.WinDLL("shcore", use_last_error=True)
            except OSError:
                shcore = None
        if shcore is not None:
            setter = getattr(shcore, "SetProcessDpiAwareness", None)
            if setter is not None:
                if not injected:
                    setter.argtypes = [ctypes.c_int]
                    setter.restype = ctypes.c_long
                if setter(2) == 0:
                    DESKTOP_LOGGER.info("Windows per-monitor DPI awareness enabled.")
                    return True
        legacy = getattr(user32, "SetProcessDPIAware", None)
        if legacy is not None and legacy():
            DESKTOP_LOGGER.warning(
                "Using system DPI awareness; desktop playback is restricted to the primary monitor."
            )
            return True
        already_aware = getattr(user32, "IsProcessDPIAware", None)
        if already_aware is not None and already_aware():
            DESKTOP_LOGGER.warning(
                "DPI awareness was already configured; verify coordinate alignment on the primary monitor."
            )
            return True
    except Exception:
        DESKTOP_LOGGER.exception("Could not configure Windows DPI awareness.")
        return False
    DESKTOP_LOGGER.error("Windows DPI awareness could not be enabled or verified.")
    return False


class _DesktopWindowsAPI:
    """Small Win32 facade with pointer-safe signatures on 64-bit Python."""

    def __init__(self) -> None:
        from ctypes import wintypes

        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        specs = {
            "GetForegroundWindow": ([], wintypes.HWND),
            "GetAncestor": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            "WindowFromPoint": ([wintypes.POINT], wintypes.HWND),
            "IsWindow": ([wintypes.HWND], wintypes.BOOL),
            "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
            "IsIconic": ([wintypes.HWND], wintypes.BOOL),
            "ShowWindow": ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
            "SetForegroundWindow": ([wintypes.HWND], wintypes.BOOL),
            "GetSystemMetrics": ([ctypes.c_int], ctypes.c_int),
        }
        for name, (args, result) in specs.items():
            fn = getattr(self._user32, name)
            fn.argtypes = args
            fn.restype = result

    def root_window(self, handle: int) -> int:
        return int(self._user32.GetAncestor(handle, 2) or handle)  # GA_ROOT

    def foreground(self) -> int:
        return int(self._user32.GetForegroundWindow() or 0)

    def is_window(self, handle: int) -> bool:
        return bool(self._user32.IsWindow(handle))

    def is_visible(self, handle: int) -> bool:
        return bool(self._user32.IsWindowVisible(handle))

    def is_minimized(self, handle: int) -> bool:
        return bool(self._user32.IsIconic(handle))

    def restore(self, handle: int) -> None:
        self._user32.ShowWindow(handle, 9)  # SW_RESTORE

    def focus(self, handle: int) -> bool:
        return bool(self._user32.SetForegroundWindow(handle))

    def window_at(self, x: int, y: int) -> int:
        return int(self._user32.WindowFromPoint(self._wintypes.POINT(x, y)) or 0)

    def primary_size(self) -> tuple[int, int]:
        return self._user32.GetSystemMetrics(0), self._user32.GetSystemMetrics(1)


def _desktop_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DesktopError(f"{name} must be an integer pixel coordinate.")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise DesktopError(f"{name} must be an integer pixel coordinate.")
    return int(numeric)


class DesktopAdapter:
    """Lazily loaded desktop capture and verified-target click adapter.

    ``backend`` and ``windows_api`` may be fake adapters in tests. The actual
    PyAutoGUI backend is configured with PAUSE=0 and FAILSAFE=True. ``click``
    and ``press_next`` require a target established by ``focus_target``.
    """

    def __init__(
        self, backend=None, windows_api=None, platform_name: str | None = None
    ) -> None:
        self._backend = backend
        self._windows_api = windows_api
        self._platform_name = platform_name or platform.system()
        self._target: int | None = None

    def _require_windows(self) -> None:
        if self._platform_name != "Windows":
            raise DesktopError(
                "Live capture and clicking are supported on Windows only. "
                "Use Open Image for offline analysis on this platform."
            )

    def _get_backend(self):
        self._require_windows()
        if self._backend is None:
            try:
                self._backend = importlib.import_module("pyautogui")
            except Exception as exc:
                raise DesktopError(
                    "Could not load PyAutoGUI. Install the desktop dependencies and retry."
                ) from exc
        self._backend.PAUSE = 0
        self._backend.FAILSAFE = True
        return self._backend

    def _get_api(self):
        self._require_windows()
        if self._windows_api is None:
            self._windows_api = _DesktopWindowsAPI()
        return self._windows_api

    def size(self) -> tuple[int, int]:
        reported = self._get_backend().size()
        try:
            width, height = reported
            width, height = _desktop_integer(width, "Screen width"), _desktop_integer(height, "Screen height")
        except (TypeError, ValueError) as exc:
            raise DesktopError("Desktop backend returned an invalid screen size.") from exc
        if width <= 0 or height <= 0:
            raise DesktopError("Primary monitor dimensions must be positive.")
        physical = tuple(self._get_api().primary_size())
        if physical != (width, height):
            raise DesktopError(
                "Windows and the capture backend disagree about screen dimensions. "
                "Restart with DPI awareness enabled before using live playback."
            )
        return width, height

    def validate_region(self, region: CaptureGeometry) -> None:
        if not isinstance(region, CaptureGeometry):
            raise DesktopError("Capture region must be a CaptureGeometry.")
        width, height = self.size()
        if (
            region.left < 0
            or region.top < 0
            or region.left + region.width > width
            or region.top + region.height > height
        ):
            raise DesktopError("The capture region must lie entirely on the primary monitor.")

    def screenshot(self, region: CaptureGeometry | None = None) -> np.ndarray:
        width, height = self.size()
        if region is not None:
            self.validate_region(region)
            expected = (region.height, region.width)
            image = self._get_backend().screenshot(region=region.rectangle)
        else:
            expected = (height, width)
            image = self._get_backend().screenshot()
        if hasattr(image, "convert"):
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        else:
            array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape != (*expected, 3):
            raise DesktopError(
                "Screenshot dimensions or color format do not match the desktop region. "
                "Live clicking is unavailable until capture coordinates align."
            )
        return np.array(array, dtype=np.uint8, order="C", copy=True)

    def _validate_handle(self, handle) -> int:
        handle = _desktop_integer(handle, "Target window handle")
        api = self._get_api()
        if handle <= 0 or not api.is_window(handle):
            raise DesktopError("The captured target window no longer exists. Capture the region again.")
        if not api.is_visible(handle):
            raise DesktopError("The captured target window is not visible.")
        return handle

    def capture_target(self) -> int | None:
        if self._platform_name != "Windows":
            return None
        api = self._get_api()
        return self._validate_handle(api.root_window(api.foreground()))

    def window_under_roi(self, region: CaptureGeometry) -> int:
        """Require ROI center and corners to belong to one top-level window."""
        self.validate_region(region)
        api = self._get_api()
        left, top = region.left, region.top
        right, bottom = left + region.width - 1, top + region.height - 1
        samples = (
            (left, top), (right, top), (left, bottom), (right, bottom),
            ((left + right) // 2, (top + bottom) // 2),
        )
        handles = {api.root_window(api.window_at(x, y)) for x, y in samples}
        if len(handles) != 1:
            raise DesktopError("The capture region overlaps more than one application window.")
        return self._validate_handle(handles.pop())

    def focus_target(self, handle) -> None:
        handle = self._validate_handle(handle)
        api = self._get_api()
        if api.is_minimized(handle):
            api.restore(handle)
        if api.root_window(api.foreground()) != handle:
            if not api.focus(handle):
                raise DesktopError(
                    "Windows did not allow the picker to focus the captured window. "
                    "Bring the picking application to the front and retry."
                )
            deadline = time.monotonic() + 0.3
            while api.root_window(api.foreground()) != handle and time.monotonic() < deadline:
                time.sleep(0.01)
        self.ensure_target(handle)
        self._target = handle

    def ensure_target(self, handle=None) -> None:
        handle = self._target if handle is None else handle
        if handle is None:
            raise DesktopError("No captured target window has been focused for playback.")
        handle = self._validate_handle(handle)
        api = self._get_api()
        if api.is_minimized(handle) or api.root_window(api.foreground()) != handle:
            raise DesktopError("The picking application lost focus. Playback has stopped.")

    def click(self, x: int, y: int) -> None:
        x, y = _desktop_integer(x, "Click x"), _desktop_integer(y, "Click y")
        width, height = self.size()
        if not (0 <= x < width and 0 <= y < height):
            raise DesktopError("Click lies outside the primary monitor.")
        self.ensure_target()
        api = self._get_api()
        if api.root_window(api.window_at(x, y)) != self._target:
            raise DesktopError("Another window covers this picking position. Playback has stopped.")
        self._get_backend().click(x, y)

    def press_next(self, key: str = "down") -> None:
        if not isinstance(key, str) or key not in {
            "down", "up", "left", "right", "pagedown", "pageup", "n"
        }:
            raise DesktopError("Choose a supported single navigation key.")
        self.ensure_target()
        self._get_backend().press(key)


class GlobalHotkeys:
    """Debounced global Escape/F8 listener with queue-only callbacks.

    The callback receives ``'stop'`` or ``'pause'`` on a listener thread and
    must only enqueue a message or update thread-safe cancellation state.
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        listener_factory=None,
        key_enum=None,
        platform_name: str | None = None,
    ) -> None:
        if not callable(callback):
            raise TypeError("Hotkey callback must be callable.")
        self._callback = callback
        self._factory = listener_factory
        self._keys = key_enum
        self._platform_name = platform_name or platform.system()
        self._listener = None
        self._pressed: set = set()
        self._lock = threading.Lock()
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        if self._platform_name != "Windows":
            raise DesktopError("Global desktop hotkeys are enabled only for Windows live mode.")
        if self._factory is None or self._keys is None:
            try:
                keyboard = importlib.import_module("pynput.keyboard")
            except Exception as exc:
                raise DesktopError("Global hotkeys could not start; install pynput.") from exc
            self._factory = self._factory or keyboard.Listener
            self._keys = self._keys or keyboard.Key
        self._pressed.clear()
        self._active = True
        try:
            self._listener = self._factory(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        except Exception:
            self._active = False
            self._listener = None
            raise

    def _on_press(self, key) -> None:
        if self._keys is None or key not in (self._keys.esc, self._keys.f8):
            return
        with self._lock:
            if not self._active or key in self._pressed:
                return
            self._pressed.add(key)
        try:
            self._callback("stop" if key == self._keys.esc else "pause")
        except Exception:
            DESKTOP_LOGGER.exception("Hotkey callback failed.")

    def _on_release(self, key) -> None:
        with self._lock:
            self._pressed.discard(key)

    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._pressed.clear()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()

