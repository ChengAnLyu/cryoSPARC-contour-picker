"""Windows screen annotations and gesture input without a screenshot editor.

Call ``desktop.enable_dpi_awareness`` before creating Tk. Coordinates here are
physical, unscaled pixels on the primary monitor, whose origin is (0, 0).
All ScreenOverlay/InputSurface methods must run on their creating GUI thread.
The controller must hide *both* surfaces and allow a paint turn before taking
a screenshot or replaying clicks. Hiding only the input surface is insufficient.
The control panel remains visible above both surfaces. Model capture regions
must exclude its native outer rectangle; hiding these surfaces never hides it.

The annotation window is explicitly mouse-transparent and nonactivating.
The separate input window has nonzero alpha and no transparent color: fully
transparent Windows pixels also pass mouse input through to the application
beneath them. InputSurface therefore intercepts gestures without replacing the
live cryoSPARC image with a captured image.

ManualMouseObserver has no Tk dependency. Its listener thread emits only small
tuples to a supplied queue callback and ignores injected desktop input.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import platform
import threading
from typing import Callable

import numpy as np
from PIL import Image

from execution import CaptureGeometry


OVERLAY_LOG = logging.getLogger(__name__)
TRANSPARENT_COLOR = "#010203"
TRANSPARENT_RGB = (1, 2, 3)
INPUT_ALPHA = 0.02


class OverlayError(RuntimeError):
    """A screen surface cannot be created or its coordinate contract failed."""


def _screen_xy(x, y) -> tuple[int, int]:
    values = (float(x), float(y))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Screen coordinates must be finite.")
    return math.floor(values[0]), math.floor(values[1])


class _OverlayWindowsAPI:
    """Pointer-safe Win32 operations, created only when an overlay is needed."""

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_NOACTIVATE = 0x08000000
    WS_EX_TOOLWINDOW = 0x00000080

    def __init__(self, user32=None):
        from ctypes import wintypes

        self.user32 = user32 or ctypes.WinDLL("user32", use_last_error=True)
        self._get_long = getattr(self.user32, "GetWindowLongPtrW", None)
        self._set_long = getattr(self.user32, "SetWindowLongPtrW", None)
        if self._get_long is None or self._set_long is None:
            self._get_long = self.user32.GetWindowLongW
            self._set_long = self.user32.SetWindowLongW
        if user32 is None:
            self._get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            self._get_long.restype = ctypes.c_ssize_t
            self._set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            self._set_long.restype = ctypes.c_ssize_t
            self.user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            self.user32.GetAncestor.restype = wintypes.HWND
            self.user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            self.user32.GetSystemMetrics.restype = ctypes.c_int
            self.user32.WindowFromPoint.argtypes = [wintypes.POINT]
            self.user32.WindowFromPoint.restype = wintypes.HWND
            self.user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
            self.user32.GetWindowRect.restype = wintypes.BOOL
            self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
            self.user32.IsWindowVisible.restype = wintypes.BOOL
            self.user32.IsIconic.argtypes = [wintypes.HWND]
            self.user32.IsIconic.restype = wintypes.BOOL
            self.user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
            self.user32.GetWindow.restype = wintypes.HWND
            self.user32.GetForegroundWindow.argtypes = []
            self.user32.GetForegroundWindow.restype = wintypes.HWND
            self.user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                               ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int,
                                               wintypes.UINT]
            self.user32.SetWindowPos.restype = wintypes.BOOL
        self._point_type = wintypes.POINT
        self._rect_type = wintypes.RECT

    def primary_size(self) -> tuple[int, int]:
        return self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1)

    def root_handle(self, child: int) -> int:
        return int(self.user32.GetAncestor(int(child), 2) or child)

    def window_at(self, x: int, y: int) -> int:
        return self.root_handle(int(self.user32.WindowFromPoint(self._point_type(x, y)) or 0))

    def styles(self, child: int) -> int:
        return int(self._get_long(self.root_handle(child), self.GWL_EXSTYLE))

    def visible(self, child: int) -> bool:
        return bool(self.user32.IsWindowVisible(self.root_handle(child)))

    def minimized(self, child: int) -> bool:
        return bool(self.user32.IsIconic(self.root_handle(child)))

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    def bounds(self, child: int) -> tuple[int, int, int, int]:
        rectangle = self._rect_type()
        if not self.user32.GetWindowRect(self.root_handle(child), ctypes.byref(rectangle)):
            raise OverlayError("The control panel's screen bounds could not be read.")
        width, height = rectangle.right - rectangle.left, rectangle.bottom - rectangle.top
        if width <= 0 or height <= 0:
            raise OverlayError("The control panel has invalid screen bounds.")
        return rectangle.left, rectangle.top, width, height

    def is_above(self, first: int, second: int) -> bool:
        """Compare two top-level windows without changing focus or z-order."""
        first, current = self.root_handle(first), self.root_handle(second)
        visited = {current}
        for _ in range(8192):
            current = int(self.user32.GetWindow(current, 3) or 0)  # GW_HWNDPREV
            if not current or current in visited:
                return False
            if current == first:
                return True
            visited.add(current)
        return False

    def configure(self, child: int, *, click_through: bool) -> int:
        handle = self.root_handle(child)
        if handle <= 0:
            raise OverlayError("The overlay has no native Windows handle.")
        style = int(self._get_long(handle, self.GWL_EXSTYLE))
        style |= self.WS_EX_LAYERED | self.WS_EX_NOACTIVATE | self.WS_EX_TOOLWINDOW
        if click_through:
            style |= self.WS_EX_TRANSPARENT
        else:
            style &= ~self.WS_EX_TRANSPARENT
        # A zero return can mean success when the former style was zero.
        if hasattr(ctypes, "set_last_error"):
            ctypes.set_last_error(0)
        previous = self._set_long(handle, self.GWL_EXSTYLE, style)
        error = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
        if previous == 0 and error:
            raise OverlayError(f"Could not configure the screen overlay (Windows error {error}).")
        return handle

    def raise_without_activation(self, child: int) -> None:
        # HWND_TOPMOST; NOSIZE | NOMOVE | NOACTIVATE | FRAMECHANGED | SHOWWINDOW.
        if not self.user32.SetWindowPos(self.root_handle(child), -1, 0, 0, 0, 0,
                                       0x0001 | 0x0002 | 0x0010 | 0x0020 | 0x0040):
            raise OverlayError("Could not show the screen overlay without activating it.")


def control_panel_bounds(root, *, style_api=None, platform_name=None) -> tuple[int, int, int, int]:
    """Return the panel's screen (left, top, width, height) on the GUI thread.

    Windows uses the complete native outer window, including caption and frame.
    The Tk client bounds are available only as an offline non-Windows fallback;
    live screen capture remains Windows-only.
    """
    if (platform_name or platform.system()) == "Windows":
        api = style_api or _OverlayWindowsAPI()
        return api.bounds(root.winfo_id())
    return (int(root.winfo_rootx()), int(root.winfo_rooty()),
            int(root.winfo_width()), int(root.winfo_height()))


def keep_control_panel_above(root, *, style_api=None, platform_name=None) -> bool:
    """Restack a visible control panel without activating it or changing input.

    Call on the GUI thread after showing a screen surface or a controller
    transition. An intentionally minimized/hidden panel is not resurrected.
    The screen surfaces call this automatically after each ``show``.
    """
    if (platform_name or platform.system()) != "Windows":
        root.attributes("-topmost", True)
        return False
    api = style_api or _OverlayWindowsAPI()
    handle = root.winfo_id()
    if not api.visible(handle) or api.minimized(handle):
        return False
    # Do not configure the panel as a passive overlay. Its buttons, caption,
    # keyboard focus, and normal mouse interaction must remain available.
    api.raise_without_activation(handle)
    return True


class _ScreenSurface:
    def __init__(self, root, *, window_factory=None, canvas_factory=None,
                 style_api=None, platform_name=None, control_panel=None):
        self.root = root
        self.control_panel = root if control_panel is None else control_panel
        self.window = None
        self.canvas = None
        self.visible = False
        self.closed = False
        self._gui_thread = threading.get_ident()
        self._window_factory = window_factory
        self._canvas_factory = canvas_factory
        self._style_api = style_api
        self._platform_name = platform_name or platform.system()

    def _check_thread(self):
        if threading.get_ident() != self._gui_thread:
            raise RuntimeError("Screen overlay methods must run on the GUI thread.")

    def _api(self):
        if self._platform_name != "Windows":
            raise OverlayError("Live screen overlays require Windows.")
        if self._style_api is None:
            self._style_api = _OverlayWindowsAPI()
        return self._style_api

    def _make_window(self, bounds, *, input_surface):
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen surface is closed.")
        api = self._api()
        width, height = api.primary_size()
        if (bounds.left < 0 or bounds.top < 0
                or bounds.left + bounds.width > width
                or bounds.top + bounds.height > height):
            raise OverlayError("Screen overlays are restricted to the primary monitor.")
        if self._window_factory is None or self._canvas_factory is None:
            import tkinter as tk
            self._window_factory = self._window_factory or tk.Toplevel
            self._canvas_factory = self._canvas_factory or tk.Canvas
        window = self._window_factory(self.root)
        try:
            window.withdraw()
            window.overrideredirect(True)
            # Do not make this transient(panel): an owned Windows popup must
            # stay above its owner, preventing the control panel from winning.
            window.attributes("-topmost", True)
            window.geometry(f"{bounds.width}x{bounds.height}+{bounds.left}+{bounds.top}")
            background = "black" if input_surface else TRANSPARENT_COLOR
            window.configure(bg=background)
            if input_surface:
                # No color key and never alpha=0: every ROI pixel must receive input.
                window.attributes("-alpha", INPUT_ALPHA)
            else:
                window.attributes("-transparentcolor", TRANSPARENT_COLOR)
            canvas = self._canvas_factory(window, bg=background, highlightthickness=0,
                                          borderwidth=0, cursor="crosshair" if input_surface else "")
            canvas.pack(fill="both", expand=True)
            window.update_idletasks()
            api.configure(window.winfo_id(), click_through=not input_surface)
            self.window, self.canvas = window, canvas
        except Exception:
            window.destroy()
            raise

    def _show(self, *, input_surface):
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen surface is closed.")
        # Tk wm state normal avoids the forced focus performed by deiconify.
        try:
            self.window.state("normal")
            self.window.update_idletasks()
            api = self._api()
            api.configure(self.window.winfo_id(), click_through=not input_surface)
            api.raise_without_activation(self.window.winfo_id())
            keep_control_panel_above(self.control_panel, style_api=api,
                                     platform_name=self._platform_name)
            self.visible = True
        except Exception:
            # A failed native update must not strand a nearly invisible mouse
            # catcher over cryoSPARC while the controller handles the error.
            self.window.withdraw()
            self.visible = False
            raise

    def hide(self):
        self._check_thread()
        if self.window is not None and not self.closed:
            self.window.withdraw()
        self.visible = False

    def close(self):
        self._check_thread()
        if self.closed:
            return
        self.closed = True
        self.visible = False
        if self.window is not None:
            self.window.destroy()
        self.window = self.canvas = None


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """Inside boundary pixels, preserving both disconnected pieces and holes."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape, default=0) == 0:
        raise ValueError("Overlay masks must be nonempty two-dimensional arrays.")
    interior = mask.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        interior[1:-1, 1:-1] &= (mask[:-2, 1:-1] & mask[2:, 1:-1]
                                  & mask[1:-1, :-2] & mask[1:-1, 2:])
    return mask & ~interior


def _mask_image(candidates, geometry, selected_ids=(), preview=None):
    image = np.empty((geometry.height, geometry.width, 3), dtype=np.uint8)
    image[:] = TRANSPARENT_RGB
    selected = set(selected_ids)
    layers = [(candidate, (80, 255, 120) if candidate.id in selected else (40, 210, 95))
              for candidate in candidates]
    if preview is not None:
        layers.append((preview, (255, 215, 70)))
    for candidate, color in layers:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != (geometry.height, geometry.width):
            raise ValueError("Overlay mask dimensions do not match the selected screen area.")
        image[_mask_outline(mask)] = color
    return Image.fromarray(image)


class ScreenOverlay(_ScreenSurface):
    """Lazy, click-through full-primary-screen annotation canvas."""

    def __init__(self, root, *, photo_factory=None, **kwargs):
        super().__init__(root, **kwargs)
        self._photo_factory = photo_factory
        self._mask_photo = None

    def _ensure(self):
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen overlay is closed.")
        if self.window is None:
            width, height = self._api().primary_size()
            self._make_window(CaptureGeometry(0, 0, width, height), input_surface=False)

    def show(self):
        self._ensure()
        self._show(input_surface=False)

    def clear(self):
        self._check_thread()
        if self.canvas is not None:
            self.canvas.delete("all")
        self._mask_photo = None

    def begin_trace(self, screen_xy):
        """Start a manual stroke on a fresh native annotation surface.

        A withdrawn layered window can retain its last painted pixels even
        after Canvas items were deleted while it was hidden. Mapping it again
        before the new Canvas repaint can briefly expose the previous stroke.
        Retiring only that temporary window avoids reusing its backing store;
        the persistent control panel is never hidden or recreated.
        """
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen overlay is closed.")
        # Validate before retiring the old surface so bad input has no effect.
        points = [_screen_xy(*point) for point in screen_xy]
        if self.window is not None:
            self.hide()
            self.clear()
            self.window.destroy()
            self.window = self.canvas = None
        self.draw_trace(points)

    def draw_trace(self, screen_xy):
        self._ensure()
        points = [_screen_xy(*point) for point in screen_xy]
        self.canvas.delete("trace")
        if len(points) > 1:
            self.canvas.create_line(*[value for point in points for value in point],
                                    fill="#ff3838", width=2, tags="trace")

    def draw_region(self, geometry: CaptureGeometry):
        self._ensure()
        self.canvas.delete("region")
        self.canvas.create_rectangle(geometry.left, geometry.top,
                                     geometry.left + geometry.width - 1,
                                     geometry.top + geometry.height - 1,
                                     outline="#63dbe9", width=2, tags="region")

    def draw_masks(self, candidates, geometry: CaptureGeometry, selected_ids=(), preview=None):
        self._ensure()
        bitmap = _mask_image(candidates, geometry, selected_ids, preview)
        if self._photo_factory is None:
            from PIL import ImageTk
            self._photo_factory = ImageTk.PhotoImage
        photo = self._photo_factory(bitmap, master=self.window)
        self.canvas.delete("masks")
        self.canvas.create_image(geometry.left, geometry.top, image=photo, anchor="nw", tags="masks")
        self._mask_photo = photo

    def draw_box(self, screen_box):
        self._ensure()
        self.canvas.delete("box")
        if screen_box is not None:
            x0, y0 = _screen_xy(*screen_box[:2])
            x1, y1 = _screen_xy(*screen_box[2:])
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         outline="#ffe164", width=2, tags="box")

    def draw_points(self, screen_xy, labels):
        self._ensure()
        points, labels = list(screen_xy), list(labels)
        if len(points) != len(labels):
            raise ValueError("Each prompt point needs a foreground/background label.")
        self.canvas.delete("points")
        for point, label in zip(points, labels):
            if label not in (0, 1):
                raise ValueError("Prompt labels must be 0 or 1.")
            x, y = _screen_xy(*point)
            color = "#6aff8b" if label else "#ff6c6c"
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                                    outline=color, width=2, tags="points")
            self.canvas.create_line(x - 3, y, x + 3, y, fill=color, width=2, tags="points")
            if label:
                self.canvas.create_line(x, y - 3, x, y + 3, fill=color, width=2, tags="points")

    def close(self):
        super().close()
        self._mask_photo = None


class InputSurface(_ScreenSurface):
    """Near-transparent mouse catcher; callbacks receive physical screen x/y.

    Press/right-click must begin inside bounds. Moves/releases from a started
    drag are delivered even outside bounds so the controller can finish or
    clamp the gesture explicitly. No screenshot or native input is generated.
    """

    def __init__(self, root, bounds: CaptureGeometry, on_press, on_move, on_release,
                 on_right=None, on_cancel=None, **kwargs):
        super().__init__(root, **kwargs)
        for callback in (on_press, on_move, on_release):
            if not callable(callback):
                raise TypeError("Input gesture callbacks must be callable.")
        self.bounds = bounds
        self._press_callback, self._move_callback = on_press, on_move
        self._release_callback = on_release
        self._right_callback, self._cancel_callback = on_right, on_cancel
        self._dragging = False

    def _ensure(self):
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen input surface is closed.")
        if self.window is not None:
            return
        self._make_window(self.bounds, input_surface=True)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-3>", self._right)
        self.window.bind("<Escape>", self._cancel)

    def show(self):
        self._ensure()
        self._show(input_surface=True)

    def hide(self):
        self._check_thread()
        self._dragging = False
        super().hide()

    def close(self):
        self._check_thread()
        try:
            super().close()
        finally:
            # A contour selection destroys its catcher from inside a release
            # callback. Do not let that retired object keep the controller
            # alive or dispatch a queued cancellation to the next catcher.
            self._dragging = False
            self._press_callback = self._move_callback = self._release_callback = None
            self._right_callback = self._cancel_callback = None

    def _event_xy(self, event):
        # Do not use canvas-local x/y or Tk scaling: coordinates are screen pixels.
        return _screen_xy(event.x_root, event.y_root)

    def _over_panel(self, point):
        # The native z-order normally routes these directly to the panel. This
        # also rejects queued events or an existing mouse grab after it moves.
        api = self._api()
        panel = self.control_panel.winfo_id()
        if not api.visible(panel) or api.minimized(panel):
            return False
        left, top, width, height = control_panel_bounds(
            self.control_panel, style_api=api, platform_name=self._platform_name)
        return left <= point[0] < left + width and top <= point[1] < top + height

    def _cancel_drag_over_panel(self):
        was_dragging, self._dragging = self._dragging, False
        if was_dragging and self._cancel_callback:
            self._cancel_callback()

    def _press(self, event):
        point = self._event_xy(event)
        if self.visible and self.bounds.contains(*point) and not self._over_panel(point):
            self._dragging = True
            self._press_callback(*point)
        return "break"

    def _move(self, event):
        if self.visible and self._dragging:
            point = self._event_xy(event)
            if self._over_panel(point):
                self._cancel_drag_over_panel()
            else:
                self._move_callback(*point)
        return "break"

    def _release(self, event):
        was_dragging, self._dragging = self._dragging, False
        if self.visible and was_dragging:
            point = self._event_xy(event)
            if self._over_panel(point):
                if self._cancel_callback:
                    self._cancel_callback()
            else:
                self._release_callback(*point)
        return "break"

    def _right(self, event):
        point = self._event_xy(event)
        if (self.visible and self.bounds.contains(*point) and self._right_callback
                and not self._over_panel(point)):
            self._right_callback(*point)
        return "break"

    def _cancel(self, event=None):
        self._dragging = False
        if self._cancel_callback:
            self._cancel_callback()
        return "break"


class ManualMouseObserver:
    """Passive global drag observer; callback must only queue its event tuple.

    Physical input continues to cryoSPARC as in the original manual mode.
    Playback-generated input is excluded by both current pynput's ``injected``
    callback argument and the Windows low-level hook flags. This listener never
    suppresses user input and never calls Tk. Stop it while picking is disabled.
    """

    def __init__(self, callback: Callable[[tuple[str, int, int]], None], *,
                 listener_factory=None, button_enum=None, platform_name=None):
        if not callable(callback):
            raise TypeError("Mouse observer callback must be callable.")
        self._callback = callback
        self._factory = listener_factory
        self._buttons = button_enum
        self._platform_name = platform_name or platform.system()
        self._listener = None
        self._active = False
        self._drawing = False
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def active(self):
        with self._lock:
            return self._active

    def start(self):
        if self._platform_name != "Windows":
            raise OverlayError("Global screen drawing is supported on Windows.")
        with self._lock:
            if self._active:
                return
            if self._factory is None or self._buttons is None:
                mouse = importlib.import_module("pynput.mouse")
                self._factory = self._factory or mouse.Listener
                self._buttons = self._buttons or mouse.Button
            self._generation += 1
            generation = self._generation
            self._active, self._drawing = True, False
            try:
                listener = self._factory(
                    on_click=lambda *args: self._click(*args, generation=generation),
                    on_move=lambda *args: self._move(*args, generation=generation),
                    win32_event_filter=lambda msg, data: self._event_filter(msg, data, generation),
                )
                self._listener = listener
                listener.start()
            except Exception:
                self._active, self._drawing = False, False
                self._listener = None
                raise

    def _event_filter(self, message, data, generation):
        with self._lock:
            # Returning False skips observer callbacks; it does not suppress
            # the Windows event. Both normal/lower-integrity injected flags.
            return (self._active and generation == self._generation
                    and not (int(getattr(data, "flags", 0)) & 0x00000003))

    def _click(self, x, y, button, pressed, injected=False, *, generation=None):
        with self._lock:
            if (not self._active or injected or button != self._buttons.left
                    or generation is not None and generation != self._generation):
                return
            if pressed:
                self._drawing = True
                self._emit("press", x, y)
            elif self._drawing:
                self._drawing = False
                self._emit("release", x, y)

    def _move(self, x, y, injected=False, *, generation=None):
        with self._lock:
            if (self._active and self._drawing and not injected
                    and (generation is None or generation == self._generation)):
                self._emit("move", x, y)

    def _emit(self, action, x, y):
        try:
            self._callback((action, *_screen_xy(x, y)))
        except Exception:
            OVERLAY_LOG.exception("The screen mouse event could not be queued.")

    def stop(self):
        with self._lock:
            self._active, self._drawing = False, False
            self._generation += 1
            listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()


def _verify_main_control_panel(root, api) -> dict:
    """Verify the real Tk root against its own full-screen overlay Toplevels.

    A sibling synthetic panel does not establish the owner/root z-order case.
    This phase uses each surface's default control_panel=root and checks the
    real Terminate button and caption without clicking or activating anything.
    """
    root.update()
    handle = root.winfo_id()
    if not api.visible(handle) or api.minimized(handle):
        raise OverlayError("The main control panel must remain visible for native verification.")
    width, height = api.primary_size()
    geometry = CaptureGeometry(0, 0, width, height)
    outer = control_panel_bounds(root, style_api=api)
    client_x, client_y = root.winfo_rootx(), root.winfo_rooty()
    if client_y <= outer[1]:
        raise OverlayError("The main control panel has no native caption to verify.")
    points = [(client_x + root.winfo_width() // 2,
               client_y + root.winfo_height() // 2),
              (client_x + root.winfo_width() // 3, (outer[1] + client_y) // 2)]
    pending = list(root.winfo_children())
    terminate = None
    while pending:
        widget = pending.pop()
        if (widget.winfo_class() in ("Button", "TButton")
                and str(widget.cget("text")).strip().lower() == "terminate"
                and widget.winfo_viewable()):
            terminate = widget
            break
        pending.extend(widget.winfo_children())
    if terminate is None:
        raise OverlayError("The main control panel has no visible Terminate button.")
    points.append((terminate.winfo_rootx() + terminate.winfo_width() // 2,
                   terminate.winfo_rooty() + terminate.winfo_height() // 2))
    if not all(geometry.contains(*point) for point in points):
        raise OverlayError("The main panel's test points lie outside the primary screen.")
    main_handle = api.root_handle(handle)
    foreground = api.foreground()
    events = []
    record_input = lambda x, y: events.append((x, y))
    # Deliberately use the normal constructor defaults here: the real root is
    # the panel, rather than a sibling Toplevel supplied through control_panel.
    overlay = ScreenOverlay(root, style_api=api)
    input_surface = InputSurface(root, geometry, record_input, record_input,
                                 record_input, style_api=api)
    try:
        overlay.draw_box((outer[0], outer[1], outer[0] + outer[2], outer[1] + outer[3]))
        input_surface.show()
        overlay.show()
        root.update()
        if not api.visible(handle) or api.minimized(handle):
            raise OverlayError("The main control panel disappeared beneath its own screen surfaces.")
        if any(not api.is_above(handle, surface.window.winfo_id())
               for surface in (input_surface, overlay)):
            raise OverlayError("A screen surface covers the main Tk control panel.")
        if any(api.window_at(*point) != main_handle for point in points):
            raise OverlayError("A screen surface intercepts the main panel's button, client, or caption.")
        if api.foreground() != foreground:
            raise OverlayError("Showing the main panel's screen surfaces changed foreground focus.")
        input_surface.hide()
        overlay.hide()
        root.update()
        if (not api.visible(handle) or api.minimized(handle)
                or any(api.window_at(*point) != main_handle for point in points)):
            raise OverlayError("Hiding screen surfaces made the main control panel inaccessible.")
        if events:
            raise OverlayError("Unexpected input arrived during main-panel verification.")
        return {"main_root_visible_during_overlay": True,
                "main_root_above_surfaces": True,
                "main_root_button_and_caption_hit_testable": True,
                "main_root_visible_after_hide": True,
                "main_root_stacking_does_not_activate": True}
    finally:
        input_surface.close()
        overlay.close()


def verify_screen_overlay(root) -> dict:
    """Exercise native Windows hit testing on synthetic windows and the real panel.

    No mouse/keyboard events, screenshots, or global listeners are generated.
    The controller should call this only from its deliberate Windows smoke test.
    """
    import tkinter as tk
    from types import SimpleNamespace

    if platform.system() != "Windows":
        raise OverlayError("Native overlay verification requires Windows.")
    api = _OverlayWindowsAPI()
    width, height = api.primary_size()
    if width < 360 or height < 300:
        raise OverlayError("The Windows smoke-test display is too small.")
    # Keep the actual control panel visible throughout the probe. Choose an
    # unused corner for our synthetic target/panel rather than hiding the root.
    occupied = (control_panel_bounds(root, style_api=api)
                if api.visible(root.winfo_id()) and not api.minimized(root.winfo_id()) else None)
    position = None
    for left, top in ((width - 340, 20), (width - 340, height - 280),
                      (20, height - 280), (20, 20)):
        if occupied is not None:
            ox, oy, ow, oh = occupied
            if left < ox + ow and left + 320 > ox and top < oy + oh and top + 260 > oy:
                continue
        position = (left, top)
        break
    if position is None:
        raise OverlayError("No free display area remains for the visible-panel overlay smoke test.")
    left, top = position
    target = tk.Toplevel(root)
    panel = overlay = input_surface = None
    events = []
    try:
        target.withdraw()
        target.overrideredirect(True)
        target.attributes("-topmost", True)
        target.geometry(f"320x260+{left}+{top}")
        target.configure(bg="#445566")
        target.state("normal")
        target.update_idletasks()
        api.raise_without_activation(target.winfo_id())
        root.update()
        target_handle = api.root_handle(target.winfo_id())
        geometry = CaptureGeometry(left + 20, top + 20, 280, 220)
        panel = tk.Toplevel(root)
        panel.withdraw()
        panel.title("SARU panel smoke test")
        panel.geometry(f"100x70+{left + 140}+{top + 70}")
        button = tk.Button(panel, text="Terminate", command=lambda: events.append("terminate"))
        button.pack(fill="both", expand=True, padx=4, pady=4)
        panel.state("normal")
        panel.update_idletasks()
        keep_control_panel_above(panel, style_api=api)
        root.update()
        panel_handle = api.root_handle(panel.winfo_id())

        def panel_points():
            outer = control_panel_bounds(panel, style_api=api)
            button_center = (button.winfo_rootx() + button.winfo_width() // 2,
                             button.winfo_rooty() + button.winfo_height() // 2)
            caption = (outer[0] + 12, outer[1] + 10)
            if not geometry.contains(*button_center) or not geometry.contains(*caption):
                raise OverlayError("The synthetic panel does not overlap the input test region.")
            return button_center, caption

        def check_panel_above(*surfaces):
            if not api.visible(panel.winfo_id()) or api.minimized(panel.winfo_id()):
                raise OverlayError("The control panel disappeared while a screen surface was shown.")
            for surface in surfaces:
                if not api.is_above(panel.winfo_id(), surface.window.winfo_id()):
                    raise OverlayError("A screen surface covers the control panel.")
            if any(api.window_at(*point) != panel_handle for point in panel_points()):
                raise OverlayError("A screen surface intercepts the panel's button or caption.")

        mask = np.zeros((geometry.height, geometry.width), dtype=bool)
        mask[25:180, 25:220] = True
        mask[55:80, 45:65] = False
        candidate = SimpleNamespace(id="synthetic-mask", mask=mask)
        overlay = ScreenOverlay(root, style_api=api, control_panel=panel)
        overlay.draw_masks([candidate], geometry, {candidate.id})
        overlay.draw_trace([(left + 40, top + 35), (left + 100, top + 35)])
        foreground = api.foreground()
        overlay.show()
        root.update()
        check_panel_above(overlay)
        styles = api.styles(overlay.window.winfo_id())
        required = api.WS_EX_LAYERED | api.WS_EX_TRANSPARENT | api.WS_EX_NOACTIVATE
        if styles & required != required:
            raise OverlayError("The visual overlay lacks its native passive styles.")
        # Include a rendered red line, a rendered green mask edge, and a hole.
        points = [(left + 80, top + 35), (left + 45, top + 80),
                  (left + 74, top + 88)]
        if any(api.window_at(*point) != target_handle for point in points):
            raise OverlayError("A visual annotation intercepted input intended for the target.")
        record_input = lambda x, y: events.append((x, y))
        input_surface = InputSurface(root, geometry, record_input,
                                     record_input, record_input, style_api=api,
                                     control_panel=panel)
        input_surface.show()
        overlay.show()  # Put visible annotations above the input catcher too.
        root.update()
        check_panel_above(overlay, input_surface)
        input_handle = api.root_handle(input_surface.window.winfo_id())
        if any(api.window_at(*point) != input_handle for point in points):
            raise OverlayError("The ROI input surface does not catch mask-interior input.")
        input_style = api.styles(input_surface.window.winfo_id())
        if input_style & api.WS_EX_TRANSPARENT or not input_style & api.WS_EX_NOACTIVATE:
            raise OverlayError("The ROI input surface has incorrect native window styles.")
        if api.window_at(left + 5, top + 5) != target_handle:
            raise OverlayError("The ROI input surface extends beyond its selected bounds.")
        original_panel_bounds = control_panel_bounds(panel, style_api=api)
        panel.geometry(f"100x70+{left + 160}+{top + 85}")
        root.update()
        input_surface.show()
        overlay.show()
        root.update()
        check_panel_above(overlay, input_surface)
        if original_panel_bounds == control_panel_bounds(panel, style_api=api):
            raise OverlayError("The synthetic control panel did not move during verification.")
        if api.foreground() != foreground:
            raise OverlayError("Showing a screen surface activated a window unexpectedly.")
        input_surface.hide()
        overlay.hide()
        root.update()
        check_panel_above()
        if any(api.window_at(*point) != target_handle for point in points):
            raise OverlayError("Hiding the screen surfaces did not restore the target window.")
        if events:
            raise OverlayError("Unexpected input arrived during the no-input overlay smoke test.")
        # Remove only our synthetic windows before testing the real main root.
        # Its own Toplevels may have different native owner/root relationships
        # from the sibling synthetic panel used in the first phase.
        panel.withdraw()
        target.withdraw()
        root.update()
        main_panel_checks = _verify_main_control_panel(root, api)
        return {"annotation_hit_transparent": True, "roi_input_intercepts": True,
                "outside_roi_passes": True, "hidden_restores_target": True,
                "panel_visible_during_overlay": True, "panel_above_surfaces": True,
                "panel_button_and_caption_hit_testable": True,
                "panel_move_supported": True, "panel_visible_after_hide": True,
                "stacking_does_not_activate": True,
                "desktop_events_sent": 0, "global_listeners_started": 0,
                **main_panel_checks}
    finally:
        if input_surface is not None:
            input_surface.close()
        if overlay is not None:
            overlay.close()
        if panel is not None:
            panel.destroy()
        target.destroy()
