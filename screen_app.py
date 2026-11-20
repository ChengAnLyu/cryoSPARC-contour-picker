"""Compact live-screen controller; cryoSPARC remains the image viewer.

All Tk work runs on the main thread. Global listeners enqueue input, and one
worker performs recognition and cancellable click playback. Manual contours
retain the original direct drag/release workflow and lattice placement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np

from application_icon import apply_application_icon
from desktop import DesktopAdapter, GlobalHotkeys
from execution import CancellationToken, CancelledOperation, CaptureGeometry, TaskRunner, play_clicks
from manual_sampling import legacy_contour_points, mask_contour_points
from micrograph_navigation import NoNewMicrographError, wait_for_next_micrograph
from recognition import (Cancelled, RecognitionConfig, SamRecognizer, union_masks,
                         DEFAULT_MIN_CAPSID_AREA_PX)
from screen_overlay import (ScreenOverlay, InputSurface, ManualMouseObserver,
                            keep_control_panel_above, control_panel_bounds)


SCREEN_LOG = logging.getLogger("saru.screen")
DEFAULT_CLICK_DELAY = .01


def screen_changed_fraction(before, after, tolerance=12):
    """Compare physical screen pixels without unsigned subtraction overflow."""
    if before is None or after is None or before.shape != after.shape:
        return 1.0
    difference = np.abs(before.astype(np.int16) - after.astype(np.int16))
    return float(np.mean(np.max(difference, axis=2) > tolerance))


def screen_mask_at(candidates, geometry, x, y):
    """Hit-test actual mask interiors, including holes and disconnected parts."""
    if geometry is None or not geometry.contains(x, y):
        return None
    px, py = int(x - geometry.left), int(y - geometry.top)
    # Prefer a smaller nested region when several proposals overlap the click.
    for candidate in sorted(candidates, key=lambda item: item.area):
        if candidate.mask.shape != (geometry.height, geometry.width):
            raise ValueError("A displayed mask no longer matches the selected screen area.")
        if candidate.mask[py, px]:
            return candidate
    return None


@dataclass(frozen=True)
class ScreenContourBatch:
    candidate_id: str
    local: np.ndarray
    screen: np.ndarray


def _screen_picked_points(picked, geometry):
    """Validate actual ROI-local pick positions before building native arrays."""
    result = set()
    for point in picked:
        try:
            x, y = point
            if any(isinstance(value, (bool, np.bool_)) for value in (x, y)):
                raise ValueError
            x, y = float(x), float(y)
            if (not math.isfinite(x) or not math.isfinite(y)
                    or not x.is_integer() or not y.is_integer()
                    or not 0 <= x < geometry.width or not 0 <= y < geometry.height):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Previously placed picks must be integer positions inside the selected screen area.") from exc
        result.add((int(x), int(y)))
    return result


def screen_model_batches(candidates, geometry, spacing, picked=(), cancel=None,
                         max_points=10000):
    """Prepare every contour before clicking, retaining detection order.

    Each contour owns its grid origin. Sites too close to earlier actual or
    planned picks are removed without moving any retained site or changing
    comparisons within the contour. The threshold is the minimum neighboring
    distance in the manual integer lattice. Empty batches still represent a
    processed contour, and the point limit applies after filtering.
    """
    from scipy.spatial import cKDTree
    if (isinstance(max_points, (bool, np.bool_))
            or not isinstance(max_points, (int, np.integer)) or max_points < 1):
        raise ValueError("max_points must be a positive integer.")
    already = _screen_picked_points(picked, geometry)
    batches, count = [], 0
    for candidate in candidates:
        if candidate.mask.shape != (geometry.height, geometry.width):
            raise ValueError("A detected contour no longer matches the selected screen area.")
        # The sampler separately bounds lattice work. Raw sites may exceed
        # the click budget when most are removed by existing picks, so enforce
        # that budget only after proximity filtering.
        positions = mask_contour_points(candidate.mask, spacing,
                                        max_points=int(candidate.mask.size), cancel=cancel)
        if already and len(positions):
            row_height = int(int(spacing) * math.sqrt(3) / 2)
            minimum = min(float(spacing), math.hypot(int(spacing)//2, row_height))
            threshold = minimum - 1e-7 * max(1.0, minimum)
            previous = cKDTree(np.asarray(tuple(already), dtype=np.float64))
            parts, accepted = [], 0
            for start in range(0, len(positions), 4096):
                if cancel is not None and cancel():
                    raise CancelledOperation("Detected contour planning cancelled.")
                chunk = positions[start:start+4096]
                distances, _ = previous.query(chunk, k=1, distance_upper_bound=minimum)
                part = chunk[distances >= threshold]
                accepted += len(part)
                if count + accepted > max_points:
                    raise ValueError(f"More than {max_points:,} pick positions were requested across the contours; increase spacing or select fewer contours.")
                parts.append(part)
            local = np.concatenate(parts)
        else:
            local = positions
        count += len(local)
        if count > max_points:
            raise ValueError(f"More than {max_points:,} pick positions were requested across the contours; increase spacing or select fewer contours.")
        already.update(tuple(map(int, point)) for point in local)
        screen = np.asarray([geometry.local_to_screen(int(x), int(y)) for x, y in local],
                            dtype=np.int32).reshape(-1, 2)
        batches.append(ScreenContourBatch(candidate.id, local, screen))
    return batches


def screen_model_points(candidates, geometry, spacing, picked=(), cancel=None):
    """Return the ordered contour plans as one array for previews and callers."""
    batches = screen_model_batches(candidates, geometry, spacing, picked, cancel)
    if not batches:
        return np.empty((0, 2), np.int32), np.empty((0, 2), np.int32)
    return np.concatenate([batch.local for batch in batches]), np.concatenate([batch.screen for batch in batches])


def screen_background_guard(candidates, geometry, spacing, picked=()):
    """Pixels outside selected regions and expected pick-marker margins.

    This is a conservative scene-change heuristic, not image registration.
    Earlier actual picks are also excluded: cryoSPARC may finish painting their
    markers after the first contour has completed. Their delayed appearance
    must not be mistaken for a moved micrograph on the next contour.
    A tiny or empty remainder cannot establish that old proposals still align.
    """
    from scipy.ndimage import distance_transform_edt
    selected = union_masks(candidates, (geometry.height, geometry.width))
    for x, y in _screen_picked_points(picked, geometry):
        selected[y, x] = True
    if not np.any(selected):
        return np.ones(selected.shape, dtype=bool)
    background = distance_transform_edt(~selected) > max(8, int(spacing))
    if np.count_nonzero(background) < max(64, int(.02 * background.size)):
        return None
    return background


def screen_background_changed(before, after, background):
    if before.shape != after.shape or background.shape != before.shape[:2]:
        return 1.0
    difference = np.max(np.abs(before.astype(np.int16)-after.astype(np.int16)), axis=2)
    return float(np.mean(difference[background] > 12))


def screen_region_overlaps_panel(region, bounds):
    """Check the visible panel's outer rectangle without touching Tk."""
    if bounds is None:
        return False
    left, top, width, height = bounds
    return (region.left < left + width and left < region.left + region.width
            and region.top < top + height and top < region.top + region.height)


class ScreenPickerApp:
    """Default screen UI. ``mode`` is str; ``disabled`` is a Tk BooleanVar.

    ``start_listeners=False`` constructs a passive panel for packaging checks;
    it never starts global hooks or sends desktop input. The overlay is lazy.
    """

    def __init__(self, root, args, log_path, checkpoint_path, *, start_listeners=True):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root, self.args, self.log_path = root, args, Path(log_path)
        apply_application_icon(root)
        self.desktop, self.runner = DesktopAdapter(), TaskRunner()
        self.events = queue.Queue()
        self.overlay = self.input_surface = self.advanced = None
        self._picking_menu = None
        self.mouse_observer = self.hotkeys = None
        self._listeners_requested = bool(start_listeners)
        self._manual_gate = threading.Event()
        self._input_lock = threading.Lock()
        self._emergency_stop = threading.Event()
        self._mouse_epoch = 0
        self.mode = "manual"
        self.closed = False
        self.scheduled = set()
        self.transition = False
        self._fit_after = None
        self._poll_after = None
        self._shutdown_after = None
        self._variable_traces = []
        self._fitting = False
        self.job_kind = None
        self.active_token = None
        self.paused = False
        self.manual_window = None
        self.trace = []
        self.trace_dirty = False
        self.drawing = False
        self.released = False
        self.manual_geometry = None
        self.panel_bounds = None
        self.geometry = self.target = self.rgb = self.reference = None
        self.candidates, self.alternatives = [], []
        self.alternative_index = 0
        self.picked_points = set()
        self.frame_id = 0
        self.auto_remaining = 0
        self.auto_completed = 0
        self.auto_history = []
        self.recognizer = self.recognizer_config = None
        self.input_kind = None
        self.input_start = None
        self._build_ui(checkpoint_path)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.stop())
        root.bind("<Configure>", lambda event: self._remember_panel())
        # These shortcuts belong to the control panel, never a global hook.
        root.bind("<F9>", self._open_picking_speed)
        root.bind("<Button-3>", self._show_picking_menu)
        if start_listeners:
            try:
                self.hotkeys = GlobalHotkeys(self._hotkey, space_pause=True)
                self.hotkeys.start()
                self.mouse_observer = ManualMouseObserver(self._mouse_event)
                self.mouse_observer.start()
            except Exception as exc:
                SCREEN_LOG.exception("Screen input hooks could not start")
                if self.hotkeys is not None:
                    self.hotkeys.stop()
                if self.mouse_observer is not None:
                    self.mouse_observer.stop()
                self.hotkeys = self.mouse_observer = None
                self.status.set(f"Screen input unavailable: {exc}")
        self._poll_after = self.root.after(30, self.poll)

    def _build_ui(self, checkpoint_path):
        tk, ttk, root = self.tk, self.ttk, self.root
        root.title("SARU PICKER")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        scale = max(1.0, float(root.tk.call("tk", "scaling")) / (96 / 72))
        self._panel_minimum = (round(420 * scale), round(320 * scale))
        wrap = round(395 * scale)
        # Reserve the footer first. Its button must never receive the few
        # leftover pixels of a fixed-height pack layout.
        self.panel_footer = tk.Frame(root)
        self.panel_footer.pack(side="bottom", fill="x")
        self.terminate_button = tk.Button(self.panel_footer, text="terminate", width=10,
                                         font=("Arial", 10), command=self.close)
        self.terminate_button.pack(pady=10)
        self.panel_content = tk.Frame(root)
        self.panel_content.pack(side="top", fill="both", expand=True)
        content = self.panel_content
        tk.Label(content, text="Mode", font=("Arial", 10)).pack(pady=5)
        self.mode_var = tk.StringVar(value="manual")
        modes = tk.Frame(content)
        modes.pack(pady=5)
        for mode in ("manual", "semi-auto", "auto"):
            tk.Radiobutton(modes, text=mode, value=mode, variable=self.mode_var,
                           indicatoron=False, width=10, font=("Arial", 10),
                           command=lambda value=mode: self.set_mode(value)).pack(side="left", padx=5)
        self.spacing = tk.IntVar(value=13)
        self.spacing_slider = tk.Scale(content, from_=5, to=25, orient="horizontal",
                                       label="Particle spacing", length=round(150 * scale),
                                       variable=self.spacing)
        self.spacing_slider.pack(padx=10, pady=10)
        self.disabled = tk.BooleanVar(value=True)
        tk.Checkbutton(content, text="Disable Picking", variable=self.disabled,
                       command=self.disabled_changed).pack(pady=5)
        self.status = tk.StringVar(value="Manual Mode: Hold left mouse button to draw a shape.")
        self.status_label = tk.Label(content, textvariable=self.status,
                                    font=("Arial", 8), wraplength=wrap)
        self.status_label.pack(padx=10, pady=5)
        self.model_controls = ttk.Frame(content, padding=(12, 0, 12, 0))
        row = ttk.Frame(self.model_controls)
        row.pack(fill="x")
        ttk.Button(row, text="Select screen area", command=self.select_region).pack(side="left")
        ttk.Button(row, text="Detect", command=self.detect).pack(side="left", padx=5)
        ttk.Button(row, text="Advanced…", command=self.open_advanced).pack(side="right")
        self.region_label = tk.StringVar(value="Select the micrograph area on the screen.")
        ttk.Label(self.model_controls, textvariable=self.region_label, wraplength=wrap).pack(anchor="w", pady=4)
        self.semi_controls = ttk.Frame(self.model_controls)
        ttk.Button(self.semi_controls, text="Box prompt", command=self.box_prompt).pack(side="left")
        self.next_micrograph_button = ttk.Button(self.semi_controls, text="Next micrograph", command=self.next_micrograph)
        self.next_micrograph_button.pack(side="left", padx=5)
        self.auto_controls = ttk.Frame(self.model_controls)
        self.frames, self.wait_seconds = tk.IntVar(value=0), tk.DoubleVar(value=10)
        ttk.Label(self.auto_controls, text="Limit (0=all)").pack(side="left")
        ttk.Spinbox(self.auto_controls, from_=0, to=1000, textvariable=self.frames, width=4).pack(side="left", padx=3)
        ttk.Label(self.auto_controls, text="Load (s)").pack(side="left")
        ttk.Spinbox(self.auto_controls, from_=1, to=60, textvariable=self.wait_seconds, width=4).pack(side="left", padx=3)
        ttk.Button(self.auto_controls, text="Start auto", command=self.start_auto).pack(side="right")
        self.model_note = ttk.Label(self.model_controls, wraplength=wrap)
        self.model_note.pack(anchor="w", pady=5)
        self.checkpoint = tk.StringVar(value=str(checkpoint_path))
        self.model_type = tk.StringVar(value=getattr(self.args, "model_type", "vit_b"))
        self.device = tk.StringVar(value=getattr(self.args, "device", "auto"))
        self.automatic_profile = tk.StringVar(value=getattr(self.args, "automatic_profile", "capsid"))
        self.min_capsid_area = tk.StringVar(value=str(getattr(self.args, "min_capsid_area", DEFAULT_MIN_CAPSID_AREA_PX)))
        self.grid_side = tk.IntVar(value=8)
        self.click_delay = tk.DoubleVar(value=DEFAULT_CLICK_DELAY)
        for variable in (self.status, self.region_label):
            identifier = variable.trace_add("write", self._queue_panel_fit)
            self._variable_traces.append((variable, identifier))
        self.fit_panel()

    def _queue_panel_fit(self, *args):
        if not self.closed and self._fit_after is None:
            self._fit_after = self.root.after_idle(self.fit_panel)

    def fit_panel(self):
        """Fit actual widget requests, including scaled fonts and long status."""
        if self.closed or self._fitting:
            return
        if self._fit_after is not None:
            self.root.after_cancel(self._fit_after)
            self._fit_after = None
        self._fitting = True
        try:
            self.root.update_idletasks()
            width = max(self._panel_minimum[0], self.root.winfo_reqwidth())
            height = max(self._panel_minimum[1], self.root.winfo_reqheight())
            self.root.minsize(width, height)
            if (self.root.winfo_width(), self.root.winfo_height()) != (width, height):
                self.root.geometry(f"{width}x{height}")
                self.root.update_idletasks()
            self._remember_panel()
        finally:
            self._fitting = False

    def ensure_overlay(self):
        if self.overlay is None:
            self.overlay = ScreenOverlay(self.root)
        return self.overlay

    def _remember_panel(self):
        if self.closed:
            return
        try:
            self.panel_bounds = control_panel_bounds(self.root)
        except self.tk.TclError:
            pass

    def _check_panel_clear(self, region):
        if screen_region_overlaps_panel(region, self.panel_bounds):
            raise ValueError("Move the SARU panel outside the selected micrograph area, then try again.")

    def _capture_region(self, region):
        # panel_bounds is an immutable snapshot updated on the Tk thread.
        # Workers can reject occlusion without accessing any Tk widget.
        self._check_panel_clear(region)
        return self.desktop.screenshot(region)

    def _in_panel(self, x, y):
        if self.panel_bounds is None:
            self._remember_panel()
        if self.panel_bounds is None:
            return False
        left, top, width, height = self.panel_bounds
        return left <= x < left + width and top <= y < top + height

    def _mouse_event(self, record):
        # pynput thread: no Tk calls and no mutable trace access.
        with self._input_lock:
            enabled, epoch = self._manual_gate.is_set(), self._mouse_epoch
        if enabled:
            kind, x, y = record
            self.events.put(("mouse", epoch, kind, int(x), int(y)))

    def _hotkey(self, action):
        if action == "stop":
            self._emergency_stop.set()
            self._gate_manual()
            if self.active_token is not None:
                self.active_token.cancel()
        self.events.put(("hotkey", action))

    def _gate_manual(self):
        with self._input_lock:
            self._manual_gate.clear()
            self._mouse_epoch += 1

    def _sync_manual(self):
        enabled = (self.mode == "manual" and not self.disabled.get() and not self.closed
                   and self.mouse_observer is not None and self.hotkeys is not None
                   and self.advanced is None and not self.released
                   and not self._emergency_stop.is_set()
                   and (self.drawing or (not self.transition and not self.runner.busy)))
        with self._input_lock:
            if enabled:
                self._manual_gate.set()
            else:
                self._manual_gate.clear()

    def _schedule(self, milliseconds, function):
        generation = self.runner.generation
        holder = []
        def invoke():
            self.scheduled.discard(holder[0])
            if not self.closed and generation == self.runner.generation and not self._emergency_stop.is_set():
                try:
                    function()
                except Exception as exc:
                    self._error(exc)
        holder.append(self.root.after(milliseconds, invoke))
        self.scheduled.add(holder[0])

    def _close_input(self):
        if self.input_surface is not None:
            self.input_surface.close()
            self.input_surface = None
        self.input_kind, self.input_start = None, None

    def _hide_overlays(self):
        """Clear temporary screen layers; the controls always remain visible."""
        self._gate_manual()
        self._remember_panel()
        self._close_input()
        if self.overlay is not None:
            self.overlay.hide()
        self._show_panel()

    def _show_panel(self):
        # Raising is nonactivating: the browser keeps keyboard focus while the
        # panel remains visible, interactive and above our full-screen layers.
        keep_control_panel_above(self.root)
        if self.advanced is not None:
            keep_control_panel_above(self.advanced)

    def _idle(self):
        if self.advanced is not None:
            raise RuntimeError("Close settings before starting a screen operation.")
        if self.runner.busy or self.transition or self.auto_remaining or self.drawing:
            raise RuntimeError("An operation is active. Press Escape to stop it first.")
        if self.input_surface is not None and self.input_kind != "masks":
            raise RuntimeError("Finish or cancel the screen selection first.")

    def _live_ready(self):
        if self.disabled.get():
            raise ValueError("Uncheck Disable Picking to send picks to cryoSPARC.")
        if self.hotkeys is None or self.mouse_observer is None:
            raise RuntimeError("Global input hooks are unavailable. Restart the Windows application.")
        delay = float(self.click_delay.get())
        if not 0 <= delay <= 2:
            raise ValueError("Click delay must be between 0 and 2 seconds.")
        return delay

    def _error(self, error):
        SCREEN_LOG.error("Screen operation stopped: %s", error)
        self.stop(message=str(error))
        self.candidates, self.alternatives = [], []
        self.reference = None

    def set_mode(self, mode):
        if mode not in {"manual", "semi-auto", "auto"}:
            return
        self.stop()
        self.mode = mode
        self.mode_var.set(mode)
        self.candidates, self.alternatives = [], []
        self.reference = self.rgb = None
        self.picked_points.clear()
        self.manual_window = None
        self.semi_controls.pack_forget()
        self.auto_controls.pack_forget()
        if mode == "manual":
            self.model_controls.pack_forget()
            self.status.set("Manual Mode: Hold left mouse button to draw a shape.")
        else:
            self.model_controls.pack(fill="x")
            if mode == "semi-auto":
                self.semi_controls.pack(fill="x", before=self.model_note, pady=3)
                self.model_note.configure(text="Click a green region to pick it. Next micrograph moves Down and detects again.")
            else:
                self.auto_controls.pack(fill="x", before=self.model_note, pady=3)
                self.model_note.configure(text="Auto picks every contour. Limit 0 repeats; Load is the maximum wait. Space pauses; Escape stops.")
            self.status.set("Select the screen area, then detect or start auto.")
        self.fit_panel()

    def disabled_changed(self):
        if self.disabled.get():
            self.stop()
            return
        try:
            self._live_ready()
            if self.mode == "manual":
                self.manual_window = None
                self.status.set("Manual Mode: Hold left mouse button to draw a shape.")
                self._sync_manual()
            elif self.mode == "semi-auto":
                self._present_masks()
            else:
                self._show_auto_ready()
        except Exception as exc:
            self._error(exc)

    def _show_auto_ready(self):
        if (self.closed or self.mode != "auto" or self.disabled.get()
                or self.auto_remaining or self.transition or self.advanced is not None):
            return
        if self.runner.busy:
            # A cancelled native model call may still be returning. Keep its
            # captures free of outlines until the worker has been collected.
            self.status.set("Waiting for the previous operation to stop. Stored contours will return when ready.")
            self._schedule(100, self._show_auto_ready)
            return
        if self.input_surface is not None and self.input_kind != "masks":
            return
        self._present_masks()
        if self.candidates:
            count = len(self.candidates)
            self.status.set(f"Auto ready. {count} detected contour{'s' if count != 1 else ''} shown. Choose Start auto.")
        else:
            self.status.set("Auto ready. Select a screen area, then choose Start auto.")

    def _manual_input(self, epoch, kind, x, y):
        if (epoch != self._mouse_epoch or self.mode != "manual" or self.disabled.get()
                or not self._manual_gate.is_set()):
            return
        if kind == "press":
            if self.drawing or self.runner.busy or self.transition or self._in_panel(x, y):
                return
            try:
                target = self.desktop.target_at(x, y)
                if self.desktop.is_own_window(target):
                    return
                # A stroke selects its own target. The initial physical click
                # may still be activating the browser when the hook fires.
                self.manual_window = target
                width, height = self.desktop.size()
                self.manual_geometry = CaptureGeometry(0, 0, width, height)
                self.trace, self.drawing, self.released = [(x, y)], True, False
                self.trace_dirty = True
                self.ensure_overlay().begin_trace(self.trace)
                self.overlay.show()
                self._show_panel()
            except Exception as exc:
                self._error(exc)
        elif self.drawing and kind in {"move", "release"}:
            try:
                if self._in_panel(x, y) or not self.manual_geometry.contains(x, y):
                    self._cancel_trace()
                    return
                if kind == "move":
                    if self.trace[-1] != (x, y):
                        self.trace.append((x, y))
                    if len(self.trace) > 50000:
                        raise ValueError("This trace is too long. Draw a smaller contour.")
                    self.trace_dirty = True
                else:
                    self.released = True
                    self._gate_manual()
                    self._finish_manual()
            except Exception as exc:
                self._error(exc)

    def _cancel_trace(self):
        self._gate_manual()
        self.runner.invalidate()
        self.transition = False
        self.trace, self.drawing, self.released = [], False, False
        self.trace_dirty = False
        self.manual_window = None
        if self.overlay is not None:
            self.overlay.clear()
            self.overlay.hide()
        self._show_panel()
        self.status.set("Trace cancelled outside the target. Draw the next shape.")

    def _finish_manual(self):
        self._gate_manual()
        points = tuple(self.trace)
        target, spacing = self.manual_window, int(self.spacing.get())
        delay = self._live_ready()
        self.trace, self.drawing, self.released = [], False, False
        self.trace_dirty = False
        self.manual_window = None
        self._hide_overlays()
        if len(points) < 3:
            self._show_panel()
            if self.overlay is not None:
                self.overlay.clear()
                self.overlay.hide()
            self._sync_manual()
            return
        def work(token):
            picks = legacy_contour_points(points, spacing, cancel=lambda: token.cancelled)
            if not len(picks):
                return {"count": 0}
            vertices = np.asarray(points, dtype=np.int64)
            left, top = vertices.min(axis=0)
            right, bottom = vertices.max(axis=0)
            region = CaptureGeometry(int(left), int(top), int(right-left+1), int(bottom-top+1))
            token.wait_until_ready()
            # The physical stroke already selected the browser. Never raise
            # it over the persistent panel or steal focus from another app.
            self.desktop.bind_target(target)
            count = play_clicks(picks, self.desktop.click, region, token, delay=delay, max_clicks=10000,
                                progress=lambda done, total: self.events.put(("progress", token.generation, done, total)))
            return {"count": count}
        self._submit("manual picks", work)

    def select_region(self):
        try:
            self._idle()
            if self.mode == "manual":
                return
            width, height = self.desktop.size()
            self._hide_overlays()
            bounds = CaptureGeometry(0, 0, width, height)
            self.input_kind = "region"
            self.ensure_overlay().begin_selection()
            self.overlay.show()
            self.input_surface = InputSurface(self.root, bounds, self._input_press,
                                              self._input_move, self._input_release,
                                              on_cancel=self.stop)
            self.input_surface.show()
        except Exception as exc:
            self._error(exc)

    def _input_press(self, x, y):
        self.input_start = None if self._in_panel(x, y) else (x, y)

    def _input_move(self, x, y):
        if self.input_start is None or self.input_kind == "masks":
            return
        left, right = sorted((self.input_start[0], x))
        top, bottom = sorted((self.input_start[1], y))
        if right > left and bottom > top:
            if self.input_kind == "region":
                self.ensure_overlay().draw_region(CaptureGeometry(int(left), int(top), int(right-left), int(bottom-top)))
            else:
                self.ensure_overlay().draw_box((left, top, right, bottom))

    def _input_release(self, x, y):
        if self.input_start is None:
            return
        if self._in_panel(x, y):
            self.input_start = None
            return
        start, kind = self.input_start, self.input_kind
        if kind == "masks":
            self.input_start = None
            # Complete the physical press/release pair while the catcher still
            # owns it. A drag is not a mask selection.
            if (x-start[0])**2 + (y-start[1])**2 <= 36:
                first = screen_mask_at(self.candidates, self.geometry, *start)
                last = screen_mask_at(self.candidates, self.geometry, x, y)
                if first is not None and last is not None and first.id == last.id:
                    self._semi_click(x, y)
            return
        bounds = self.geometry if kind == "box" else CaptureGeometry(0, 0, *self.desktop.size())
        left, right = sorted((max(bounds.left, min(bounds.left+bounds.width-1, int(start[0]))),
                              max(bounds.left, min(bounds.left+bounds.width-1, int(x)))))
        top, bottom = sorted((max(bounds.top, min(bounds.top+bounds.height-1, int(start[1]))),
                              max(bounds.top, min(bounds.top+bounds.height-1, int(y)))))
        if right-left < (4 if kind == "box" else 32) or bottom-top < (4 if kind == "box" else 32):
            return
        self._hide_overlays()
        if kind == "box":
            box = (left-bounds.left, top-bounds.top, right-bounds.left, bottom-bounds.top)
            self._start_detection(box=box)
            return
        region = CaptureGeometry(left, top, right-left+1, bottom-top+1)
        self.transition = True
        def complete():
            self.transition = False
            self._remember_panel()
            self._check_panel_clear(region)
            target = self.desktop.window_under_roi(region)
            self.geometry, self.target = region, target
            self.rgb = self.reference = None
            self.candidates, self.alternatives = [], []
            self.picked_points.clear()
            self.region_label.set(f"Screen area: {region.width} × {region.height} px at {region.left}, {region.top}")
            self._show_panel()
            self.status.set("Screen area selected. Choose Detect or draw a Box prompt.")
        self._schedule(90, complete)

    def _model_config(self):
        return RecognitionConfig(checkpoint=self.checkpoint.get(), model_type=self.model_type.get(),
                                 device=self.device.get(), points_per_side=int(self.grid_side.get()),
                                 automatic_profile=self.automatic_profile.get(),
                                 min_capsid_area_px=int(self.min_capsid_area.get()))

    def _apply_model_settings(self, previous_config):
        """Validate settings and retire contours made with an older detector.

        Keep actual pick history and its scene reference: changing a size limit
        must not make a repeated detection place duplicate picks on this image.
        """
        current = self._model_config()
        if current == previous_config:
            return False
        self._close_input()
        self.candidates, self.alternatives = [], []
        self.alternative_index = 0
        self.frame_id += 1
        if self.overlay is not None:
            self.overlay.hide()
            self.overlay.clear()
        self.status.set("Detection settings changed. Choose Detect to refresh the contours.")
        return True

    def _get_recognizer(self, config):
        if self.recognizer is None or self.recognizer_config != config:
            if self.recognizer is not None:
                self.recognizer.close()
            self.recognizer = SamRecognizer(config)
            self.recognizer_config = config
        return self.recognizer

    def detect(self):
        try:
            self._idle()
            self._start_detection()
        except Exception as exc:
            self._error(exc)

    def _start_detection(self, *, auto=False, box=None, expected=None):
        if self.geometry is None or self.target is None:
            raise ValueError("Select the micrograph screen area first.")
        config, geometry, target = self._model_config(), self.geometry, self.target
        self._hide_overlays()
        self._check_panel_clear(geometry)
        self.transition = True
        def begin():
            self.transition = False
            def work(token):
                self.desktop.focus_target(target)
                token.sleep(.12)
                if self.desktop.window_under_roi(geometry) != target:
                    raise RuntimeError("Another window covers the selected screen area.")
                rgb = self._capture_region(geometry)
                if expected is not None and screen_changed_fraction(expected, rgb) > .01:
                    raise RuntimeError("The micrograph changed after navigation. Detect again before picking.")
                model = self._get_recognizer(config)
                candidates = (model.prompt(rgb, box=box, cancel=lambda: token.cancelled) if box is not None
                              else model.automatic(rgb, cancel=lambda: token.cancelled))
                token.wait_until_ready()
                self.desktop.ensure_target(target)
                if screen_changed_fraction(rgb, self._capture_region(geometry)) > .03:
                    raise RuntimeError("The micrograph moved during recognition. Detect again.")
                return {"rgb": rgb, "candidates": candidates, "auto": auto, "prompt": box is not None}
            self._submit("recognition", work)
        self._schedule(90, begin)

    def _present_masks(self):
        self._show_panel()
        self._close_input()
        if self.mode == "manual" or self.geometry is None or not self.candidates:
            if self.overlay is not None:
                self.overlay.hide()
                self.overlay.clear()
            return
        overlay = self.ensure_overlay()
        overlay.draw_masks(self.candidates, self.geometry, selected_ids=[c.id for c in self.candidates])
        overlay.show()
        if (self.mode == "semi-auto" and not self.disabled.get() and not self.runner.busy
                and not self.transition and self.advanced is None):
            self.input_kind = "masks"
            self.input_surface = InputSurface(self.root, self.geometry, self._input_press,
                                              self._input_move, self._input_release,
                                              on_cancel=self.stop)
            self.input_surface.show()
        self._show_panel()

    def _semi_click(self, x, y):
        try:
            if self.mode != "semi-auto" or self.disabled.get() or self.runner.busy or self.transition:
                return
            candidate = screen_mask_at(self.candidates, self.geometry, x, y)
            if candidate is not None:
                self._start_model_picks([candidate])
        except Exception as exc:
            self._error(exc)

    def box_prompt(self):
        try:
            self._idle()
            if self.mode != "semi-auto" or self.geometry is None:
                raise ValueError("In semi-auto mode, select the screen area first.")
            self._hide_overlays()
            self._check_panel_clear(self.geometry)
            self.input_kind = "box"
            self.ensure_overlay().begin_selection()
            self.overlay.draw_region(self.geometry)
            self.overlay.show()
            self.input_surface = InputSurface(self.root, self.geometry, self._input_press,
                                              self._input_move, self._input_release,
                                              on_cancel=self.stop)
            self.input_surface.show()
        except Exception as exc:
            self._error(exc)

    def next_micrograph(self):
        """Advance one screen image and show fresh contours without picking."""
        try:
            self._idle()
            if self.mode != "semi-auto":
                return
            if self.hotkeys is None:
                raise RuntimeError("Global hotkeys are unavailable. Restart the Windows application.")
            # Disable Picking controls generated picks, not this explicit
            # navigation request. Recognition can still run for review.
            self._model_config()
            self._next_frame(auto=False)
        except Exception as exc:
            self._error(exc)

    def _start_model_picks(self, candidates, *, auto=False):
        delay = self._live_ready()
        geometry, target, reference = self.geometry, self.target, self.reference
        if geometry is None or target is None or reference is None:
            raise ValueError("Detect the selected screen area before picking.")
        candidates, picked = tuple(candidates), set(self.picked_points)
        spacing, frame_id = int(self.spacing.get()), self.frame_id
        SCREEN_LOG.info("Starting model picks: frame=%s contours=%s previous_picks=%s auto=%s ids=%s",
                        frame_id, len(candidates), len(picked), auto, [candidate.id for candidate in candidates])
        self._hide_overlays()
        self._check_panel_clear(geometry)
        self.transition = True
        def begin():
            self.transition = False
            def work(token):
                batches = screen_model_batches(candidates, geometry, spacing, picked,
                                               cancel=lambda: token.cancelled)
                if not batches:
                    raise ValueError("There are no detected contours to pick.")
                planned = sum(len(batch.screen) for batch in batches)
                self.desktop.focus_target(target)
                token.sleep(.12)
                if self.desktop.window_under_roi(geometry) != target:
                    raise RuntimeError("Another window covers the micrograph.")
                alignment = screen_background_guard((), geometry, spacing, picked)
                if alignment is None:
                    raise RuntimeError("Too little unchanged background to verify the remaining contours. Select a larger area and Detect again.")
                if screen_background_changed(reference, self._capture_region(geometry), alignment) > .03:
                    raise RuntimeError("The micrograph changed before picking. Detect again.")
                background = screen_background_guard(candidates if planned else (), geometry, spacing, picked)
                next_check = time.monotonic() + .75
                def verify_scene(*, force=False):
                    nonlocal next_check
                    self.desktop.ensure_target(target)
                    if background is not None and (force or time.monotonic() >= next_check):
                        current = self._capture_region(geometry)
                        if screen_background_changed(reference, current, background) > .03:
                            raise RuntimeError("The micrograph changed during picking. Detect again.")
                        next_check = time.monotonic() + .75
                def click(x, y):
                    verify_scene()
                    self.desktop.click(x, y)
                count, completed_local = 0, []
                for index, batch in enumerate(batches, 1):
                    token.wait_until_ready()
                    if index > 1:
                        # A separate play_clicks call must not remove the wait
                        # between the previous contour's last pick and this one.
                        if count and len(batch.screen):
                            token.sleep(delay)
                        verify_scene(force=True)
                    self.events.put(("contour_progress", token.generation, frame_id,
                                     index, len(batches), 0, len(batch.screen)))
                    if not len(batch.screen):
                        self.events.put(("contour_done", token.generation, frame_id, batch.candidate_id))
                        continue
                    def progress(done, total):
                        actual = tuple(map(int, batch.local[done-1]))
                        completed_local.append(actual)
                        self.events.put(("picked", token.generation, frame_id,
                                         actual, done, total, index, len(batches)))
                        # Record actual completion before a cancellation check
                        # can discard the final worker result.
                        if done == total:
                            self.events.put(("contour_done", token.generation, frame_id, batch.candidate_id))
                    count += play_clicks(batch.screen, click, geometry, token, delay=delay,
                                         max_clicks=10000, progress=progress)
                    SCREEN_LOG.info("Completed model contour: frame=%s contour=%s/%s id=%s picks=%s",
                                    frame_id, index, len(batches), batch.candidate_id, len(batch.screen))
                token.sleep(.12)
                self.desktop.ensure_target(target)
                after = self._capture_region(geometry)
                if background is not None and screen_background_changed(reference, after, background) > .03:
                    raise RuntimeError("The micrograph changed during picking. Detect again.")
                return {"count": count, "ids": {batch.candidate_id for batch in batches}, "reference": after,
                        "auto": auto, "contours": len(batches), "scene_verified": background is not None,
                        "frame_id": frame_id, "picked": tuple(completed_local)}
            self._submit("model picks", work)
        self._schedule(90, begin)

    def start_auto(self):
        try:
            self._idle()
            if self.mode != "auto":
                return
            self._live_ready()
            count, wait = int(self.frames.get()), float(self.wait_seconds.get())
            if not 0 <= count <= 1000 or not 1 <= wait <= 60:
                raise ValueError("Choose a limit of 0–1000 micrographs (0 repeats) and a load timeout of 1–60 seconds.")
            # -1 is an active continuous run; zero always means no active run.
            self.auto_remaining, self.auto_history, self.paused = count if count else -1, [], False
            self.auto_completed = 0
            self.hotkeys.enable_space_pause(True)
            if self.candidates and self.reference is not None:
                # Execute the exact displayed proposals. The picking worker
                # first compares a fresh clean capture with this reference.
                self.auto_history = [self.reference.copy()]
                self._start_model_picks(self.candidates, auto=True)
            else:
                self._start_detection(auto=True)
        except Exception as exc:
            self._error(exc)

    def _finish_auto_frame(self):
        self.auto_completed += 1
        if self.reference is not None:
            self.auto_history = (self.auto_history + [self.reference.copy()])[-20:]
        if self.auto_remaining > 0:
            self.auto_remaining -= 1
        if self.auto_remaining:
            self._next_frame()
        else:
            self.paused = False
            self.hotkeys.enable_space_pause(False)
            self._show_panel()
            self.status.set(f"Auto finished: {self.auto_completed} micrograph(s) processed. Review the picks in cryoSPARC.")

    def _next_frame(self, *, auto=True):
        geometry, target = self.geometry, self.target
        if geometry is None or target is None:
            raise ValueError("Select the micrograph screen area first.")
        wait = float(self.wait_seconds.get())
        if not 1 <= wait <= 60:
            raise ValueError("Choose a load timeout of 1–60 seconds.")
        history = tuple(self.auto_history) if auto else ()
        reference, picked = self.reference, set(self.picked_points)
        spacing = int(self.spacing.get())
        self._hide_overlays()
        self._check_panel_clear(geometry)
        # Down may advance before a cancelled worker returns. Retire the old
        # contours now so re-enabling picking cannot draw them on a new image.
        self.candidates, self.alternatives = [], []
        self.transition = True
        def begin():
            self.transition = False
            def work(token):
                self.desktop.focus_target(target)
                token.wait_until_ready()
                def check_target():
                    self.desktop.ensure_target(target)
                    if self.desktop.window_under_roi(geometry) != target:
                        raise RuntimeError("Another window covers the micrograph. Navigation stopped.")
                check_target()
                background = screen_background_guard((), geometry, spacing, picked)
                if background is None:
                    raise RuntimeError("Too little unchanged background to verify the next micrograph. Select a larger screen area.")
                before = self._capture_region(geometry)
                if reference is not None and screen_background_changed(reference, before, background) > .03:
                    raise RuntimeError("The micrograph changed before navigation. Detect again.")
                token.wait_until_ready()
                self.desktop.press_next("down")
                SCREEN_LOG.info("Sent one Down key for %s navigation", "auto" if auto else "semi-auto")
                try:
                    after = wait_for_next_micrograph(
                        lambda: self._capture_region(geometry), before, token,
                        unchanged_mask=background, history=history, timeout=wait,
                        check_target=check_target)
                except NoNewMicrographError as exc:
                    # At the last image (or if Down was not handled), do not
                    # detect or pick the old image again. Preserve actual picks.
                    return {"rgb": before, "auto": auto, "advanced": False, "reason": str(exc)}
                return {"rgb": after, "auto": auto, "advanced": True}
            self._submit("next frame", work)
        # Let the native input/annotation layers disappear before capturing.
        self._schedule(90, begin)

    def _submit(self, kind, function):
        token = CancellationToken(self.runner.generation)
        if self.paused:
            token.pause()
        self.active_token = token
        if self._emergency_stop.is_set():
            token.cancel()
            raise CancelledOperation("Stop requested.")
        try:
            self.runner.submit(function, token=token)
        except Exception:
            self.active_token = None
            raise
        self.job_kind = kind
        self.status.set(f"{kind.capitalize()}… Escape stops.")

    def _result(self, kind, value):
        if kind == "manual picks":
            if self.overlay is not None:
                self.overlay.clear()
                self.overlay.hide()
            self._show_panel()
            self.status.set(f"{value['count']} picks placed. Hold left mouse button to draw the next shape.")
        elif kind == "recognition":
            same_scene = False
            if self.reference is not None:
                guard = screen_background_guard((), self.geometry, int(self.spacing.get()), self.picked_points)
                # Late marker painting is not a new micrograph. If so much of
                # the old image was picked that identity is uncertain, retain
                # its actual picks until navigation/area selection verifies a
                # new frame; guessing here could place duplicate picks.
                same_scene = (guard is None or
                              screen_background_changed(self.reference, value["rgb"], guard) < .02)
            if not same_scene:
                self.picked_points.clear()
            self.rgb = value["rgb"]
            self.reference = self.rgb.copy()
            self.frame_id += 1
            self.alternatives = list(value["candidates"]) if value["prompt"] else []
            self.alternative_index = 0
            self.candidates = self.alternatives[:1] if value["prompt"] else list(value["candidates"])
            if value["auto"]:
                self.auto_history = (self.auto_history + [self.reference.copy()])[-20:]
                if self.candidates:
                    self._start_model_picks(self.candidates, auto=True)
                else:
                    SCREEN_LOG.info("No contours on auto micrograph %s; continuing without picks", self.auto_completed + 1)
                    self._finish_auto_frame()
            else:
                self._present_masks()
                if self.candidates:
                    self.status.set(f"{len(self.candidates)} proposals. " + ("Uncheck Disable Picking to select on screen." if self.disabled.get()
                                    else "Click an intended green region to pick it."))
                else:
                    self.status.set("No proposals. Try a tight Box prompt around a target.")
        elif kind == "model picks":
            if value.get("frame_id", self.frame_id) != self.frame_id:
                return
            # Progress events can trail a completed result or exceed a poll's
            # event budget. Commit actual clicks before making the next
            # contour selectable; its planner must see the complete history.
            self.picked_points.update(value.get("picked", ()))
            if not value.get("scene_verified", True):
                self.stop(message=f"{value['count']} picks placed. Too little unchanged background to verify alignment; select a larger area and Detect again.")
                self.candidates, self.alternatives = [], []
                self.reference = self.rgb = None
                return
            self.reference = value["reference"]
            self.candidates = [c for c in self.candidates if c.id not in value["ids"]]
            self.alternatives = []
            if value["auto"]:
                self._finish_auto_frame()
            else:
                self._present_masks()
                self.status.set(f"{value['count']} picks placed. Click another green region or detect again.")
        elif kind == "next frame":
            if not value["advanced"]:
                self.reference = self.rgb = value["rgb"]
                self.candidates, self.alternatives = [], []
                count = f" {self.auto_completed} micrograph(s) processed." if value["auto"] else ""
                self.stop(message=f"No new micrograph appeared. Stopped; this may be the last image or Down was not handled.{count}")
                return
            self.picked_points.clear()
            self.candidates, self.alternatives = [], []
            # Retire old queued progress before recognition of this image.
            self.frame_id += 1
            self.reference = self.rgb = value["rgb"]
            self._start_detection(auto=value["auto"], expected=value["rgb"])

    def pause(self):
        if not self.runner.busy and not self.transition and not self.auto_remaining:
            return
        self.paused = not self.paused
        if self.active_token is not None:
            self.active_token.pause() if self.paused else self.active_token.resume()
        self.status.set("Paused. Space/F8 resumes; Escape stops." if self.paused else "Resumed.")

    def stop(self, *, message=None):
        self._gate_manual()
        self.runner.invalidate()
        for identifier in tuple(self.scheduled):
            self.root.after_cancel(identifier)
        self.scheduled.clear()
        self.transition = False
        self.auto_remaining, self.paused = 0, False
        self.drawing = self.released = False
        self.trace, self.manual_window = [], None
        self.trace_dirty = False
        self._close_input()
        if self.overlay is not None:
            self.overlay.clear()
            self.overlay.hide()
        if self.hotkeys is not None:
            self.hotkeys.enable_space_pause(False)
        self.disabled.set(True)
        self._emergency_stop.clear()
        self._show_panel()
        self.status.set(message or "Picking disabled. Uncheck Disable Picking when ready.")

    def poll(self):
        """Keep queue processing alive after a recoverable callback failure."""
        identifier = getattr(self, "_poll_after", None)
        self._poll_after = None
        if identifier is not None:
            self.root.after_cancel(identifier)
        if self.closed:
            return
        try:
            self._poll_once()
        except Exception as exc:
            SCREEN_LOG.exception("Screen event processing failed; stopping the active operation")
            try:
                self._error(exc)
            except Exception:
                # Cancellation must survive even a failed native window update.
                self.runner.invalidate()
                SCREEN_LOG.exception("Screen error recovery could not update the panel")
        finally:
            if not self.closed:
                self._poll_after = self.root.after(30, self.poll)

    def _poll_once(self):
        # A collected completion guarantees that all its progress events have
        # already been queued. Collect before draining so the final event
        # cannot arrive between an empty-queue check and enabling selection.
        completed_results = self.runner.poll_results()
        try:
            for _ in range(4000):
                item = self.events.get_nowait()
                if item[0] == "mouse":
                    self._manual_input(*item[1:])
                elif item[0] == "hotkey":
                    if item[1] == "stop":
                        self.stop()
                    elif item[1] == "pause" or (item[1] == "space_pause" and self.auto_remaining):
                        self.pause()
                elif item[0] == "progress" and item[1] == self.runner.generation:
                    self.status.set(f"Picking: {item[2]} / {item[3]}. Escape stops.")
                elif item[0] == "picked":
                    if item[2] == self.frame_id:
                        self.picked_points.add(item[3])
                    if item[1] == self.runner.generation and item[2] == self.frame_id:
                        self.status.set(f"Contour {item[6]}/{item[7]}: {item[4]}/{item[5]} picks. Escape stops.")
                elif item[0] == "contour_progress":
                    if item[1] == self.runner.generation and item[2] == self.frame_id:
                        self.status.set(f"Contour {item[3]}/{item[4]}: {item[5]}/{item[6]} picks. Escape stops.")
                elif item[0] == "contour_done" and item[2] == self.frame_id:
                    # A successful final click is still real when Escape
                    # invalidates the worker before its result is delivered.
                    self.candidates = [candidate for candidate in self.candidates if candidate.id != item[3]]
                    if any(candidate.id == item[3] for candidate in self.alternatives):
                        self.alternatives = []
        except queue.Empty:
            pass
        for result in completed_results:
            # Escape may invalidate this completion while its earlier progress
            # is being processed. Real clicks stay recorded; stale UI state
            # must not be presented again.
            if result.token.generation != self.runner.generation:
                continue
            kind = self.job_kind
            self.active_token = None
            if result.error is not None or result.cancelled:
                if result.traceback_text:
                    SCREEN_LOG.error("%s", result.traceback_text)
                self._error("Cancelled." if result.cancelled or isinstance(result.error, (Cancelled, CancelledOperation)) else result.error)
                continue
            try:
                self._result(kind, result.value)
            except Exception as exc:
                SCREEN_LOG.exception("Screen result handling failed")
                self._error(exc)
        if self.drawing and self.trace_dirty:
            try:
                self.ensure_overlay().draw_trace(self.trace)
                if not self.overlay.visible:
                    self.overlay.show()
                self.trace_dirty = False
            except Exception as exc:
                self._error(exc)
        self._sync_manual()

    def _settings_available(self):
        """Panel settings access must not interrupt or focus over active work."""
        if self.closed:
            return False
        if self.advanced is not None:
            return True
        try:
            self._idle()
        except RuntimeError:
            return False
        return True

    def _open_picking_speed(self, event=None):
        if self._settings_available():
            self.open_advanced()
        return "break"

    def _show_picking_menu(self, event):
        if not self._settings_available():
            return "break"
        if self._picking_menu is None:
            self._picking_menu = self.tk.Menu(self.root, tearoff=False)
            self._picking_menu.add_command(label="Picking speed…", accelerator="F9",
                                           command=self._open_picking_speed)
        try:
            self._picking_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._picking_menu.grab_release()
        return "break"

    def open_advanced(self):
        try:
            if self.advanced is not None:
                self.advanced.lift()
                return
            self._idle()
            previous_config = self._model_config()
            self._gate_manual()
            self._close_input()
            window = self.tk.Toplevel(self.root)
            self.advanced = window
            window.title("SARU Picker — settings")
            window.attributes("-topmost", True)
            window.resizable(False, False)
            frame = self.ttk.Frame(window, padding=12)
            frame.pack(fill="both", expand=True)
            settings_wrap = max(410, round(410 * self.root.winfo_fpixels("1i") / 96))
            self.ttk.Label(frame, text="SAM checkpoint").grid(row=0, column=0, sticky="w")
            self.ttk.Entry(frame, textvariable=self.checkpoint, width=48).grid(row=1, column=0, columnspan=2, sticky="ew")
            def browse():
                path = self.filedialog.askopenfilename(parent=window, filetypes=[("SAM checkpoint", "*.pth")])
                if path:
                    self.checkpoint.set(path)
                    for model_type in ("vit_b", "vit_l", "vit_h"):
                        if model_type in Path(path).name:
                            self.model_type.set(model_type)
                            break
            self.ttk.Button(frame, text="Browse…", command=browse).grid(row=1, column=2, padx=4)
            for index, (label, variable, values) in enumerate((("Device", self.device, ("auto", "cpu", "cuda")),
                                                               ("Model type", self.model_type, ("vit_b", "vit_l", "vit_h")),
                                                               ("Detection profile", self.automatic_profile, ("capsid", "capsid-learned", "generic"))), 2):
                self.ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
                self.ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=18 if label == "Detection profile" else 10).grid(row=index, column=1)
            for index, (label, variable, low, high, step) in enumerate((("Automatic grid / side", self.grid_side, 4, 32, 1),
                                                                        ("Click delay (all modes; 0 = maximum)", self.click_delay, 0, 2, .01),
                                                                        ("Minimum capsid area (screen px²)", self.min_capsid_area, 0, 1000000, 500)), 5):
                self.ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
                self.ttk.Spinbox(frame, textvariable=variable, from_=low, to=high, increment=step, width=10).grid(row=index, column=1)
            def close_settings():
                try:
                    self._apply_model_settings(previous_config)
                except (ValueError, self.tk.TclError) as exc:
                    self.messagebox.showerror("SARU Picker", str(exc), parent=window)
                    return
                window.destroy()
                self.advanced = None
                self._gate_manual()
                self._sync_manual()
                if self.mode == "semi-auto":
                    self._present_masks()
            self.ttk.Label(frame, text="capsid: existing dense-interior filter (default).\ncapsid-learned: learned capsids — experimental; review contours.\ngeneric: all SAM regions.", wraplength=settings_wrap).grid(
                row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))
            self.ttk.Label(frame, text="Both capsid profiles skip regions below the minimum area. 3,500 px² is the large-only default; 0 allows all sizes. Screen zoom changes apparent size. Manual and Box prompt selections are unchanged.",
                           wraplength=settings_wrap).grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))
            self.ttk.Label(frame, text="Click delay is in seconds. 0 removes the added wait; increase it if cryoSPARC misses picks.",
                           wraplength=settings_wrap).grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))
            self.ttk.Button(frame, text="Done", command=close_settings).grid(row=11, column=0, columnspan=3, pady=(10, 0))
            window.protocol("WM_DELETE_WINDOW", close_settings)
        except Exception as exc:
            self._error(exc)

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Variable traces own Tcl commands separately from root.bind/after.
        # root.destroy alone does not delete those commands, which otherwise
        # keep bound controller callbacks and the interpreter graph alive.
        traces, self._variable_traces = getattr(self, "_variable_traces", []), []
        for variable, identifier in traces:
            try:
                variable.trace_remove("write", identifier)
            except self.tk.TclError:
                SCREEN_LOG.warning("A panel variable trace was already unavailable during shutdown.")
        if getattr(self, "_poll_after", None) is not None:
            self.root.after_cancel(self._poll_after)
            self._poll_after = None
        if self._fit_after is not None:
            self.root.after_cancel(self._fit_after)
            self._fit_after = None
        self._gate_manual()
        stopped = self.runner.shutdown(wait=False)
        for identifier in tuple(self.scheduled):
            self.root.after_cancel(identifier)
        self.scheduled.clear()
        if self.mouse_observer is not None:
            self.mouse_observer.stop()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self._close_input()
        if self.overlay is not None:
            self.overlay.close()
        if self.advanced is not None:
            self.advanced.destroy()
            self.advanced = None
        if stopped:
            self.root.destroy()
        else:
            # A native model pass cannot be interrupted. Keep its interpreter
            # alive until callbacks/results can be released on the GUI thread.
            self.status.set("Stopping the active operation… the window will close when it has stopped.")
            self._shutdown_after = self.root.after(30, self._finish_close)

    def _finish_close(self):
        self._shutdown_after = None
        self.runner.poll_results()
        if self.runner.shutdown(wait=False):
            self.root.destroy()
        else:
            self._shutdown_after = self.root.after(30, self._finish_close)
