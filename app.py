"""Screen picker entry point and optional offline review UI for SARU Picker.

The desktop image remains the coordinate source. Resizing this preview never
changes the sampling interval in the captured image. Tk is used only on the
main thread; inference and click playback share one cancellable worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import queue
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from application_icon import (apply_application_icon, application_icon_paths,
                              configure_application_identity, windows_application_icon_status)
from capsid import filter_capsid_proposals, polygonal_capsid_mask
from capsid_prior import load_profile, resolve_profile_path, score_image
from carbon_support import detect_carbon_edges, guard_proposals

from recognition import (
    Candidate, RecognitionConfig, RecognitionError, Cancelled, SamRecognizer, create_recognizer, DEFAULT_MIN_CAPSID_AREA_PX,
    _filter_capsid_area,
    make_candidate, polygon_mask, union_masks, sample_hex_grid, validate_rgb,
)
from execution import (
    CaptureGeometry, CancellationToken, CancelledOperation, TaskRunner,
    WorkerBusyError, play_clicks,
)
from desktop import DesktopAdapter, GlobalHotkeys, enable_dpi_awareness
from screen_app import (ScreenPickerApp, screen_model_batches, screen_changed_fraction,
                        screen_region_overlaps_panel, initialize_detector_fields, detector_config_from_fields,
                        detection_profile_value, persist_detector_fields, apply_detector_args,
                        begin_recognition_progress, queue_recognition_progress, accept_recognition_progress,
                        clear_recognition_progress, refresh_recognition_progress)
from detector_selection import (DETECTION_PROFILES, canonical_detection_profile, resolve_detection_profile,
                                load_detector_settings)
from screen_overlay import TRANSPARENT_RGB, control_panel_bounds, verify_screen_overlay

APP_LOG = logging.getLogger("saru_picker")
APP_VERSION = "3.0.0"
DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"
DEFAULT_CRYOLO_CHECKPOINT = "cryolo-capsids.onnx"
ACTIVE_BACKEND = "sam"


def user_state_directory() -> Path:
    """Writable application data, independent of the installation directory."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data and Path(local_app_data).is_absolute():
            return Path(local_app_data) / "SARU_Picker"
        return Path.home() / "AppData" / "Local" / "SARU_Picker"
    return Path.home() / ".saru_picker"


def resolve_checkpoint(explicit: str | Path | None = None, *, backend: str = "sam") -> Path:
    """Locate the installed model without relying on the launch directory.

    An explicit choice is never replaced by a different checkpoint. PyInstaller
    puts onedir data beneath _MEIPASS (usually the _internal directory).
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    directories = []
    if getattr(sys, "frozen", False):
        if getattr(sys, "_MEIPASS", None):
            bundle = Path(sys._MEIPASS)
            directories.extend((bundle / "models", bundle))
        executable_directory = Path(sys.executable).resolve().parent
        directories.extend((executable_directory / "models", executable_directory))
    source_directory = Path(__file__).resolve().parent
    directories.extend((source_directory / "models", source_directory))
    filename = DEFAULT_CRYOLO_CHECKPOINT if backend == "cryolo" else DEFAULT_CHECKPOINT
    for directory in directories:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    # Keep a concrete expected path for a useful missing-model message.
    return directories[0] / filename


def model_runtime_options(args) -> dict:
    """Keep GUI, headless detection and diagnostics on the same backend settings."""
    return {"backend": getattr(args, "backend", "sam"),
            "cryolo_config": getattr(args, "cryolo_config", None),
            "cryolo_python": getattr(args, "cryolo_python", None),
            "cryolo_threshold": getattr(args, "cryolo_threshold", .3),
            "cryolo_timeout": getattr(args, "cryolo_timeout", 180.0)}


@contextmanager
def windowed_console_streams():
    """Give dependencies valid streams when pythonw/PyInstaller has no console."""
    replacements = []
    try:
        for name in ("stdout", "stderr"):
            if getattr(sys, name) is None:
                stream = open(os.devnull, "w", encoding="utf-8")
                setattr(sys, name, stream)
                replacements.append((name, stream))
        yield
    finally:
        for name, stream in replacements:
            if getattr(sys, name) is stream:
                setattr(sys, name, None)
            stream.close()


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write an explicit diagnostic destination, including failures."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".saru-report-", suffix=".json",
                                         delete=False) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def diagnostic_report(log_path: Path, checkpoint: Path) -> dict[str, Any]:
    dependencies = {}
    for package in ("numpy", "Pillow", "scipy", "torch", "torchvision",
                    "segment-anything", "PyAutoGUI", "pynput"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "status": "ok", "mode": "diagnostics", "version": APP_VERSION, "backend": "sam",
        "python": sys.version, "platform": sys.platform,
        "frozen": bool(getattr(sys, "frozen", False)),
        "dependencies": dependencies,
        "checkpoint": str(checkpoint), "checkpoint_exists": checkpoint.is_file(),
        "log_path": str(log_path),
        "note": "Metadata inspection only; use --smoke-test for native libraries and model inference.",
    }


def _smoke_image_processing() -> dict[str, Any]:
    mask = np.ones((16, 24), dtype=bool)
    mask[4:12, 8:16] = False
    points = sample_hex_grid(mask, 3, inset=1)
    if not len(points) or not mask[points[:, 1], points[:, 0]].all():
        raise RuntimeError("Bitmap sampling failed its inclusion check.")
    # Exercise the actual model-mode planner and its compiled spatial index
    # in the frozen binary without sending any desktop input.
    batches = screen_model_batches([make_candidate(mask), make_candidate(mask.copy())],
                                   CaptureGeometry(40, 50, 24, 16), 3)
    if not len(batches[0].local) or len(batches[1].local):
        raise RuntimeError("Per-contour duplicate filtering failed.")
    if not mask[batches[0].local[:, 1], batches[0].local[:, 0]].all():
        raise RuntimeError("Per-contour points escaped the mask.")
    # A uniform scene must not become a capsid merely because SAM proposed a
    # region. This exercises the new SciPy image-processing runtime as well.
    blank = np.full((64, 96, 3), 125, np.uint8)
    proposal = np.zeros((64, 96), dtype=bool)
    proposal[20:44, 24:72] = True
    if filter_capsid_proposals(blank, [{"segmentation": proposal}]):
        raise RuntimeError("Uniform-image capsid rejection failed.")
    # Exercise the exact final size gate in source, frozen and installed builds.
    # A disjoint small island or a nearly-large ring must not pool area to pass.
    components = np.zeros((200, 220), dtype=bool)
    components[10:60, 10:80] = True  # Exactly 3,500 foreground pixels.
    components[80:110, 10:110] = True  # 3,000-pixel island.
    components[120:180, 120:180] = True
    components[140:151, 140:151] = False  # Ring has 3,479 foreground pixels.
    sized, size_evidence = _filter_capsid_area(
        [{"segmentation": components}], DEFAULT_MIN_CAPSID_AREA_PX)
    expected = np.zeros_like(components)
    expected[10:60, 10:80] = True
    if (DEFAULT_MIN_CAPSID_AREA_PX != 3500 or len(sized) != 1
            or not np.array_equal(sized[0]["segmentation"], expected)):
        raise RuntimeError("Automatic capsid size filtering failed its component/holes check.")
    core = np.zeros((80, 100), dtype=bool)
    core[10:70, 10:90] = True
    core[10:18, 40:50] = False
    core[35:39, 25:29] = False
    allowed = np.ones_like(core)
    allowed[35:39, 25:29] = False
    polygon, evidence = polygonal_capsid_mask(core, allowed_mask=allowed)
    if (not evidence["components"][0]["applied"] or np.any(polygon & ~allowed)
            or not polygon[12, 44] or np.any(polygon[35:39, 25:29])):
        raise RuntimeError("Polygon fitting or its excluded-area constraint failed.")
    # Exercise the packaged shared guard with a long support edge and a real
    # compact target immediately beside it. No desktop events or model data.
    yy, xx = np.mgrid[:320, :480]
    distance = xx - .35 * yy - 150
    rng = np.random.default_rng(261)
    grey = 125 + rng.normal(0, 4, (320, 480))
    grey[distance < 0] -= 35
    grey[np.abs(distance) < 1.7] = 25
    film_fragment = ((distance >= -20) & (distance <= -2) & (yy >= 95) & (yy <= 220))
    compact = (np.abs(xx - 235) < 30) & (np.abs(yy - 160) < 30)
    grey[compact] += rng.normal(0, 25, int(compact.sum()))
    scene = np.repeat(np.uint8(np.clip(grey, 0, 255))[..., None], 3, axis=2)
    records = [{"segmentation": film_fragment}, {"segmentation": compact}]
    context = detect_carbon_edges(scene)
    retained, guard = guard_proposals(context, records)
    if (not context.segments or guard["rejected_indices"] != [0]
            or len(retained) != 1 or retained[0] is not records[1]
            or not np.array_equal(retained[0]["segmentation"], compact)):
        raise RuntimeError("Carbon guard failed to reject the film fragment or preserve its neighboring target.")
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        if decoded.size != (12, 8):
            raise RuntimeError("Pillow PNG decoding returned the wrong dimensions.")
    return {"grid_points": int(len(points)), "png_roundtrip": True,
            "per_contour_grid_points": int(len(batches[0].local)),
            "cross_contour_duplicate_guard": True,
            "capsid_uniform_scene_rejected": True,
            "capsid_polygon_fitting": True,
            "polygon_exclusions_preserved": True,
            "capsid_size_filter": {"status": "ok", "default_minimum_area_px": DEFAULT_MIN_CAPSID_AREA_PX,
                                   "disconnected_small_islands_rejected": True,
                                   "holes_excluded_from_area": True, "exact_boundary_retained": True,
                                   "diagnostics": size_evidence},
            "carbon_guard": {"status": "ok", "detected_edges": len(context.segments),
                             "rejected_film_contours": guard["rejected_count"],
                             "nearby_capsid_preserved": True}}


def _smoke_torchvision() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 25., 25.]])
    scores = torch.tensor([.9, .8, .7])
    retained = torchvision.ops.nms(boxes, scores, .5).tolist()
    if retained != [0, 2]:
        raise RuntimeError("The compiled torchvision NMS operation returned an unexpected result.")
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "native_nms": True}


def _smoke_layout_diagnostics(window, widget=None, ancestor=None) -> str:
    """Include native mapping and geometry in build-host layout failures."""
    def describe(control):
        return {"path": str(control),
                "actual": [int(control.winfo_width()), int(control.winfo_height())],
                "requested": [int(control.winfo_reqwidth()), int(control.winfo_reqheight())],
                "screen_xy": [int(control.winfo_rootx()), int(control.winfo_rooty())],
                "mapped": bool(control.winfo_ismapped()),
                "viewable": bool(control.winfo_viewable()),
                "geometry": str(control.winfo_geometry()),
                "manager": str(control.winfo_manager())}
    details = {"state": str(window.state()), "panel": describe(window)}
    if widget is not None:
        details["widget"] = describe(widget)
    if ancestor is not None:
        details["clipping_ancestor"] = describe(ancestor)
    return json.dumps(details, sort_keys=True)


def _smoke_settle_panel(picker) -> None:
    """Dispatch native mapping events, fit content, then dispatch the resize.

    Idle-only processing can leave Windows child HWNDs unmapped after the root
    is shown. Two fixed update calls service the native queue without a polling
    loop, delayed retry, or any visibility/state repair that could hide a bug.
    The smoke picker has no input listeners and remains disabled throughout.
    """
    picker.root.update()
    picker.fit_panel()
    picker.root.update()


def _smoke_widget_layout(window, widget, description: str) -> dict[str, Any]:
    """Reject a managed control that is hidden, shrunk, or ancestor-clipped.

    Tk's packer can shrink a button to only a few visible pixels while its
    bottom edge still fits the root. Comparing only the two bottom edges does
    not establish that the label or complete clickable area is available.
    """
    size = [int(widget.winfo_width()), int(widget.winfo_height())]
    requested = [int(widget.winfo_reqwidth()), int(widget.winfo_reqheight())]
    if not widget.winfo_viewable():
        raise RuntimeError(f"{description} is not visible in the floating panel; "
                           f"layout={_smoke_layout_diagnostics(window, widget)}")
    if min(size) <= 0 or any(actual < needed for actual, needed in zip(size, requested)):
        raise RuntimeError(
            f"{description} is clipped below its requested size: actual {size}, requested {requested}; "
            f"layout={_smoke_layout_diagnostics(window, widget)}")
    left, top = int(widget.winfo_rootx()), int(widget.winfo_rooty())
    right, bottom = left + size[0], top + size[1]
    parent = widget.master
    while parent is not None:
        parent_left, parent_top = int(parent.winfo_rootx()), int(parent.winfo_rooty())
        parent_right = parent_left + int(parent.winfo_width())
        parent_bottom = parent_top + int(parent.winfo_height())
        if (left < parent_left or top < parent_top or right > parent_right or bottom > parent_bottom):
            raise RuntimeError(f"{description} extends outside its parent and is not fully usable; "
                               f"layout={_smoke_layout_diagnostics(window, widget, parent)}")
        if parent is window:
            break
        parent = parent.master
    if parent is not window:
        raise RuntimeError(f"{description} is not attached to the floating panel; "
                           f"layout={_smoke_layout_diagnostics(window, widget)}")
    return {"actual": size, "requested": requested,
            "panel_rect": [left-int(window.winfo_rootx()), top-int(window.winfo_rooty()), *size],
            "viewable": True, "full_requested_size": True}


def _smoke_terminate_text(window, button) -> dict[str, Any]:
    """Check the full text line, including ascent/descent and button padding."""
    tk_font = importlib.import_module("tkinter.font")
    font = tk_font.Font(root=window, font=button.cget("font"))
    lines = str(button.cget("text")).splitlines() or [""]
    text_width = max(int(font.measure(line)) for line in lines)
    ascent, descent = int(font.metrics("ascent")), int(font.metrics("descent"))
    line_height = int(font.metrics("linespace"))
    border = sum(int(window.winfo_pixels(button.cget(name)))
                 for name in ("borderwidth", "highlightthickness"))
    pad_x, pad_y = (int(window.winfo_pixels(button.cget(name))) for name in ("padx", "pady"))
    usable = [int(button.winfo_width())-2*(border+pad_x),
              int(button.winfo_height())-2*(border+pad_y)]
    text_height = line_height * len(lines)
    if text_width > usable[0] or text_height > usable[1]:
        raise RuntimeError(
            f"Terminate's text or font baseline is clipped: usable {usable}, text {[text_width, text_height]}; "
            f"layout={_smoke_layout_diagnostics(window, button)}")
    # The actual button retains its default centered anchor. This derived
    # baseline records the complete font line that must fit the usable area.
    baseline = border + pad_y + (usable[1]-text_height)//2 + ascent
    if baseline + (len(lines)-1)*line_height + descent > int(button.winfo_height())-border-pad_y:
        raise RuntimeError("Terminate's text descender falls outside its usable button area; "
                           f"layout={_smoke_layout_diagnostics(window, button)}")
    return {"text_size": [text_width, text_height], "usable_text_size": usable,
            "font_metrics": {"ascent": ascent, "descent": descent, "linespace": line_height},
            "centered_first_baseline_from_top": baseline, "full_font_line_fits": True}


def _smoke_scale_label(window, slider) -> dict[str, Any]:
    """Scale's requested size does not include its internal label's width."""
    tk_font = importlib.import_module("tkinter.font")
    font = tk_font.Font(root=window, font=slider.cget("font"))
    label = str(slider.cget("label"))
    text_width = int(font.measure(label))
    border = sum(int(window.winfo_pixels(slider.cget(name)))
                 for name in ("borderwidth", "highlightthickness"))
    available_width = int(slider.winfo_width()) - 2 * border
    if text_width > available_width:
        raise RuntimeError(
            f"Particle spacing caption is clipped: label {text_width}px, interior {available_width}px; "
            f"layout={_smoke_layout_diagnostics(window, slider)}")
    return {"text": label, "text_width": text_width,
            "available_width": available_width, "full_label_width_fits": True}


def _smoke_panel_layout(picker, mode: str, status_case: str) -> dict[str, Any]:
    window = picker.root
    _smoke_settle_panel(picker)
    if window.state() != "normal" or not window.winfo_viewable():
        raise RuntimeError(f"The {mode} floating control panel is not visible; "
                           f"layout={_smoke_layout_diagnostics(window)}")
    if not window.tk.getboolean(window.attributes("-topmost")):
        raise RuntimeError("The floating control panel is not kept above the target application; "
                           f"layout={_smoke_layout_diagnostics(window)}")
    controls = []

    def inspect(parent):
        for child in parent.winfo_children():
            # pack_forget/grid_remove intentionally hide inactive mode groups.
            # A managed child clipped away by pack still has a manager and must
            # fail the visibility/size checks instead of being silently skipped.
            if not child.winfo_manager():
                continue
            description = f"{mode}/{status_case} {child.winfo_class()} {child}"
            controls.append(_smoke_widget_layout(window, child, description))
            inspect(child)

    inspect(window)
    terminate = _smoke_widget_layout(window, picker.terminate_button, f"{mode}/{status_case} Terminate")
    terminate["text"] = _smoke_terminate_text(window, picker.terminate_button)
    status = _smoke_widget_layout(window, picker.status_label, f"{mode}/{status_case} status")
    spacing = _smoke_scale_label(window, picker.spacing_slider)
    return {"panel_size": [int(window.winfo_width()), int(window.winfo_height())],
            "panel_requested_size": [int(window.winfo_reqwidth()), int(window.winfo_reqheight())],
            "managed_controls_checked": len(controls), "all_managed_controls_fit": True,
            "terminate": terminate, "status": status, "spacing_caption": spacing,
            "viewable": True, "topmost": True}


def _smoke_panel_screenshot(window, destination: Path) -> dict[str, Any]:
    """Validate two painted client captures without repairing window state.

    Tk mapping/layout completion does not establish that the compositor has
    presented the client pixels yet. Pump normally for a short fixed interval
    and use DwmFlush as a synchronization aid before each real screen capture.
    DwmFlush does not flush the whole session, so both pixel checks must pass.
    No visibility, focus, geometry or repaint repair is performed here.
    """
    import ctypes

    image_grab = importlib.import_module("PIL.ImageGrab")
    flush = ctypes.windll.dwmapi.DwmFlush
    flush.argtypes, flush.restype = [], ctypes.c_long
    background = [int(round(value / 257))
                  for value in window.winfo_rgb(window.cget("background"))]
    inset, edge_band, tolerance, variation_delta = 3, 4, 12, 35
    minimum_background, minimum_edge, minimum_varied = .65, .85, .003
    frames, original_geometry = [], None
    destination.parent.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        # Give native events elapsed time without remapping, lifting or forcing
        # a repaint. Both fixed captures must pass; there is no retry-to-pass.
        for _ in range(5):
            window.update()
            time.sleep(.01)
        window.update()
        hresult = int(flush())
        if hresult != 0:
            raise RuntimeError(f"DwmFlush failed before panel capture: HRESULT {hresult:#x}.")
        if window.state() != "normal" or not window.winfo_viewable():
            raise RuntimeError("The panel lost its visible state before pixel validation.")
        left, top, width, height = control_panel_bounds(window)
        bbox = (left, top, left + width, top + height)
        client_left, client_top = int(window.winfo_rootx()), int(window.winfo_rooty())
        client_width, client_height = int(window.winfo_width()), int(window.winfo_height())
        client_bbox = (client_left, client_top,
                       client_left + client_width, client_top + client_height)
        geometry = (bbox, client_bbox)
        if original_geometry is not None and geometry != original_geometry:
            raise RuntimeError("Panel geometry changed between the two pixel-validation captures.")
        original_geometry = geometry
        capture = image_grab.grab(bbox=bbox, include_layered_windows=True, all_screens=True).convert("RGB")
        frame_path = (destination.with_name(f"{destination.stem}-frame-1{destination.suffix}")
                      if index == 0 else destination)
        # Retain the unmodified capture even if it fails, as build diagnostics.
        capture.save(frame_path)
        if capture.size != (width, height):
            raise RuntimeError("Panel screenshot dimensions differ from its native frame bounds.")
        x0, y0 = client_left - left, client_top - top
        x1, y1 = x0 + client_width, y0 + client_height
        if (min(client_width, client_height) < 32 or x0 < 0 or y0 < 0
                or x1 > width or y1 > height):
            raise RuntimeError("The actual panel client rectangle is outside its captured native frame.")
        pixels = np.asarray(capture, dtype=np.int16)[y0+inset:y1-inset, x0+inset:x1-inset]
        delta = np.max(np.abs(pixels - np.asarray(background, dtype=np.int16)), axis=2)
        matches = delta <= tolerance
        edges = {"top": float(matches[:edge_band, :].mean()),
                 "bottom": float(matches[-edge_band:, :].mean()),
                 "left": float(matches[:, :edge_band].mean()),
                 "right": float(matches[:, -edge_band:].mean())}
        evidence = {"path": str(frame_path), "image_dimensions": [width, height],
                    "background_fraction": float(matches.mean()),
                    "edge_background_fraction": edges,
                    "varied_fraction": float((delta > variation_delta).mean())}
        if (evidence["background_fraction"] < minimum_background
                or min(edges.values()) < minimum_edge
                or evidence["varied_fraction"] < minimum_varied):
            raise RuntimeError(f"Panel capture {index+1} does not show a fully painted client "
                               f"with visible controls: {json.dumps(evidence, sort_keys=True)}")
        frames.append(evidence)
    return {"path": str(destination), "native_outer_bbox": list(bbox),
            "render_check": {"status": "ok", "frames_checked": 2,
                             "event_pump_seconds_before_each": .05,
                             "background_rgb": background, "color_tolerance": tolerance,
                             "variation_delta": variation_delta, "client_inset_pixels": inset,
                             "edge_band_pixels": edge_band, "client_bbox": list(client_bbox),
                             "minimum_background_fraction": minimum_background,
                             "minimum_edge_background_fraction": minimum_edge,
                             "minimum_varied_fraction": minimum_varied, "frames": frames,
                             "method": "Two actual client captures after normal event pumping and DwmFlush; "
                                       "no visibility, focus, geometry or repaint repair. "
                                       "Paint coverage and control variation, not text-pixel identity."}}


def _smoke_repeated_screen_picks(picker) -> dict[str, Any]:
    """Exercise real Tk surfaces and the real worker with an in-memory desktop.

    Events are delivered only to our own Tk canvas. The desktop adapter below
    cannot focus another application or send system input. This catches the
    lifecycle between separate semi-auto jobs in source and frozen builds.
    """
    import threading

    root = picker.root
    previous_mode = picker.mode
    previous_desktop, previous_hotkeys, previous_mouse = (picker.desktop, picker.hotkeys,
                                                         picker.mouse_observer)
    picker.set_mode("semi-auto")
    _smoke_settle_panel(picker)
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    occupied = control_panel_bounds(root)
    geometry = next((CaptureGeometry(x, y, 480, 360)
                     for x, y in ((width-500, 30), (width-500, height-390),
                                  (20, height-390), (20, 30))
                     if x >= 0 and y >= 0
                     and not screen_region_overlaps_panel(CaptureGeometry(x, y, 480, 360), occupied)), None)
    if geometry is None:
        raise RuntimeError("No free screen area for the repeated-contour UI check.")

    class Hooks:
        def enable_space_pause(self, enabled):
            pass
        def stop(self):
            pass

    class MemoryDesktop:
        def __init__(self):
            yy, xx = np.mgrid[:geometry.height, :geometry.width]
            grey = (80 + (xx * 3 + yy * 7) % 100).astype(np.uint8)
            self.rgb = np.repeat(grey[..., None], 3, axis=2)
            self.clicks = []
            self.thread_ids = set()
        def focus_target(self, target):
            self.ensure_target(target)
        def ensure_target(self, target):
            if target != 12345:
                raise RuntimeError("Unexpected synthetic picking target.")
        def window_under_roi(self, region):
            if region != geometry:
                raise RuntimeError("Unexpected synthetic capture area.")
            return 12345
        def _layers_hidden(self):
            if ((picker.overlay is not None and picker.overlay.visible)
                    or (picker.input_surface is not None and picker.input_surface.visible)):
                raise RuntimeError("A screen layer remained visible during picking/capture.")
        def screenshot(self, region):
            self.window_under_roi(region)
            self._layers_hidden()
            return self.rgb.copy()
        def click(self, x, y):
            self._layers_hidden()
            if not geometry.contains(x, y):
                raise RuntimeError("A generated test pick escaped the screen area.")
            self.thread_ids.add(threading.get_ident())
            self.clicks.append((x, y))
        def paint_delayed_markers(self):
            image = Image.fromarray(self.rgb)
            draw = ImageDraw.Draw(image)
            for x, y in self.clicks:
                x, y = x-geometry.left, y-geometry.top
                draw.ellipse((x-15, y-15, x+15, y+15), outline=(40, 240, 60), width=5)
            self.rgb = np.array(image)

    desktop = MemoryDesktop()
    candidates = []
    for top in (30, 205):
        for left in (20, 134, 248, 362):
            mask = np.zeros((geometry.height, geometry.width), dtype=bool)
            mask[top:top+80, left:left+80] = True
            candidates.append(make_candidate(mask, source="synthetic lifecycle check"))
    overlap = np.zeros_like(candidates[0].mask)
    overlap[30:110, 22:102] = True
    exhausted = make_candidate(overlap, source="synthetic exhausted overlap check")
    completed = []
    annotation_presentations, retired_windows = [], []
    panel_handle = int(root.winfo_id())
    delayed_fraction = 0.0
    attempts = []
    previous_result_override = picker.__dict__.get("_result")
    original_result = picker._result

    def record_result(kind, value):
        original_result(kind, value)
        if kind == "model picks":
            planning = value["planning"]
            attempts.append({"candidate_id": next(iter(value["ids"])),
                             "frame_id": value["frame_id"],
                             "raw_grid_points": planning["raw_grid_points"],
                             "blocked_grid_points": planning["blocked_grid_points"],
                             "eligible_grid_points": planning["eligible_grid_points"],
                             "sent_points": planning["sent_points"],
                             "exhausted_ids": sorted(value["exhausted_ids"]),
                             "remaining_plans": list(value["remaining_plans"]),
                             "remaining_contour_ids": [c.id for c in picker.candidates],
                             "status": picker.status.get()})

    def inspect_annotation(expected_candidates):
        # Inspect the actual Tk PhotoImage attached to the sole Canvas item,
        # not a second rendering through the production outline function.
        from PIL import ImageTk
        overlay = picker.overlay
        if overlay is None or not overlay.visible or overlay._mask_photo is None:
            raise RuntimeError("Remaining contours have no visible annotation bitmap.")
        items = overlay.canvas.find_all()
        if (len(items) != 1 or len(overlay.canvas.find_withtag("masks")) != 1
                or overlay.canvas.itemcget(items[0], "image") != str(overlay._mask_photo)):
            raise RuntimeError("A previous annotation remained alongside the current mask bitmap.")
        pixels = np.asarray(ImageTk.getimage(overlay._mask_photo).convert("RGB"))
        painted = np.any(pixels != TRANSPARENT_RGB, axis=2)
        remaining = np.zeros((geometry.height, geometry.width), dtype=bool)
        for candidate in expected_candidates:
            remaining |= candidate.mask
            if not np.any(painted & candidate.mask):
                raise RuntimeError("A remaining contour disappeared from its annotation bitmap.")
        if np.any(painted & ~remaining):
            raise RuntimeError("Completed contours or stale geometry remained in the annotation bitmap.")
        if any(window.winfo_exists() for window in retired_windows):
            raise RuntimeError("A retired contour window was reused or remained alive.")
        annotation_presentations.append({
            "tk_window": str(overlay.window),
            "native_window": int(overlay.window.winfo_id()),
            "remaining_contour_ids": [candidate.id for candidate in expected_candidates],
            "only_remaining_contours_in_bitmap": True,
            "one_current_canvas_image": True,
        })

    def select_and_check(candidate, expected_remaining, expected_picks):
        inspect_annotation(picker.candidates)
        previous_annotation = picker.overlay.window
        previous_canvas = picker.overlay.canvas
        surface = picker.input_surface
        if surface is None or not surface.visible:
            raise RuntimeError("A remaining contour has no clickable overlay.")
        before = len(desktop.clicks)
        previous_attempts = len(attempts)
        ys, xs = np.nonzero(candidate.mask)
        mid = len(xs)//2
        local_x, local_y = int(xs[mid]), int(ys[mid])
        x, y = geometry.local_to_screen(local_x, local_y)
        surface.canvas.event_generate("<ButtonPress-1>", x=local_x, y=local_y,
                                      rootx=x, rooty=y)
        surface.canvas.event_generate("<ButtonRelease-1>", x=local_x, y=local_y,
                                      rootx=x, rooty=y)
        if not (picker.transition or picker.runner.busy):
            raise RuntimeError(f"Contour selection did not start: {picker.status.get()}")
        deadline = time.monotonic() + 15
        while picker.transition or picker.runner.busy:
            root.update()
            if time.monotonic() > deadline:
                raise RuntimeError(f"Contour selection did not finish: {picker.status.get()}")
            time.sleep(.005)
        root.update()
        if picker.disabled.get() or len(attempts) != previous_attempts+1:
            raise RuntimeError(f"Contour result failed: {picker.status.get()}")
        if len(desktop.clicks)-before != expected_picks:
            raise RuntimeError("A fresh or exhausted contour sent an unexpected number of picks.")
        if [c.id for c in picker.candidates] != [c.id for c in expected_remaining]:
            raise RuntimeError("Remaining contours include an exhausted proposal or omit a fresh one.")
        plan = attempts[-1]
        if (plan["candidate_id"] != candidate.id or plan["raw_grid_points"] != 20
                or plan["eligible_grid_points"] != expected_picks
                or plan["blocked_grid_points"] != 20-expected_picks
                or plan["sent_points"] != expected_picks):
            raise RuntimeError("Actual pick planning diagnostics do not match the synthetic grid.")
        if (picker.root is not root or int(root.winfo_id()) != panel_handle
                or root.state() != "normal" or not root.winfo_viewable()
                or not root.tk.getboolean(root.attributes("-topmost"))):
            raise RuntimeError("The control panel disappeared during repeated selections.")
        if expected_remaining:
            if picker.input_surface is None or not picker.input_surface.visible:
                raise RuntimeError("The next fresh contour is not selectable.")
            if (picker.overlay.window is previous_annotation
                    or picker.overlay.canvas is previous_canvas
                    or previous_annotation.winfo_exists() or previous_canvas.winfo_exists()):
                raise RuntimeError("Semi-auto remapped the previous contour's painted window.")
            retired_windows.append(previous_annotation)
        elif (picker.overlay.visible or picker.overlay.canvas.find_all()
                or picker.overlay._mask_photo is not None or picker.input_surface is not None):
            raise RuntimeError("The final contour left a visible or uncleared annotation surface.")

    try:
        picker._result = record_result
        picker.desktop, picker.hotkeys, picker.mouse_observer = desktop, Hooks(), Hooks()
        picker.geometry, picker.target = geometry, 12345
        picker.rgb = desktop.rgb.copy()
        picker.reference = desktop.rgb.copy()
        picker.frame_id += 1
        picker.candidates = candidates + [exhausted]
        picker.picked_points.clear()
        picker.spacing.set(20)
        picker.click_delay.set(.02)
        picker.disabled.set(False)
        picker._present_masks()
        root.update()
        for index, candidate in enumerate(candidates):
            select_and_check(candidate, candidates[index+1:], 20)
            completed.append(len(desktop.clicks))
            if index == 0:
                if exhausted.id not in attempts[-1]["exhausted_ids"]:
                    raise RuntimeError("The covered overlapping proposal was not retired after the first pick.")
                desktop.paint_delayed_markers()
                delayed_fraction = screen_changed_fraction(picker.reference, desktop.rgb)
                if delayed_fraction <= .03:
                    raise RuntimeError("Delayed-marker fixture did not cross the former drift threshold.")
            if index == 4:
                # A synthetic recognition result deliberately reintroduces a
                # covered proposal after five jobs. This exercises the empty
                # plan path as well as proactive retirement of initial overlaps.
                remembered = set(picker.picked_points)
                previous_annotation = picker.overlay.window
                picker._result("recognition", {"rgb": desktop.rgb.copy(),
                               "candidates": [exhausted] + candidates[index+1:],
                               "auto": False, "prompt": False})
                root.update()
                if picker.picked_points != remembered or previous_annotation.winfo_exists():
                    raise RuntimeError("Re-detection lost actual picks or reused its previous annotation surface.")
                retired_windows.append(previous_annotation)
                select_and_check(exhausted, candidates[index+1:], 0)
                if picker.picked_points != remembered or "already" not in picker.status.get().lower():
                    raise RuntimeError("The zero-plan selection did not explain coverage or preserve actual picks.")
        if len(desktop.clicks) != len(set(desktop.clicks)) or not desktop.clicks:
            raise RuntimeError("Repeated contours produced duplicate or empty test picks.")
        if desktop.thread_ids == {threading.get_ident()} or len(desktop.thread_ids) != 1:
            raise RuntimeError("Picking did not use the serialized background worker.")
        if len(attempts) != 9 or completed != list(range(20, 161, 20)):
            raise RuntimeError("Eight fresh contours did not complete across the zero-plan detour.")
        if "next micrograph" not in picker.status.get().lower() or "click another green" in picker.status.get().lower():
            raise RuntimeError("The completed scene still offers a nonexistent green contour.")
        return {"contours_completed": len(completed), "cumulative_pick_counts": completed,
                "selection_attempts": len(attempts), "zero_plan_attempts": 1,
                "exhausted_proposals_retired": True,
                "zero_plan_did_not_block_next_selection": True,
                "final_completion_status_clear": True,
                "attempts": attempts, "final_status": picker.status.get(),
                "delayed_marker_changed_fraction": delayed_fraction,
                "separate_worker_jobs": True, "real_tk_input_events": True,
                "remaining_contours_selectable": True, "panel_visible_throughout": True,
                "panel_identity_preserved": True,
                "fresh_annotation_surface_after_selection": True,
                "retired_annotation_windows_destroyed": len(retired_windows),
                "completed_contours_absent_from_bitmaps": True,
                "final_annotation_cleared": True,
                "annotation_presentations": annotation_presentations,
                "unique_test_picks": len(desktop.clicks), "desktop_events_sent": 0,
                "note": "Real controller/worker/Windows overlays; in-memory target and click sink, not cryoSPARC. Checks window retirement and actual Tk bitmap contents, not every compositor frame."}
    finally:
        if previous_result_override is None:
            picker.__dict__.pop("_result", None)
        else:
            picker._result = previous_result_override
        picker.stop()
        if not picker.runner.shutdown(wait=True, timeout=5):
            raise RuntimeError("Repeated-contour test worker did not terminate.")
        picker.runner.poll_results()
        picker.runner = TaskRunner()
        picker.desktop, picker.hotkeys, picker.mouse_observer = (previous_desktop,
                                                               previous_hotkeys, previous_mouse)
        picker.set_mode(previous_mode)


def _smoke_micrograph_cycle(picker) -> dict[str, Any]:
    """Drive actual UI/worker navigation against a synthetic micrograph list.

    Only our own Tk button is invoked. Keyboard and picking effects terminate
    in the in-memory adapter, so this check cannot operate another application.
    The real screen controller and changed/stable-frame waiter remain in use.
    """
    import threading
    from screen_app import screen_background_guard, screen_background_changed

    root = picker.root
    previous_mode = picker.mode
    previous_desktop, previous_hotkeys, previous_mouse = (picker.desktop, picker.hotkeys,
                                                         picker.mouse_observer)
    previous_model_override = picker.__dict__.get("_get_recognizer")
    picker.set_mode("semi-auto")
    _smoke_settle_panel(picker)
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    occupied = control_panel_bounds(root)
    geometry = next((CaptureGeometry(x, y, 240, 160)
                     for x, y in ((width-260, 30), (width-260, height-190),
                                  (20, height-190), (20, 30))
                     if x >= 0 and y >= 0
                     and not screen_region_overlaps_panel(CaptureGeometry(x, y, 240, 160), occupied)), None)
    if geometry is None:
        raise RuntimeError("No free screen area for the micrograph-cycle UI check.")

    class Hooks:
        def enable_space_pause(self, enabled):
            pass
        def stop(self):
            pass

    class MemoryDesktop:
        def __init__(self):
            self.original_frames = [np.repeat(np.random.default_rng(210+index).integers(
                60, 190, (geometry.height, geometry.width), dtype=np.uint8)[..., None], 3, axis=2)
                for index in range(4)]
            self.reset()
        def reset(self, index=0):
            self.frames = [frame.copy() for frame in self.original_frames]
            self.index = index
            self.keys, self.clicks, self.thread_ids = [], [], set()
            self.paint_auto_markers = False
            self.marked_frames, self.late_marked_frames = set(), set()
            self.marker_updates, self.marker_guard_differences = 0, []
        def _paint_marker_reflow(self, *, late=False):
            frame = self.frames[self.index]
            before = frame.copy()
            if late:
                # Paint after playback's final capture but before Down. This
                # deliberately changes pixels outside the known pick margins.
                frame[:16] = (245, 15, 25)
                self.late_marked_frames.add(self.index)
            else:
                # A wide repaint makes the previous background check fail,
                # even though the synthetic target and all pick positions stay
                # fixed. Alternating colours exercises repeated marker redraws.
                frame[-24:] = (15, 245, 25) if self.marker_updates % 2 else (245, 25, 15)
                self.marker_updates += 1
                self.marked_frames.add(self.index)
            guard = screen_background_guard([candidates[self.index], extra_candidates[self.index]],
                                            geometry, 20)
            if guard is None:
                raise RuntimeError("Synthetic marker fixture lost its navigation background.")
            self.marker_guard_differences.append(screen_background_changed(before, frame, guard))
        def focus_target(self, target):
            self.ensure_target(target)
        def ensure_target(self, target):
            if target != 12345:
                raise RuntimeError("Unexpected synthetic navigation target.")
        def window_under_roi(self, region):
            if region != geometry:
                raise RuntimeError("Unexpected synthetic capture area.")
            return 12345
        def _layers_hidden(self):
            if ((picker.overlay is not None and picker.overlay.visible)
                    or (picker.input_surface is not None and picker.input_surface.visible)):
                raise RuntimeError("A screen layer remained visible during navigation/capture.")
        def screenshot(self, region):
            self.window_under_roi(region)
            self._layers_hidden()
            if (self.paint_auto_markers and picker.job_kind == "next frame"
                    and self.index in self.marked_frames and self.index not in self.late_marked_frames):
                self._paint_marker_reflow(late=True)
            return self.frames[self.index].copy()
        def click(self, x, y):
            self._layers_hidden()
            if not geometry.contains(x, y):
                raise RuntimeError("A generated cycle test pick escaped the selected area.")
            self.thread_ids.add(threading.get_ident())
            self.clicks.append((self.index, x, y))
            if self.paint_auto_markers:
                self._paint_marker_reflow()
        def press_next(self, key):
            self._layers_hidden()
            if key != "down":
                raise RuntimeError(f"Unexpected navigation key: {key!r}")
            self.thread_ids.add(threading.get_ident())
            self.keys.append(key)
            self.index = min(self.index + 1, len(self.frames)-1)

    desktop = MemoryDesktop()
    candidates, extra_candidates = [], []
    for index in range(4):
        left = 15 + index * 30
        mask = polygon_mask([(left+9, 30), (left+43, 30), (left+54, 48),
                             (left+49, 94), (left+10, 101), (left, 73)],
                            (geometry.height, geometry.width))
        candidates.append(make_candidate(mask, source="synthetic micrograph-cycle check"))
        extra_mask = polygon_mask([(176, 32), (219, 32), (224, 56),
                                   (216, 102), (175, 99), (168, 66)],
                                  (geometry.height, geometry.width))
        extra_candidates.append(make_candidate(extra_mask, source="synthetic second auto contour"))

    class Recognizer:
        def __init__(self):
            self.calls = []
            self.empty_frames = set()
        def automatic(self, rgb, *, cancel):
            if cancel():
                raise CancelledOperation("Synthetic recognition was cancelled.")
            index = desktop.index
            if not np.array_equal(rgb, desktop.frames[index]):
                raise RuntimeError("Recognition used a stale synthetic micrograph.")
            self.calls.append(index)
            if index in self.empty_frames:
                return []
            return ([candidates[index], extra_candidates[index]] if desktop.paint_auto_markers
                    else [candidates[index]])

    model = Recognizer()

    def panel_visible():
        if (root.state() != "normal" or not root.winfo_viewable()
                or not root.tk.getboolean(root.attributes("-topmost"))):
            raise RuntimeError("Control panel disappeared during the micrograph cycle.")

    def pump_until(predicate, label, *, seconds=20):
        deadline = time.monotonic() + seconds
        while True:
            root.update()
            panel_visible()
            if predicate():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"{label} did not complete: {picker.status.get()}")
            time.sleep(.005)

    def idle():
        return not picker.transition and not picker.runner.busy and not picker.auto_remaining

    def next_button():
        buttons = [child for child in picker.semi_controls.winfo_children()
                   if child.winfo_class() == "TButton" and child.cget("text") == "Next micrograph"]
        if len(buttons) != 1:
            raise RuntimeError("The semi-auto Next micrograph button is missing or ambiguous.")
        return buttons[0]

    def prepare(mode, index=0, *, show_contour=False):
        picker.set_mode(mode)
        _smoke_settle_panel(picker)
        desktop.reset(index)
        model.calls.clear()
        model.empty_frames.clear()
        picker.geometry, picker.target = geometry, 12345
        picker.rgb = desktop.frames[index].copy()
        picker.reference = picker.rgb.copy()
        picker.frame_id += 1
        picker.candidates = [candidates[index]] if show_contour else []
        picker.alternatives = []
        picker.picked_points.clear()
        picker.spacing.set(20)
        picker.click_delay.set(.03)
        picker.wait_seconds.set(2)
        picker.disabled.set(False)
        if show_contour:
            picker._present_masks()
        root.update()

    try:
        picker.desktop, picker.hotkeys, picker.mouse_observer = desktop, Hooks(), Hooks()
        picker._get_recognizer = lambda config: model

        # A real Tk button must advance once, clear the old frame's picks, and
        # detect the next frame without placing any picks automatically.
        prepare("semi-auto", show_contour=True)
        picker.picked_points.add((30, 45))
        next_button().invoke()
        pump_until(idle, "Semi-auto next micrograph")
        if (desktop.keys != ["down"] or desktop.clicks or model.calls != [1]
                or picker.picked_points or picker.disabled.get()
                or [candidate.id for candidate in picker.candidates] != [candidates[1].id]
                or picker.input_surface is None or not picker.input_surface.visible):
            raise RuntimeError(f"Semi-auto next micrograph failed: {picker.status.get()}")
        semi = {"down_keys": len(desktop.keys), "detections": list(model.calls),
                "picks_sent": len(desktop.clicks), "old_pick_history_cleared": True,
                "new_contour_selectable": True, "real_tk_button_invoked": True}

        # Continuous auto must cross a zero-proposal frame and keep going.
        # Stop while the next navigation job is waiting: no stale continuation
        # may detect or pick the fourth frame after this explicit stop.
        prepare("auto")
        desktop.paint_auto_markers = True
        model.empty_frames = {1}
        picker.frames.set(0)
        picker.start_auto()
        pump_until(lambda: len(desktop.keys) >= 3 or idle(), "Continuous micrograph cycle")
        if len(desktop.keys) != 3:
            raise RuntimeError(f"Continuous auto stopped before its third advance: {picker.status.get()}")
        picker.stop(message="Synthetic Stop during navigation.")
        pump_until(idle, "Stop during navigation")
        if model.calls != [0, 1, 2] or desktop.keys != ["down"] * 3:
            raise RuntimeError("Auto navigation skipped a detection or continued after Stop.")
        if {index for index, _, _ in desktop.clicks} != {0, 2}:
            raise RuntimeError("Auto picked an empty or stopped micrograph.")
        if len(desktop.clicks) != len(set(desktop.clicks)) or not picker.disabled.get():
            raise RuntimeError("Auto produced duplicate test picks or did not disable after Stop.")
        if len(desktop.thread_ids) != 1 or threading.get_ident() in desktop.thread_ids:
            raise RuntimeError("Navigation/picking did not use the serialized background worker.")
        expected = []
        for index in (0, 2):
            for batch in screen_model_batches([candidates[index], extra_candidates[index]], geometry, 20):
                expected.extend((index, int(x), int(y)) for x, y in batch.screen)
        if desktop.clicks != expected:
            raise RuntimeError("Marker redraws interrupted or changed the ordered multi-contour auto picks.")
        if (desktop.marked_frames != {0, 2} or desktop.late_marked_frames != {0, 2}
                or desktop.marker_updates != len(expected)
                or not desktop.marker_guard_differences
                or min(desktop.marker_guard_differences) <= .03):
            raise RuntimeError("The auto fixture did not exercise marker changes beyond the former background threshold.")
        auto = {"continuous_setting": 0, "detected_frames": list(model.calls),
                "picked_frames": sorted({index for index, _, _ in desktop.clicks}),
                "empty_frame_skipped": True, "down_keys": len(desktop.keys),
                "unique_test_picks": len(desktop.clicks), "stop_during_navigation": True,
                "no_detection_or_picks_after_stop": True,
                "contours_per_picked_frame": 2,
                "marker_repaints_during_picking": desktop.marker_updates,
                "late_marker_frames_before_down": sorted(desktop.late_marked_frames),
                "minimum_marker_background_change": min(desktop.marker_guard_differences),
                "ordered_picks_unchanged_by_markers": True}

        # The last list item returns the same screenshot after Down. This must
        # stop with no new recognition or picking, rather than re-picking it.
        prepare("semi-auto", index=3, show_contour=True)
        next_button().invoke()
        pump_until(idle, "End-of-list handling")
        if (desktop.keys != ["down"] or model.calls or desktop.clicks
                or not picker.disabled.get() or picker.candidates):
            raise RuntimeError(f"An unchanged micrograph was accepted: {picker.status.get()}")
        unchanged = {"down_keys": 1, "new_detections": 0, "picks_sent": 0,
                     "stopped_before_repeated_picking": True}
        return {"semi_auto_next": semi, "continuous_auto": auto,
                "unchanged_frame": unchanged, "panel_visible_throughout": True,
                "real_worker_and_navigation_waiter": True, "desktop_events_sent": 0,
                "note": "Real Tk/controller/worker; in-memory frame list, recognizer, key and click sinks; not cryoSPARC."}
    finally:
        picker.stop()
        if not picker.runner.shutdown(wait=True, timeout=5):
            raise RuntimeError("Micrograph-cycle test worker did not terminate.")
        picker.runner.poll_results()
        picker.runner = TaskRunner()
        if previous_model_override is None:
            picker.__dict__.pop("_get_recognizer", None)
        else:
            picker._get_recognizer = previous_model_override
        picker.desktop, picker.hotkeys, picker.mouse_observer = (previous_desktop,
                                                               previous_hotkeys, previous_mouse)
        picker.set_mode(previous_mode)


def _smoke_manual_strokes(picker) -> dict[str, Any]:
    """Check fresh manual surfaces and playback without native input events.

    Synthetic hook records enter the real controller queue. Real Tk surfaces
    and the actual worker are exercised, but all generated picks terminate in
    the memory adapter. This verifies window replacement and Canvas contents;
    it is not a recording of every Windows compositor frame.
    """
    import threading

    root = picker.root
    previous_mode = picker.mode
    previous_desktop, previous_hotkeys, previous_mouse = (picker.desktop, picker.hotkeys,
                                                         picker.mouse_observer)
    previous_delay, previous_spacing = float(picker.click_delay.get()), int(picker.spacing.get())
    if previous_delay != .01:
        raise RuntimeError("The new application did not start with the shared 0.01-second click delay.")
    picker.set_mode("manual")
    _smoke_settle_panel(picker)
    panel_identity = int(root.winfo_id())
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    occupied = control_panel_bounds(root)
    geometry = next((CaptureGeometry(x, y, 240, 160)
                     for x, y in ((width-260, 30), (width-260, height-190),
                                  (20, height-190), (20, 30))
                     if x >= 0 and y >= 0
                     and not screen_region_overlaps_panel(CaptureGeometry(x, y, 240, 160), occupied)), None)
    if geometry is None:
        raise RuntimeError("No free screen area for the consecutive-manual-stroke UI check.")

    class Hooks:
        def enable_space_pause(self, enabled):
            pass
        def stop(self):
            pass

    class MemoryDesktop:
        def __init__(self):
            self.clicks, self.thread_ids, self.bound = [], set(), None
        def target_at(self, x, y):
            if not geometry.contains(x, y):
                raise RuntimeError("The synthetic manual press escaped the test area.")
            return 12345
        def is_own_window(self, target):
            return False
        def size(self):
            return width, height
        def bind_target(self, target):
            if target != 12345:
                raise RuntimeError("Unexpected synthetic manual target.")
            self.bound = target
        def click(self, x, y):
            if self.bound != 12345 or not geometry.contains(x, y):
                raise RuntimeError("A synthetic manual pick escaped the bound target.")
            if ((picker.overlay is not None and picker.overlay.visible)
                    or (picker.input_surface is not None and picker.input_surface.visible)):
                raise RuntimeError("A screen layer remained visible during manual playback.")
            self.thread_ids.add(threading.get_ident())
            self.clicks.append((int(x), int(y)))

    desktop = MemoryDesktop()

    def panel_visible():
        if (int(root.winfo_id()) != panel_identity or root.state() != "normal"
                or not root.winfo_viewable()
                or not root.tk.getboolean(root.attributes("-topmost"))):
            raise RuntimeError("The persistent control panel changed or disappeared during a manual stroke.")

    def pump_until(predicate, label, *, seconds=10):
        deadline = time.monotonic() + seconds
        while True:
            root.update()
            panel_visible()
            if predicate():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"{label} did not complete: {picker.status.get()}")
            time.sleep(.005)

    left, top = geometry.left, geometry.top
    strokes = [
        [(left+10, top+20), (left+70, top+20), (left+70, top+80), (left+10, top+80)],
        [(left+130, top+30), (left+210, top+30), (left+210, top+115), (left+130, top+115)],
    ]
    # Independent fixtures for the original spacing=20 lattice: row height 17,
    # odd-row shift 10, excluded top/left and included bottom/right boundaries.
    expected_local = [
        [(20, 37), (40, 37), (60, 37), (30, 54), (50, 54), (70, 54),
         (20, 71), (40, 71), (60, 71)],
        [(140, 47), (160, 47), (180, 47), (200, 47),
         (150, 64), (170, 64), (190, 64), (210, 64),
         (140, 81), (160, 81), (180, 81), (200, 81),
         (150, 98), (170, 98), (190, 98), (210, 98),
         (140, 115), (160, 115), (180, 115), (200, 115)],
    ]
    expected = [[(left+x, top+y) for x, y in points] for points in expected_local]
    surfaces, counts = [], []
    delays = {}
    try:
        picker.desktop, picker.hotkeys, picker.mouse_observer = desktop, Hooks(), Hooks()
        for delay in (previous_delay, 0.0):
            values = {}
            for mode in ("manual", "semi-auto", "auto"):
                picker.set_mode(mode)
                picker.disabled.set(False)
                picker.click_delay.set(delay)
                values[mode] = picker._live_ready()
            if set(values.values()) != {delay}:
                raise RuntimeError("Click pacing is not shared by all picking modes.")
            delays["default" if delay else "zero"] = values
        picker.set_mode("manual")
        picker.disabled.set(False)
        picker.spacing.set(20)
        picker._sync_manual()
        _smoke_settle_panel(picker)
        # Paint an old trace before stroke 1, then hide it, reproducing the
        # surface lifecycle that previously reused a layered backing store.
        overlay = picker.ensure_overlay()
        overlay.draw_trace(strokes[1])
        overlay.show()
        root.update()
        overlay.hide()
        root.update()
        for index, points in enumerate(strokes):
            picker.click_delay.set(previous_delay if index == 0 else 0.0)
            previous_window, previous_canvas = overlay.window, overlay.canvas
            offset = len(desktop.clicks)
            picker._mouse_event(("press", *points[0]))
            pump_until(lambda: picker.drawing, f"Manual stroke {index+1} press")
            if (overlay is not picker.overlay or overlay.window is previous_window
                    or overlay.canvas is previous_canvas or previous_window.winfo_exists()
                    or previous_canvas.winfo_exists() or not overlay.visible
                    or overlay.canvas.find_all()):
                raise RuntimeError(f"Manual stroke {index+1} reused old surface geometry.")
            surfaces.append({"tk_window": str(overlay.window),
                             "native_window": int(overlay.window.winfo_id()),
                             "previous_window_destroyed": True,
                             "initial_canvas_empty": True})
            for point in points[1:]:
                picker._mouse_event(("move", *point))
            coordinates = [float(value) for point in points for value in point]
            pump_until(lambda: list(overlay.canvas.coords("trace")) == coordinates,
                       f"Manual stroke {index+1} drawing")
            if (len(overlay.canvas.find_all()) != 1
                    or len(overlay.canvas.find_withtag("trace")) != 1):
                raise RuntimeError("Manual drawing contains geometry outside the current trace.")
            picker._mouse_event(("release", *points[0]))
            pump_until(lambda: (not picker.drawing and not picker.released
                               and not picker.runner.busy and not picker.transition
                               and picker._manual_gate.is_set()),
                       f"Manual stroke {index+1} playback")
            if desktop.clicks[offset:] != expected[index]:
                raise RuntimeError(f"Manual stroke {index+1} changed the original lattice or pick order.")
            if picker.disabled.get() or overlay.visible or overlay.canvas.find_all():
                raise RuntimeError("Manual playback did not return to a clean, enabled drawing state.")
            counts.append(len(desktop.clicks)-offset)
        if len(desktop.thread_ids) != 1 or threading.get_ident() in desktop.thread_ids:
            raise RuntimeError("Manual picking did not use the serialized background worker.")
        return {"strokes_completed": len(counts), "picks_per_stroke": counts,
                "total_test_picks": len(desktop.clicks), "manual_grid_and_order_unchanged": True,
                "shared_delays_by_mode": delays, "stroke_delays_seconds": [previous_delay, 0.0],
                "fresh_annotation_surface_per_stroke": True, "surfaces": surfaces,
                "only_current_trace_geometry": True, "panel_identity_preserved": True,
                "panel_visible_throughout": True, "real_worker_and_hook_queue": True,
                "desktop_events_sent": 0,
                "note": "Real Tk surfaces/controller/worker with synthetic hook records and an in-memory click sink. "
                        "Verifies fresh native windows and Canvas contents, not every compositor frame or live cryoSPARC."}
    finally:
        picker.stop()
        if not picker.runner.shutdown(wait=True, timeout=5):
            raise RuntimeError("Manual-stroke test worker did not terminate.")
        picker.runner.poll_results()
        picker.runner = TaskRunner()
        picker.desktop, picker.hotkeys, picker.mouse_observer = (previous_desktop,
                                                               previous_hotkeys, previous_mouse)
        picker.click_delay.set(previous_delay)
        picker.spacing.set(previous_spacing)
        picker.set_mode(previous_mode)


def _smoke_windows_work_area() -> tuple[int, int, int, int]:
    """Read the primary monitor's usable rectangle, excluding its taskbar."""
    import ctypes
    from ctypes import wintypes

    query = ctypes.windll.user32.SystemParametersInfoW
    query.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT]
    query.restype = wintypes.BOOL
    rectangle = wintypes.RECT()
    if not query(0x0030, 0, ctypes.byref(rectangle), 0):  # SPI_GETWORKAREA
        raise RuntimeError("Could not read the Windows work area for native UI checks.")
    bounds = tuple(int(value) for value in
                   (rectangle.left, rectangle.top, rectangle.right, rectangle.bottom))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise RuntimeError(f"Windows returned an invalid work area: {bounds}.")
    return bounds


def _smoke_position_fixture(window, work_area) -> dict[str, Any]:
    """Place a mapped fixture once without size, focus or z-order changes.

    Fresh Tk roots and settings windows otherwise use Windows' cascaded
    placement. That moving origin can put a later, larger scale off-screen.
    A Tk geometry request made before mapping can be superseded by the native
    wrapper's initial placement. The caller first maps and pumps the window;
    use its actual outer HWND before the first layout/capture validation.
    Never reposition in response to a failed check or between capture frames.
    """
    import ctypes
    from ctypes import wintypes

    if window.state() != "normal" or not window.winfo_viewable():
        raise RuntimeError("Native UI fixture must already be mapped before placement.")
    user32 = ctypes.windll.user32
    ancestor = user32.GetAncestor
    ancestor.argtypes, ancestor.restype = [wintypes.HWND, wintypes.UINT], wintypes.HWND
    position = user32.SetWindowPos
    position.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, wintypes.UINT]
    position.restype = wintypes.BOOL
    handle = int(ancestor(int(window.winfo_id()), 2) or 0)  # GA_ROOT: native outer wrapper.
    if not handle:
        raise RuntimeError("Native UI fixture has no outer Windows handle.")
    before = control_panel_bounds(window)
    left, top = work_area[0] + 16, work_area[1] + 16
    flags = 0x0001 | 0x0004 | 0x0010  # NOSIZE | NOZORDER | NOACTIVATE.
    if not position(handle, None, left, top, 0, 0, flags):
        raise RuntimeError("Could not position the native UI fixture.")
    window.update()  # Dispatch this one move's Configure event before validation.
    after = control_panel_bounds(window)
    if after[:2] != (left, top) or after[2:] != before[2:]:
        raise RuntimeError(f"Native UI fixture placement did not preserve its requested origin/size: "
                           f"before={before}, after={after}, requested_origin={(left, top)}.")
    return {"method": "SetWindowPos", "requested_outer_origin": [left, top],
            "native_bounds_before": list(before), "native_bounds_after": list(after),
            "flags": flags, "size_unchanged": True, "mapped_before_placement": True}


def _smoke_assert_work_area(window, work_area) -> dict[str, Any]:
    """Require the complete native frame to fit without moving or resizing it."""
    left, top, width, height = control_panel_bounds(window)
    frame = (left, top, left + width, top + height)
    if (width <= 0 or height <= 0 or left < work_area[0] or top < work_area[1]
            or frame[2] > work_area[2] or frame[3] > work_area[3]):
        raise RuntimeError(f"Native UI fixture is outside the Windows work area: "
                           f"frame={frame}, work_area={tuple(work_area)}. "
                           "The smoke check will not resize or reposition it to pass.")
    return {"native_outer_bbox": list(frame), "work_area_bbox": list(work_area),
            "complete_frame_inside_work_area": True}


def _smoke_advanced_settings(picker, percent, screenshot_dir=None, *, work_area=None):
    """Check the actual native size control and settings close callback."""
    if work_area is None:
        work_area = _smoke_windows_work_area()
    picker.open_advanced()
    window = picker.advanced
    if window is None:
        raise RuntimeError("Advanced settings did not open.")
    picker.root.update()
    if not window.winfo_viewable():
        raise RuntimeError("Advanced settings did not open.")
    fixture_position = _smoke_position_fixture(window, work_area)
    screen_bounds = _smoke_assert_work_area(window, work_area)
    width, height = window.winfo_width(), window.winfo_height()
    if width < window.winfo_reqwidth() or height < window.winfo_reqheight():
        raise RuntimeError(f"Advanced settings controls do not fit their window at {percent}%: "
                           f"actual={width}x{height}, requested={window.winfo_reqwidth()}x{window.winfo_reqheight()}")
    widgets, pending = [], list(window.winfo_children())
    while pending:
        widget = pending.pop()
        widgets.append(widget)
        pending.extend(widget.winfo_children())
    for widget in widgets:
        if not widget.winfo_ismapped():
            raise RuntimeError("An Advanced settings control is not mapped.")
        x, y = widget.winfo_rootx()-window.winfo_rootx(), widget.winfo_rooty()-window.winfo_rooty()
        if x < 0 or y < 0 or x+widget.winfo_width() > width or y+widget.winfo_height() > height:
            raise RuntimeError("An Advanced settings control is clipped.")
    controls = [w for w in widgets if w.winfo_class() == 'TSpinbox'
                and str(w.cget('textvariable')) == str(picker.min_capsid_area)]
    done = [w for w in widgets if w.winfo_class() == 'TButton' and str(w.cget('text')) == 'Done']
    if len(controls) != 1 or len(done) != 1 or picker._model_config().min_capsid_area_px != 3500:
        raise RuntimeError("The default minimum capsid area is not exposed correctly.")
    record = {'status':'ok', 'scale_percent':percent, 'window_size':[width,height],
              'default_minimum_area_px':3500, 'all_controls_mapped_and_inside_window':True,
              'fixture_position':fixture_position, 'screen_bounds':screen_bounds}
    if screenshot_dir is not None:
        record['screenshot'] = _smoke_panel_screenshot(
            window, Path(screenshot_dir)/f'settings-{percent}.png')
    picker.min_capsid_area.set('5000')
    done[0].invoke()
    picker.root.update()
    if picker.advanced is not None or picker._model_config().min_capsid_area_px != 5000:
        raise RuntimeError("Advanced size-setting edit did not close and apply.")
    picker.min_capsid_area.set('3500')
    record['edit_and_done_callback_verified'] = True
    return record


def _smoke_windows_runtime(log_path: Path, *, screenshot_dir: Path | None = None) -> dict[str, Any]:
    # These are native Windows widgets at simulated Tk font/layout scales. The
    # Windows monitor's DPI setting is never changed. Each fresh interpreter is
    # scaled before widgets/fonts exist, because changing Tk scaling afterwards
    # does not reliably resize previously constructed controls.
    enable_dpi_awareness()
    configure_application_identity()
    work_area = _smoke_windows_work_area()
    tkinter = importlib.import_module("tkinter")
    scale_checks = []
    panel_layouts = {}
    overlay_checks = None
    repeated_contours = None
    micrograph_cycle = None
    manual_strokes = None
    icon_checks = None
    tk_version = None
    long_status = ("Recognition is running on the selected screen area.\n"
                   "Escape stops picking; F8 pauses or resumes.\n"
                   "Keep cryoSPARC's image position and zoom fixed.")
    for percent in (100, 125, 150, 200):
        window = tkinter.Tk()
        picker = None
        original_scaling = None
        try:
            window.withdraw()
            original_scaling = float(window.tk.call("tk", "scaling"))
            requested_dpi = 96 * percent / 100
            window.tk.call("tk", "scaling", requested_dpi / 72)
            actual_scaling = float(window.tk.call("tk", "scaling"))
            args = argparse.Namespace(checkpoint=None, device="cpu", model_type="vit_b", image=None)
            picker = ScreenPickerApp(window, args, log_path, resolve_checkpoint(), start_listeners=False)
            if (picker.mode != "manual" or not picker.disabled.get()
                    or picker.hotkeys is not None or picker.mouse_observer is not None
                    or picker.overlay is not None):
                raise RuntimeError("The compact screen application did not start in passive manual mode.")
            window.state("normal")
            _smoke_settle_panel(picker)
            fixture_position = _smoke_position_fixture(window, work_area)
            layouts = {}
            for mode in ("manual", "semi-auto", "auto"):
                if picker.mode != mode:
                    picker.set_mode(mode)
                initial_status = picker.status.get()
                normal = _smoke_panel_layout(picker, mode, "default")
                normal["screen_bounds"] = _smoke_assert_work_area(window, work_area)
                if screenshot_dir is not None:
                    normal["screenshot"] = _smoke_panel_screenshot(
                        window, Path(screenshot_dir) / f"panel-{percent}-{mode}.png")
                picker.status.set(long_status)
                multiline = _smoke_panel_layout(picker, mode, "multiline_status")
                multiline["screen_bounds"] = _smoke_assert_work_area(window, work_area)
                picker._hide_overlays()
                _smoke_settle_panel(picker)
                if (window.state() != "normal" or not window.winfo_viewable()
                        or not window.tk.getboolean(window.attributes("-topmost"))):
                    raise RuntimeError(f"The {mode} control panel disappeared while hiding the screen overlays; "
                                       f"layout={_smoke_layout_diagnostics(window)}")
                layouts[mode] = {"default": normal, "multiline_status": multiline,
                                 "panel_persists_when_overlays_hidden": True}
                if percent == 100:
                    panel_layouts[mode] = normal["panel_size"]
                picker.status.set(initial_status)
                _smoke_settle_panel(picker)
            settings_check = _smoke_advanced_settings(picker, percent, screenshot_dir,
                                                       work_area=work_area)
            tk_version = str(window.tk.call("info", "patchlevel"))
            scale_checks.append({"scale_percent": percent, "requested_dpi": requested_dpi,
                                 "native_tk_scaling_before_test": original_scaling,
                                 "actual_tk_pixels_per_point": actual_scaling, "layouts": layouts,
                                 "fixture_position": fixture_position,
                                 "advanced_settings":settings_check})
            if overlay_checks is None:
                overlay_checks = verify_screen_overlay(window)
                icon_checks = windows_application_icon_status(window)
                if not all(icon_checks.values()) or not all(path.is_file() for path in application_icon_paths()):
                    raise RuntimeError(f"The native application icon or bundled artwork is missing: "
                                       f"native={icon_checks}, assets="
                                       f"{[(str(path), path.is_file()) for path in application_icon_paths()]}")
                manual_strokes = _smoke_manual_strokes(picker)
                repeated_contours = _smoke_repeated_screen_picks(picker)
                micrograph_cycle = _smoke_micrograph_cycle(picker)
        finally:
            if original_scaling is not None:
                window.tk.call("tk", "scaling", original_scaling)
            if picker is not None:
                # Each simulated scale owns a fresh Tcl interpreter. Finish
                # its idle worker before creating the next root; production
                # close remains asynchronous while native work is active.
                runner = getattr(picker, "runner", None)
                if runner is not None and not runner.shutdown(wait=True, timeout=5):
                    raise RuntimeError("Native UI check worker did not stop before interpreter teardown.")
                picker.close()
            else:
                window.destroy()
    pyautogui = importlib.import_module("pyautogui")
    keyboard = importlib.import_module("pynput.keyboard")
    mouse = importlib.import_module("pynput.mouse")
    if not callable(pyautogui.click) or keyboard.Key.esc is None or mouse.Button.left is None:
        raise RuntimeError("Desktop dependencies are incomplete.")
    return {"tk_version": tk_version, "desktop_modules_imported": True,
            "full_application_constructed": True, "synthetic_mask_rendered": True,
            "compact_manual_default": True, "panel_layouts": panel_layouts,
            "fixture_work_area_bbox": list(work_area),
            "fixture_placement_method": "One native SetWindowPos after initial mapping, before layout validation; NOSIZE|NOZORDER|NOACTIVATE. Complete native frames must fit the Windows work area. No resize or retry-to-pass.",
            "layout_scaling_method": "Native Windows Tk widgets; pixels-per-point set before construction to simulate 100/125/150/200% of 96 DPI. Windows monitor scaling is unchanged.",
            "layout_scale_checks": scale_checks,
            "screen_overlay": overlay_checks,
            "application_icon": icon_checks,
            "repeated_semi_auto_contours": repeated_contours,
            "micrograph_navigation_cycle": micrograph_cycle,
            "consecutive_manual_strokes": manual_strokes,
            "desktop_events_sent": 0, "global_listeners_started": 0}


def _smoke_sam(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"The bundled SAM model is missing: {checkpoint}")
    recognizer = SamRecognizer(RecognitionConfig(str(checkpoint), model_type="vit_b", device="cpu"))
    try:
        yy, xx = np.mgrid[:96, :128]
        grey = (70 + (xx * 3 + yy * 5) % 80).astype(np.uint8)
        grey[((xx - 64) / 30) ** 2 + ((yy - 48) / 24) ** 2 <= 1] = 210
        rgb = np.repeat(grey[..., None], 3, axis=2)
        candidates = recognizer.prompt(rgb, box=(30, 20, 99, 77))
        if any(candidate.mask.shape != rgb.shape[:2] for candidate in candidates):
            raise RuntimeError("SAM returned an incorrectly sized mask.")
        return {"device": recognizer.device, "model_type": "vit_b",
                "checkpoint_bytes": checkpoint.stat().st_size,
                "input_shape": list(rgb.shape), "candidate_count": len(candidates),
                "note": "Synthetic inference check only; not a recognition accuracy test."}
    finally:
        recognizer.close()


def _smoke_capsid_profile(checkpoint: Path) -> dict[str, Any]:
    """Resolve the production sidecar and execute its actual numeric model.

    Public builds normally have no learned profile and retain the existing
    detector. The installer integration check supplies a nonprivate constant
    model whose expected output can be verified independently after install.
    """
    path = resolve_profile_path(checkpoint)
    if path is None:
        return {"profile_status": "missing", "fallback_used": True,
                "profile_inference_executed": False,
                "note": "No optional capsid profile; the existing capsid heuristic remains available."}
    profile = load_profile(path)
    yy, xx = np.mgrid[:96, :128]
    grey = (70 + (xx * 3 + yy * 5) % 80).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    scores = score_image(rgb, profile)
    if (scores.shape != rgb.shape[:2] or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise RuntimeError("The capsid profile returned an invalid inference map.")
    return {"profile_status": "loaded", "fallback_used": False,
            "profile_inference_executed": True,
            "profile_path": profile.path, "profile_sha256": profile.sha256,
            "format": profile.format,
            "inference": {"input_shape": list(rgb.shape), "score_shape": list(scores.shape),
                          "scored_pixels": int(scores.size), "minimum": float(scores.min()),
                          "maximum": float(scores.max()), "mean": float(scores.mean(dtype=np.float64)),
                          "score_sha256": hashlib.sha256(scores.astype("<f4").tobytes()).hexdigest()},
            "score_is_calibrated_probability": False,
            "note": "Actual profile inference on a synthetic image; not recognition accuracy."}


def run_smoke_test(checkpoint: Path, log_path: Path) -> dict[str, Any]:
    """Noninteractive build validation; deliberately never sends desktop input."""
    report = diagnostic_report(log_path, checkpoint)
    report.update(mode="smoke-test", status="running", checks=[],
                  note="Synthetic packaging validation of native libraries, application startup, and model inference; no recognition accuracy measurement.")
    checks = [("image_processing", _smoke_image_processing),
              ("torchvision_native", _smoke_torchvision)]
    if sys.platform == "win32":
        checks.append(("windows_runtime", lambda: _smoke_windows_runtime(log_path)))
    else:
        report["checks"].append({"name": "windows_runtime", "status": "skipped",
                                  "reason": "Requires the Windows build host."})
    checks.append(("sam_cpu_inference", lambda: _smoke_sam(checkpoint)))
    checks.append(("capsid_profile", lambda: _smoke_capsid_profile(checkpoint)))
    for name, function in checks:
        started = time.perf_counter()
        try:
            details = function()
        except Exception as exc:
            APP_LOG.exception("Smoke-test step failed: %s", name)
            report["checks"].append({"name": name, "status": "error",
                                      "seconds": time.perf_counter() - started,
                                      "error": str(exc), "error_type": type(exc).__name__})
            report.update(status="error", error=f"{name}: {exc}")
            return report
        report["checks"].append({"name": name, "status": "ok",
                                  "seconds": time.perf_counter() - started,
                                  "details": details})
    report["status"] = "ok"
    return report


def cryolo_diagnostic_report(log_path: Path, config: RecognitionConfig) -> dict[str, Any]:
    from cryolo_onnx import resolve_onnx_checkpoint
    explicit_legacy = (Path(config.checkpoint).suffix.lower() == ".h5"
                       and (config.cryolo_python or os.environ.get("SARU_CRYOLO_PYTHON")))
    onnx_checkpoint = None if explicit_legacy else resolve_onnx_checkpoint(config.checkpoint)
    report = diagnostic_report(log_path, Path(config.checkpoint))
    report.update(backend="cryolo", cryolo_config=config.cryolo_config,
                  cryolo_config_exists=bool(config.cryolo_config and Path(config.cryolo_config).is_file()),
                  cryolo_python=config.cryolo_python,
                  inference_runtime=("onnxruntime" if onnx_checkpoint is not None else
                                     "legacy H5" if explicit_legacy else "offline model missing"),
                  effective_checkpoint=str(onnx_checkpoint) if onnx_checkpoint is not None else config.checkpoint,
                  effective_checkpoint_exists=(onnx_checkpoint if onnx_checkpoint is not None else Path(config.checkpoint)).is_file(),
                  cryolo_config_required=bool(explicit_legacy),
                  note=("Metadata inspection only. Bundled crYOLO runs offline with ONNX Runtime and embedded model metadata; use --smoke-test for real inference."
                        if onnx_checkpoint is not None else
                        "Metadata inspection only. Using the explicit legacy H5 runtime override; automatic runtime installation is disabled."
                        if explicit_legacy else
                        "Offline crYOLO model not found. Install the current SARU Picker model bundle or select cryolo-capsids.onnx in Advanced. No Python installation or runtime repair is needed."))
    report["dependencies"] = {name: version for name, version in report["dependencies"].items()
                              if name not in {"torch", "torchvision", "segment-anything"}}
    try:
        report["dependencies"]["onnxruntime"] = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        report["dependencies"]["onnxruntime"] = None
    return report


def run_cryolo_smoke_test(log_path: Path, config: RecognitionConfig) -> dict[str, Any]:
    """Exercise the selected crYOLO detector without sending desktop events."""
    report = cryolo_diagnostic_report(log_path, config)
    report.update(mode="smoke-test", status="running", checks=[])
    recognizer = None
    try:
        _smoke_image_processing()
        report["checks"].append({"name": "image_processing", "status": "ok"})
        recognizer = create_recognizer(config)
        rgb = np.random.default_rng(29).integers(80, 180, (512, 512, 3), dtype=np.uint8)
        rgb[:] = rgb[:, :, :1]
        candidates = recognizer.automatic(rgb)
        if any(candidate.mask.shape != rgb.shape[:2] for candidate in candidates):
            raise RuntimeError("crYOLO returned a contour with the wrong image dimensions.")
        report["checks"].append({"name": "cryolo_inference", "status": "ok",
                                  "candidates": len(candidates), "details": recognizer.last_detection})
        report.update(status="ok", note="Real crYOLO inference on synthetic noise. This checks the installed runtime, not capsid accuracy.")
    except Exception as exc:
        report.update(status="error", error=str(exc), error_type=type(exc).__name__)
        report["checks"].append({"name": "cryolo_inference", "status": "error", "error": str(exc)})
    finally:
        if recognizer is not None:
            recognizer.close()
    return report


def show_startup_error(error: Exception, log_path: Path | None, parent=None) -> None:
    """Display a fatal GUI startup failure without creating another Tk root."""
    message = f"SARU Picker could not start.\n\n{error}"
    if log_path is not None:
        message += f"\n\nDiagnostic log: {log_path}"
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "SARU Picker", 0x10)
        elif parent is not None:
            from tkinter import messagebox
            messagebox.showerror("SARU Picker", message, parent=parent)
        elif sys.stderr is not None:
            print(message, file=sys.stderr)
    except Exception:
        APP_LOG.exception("Could not display the startup failure dialog")


def image_digest(rgb: np.ndarray) -> str:
    rgb = validate_rgb(rgb)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def load_rgb_file(path: Path | str) -> np.ndarray:
    """Load a display-ready image without silently clipping raw intensities."""
    with Image.open(path) as image:
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError("This is a raw 16-bit or floating-point image. Export a contrast-normalized 8-bit image first; raw intensities are not silently clipped.")
        return np.asarray(image.convert("RGB")).copy()


def changed_fraction(before: np.ndarray, after: np.ndarray, tolerance: int = 12) -> float:
    """A conservative screenshot drift guard, not a registration algorithm."""
    a, b = validate_rgb(before), validate_rgb(after)
    if a.shape != b.shape:
        return 1.0
    difference = np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=2)
    return float(np.mean(difference > tolerance))


def selected_grid(candidates: list[Candidate], selected: set[str], shape: tuple[int, int],
                  spacing: int, inset: float, max_points: int) -> np.ndarray:
    masks = [c.mask for c in candidates if c.id in selected]
    if not masks:
        return np.empty((0, 2), dtype=np.int32)
    return sample_hex_grid(union_masks(masks, shape=shape), spacing,
                           inset=inset, max_points=max_points)


def export_review(destination: Path, rgb: np.ndarray, candidates: list[Candidate],
                  selected: set[str], points: np.ndarray, geometry: CaptureGeometry | None,
                  metadata: dict[str, Any] | None = None) -> None:
    """Export aligned image/masks and explicitly labeled local coordinates.

    These coordinates are capture pixels, not physical pixels or cryoSPARC .cs
    coordinates. Saved mask labels are user selections, not verified truth.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image = validate_rgb(rgb)
    accepted = [c for c in candidates if c.id in selected]
    Image.fromarray(image).save(destination / "image.png")
    mask = union_masks([c.mask for c in accepted], shape=image.shape[:2])
    Image.fromarray(mask.astype(np.uint8) * 255).save(destination / "selected_mask.png")
    # Separate bitmaps preserve overlapping instances without an arbitrary label priority.
    np.savez_compressed(destination / "selected_masks.npz",
                        **{f"mask_{i:04d}": c.mask for i, c in enumerate(accepted)})
    with (destination / "points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["capture_x_px", "capture_y_px", "screen_x_px", "screen_y_px"])
        for x, y in points:
            screen = geometry.local_to_screen(int(x), int(y)) if geometry else ("", "")
            writer.writerow([int(x), int(y), *screen])
    details = {
        "version": APP_VERSION,
        "coordinate_system": "captured_image_pixels; no micrograph or physical calibration",
        "source_sha256": image_digest(image),
        "shape_hw": list(image.shape[:2]),
        "capture_region_xywh": list(geometry.rectangle) if geometry else None,
        "selection_status": "user-selected or explicitly accepted proposals; not expert-verified ground truth",
        "point_count": int(len(points)),
        "candidates": [dict(id=c.id, source=c.source, area=c.area,
                            bbox_xywh=list(c.bbox), predicted_iou=c.predicted_iou,
                            stability_score=c.stability_score,
                            selected=c.id in selected) for c in candidates],
        "settings": metadata or {},
    }
    (destination / "review.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


def setup_logging(directory: Path | None = None, *, include_console: bool | None = None) -> Path:
    directory = directory or user_state_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "saru_picker.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    existing = [handler for handler in root_logger.handlers if getattr(handler, "_saru_file", False)]
    for handler in existing:
        if Path(handler.baseFilename) != log_path.resolve():
            root_logger.removeHandler(handler)
            handler.close()
    if not any(getattr(handler, "_saru_file", False) for handler in root_logger.handlers):
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2,
                                      encoding="utf-8")
        handler._saru_file = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root_logger.addHandler(handler)
    if include_console is None:
        include_console = sys.stderr is not None
    for handler in list(root_logger.handlers):
        if getattr(handler, "_saru_console", False) and not include_console:
            root_logger.removeHandler(handler)
            handler.close()
    if include_console and not any(getattr(handler, "_saru_console", False) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler._saru_console = True
        root_logger.addHandler(console_handler)
    return log_path


class PickerApp:
    def __init__(self, root, args, log_path: Path, *, start_hotkeys: bool = True):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.root, self.args, self.log_path = root, args, log_path
        apply_application_icon(root)
        self.runner = TaskRunner()
        self.events: queue.Queue = queue.Queue()
        self.desktop = DesktopAdapter()
        self.hotkeys = None
        self.active_token = None
        self._recognition_progress = None
        self.recognizer = None
        self.recognizer_config = None
        self.closed = False
        self.scheduled: set[str] = set()
        self.rgb: np.ndarray | None = None
        self.playback_reference: np.ndarray | None = None
        self.capture_id = 0
        self.geometry: CaptureGeometry | None = None
        self.target = None
        self.candidates: list[Candidate] = []
        self.selected: set[str] = set()
        self.picked: set[str] = set()
        self.picked_points: set[tuple[int, int]] = set()
        self.prompt_box = None
        self.prompt_points: list[tuple[float, float]] = []
        self.prompt_labels: list[int] = []
        self.alternatives: list[Candidate] = []
        self.alternative_index = 0
        self.drag_start = None
        self.trace: list[tuple[float, float]] = []
        self.display_rect = (0, 0, 1, 1)
        self.last_grid = np.empty((0, 2), np.int32)
        self.batch_remaining = 0
        self.batch_seen: list[np.ndarray] = []
        self.batch_paused = False
        self.playback_pending = False
        self.playback_active = False
        self.job_kind = None
        self.job_started = 0.0
        self.capture_select = False
        self.capture_pending = False
        self.selector = None
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda e: self.stop())
        # Global F8 only, avoiding double toggles from local and global bindings.
        if start_hotkeys:
            try:
                self.hotkeys = GlobalHotkeys(self._hotkey)
                self.hotkeys.start()
            except Exception:
                APP_LOG.exception("Global hotkeys unavailable")
                self.hotkeys = None
                self.status.set("Global hotkeys unavailable: live picking disabled. Offline review remains available.")
        else:
            self.status.set("Offline review: live desktop control is disabled.")
        self.root.after(40, self.poll)
        if args.image:
            self.load_image(Path(args.image))

    def _build_ui(self):
        tk, ttk, root = self.tk, self.ttk, self.root
        is_cryolo = getattr(self.args, "backend", "sam") == "cryolo"
        root.title("SARU Picker 3.0.0 — review masks, then pick")
        root.geometry("1280x900")
        root.minsize(1040, 760)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        controls_shell = ttk.Frame(root)
        controls_shell.grid(row=0, column=0, sticky="ns")
        controls_scroll = ttk.Scrollbar(controls_shell, orient="vertical")
        controls_scroll.pack(side="right", fill="y")
        controls_canvas = tk.Canvas(controls_shell, width=350, highlightthickness=0,
                                   yscrollcommand=controls_scroll.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scroll.configure(command=controls_canvas.yview)
        controls = ttk.Frame(controls_canvas, padding=12)
        controls_canvas.create_window((0, 0), window=controls, anchor="nw", width=350)
        controls.bind("<Configure>", lambda e: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))
        right = ttk.Frame(root, padding=(0, 12, 12, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(controls, text="SARU PICKER 3.0.0", font=("Arial", 17, "bold")).pack(anchor="w")
        ttk.Label(controls, text="Capture • segment • review • pick").pack(anchor="w", pady=(0, 8))
        source = ttk.LabelFrame(controls, text="1. Image", padding=7)
        source.pack(fill="x", pady=4)
        ttk.Button(source, text="Open image…", command=self.open_image).pack(fill="x")
        ttk.Button(source, text="Select screen region (3 s)",
                   command=lambda: self.capture(True)).pack(fill="x", pady=3)
        ttk.Button(source, text="Recapture same region (3 s)",
                   command=lambda: self.capture(False)).pack(fill="x")
        self.source_label = tk.StringVar(value="No image loaded")
        ttk.Label(source, textvariable=self.source_label, wraplength=305).pack(anchor="w", pady=3)
        model = ttk.LabelFrame(controls, text="2. Detection", padding=7)
        model.pack(fill="x", pady=4)
        checkpoint = (resolve_checkpoint(self.args.checkpoint, backend="cryolo") if is_cryolo
                      else resolve_checkpoint(self.args.checkpoint))
        initialize_detector_fields(self, tk, self.args, checkpoint)
        ttk.Label(model, text="Detection profile").pack(anchor="w")
        self.profile_selector = ttk.Combobox(model, textvariable=self.detection_profile,
                                             values=DETECTION_PROFILES, state="readonly", width=24)
        self.profile_selector.pack(fill="x")
        self.profile_selector.bind("<<ComboboxSelected>>", self._profile_draft_changed)
        for label, variable, backend in (("SAM checkpoint", self.sam_checkpoint, "sam"),
                                         ("crYOLO weights", self.cryolo_checkpoint, "cryolo")):
            ttk.Label(model, text=label).pack(anchor="w")
            row = ttk.Frame(model)
            row.pack(fill="x")
            ttk.Entry(row, textvariable=variable, width=29).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="…", width=3, command=lambda backend=backend: self.browse_model(backend)).pack(side="right")
        ttk.Label(model, text="crYOLO config (H5 only)").pack(anchor="w")
        ttk.Entry(model, textvariable=self.cryolo_config, width=36).pack(fill="x")
        row = ttk.Frame(model)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Device").pack(side="left")
        ttk.Combobox(row, textvariable=self.device, values=("auto", "cpu", "cuda"), state="readonly", width=7).pack(side="left", padx=4)
        ttk.Label(row, text="SAM").pack(side="left")
        ttk.Combobox(row, textvariable=self.model_type, values=("vit_b", "vit_l", "vit_h"), state="readonly", width=6).pack(side="left", padx=3)
        for label, variable in (("SAM auto grid / side", self.grid_side),
                                ("crYOLO confidence", self.cryolo_threshold),
                                ("Minimum capsid area (px²)", self.min_capsid_area)):
            row = ttk.Frame(model)
            row.pack(fill="x")
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=variable, width=9).pack(side="right")
        self._applied_model_config = detector_config_from_fields(self)
        ttk.Button(model, text="Apply detection settings", command=self._apply_model_settings).pack(fill="x", pady=5)
        self.tool = tk.StringVar(value="box")
        tools_frame = ttk.LabelFrame(controls, text="3. Draw on preview", padding=7)
        tools_frame.pack(fill="x", pady=4)
        for value, label in (("freehand", "Manual contour"), ("box", "Box around target"),
                             ("positive", "Foreground point (+)"), ("negative", "Background point (−)"),
                             ("select", "Select / deselect mask")):
            ttk.Radiobutton(tools_frame, text=label, value=value, variable=self.tool).pack(anchor="w")
        row = ttk.Frame(tools_frame)
        row.pack(fill="x", pady=(5, 0))
        self.prompt_button = ttk.Button(row, text="Select detections" if is_cryolo else "Segment prompt", command=self.segment_prompt)
        self.prompt_button.pack(side="left")
        ttk.Button(row, text="Clear prompt", command=self.clear_prompt).pack(side="left", padx=4)
        row = ttk.Frame(tools_frame)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Next alternative", command=self.cycle_alternative).pack(side="left")
        ttk.Button(row, text="Accept mask", command=self.accept_alternative).pack(side="left", padx=4)
        ttk.Button(tools_frame, text="Generate automatic proposals", command=self.auto_proposals).pack(fill="x")
        self.detector_note = ttk.Label(tools_frame, text=("Yellow: proposal / preview. Green: selected.\n"
            "SAM profiles segment prompts. crYOLO boxes and points select existing detections; use manual contours for missed targets."),
            wraplength=310)
        self.detector_note.pack(anchor="w", pady=4)
        picking = ttk.LabelFrame(controls, text="4. Sampling and playback", padding=7)
        picking.pack(fill="x", pady=4)
        self.spacing, self.inset = tk.IntVar(value=13), tk.DoubleVar(value=2.0)
        self.click_delay = tk.DoubleVar(value=0.08)
        for label, variable, low, high, step in (("Spacing (capture px)", self.spacing, 2, 100, 1),
                                                ("Boundary inset (px)", self.inset, 0, 50, 1),
                                                ("Click delay (seconds)", self.click_delay, .02, 1, .01)):
            row = ttk.Frame(picking)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Spinbox(row, textvariable=variable, from_=low, to=high,
                        increment=step, width=7, command=self.render).pack(side="right")
        self.live = tk.BooleanVar(value=False)
        ttk.Checkbutton(picking, text="Enable live desktop clicks", variable=self.live,
                        command=self.live_changed).pack(anchor="w", pady=4)
        ttk.Button(picking, text="Pick selected masks", command=self.start_playback).pack(fill="x")
        row = ttk.Frame(picking)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Pause / resume (F8)", command=self.pause).pack(side="left")
        ttk.Button(row, text="Stop (Esc)", command=self.stop).pack(side="right")
        ttk.Button(picking, text="Export image, masks and points…", command=self.export).pack(fill="x")
        batch = ttk.LabelFrame(controls, text="Optional automatic batch", padding=7)
        batch.pack(fill="x", pady=4)
        self.batch_accept = tk.BooleanVar(value=False)
        ttk.Checkbutton(batch, text="Accept all automatic proposals", variable=self.batch_accept).pack(anchor="w")
        self.frame_count, self.wait_seconds = tk.IntVar(value=1), tk.DoubleVar(value=5.0)
        self.next_key = tk.StringVar(value="down")
        row = ttk.Frame(batch)
        row.pack(fill="x")
        ttk.Label(row, text="Frames").pack(side="left")
        ttk.Spinbox(row, from_=1, to=1000, textvariable=self.frame_count, width=4).pack(side="left", padx=4)
        ttk.Label(row, text="Wait (s)").pack(side="left")
        ttk.Spinbox(row, from_=1, to=60, textvariable=self.wait_seconds, width=4).pack(side="left", padx=4)
        ttk.Button(batch, text="Run bounded auto batch", command=self.start_batch).pack(fill="x", pady=3)
        ttk.Label(batch, text="Next-image key: Down. Stops on error, unchanged image, or no candidates.",
                  wraplength=305).pack(anchor="w")
        ttk.Label(right, text="Snapshot preview — drawing here does not click cryoSPARC",
                  font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.canvas = tk.Canvas(right, bg="#1c2430", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<B1-Motion>", self.pointer_move)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.canvas.bind("<ButtonPress-3>", self.background_point)
        self.grid_label = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.grid_label).grid(row=2, column=0, sticky="w", pady=4)
        self.table = ttk.Treeview(right, columns=("use", "source", "area", "quality"),
                                  show="headings", height=5, selectmode="browse")
        for col, label, width in (("use", "Use", 55), ("source", "Source", 200),
                                  ("area", "Area px", 100), ("quality", "Predicted IoU", 120)):
            self.table.heading(col, text=label)
            self.table.column(col, width=width, stretch=col == "source")
        self.table.grid(row=3, column=0, sticky="ew")
        self.table.bind("<ButtonRelease-1>", self.table_toggle)
        row = ttk.Frame(right)
        row.grid(row=4, column=0, sticky="ew", pady=5)
        ttk.Button(row, text="Select all", command=self.select_all).pack(side="left")
        ttk.Button(row, text="Clear selection", command=self.clear_selection).pack(side="left", padx=5)
        ttk.Button(row, text="Delete selected", command=self.delete_selected).pack(side="left")
        self.status = tk.StringVar(value="Open a clean image or capture a screen region to begin.")
        ttk.Label(root, textvariable=self.status, wraplength=1200, padding=(12, 7),
                  relief="sunken").grid(row=1, column=0, columnspan=2, sticky="ew")

    def _hotkey(self, action):
        # Never call Tk from pynput. Cancellation itself is a thread-safe Event.
        if action == "stop" and self.active_token is not None:
            self.active_token.cancel()
        self.events.put(("hotkey", action))

    def _later(self, milliseconds: int, function):
        generation = self.runner.generation
        holder = []
        def wrapped():
            self.scheduled.discard(holder[0])
            if not self.closed and generation == self.runner.generation:
                try:
                    function()
                except Exception as exc:
                    APP_LOG.exception("Delayed operation failed")
                    self.stop()
                    self._error(exc)
        holder.append(self.root.after(milliseconds, wrapped))
        self.scheduled.add(holder[0])

    def _cancel_timers(self):
        for identifier in list(self.scheduled):
            self.root.after_cancel(identifier)
        self.scheduled.clear()

    def _ensure_idle(self, *, allow_model_changes=False):
        if self.runner.busy or self.playback_pending or self.capture_pending or self.batch_remaining or getattr(self, "auto_running", False):
            raise RuntimeError("An operation is active. Stop it and wait for the worker to finish first.")
        if not allow_model_changes and self._model_settings_pending():
            raise RuntimeError("Apply detection settings before detecting or selecting contours.")

    def _error(self, error):
        APP_LOG.error("%s", error)
        self.status.set(str(error))
        self.messagebox.showerror("SARU Picker", f"{error}\n\nDetails: {self.log_path}", parent=self.root)

    def browse_model(self, backend=None):
        backend = backend or resolve_detection_profile(detection_profile_value(self))[0]
        path = self.filedialog.askopenfilename(filetypes=([("crYOLO weights", "*.onnx *.h5")] if backend == "cryolo" else
                                                        [("PyTorch checkpoint", "*.pth *.pt")]) + [("All files", "*")])
        if path:
            (self.cryolo_checkpoint if backend == "cryolo" else self.sam_checkpoint).set(path)
            if backend == "sam":
                for kind in ("vit_b", "vit_l", "vit_h"):
                    if kind in Path(path).name:
                        self.model_type.set(kind)
                        break

    def _model_config(self):
        applied = getattr(self, "_applied_model_config", None)
        return applied if applied is not None else detector_config_from_fields(self)

    def _model_settings_pending(self):
        applied = getattr(self, "_applied_model_config", None)
        if applied is None:
            return False
        try:
            return detector_config_from_fields(self) != applied
        except (ValueError, TypeError, self.tk.TclError):
            return True

    def _profile_draft_changed(self, event=None):
        if self.runner.busy or self.playback_pending or self.capture_pending or self.batch_remaining or getattr(self, "auto_running", False):
            config = self._model_config()
            self.detection_profile.set("crYOLO" if config.backend == "cryolo" else config.automatic_profile)
            self.status.set("Stop the active operation before changing the detection profile.")
        else:
            self.status.set("Choose Apply detection settings to use the selected profile.")

    def _apply_model_settings(self):
        try:
            self._ensure_idle(allow_model_changes=True)
            config = detector_config_from_fields(self)
            persist_detector_fields(self)
            changed = config != getattr(self, "_applied_model_config", None)
            self._applied_model_config = config
            apply_detector_args(self, config)
            if getattr(self, "prompt_button", None) is not None:
                self.prompt_button.configure(text="Select detections" if config.backend == "cryolo" else "Segment prompt")
            if changed:
                # Retire proposals and prompts, retaining actual click history
                # and the same image/coordinate frame for duplicate prevention.
                self.candidates, self.selected, self.alternatives = [], set(), []
                self.prompt_box, self.prompt_points, self.prompt_labels = None, [], []
                self.alternative_index = 0
                self.drag_start, self.trace = None, []
                self.refresh_table()
                self.render()
            self.status.set("Detection settings applied. Generate new proposals for this image.")
            return changed
        except Exception as exc:
            self._error(exc)
            return False

    def _get_recognizer(self, config):
        # Called only by the serialized worker.
        if self.recognizer is None or self.recognizer_config != config:
            if self.recognizer is not None:
                self.recognizer.close()
            self.recognizer = create_recognizer(config) if config.backend == "cryolo" else SamRecognizer(config)
            self.recognizer_config = config
        return self.recognizer

    def _set_image(self, rgb, geometry=None, target=None, description="Image"):
        self.capture_id += 1
        self.rgb = validate_rgb(rgb).copy()
        self.rgb.flags.writeable = False
        self.playback_reference = self.rgb
        self.geometry, self.target = geometry, target
        self.candidates, self.selected, self.picked = [], set(), set()
        self.picked_points.clear()
        self.clear_prompt()
        self.source_label.set(f"{description} — {self.rgb.shape[1]} × {self.rgb.shape[0]} px")
        self.refresh_table()
        self.render()

    def open_image(self):
        try:
            self._ensure_idle()
            path = self.filedialog.askopenfilename(filetypes=[("Images", "*.png *.tif *.tiff *.jpg *.jpeg *.bmp"), ("All files", "*")])
            if path:
                self.load_image(Path(path))
        except Exception as exc:
            self._error(exc)

    def load_image(self, path):
        try:
            self._ensure_idle()
            self._set_image(load_rgb_file(path), description=Path(path).name)
            self.live.set(False)
            self.status.set("Offline image loaded. Use a box or manual contour; live clicks require a fresh screen capture.")
        except Exception as exc:
            self._error(exc)

    def capture(self, select_region=False):
        try:
            self._ensure_idle()
            if not select_region and self.geometry is None:
                raise ValueError("Select a screen region first.")
            self.desktop.size()  # Validate platform/dependencies before hiding UI.
            self.capture_pending = True
            self.status.set("Capturing in 3 seconds. Activate the intended cryoSPARC image.")
            self.root.withdraw()
            self._later(3000, lambda: self._capture_now(select_region))
        except Exception as exc:
            self.capture_pending = False
            self.root.deiconify()
            self._error(exc)

    def _capture_now(self, select_region):
        try:
            if select_region:
                screenshot = self.desktop.screenshot()
                target = self.desktop.capture_target()
                self._select_region(screenshot, target)
            else:
                target = self.desktop.window_under_roi(self.geometry)
                screenshot = self.desktop.screenshot(self.geometry)
                self._set_image(screenshot, self.geometry, target, "Screen capture")
                self.capture_pending = False
                self.root.deiconify()
                self.status.set("Capture ready. Keep cryoSPARC image position and zoom fixed until playback.")
        except Exception as exc:
            self.capture_pending = False
            self.root.deiconify()
            self._error(exc)

    def _select_region(self, screenshot, target):
        tk = self.tk
        window = tk.Toplevel(self.root)
        self.selector = window
        h, w = screenshot.shape[:2]
        window.overrideredirect(True)
        window.geometry(f"{w}x{h}+0+0")
        window.attributes("-topmost", True)
        canvas = tk.Canvas(window, width=w, height=h, highlightthickness=0, cursor="crosshair")
        canvas.pack()
        photo = self.ImageTk.PhotoImage(Image.fromarray(screenshot), master=window)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.image = photo
        canvas.create_rectangle(8, 8, 630, 50, fill="#172236", outline="")
        canvas.create_text(20, 28, anchor="w", fill="white",
                           text="Drag around the micrograph only. Escape cancels.", font=("Arial", 14))
        start = []
        def press(event):
            start[:] = [event.x, event.y]
        def move(event):
            if start:
                canvas.delete("region")
                canvas.create_rectangle(*start, event.x, event.y, outline="#64e1b8", width=3, tags="region")
        def release(event):
            if not start:
                return
            left, right = sorted((max(0, min(w, start[0])), max(0, min(w, event.x))))
            top, bottom = sorted((max(0, min(h, start[1])), max(0, min(h, event.y))))
            if right - left < 32 or bottom - top < 32:
                return
            geometry = CaptureGeometry(left, top, right-left, bottom-top)
            window.destroy()
            self.selector = None
            self.capture_pending = False
            try:
                self.desktop.focus_target(target)
                actual_target = self.desktop.window_under_roi(geometry)
                if actual_target != target:
                    raise RuntimeError("The selected region belongs to another window. Recapture with cryoSPARC active.")
                self._set_image(screenshot[top:bottom, left:right], geometry, target, "Screen capture")
                self.status.set("Capture ready. Select masks and review grid points before enabling live clicks.")
            except Exception as exc:
                self._error(exc)
            finally:
                self.root.deiconify()
        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", move)
        canvas.bind("<ButtonRelease-1>", release)
        window.bind("<Escape>", lambda e: self.stop())
        window.focus_force()

    def _local(self, event):
        if self.rgb is None:
            return None
        ox, oy, width, height = self.display_rect
        geometry = CaptureGeometry(0, 0, self.rgb.shape[1], self.rgb.shape[0])
        try:
            return geometry.viewer_to_local(event.x, event.y, width, height, ox, oy)
        except ValueError:
            return None

    def _editable(self):
        return (self.rgb is not None and not self.runner.busy and not self.playback_pending
                and not self.batch_remaining and not self._model_settings_pending())

    def pointer_down(self, event):
        if not self._editable():
            return
        point = self._local(event)
        if point is None:
            return
        mode = self.tool.get()
        if mode in ("positive", "negative"):
            self.prompt_points.append(point)
            self.prompt_labels.append(1 if mode == "positive" else 0)
            self.alternatives.clear()
        elif mode == "select":
            x, y = map(int, point)
            hits = [c for c in self.candidates if c.mask[y, x] and c.id not in self.picked]
            if hits:
                self._toggle(min(hits, key=lambda c: c.area).id)
        else:
            self.drag_start = point
            self.trace = [point]
        self.render()

    def pointer_move(self, event):
        if self.drag_start is None or not self._editable():
            return
        point = self._local(event)
        if point is None:
            return
        if self.tool.get() == "freehand":
            if np.linalg.norm(np.array(point) - self.trace[-1]) >= 1:
                self.trace.append(point)
        else:
            x0, y0 = self.drag_start
            x1, y1 = point
            self.prompt_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.alternatives.clear()
        self.render()

    def pointer_up(self, event):
        if self.drag_start is None:
            return
        try:
            if self.tool.get() == "freehand" and len(self.trace) >= 3:
                candidate = make_candidate(polygon_mask(self.trace, self.rgb.shape[:2]),
                                           source="manual", image_key=image_digest(self.rgb))
                self._add_candidate(candidate, select=True)
        except Exception as exc:
            self._error(exc)
        finally:
            self.drag_start = None
            self.trace = []
            self.render()

    def background_point(self, event):
        if self._editable():
            point = self._local(event)
            if point is not None:
                self.prompt_points.append(point)
                self.prompt_labels.append(0)
                self.alternatives.clear()
                self.render()

    def clear_prompt(self):
        if self.runner.busy:
            return
        self.prompt_box = None
        self.prompt_points, self.prompt_labels, self.alternatives = [], [], []
        self.drag_start, self.trace = None, []
        self.render()

    def _submit(self, kind, function):
        token = CancellationToken(self.runner.generation)
        if self.batch_paused:
            token.pause()
        def guarded(token):
            token.wait_until_ready()
            return function(token)
        submitted = self.runner.submit(guarded, token=token)
        self.job_kind, self.job_started = kind, time.perf_counter()
        self.active_token = submitted
        if kind in {"prompt", "automatic proposals", "batch recognition"}:
            begin_recognition_progress(self, submitted, self.job_started)
        else:
            clear_recognition_progress(self)
        self.status.set(f"{kind}: running. Esc cancels; an active model forward pass may need to finish.")
        refresh_recognition_progress(self, self.batch_paused)

    def segment_prompt(self):
        try:
            self._ensure_idle()
            if self.rgb is None or (self.prompt_box is None and not any(self.prompt_labels)):
                raise ValueError("Draw a box or add a foreground point first.")
            config, rgb = self._model_config(), self.rgb
            box = self.prompt_box
            points, labels = tuple(self.prompt_points), tuple(self.prompt_labels)
            def work(token):
                options = {"cancel": lambda: token.cancelled}
                if config.backend == "cryolo":
                    options["progress"] = lambda phase: queue_recognition_progress(self, token, phase)
                else:
                    queue_recognition_progress(self, token, "Running SAM")
                return self._get_recognizer(config).prompt(
                    rgb, box=box, points=points or None, labels=labels or None, **options)
            self._submit("prompt", work)
        except Exception as exc:
            self._error(exc)

    def cycle_alternative(self):
        if self.alternatives and not self.runner.busy:
            self.alternative_index = (self.alternative_index + 1) % len(self.alternatives)
            self.render()

    def accept_alternative(self):
        if self.alternatives and self._editable():
            self._add_candidate(self.alternatives[self.alternative_index], select=True)
            self.clear_prompt()
            self.status.set("Mask accepted. Draw the next target or review the sampling points.")

    def auto_proposals(self):
        try:
            self._ensure_idle()
            self._start_auto_detection(False)
        except Exception as exc:
            self._error(exc)

    def _start_auto_detection(self, batch):
        if self.rgb is None:
            raise ValueError("Capture or open an image first.")
        self.alternatives.clear()
        config, rgb = self._model_config(), self.rgb
        def work(token):
            options = {"cancel": lambda: token.cancelled}
            if config.backend == "cryolo":
                options["progress"] = lambda phase: queue_recognition_progress(self, token, phase)
            else:
                queue_recognition_progress(self, token, "Running SAM")
            return self._get_recognizer(config).automatic(rgb, **options)
        self._submit("batch recognition" if batch else "automatic proposals", work)

    def _add_candidate(self, candidate, select=False):
        if candidate.id not in {c.id for c in self.candidates}:
            self.candidates.append(candidate)
        if select and candidate.id not in self.picked:
            self.selected.add(candidate.id)
        self.refresh_table()
        self.render()

    def _toggle(self, candidate_id):
        if candidate_id in self.picked:
            return
        if candidate_id in self.selected:
            self.selected.remove(candidate_id)
        else:
            self.selected.add(candidate_id)
        self.refresh_table()
        self.render()

    def table_toggle(self, event):
        if self._editable():
            candidate_id = self.table.identify_row(event.y)
            if candidate_id:
                self._toggle(candidate_id)

    def select_all(self):
        if self._editable():
            self.selected = {c.id for c in self.candidates} - self.picked
            self.refresh_table()
            self.render()

    def clear_selection(self):
        if self._editable():
            self.selected.clear()
            self.refresh_table()
            self.render()

    def delete_selected(self):
        if self._editable():
            self.candidates = [c for c in self.candidates if c.id not in self.selected]
            self.selected.clear()
            self.refresh_table()
            self.render()

    def refresh_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        for candidate in self.candidates:
            quality = "—" if candidate.predicted_iou is None else f"{candidate.predicted_iou:.3f}"
            used = "picked" if candidate.id in self.picked else ("yes" if candidate.id in self.selected else "")
            self.table.insert("", "end", iid=candidate.id,
                              values=(used, candidate.source, candidate.area, quality))

    def _grid(self):
        if self.rgb is None:
            return np.empty((0, 2), np.int32)
        return selected_grid(self.candidates, self.selected, self.rgb.shape[:2],
                             int(self.spacing.get()), float(self.inset.get()), 10000)

    def render(self):
        if not hasattr(self, "canvas") or self.closed:
            return
        self.canvas.delete("all")
        if self.rgb is None:
            self.canvas.create_text(24, 30, anchor="nw", fill="#e2e8f0", text="Open an image to review segmentation.")
            return
        image = self.rgb.copy()
        for candidate in self.candidates:
            color = (80, 140, 235) if candidate.id in self.picked else ((52, 220, 154) if candidate.id in self.selected else (245, 193, 67))
            alpha = .28 if candidate.id in self.selected else .12
            image[candidate.mask] = (image[candidate.mask] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
        preview = self.alternatives[self.alternative_index] if self.alternatives else None
        if preview is not None:
            image[preview.mask] = (image[preview.mask] * .62 + np.array((255, 191, 45)) * .38).astype(np.uint8)
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        try:
            self.last_grid = self._grid()
            visible_grid = self.last_grid[:3000]
            for x, y in visible_grid:
                draw.ellipse((int(x)-1, int(y)-1, int(x)+1, int(y)+1), fill=(110, 255, 215))
            label = f"{len(self.selected)} selected masks • {len(self.last_grid)} grid points • capture pixels"
            if len(self.last_grid) > 3000:
                label += " • displaying first 3,000 points"
            if preview is not None:
                label += f" • alternative {self.alternative_index+1}/{len(self.alternatives)}"
                if preview.predicted_iou is not None:
                    label += f", predicted IoU {preview.predicted_iou:.3f}"
            self.grid_label.set(label)
        except (ValueError, self.tk.TclError) as exc:
            self.last_grid = np.empty((0, 2), np.int32)
            self.grid_label.set(str(exc))
        if self.prompt_box:
            draw.rectangle(self.prompt_box, outline=(0, 230, 255), width=2)
        if len(self.trace) > 1:
            draw.line(self.trace, fill=(255, 85, 115), width=2)
        for (x, y), label in zip(self.prompt_points, self.prompt_labels):
            color = (20, 255, 125) if label else (255, 75, 95)
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline="white")
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(width/pil.width, height/pil.height)
        dw, dh = max(1, int(pil.width*scale)), max(1, int(pil.height*scale))
        ox, oy = (width-dw)//2, (height-dh)//2
        self.display_rect = (ox, oy, dw, dh)
        self.photo = self.ImageTk.PhotoImage(pil.resize((dw, dh), Image.Resampling.BILINEAR), master=self.root)
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")

    def live_changed(self):
        if not self.live.get():
            self.stop()

    def _validate_live(self):
        if not self.live.get():
            raise ValueError("Enable live desktop clicks to start playback.")
        if self.geometry is None or self.target is None:
            raise ValueError("Live picking requires a fresh screen capture, not an opened image file.")
        if self.hotkeys is None:
            raise ValueError("Global hotkeys are unavailable. Install pynput and restart before live playback.")
        delay = float(self.click_delay.get())
        if not .02 <= delay <= 2:
            raise ValueError("Click delay must be between 0.02 and 2 seconds.")
        return delay

    def start_playback(self, batch=False):
        try:
            if not batch:
                self._ensure_idle()
            delay = self._validate_live()
            points = self._grid()
            points = np.asarray([tuple(p) for p in points if tuple(map(int, p)) not in self.picked_points], dtype=np.int32).reshape(-1, 2)
            if not len(points):
                raise ValueError("No unpicked points. Select masks, reduce inset, or capture a new image.")
            local_points = points.copy()
            screen_points = tuple(self.geometry.local_to_screen(int(x), int(y)) for x, y in points)
            selected_ids, reference, geometry, target = set(self.selected), self.playback_reference, self.geometry, self.target
            capture_id = self.capture_id
            self.playback_pending = True
            self.root.withdraw()
            def begin():
                self.playback_pending = False
                self.playback_active = True
                def work(token):
                    token.wait_until_ready()
                    self.desktop.focus_target(target)
                    token.sleep(.20)
                    current = self.desktop.screenshot(geometry)
                    if changed_fraction(reference, current) > .03:
                        raise RuntimeError("The captured image changed before playback. Recapture to prevent misplaced picks.")
                    def click(x, y):
                        self.desktop.ensure_target(target)
                        self.desktop.click(x, y)
                    def progress(done, total):
                        # Remember each completed point even when cancellation follows.
                        self.events.put(("clicked", token.generation, tuple(map(int, local_points[done-1])), done, total, capture_id))
                    count = play_clicks(screen_points, click, geometry, token,
                                        delay=delay, max_clicks=10000, progress=progress)
                    return {"count": count, "selected": selected_ids, "batch": batch}
                self._submit("batch playback" if batch else "playback", work)
            self._later(350, begin)
        except Exception as exc:
            self.playback_pending = self.playback_active = False
            self.batch_remaining = 0
            self.root.deiconify()
            self._error(exc)

    def start_batch(self):
        try:
            self._ensure_idle()
            self._validate_live()
            if not self.batch_accept.get():
                raise ValueError("For unattended runs, explicitly enable 'Accept all automatic proposals' after testing on representative images.")
            count, wait = int(self.frame_count.get()), float(self.wait_seconds.get())
            if not 1 <= count <= 1000 or not 1 <= wait <= 60:
                raise ValueError("Choose 1–1000 frames and a 1–60 second load delay.")
            self.batch_remaining = count
            self.batch_paused = False
            self.batch_seen = [self.rgb.copy()]
            self._start_auto_detection(True)
        except Exception as exc:
            self.batch_remaining = 0
            self._error(exc)

    def _next_batch_frame(self):
        if not self.batch_remaining:
            return
        if self.batch_paused:
            self._later(150, self._next_batch_frame)
            return
        geometry, target = self.geometry, self.target
        wait, key = float(self.wait_seconds.get()), self.next_key.get()
        previous = tuple(self.batch_seen)
        self.root.withdraw()
        def work(token):
            self.desktop.focus_target(target)
            token.wait_until_ready()
            self.desktop.press_next(key)
            token.sleep(wait)
            first = self.desktop.screenshot(geometry)
            token.sleep(.35)
            second = self.desktop.screenshot(geometry)
            if changed_fraction(first, second) > .01:
                raise RuntimeError("The next image is still changing. Increase the load delay and recapture.")
            if any(changed_fraction(old, second) < .02 for old in previous):
                raise RuntimeError("The image did not change, or a previous image reappeared. Batch stopped to prevent repeated picks.")
            return second
        self._submit("next image", work)

    def pause(self):
        if not self.runner.busy and not self.batch_remaining and not self.playback_pending:
            return
        self.batch_paused = not self.batch_paused
        if self.active_token:
            self.active_token.pause() if self.batch_paused else self.active_token.resume()
        self.status.set("Paused. F8 resumes; Esc stops." if self.batch_paused else "Resumed.")

    def stop(self):
        self.runner.invalidate()
        clear_recognition_progress(self)
        self.active_token = None
        self._cancel_timers()
        self.batch_remaining = 0
        self.batch_paused = False
        self.playback_pending = self.playback_active = False
        self.capture_pending = False
        if self.selector is not None:
            self.selector.destroy()
            self.selector = None
        self.live.set(False)
        self.root.deiconify()
        self.status.set("Stopped. Live picking is disabled. An active model calculation may still be finishing.")

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "hotkey":
                    self.stop() if event[1] == "stop" else self.pause()
                elif event[0] == "recognition_phase":
                    accept_recognition_progress(self, event[1], event[2])
                elif event[0] == "clicked":
                    # Points belong to the unchanged capture while playback is active;
                    # stop invalidation must not forget clicks already delivered.
                    if event[5] != self.capture_id:
                        continue
                    self.picked_points.add(event[2])
                    if event[1] == self.runner.generation:
                        self.status.set(f"Picking: {event[3]} / {event[4]} points. F8 pauses; Esc stops.")
        except queue.Empty:
            pass
        for result in self.runner.poll_results():
            kind = self.job_kind
            elapsed = time.perf_counter() - self.job_started
            clear_recognition_progress(self, result.token)
            self.active_token = None
            APP_LOG.info("%s finished in %.3fs; error=%r cancelled=%s", kind, elapsed, result.error, result.cancelled)
            if result.error is not None or result.cancelled:
                self.batch_remaining = 0
                self.playback_active = False
                self.root.deiconify()
                if result.traceback_text:
                    APP_LOG.error("%s", result.traceback_text)
                self.status.set("Cancelled." if result.cancelled else str(result.error))
                if result.error and not isinstance(result.error, (Cancelled, CancelledOperation)):
                    self._error(result.error)
                continue
            try:
                self._handle_result(kind, result.value, elapsed)
            except Exception as exc:
                APP_LOG.exception("Result handling failed")
                self.batch_remaining = 0
                self.playback_active = False
                self.root.deiconify()
                self._error(exc)
        refresh_recognition_progress(self, self.batch_paused)
        self.root.after(40, self.poll)

    def _handle_result(self, kind, value, elapsed):
        if kind == "prompt":
            self.alternatives, self.alternative_index = value, 0
            self.render()
            self.status.set(f"Prompt finished in {elapsed:.2f}s. Review alternatives, then accept the intended inner region.")
        elif kind in ("automatic proposals", "batch recognition"):
            # Preserve manually accepted masks, discard previous unselected proposals.
            retained = [c for c in self.candidates if c.id in self.selected or c.id in self.picked]
            by_id = {c.id: c for c in retained + value}
            self.candidates = list(by_id.values())
            self.refresh_table()
            self.render()
            self.status.set(f"{len(value)} unclassified proposals in {elapsed:.2f}s. Select target masks before picking.")
            if kind == "batch recognition":
                if not value:
                    raise RuntimeError("No automatic candidates: batch stopped for review.")
                self.selected = {c.id for c in value} - self.picked
                self.refresh_table()
                self.start_playback(batch=True)
        elif kind in ("playback", "batch playback"):
            self.playback_active = False
            self.picked |= value["selected"]
            self.selected -= value["selected"]
            # Preserve the clean source for SAM/exports. Only the playback drift
            # reference is updated to include application pick markers.
            self.playback_reference = self.desktop.screenshot(self.geometry)
            self.playback_reference.flags.writeable = False
            self.refresh_table()
            self.render()
            if value["batch"]:
                # Compare both pre-pick and post-pick views: markers themselves must
                # not be mistaken for a successfully loaded new micrograph.
                self.batch_seen.append(self.playback_reference.copy())
                self.batch_seen = self.batch_seen[-20:]
                self.batch_remaining -= 1
                if self.batch_remaining:
                    self._later(400, self._next_batch_frame)
                else:
                    self.root.deiconify()
                    self.status.set(f"Batch finished. Last frame: {value['count']} clicks.")
            else:
                self.root.deiconify()
                self.status.set(f"Completed {value['count']} clicks in {elapsed:.2f}s. Review picks in cryoSPARC.")
        elif kind == "next image":
            self._set_image(value, self.geometry, self.target, "Batch screen capture")
            self.batch_seen.append(self.rgb.copy())
            # Bound memory for long batches; exact detection covers the last 20 images.
            self.batch_seen = self.batch_seen[-20:]
            self._start_auto_detection(True)

    def export(self):
        try:
            self._ensure_idle()
            if self.rgb is None:
                raise ValueError("Open or capture an image first.")
            directory = self.filedialog.askdirectory(title="Choose a parent folder for this review")
            if not directory:
                return
            destination = Path(directory) / f"saru_review_{time.strftime('%Y%m%d_%H%M%S')}"
            export_review(destination, self.rgb, self.candidates, self.selected,
                          self._grid(), self.geometry,
                          {"spacing": self.spacing.get(), "inset": self.inset.get()})
            self.status.set(f"Exported review to {destination}")
        except Exception as exc:
            self._error(exc)

    def close(self):
        self.closed = True
        clear_recognition_progress(self)
        self.runner.shutdown(wait=False)
        self._cancel_timers()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        if self.selector is not None:
            self.selector.destroy()
        self.root.destroy()


class PickerArgumentParser(argparse.ArgumentParser):
    """Resolve four user-facing profiles into an explicit detector configuration."""
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        choice = result.detection_profile
        if choice is None:
            choice = "crYOLO" if result.backend == "cryolo" else "capsid"
        choice = canonical_detection_profile(choice)
        # Preserve the previous explicit crYOLO + learned-refinement CLI spelling.
        if result.backend == "cryolo" and choice == "capsid-learned":
            choice = "crYOLO"
        backend, profile = resolve_detection_profile(choice)
        if result.backend is not None and result.backend != backend:
            self.error("The selected detection profile conflicts with --backend.")
        result.detection_profile = choice
        result.backend, result.automatic_profile = backend, profile
        return result


def build_parser():
    parser = PickerArgumentParser(description="SARU Picker 3.0.0: manual and model-assisted picking directly over cryoSPARC.")
    parser.add_argument("--review", action="store_true", help="Open the optional snapshot review editor instead of the screen overlay.")
    parser.add_argument("--image", help="Open an image in the optional offline review editor.")
    parser.add_argument("--backend", choices=("sam", "cryolo"), default=None,
                        help="Optional explicit backend. By default the detection profile selects SAM or crYOLO.")
    parser.add_argument("--checkpoint", help="Model weights: SAM .pth or crYOLO .onnx; otherwise use the selected backend's bundled model.")
    parser.add_argument("--sam-checkpoint", help="Separate SAM weights used by the three original detection profiles.")
    parser.add_argument("--cryolo-checkpoint", help="crYOLO ONNX model (bundled by default). Saved standard H5 paths use the matching adjacent or installed ONNX export; other H5 files require explicit --cryolo-python.")
    parser.add_argument("--cryolo-config", help="Legacy H5 JSON configuration. ONNX uses embedded metadata and ignores this option.")
    parser.add_argument("--cryolo-python", help="Expert override for a separately managed legacy H5 crYOLO runtime; unused for ONNX.")
    parser.add_argument("--cryolo-threshold", type=float, default=.3, help="crYOLO detector confidence threshold (0 to 1).")
    parser.add_argument("--cryolo-timeout", type=float, default=180.0, help="Maximum seconds for a crYOLO prediction worker.")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto or cpu for bundled crYOLO ONNX; cuda requires a supported GPU runtime.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--segment-image", help="Run headless recognition on a clean image, without any desktop events.")
    modes.add_argument("--diagnostics", action="store_true", help="Report installed dependency metadata without loading a detector.")
    modes.add_argument("--smoke-test", action="store_true", help="Noninteractive native-library, Tk, and CPU model check; never sends desktop input.")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--point", nargs=3, action="append", type=float, metavar=("X", "Y", "LABEL"), help="Repeatable foreground(1)/background(0) prompt.")
    parser.add_argument("--points-per-side", type=int, default=8)
    parser.add_argument("--detection-profile", "--automatic-profile", dest="detection_profile",
                        choices=(*DETECTION_PROFILES, "cryolo"), default=None,
                        help="capsid (default), capsid-learned and generic use SAM; crYOLO uses the trained detector with learned contour refinement.")
    parser.add_argument("--min-capsid-area", type=int, default=DEFAULT_MIN_CAPSID_AREA_PX, metavar="PIXELS",
                        help="Minimum connected contour area in screenshot pixels (default: 3500; 0 allows all sizes). Applies to every crYOLO automatic profile and SAM capsid profiles; manual, explicit prompts and SAM generic bypass it.")
    parser.add_argument("--spacing", type=int, default=13)
    parser.add_argument("--inset", type=float, default=2)
    parser.add_argument("--output", type=Path, help="Headless segmentation output directory; defaults to writable user data when installed.")
    parser.add_argument("--report", type=Path, help="Write diagnostics/smoke-test JSON here, including failures.")
    return parser


def _run_application(args, *, console_available: bool) -> int:
    log_path = None
    root = None
    interactive = not (args.diagnostics or args.smoke_test or args.segment_image)
    try:
        log_path = setup_logging(include_console=console_available)
        APP_LOG.info("Starting SARU Picker %s; platform=%s frozen=%s", APP_VERSION,
                     sys.platform, bool(getattr(sys, "frozen", False)))
        is_cryolo = getattr(args, "backend", "sam") == "cryolo"
        sam_explicit = getattr(args, "sam_checkpoint", None) or (args.checkpoint if not is_cryolo else None)
        cryolo_explicit = getattr(args, "cryolo_checkpoint", None) or (args.checkpoint if is_cryolo else None)
        # The generic checkpoint argument overrides only the selected backend.
        if args.checkpoint:
            if is_cryolo:
                cryolo_explicit = args.checkpoint
            else:
                sam_explicit = args.checkpoint
        args.sam_checkpoint = str(resolve_checkpoint(sam_explicit))
        args.cryolo_checkpoint = str(resolve_checkpoint(cryolo_explicit, backend="cryolo"))
        checkpoint = Path(args.cryolo_checkpoint if is_cryolo else args.sam_checkpoint)
        if not args.cryolo_config:
            args.cryolo_config = str(Path(args.cryolo_checkpoint).with_name("cryolo-config.json"))
        if args.diagnostics or args.smoke_test:
            if is_cryolo:
                config = RecognitionConfig(str(checkpoint), device=args.device,
                                           automatic_profile=args.automatic_profile,
                                           **model_runtime_options(args))
                report = (run_cryolo_smoke_test(log_path, config) if args.smoke_test
                          else cryolo_diagnostic_report(log_path, config))
            else:
                report = (run_smoke_test(checkpoint, log_path) if args.smoke_test
                          else diagnostic_report(log_path, checkpoint))
            report.update(backend=args.backend, detection_profile=args.detection_profile)
            if args.report:
                write_json_report(args.report, report)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "ok" else 1
        if args.segment_image:
            output = args.output or (user_state_directory() / "exports" / "saru_output"
                                    if getattr(sys, "frozen", False) else Path("saru_output"))
            config = RecognitionConfig(str(checkpoint), model_type=args.model_type, device=args.device,
                                       points_per_side=args.points_per_side, automatic_profile=args.automatic_profile,
                                       min_capsid_area_px=args.min_capsid_area, **model_runtime_options(args))
            rgb = load_rgb_file(args.segment_image)
            recognizer = create_recognizer(config) if is_cryolo else SamRecognizer(config)
            try:
                started = time.perf_counter()
                if args.box or args.point:
                    prompts = args.point or []
                    candidates = recognizer.prompt(rgb, box=args.box,
                                                    points=[p[:2] for p in prompts] or None,
                                                    labels=[p[2] for p in prompts] or None)
                    selected = {candidates[0].id} if candidates else set()
                else:
                    candidates = recognizer.automatic(rgb)
                    selected = {c.id for c in candidates}
                points = selected_grid(candidates, selected, rgb.shape[:2], args.spacing, args.inset, 10000)
                elapsed = time.perf_counter() - started
                export_review(output, rgb, candidates, selected, points, None,
                              {"config": asdict(config), "elapsed_seconds": elapsed,
                               "recognition": recognizer.last_detection,
                               "review_status": "unreviewed CLI outputs; no desktop clicks were sent"})
                report = {"status": "ok", "mode": "segment-image", "version": APP_VERSION,
                          "backend": args.backend, "detection_profile": args.detection_profile,
                          "candidates": len(candidates), "grid_points": len(points),
                          "elapsed_seconds": elapsed, "output": str(output)}
                if args.report:
                    write_json_report(args.report, report)
                print(json.dumps(report))
            finally:
                recognizer.close()
            return 0
        enable_dpi_awareness()
        configure_application_identity("SARU.Picker.crYOLO")
        import tkinter as tk
        root = tk.Tk()
        if args.review or args.image:
            PickerApp(root, args, log_path)
        else:
            ScreenPickerApp(root, args, log_path, checkpoint)
        root.mainloop()
        return 0
    except Exception as exc:
        APP_LOG.exception("Application operation failed")
        if args.report:
            try:
                write_json_report(args.report, {
                    "status": "error", "version": APP_VERSION,
                    "error": str(exc), "error_type": type(exc).__name__,
                    "log_path": str(log_path) if log_path else None,
                })
            except Exception:
                APP_LOG.exception("Could not write the failure report")
        if interactive:
            show_startup_error(exc, log_path, parent=root)
        else:
            print(f"SARU Picker: {exc}", file=sys.stderr)
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        return 1


def main(argv=None, *, default_backend="sam"):
    console_available = sys.stderr is not None
    with windowed_console_streams():
        parser = build_parser()
        arguments = list(sys.argv[1:] if argv is None else argv)
        settings_path = user_state_directory() / "detector-settings.json"
        try:
            settings = load_detector_settings(settings_path)
        except (OSError, ValueError, TypeError) as exc:
            APP_LOG.warning("Ignoring invalid detector settings: %s", exc)
            settings = {}
        if "sam_model_type" in settings:
            settings["model_type"] = settings.pop("sam_model_type")
        has_backend = any(value == "--backend" or value.startswith("--backend=") for value in arguments)
        has_profile = any(value.split("=", 1)[0] in {"--detection-profile", "--automatic-profile"} for value in arguments)
        if has_backend and not has_profile:
            settings.pop("detection_profile", None)
        if default_backend == "cryolo" and not has_backend and not has_profile and "detection_profile" not in settings:
            settings["detection_profile"] = "crYOLO"
        parser.set_defaults(**settings)
        args = parser.parse_args(arguments)
        args.detector_settings_path = str(settings_path)
        return _run_application(args, console_available=console_available)


if __name__ == "__main__":
    raise SystemExit(main())
