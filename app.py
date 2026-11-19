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

from recognition import (
    Candidate, RecognitionConfig, RecognitionError, Cancelled, SamRecognizer,
    make_candidate, polygon_mask, union_masks, sample_hex_grid, validate_rgb,
)
from execution import (
    CaptureGeometry, CancellationToken, CancelledOperation, TaskRunner,
    WorkerBusyError, play_clicks,
)
from desktop import DesktopAdapter, GlobalHotkeys, enable_dpi_awareness
from screen_app import ScreenPickerApp
from screen_overlay import verify_screen_overlay

APP_LOG = logging.getLogger("saru_picker")
APP_VERSION = "2.1.1"
DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"


def user_state_directory() -> Path:
    """Writable application data, independent of the installation directory."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data and Path(local_app_data).is_absolute():
            return Path(local_app_data) / "SARU_Picker"
        return Path.home() / "AppData" / "Local" / "SARU_Picker"
    return Path.home() / ".saru_picker"


def resolve_checkpoint(explicit: str | Path | None = None) -> Path:
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
    for directory in directories:
        candidate = directory / DEFAULT_CHECKPOINT
        if candidate.is_file():
            return candidate
    # Keep a concrete expected path for a useful missing-model message.
    return directories[0] / DEFAULT_CHECKPOINT


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
        "status": "ok", "mode": "diagnostics", "version": APP_VERSION,
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
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        if decoded.size != (12, 8):
            raise RuntimeError("Pillow PNG decoding returned the wrong dimensions.")
    return {"grid_points": int(len(points)), "png_roundtrip": True}


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


def _smoke_widget_layout(window, widget, description: str) -> dict[str, Any]:
    """Reject a managed control that is hidden, shrunk, or ancestor-clipped.

    Tk's packer can shrink a button to only a few visible pixels while its
    bottom edge still fits the root. Comparing only the two bottom edges does
    not establish that the label or complete clickable area is available.
    """
    size = [int(widget.winfo_width()), int(widget.winfo_height())]
    requested = [int(widget.winfo_reqwidth()), int(widget.winfo_reqheight())]
    if not widget.winfo_viewable():
        raise RuntimeError(f"{description} is not visible in the floating panel.")
    if min(size) <= 0 or any(actual < needed for actual, needed in zip(size, requested)):
        raise RuntimeError(
            f"{description} is clipped below its requested size: actual {size}, requested {requested}.")
    left, top = int(widget.winfo_rootx()), int(widget.winfo_rooty())
    right, bottom = left + size[0], top + size[1]
    parent = widget.master
    while parent is not None:
        parent_left, parent_top = int(parent.winfo_rootx()), int(parent.winfo_rooty())
        parent_right = parent_left + int(parent.winfo_width())
        parent_bottom = parent_top + int(parent.winfo_height())
        if (left < parent_left or top < parent_top or right > parent_right or bottom > parent_bottom):
            raise RuntimeError(f"{description} extends outside its parent and is not fully usable.")
        if parent is window:
            break
        parent = parent.master
    if parent is not window:
        raise RuntimeError(f"{description} is not attached to the floating panel.")
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
            f"Terminate's text or font baseline is clipped: usable {usable}, text {[text_width, text_height]}.")
    # The actual button retains its default centered anchor. This derived
    # baseline records the complete font line that must fit the usable area.
    baseline = border + pad_y + (usable[1]-text_height)//2 + ascent
    if baseline + (len(lines)-1)*line_height + descent > int(button.winfo_height())-border-pad_y:
        raise RuntimeError("Terminate's text descender falls outside its usable button area.")
    return {"text_size": [text_width, text_height], "usable_text_size": usable,
            "font_metrics": {"ascent": ascent, "descent": descent, "linespace": line_height},
            "centered_first_baseline_from_top": baseline, "full_font_line_fits": True}


def _smoke_panel_layout(picker, mode: str, status_case: str) -> dict[str, Any]:
    window = picker.root
    picker.fit_panel()
    window.update_idletasks()
    if window.state() != "normal" or not window.winfo_viewable():
        raise RuntimeError(f"The {mode} floating control panel is not visible.")
    if not window.tk.getboolean(window.attributes("-topmost")):
        raise RuntimeError("The floating control panel is not kept above the target application.")
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
    return {"panel_size": [int(window.winfo_width()), int(window.winfo_height())],
            "panel_requested_size": [int(window.winfo_reqwidth()), int(window.winfo_reqheight())],
            "managed_controls_checked": len(controls), "all_managed_controls_fit": True,
            "terminate": terminate, "status": status, "viewable": True, "topmost": True}


def _smoke_windows_runtime(log_path: Path) -> dict[str, Any]:
    # These are native Windows widgets at simulated Tk font/layout scales. The
    # Windows monitor's DPI setting is never changed. Each fresh interpreter is
    # scaled before widgets/fonts exist, because changing Tk scaling afterwards
    # does not reliably resize previously constructed controls.
    enable_dpi_awareness()
    tkinter = importlib.import_module("tkinter")
    scale_checks = []
    panel_layouts = {}
    overlay_checks = None
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
            window.update_idletasks()
            layouts = {}
            for mode in ("manual", "semi-auto", "auto"):
                if picker.mode != mode:
                    picker.set_mode(mode)
                window.update_idletasks()
                initial_status = picker.status.get()
                normal = _smoke_panel_layout(picker, mode, "default")
                picker.status.set(long_status)
                multiline = _smoke_panel_layout(picker, mode, "multiline_status")
                picker._hide_overlays()
                window.update_idletasks()
                if (window.state() != "normal" or not window.winfo_viewable()
                        or not window.tk.getboolean(window.attributes("-topmost"))):
                    raise RuntimeError(f"The {mode} control panel disappeared while hiding the screen overlays.")
                layouts[mode] = {"default": normal, "multiline_status": multiline,
                                 "panel_persists_when_overlays_hidden": True}
                if percent == 100:
                    panel_layouts[mode] = normal["panel_size"]
                picker.status.set(initial_status)
                picker.fit_panel()
            tk_version = str(window.tk.call("info", "patchlevel"))
            scale_checks.append({"scale_percent": percent, "requested_dpi": requested_dpi,
                                 "native_tk_scaling_before_test": original_scaling,
                                 "actual_tk_pixels_per_point": actual_scaling, "layouts": layouts})
            if overlay_checks is None:
                overlay_checks = verify_screen_overlay(window)
        finally:
            if original_scaling is not None:
                window.tk.call("tk", "scaling", original_scaling)
            if picker is not None:
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
            "layout_scaling_method": "Native Windows Tk widgets; pixels-per-point set before construction to simulate 100/125/150/200% of 96 DPI. Windows monitor scaling is unchanged.",
            "layout_scale_checks": scale_checks,
            "screen_overlay": overlay_checks,
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
        self.runner = TaskRunner()
        self.events: queue.Queue = queue.Queue()
        self.desktop = DesktopAdapter()
        self.hotkeys = None
        self.active_token = None
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
        root.title("SARU Picker 2 — review masks, then pick")
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
        ttk.Label(controls, text="SARU PICKER", font=("Arial", 17, "bold")).pack(anchor="w")
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
        model = ttk.LabelFrame(controls, text="2. SAM model", padding=7)
        model.pack(fill="x", pady=4)
        checkpoint = str(resolve_checkpoint(self.args.checkpoint))
        self.checkpoint = tk.StringVar(value=checkpoint)
        ttk.Entry(model, textvariable=self.checkpoint, width=36).pack(fill="x")
        row = ttk.Frame(model)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Browse…", command=self.browse_model).pack(side="left")
        self.device = tk.StringVar(value=self.args.device)
        ttk.Combobox(row, textvariable=self.device, values=("auto", "cpu", "cuda"),
                     state="readonly", width=7).pack(side="left", padx=5)
        self.model_type = tk.StringVar(value=self.args.model_type)
        ttk.Combobox(row, textvariable=self.model_type, values=("vit_b", "vit_l", "vit_h"),
                     state="readonly", width=6).pack(side="left")
        self.grid_side = tk.IntVar(value=8)
        row = ttk.Frame(model)
        row.pack(fill="x")
        ttk.Label(row, text="Auto grid / side").pack(side="left")
        ttk.Spinbox(row, from_=4, to=32, width=5, textvariable=self.grid_side).pack(side="right")
        self.tool = tk.StringVar(value="box")
        tools_frame = ttk.LabelFrame(controls, text="3. Draw on preview", padding=7)
        tools_frame.pack(fill="x", pady=4)
        for value, label in (("freehand", "Manual contour"), ("box", "Box around target"),
                             ("positive", "Foreground point (+)"), ("negative", "Background point (−)"),
                             ("select", "Select / deselect mask")):
            ttk.Radiobutton(tools_frame, text=label, value=value, variable=self.tool).pack(anchor="w")
        row = ttk.Frame(tools_frame)
        row.pack(fill="x", pady=(5, 0))
        ttk.Button(row, text="Segment prompt", command=self.segment_prompt).pack(side="left")
        ttk.Button(row, text="Clear prompt", command=self.clear_prompt).pack(side="left", padx=4)
        row = ttk.Frame(tools_frame)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Next alternative", command=self.cycle_alternative).pack(side="left")
        ttk.Button(row, text="Accept mask", command=self.accept_alternative).pack(side="left", padx=4)
        ttk.Button(tools_frame, text="Generate automatic proposals", command=self.auto_proposals).pack(fill="x")
        ttk.Label(tools_frame, text="Yellow: proposal / preview. Green: selected.\nSAM scores estimate mask quality, not target identity.",
                  wraplength=310).pack(anchor="w", pady=4)
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

    def _ensure_idle(self):
        if self.runner.busy or self.playback_pending or self.capture_pending or self.batch_remaining:
            raise RuntimeError("An operation is active. Stop it and wait for the worker to finish first.")

    def _error(self, error):
        APP_LOG.error("%s", error)
        self.status.set(str(error))
        self.messagebox.showerror("SARU Picker", f"{error}\n\nDetails: {self.log_path}", parent=self.root)

    def browse_model(self):
        path = self.filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*")])
        if path:
            self.checkpoint.set(path)
            for kind in ("vit_b", "vit_l", "vit_h"):
                if kind in Path(path).name:
                    self.model_type.set(kind)
                    break

    def _model_config(self):
        return RecognitionConfig(checkpoint=self.checkpoint.get(), model_type=self.model_type.get(),
                                 device=self.device.get(), points_per_side=int(self.grid_side.get()))

    def _get_recognizer(self, config):
        # Called only by the serialized worker.
        if self.recognizer is None or self.recognizer_config != config:
            if self.recognizer is not None:
                self.recognizer.close()
            self.recognizer = SamRecognizer(config)
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
        return self.rgb is not None and not self.runner.busy and not self.playback_pending and not self.batch_remaining

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
        self.status.set(f"{kind}: running. Esc cancels; an active model forward pass may need to finish.")

    def segment_prompt(self):
        try:
            self._ensure_idle()
            if self.rgb is None or (self.prompt_box is None and not any(self.prompt_labels)):
                raise ValueError("Draw a box or add a foreground point first.")
            config, rgb = self._model_config(), self.rgb
            box = self.prompt_box
            points, labels = tuple(self.prompt_points), tuple(self.prompt_labels)
            self._submit("prompt", lambda token: self._get_recognizer(config).prompt(
                rgb, box=box, points=points or None, labels=labels or None,
                cancel=lambda: token.cancelled))
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
        self._submit("batch recognition" if batch else "automatic proposals",
                     lambda token: self._get_recognizer(config).automatic(rgb, cancel=lambda: token.cancelled))

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
                label += f" • alternative {self.alternative_index+1}/{len(self.alternatives)}, predicted IoU {preview.predicted_iou:.3f}"
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
        self.runner.shutdown(wait=False)
        self._cancel_timers()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        if self.selector is not None:
            self.selector.destroy()
        self.root.destroy()


def build_parser():
    parser = argparse.ArgumentParser(description="SARU Picker: manual and model-assisted picking directly over cryoSPARC.")
    parser.add_argument("--review", action="store_true", help="Open the optional snapshot review editor instead of the screen overlay.")
    parser.add_argument("--image", help="Open an image in the optional offline review editor.")
    parser.add_argument("--checkpoint", help="Explicit original SAM checkpoint (.pth); otherwise use the bundled model.")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--segment-image", help="Run headless recognition on a clean image, without any desktop events.")
    modes.add_argument("--diagnostics", action="store_true", help="Report installed dependency metadata without loading SAM.")
    modes.add_argument("--smoke-test", action="store_true", help="Noninteractive native-library, Tk, and CPU model check; never sends desktop input.")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--point", nargs=3, action="append", type=float, metavar=("X", "Y", "LABEL"), help="Repeatable foreground(1)/background(0) prompt.")
    parser.add_argument("--points-per-side", type=int, default=8)
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
        checkpoint = resolve_checkpoint(args.checkpoint)
        if args.diagnostics or args.smoke_test:
            report = (run_smoke_test(checkpoint, log_path) if args.smoke_test
                      else diagnostic_report(log_path, checkpoint))
            if args.report:
                write_json_report(args.report, report)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "ok" else 1
        if args.segment_image:
            output = args.output or (user_state_directory() / "exports" / "saru_output"
                                    if getattr(sys, "frozen", False) else Path("saru_output"))
            config = RecognitionConfig(str(checkpoint), model_type=args.model_type, device=args.device,
                                       points_per_side=args.points_per_side)
            rgb = load_rgb_file(args.segment_image)
            recognizer = SamRecognizer(config)
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
                               "review_status": "unreviewed CLI outputs; no desktop clicks were sent"})
                report = {"status": "ok", "mode": "segment-image", "version": APP_VERSION,
                          "candidates": len(candidates), "grid_points": len(points),
                          "elapsed_seconds": elapsed, "output": str(output)}
                if args.report:
                    write_json_report(args.report, report)
                print(json.dumps(report))
            finally:
                recognizer.close()
            return 0
        enable_dpi_awareness()
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


def main(argv=None):
    console_available = sys.stderr is not None
    with windowed_console_streams():
        args = build_parser().parse_args(argv)
        return _run_application(args, console_available=console_available)


if __name__ == "__main__":
    raise SystemExit(main())
