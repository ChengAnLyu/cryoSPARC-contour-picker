#!/usr/bin/env python3
"""SARU Picker 2.9.0: direct screen contours and model-assisted desktop picking.
Generated from the companion tested modules. Run python main.py --help.
"""
from __future__ import annotations


# ---------------- application_icon.py ----------------


import ctypes
import logging
from pathlib import Path
import sys


APPLICATION_ICON_LOG = logging.getLogger(__name__)
APPLICATION_USER_MODEL_ID = "SARU.Picker.Desktop"


def application_icon_paths() -> tuple[Path, Path]:
    """Locate artwork alongside source code or in PyInstaller's data folder."""
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return base / "assets" / "app-icon.ico", base / "assets" / "app-icon.png"


def configure_application_identity(application_id: str = APPLICATION_USER_MODEL_ID) -> bool:
    """Give source and installed Windows launches the same taskbar identity.

    Invoke before creating the first Tk window. Failure to set this optional
    Windows hint must not prevent particle picking from starting.
    """
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        setter = shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        result = setter(application_id)
        if result != 0:
            raise OSError(f"SetCurrentProcessExplicitAppUserModelID returned {result:#x}.")
        return True
    except Exception:
        APPLICATION_ICON_LOG.warning("Could not set the Windows taskbar identity.", exc_info=True)
        return False


def apply_application_icon(root) -> bool:
    """Set this window's icon and the default for its future child windows.

    A plain copied standalone script may not include assets. That remains a
    supported launch: missing artwork does not interrupt the application.
    The Windows packaging contract, in contrast, requires both icon assets.
    """
    ico, png = application_icon_paths()
    # A plain string records successful custom-artwork setup without retaining
    # a Tcl image. Native smoke checks must not accept an unrelated class icon.
    root._saru_application_icon_path = None
    if sys.platform == "win32" and ico.is_file():
        try:
            # Tk's -default path changes the native window *class* icons.
            # Set this root explicitly first (WM_SETICON); this also creates
            # its wrapper before setting the defaults for future toplevels.
            root.iconbitmap(str(ico))
            root.iconbitmap(default=str(ico))
            root._saru_application_icon_path = str(ico)
            return True
        except Exception:
            APPLICATION_ICON_LOG.warning("Could not set the ICO window icon; trying PNG.", exc_info=True)
    if not png.is_file():
        APPLICATION_ICON_LOG.warning("Application icon assets are unavailable in %s.", png.parent)
        return False
    try:
        import tkinter as tk

        photo = tk.PhotoImage(master=root, file=str(png))
        try:
            # Tk snapshots the pixels here; no retained PhotoImage is needed.
            # Its destructor therefore runs now, on this same GUI thread.
            root.iconphoto(True, photo)
        finally:
            del photo
        root._saru_application_icon_path = str(png)
        return True
    except Exception:
        APPLICATION_ICON_LOG.warning("Could not set the application window icon.", exc_info=True)
        return False


def windows_application_icon_status(root) -> dict[str, bool]:
    """Verify custom setup and real titlebar/taskbar icons on Windows.

    This intentionally checks the native top-level wrapper rather than Tk's
    child drawing window. WM_GETICON returns zero for class-only icons, so
    follow the documented GetClassLongPtr fallback (also used by Tk itself).
    Both outcomes require successful custom-artwork setup on this root; an
    untouched Python/Tk class icon is not evidence of application artwork.
    No icon is changed and no window is activated.
    """
    if sys.platform != "win32":
        raise RuntimeError("Native application icon inspection requires Windows.")
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    class_long = getattr(user32, "GetClassLongPtrW", None)
    if class_long is None:
        class_long = user32.GetClassLongW
    class_long.argtypes = [wintypes.HWND, ctypes.c_int]
    class_long.restype = ctypes.c_size_t
    child = root.winfo_id()
    handle = user32.GetAncestor(child, 2) or child  # GA_ROOT
    expected_paths = {str(path) for path in application_icon_paths() if path.is_file()}
    custom_applied = getattr(root, "_saru_application_icon_path", None) in expected_paths
    # WM_GETICON: ICON_SMALL=0, ICON_BIG=1, ICON_SMALL2=2. The last
    # fallbacks read GCLP_HICONSM=-34 and GCLP_HICON=-14 respectively.
    small = (user32.SendMessageW(handle, 0x007F, 0, 0)
             or user32.SendMessageW(handle, 0x007F, 2, 0)
             or class_long(handle, -34))
    large = user32.SendMessageW(handle, 0x007F, 1, 0) or class_long(handle, -14)
    return {
        "small_icon": bool(custom_applied and small),
        "large_icon": bool(custom_applied and large),
    }


# ---------------- capsid_prior.py ----------------

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage as ndi

PROFILE_FORMAT = "saru-capsid-appearance-v1"
PROFILE_FILENAME = "capsid-profile.json"
PROFILE_MAX_BYTES = 262144
FEATURE_NAMES = (
    'mean_1', 'mean_4', 'mean_12', 'mean_32',
    'fine_rms_5', 'fine_rms_12', 'fine_rms_24',
    'mid_rms_5', 'mid_rms_12', 'mid_rms_24',
    'coarse_rms_8', 'coarse_rms_16', 'coarse_rms_32',
    'minor_3', 'coherence_3', 'minor_6', 'coherence_6', 'minor_12', 'coherence_12',
    'grain_local_context_6', 'grain_local_context_12', 'black_6', 'black_16',
)


def _prior_check(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()


@dataclass(frozen=True)
class CapsidProfile:
    path: str
    sha256: str
    format: str
    mean: np.ndarray
    scale: np.ndarray
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]

    def describe(self) -> dict:
        return {'status': 'loaded', 'path': self.path, 'sha256': self.sha256,
                'format': self.format, 'feature_count': len(FEATURE_NAMES),
                'score_is_calibrated_probability': False}


def resolve_profile_path(checkpoint: str | Path,
                         explicit_path: str | Path | None = None) -> Path | None:
    """Resolve only a caller-selected file or the sidecar next to SAM weights."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise ValueError(f'Capsid profile not found: {path}')
        return path
    path = Path(checkpoint).expanduser().parent / PROFILE_FILENAME
    return path if path.is_file() else None


def load_profile(path: str | Path) -> CapsidProfile:
    """Load bounded numeric JSON, rejecting schema drift and invalid numbers."""
    path = Path(path).expanduser()
    with path.open('rb') as handle:
        blob = handle.read(PROFILE_MAX_BYTES + 1)
    if len(blob) > PROFILE_MAX_BYTES:
        raise ValueError('Capsid profile exceeds the 256 KiB size limit.')
    try:
        data = json.loads(blob.decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('Capsid profile must be valid UTF-8 JSON.') from exc
    if not isinstance(data, dict) or data.get('format') != PROFILE_FORMAT:
        raise ValueError('Unsupported capsid profile format.')
    if data.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Capsid profile feature schema does not match this application.')

    def numbers(value, shape, label, *, positive=False):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'fiu':
                raise ValueError('Profile parameters must be JSON numbers.')
            array = raw.astype(np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Invalid capsid profile {label}.') from exc
        if (array.shape != shape or not np.isfinite(array).all()
                or np.any(np.abs(array) > 10000)
                or (positive and np.any(array < 1e-5))):
            raise ValueError(f'Invalid capsid profile {label} dimensions or values.')
        array = np.ascontiguousarray(array)
        array.flags.writeable = False
        return array

    mean = numbers(data.get('mean'), (23,), 'mean')
    scale = numbers(data.get('scale'), (23,), 'scale', positive=True)
    raw_layers = data.get('layers')
    if not isinstance(raw_layers, list) or len(raw_layers) != 3:
        raise ValueError('Capsid profile must contain exactly three neural layers.')
    layers = []
    for index, (raw, dimensions) in enumerate(zip(raw_layers, ((32, 23), (16, 32), (1, 16)))):
        if not isinstance(raw, dict):
            raise ValueError('Capsid profile layers must be numeric objects.')
        layers.append((numbers(raw.get('weight'), dimensions, f'layer {index} weight'),
                       numbers(raw.get('bias'), (dimensions[0],), f'layer {index} bias')))
    return CapsidProfile(str(path.resolve()), hashlib.sha256(blob).hexdigest(),
                         PROFILE_FORMAT, mean, scale, tuple(layers))


def appearance_features(rgb: np.ndarray,
                        check_cancel: Callable[[], None] | None = None):
    """Fixed, contrast-normalized image features, with bounded temporary storage.

    Feature scales are screen pixels, matching the screenshot training domain.
    Feature normalization uses only the current clean screenshot, never marks.
    The 23-channel output is preallocated; neural activations are chunked later.
    """
    _prior_check(check_cancel)
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 4):
        raise ValueError('Capsid appearance requires an H x W x 3 uint8 RGB image, at least 4 pixels per side.')
    if rgb.shape[0] * rgb.shape[1] > 4_194_304:
        raise ValueError('Selected screen area is too large for the capsid profile; select a smaller micrograph area.')
    gray = rgb.mean(2, dtype=np.float32) / 255
    chroma = rgb.max(2).astype(np.int16) - rgb.min(2)
    valid = (chroma < 8) & (ndi.uniform_filter(gray, 7) < .97)
    valid = ndi.binary_erosion(valid, iterations=3, border_value=1)
    features = np.zeros((*gray.shape, len(FEATURE_NAMES)), np.float32)
    if not valid.any():
        return features, valid
    q10, median, q90 = np.quantile(gray[valid], [.1, .5, .9])
    contrast = max(float(q90 - q10), 1 / 255)
    z = np.clip((gray - median) / contrast, -4, 4).astype(np.float32)
    smooth = {}
    for sigma in (1, 3, 4, 8, 12, 32):
        _prior_check(check_cancel)
        smooth[sigma] = ndi.gaussian_filter(z, sigma)
    cursor = 0
    def append(value):
        nonlocal cursor
        features[..., cursor] = np.clip(value, -6, 6)
        cursor += 1
        _prior_check(check_cancel)
    for sigma in (1, 4, 12, 32):
        append(smooth[sigma])
    epsilon = np.float32(1e-5)
    for band, scales in ((z - smooth[1], (5, 12, 24)),
                         (smooth[1] - smooth[3], (5, 12, 24)),
                         (smooth[3] - smooth[8], (8, 16, 32))):
        squared = band * band
        for sigma in scales:
            _prior_check(check_cancel)
            rms = np.sqrt(np.maximum(ndi.gaussian_filter(squared, sigma), 0))
            append(np.log((rms + epsilon) / (np.median(rms[valid]) + epsilon)))
    gy, gx = np.gradient(smooth[1])
    xx, xy, yy = gx * gx, gx * gy, gy * gy
    for sigma in (3, 6, 12):
        _prior_check(check_cancel)
        a = ndi.gaussian_filter(xx, sigma)
        b = ndi.gaussian_filter(xy, sigma)
        c = ndi.gaussian_filter(yy, sigma)
        discriminant = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0))
        trace = a + c
        minor = np.sqrt(np.maximum((trace - discriminant) * .5, 0))
        append(np.log((minor + epsilon) / (np.median(minor[valid]) + epsilon)))
        append(discriminant / (trace + epsilon))
    fine = ndi.gaussian_filter((z - smooth[1]) ** 2, 3)
    context = ndi.gaussian_filter(fine, 32)
    for sigma in (6, 12):
        local = ndi.gaussian_filter(fine, sigma)
        append(np.log((local + epsilon) / (context + epsilon)))
    dark = (z < -.8).astype(np.float32)
    for sigma in (6, 16):
        append(ndi.gaussian_filter(dark, sigma))
    if cursor != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError('Capsid appearance feature computation failed.')
    return features, valid


def score_features(features: np.ndarray, profile: CapsidProfile,
                   check_cancel: Callable[[], None] | None = None) -> np.ndarray:
    """Apply the profile in fixed-size chunks without a full-image hidden tensor."""
    _prior_check(check_cancel)
    if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError('Capsid feature dimensions do not match the profile.')
    flat = features.reshape(-1, features.shape[-1])
    result = np.empty(len(flat), np.float32)
    for start in range(0, len(flat), 32768):
        _prior_check(check_cancel)
        hidden = np.clip((flat[start:start + 32768] - profile.mean) / profile.scale, -8, 8)
        for index, (weight, bias) in enumerate(profile.layers):
            hidden = hidden @ weight.T + bias
            if index < len(profile.layers) - 1:
                hidden = np.tanh(hidden)
        result[start:start + len(hidden)] = 1 / (1 + np.exp(-np.clip(hidden[:, 0], -30, 30)))
    if not np.isfinite(result).all():
        raise ValueError('Capsid appearance profile produced nonfinite scores.')
    _prior_check(check_cancel)
    return result.reshape(features.shape[:2])


def score_image(rgb: np.ndarray, profile: CapsidProfile,
                check_cancel: Callable[[], None] | None = None) -> np.ndarray:
    """Return one uncalibrated score per input pixel; UI-coloured pixels are zero."""
    features, valid = appearance_features(rgb, check_cancel)
    result = score_features(features, profile, check_cancel)
    result[~valid] = 0
    return result


# ---------------- carbon_support.py ----------------

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

_MAX_DIMENSION = 640
_MAX_SEGMENTS = 8
_MAX_EDGE_POINTS = 40_000
_MAX_PEAKS = 32
_MAX_MASK_POINTS = 8192


def _check(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()


@dataclass(frozen=True)
class CarbonSegment:
    """A finite, coherent straight edge in original screenshot coordinates."""
    endpoints_xy: tuple[tuple[float, float], tuple[float, float]]
    normal_xy: tuple[float, float]
    rho: float
    length: float
    coverage: float
    residual: float
    vote_density: float
    alignment: float
    support_kind: str = "frame-span"

    def describe(self) -> dict[str, Any]:
        return {
            "endpoints_xy": [list(p) for p in self.endpoints_xy],
            "length": round(self.length, 3),
            "coverage": round(self.coverage, 4),
            "residual": round(self.residual, 3),
            "vote_density": round(self.vote_density, 4),
            "alignment": round(self.alignment, 4),
            "support_kind": self.support_kind,
        }


@dataclass(frozen=True)
class CarbonContext:
    """Bounded full-ROI context; contains no original image or model state."""
    image_shape: tuple[int, int]
    work_shape: tuple[int, int]
    segments: tuple[CarbonSegment, ...]
    status: str = "ok"

    def describe(self) -> dict[str, Any]:
        return {
            "version": 1,
            "status": self.status,
            "image_shape": list(self.image_shape),
            "work_shape": list(self.work_shape),
            "detected_edges": len(self.segments),
            "segments": [segment.describe() for segment in self.segments],
            "scope": "final capsid contours; no blanket film-side exclusion",
        }


def detect_carbon_edges(rgb: np.ndarray,
                        check_cancel: Callable[[], None] | None = None) -> CarbonContext:
    """Find strong, long straight edges using a bounded, oriented Hough pass.

    Work is capped at 640 pixels on the longer side, 40,000 edge points,
    32 peaks and eight accepted segments.  Both screenshot grain and large
    support geometry contribute: fine gradient maxima vote for a long line,
    then finite support, direction coherence and residual checks verify it.
    The caller's cancellation callback is checked between bounded operations.
    """
    _check(check_cancel)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("carbon edge input must be an RGB uint8 image")
    h0, w0 = image.shape[:2]
    if h0 < 16 or w0 < 16:
        return CarbonContext((h0, w0), (h0, w0), (), "too-small")
    scale = min(1.0, _MAX_DIMENSION / max(h0, w0))
    w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    if (w, h) != (w0, h0):
        image = np.asarray(Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR))
    sx, sy = w / w0, h / h0
    _check(check_cancel)
    gray = np.mean(image, axis=2, dtype=np.float32) / 255.0
    # Exclude nearly uniform bright UI, rather than individual bright or dark
    # grain pixels.  Pointwise black exclusion would erase the carbon line.
    chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2)
    white_ui = ndi.gaussian_filter(gray, 3.0) > .90
    valid = ndi.binary_erosion((chroma < 12) & ~white_ui, iterations=10, border_value=0)
    if np.count_nonzero(valid) < 1024:
        return CarbonContext((h0, w0), (h, w), (), "insufficient-field")
    smooth = ndi.gaussian_filter(gray, 2.5)
    gy, gx = np.gradient(smooth)
    magnitude = np.hypot(gx, gy)
    threshold = max(.004, float(np.quantile(magnitude[valid], .85)))
    _check(check_cancel)
    yy, xx = np.mgrid[:h, :w]
    nx = gx / np.maximum(magnitude, 1e-8)
    ny = gy / np.maximum(magnitude, 1e-8)
    forward = ndi.map_coordinates(magnitude, [yy + ny, xx + nx], order=1, mode="nearest")
    backward = ndi.map_coordinates(magnitude, [yy - ny, xx - nx], order=1, mode="nearest")
    edges = valid & (magnitude >= threshold) & (magnitude >= forward) & (magnitude >= backward)
    ys, xs = np.nonzero(edges)
    if len(xs) < 40:
        return CarbonContext((h0, w0), (h, w), (), "no-long-edges")
    strengths = magnitude[ys, xs]
    if len(xs) > _MAX_EDGE_POINTS:
        # Stable ordering makes a fixed screenshot deterministic even if many
        # pixels tie at the threshold.
        keep = np.argsort(strengths, kind="stable")[-_MAX_EDGE_POINTS:]
        ys, xs, strengths = ys[keep], xs[keep], strengths[keep]
    normals_x, normals_y = nx[ys, xs], ny[ys, xs]
    angle = np.mod(np.arctan2(normals_y, normals_x), np.pi)
    bins = np.rint(angle * (180 / np.pi)).astype(np.int16) % 180
    diagonal = math.hypot(w, h)
    rho_max = math.ceil(diagonal)
    accumulator = np.zeros((180, 2 * rho_max + 1), np.float32)
    weights = np.clip(strengths / threshold, 1, 3)
    for delta in range(-15, 16):
        if delta % 4 == 0:
            _check(check_cancel)
        angles = (bins + delta) % 180
        theta = angles * (np.pi / 180)
        rhos = np.rint(xs * np.cos(theta) + ys * np.sin(theta)).astype(np.int32) + rho_max
        np.add.at(accumulator, (angles, rhos), weights)
    maxima = accumulator == ndi.maximum_filter(accumulator, size=(9, 13), mode="constant")
    peak_indices = np.argwhere(maxima & (accumulator > max(35, diagonal * .10)))
    if len(peak_indices):
        strength_order = np.argsort(-accumulator[peak_indices[:, 0], peak_indices[:, 1]], kind="stable")
        peak_indices = peak_indices[strength_order[:_MAX_PEAKS]]
    minimum_length = max(100.0, .30 * diagonal)
    work_segments: list[dict[str, Any]] = []
    for angle_bin, rho_bin in peak_indices:
        _check(check_cancel)
        theta = angle_bin * (np.pi / 180)
        initial_normal = np.array([math.cos(theta), math.sin(theta)])
        initial_tangent = np.array([-initial_normal[1], initial_normal[0]])
        initial_rho = float(rho_bin - rho_max)
        distances = xs * initial_normal[0] + ys * initial_normal[1] - initial_rho
        alignments = np.abs(normals_x * initial_normal[0] + normals_y * initial_normal[1])
        near = (np.abs(distances) <= 3.0) & (alignments > .80)
        if np.count_nonzero(near) < minimum_length * .35:
            continue
        tangent_positions = xs * initial_tangent[0] + ys * initial_tangent[1]
        near_positions = tangent_positions[near]
        lower = math.floor(float(near_positions.min()) / 5)
        upper = math.ceil(float(near_positions.max()) / 5)
        support = np.bincount(np.floor(near_positions / 5).astype(int) - lower,
                              minlength=upper - lower + 1) > 0
        connected = ndi.binary_closing(np.pad(support, 2), iterations=2)[2:-2]
        labels, count = ndi.label(connected)
        for label_id in range(1, count + 1):
            indices = np.flatnonzero(labels == label_id)
            t0 = (lower + int(indices.min())) * 5
            t1 = (lower + int(indices.max()) + 1) * 5
            if t1 - t0 < minimum_length:
                continue
            coverage = float(np.mean(support[indices]))
            if coverage < .86:
                continue
            members = near & (tangent_positions >= t0) & (tangent_positions <= t1)
            points = np.column_stack((xs[members], ys[members]))
            if len(points) < 20:
                continue
            center = np.mean(points, axis=0)
            _, vectors = np.linalg.eigh(np.cov(points.T))
            normal = vectors[:, 0]
            if float(normal @ initial_normal) < 0:
                normal = -normal
            if float(normal @ initial_normal) < .99:
                continue
            tangent = np.array([-normal[1], normal[0]])
            rho = float(center @ normal)
            positions = points @ tangent
            low, high = np.quantile(positions, [.01, .99])
            length = float(high - low)
            residual = float(np.quantile(np.abs(points @ normal - rho), .90))
            alignment = float(np.median(alignments[members]))
            vote_density = float(accumulator[angle_bin, rho_bin] / max(length, 1))
            if length < minimum_length or residual > 2.8 or alignment < .98 or vote_density < .46:
                continue
            p0, p1 = normal * rho + tangent * low, normal * rho + tangent * high
            # Prefer the strongest already-verified line when two fits describe
            # the same support, including slightly different fits to a curve.
            if any(abs(float(normal @ q["normal"])) > .995 and
                   max(abs(float(p0 @ q["normal"] - q["rho"])),
                       abs(float(p1 @ q["normal"] - q["rho"]))) < 8
                   for q in work_segments):
                continue
            work_segments.append(dict(normal=normal, rho=rho, p0=p0, p1=p1,
                                      length=length, coverage=coverage, residual=residual,
                                      vote_density=vote_density, alignment=alignment))
            if len(work_segments) >= _MAX_SEGMENTS:
                break
        if len(work_segments) >= _MAX_SEGMENTS:
            break
    _check(check_cancel)
    # An isolated long membrane facet can also win a Hough vote. Require
    # full-field support: a line reaches two field sides, or reaches one side
    # while spanning >=40% of the field diagonal. A nearby parallel segment
    # can borrow that context (e.g. the second edge of a thin carbon strip).
    columns, rows = np.mean(valid, axis=0), np.mean(valid, axis=1)
    field_x = np.flatnonzero(columns >= max(.10, float(columns.max()) * .50))
    field_y = np.flatnonzero(rows >= max(.10, float(rows.max()) * .50))
    if not len(field_x) or not len(field_y):
        return CarbonContext((h0, w0), (h, w), (), "insufficient-field")
    field_bounds = (float(field_x[0]), float(field_y[0]),
                    float(field_x[-1]), float(field_y[-1]))
    frame_tolerance = max(10., .045 * min(h, w))
    def near_frame(point):
        x, y = point
        x0, y0, x1, y1 = field_bounds
        return min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)) <= frame_tolerance
    anchored = [segment for segment in work_segments
                if ((near_frame(segment["p0"]) and near_frame(segment["p1"])) or
                    ((near_frame(segment["p0"]) or near_frame(segment["p1"])) and
                     segment["length"] >= .40 * diagonal))]
    supported = []
    for segment in work_segments:
        if any(segment is anchor for anchor in anchored):
            segment["support_kind"] = "frame-span"
            supported.append(segment)
            continue
        center = (segment["p0"] + segment["p1"]) * .5
        tangent = np.array([-segment["normal"][1], segment["normal"][0]])
        segment_low, segment_high = sorted([segment["p0"] @ tangent, segment["p1"] @ tangent])
        for anchor in anchored:
            if abs(float(segment["normal"] @ anchor["normal"])) < .985:
                continue
            if abs(float(center @ anchor["normal"] - anchor["rho"])) > .08 * min(h, w):
                continue
            anchor_low, anchor_high = sorted([anchor["p0"] @ tangent, anchor["p1"] @ tangent])
            overlap = min(segment_high, anchor_high) - max(segment_low, anchor_low)
            if overlap >= .50 * min(segment["length"], anchor["length"]):
                segment["support_kind"] = "parallel-continuation"
                supported.append(segment)
                break
    segments = []
    scales = np.array([sx, sy])
    for segment in supported:
        p0, p1 = segment["p0"] / scales, segment["p1"] / scales
        transformed_normal = segment["normal"] * scales
        normal_length = float(np.linalg.norm(transformed_normal))
        transformed_normal /= normal_length
        segments.append(CarbonSegment(
            endpoints_xy=(tuple(map(float, p0)), tuple(map(float, p1))),
            normal_xy=tuple(map(float, transformed_normal)),
            rho=segment["rho"] / normal_length,
            length=float(np.linalg.norm(p1 - p0)),
            coverage=segment["coverage"], residual=segment["residual"] / normal_length,
            vote_density=segment["vote_density"], alignment=segment["alignment"],
            support_kind=segment["support_kind"],
        ))
    return CarbonContext((h0, w0), (h, w), tuple(segments), "ok" if segments else "no-long-edges")


# The descriptive alias is also used by the independent private replay script.
detect_carbon_context = detect_carbon_edges


def carbon_candidate_evidence(mask: np.ndarray, context: CarbonContext,
                              check_cancel: Callable[[], None] | None = None) -> dict[str, Any]:
    """Adjudicate this actual mask, independently of its parent or neighbors.

    Kept/rejected decisions are intentionally based on multiple independent
    geometric facts.  Long-line context alone never vetoes a compact target.
    All numbers are diagnostics rather than calibrated probabilities.
    """
    _check(check_cancel)
    source_mask = np.asarray(mask, dtype=bool)
    if source_mask.ndim != 2 or source_mask.shape != context.image_shape:
        raise ValueError("carbon guard mask has a different size from the screenshot")
    evidence: dict[str, Any] = {"reject": False, "reason": "no-long-edge", "line_count": len(context.segments)}
    if not context.segments:
        return evidence
    h0, w0 = context.image_shape
    h, w = context.work_shape
    row_ids = np.minimum((np.arange(h) * h0 / h).astype(int), h0 - 1)
    col_ids = np.minimum((np.arange(w) * w0 / w).astype(int), w0 - 1)
    work_mask = source_mask[np.ix_(row_ids, col_ids)]
    area = int(np.count_nonzero(work_mask))
    evidence.update(area_fraction=round(area / (h * w), 5))
    if area < 32:
        evidence["reason"] = "small-or-empty-mask"
        return evidence
    # Interior holes represent valid exclusions, not outer contour evidence.
    # Fill only this temporary diagnostic silhouette; never alter the output.
    silhouette = ndi.binary_fill_holes(work_mask)
    boundary = silhouette & ~ndi.binary_erosion(silhouette, structure=np.ones((3, 3), bool), border_value=0)
    by, bx = np.nonzero(boundary)
    iy, ix = np.nonzero(work_mask)
    if len(bx) > _MAX_MASK_POINTS:
        stride = math.ceil(len(bx) / _MAX_MASK_POINTS)
        bx, by = bx[::stride], by[::stride]
    if len(ix) > _MAX_MASK_POINTS:
        stride = math.ceil(len(ix) / _MAX_MASK_POINTS)
        ix, iy = ix[::stride], iy[::stride]
    scales = np.array([w / w0, h / h0])
    records = []
    for line_id, segment in enumerate(context.segments):
        _check(check_cancel)
        endpoints = np.asarray(segment.endpoints_xy) * scales
        delta = endpoints[1] - endpoints[0]
        line_length = float(np.linalg.norm(delta))
        tangent = delta / line_length
        normal = np.array([-tangent[1], tangent[0]])
        rho = float(endpoints[0] @ normal)
        t0, t1 = sorted(endpoints @ tangent)
        bdist = bx * normal[0] + by * normal[1] - rho
        btangent = bx * tangent[0] + by * tangent[1]
        # Six working pixels accommodate antialiasing, edge blur and polygon
        # simplification, while preventing proximity from becoming association.
        near = (np.abs(bdist) <= 6.0) & (btangent >= t0 - 6) & (btangent <= t1 + 6)
        contact_fraction = float(np.mean(near))
        if np.count_nonzero(near) < 12:
            continue
        contact_low, contact_high = np.quantile(btangent[near], [.02, .98])
        contact_span = float(contact_high - contact_low)
        positions = ix * tangent[0] + iy * tangent[1]
        distances = ix * normal[0] + iy * normal[1] - rho
        along_low, along_high = np.quantile(positions, [.02, .98])
        across_low, across_high = np.quantile(distances, [.02, .98])
        along_span = max(1., float(along_high - along_low))
        across_span = max(1., float(across_high - across_low))
        aspect = along_span / across_span
        side_fraction = max(float(np.mean(distances >= -6)), float(np.mean(distances <= 6)))
        record = {
            "line_id": line_id,
            "boundary_contact_fraction": round(contact_fraction, 5),
            "contact_span": round(contact_span, 3),
            "line_length": round(line_length, 3),
            "line_to_mask_scale": round(line_length / math.sqrt(area), 4),
            "along_span": round(along_span, 3),
            "across_span": round(across_span, 3),
            "aligned_aspect": round(aspect, 4),
            "one_side_fraction": round(side_fraction, 5),
        }
        reject = (contact_fraction >= .25 and
                  contact_span >= max(45., 1.10 * math.sqrt(area)) and
                  contact_span >= .55 * along_span and
                  line_length >= 3.0 * math.sqrt(area) and
                  line_length >= 1.75 * along_span and
                  aspect >= 2.0 and side_fraction >= .94)
        record["reject"] = bool(reject)
        records.append(record)
    evidence["reason"] = "independent-contour"
    evidence["matches"] = records
    rejected = [record for record in records if record["reject"]]
    if rejected:
        evidence["reject"] = True
        evidence["reason"] = "elongated-contour-follows-long-support-edge"
        evidence["matched_line"] = max(rejected, key=lambda r: r["boundary_contact_fraction"])["line_id"]
    return evidence


def guard_proposals(context: CarbonContext, proposals: Iterable[Mapping[str, Any]],
                    check_cancel: Callable[[], None] | None = None
                    ) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Keep original proposal objects and order; return a bounded audit summary."""
    retained = []
    accepted_indices, rejected_indices, decisions = [], [], []
    for index, proposal in enumerate(proposals):
        _check(check_cancel)
        evidence = carbon_candidate_evidence(proposal["segmentation"], context, check_cancel)
        if evidence["reject"]:
            rejected_indices.append(index)
        else:
            retained.append(proposal)
            accepted_indices.append(index)
        # Keep the summary bounded even with pathological external proposal
        # iterables. Full evidence remains available through the scoring API.
        if len(decisions) < 64 and (evidence["reject"] or evidence.get("matches")):
            decisions.append({"proposal_index": index, **evidence})
    summary = context.describe()
    summary.update(input_count=len(accepted_indices) + len(rejected_indices),
                   retained_count=len(retained), rejected_count=len(rejected_indices),
                   accepted_indices=accepted_indices, rejected_indices=rejected_indices,
                   decisions=decisions)
    return retained, summary


# ---------------- capsid.py ----------------

from collections.abc import Callable, Iterable, Mapping
import math
import heapq
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def _capsid_check(check_cancel: Callable[[], None] | None) -> None:
    if check_cancel is not None:
        check_cancel()


def _capsid_texture(rgb: np.ndarray, check_cancel: Callable[[], None] | None):
    """Return isotropic grain contrast and a conservative screenshot ROI mask.

    The smaller structure-tensor eigenvalue suppresses the coherent gradient
    along a membrane while retaining gradients in both directions inside
    granular density. All comparisons below are relative, so a linear contrast
    adjustment does not introduce a fixed grayscale density threshold.
    """
    _capsid_check(check_cancel)
    gray = np.mean(rgb, axis=2, dtype=np.float32) / 255.0
    # Pure white margins, UI colour and scale bars must not become grain cues.
    # Rejecting these pixels is distinct from treating dark pixels as capsids.
    chroma = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2)
    # Restrict the exclusion to nearly uniform white UI; bright/dark grain
    # pixels inside a capsid are legitimate data and must remain available.
    white_ui = ndi.uniform_filter(gray, size=7) > .97
    valid = (chroma < 8) & ~white_ui
    valid = ndi.binary_erosion(valid, iterations=3, border_value=1)
    _capsid_check(check_cancel)
    smooth = ndi.gaussian_filter(gray, 1.5)
    gy, gx = np.gradient(smooth)
    # A narrow orientation window avoids mistaking a curved membrane for
    # isotropic grain simply because its tangent changes across the window.
    xx = ndi.gaussian_filter(gx * gx, 4)
    _capsid_check(check_cancel)
    xy = ndi.gaussian_filter(gx * gy, 4)
    yy = ndi.gaussian_filter(gy * gy, 4)
    discriminant = np.sqrt(np.maximum((xx - yy) ** 2 + 4 * xy * xy, 0))
    minor = np.sqrt(np.maximum((xx + yy - discriminant) * .5, 0))
    coherence = discriminant / (xx + yy + 1e-9)
    texture = ndi.gaussian_filter(minor * np.clip(1 - coherence, 0, 1), 4)
    _capsid_check(check_cancel)
    return texture, valid


def _capsid_bbox(mask: np.ndarray) -> list[int]:
    yy, xx = np.nonzero(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1),
            int(yy.max() - yy.min() + 1)]


def _capsid_field_bounds(valid: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the grayscale field, allowing screenshot margins around it."""
    columns = np.mean(valid, axis=0)
    rows = np.mean(valid, axis=1)
    xx = np.flatnonzero(columns >= max(.1, float(columns.max()) * .5))
    yy = np.flatnonzero(rows >= max(.1, float(rows.max()) * .5))
    if not len(xx) or not len(yy):
        return 0, 0, valid.shape[1] - 1, valid.shape[0] - 1
    return int(xx[0]), int(yy[0]), int(xx[-1]), int(yy[-1])


def _capsid_screen_sides(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bounds
    horizontal_min = max(3, (x1 - x0 + 1) * .02)
    vertical_min = max(3, (y1 - y0 + 1) * .02)
    return sum((
        np.count_nonzero(mask[y0:min(y0 + 3, y1 + 1), x0:x1 + 1]) >= horizontal_min,
        np.count_nonzero(mask[max(y0, y1 - 2):y1 + 1, x0:x1 + 1]) >= horizontal_min,
        np.count_nonzero(mask[y0:y1 + 1, x0:min(x0 + 3, x1 + 1)]) >= vertical_min,
        np.count_nonzero(mask[y0:y1 + 1, max(x0, x1 - 2):x1 + 1]) >= vertical_min,
    ))


def _capsid_boundary_rings(mask: np.ndarray, check_cancel=None) -> list[np.ndarray]:
    """Trace cell edges, including the image edge, in image coordinates.

    A foreground pixel (x, y) occupies [x, x+1] x [y, y+1]. The right-turn
    rule keeps diagonal contacts separate. This avoids introducing a hull
    across concavities or joining distinct regions at a shared corner.
    """
    padded = np.pad(mask, 1)
    edge_sets = (
        mask & ~padded[:-2, 1:-1], mask & ~padded[1:-1, 2:],
        mask & ~padded[2:, 1:-1], mask & ~padded[1:-1, :-2],
    )
    if sum(np.count_nonzero(edges) for edges in edge_sets) > 50_000:
        return []  # Bound pathological/noisy boundaries; caller keeps mask.
    outgoing = {}
    for direction, edges in enumerate(edge_sets):
        yy, xx = np.nonzero(edges)
        for x, y in zip(xx.tolist(), yy.tolist()):
            if direction == 0:
                start, end = (x, y), (x + 1, y)
            elif direction == 1:
                start, end = (x + 1, y), (x + 1, y + 1)
            elif direction == 2:
                start, end = (x + 1, y + 1), (x, y + 1)
            else:
                start, end = (x, y + 1), (x, y)
            outgoing.setdefault(start, {})[direction] = end
        _capsid_check(check_cancel)
    rings = []
    while outgoing:
        _capsid_check(check_cancel)
        start = next(iter(outgoing))
        current = start
        direction = next(iter(outgoing[start]))
        ring = []
        while True:
            ring.append(current)
            choices = outgoing[current]
            end = choices.pop(direction)
            if not choices:
                del outgoing[current]
            current = end
            if current == start:
                break
            choices = outgoing.get(current, {})
            direction = next((d for d in ((direction + 1) % 4, direction,
                                          (direction + 3) % 4, (direction + 2) % 4)
                              if d in choices), None)
            if direction is None:
                return []
        points = np.asarray(ring, dtype=np.float64)
        # Removing collinear cell-edge vertices is exact, and makes later
        # simplification much cheaper than processing every boundary pixel.
        before = points - np.roll(points, 1, axis=0)
        after = np.roll(points, -1, axis=0) - points
        corners = before[:, 0] * after[:, 1] != before[:, 1] * after[:, 0]
        rings.append(points[corners])
    return rings


def _capsid_simplify_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker simplification, with segment distances."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        vector = points[last] - points[first]
        length_squared = float(vector @ vector)
        offsets = points[first + 1:last] - points[first]
        if length_squared:
            fractions = np.clip(offsets @ vector / length_squared, 0, 1)
            offsets = offsets - fractions[:, None] * vector
        distances = np.einsum("ij,ij->i", offsets, offsets)
        farthest = int(np.argmax(distances))
        if distances[farthest] > tolerance * tolerance:
            split = first + 1 + farthest
            keep[split] = True
            stack.extend(((first, split), (split, last)))
    return points[keep]


def _capsid_simplify_ring(points: np.ndarray, tolerance: float, fixed=None) -> np.ndarray:
    if len(points) <= 4:
        return points
    # Two separated anchors avoid the closed-polyline zero-length-baseline
    # problem. A ring is simplified on both sides of these fixed anchors.
    first = int(np.argmax(np.sum((points - points[0]) ** 2, axis=1)))
    opposite = int(np.argmax(np.sum((points - points[first]) ** 2, axis=1)))
    anchors = np.unique([first, opposite, *(np.flatnonzero(fixed) if fixed is not None else [])])
    simplified = []
    for start, end in zip(anchors, np.roll(anchors, -1)):
        indices = (np.arange(start, end + 1) if end > start else
                   np.concatenate((np.arange(start, len(points)), np.arange(end + 1))))
        simplified.append(_capsid_simplify_open(points[indices], tolerance)[:-1])
    return np.vstack(simplified)


def _capsid_convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull used only when bounded shape checks permit it."""
    vertices = sorted(set(map(tuple, points)))
    if len(vertices) <= 2:
        return np.asarray(vertices, dtype=np.float64)
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for point in vertices:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(vertices):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _capsid_simple_polygon(points: np.ndarray) -> bool:
    """Reject degenerate/self-crossing RDP outputs before rasterization."""
    count = len(points)
    if count < 3 or count > 512:
        return False
    for index in range(count):
        a, b = points[index], points[(index + 1) % count]
        c = points[index + 2:count - (1 if index == 0 else 0)]
        if not len(c):
            continue
        d = points[(np.arange(index + 2, index + 2 + len(c)) + 1) % count]
        ab = b - a
        ac, ad = c - a, d - a
        cd = d - c
        ca, cb = a - c, b - c
        cross_ac = ab[0] * ac[:, 1] - ab[1] * ac[:, 0]
        cross_ad = ab[0] * ad[:, 1] - ab[1] * ad[:, 0]
        cross_ca = cd[:, 0] * ca[:, 1] - cd[:, 1] * ca[:, 0]
        cross_cb = cd[:, 0] * cb[:, 1] - cd[:, 1] * cb[:, 0]
        overlapping_boxes = (
            np.maximum(np.minimum(a[0], b[0]), np.minimum(c[:, 0], d[:, 0]))
            <= np.minimum(np.maximum(a[0], b[0]), np.maximum(c[:, 0], d[:, 0]))
        ) & (
            np.maximum(np.minimum(a[1], b[1]), np.minimum(c[:, 1], d[:, 1]))
            <= np.minimum(np.maximum(a[1], b[1]), np.maximum(c[:, 1], d[:, 1]))
        )
        if np.any(overlapping_boxes & (cross_ac * cross_ad <= 0)
                  & (cross_ca * cross_cb <= 0)):
            return False
    return True


def _capsid_raster_polygon(points: np.ndarray, shape, check_cancel=None) -> np.ndarray:
    """Rasterize at pixel centres with half-open scanline intersections.

    The returned bitmap is the single source for overlay, hit testing and
    picking. Drawing a polygon over a different sampling mask would be unsafe.
    """
    result = np.zeros(shape, dtype=bool)
    edges = np.roll(points, -1, axis=0) - points
    low_y = np.minimum(points[:, 1], points[:, 1] + edges[:, 1])
    high_y = np.maximum(points[:, 1], points[:, 1] + edges[:, 1])
    first = max(0, int(math.ceil(float(low_y.min()) - .5)))
    last = min(shape[0], int(math.ceil(float(high_y.max()) - .5)))
    for y in range(first, last):
        if y % 64 == 0:
            _capsid_check(check_cancel)
        active = (low_y <= y + .5) & (y + .5 < high_y)
        x = points[active, 0] + ((y + .5 - points[active, 1])
                                * edges[active, 0] / edges[active, 1])
        x.sort()
        for left, right in zip(x[::2], x[1::2]):
            begin = max(0, int(math.ceil(float(left) - .5)))
            end = min(shape[1], int(math.ceil(float(right) - .5)))
            result[y, begin:end] = True
    return result


def polygonal_capsid_mask(mask: np.ndarray, *, allowed_mask=None,
                         check_cancel=None) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit bounded polygonal outer contours while retaining internal exclusions.

    Each connected component is fitted separately. A near-convex component
    may receive a simpler hull-derived polygon if its filled support occupies
    at least 80% of that polygon, it adds at most 25% area, and outward growth
    stays within a bounded 4--24 pixel band. Otherwise RDP fits a concave
    polygon with at most 10% area added. Both paths retain at least 90% of the
    original support and preserve connectivity. Tiny threshold pinholes may
    be filled (each <=0.5% of the core, capped at 64 pixels; total <=2%).
    Larger holes and all forbidden pixels stay unpickable. Very thin or
    degenerate shapes fall back to their original outer boundary.

    This regularizes an existing proposal; it does not supply new biological
    recognition evidence or repair a wrongly selected membrane/background.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape, default=0) == 0:
        raise ValueError("Polygon fitting requires a nonempty two-dimensional mask.")
    if allowed_mask is None:
        allowed = np.ones(mask.shape, dtype=bool)
    else:
        allowed = np.asarray(allowed_mask, dtype=bool)
        if allowed.shape != mask.shape:
            raise ValueError("Polygon constraints must match the mask dimensions.")
    source = mask & allowed
    labels, count = ndi.label(source)
    result = np.zeros(mask.shape, dtype=bool)
    details = {"method": "bounded polygon simplification", "components": [],
               "original_area": int(source.sum()), "polygon_area": 0,
               "hard_exclusions_preserved": True,
               "all_internal_exclusions_preserved": True}
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        _capsid_check(check_cancel)
        if bounds is None:
            continue
        # Work only on the component bounding box. This also keeps the native
        # image border exact for truncated capsids without closing it inwards.
        component = labels[bounds] == label
        constraint = allowed[bounds].copy()
        if count > 1:
            # Neither fit may bridge a gap to another connected component,
            # including one just outside this component's tight bounding box.
            halo_bounds = tuple(slice(max(0, part.start - 1),
                                      min(limit, part.stop + 1))
                                for part, limit in zip(bounds, mask.shape))
            halo = labels[halo_bounds]
            other = (halo != 0) & (halo != label)
            excluded = ndi.binary_dilation(other)
            inner = tuple(slice(part.start - halo_part.start,
                                part.stop - halo_part.start)
                          for part, halo_part in zip(bounds, halo_bounds))
            constraint &= ~excluded[inner]
        area = int(component.sum())
        filled = ndi.binary_fill_holes(component)
        holes = filled & ~component
        # Fine texture thresholding can puncture a capsid with grain-sized
        # holes. Remove only tiny holes entirely within the allowed parent;
        # a SAM exclusion, coloured screen mark, or larger dark patch is not
        # reclassified as capsid merely to make the polygon look cleaner.
        hole_labels, _ = ndi.label(holes)
        sizes = np.bincount(hole_labels.ravel())
        hole_limit = max(1, min(64, math.floor(area * .005)))
        budget = math.floor(area * .02)
        filled_hole_pixels = 0
        forbidden_holes = np.bincount(hole_labels[~constraint], minlength=len(sizes)) > 0
        eligible = np.flatnonzero((sizes > 0) & (sizes <= hole_limit) & ~forbidden_holes)
        eligible = eligible[eligible != 0]
        eligible = eligible[np.argsort(sizes[eligible], kind="stable")]
        fill_holes = np.zeros(len(sizes), dtype=bool)
        for index, hole_id in enumerate(eligible):
            if index % 256 == 0:
                _capsid_check(check_cancel)
            size = int(sizes[hole_id])
            if size > budget:
                break
            fill_holes[hole_id] = True
            filled_hole_pixels += size
            budget -= size
        holes &= ~fill_holes[hole_labels]
        rings = _capsid_boundary_rings(filled, check_cancel)
        fitted = filled & ~holes
        metadata = {"applied": False, "strategy": "original mask", "vertices": 0,
                    "tolerance_pixels": 0.0,
                    "filled_pinhole_pixels": filled_hole_pixels,
                    "preserved_hole_pixels": int(holes.sum())}
        if filled_hole_pixels:
            details["all_internal_exclusions_preserved"] = False
        if len(rings) == 1 and len(rings[0]) >= 3:
            boundary = rings[0]
            def fixed_border_vertices(vertices):
                return ((bounds[1].start == 0) & (vertices[:, 0] == 0)
                        | (bounds[1].stop == mask.shape[1]) & (vertices[:, 0] == component.shape[1])
                        | (bounds[0].start == 0) & (vertices[:, 1] == 0)
                        | (bounds[0].stop == mask.shape[0]) & (vertices[:, 1] == component.shape[0]))

            # Many threshold notches are grain variation inside one polygonal
            # core. Prefer a simpler capsid-like hull only when its extent is
            # already mostly supported. Do not convexify deep notches, merged
            # branches, separate components, or missing forbidden regions.
            hull = _capsid_convex_hull(boundary)
            hull_tolerance = max(1.5, min(8., math.sqrt(area) * .04))
            polygon = _capsid_simplify_ring(hull, hull_tolerance, fixed_border_vertices(hull))
            if _capsid_simple_polygon(polygon):
                hull_mask = _capsid_raster_polygon(polygon, component.shape, check_cancel)
                candidate = hull_mask & constraint & ~holes
                retained = int(np.count_nonzero(candidate & component))
                added = int(np.count_nonzero(candidate & ~component))
                if (retained >= .90 * area and added <= .25 * area
                        and int(np.count_nonzero(hull_mask & filled)) >= .80 * int(hull_mask.sum())):
                    distances = ndi.distance_transform_edt(~filled)
                    added_pixels = candidate & ~component
                    growth = float(distances[added_pixels].max()) if added else 0.0
                    if growth <= max(4., min(24., math.sqrt(area) * .20)) and ndi.label(candidate)[1] == 1:
                        fitted = candidate
                        metadata.update(applied=True, strategy="near-convex polygon",
                                        vertices=len(polygon), tolerance_pixels=float(hull_tolerance),
                                        maximum_outward_growth_pixels=growth)
            tolerance = max(2., min(10., math.sqrt(area) * .06))
            fixed = fixed_border_vertices(boundary)
            for attempt in range(0 if metadata["applied"] else 5):
                _capsid_check(check_cancel)
                polygon = _capsid_simplify_ring(rings[0], tolerance, fixed)
                if _capsid_simple_polygon(polygon):
                    candidate = _capsid_raster_polygon(polygon, component.shape, check_cancel)
                    candidate &= constraint & ~holes
                    retained = int(np.count_nonzero(candidate & component))
                    added = int(np.count_nonzero(candidate & ~component))
                    _, pieces = ndi.label(candidate)
                    if retained >= .90 * area and added <= .10 * area and pieces == 1:
                        fitted = candidate
                        metadata.update(applied=True, strategy="concave polygon", vertices=len(polygon),
                                        tolerance_pixels=float(tolerance))
                        break
                tolerance *= .5
        result[bounds] |= fitted
        details["components"].append(metadata)
    details["polygon_area"] = int(result.sum())
    return result, details


def filter_capsid_proposals(
    rgb: np.ndarray,
    raw_proposals: Iterable[Mapping[str, Any]],
    *,
    check_cancel: Callable[[], None] | None = None,
    min_area_ratio: float = .002,
    max_area_ratio: float = .40,
    min_local_texture_ratio: float = 1.18,
    min_global_texture_ratio: float = 1.10,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Select dense interiors of permissive SAM proposals, in evidence order.

    ``raw_proposals`` use SAM's ``segmentation`` dictionary field. Returned
    records have new masks/areas/bboxes, ``capsid_score`` (an uncalibrated rank),
    and numerical ``capsid_evidence``. The inherited SAM quality numbers are
    moved into evidence: they do not estimate the modified core masks' IoU.

    Use permissive SAM proposal thresholds before this function (currently
    .50 predicted IoU / .75 stability); filtering strict generic masks first
    can remove the inner core before the content checks ever see it.

    The work is bounded to 128 area-eligible proposals and 24 returned regions
    by default. Cancellation is checked between feature and proposal passes.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Capsid proposals require an H x W x 3 uint8 RGB image.")
    if min(rgb.shape[:2]) < 4:
        return []
    if not (0 <= min_area_ratio < max_area_ratio <= 1):
        raise ValueError("Capsid area ratios must satisfy 0 <= min < max <= 1.")
    if max_candidates <= 0:
        return []
    _capsid_check(check_cancel)
    shape = rgb.shape[:2]
    pixels = int(np.prod(shape))
    min_area = max(64, math.ceil(min_area_ratio * pixels))
    max_area = max_area_ratio * pixels
    proposals = []
    for proposal in raw_proposals:
        _capsid_check(check_cancel)
        mask = np.asarray(proposal["segmentation"], dtype=bool)
        if mask.shape != shape:
            raise ValueError("A SAM proposal has a different size from the screenshot.")
        area = int(np.count_nonzero(mask))
        if min_area <= area <= max_area:
            proposals.append((proposal, mask, area))
    if not proposals:
        return []
    # Small isolated fragments must not hide larger cores just by arriving
    # first. Bound the expensive per-region passes using proposal objectness.
    proposals.sort(key=lambda entry: float(entry[0].get("predicted_iou") or 0), reverse=True)
    proposals = proposals[:128]
    texture, valid = _capsid_texture(rgb, check_cancel)
    if np.count_nonzero(valid) < min_area:
        return []
    global_median = float(np.median(texture[valid]))
    if global_median < 1e-7:
        return []
    field_bounds = _capsid_field_bounds(valid)
    refined = []
    for proposal, mask, area in proposals:
        _capsid_check(check_cancel)
        screen_sides = _capsid_screen_sides(mask & valid, field_bounds)
        # Background/carbon partitions commonly follow three screen sides.
        # Keep one- or two-sided cropped objects; a very tight crop can need
        # the existing box-prompt mode instead of automatic classification.
        if screen_sides >= 3:
            continue
        # Keep the acquisition/screen edge open for partially visible capsids.
        inset = max(2, min(6, round(math.sqrt(area / math.pi) * .045)))
        interior = ndi.binary_erosion(mask, iterations=inset, border_value=1) & valid
        annulus = (ndi.binary_dilation(mask, iterations=20)
                   & ~ndi.binary_dilation(mask, iterations=3) & valid)
        if np.count_nonzero(interior) < min_area or np.count_nonzero(annulus) < 50:
            continue
        outside = float(np.median(texture[annulus]))
        upper_inside = float(np.quantile(texture[interior], .85))
        upper_outside = float(np.quantile(texture[annulus], .85))
        upper_ratio = upper_inside / max(upper_outside, 1e-7)
        # Compare like quantiles before selecting a bright tail. Without this
        # check even an empty, textured background has a selected upper tail.
        # Unlike an envelope-median gate, a minority dense core can pass.
        if upper_ratio < 1.08:
            continue
        _capsid_check(check_cancel)
        # Separate the robust upper interior signal from the local background
        # at their contrast midpoint. A threshold barely above background
        # would otherwise retain a diffuse halo around a dense core.
        threshold = max(outside * 1.08,
                        (outside + upper_inside) * .5,
                        global_median * 1.02)
        support = interior & (texture > threshold)
        support = ndi.binary_closing(support, iterations=4, border_value=1) & interior
        # Do not fill low-texture holes or excluded pixels: the examples leave
        # dense dark contaminants unpicked, and original SAM holes must remain.
        support &= mask & valid
        labels, count = ndi.label(support)
        component_min = max(min_area, math.ceil(area * .10))
        for label in range(1, count + 1):
            _capsid_check(check_cancel)
            component = labels == label
            component_area = int(np.count_nonzero(component))
            if component_area < component_min:
                continue
            # Score the candidate core itself. A real capsid can occupy a
            # minority of a large envelope, so the envelope-wide median is
            # not a valid prerequisite for finding it.
            inside = float(np.median(texture[component]))
            local_ratio = inside / max(outside, 1e-7)
            global_ratio = inside / global_median
            if local_ratio < min_local_texture_ratio or global_ratio < min_global_texture_ratio:
                continue
            # Density score is explicitly not a probability or SAM IoU.
            density_rank = min(local_ratio, 3.0) * min(global_ratio, 3.0)
            _, _, proposal_width, proposal_height = _capsid_bbox(mask)
            elongation = max(proposal_width, proposal_height) / max(1, min(proposal_width, proposal_height))
            # A soft ordering cue, never a polygonality/roundness exclusion.
            # It prevents a long merged screen-edge region from hiding a
            # compact dense core already present among nested SAM proposals.
            rank = density_rank * math.sqrt(component_area) / max(1.0, elongation)
            result = dict(proposal)
            evidence = {
                "local_texture_ratio": local_ratio,
                "global_texture_ratio": global_ratio,
                "texture_threshold": threshold,
                "upper_texture_ratio": upper_ratio,
                "screen_sides": int(screen_sides),
                "proposal_area": area,
                "core_area": component_area,
                "proposal_predicted_iou": proposal.get("predicted_iou"),
                "proposal_stability_score": proposal.get("stability_score"),
                "is_calibrated_probability": False,
            }
            result.update(segmentation=component, area=component_area,
                          bbox=_capsid_bbox(component), predicted_iou=None,
                          stability_score=None, capsid_score=float(rank),
                          capsid_evidence=evidence,
                          _capsid_parent_mask=mask, _capsid_inset=inset,
                          source="capsid texture core (review)")
            refined.append(result)
    # Resolve duplicate/nested proposals only after the content check. Favour
    # supported target extent over a tiny, very dark patch nested inside it.
    refined.sort(key=lambda record: record["capsid_score"], reverse=True)
    selected = []
    for record in refined:
        _capsid_check(check_cancel)
        mask, area = record["segmentation"], record["area"]
        duplicate = False
        for previous in selected:
            overlap = int(np.count_nonzero(mask & previous["segmentation"]))
            if overlap / min(area, previous["area"]) >= .80:
                duplicate = True
                break
        if not duplicate:
            selected.append(record)
            if len(selected) >= max_candidates:
                break
    # Fit only the final proposals. Texture scoring and duplicate selection
    # remain based on the observed density; polygon shape is not evidence of
    # a capsid. Avoid repeating this work for all nested SAM alternatives.
    for record in selected:
        _capsid_check(check_cancel)
        parent = record.pop("_capsid_parent_mask")
        inset = record.pop("_capsid_inset")
        allowed = ndi.binary_erosion(parent, iterations=inset, border_value=1) & valid
        fitted, polygon_evidence = polygonal_capsid_mask(
            record["segmentation"], allowed_mask=allowed, check_cancel=check_cancel)
        record.update(segmentation=fitted, area=int(fitted.sum()), bbox=_capsid_bbox(fitted),
                      source="capsid polygon (review)")
        record["capsid_evidence"]["polygon"] = polygon_evidence
    return selected



def _profile_shape(mask: np.ndarray, check_cancel=None) -> tuple[bool, dict[str, float]]:
    """Broad compactness support, rejecting long strips and thin membrane rims.

    This is an extent prior, not an exact side-count/roundness template. Holes
    are retained in masks. Filled-extent thickness and supported occupancy are
    separate, so an internal contaminant does not make a valid core a thin rim.
    """
    _capsid_check(check_cancel)
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        return False, {}
    covariance = np.cov(np.stack((xx, yy)))
    eigenvalues = np.linalg.eigvalsh(covariance)
    elongation = math.sqrt(float(eigenvalues[-1]) / max(float(eigenvalues[0]), 1e-6))
    if elongation > 4.0:
        return False, {"moment_elongation": elongation}
    filled = ndi.binary_fill_holes(mask)
    occupancy = len(xx) / int(filled.sum())
    if occupancy < .50:
        return False, {'supported_occupancy': occupancy, 'moment_elongation': elongation}
    distance = ndi.distance_transform_edt(np.pad(filled, 1))[1:-1, 1:-1]
    thickness = float(distance.max()) / math.sqrt(int(filled.sum()))
    if thickness < .24:
        return False, {"moment_elongation": elongation, "normalized_thickness": thickness}
    rings = _capsid_boundary_rings(filled, check_cancel)
    if not rings:
        return False, {}
    hull = _capsid_convex_hull(np.vstack(rings))
    following = np.roll(hull, -1, axis=0)
    hull_area = abs(float(np.sum(hull[:, 0] * following[:, 1] - hull[:, 1] * following[:, 0]))) * .5
    solidity = int(filled.sum()) / max(hull_area, 1)
    evidence = {"moment_elongation": elongation, "normalized_thickness": thickness,
                "filled_support_solidity": solidity, "supported_occupancy": occupancy}
    return solidity >= .72, evidence


def _compact_prompt_seeds(part, bounds, image_shape, min_area, check_cancel=None):
    """Remove thick attachments before asking SAM for a compact object's mask."""
    radius = max(4, min(24, round(math.sqrt(int(part.sum())) * .15)))
    padded = np.pad(ndi.binary_fill_holes(part), radius)
    if bounds[0].start == 0:
        padded[:radius, radius:-radius] = True
    if bounds[0].stop == image_shape[0]:
        padded[-radius:, radius:-radius] = True
    if bounds[1].start == 0:
        padded[radius:-radius, :radius] = True
    if bounds[1].stop == image_shape[1]:
        padded[radius:-radius, -radius:] = True
    _capsid_check(check_cancel)
    opened = ndi.binary_opening(padded, iterations=radius)[radius:-radius, radius:-radius] & part
    labels, count = ndi.label(opened)
    for label in range(1, count + 1):
        _capsid_check(check_cancel)
        seed = labels == label
        if int(seed.sum()) < min_area:
            continue
        accepted, evidence = _profile_shape(seed, check_cancel)
        if accepted:
            yield seed, {**evidence, 'compact_seed_opening_radius': radius}


def _profile_components(support: np.ndarray, min_area: int, max_area: float,
                        check_cancel=None, *, allow_provisional=False):
    """Assess components, optionally deferring modest concavity to SAM boxes.

    Final masks always use the strict default. A provisional prompt may have
    an attached dense neighbour; SAM must separate it before final refinement.
    All area, elongation, occupancy and thickness checks still apply.
    """
    labels, count = ndi.label(support)
    # Object IDs, rather than any positive pixel inside a bounding box, keep
    # nested/disconnected components separate when their boxes overlap.
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        _capsid_check(check_cancel)
        if bounds is None:
            continue
        local = labels[bounds] == label
        area = int(local.sum())
        if not min_area <= area <= max_area:
            continue
        radius = max(2, min(8, round(math.sqrt(area) * .07)))
        # Pad as foreground only at actual image edges. Partial capsids keep
        # their open screen boundary; internal narrow tails are trimmed.
        padded = np.pad(ndi.binary_fill_holes(local), radius)
        if bounds[0].start == 0:
            padded[:radius, radius:-radius] = True
        if bounds[0].stop == support.shape[0]:
            padded[-radius:, radius:-radius] = True
        if bounds[1].start == 0:
            padded[radius:-radius, :radius] = True
        if bounds[1].stop == support.shape[1]:
            padded[radius:-radius, -radius:] = True
        opened = ndi.binary_opening(padded, iterations=radius)[radius:-radius, radius:-radius]
        cleaned = opened & local  # Never grow beyond support or fill an exclusion.
        pieces, number = ndi.label(cleaned)
        for piece in range(1, number + 1):
            _capsid_check(check_cancel)
            part = pieces == piece
            if int(part.sum()) < min_area:
                continue
            accepted, evidence = _profile_shape(part, check_cancel)
            if accepted:
                if allow_provisional:
                    evidence = {**evidence, 'prompt_only': True, 'provisional': False,
                                'prompt_solidity_threshold': .72, 'final_solidity_threshold': .72}
                mask = np.zeros_like(support)
                mask[bounds] = part
                yield mask, evidence
            elif allow_provisional and 'filled_support_solidity' in evidence:
                for seed, shape in _compact_prompt_seeds(part, bounds, support.shape, min_area, check_cancel):
                    shape = {**shape, 'prompt_only': True, 'provisional': True,
                             'rough_support_solidity': evidence['filled_support_solidity'],
                             'prompt_solidity_threshold': .72, 'final_solidity_threshold': .72}
                    mask = np.zeros_like(support)
                    mask[bounds] = seed
                    yield mask, shape


def _capsid_guided_prompts_at_threshold(scores: np.ndarray, *, threshold=.5,
                          allow_provisional=True, min_area_ratio=.002,
                          max_area_ratio=.40, max_prompts=24, check_cancel=None):
    """Run the existing prompt geometry at one appearance threshold."""
    _capsid_check(check_cancel)
    scores = np.asarray(scores)
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).all():
        raise ValueError('Capsid appearance scores must be a finite two-dimensional image.')
    if not 0 <= min_area_ratio < max_area_ratio <= 1:
        raise ValueError('Capsid area ratios must satisfy 0 <= min < max <= 1.')
    if max_prompts <= 0:
        return []
    max_prompts = min(24, max_prompts)
    min_area = max(64, math.ceil(min_area_ratio * scores.size))
    max_area = max_area_ratio * scores.size
    support = ndi.binary_closing(scores >= threshold, iterations=4, border_value=1)
    _capsid_check(check_cancel)
    support &= scores > 0
    support = ndi.binary_opening(support, iterations=3, border_value=0)
    proposals = []
    for mask, shape in _profile_components(support, min_area, max_area, check_cancel,
                                           allow_provisional=allow_provisional):
        _capsid_check(check_cancel)
        x, y, width, height = _capsid_bbox(mask)
        margin = max(6, min(24, round(max(width, height) * .12)))
        box = [max(0, x - margin), max(0, y - margin),
               min(scores.shape[1], x + width + margin),
               min(scores.shape[0], y + height + margin)]
        # One background-pixel halo contains every nearest zero needed by the
        # foreground EDT. Clip at real image edges: padding those with zeros
        # would change point placement for a frame-truncated capsid.
        region = (slice(max(0, y - 1), min(scores.shape[0], y + height + 1)),
                  slice(max(0, x - 1), min(scores.shape[1], x + width + 1)))
        local_mask, local_scores = mask[region], scores[region]
        distances = ndi.distance_transform_edt(local_mask)
        cy, cx = np.unravel_index(np.argmax(distances * local_scores), local_mask.shape)
        cy, cx = cy + region[0].start, cx + region[1].start
        area = int(local_mask.sum())
        rank = float(np.mean(local_scores[local_mask]) * math.sqrt(area))
        proposals.append({'box': box, 'point': [int(cx), int(cy)], 'rank': rank,
                          'support_area': area, 'shape': shape})
    # Existing eligible prompts retain their order and budget. New provisional
    # boxes only fill unused slots, even if a broad component has a high rank.
    return sorted(proposals, key=lambda item: (item['shape'].get('provisional', False), -item['rank']))[:max_prompts]


def _prompt_box_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / max(1, union)


def capsid_guided_prompts(scores: np.ndarray, *, min_area_ratio=.002,
                          max_area_ratio=.40, max_prompts=24, check_cancel=None):
    """Append compact high-score seeds to unused prompt slots.

    The original .5 prompts retain exact geometry, rank and priority. Strict .9
    components may add a tighter box when a broad .5 component merges the capsid
    with surrounding dense texture. Box IoU, not containment, rejects duplicate
    prompts so that a compact box inside a much larger one can still be useful.
    No final acceptance, refinement or size-filter thresholds are changed.
    """
    primary = _capsid_guided_prompts_at_threshold(
        scores, threshold=.5, allow_provisional=True, min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio, max_prompts=max_prompts, check_cancel=check_cancel)
    selected = [{**prompt, 'seed_threshold': .5} for prompt in primary]
    budget = min(24, max_prompts)
    if len(selected) >= budget or budget <= 0:
        return selected
    compact = _capsid_guided_prompts_at_threshold(
        scores, threshold=.9, allow_provisional=False, min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio, max_prompts=24, check_cancel=check_cancel)
    for prompt in compact:
        _capsid_check(check_cancel)
        if any(_prompt_box_iou(prompt['box'], old['box']) >= .80 for old in selected):
            continue
        selected.append({**prompt, 'seed_threshold': .9})
        if len(selected) >= budget:
            break
    return selected


def _profile_pool_offer(pool, record, serial):
    """Retain at most64 full-image candidate records, stable by rank then arrival."""
    entry = (float(record['capsid_score']), -serial, record)
    if len(pool) < 64:
        heapq.heappush(pool, entry)
    else:
        heapq.heappushpop(pool, entry)


def _profile_masks_overlap(first, second):
    ax, ay, aw, ah = first['bbox']
    bx, by, bw, bh = second['bbox']
    if ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay:
        return False
    intersection = (slice(max(ay, by), min(ay + ah, by + bh)),
                    slice(max(ax, bx), min(ax + aw, bx + bw)))
    return (np.count_nonzero(first['segmentation'][intersection]
                            & second['segmentation'][intersection])
            / min(first['area'], second['area']) >= .80)


def _profile_fitted_eligible_records(record, minimum_area_px, check_cancel=None):
    """Yield actual final polygons eligible to participate in overlap suppression.

    The minimum counts four-connected foreground pixels after polygon fitting,
    excluding holes. A small rejected polygon must not suppress a larger usable
    alternative. This is intentionally not a cutoff on the pre-polygon mask.
    """
    _capsid_check(check_cancel)
    fitted, polygon_evidence = polygonal_capsid_mask(
        record['segmentation'], allowed_mask=record['_allowed'], check_cancel=check_cancel)
    labels, count = ndi.label(fitted)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = np.flatnonzero(sizes[1:] >= minimum_area_px) + 1
    for label in eligible:
        _capsid_check(check_cancel)
        component = fitted if count == 1 else labels == label
        evidence = dict(record['capsid_evidence'])
        evidence['polygon'] = polygon_evidence
        evidence['area_filter'] = {
            'minimum_area_px': minimum_area_px, 'source_area': int(sizes[1:].sum()),
            'source_component_count': count, 'component_area': int(sizes[label]),
            'before_overlap_suppression': True,
            'inherited_evidence_applies_to_source_mask': count > 1}
        result = {**record, 'segmentation': component, 'area': int(sizes[label]),
                  'bbox': _capsid_bbox(component), 'capsid_evidence': evidence,
                  'source': 'learned capsid polygon (review)'}
        result.pop('_allowed', None)
        yield result


def filter_capsid_with_profile(rgb: np.ndarray, raw_proposals, scores: np.ndarray, *,
                               check_cancel=None, min_area_ratio=.002,
                               max_area_ratio=.40, max_candidates=24, min_capsid_area_px=0):
    """Refine guided SAM masks with the same learned support used for prompting.

    The resulting bitmap is used for overlay, hit testing and picking. This
    path keeps the geometric exclusions and never inherits SAM IoU estimates
    as scores for the modified region. Appearance scores are uncalibrated.
    """
    rgb = np.asarray(rgb)
    scores = np.asarray(scores)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or scores.shape != rgb.shape[:2] or not np.isfinite(scores).all()):
        raise ValueError('Capsid profile scores and screenshot must have matching image dimensions.')
    if not 0 <= min_area_ratio < max_area_ratio <= 1:
        raise ValueError('Capsid area ratios must satisfy 0 <= min < max <= 1.')
    if (isinstance(min_capsid_area_px, bool) or not isinstance(min_capsid_area_px, int)
            or min_capsid_area_px < 0):
        raise ValueError('min_capsid_area_px must be a nonnegative integer.')
    if max_candidates <= 0:
        return []
    max_candidates = min(24, max_candidates)
    min_area = max(64, math.ceil(min_area_ratio * scores.size))
    max_area = max_area_ratio * scores.size
    texture, valid = _capsid_texture(rgb, check_cancel)
    if np.count_nonzero(valid) < min_area:
        return []
    global_median = float(np.median(texture[valid]))
    if global_median < 1e-7:
        return []
    bounds = _capsid_field_bounds(valid)
    refined = []
    serial = 0
    for index, proposal in enumerate(raw_proposals):
        _capsid_check(check_cancel)
        if index >= 128:
            break
        parent = np.asarray(proposal['segmentation'], dtype=bool)
        if parent.shape != scores.shape:
            raise ValueError('A guided SAM proposal has a different size from the screenshot.')
        area = int(parent.sum())
        if not min_area <= area <= max_area or _capsid_screen_sides(parent & valid, bounds) >= 3:
            continue
        inset = max(2, min(6, round(math.sqrt(area / math.pi) * .045)))
        allowed = ndi.binary_erosion(parent, iterations=inset, border_value=1) & valid
        support = ndi.binary_closing(allowed & (scores >= .5), iterations=4, border_value=1) & allowed
        for component, shape in _profile_components(support, min_area, max_area, check_cancel):
            component_area = int(component.sum())
            bbox = _capsid_bbox(component)
            x, y, width, height = bbox
            # A dilation cannot reach beyond its iteration radius. Work on
            # that exact halo instead of traversing the whole screenshot for
            # every alternative SAM mask; preserve real image-edge clipping.
            region = (slice(max(0, y - 20), min(scores.shape[0], y + height + 20)),
                      slice(max(0, x - 20), min(scores.shape[1], x + width + 20)))
            local_component = component[region]
            annulus = (ndi.binary_dilation(local_component, iterations=20)
                       & ~ndi.binary_dilation(local_component, iterations=3) & valid[region])
            inside = float(np.median(texture[component]))
            outside = float(np.median(texture[region][annulus])) if np.any(annulus) else global_median
            local_ratio = inside / max(outside, 1e-7)
            global_ratio = inside / global_median
            # Require either a local grain increase or strong field-relative
            # grain. This removes flat rims while allowing crowded neighbours.
            if local_ratio < 1.04 and global_ratio < 1.15:
                continue
            mean_score = float(np.mean(scores[component]))
            rank = mean_score * math.sqrt(component_area) / max(1, shape['moment_elongation'])
            evidence = {'appearance_score_mean': mean_score,
                        'appearance_score_median': float(np.median(scores[component])),
                        'appearance_threshold': .5, 'shape': shape,
                        'local_texture_ratio': local_ratio, 'global_texture_ratio': global_ratio,
                        'proposal_predicted_iou': proposal.get('predicted_iou'),
                        'proposal_stability_score': proposal.get('stability_score'),
                        'is_calibrated_probability': False}
            record = {'segmentation': component, 'area': component_area,
                            'bbox': bbox, 'predicted_iou': None,
                            'stability_score': None, 'capsid_score': rank,
                            'capsid_evidence': evidence, '_allowed': allowed,
                            'source': 'learned capsid core (review)'}
            _profile_pool_offer(refined, record, serial)
            serial += 1
    ranked = [entry[2] for entry in sorted(refined, key=lambda entry: (entry[0], entry[1]), reverse=True)]
    refined.clear()
    selected = []
    for record in ranked:
        _capsid_check(check_cancel)
        if min_capsid_area_px:
            for fitted in _profile_fitted_eligible_records(record, min_capsid_area_px, check_cancel):
                if any(_profile_masks_overlap(fitted, previous) for previous in selected):
                    continue
                selected.append(fitted)
                if len(selected) >= max_candidates:
                    break
            if len(selected) >= max_candidates:
                break
            continue
        if any(_profile_masks_overlap(record, previous) for previous in selected):
            continue
        selected.append(record)
        if len(selected) >= max_candidates:
            break
    ranked.clear()
    if min_capsid_area_px:
        return selected
    for record in selected:
        fitted, evidence = polygonal_capsid_mask(record['segmentation'],
                                                allowed_mask=record.pop('_allowed'),
                                                check_cancel=check_cancel)
        record.update(segmentation=fitted, area=int(fitted.sum()), bbox=_capsid_bbox(fitted),
                      source='learned capsid polygon (review)')
        record['capsid_evidence']['polygon'] = evidence
    return selected


# ---------------- recognition.py ----------------

from dataclasses import dataclass
import gc
import logging
import time
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


CancelCheck = Callable[[], bool] | None
DEFAULT_MIN_CAPSID_AREA_PX = 3500


class RecognitionError(RuntimeError):
    """A model, input, or resource failure with an actionable message."""


class Cancelled(RecognitionError):
    """The caller cancelled an operation; no result should be acted upon."""


def _check_cancel(cancel: CancelCheck) -> None:
    if cancel is not None and cancel():
        raise Cancelled("Operation cancelled.")


@dataclass(frozen=True)
class RecognitionConfig:
    checkpoint: str
    model_type: str = "vit_b"
    device: str = "auto"
    cpu_threads: int = 4
    points_per_side: int = 8
    points_per_batch: int | None = None
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.90
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.50
    max_border_fraction: float = 0.25
    exclude_border: bool = False
    overlap_threshold: float = 0.80
    prefer_smaller: bool = True
    automatic_profile: str = "capsid"
    capsid_prior_path: str | None = None
    min_capsid_area_px: int = DEFAULT_MIN_CAPSID_AREA_PX
    backend: str = "sam"
    cryolo_config: str | None = None
    cryolo_python: str | None = None
    cryolo_threshold: float = 0.3
    cryolo_timeout: float = 180.0

    def __post_init__(self) -> None:
        if self.backend not in {"sam", "cryolo"}:
            raise ValueError("backend must be sam or cryolo.")
        for name in ("cryolo_config", "cryolo_python"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty path string or None.")
        if (isinstance(self.cryolo_threshold, bool) or not math.isfinite(self.cryolo_threshold)
                or not 0 <= self.cryolo_threshold <= 1):
            raise ValueError("cryolo_threshold must be between 0 and 1.")
        if (isinstance(self.cryolo_timeout, bool) or not math.isfinite(self.cryolo_timeout)
                or self.cryolo_timeout <= 0):
            raise ValueError("cryolo_timeout must be positive and finite.")
        if self.model_type not in {"vit_b", "vit_l", "vit_h"}:
            raise ValueError("model_type must be vit_b, vit_l, or vit_h.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.automatic_profile not in {"capsid", "capsid-learned", "generic"}:
            raise ValueError("automatic_profile must be capsid, capsid-learned, or generic.")
        if self.capsid_prior_path is not None and (not isinstance(self.capsid_prior_path, str) or not self.capsid_prior_path.strip()):
            raise ValueError("capsid_prior_path must be a nonempty path string or None.")
        if (isinstance(self.min_capsid_area_px, bool)
                or not isinstance(self.min_capsid_area_px, int) or self.min_capsid_area_px < 0):
            raise ValueError("min_capsid_area_px must be a nonnegative integer; 0 disables the size filter.")
        for name in ("cpu_threads", "points_per_side"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.points_per_side > 64:
            raise ValueError("points_per_side above 64 is disabled to limit memory use.")
        if self.points_per_batch is not None and (
            isinstance(self.points_per_batch, bool)
            or not isinstance(self.points_per_batch, int)
            or self.points_per_batch < 1
        ):
            raise ValueError("points_per_batch must be a positive integer or None.")
        for name in ("pred_iou_thresh", "stability_score_thresh", "min_area_ratio",
                     "max_area_ratio", "max_border_fraction", "overlap_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not self.min_area_ratio < self.max_area_ratio:
            raise ValueError("min_area_ratio must be smaller than max_area_ratio.")


@dataclass(frozen=True)
class Candidate:
    id: str
    mask: np.ndarray
    predicted_iou: float | None
    stability_score: float | None
    area: int
    bbox: tuple[int, int, int, int]
    source: str


def validate_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    """Return contiguous uint8 RGB; reject silently mis-scaled image data."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input must be an H x W x 3 uint8 RGB image.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Input image is empty.")
    return np.ascontiguousarray(arr)


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("A mask must be a nonempty two-dimensional array.")
    if arr.dtype != np.bool_:
        if not np.isin(arr, (0, 1)).all():
            raise ValueError("Masks must contain only Boolean or 0/1 values.")
        arr = arr.astype(bool)
    return arr


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1),
            int(ys[-1] - ys[0] + 1))


def make_candidate(mask: np.ndarray, *, predicted_iou: float | None = None,
                   stability_score: float | None = None, source: str = "manual",
                   image_key: str = "") -> Candidate:
    """Copy a mask into an immutable candidate with a content-derived stable ID."""
    mask = np.ascontiguousarray(_validate_mask(mask)).copy()
    area = int(np.count_nonzero(mask))
    if area == 0:
        raise ValueError("Cannot create an empty candidate.")
    for label, score in (("predicted_iou", predicted_iou),
                         ("stability_score", stability_score)):
        if score is not None and not math.isfinite(float(score)):
            raise ValueError(f"{label} must be finite or None.")
    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(image_key.encode("ascii"))
    digest.update(np.packbits(mask).tobytes())
    mask.flags.writeable = False
    return Candidate(digest.hexdigest(), mask,
                     None if predicted_iou is None else float(predicted_iou),
                     None if stability_score is None else float(stability_score),
                     area, _bbox(mask), source)


def _image_key(rgb: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(memoryview(rgb))
    return digest.hexdigest()


def filter_candidates(candidates: Iterable[Candidate], config: RecognitionConfig,
                      cancel: CancelCheck = None, *,
                      suppress_duplicates: bool = True) -> list[Candidate]:
    """Geometric filtering only: this does not classify inner capsids.

    Duplicate suppression uses intersection / smaller-mask area.  With
    prefer_smaller=True, a nested inner mask can suppress its enclosing mask.
    This is a configurable heuristic: a small wrong fragment can also win.
    """
    survivors: list[Candidate] = []
    shape = None
    for candidate in candidates:
        _check_cancel(cancel)
        mask = _validate_mask(candidate.mask)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise ValueError("All candidate masks must have the same shape.")
        area = int(mask.sum())
        ratio = area / mask.size
        if not config.min_area_ratio <= ratio <= config.max_area_ratio:
            continue
        border = np.zeros_like(mask)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        border_occupancy = int(np.count_nonzero(mask & border))
        if config.exclude_border and border_occupancy:
            continue
        if border_occupancy / int(border.sum()) > config.max_border_fraction:
            continue
        survivors.append(candidate)

    if not suppress_duplicates:
        # The capsid path has already resolved nested proposals by image
        # evidence. Do not run the generic smallest-mask preference again.
        return sorted(survivors, key=lambda c: (c.bbox[1], c.bbox[0], c.id))

    def quality(candidate: Candidate) -> float:
        return candidate.predicted_iou if candidate.predicted_iou is not None else -1.0

    if config.prefer_smaller:
        survivors.sort(key=lambda c: (c.area, -quality(c)))
    else:
        survivors.sort(key=lambda c: (-quality(c), -c.area))
    kept: list[Candidate] = []
    for candidate in survivors:
        _check_cancel(cancel)
        duplicate = False
        for other in kept:
            ax, ay, aw, ah = candidate.bbox
            bx, by, bw, bh = other.bbox
            if ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay:
                continue
            overlap = int(np.count_nonzero(candidate.mask & other.mask))
            if overlap and overlap / min(candidate.area, other.area) >= config.overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    # Stable spatial presentation, independent of score tie ordering.
    return sorted(kept, key=lambda c: (c.bbox[1], c.bbox[0], c.id))


def _filter_capsid_area(proposals, minimum_area_px: int, cancel: CancelCheck = None):
    """Keep large final connected contours; never pool disconnected small pieces.

    This runs only after automatic capsid refinement and carbon rejection.
    Holes stay excluded, and an unchanged single component keeps its original
    mask object. Upstream proposal bounds remain in the capsid refiners; this
    final filter does not impose another cap on already eligible components.
    """
    summary = {'enabled': minimum_area_px > 0, 'minimum_area_px': minimum_area_px,
               'units': 'screen pixels', 'input_count': len(proposals),
               'connected_components': 0, 'rejected_small_components': 0,
               'removed_small_pixels': 0, 'empty_proposals': 0,
               'split_proposals': 0, 'retained_count': 0}
    if not minimum_area_px:
        summary['retained_count'] = len(proposals)
        return proposals, summary
    retained = []
    for record in proposals:
        _check_cancel(cancel)
        mask = _validate_mask(record['segmentation'])
        labels, count = ndi.label(mask)  # Four-connected, as in both capsid refiners.
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        summary['connected_components'] += count
        if not count:
            summary['empty_proposals'] += 1
            continue
        eligible = np.flatnonzero(sizes[1:] >= minimum_area_px) + 1
        small = sizes[1:] < minimum_area_px
        summary['rejected_small_components'] += int(np.count_nonzero(small))
        summary['removed_small_pixels'] += int(sizes[1:][small].sum())
        if count > 1 and len(eligible):
            summary['split_proposals'] += 1
        for label in eligible:
            _check_cancel(cancel)
            if count == 1:
                retained.append(record)
                continue
            component = labels == label
            evidence = dict(record.get('capsid_evidence') or {})
            evidence['area_filter'] = {'minimum_area_px': minimum_area_px,
                                      'source_area': int(sizes[1:].sum()),
                                      'source_component_count': count,
                                      'component_area': int(sizes[label]),
                                      'inherited_evidence_applies_to_source_mask': True}
            retained.append({**record, 'segmentation': component,
                             'area': int(sizes[label]), 'bbox': list(_bbox(component)),
                             'capsid_evidence': evidence})
    summary['retained_count'] = len(retained)
    return retained, summary


def create_recognizer(config: RecognitionConfig):
    """Select the explicit backend; a missing crYOLO model never falls back to SAM."""
    if config.backend == "cryolo":
        return CryoloRecognizer(config)
    return SamRecognizer(config)


class SamRecognizer:
    """One worker's reusable model; call serially, never from a Tk callback.

    Cancellation is checked around image encoding and between automatic prompt
    batches.  A running encoder/CUDA kernel cannot be interrupted here; the
    application must discard stale results when a request is cancelled.
    """
    def __init__(self, config: RecognitionConfig):
        self.config = config
        self._torch = None
        self._model = None
        self._predictor = None
        self._generator = None
        self._predictor_image_key: str | None = None
        self._automatic_image_key: str | None = None
        self._automatic_config: RecognitionConfig | None = None
        self._automatic_result: list[Candidate] | None = None
        self.last_detection: dict | None = None
        self._appearance_profile = None
        self._appearance_profile_loaded = False
        self._active_cancel: CancelCheck = None
        self.device: str | None = None

    def _ensure_model(self, cancel: CancelCheck) -> None:
        _check_cancel(cancel)
        if self._predictor is not None:
            return
        checkpoint = Path(self.config.checkpoint).expanduser()
        if not checkpoint.is_file():
            raise RecognitionError(f"SAM checkpoint not found: {checkpoint}")
        known_name = re.search(r"sam_(vit_[blh])_", checkpoint.name)
        if known_name and known_name.group(1) != self.config.model_type:
            raise RecognitionError(
                f"Checkpoint filename indicates {known_name.group(1)}, but "
                f"{self.config.model_type} is selected. Select matching model type.")
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except (ImportError, RuntimeError, OSError) as exc:
            raise RecognitionError(
                "SAM dependencies could not be imported. Install compatible torch "
                "and torchvision builds, then install Meta's segment-anything. "
                f"Original error: {exc}") from exc
        requested = self.config.device
        if requested == "cuda" and not torch.cuda.is_available():
            raise RecognitionError("CUDA was selected, but PyTorch cannot access a CUDA device.")
        device = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # PyTorch's thread setting is process-wide; use a dedicated worker.
            torch.set_num_threads(min(self.config.cpu_threads, os.cpu_count() or 1))
        try:
            _check_cancel(cancel)
            # Building with checkpoint=None avoids SAM's legacy unrestricted
            # torch.load path. Never retry with weights_only=False.
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            _check_cancel(cancel)
            model = sam_model_registry[self.config.model_type](checkpoint=None)
            model.load_state_dict(state, strict=True)
            del state
            model.eval()
            model.to(device=device)
            predictor = SamPredictor(model)
            owner = self

            class CancellableGenerator(SamAutomaticMaskGenerator):
                # SAM's public generate() has no cancellation hook. This narrow
                # extension matches the upstream _process_batch(*args) API.
                def _process_batch(self, *args, **kwargs):
                    _check_cancel(owner._active_cancel)
                    result = super()._process_batch(*args, **kwargs)
                    _check_cancel(owner._active_cancel)
                    return result

            generator = CancellableGenerator(
                model=model,
                points_per_side=self.config.points_per_side,
                points_per_batch=self.config.points_per_batch or (32 if device == "cuda" else 8),
                # The capsid pass needs inner candidates that generic SAM's
                # stability gate can omit. Domain filtering follows inference;
                # these permissive proposal thresholds are not acceptance scores.
                pred_iou_thresh=(0.50 if self.config.automatic_profile in {"capsid", "capsid-learned"}
                                 else self.config.pred_iou_thresh),
                stability_score_thresh=(0.75 if self.config.automatic_profile in {"capsid", "capsid-learned"}
                                       else self.config.stability_score_thresh),
                box_nms_thresh=0.7,
                crop_n_layers=0,
                min_mask_region_area=0,  # Preserve holes and disconnected regions.
                output_mode="binary_mask",
            )
            _check_cancel(cancel)
        except Cancelled:
            raise
        except TypeError as exc:
            raise RecognitionError(
                "This PyTorch/SAM combination could not load the checkpoint safely. "
                "Use a PyTorch version supporting torch.load(weights_only=True). "
                f"Original error: {exc}") from exc
        except Exception as exc:
            raise RecognitionError(
                f"Could not load SAM {self.config.model_type} on {device}: {exc}. "
                "Check the checkpoint/model match; choose CPU if GPU memory is insufficient.") from exc
        self._torch, self._model = torch, model
        self._predictor, self._generator = predictor, generator
        self.device = device

    def _set_image(self, rgb: np.ndarray, key: str, cancel: CancelCheck) -> None:
        _check_cancel(cancel)
        if key != self._predictor_image_key:
            # Do not mark the new embedding as cached until encoding succeeds.
            self._predictor_image_key = None
            self._predictor.set_image(rgb, image_format="RGB")
            _check_cancel(cancel)
            self._predictor_image_key = key

    def prompt(self, rgb: np.ndarray | Image.Image,
               box: Sequence[float] | None = None,
               points: Sequence[Sequence[float]] | None = None,
               labels: Sequence[int] | None = None,
               cancel: CancelCheck = None) -> list[Candidate]:
        """Return up to three mask alternatives, best predicted IoU first.

        Coordinates are image-local (x, y). Box is (x0, y0, x1, y1).
        Point label 1 means foreground; 0 means background. A box or at least
        one positive point is required. No automatic shape filter is applied to
        prompted masks because the user, not an area heuristic, selects them.
        """
        rgb = validate_rgb(rgb)
        height, width = rgb.shape[:2]
        point_array = label_array = box_array = None
        if points is not None:
            point_array = np.asarray(points, dtype=np.float32)
            if point_array.ndim != 2 or point_array.shape[1:] != (2,) or len(point_array) == 0:
                raise ValueError("points must be a nonempty N x 2 list of image-local coordinates.")
            label_array = np.asarray(labels) if labels is not None else np.ones(len(point_array), dtype=np.int32)
            if label_array.shape != (len(point_array),) or not np.isin(label_array, (0, 1)).all():
                raise ValueError("Provide one label (0 or 1) per prompt point.")
            if (not np.isfinite(point_array).all()
                or (point_array < 0).any()
                or (point_array[:, 0] >= width).any()
                or (point_array[:, 1] >= height).any()):
                raise ValueError("Prompt points must lie inside the image.")
            label_array = label_array.astype(np.int32)
        elif labels is not None:
            raise ValueError("Point labels were supplied without points.")
        if box is not None:
            box_array = np.asarray(box, dtype=np.float32)
            if (box_array.shape != (4,) or not np.isfinite(box_array).all()
                or not 0 <= box_array[0] < box_array[2] <= width
                or not 0 <= box_array[1] < box_array[3] <= height):
                raise ValueError("Box must be (x0,y0,x1,y1), with positive size, inside the image.")
        if box_array is None and (label_array is None or not label_array.any()):
            raise ValueError("Supply a box or at least one positive point.")
        self._ensure_model(cancel)
        key = _image_key(rgb)
        try:
            with self._torch.inference_mode():
                self._set_image(rgb, key, cancel)
                _check_cancel(cancel)
                masks, scores, _ = self._predictor.predict(
                    point_coords=point_array, point_labels=label_array,
                    box=box_array, multimask_output=True)
                _check_cancel(cancel)
            result = []
            for mask, score in zip(masks, scores):
                if np.asarray(mask).shape != (height, width):
                    raise RecognitionError("SAM returned a mask with the wrong image dimensions.")
                if np.any(mask):
                    result.append(make_candidate(mask, predicted_iou=float(score),
                                                 source="prompt", image_key=key))
            return sorted(result, key=lambda c: c.predicted_iou, reverse=True)
        except (Cancelled, RecognitionError):
            raise
        except Exception as exc:
            raise RecognitionError(f"SAM prompted segmentation failed: {exc}") from exc

    def _get_appearance_profile(self):
        """Load a stable optional numeric profile once per recognizer instance."""
        if self.config.automatic_profile != "capsid-learned":
            return None
        if not self._appearance_profile_loaded:
            try:
                path = resolve_profile_path(self.config.checkpoint, self.config.capsid_prior_path)
                if path is None:
                    raise ValueError("The experimental capsid profile is not installed. Keep capsid-profile.json beside Setup and reinstall, Select capsid for the earlier texture detector.")
                self._appearance_profile = load_profile(path)
            except (OSError, ValueError) as exc:
                raise RecognitionError(
                    f"Could not load the capsid appearance profile: {exc}. "
                    "Reinstall the accompanying capsid-profile.json. Select capsid for the earlier texture detector.") from exc
            self._appearance_profile_loaded = True
            if self._appearance_profile is not None:
                logging.getLogger("saru.recognition").info(
                    "Using capsid appearance profile %s from %s", self._appearance_profile.sha256,
                    self._appearance_profile.path)
            else:
                logging.getLogger("saru.recognition").info(
                    "No capsid appearance sidecar found; using the existing SAM texture filter.")
        return self._appearance_profile

    def automatic(self, rgb: np.ndarray | Image.Image,
                  cancel: CancelCheck = None) -> list[Candidate]:
        """Generate reviewable regions with a private learned profile when present.

        The opt-in capsid-learned path proposes at most 24 boxes from positive
        appearance support and encodes the screenshot once. Default capsid mode
        retains its earlier sparse-grid texture path. A missing selected learned
        profile raises an actionable error; generic and user prompts are unchanged.
        """
        rgb = validate_rgb(rgb)
        _check_cancel(cancel)
        key = _image_key(rgb)
        if (key == self._automatic_image_key and self._automatic_result is not None
                and self._automatic_config == self.config):
            return list(self._automatic_result)
        profile = self._get_appearance_profile()
        started = time.perf_counter()
        # Validate and score before loading SAM: oversized/corrupt inputs fail
        # before expensive model allocation, and cancellation stays responsive.
        appearance_scores = None
        guided = []
        feature_seconds = proposal_seconds = 0.0
        if profile is not None:
            feature_started = time.perf_counter()
            try:
                appearance_scores = score_image(rgb, profile, lambda: _check_cancel(cancel))
                feature_seconds = time.perf_counter() - feature_started
                proposal_started = time.perf_counter()
                guided = capsid_guided_prompts(
                    appearance_scores, check_cancel=lambda: _check_cancel(cancel),
                    min_area_ratio=self.config.min_area_ratio,
                    max_area_ratio=min(self.config.max_area_ratio, .40))
                proposal_seconds = time.perf_counter() - proposal_started
            except Cancelled:
                raise
            except (ValueError, MemoryError) as exc:
                raise RecognitionError(f"Capsid appearance scoring failed: {exc}") from exc
        self._ensure_model(cancel)
        self._active_cancel = cancel
        try:
            sam_started = time.perf_counter()
            with self._torch.inference_mode():
                if profile is None:
                    raw = self._generator.generate(rgb)
                else:
                    raw = []
                    if guided:
                        self._set_image(rgb, key, cancel)
                    for prompt in guided:
                        _check_cancel(cancel)
                        masks, scores, _ = self._predictor.predict(
                            point_coords=np.asarray([prompt['point']], dtype=np.float32),
                            point_labels=np.asarray([1], dtype=np.int32),
                            box=np.asarray(prompt['box'], dtype=np.float32),
                            multimask_output=True)
                        _check_cancel(cancel)
                        for mask, score in zip(masks, scores):
                            if float(score) >= .50:
                                raw.append({'segmentation': mask, 'predicted_iou': float(score),
                                            'source': 'learned capsid box prompt'})
            sam_seconds = time.perf_counter() - sam_started
            _check_cancel(cancel)
            for item in raw:
                _check_cancel(cancel)
                mask = _validate_mask(item["segmentation"])
                if mask.shape != rgb.shape[:2]:
                    raise RecognitionError("SAM returned a mask with the wrong image dimensions.")
            raw_count = len(raw)
            is_capsid = self.config.automatic_profile in {"capsid", "capsid-learned"}
            refinement_started = time.perf_counter()
            carbon_seconds = 0.0
            carbon_summary = {"enabled": False}
            area_seconds = 0.0
            area_summary = {'enabled': False, 'minimum_area_px': 0, 'reason': 'generic profile'}
            if is_capsid:
                if profile is not None:
                    raw = filter_capsid_with_profile(
                        rgb, raw, appearance_scores, check_cancel=lambda: _check_cancel(cancel),
                        min_area_ratio=self.config.min_area_ratio,
                        max_area_ratio=min(self.config.max_area_ratio, .40),
                        min_capsid_area_px=self.config.min_capsid_area_px)
                else:
                    raw = filter_capsid_proposals(
                        rgb, raw, check_cancel=lambda: _check_cancel(cancel),
                        min_area_ratio=self.config.min_area_ratio,
                        max_area_ratio=min(self.config.max_area_ratio, .40))
                # Adjudicate the extracted interiors, not their broad SAM
                # parents: an edge-bound parent can contain a valid capsid.
                # Retained masks and their order are never modified here.
                carbon_started = time.perf_counter()
                context = detect_carbon_edges(rgb, check_cancel=lambda: _check_cancel(cancel))
                raw, carbon_summary = guard_proposals(
                    context, raw, check_cancel=lambda: _check_cancel(cancel))
                carbon_summary["enabled"] = True
                carbon_seconds = time.perf_counter() - carbon_started
                for item in raw:
                    _check_cancel(cancel)
                    if _validate_mask(item['segmentation']).shape != rgb.shape[:2]:
                        raise RecognitionError("A refined capsid mask has the wrong image dimensions.")
                area_started = time.perf_counter()
                raw, area_summary = _filter_capsid_area(raw, self.config.min_capsid_area_px, cancel)
                area_seconds = time.perf_counter() - area_started
            candidates = []
            evidence = {}
            for item in raw:
                _check_cancel(cancel)
                mask = np.asarray(item["segmentation"])
                if mask.shape != rgb.shape[:2]:
                    raise RecognitionError("SAM returned a mask with the wrong image dimensions.")
                if np.any(mask):
                    candidate = make_candidate(
                        mask, predicted_iou=item.get("predicted_iou"),
                        stability_score=item.get("stability_score"),
                        source=item.get("source", "automatic proposal"), image_key=key)
                    candidates.append(candidate)
                    if item.get("capsid_evidence") is not None:
                        evidence[candidate.id] = dict(item["capsid_evidence"])
            result = filter_candidates(candidates, self.config, cancel,
                                       suppress_duplicates=not is_capsid)
            _check_cancel(cancel)
            appearance = (profile.describe() if profile is not None else
                          {'status': 'disabled', 'fallback_used': False})
            if profile is not None:
                appearance.update(inference_executed=True, scored_pixels=int(rgb.shape[0] * rgb.shape[1]),
                                  guided_boxes=len(guided), fallback_used=False)
            self.last_detection = {
                "profile": self.config.automatic_profile,
                "sam_proposals": raw_count, "selected_contours": len(result),
                "appearance_model": appearance,
                "carbon_guard": carbon_summary,
                "capsid_area_filter": area_summary,
                "timing_seconds": {"appearance": feature_seconds, "guided_proposals": proposal_seconds,
                                   "sam": sam_seconds, "carbon_guard": carbon_seconds, "capsid_area_filter": area_seconds,
                                   "refinement": time.perf_counter() - refinement_started,
                                   "total_including_initial_model_load": time.perf_counter() - started},
                "capsid_evidence": {c.id: evidence[c.id] for c in result if c.id in evidence},
                "note": ("Positive-unlabeled appearance profile guides SAM boxes and bounded polygonal cores; "
                         "scores are not calibrated capsid probabilities."
                         if profile is not None else
                         "Dense-core texture filter with bounded polygonal boundaries; no trained appearance profile."
                         if is_capsid else "Unclassified generic SAM proposals."),
            }
            self._automatic_image_key, self._automatic_result = key, result
            self._automatic_config = self.config
            return list(result)
        except (Cancelled, RecognitionError):
            raise
        except Exception as exc:
            raise RecognitionError(
                f"SAM automatic proposals failed: {exc}. For memory errors, reduce "
                "the selected screen area or points_per_batch; for missing targets, use a box/point prompt.") from exc
        finally:
            self._active_cancel = None

    def close(self) -> None:
        """Release model/cache resources when no inference is running."""
        torch, device = self._torch, self.device
        self._model = self._predictor = self._generator = None
        self._predictor_image_key = self._automatic_image_key = None
        self._automatic_config = None
        self._automatic_result = None
        self.last_detection = None
        self._torch, self.device = None, None
        gc.collect()
        if torch is not None and device == "cuda":
            torch.cuda.empty_cache()


def polygon_mask(points: Sequence[Sequence[float]], shape: Sequence[int]) -> np.ndarray:
    """Rasterize a manually traced polygon; shape is (height, width)."""
    if len(shape) != 2 or any(int(x) != x or x < 1 for x in shape):
        raise ValueError("shape must be positive integer (height, width).")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("A polygon needs at least three finite (x,y) points.")
    if len(np.unique(points, axis=0)) < 3:
        raise ValueError("A polygon needs at least three distinct points.")
    if np.linalg.matrix_rank(points - points[0]) < 2:
        raise ValueError("A polygon must enclose an area, not a straight line.")
    height, width = map(int, shape)
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in points], fill=1)
    return np.asarray(canvas, dtype=bool).copy()


def union_masks(masks: Iterable[Candidate | np.ndarray],
                shape: Sequence[int] | None = None) -> np.ndarray:
    """Union exact bitmaps so overlapping candidates cannot double-pick pixels."""
    union = None
    if shape is not None:
        if len(shape) != 2 or any(int(x) != x or x < 1 for x in shape):
            raise ValueError("shape must be positive integer (height, width).")
        union = np.zeros(tuple(map(int, shape)), dtype=bool)
    for item in masks:
        mask = _validate_mask(item.mask if isinstance(item, Candidate) else item)
        if union is None:
            union = np.zeros_like(mask)
        elif union.shape != mask.shape:
            raise ValueError("All masks must have the same image dimensions.")
        np.logical_or(union, mask, out=union)
    if union is None:
        raise ValueError("Provide at least one mask, or supply shape for an empty union.")
    return union


def sample_hex_grid(mask: np.ndarray, spacing: int, inset: float = 0,
                    max_points: int = 20000, origin: Sequence[float] = (0, 0),
                    cancel: CancelCheck = None) -> np.ndarray:
    """Sample an image-local, staggered grid using direct bitmap lookup.

    The lattice origin belongs to the whole ROI, not each candidate's bbox.
    All selected masks should be unioned before this function is called. Holes
    and disconnected components survive. inset is an inward Euclidean distance
    in image pixels (not Angstroms); boundaries of the image count as background.
    Integer screen sampling rounds the ideal hex lattice to the nearest pixel.
    Raises when max_points would be exceeded; never silently truncates picks.
    """
    _check_cancel(cancel)
    mask = _validate_mask(mask)
    if isinstance(spacing, bool) or not isinstance(spacing, (int, np.integer)) or spacing < 2:
        raise ValueError("spacing must be an integer of at least 2 pixels.")
    if not math.isfinite(inset) or inset < 0:
        raise ValueError("inset must be finite and nonnegative.")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")
    if len(origin) != 2 or not all(math.isfinite(v) for v in origin):
        raise ValueError("origin must contain two finite coordinates.")
    if inset:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise RecognitionError("Install scipy to use an inset margin.") from exc
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        distance = distance_transform_edt(padded)[1:-1, 1:-1]
        mask = mask & (distance > inset)
        _check_cancel(cancel)
    x0, y0, width, height = _bbox(mask)
    if not width:
        return np.empty((0, 2), dtype=np.int32)
    ox, oy = map(float, origin)
    row_step = float(spacing) * math.sqrt(3) / 2
    start_row = math.floor((y0 - oy) / row_step) - 1
    end_row = math.ceil((y0 + height - 1 - oy) / row_step) + 1
    result: list[np.ndarray] = []
    total = 0
    for row in range(start_row, end_row + 1):
        _check_cancel(cancel)
        y = int(round(oy + row * row_step))
        if not y0 <= y < y0 + height:
            continue
        offset = ox + (spacing / 2 if row % 2 else 0)
        first = math.floor((x0 - offset) / spacing) - 1
        last = math.ceil((x0 + width - 1 - offset) / spacing) + 1
        xs = np.rint(offset + np.arange(first, last + 1, dtype=np.float64) * spacing).astype(np.int64)
        xs = xs[(xs >= x0) & (xs < x0 + width)]
        xs = xs[mask[y, xs]]
        total += len(xs)
        if total > max_points:
            raise RecognitionError(
                f"More than {max_points:,} pick positions were requested. "
                "Increase spacing, select fewer regions, or explicitly raise the point limit.")
        if len(xs):
            result.append(np.column_stack((xs, np.full(len(xs), y, dtype=np.int64))))
    _check_cancel(cancel)
    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(result, axis=0).astype(np.int32)


# ---------------- cryolo_recognition.py ----------------

from dataclasses import dataclass
import copy
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np
from PIL import Image



@dataclass(frozen=True)
class CryoloBox:
    """Detector-estimated rectangle, in top-left screenshot coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    def clipped_mask(self, shape):
        height, width = shape
        x0, x1 = max(0, math.floor(self.x0)), min(width, math.ceil(self.x1))
        y0, y1 = max(0, math.floor(self.y0)), min(height, math.ceil(self.y1))
        mask = np.zeros(shape, dtype=bool)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask


def read_cbox(path: str | Path, image_shape, threshold: float) -> list[CryoloBox]:
    """Read crYOLO >=1.8 STAR CBOX, converting bottom-left output to top-left.

    A present zero-byte file is the official writer's empty-prediction format.
    Missing/malformed files are failures, never evidence of a blank image.
    CBOX contains below-threshold predictions: threshold is applied here too.
    Width/Height locate the center of the extraction box. EstWidth/EstHeight,
    when available, describe the detected object's extent around that center.
    """
    path = Path(path)
    if not path.is_file():
        raise RecognitionError(f"crYOLO did not produce the expected CBOX file: {path.name}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise RecognitionError("crYOLO CBOX output exceeds the 16 MiB limit.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RecognitionError(f"Cannot read crYOLO CBOX output: {exc}") from exc
    if not text.strip():
        return []
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("crYOLO threshold must be between 0 and 1.")
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")
    columns, rows = [], []
    in_block = in_loop = False
    found_block = False
    try:
        for line in text.splitlines():
            tokens = shlex.split(line, comments=True)
            if not tokens:
                continue
            head = tokens[0]
            if head.startswith("data_"):
                in_block = head.lower() == "data_cryolo"
                in_loop = False
                found_block |= in_block
                continue
            if not in_block:
                continue
            if head.lower() == "loop_":
                if columns or rows:
                    raise ValueError("Multiple coordinate loops are unsupported.")
                in_loop = True
                continue
            if head.startswith("_"):
                if not in_loop or rows or len(tokens) != 1:
                    raise ValueError("Invalid coordinate column header.")
                columns.append(head.lower())
                continue
            if not in_loop or not columns or len(tokens) != len(columns):
                raise ValueError("A CBOX row has the wrong number of columns.")
            rows.append(tokens)
        required = {"_coordinatex", "_coordinatey", "_width", "_height", "_confidence"}
        if not found_block or not required.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate coordinate/confidence columns.")
        indices = {name: i for i, name in enumerate(columns)}
        boxes = []
        for row in rows:
            def number(name):
                value = float(row[indices[name]])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name}.")
                return value
            x, y = number("_coordinatex"), number("_coordinatey")
            box_w, box_h = number("_width"), number("_height")
            confidence = number("_confidence")
            if box_w <= 0 or box_h <= 0 or not 0 <= confidence <= 1:
                raise ValueError("Invalid dimensions or confidence.")
            est_w, est_h = box_w, box_h
            for name in ("_estwidth", "_estheight"):
                if name in indices and row[indices[name]].lower() not in {"<na>", "nan", "none", "?", "."}:
                    estimate = number(name)
                    if estimate <= 0:
                        raise ValueError("Nonpositive estimated object size.")
                    if name == "_estwidth":
                        est_w = estimate
                    else:
                        est_h = estimate
            cx, cy = x + box_w / 2, height - (y + box_h / 2)
            if not 0 <= cx <= width or not 0 <= cy <= height:
                raise ValueError("Detection center lies outside the input image.")
            if confidence >= threshold:
                boxes.append(CryoloBox(cx-est_w/2, cy-est_h/2,
                                       cx+est_w/2, cy+est_h/2, confidence))
        return sorted(boxes, key=lambda b: (-b.confidence, b.y0, b.x0))
    except (ValueError, IndexError, OverflowError) as exc:
        raise RecognitionError(f"Malformed crYOLO CBOX output: {exc}") from exc


def _stop_process(process):
    """Stop only this request's process group, including preprocessing workers."""
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        # Python preprocessing pools can outlive their parent on Windows.
        # taskkill is restricted to this freshly created request's PID tree.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=2)


class CryoloRecognizer:
    """One serial worker's detector with cached predictions for prompt review."""

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self.device = "cpu" if config.device == "cpu" else config.device
        self.last_detection = None
        self._key = None
        self._artifact_signature = None
        self._boxes = None
        self._automatic_key = None
        self._automatic_result = None
        self._automatic_report = None
        self._profile = None
        self._process = None

    def _paths(self):
        weights = Path(self.config.checkpoint).expanduser().resolve()
        config_value = getattr(self.config, "cryolo_config", None)
        config_path = (Path(config_value).expanduser().resolve() if config_value
                       else weights.with_name("cryolo-config.json"))
        python_value = getattr(self.config, "cryolo_python", None) or os.environ.get("SARU_CRYOLO_PYTHON")
        if not python_value and os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                runtime = Path(local_app_data) / "SARU_Picker_crYOLO" / "cryolo-env"
                for installed in (runtime / "Scripts" / "python.exe", runtime / "python.exe"):
                    if installed.is_file():
                        python_value = str(installed)
                        break
        if not python_value:
            raise RecognitionError("Select the Python executable in your installed crYOLO environment (crYOLO settings).")
        executable = shutil.which(str(python_value))
        if executable is None and Path(python_value).expanduser().is_file():
            executable = os.path.abspath(os.path.expanduser(python_value))
        if not executable:
            raise RecognitionError(f"crYOLO Python executable was not found: {python_value}")
        # Keep the venv executable spelling: resolving its symlink would
        # bypass pyvenv.cfg and lose the separately installed crYOLO packages.
        executable = os.path.abspath(executable)
        for label, path in (("weights", weights), ("configuration", config_path)):
            if not path.is_file():
                raise RecognitionError(f"crYOLO {label} not found: {path}")
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("model"), dict):
                raise ValueError("Missing model configuration.")
        except (OSError, ValueError) as exc:
            raise RecognitionError(f"Invalid crYOLO configuration: {exc}") from exc
        return executable, weights, config_path

    def _predict(self, rgb, cancel):
        _check_cancel(cancel)
        executable, weights, config_path = self._paths()
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in (weights, config_path))
        key = _image_key(rgb)
        if key == self._key and signature == self._artifact_signature and self._boxes is not None:
            return list(self._boxes), True
        self._automatic_key = None
        self._automatic_result = None
        self._automatic_report = None
        self._profile = None
        threshold = getattr(self.config, "cryolo_threshold", .3)
        with tempfile.TemporaryDirectory(prefix="saru-cryolo-") as folder:
            folder = Path(folder)
            inputs, output = folder / "input", folder / "output"
            inputs.mkdir()
            # The screenshot is top-left origin. crYOLO's PNG loader keeps this
            # orientation; its CBOX exporter flips Y to bottom-left coordinates.
            Image.fromarray(rgb).convert("L").save(inputs / "screen.png")
            command = [executable, "-m", "cryolo.predict", "-c", str(config_path),
                       "-w", str(weights), "-i", str(inputs), "-o", str(output),
                       "-t", str(threshold), "-pbs", "1", "--num_cpu",
                       str(self.config.cpu_threads), "--write_empty"]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            if self.config.device == "cpu":
                # crYOLO 1.9.9 strips the minus sign from `-g -1`; use the
                # TensorFlow/CUDA environment control and leave -g unspecified.
                env["CUDA_VISIBLE_DEVICES"] = "-1"
            elif self.config.device == "cuda":
                # Fail an explicit GPU choice rather than letting TensorFlow
                # silently place the requested run on the CPU.
                probe = ("import runpy; import tensorflow as tf; "
                         "assert tf.test.is_gpu_available(cuda_only=True), "
                         "'CUDA selected, but the crYOLO runtime has no CUDA device'; "
                         "runpy.run_module('cryolo.predict', run_name='__main__')")
                command[1:3] = ["-c", probe]
                command.extend(["-g", "0"])
            log_path = folder / "predict.log"
            started = time.monotonic()
            try:
                with log_path.open("wb") as log:
                    self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     cwd=folder, env=env,
                                                     start_new_session=os.name != "nt")
                    while self._process.poll() is None:
                        _check_cancel(cancel)
                        if time.monotonic() - started > getattr(self.config, "cryolo_timeout", 180):
                            raise RecognitionError("crYOLO detection timed out. Check its runtime/model or increase the timeout.")
                        time.sleep(.05)
                    code = self._process.returncode
                _check_cancel(cancel)
                if code:
                    with log_path.open("rb") as log:
                        log.seek(max(0, log_path.stat().st_size - 8000))
                        tail = log.read().decode("utf-8", errors="replace")
                    raise RecognitionError(f"crYOLO prediction failed (exit {code}).\n{tail}")
                boxes = read_cbox(output / "CBOX" / "screen.cbox", rgb.shape[:2], threshold)
            except OSError as exc:
                raise RecognitionError(f"Could not start the crYOLO Python runtime: {exc}") from exc
            finally:
                if self._process is not None:
                    _stop_process(self._process)
                    self._process = None
        _check_cancel(cancel)
        self._key, self._artifact_signature, self._boxes = key, signature, list(boxes)
        return boxes, False

    def _refine(self, rgb, boxes, cancel, *, prompt=False):
        check = lambda: _check_cancel(cancel)
        raw = []
        for box in boxes[:128]:
            check()
            mask = box.clipped_mask(rgb.shape[:2])
            if mask.any():
                raw.append({"segmentation": mask, "predicted_iou": None,
                            "stability_score": None, "source": "crYOLO detection box (review)"})
        profile = None
        minimum = 0 if prompt else self.config.min_capsid_area_px
        appearance = {"status": "disabled"}
        # Even generic crYOLO mode traces image-supported interiors: a detector
        # rectangle is never presented as a biological segmentation mask.
        if self.config.automatic_profile == "capsid-learned":
            if self._profile is None:
                try:
                    path = resolve_profile_path(self.config.checkpoint, self.config.capsid_prior_path)
                    if path is None:
                        raise ValueError("capsid-profile.json is missing beside the crYOLO weights")
                    self._profile = load_profile(path)
                except (OSError, ValueError) as exc:
                    raise RecognitionError(f"Could not load the capsid appearance profile: {exc}") from exc
            profile = self._profile
            appearance = profile.describe()
        if not raw:
            return [], {}, {"enabled": False}, {"enabled": False}, appearance
        if profile is not None:
            scores = score_image(rgb, profile, check)
            raw = filter_capsid_with_profile(rgb, raw, scores, check_cancel=check,
                                             min_area_ratio=0 if prompt else self.config.min_area_ratio,
                                             max_area_ratio=1 if prompt else min(self.config.max_area_ratio, .4),
                                             min_capsid_area_px=minimum)
        else:
            raw = filter_capsid_proposals(rgb, raw, check_cancel=check,
                                         min_area_ratio=0 if prompt else self.config.min_area_ratio,
                                         max_area_ratio=1 if prompt else min(self.config.max_area_ratio, .4))
        context = detect_carbon_edges(rgb, check_cancel=check)
        raw, carbon = guard_proposals(context, raw, check_cancel=check)
        carbon["enabled"] = True
        raw, area = _filter_capsid_area(raw, minimum, cancel)
        candidates, evidence = [], {}
        key = _image_key(rgb)
        for record in raw:
            check()
            mask = np.asarray(record["segmentation"], dtype=bool)
            if mask.shape != rgb.shape[:2]:
                raise RecognitionError("A refined crYOLO contour has incorrect image dimensions.")
            if not mask.any():
                continue
            candidate = make_candidate(mask, source="crYOLO + capsid contour refinement (review)", image_key=key)
            candidates.append(candidate)
            supporting = []
            for b in boxes:
                x0, x1 = max(0, math.floor(b.x0)), min(mask.shape[1], math.ceil(b.x1))
                y0, y1 = max(0, math.floor(b.y0)), min(mask.shape[0], math.ceil(b.y1))
                if x1 > x0 and y1 > y0 and np.count_nonzero(mask[y0:y1, x0:x1]) / candidate.area >= .5:
                    supporting.append(b)
            item = dict(record.get("capsid_evidence") or {})
            item.update(cryolo_confidence=max((b.confidence for b in supporting), default=None),
                        cryolo_boxes=[[b.x0, b.y0, b.x1, b.y1] for b in supporting],
                        confidence_is_mask_iou=False, is_calibrated_probability=False)
            evidence[candidate.id] = item
        if not prompt:
            candidates = filter_candidates(candidates, self.config, cancel, suppress_duplicates=False)
        return candidates, {c.id: evidence[c.id] for c in candidates}, carbon, area, appearance

    def automatic(self, rgb, cancel=None):
        rgb = validate_rgb(rgb)
        started = time.perf_counter()
        boxes, cached = self._predict(rgb, cancel)
        key = _image_key(rgb)
        if cached and self._automatic_key == key and self._automatic_result is not None:
            self.last_detection = copy.deepcopy(self._automatic_report)
            self.last_detection["detections_cached"] = True
            return list(self._automatic_result)
        candidates = self.refine_detections(rgb, boxes, cancel=cancel)
        self.last_detection["detections_cached"] = cached
        self.last_detection["detections_supplied"] = False
        self.last_detection["timing_seconds"]["total_including_initial_model_load"] = time.perf_counter() - started
        self._automatic_key, self._automatic_result = key, list(candidates)
        self._automatic_report = copy.deepcopy(self.last_detection)
        return candidates

    def refine_detections(self, rgb, boxes, cancel=None, *, prompt=False):
        """Run the exact production contour tail on parsed batch CBOX detections.

        This supports evaluation with one crYOLO process for a whole dataset.
        It does not perform detection or imply that external boxes are held out.
        """
        rgb = validate_rgb(rgb)
        boxes = list(boxes)
        if any(not isinstance(box, CryoloBox) for box in boxes):
            raise ValueError("Detections must be CryoloBox instances from read_cbox.")
        started = time.perf_counter()
        candidates, evidence, carbon, area, appearance = self._refine(rgb, boxes, cancel, prompt=prompt)
        _check_cancel(cancel)
        self.last_detection = {"backend": "cryolo", "profile": self.config.automatic_profile,
                               "cryolo_detections": len(boxes), "selected_contours": len(candidates),
                               "confidence_threshold": getattr(self.config, "cryolo_threshold", .3),
                               "detections_supplied": True, "requested_device": self.config.device,
                               "appearance_model": appearance, "carbon_guard": carbon,
                               "capsid_area_filter": area, "capsid_evidence": evidence,
                               "timing_seconds": {"total_including_initial_model_load": time.perf_counter() - started},
                               "note": "crYOLO predicts boxes; existing SARU appearance/texture refinement traces contours. Objectness is not mask IoU or a calibrated capsid probability. No SAM inference is used."}
        return candidates

    def prompt(self, rgb, box: Sequence[float] | None = None,
               points: Sequence[Sequence[float]] | None = None,
               labels: Sequence[int] | None = None, cancel=None):
        """Select crYOLO detections by prompts; prompts do not retrain the detector."""
        rgb = validate_rgb(rgb)
        height, width = rgb.shape[:2]
        box_array = point_array = label_array = None
        if box is not None:
            box_array = np.asarray(box, dtype=float)
            if (box_array.shape != (4,) or not np.isfinite(box_array).all()
                    or not 0 <= box_array[0] < box_array[2] <= width
                    or not 0 <= box_array[1] < box_array[3] <= height):
                raise ValueError("Box must be (x0,y0,x1,y1), with positive size, inside the image.")
        if points is not None:
            point_array = np.asarray(points, dtype=float)
            label_array = np.asarray(labels) if labels is not None else np.ones(len(point_array), dtype=int)
            if (point_array.ndim != 2 or point_array.shape[1:] != (2,) or not len(point_array)
                    or not np.isfinite(point_array).all() or (point_array < 0).any()
                    or (point_array[:, 0] >= width).any() or (point_array[:, 1] >= height).any()):
                raise ValueError("Prompt points must lie inside the image.")
            if label_array.shape != (len(point_array),) or not np.isin(label_array, (0, 1)).all():
                raise ValueError("Provide one label (0 or 1) per prompt point.")
        elif labels is not None:
            raise ValueError("Point labels were supplied without points.")
        if box_array is None and (label_array is None or not label_array.any()):
            raise ValueError("Supply a box or at least one positive point.")
        boxes, cached = self._predict(rgb, cancel)
        selected = []
        for detection in boxes:
            cx, cy = (detection.x0+detection.x1)/2, (detection.y0+detection.y1)/2
            if box_array is not None and not (box_array[0] <= cx <= box_array[2] and box_array[1] <= cy <= box_array[3]):
                continue
            if point_array is not None:
                inside = ((point_array[:, 0] >= detection.x0) & (point_array[:, 0] <= detection.x1)
                          & (point_array[:, 1] >= detection.y0) & (point_array[:, 1] <= detection.y1))
                if np.any(inside & (label_array == 0)) or (np.any(label_array == 1) and not np.any(inside & (label_array == 1))):
                    continue
            selected.append(detection)
        candidates, evidence, carbon, area, appearance = self._refine(rgb, selected, cancel, prompt=True)
        self.last_detection = {"backend": "cryolo", "mode": "prompt_selection",
                               "cryolo_detections": len(boxes), "selected_detection_boxes": len(selected),
                               "selected_contours": len(candidates), "detections_cached": cached,
                               "capsid_evidence": evidence, "carbon_guard": carbon,
                               "capsid_area_filter": area, "appearance_model": appearance,
                               "note": "Prompts select existing crYOLO detections. crYOLO does not provide SAM-style prompted segmentation; a missed detection can still be drawn manually."}
        return candidates

    def close(self):
        if self._process is not None:
            _stop_process(self._process)
        self._process = self._profile = self._boxes = self._automatic_result = None
        self._key = self._artifact_signature = self._automatic_key = None
        self._automatic_report = None
        self.last_detection = None
        self.device = None


# ---------------- execution.py ----------------


from dataclasses import dataclass
import logging
import math
from numbers import Integral, Real
import queue
import threading
import time
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

    ``delay`` is the wait *between* clicks. Zero removes the configured wait
    but yields the worker between events so the GUI/input threads can process
    Stop and pause requests. The click adapter should disable additional
    automatic pauses while leaving its fail-safe enabled.
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
            if delay:
                token.sleep(delay)
            else:
                # Event.wait(0) does not yield a busy worker. Release its time
                # slice without imposing a fixed delay; the next iteration
                # still checks cancellation/pause before sending any input.
                time.sleep(0)
    return completed


# ---------------- desktop.py ----------------


import ctypes
import importlib
import logging
import math
import os
from numbers import Real
import platform
import threading
import time
from typing import Callable

import numpy as np



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
            "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
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

    def process_id(self, handle: int) -> int:
        process = self._wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
        return int(process.value)


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
    and ``press_next`` require a target established by ``focus_target`` or
    ``bind_target``. Manual mode can bind a window already activated by the
    user's physical mouse gesture without moving focus itself.
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

    def target_at(self, x: int, y: int) -> int:
        """Resolve the live window at a physical screen point without focusing it."""
        x, y = _desktop_integer(x, "Pointer x"), _desktop_integer(y, "Pointer y")
        width, height = self.size()
        if not (0 <= x < width and 0 <= y < height):
            raise DesktopError("Draw on the primary monitor.")
        api = self._get_api()
        return self._validate_handle(api.root_window(api.window_at(x, y)))

    def is_target_at(self, x: int, y: int, target: int) -> bool:
        try:
            return self.target_at(x, y) == self._validate_handle(target)
        except DesktopError:
            return False

    def is_own_window(self, handle: int) -> bool:
        """Exclude our own controls, title bars and dialogs from manual traces."""
        handle = self._validate_handle(handle)
        process_id = self._get_api().process_id(handle)
        if not process_id:
            raise DesktopError("The window owning this screen position could not be identified.")
        return process_id == os.getpid()

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

    def bind_target(self, handle) -> None:
        """Use an already foreground target without focusing or restoring it.

        A rejected new batch must not retain an older batch's implicit target.
        ``click`` still validates foreground and the window under each point.
        """
        self._target = None
        handle = _desktop_integer(handle, "Target window handle")
        self.ensure_target(handle)
        self._target = handle

    def ensure_target(self, handle=None) -> None:
        handle = self._target if handle is None else handle
        if handle is None:
            raise DesktopError("No picking target has been selected for playback.")
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
    """Debounced global Escape/F8 listener with optional scoped Space pause.

    The callback receives ``'stop'``, ``'pause'`` or ``'space_pause'`` on a listener thread and
    must only enqueue a message or update thread-safe cancellation state.
    Space is consumed only while explicitly enabled for an automatic run;
    otherwise ordinary browser keyboard input is left alone.
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        listener_factory=None,
        key_enum=None,
        platform_name: str | None = None,
        *,
        space_pause: bool = False,
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
        self._space_pause = bool(space_pause)
        self._space_enabled = False
        self._space_held = False
        self._space_passthrough_held = False

    def enable_space_pause(self, enabled: bool) -> None:
        with self._lock:
            self._space_enabled = bool(enabled) and self._space_pause

    def _win32_filter(self, msg, data):
        """Consume only the Space press/release pair owned by an auto run.

        Pynput's filter precedes its ordinary callbacks. Calling suppress_event
        intentionally raises its internal suppression signal, so this method
        must not catch that signal. Holding Space cannot toggle pause repeatedly.
        """
        if getattr(data, "vkCode", None) != 0x20 or getattr(data, "flags", 0) & 0x10:
            return True
        down, up = msg in (0x0100, 0x0104), msg in (0x0101, 0x0105)
        if not (down or up):
            return True
        with self._lock:
            if not self._active or not self._space_pause:
                return True
            # A key press that began outside the run belongs to the browser,
            # including its repeats and release even if a run starts meanwhile.
            if self._space_passthrough_held:
                if up:
                    self._space_passthrough_held = False
                return True
            if up and not self._space_held:
                return True
            if not (self._space_enabled or self._space_held):
                if down:
                    self._space_passthrough_held = True
                return True
            emit = down and not self._space_held
            if down:
                self._space_held = True
            elif up:
                self._space_held = False
        if emit:
            try:
                self._callback("space_pause")
            except Exception:
                DESKTOP_LOGGER.exception("Space pause callback failed.")
        listener = self._listener
        if listener is not None:
            listener.suppress_event()
        return False

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
            callbacks = {"on_press": self._on_press, "on_release": self._on_release}
            if self._space_pause:
                callbacks["win32_event_filter"] = self._win32_filter
            self._listener = self._factory(**callbacks)
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
            self._space_enabled = False
            self._space_held = False
            self._space_passthrough_held = False
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()



# ---------------- manual_sampling.py ----------------

import math

import numpy as np



# Ordinary desktop contours are far below these limits. They also bound work
# for a thin or self-intersecting contour whose bounding box contains very few
# accepted points, where the output-point limit alone would not protect us.
_MANUAL_MAX_VERTICES = 50_000
_MANUAL_MAX_ROWS = 20_000
_MANUAL_MAX_GRID_SITES = 2_000_000
_MANUAL_MAX_EDGE_ROWS = 20_000_000


def _manual_check_cancel(cancel):
    if cancel is not None and cancel():
        raise CancelledOperation("Manual contour sampling cancelled.")


def legacy_contour_points(points, spacing: int, max_points: int = 10_000,
                          cancel=None) -> np.ndarray:
    """Return the original contour's picks as an ``N x 2`` int32 array.

    ``points`` are finite numeric screen coordinates; negative origins and
    fractional vertices are supported. Vertices must fit signed 32-bit screen
    coordinates. ``spacing`` is an integer of at least two pixels; the GUI's
    original slider uses 5–25. No boundary inset or image-mask rasterization is
    applied. Self-intersecting traces retain the original even/odd fill rule.

    Malformed or collinear contours, an exceeded ``max_points``, or excessive
    bounding-box work raise ValueError. Nothing is silently truncated. A true
    ``cancel()`` raises execution.CancelledOperation, allowing TaskRunner to
    discard the entire request without publishing partial pick positions.

    The implementation evaluates the original ray rule once per scanline and
    uses sorted crossings to find the same parity at each original grid point.
    It avoids a slow Python loop over every vertex for every possible pick.
    """
    if cancel is not None and not callable(cancel):
        raise ValueError("cancel must be callable or None.")
    _manual_check_cancel(cancel)
    integer_limit = int(np.iinfo(np.int32).max)
    if (isinstance(spacing, (bool, np.bool_))
            or not isinstance(spacing, (int, np.integer))
            or not 2 <= spacing <= integer_limit):
        raise ValueError("spacing must be an integer of at least 2 pixels within the screen-coordinate range.")
    if (isinstance(max_points, (bool, np.bool_))
            or not isinstance(max_points, (int, np.integer)) or max_points < 1):
        raise ValueError("max_points must be a positive integer.")
    spacing, max_points = int(spacing), int(max_points)
    try:
        vertices = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise ValueError("A manual contour needs at least three finite numeric (x,y) points.") from exc
    if (vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3
            or vertices.dtype.kind not in "iuf"):
        raise ValueError("A manual contour needs at least three finite numeric (x,y) points.")
    if len(vertices) > _MANUAL_MAX_VERTICES:
        raise ValueError("The manual contour has too many vertices; draw a shorter contour.")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = vertices.astype(np.float64, copy=True)
    minimum_coordinate = int(np.iinfo(np.int32).min)
    if (not np.isfinite(vertices).all() or (vertices < minimum_coordinate).any()
            or (vertices > integer_limit).any()):
        raise ValueError("Manual contour coordinates must be finite and fit signed 32-bit screen coordinates.")

    # A signed polygon area would incorrectly reject a valid figure-eight or
    # bow-tie trace. Reject only a genuinely collinear/point-like contour.
    differences = vertices - vertices[0]
    distinct = np.flatnonzero(np.any(differences != 0, axis=1))
    if not len(distinct):
        raise ValueError("A manual contour must enclose an area, not a point or straight line.")
    direction = differences[distinct[0]]
    cross = direction[0] * differences[:, 1] - direction[1] * differences[:, 0]
    if not np.any(cross != 0):
        raise ValueError("A manual contour must enclose an area, not a point or straight line.")

    x_min, y_min = (int(v) for v in np.min(vertices, axis=0))
    x_max, y_max = (int(v) for v in np.max(vertices, axis=0))
    # Use the original arithmetic, including truncation and odd-row offset.
    row_height = int(spacing * math.sqrt(3) / 2)
    rows = (y_max - y_min) // row_height + 1
    columns = (x_max - x_min) // spacing + 1
    if (rows > _MANUAL_MAX_ROWS or rows * columns > _MANUAL_MAX_GRID_SITES
            or rows * len(vertices) > _MANUAL_MAX_EDGE_ROWS):
        raise ValueError("The manual contour requires too many sampling checks; increase spacing or draw a smaller contour.")

    x1, y1 = vertices[:, 0], vertices[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    lower_y, upper_y = np.minimum(y1, y2), np.maximum(y1, y2)
    rightmost_x = np.maximum(x1, x2)
    result = []
    total = 0
    for row, y in enumerate(range(y_min, y_max + 1, row_height)):
        _manual_check_cancel(cancel)
        active = (lower_y < y) & (y <= upper_y) & (y2 != y1)
        if not active.any():
            continue
        # Original rule: an edge toggles a point when both x <= max(x1,x2)
        # and x <= its intersection at y. Their conjunction is this threshold.
        crossings = (x1[active]
                     + (y - y1[active]) * (x2[active] - x1[active])
                     / (y2[active] - y1[active]))
        thresholds = np.minimum(rightmost_x[active], crossings)
        thresholds.sort()
        x_start = x_min + (spacing // 2 if row % 2 else 0)
        xs = np.arange(x_start, x_max + 1, spacing, dtype=np.int64)
        crossing_count = len(thresholds) - np.searchsorted(thresholds, xs, side="left")
        xs = xs[(crossing_count % 2) != 0]
        total += len(xs)
        if total > max_points:
            raise ValueError(f"More than {max_points:,} manual pick positions were requested; increase spacing or draw a smaller contour.")
        if len(xs):
            result.append(np.column_stack((xs, np.full(len(xs), y, dtype=np.int64))))
    _manual_check_cancel(cancel)
    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(result).astype(np.int32)


def mask_contour_points(mask, spacing: int, max_points: int = 10_000,
                        cancel=None) -> np.ndarray:
    """Sample one detected contour on the manual grid, without a boundary inset.

    The origin is this mask's true-pixel bounding box, not the whole screenshot
    or a union of detections. Row height and alternate-row offset use the same
    integer arithmetic as ``legacy_contour_points``. Membership comes from the
    actual mask rather than a polygon: holes and empty spaces between mask
    components remain empty, and true boundary pixels remain eligible.

    Return an ``N x 2`` int32 array of image-local coordinates in row order.
    Invalid input or excessive work raises ValueError before returning any
    positions. Cancellation raises CancelledOperation. The mask is never
    changed, and an all-false mask has no positions.
    """
    if cancel is not None and not callable(cancel):
        raise ValueError("cancel must be callable or None.")
    def check_cancel():
        if cancel is not None and cancel():
            raise CancelledOperation("Detected contour sampling cancelled.")
    check_cancel()
    integer_limit = int(np.iinfo(np.int32).max)
    if (isinstance(spacing, (bool, np.bool_))
            or not isinstance(spacing, (int, np.integer))
            or not 2 <= spacing <= integer_limit):
        raise ValueError("spacing must be an integer of at least 2 pixels within the screen-coordinate range.")
    if (isinstance(max_points, (bool, np.bool_))
            or not isinstance(max_points, (int, np.integer)) or max_points < 1):
        raise ValueError("max_points must be a positive integer.")
    spacing, max_points = int(spacing), int(max_points)
    try:
        bitmap = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise ValueError("A contour mask must be a nonempty two-dimensional Boolean or 0/1 array.") from exc
    if (bitmap.ndim != 2 or 0 in bitmap.shape
            or any(length > integer_limit for length in bitmap.shape)
            or bitmap.dtype.kind not in "biuf"):
        raise ValueError("A contour mask must be a nonempty two-dimensional Boolean or 0/1 array.")
    if bitmap.dtype != np.bool_:
        if not np.isin(bitmap, (0, 1)).all():
            raise ValueError("A contour mask must contain only Boolean or 0/1 values.")
        bitmap = bitmap.astype(bool)
    check_cancel()
    ys = np.flatnonzero(bitmap.any(axis=1))
    if not len(ys):
        return np.empty((0, 2), dtype=np.int32)
    xs = np.flatnonzero(bitmap.any(axis=0))
    x_min, x_max, y_min, y_max = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
    row_height = int(spacing * math.sqrt(3) / 2)
    rows = (y_max - y_min) // row_height + 1
    columns = (x_max - x_min) // spacing + 1
    if rows > _MANUAL_MAX_ROWS or rows * columns > _MANUAL_MAX_GRID_SITES:
        raise ValueError("The contour requires too many sampling checks; increase spacing or select a smaller area.")
    result = []
    total = 0
    for row, y in enumerate(range(y_min, y_max + 1, row_height)):
        check_cancel()
        x_start = x_min + (spacing // 2 if row % 2 else 0)
        row_xs = np.arange(x_start, x_max + 1, spacing, dtype=np.int64)
        row_xs = row_xs[bitmap[y, row_xs]]
        total += len(row_xs)
        if total > max_points:
            raise ValueError(f"More than {max_points:,} contour pick positions were requested; increase spacing or select a smaller area.")
        if len(row_xs):
            result.append(np.column_stack((row_xs, np.full(len(row_xs), y, dtype=np.int64))))
    check_cancel()
    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(result).astype(np.int32)


# ---------------- micrograph_navigation.py ----------------


import math
from numbers import Real
import time
from typing import Callable, Iterable

import numpy as np



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


# ---------------- screen_overlay.py ----------------


import ctypes
import importlib
import logging
import math
import platform
import threading
from typing import Callable

import numpy as np
from PIL import Image



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

    def _retire_surface(self):
        """Release a previous painted buffer without closing the overlay owner.

        Clearing Canvas items on a withdrawn layered HWND does not clear the
        compositor's last frame. New strokes, selections, and mask sets must
        not remap that HWND. The control panel is a separate persistent window.
        """
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen overlay is closed.")
        if self.window is not None:
            self.hide()
            self.clear()
            self.window.destroy()
            self.window = self.canvas = None

    def begin_selection(self):
        """Prepare a fresh hidden, blank surface for a region or box gesture.

        A selection can follow displayed contours or an earlier selection.
        Retire that painted window before showing any new selection geometry.
        """
        self._retire_surface()
        self._ensure()

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
        self._retire_surface()
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
        """Prepare a complete current mask set on a fresh hidden surface.

        In semi-auto mode the previous surface was hidden during picking and
        can still contain the completed contour in its native painted buffer.
        Validate the new bitmap before retiring it, then populate the new
        canvas before the controller calls show(). No old contour is remapped.
        """
        self._check_thread()
        if self.closed:
            raise OverlayError("The screen overlay is closed.")
        bitmap = _mask_image(candidates, geometry, selected_ids, preview)
        self._retire_surface()
        self._ensure()
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


# ---------------- screen_app.py ----------------

import logging
import json
import os
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time
import tempfile

import numpy as np



SCREEN_LOG = logging.getLogger("saru.screen")
DEFAULT_CLICK_DELAY = .01


def save_cryolo_settings(config, destination) -> None:
    """Persist only this parallel backend's validated user choices."""
    if config.backend != "cryolo" or not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: getattr(config, name) for name in
               ("checkpoint", "device", "cryolo_config", "cryolo_python", "cryolo_threshold",
                "cryolo_timeout", "automatic_profile")}
    payload["min_capsid_area"] = config.min_capsid_area_px
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".cryolo-settings-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    raw_count: int

    @property
    def blocked_count(self):
        return self.raw_count - len(self.local)


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
        batches.append(ScreenContourBatch(candidate.id, local, screen, len(positions)))
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
        is_cryolo = getattr(self.args, "backend", "sam") == "cryolo"
        root.title("SARU PICKER crYOLO" if is_cryolo else "SARU PICKER")
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
        ttk.Button(self.semi_controls, text="Select by box" if is_cryolo else "Box prompt", command=self.box_prompt).pack(side="left")
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
        self.cryolo_config = tk.StringVar(value=getattr(self.args, "cryolo_config", None) or "")
        self.cryolo_python = tk.StringVar(value=getattr(self.args, "cryolo_python", None) or "")
        self.cryolo_threshold = tk.StringVar(value=str(getattr(self.args, "cryolo_threshold", .3)))
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
        args = getattr(self, "args", None)
        runtime = {"backend": getattr(args, "backend", "sam")}
        if runtime["backend"] == "cryolo":
            runtime.update(cryolo_config=self.cryolo_config.get().strip() or None,
                           cryolo_python=self.cryolo_python.get().strip() or None,
                           cryolo_threshold=float(self.cryolo_threshold.get()),
                           cryolo_timeout=getattr(args, "cryolo_timeout", 180.0))
        return RecognitionConfig(checkpoint=self.checkpoint.get(), model_type=self.model_type.get(),
                                 device=self.device.get(), points_per_side=int(self.grid_side.get()),
                                 automatic_profile=self.automatic_profile.get(),
                                 min_capsid_area_px=int(self.min_capsid_area.get()), **runtime)

    def _apply_model_settings(self, previous_config):
        """Validate settings and retire contours made with an older detector.

        Keep actual pick history and its scene reference: changing a size limit
        must not make a repeated detection place duplicate picks on this image.
        """
        current = self._model_config()
        save_cryolo_settings(current, getattr(getattr(self, "args", None), "cryolo_settings_path", None))
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
            self.recognizer = create_recognizer(config) if config.backend == "cryolo" else SamRecognizer(config)
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
        selected_ids = {candidate.id for candidate in candidates}
        remaining_candidates = tuple(candidate for candidate in self.candidates
                                     if candidate.id not in selected_ids)
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
                raw_count = sum(batch.raw_count for batch in batches)
                blocked_count = sum(batch.blocked_count for batch in batches)
                SCREEN_LOG.info("Model pick plan: frame=%s spacing=%s raw=%s blocked=%s eligible=%s",
                                frame_id, spacing, raw_count, blocked_count, planned)
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
                # Retire stale overlapping proposals before restoring the
                # selectable overlay. Check every remaining region separately
                # against actual clicks only: an unselected proposal must not
                # suppress another unselected proposal's available sites.
                # This bitmap/grid work stays on the worker thread.
                actual_picks = picked | set(completed_local)
                remaining_plans, exhausted_ids = [], set()
                for candidate in remaining_candidates:
                    available = screen_model_batches([candidate], geometry, spacing, actual_picks,
                                                     cancel=lambda: token.cancelled,
                                                     max_points=int(candidate.mask.size))[0]
                    record = {"candidate_id": candidate.id, "raw_grid_points": available.raw_count,
                              "blocked_grid_points": available.blocked_count,
                              "eligible_grid_points": len(available.local)}
                    remaining_plans.append(record)
                    if not len(available.local):
                        exhausted_ids.add(candidate.id)
                    SCREEN_LOG.info("Remaining contour plan: frame=%s id=%s raw=%s blocked=%s eligible=%s",
                                    frame_id, candidate.id, available.raw_count,
                                    available.blocked_count, len(available.local))
                token.sleep(.12)
                self.desktop.ensure_target(target)
                after = self._capture_region(geometry)
                if background is not None and screen_background_changed(reference, after, background) > .03:
                    raise RuntimeError("The micrograph changed during picking. Detect again.")
                return {"count": count, "ids": {batch.candidate_id for batch in batches}, "reference": after,
                        "auto": auto, "contours": len(batches), "scene_verified": background is not None,
                        "frame_id": frame_id, "picked": tuple(completed_local),
                        "planning": {"spacing": spacing, "raw_grid_points": raw_count,
                                     "blocked_grid_points": blocked_count,
                                     "eligible_grid_points": planned, "sent_points": count},
                        "remaining_plans": tuple(remaining_plans), "exhausted_ids": exhausted_ids}
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
            planning = value.get("planning", {})
            # The slider remains usable during playback. A plan made with the
            # old spacing cannot retire a region which may have sites at the
            # user's new spacing. Frame identity was checked above.
            exhausted_ids = (value.get("exhausted_ids", set())
                             if planning.get("spacing") == int(self.spacing.get()) else set())
            retired = sum(c.id in exhausted_ids and c.id not in value["ids"] for c in self.candidates)
            self.candidates = [c for c in self.candidates
                               if c.id not in value["ids"] and c.id not in exhausted_ids]
            self.alternatives = []
            if value["auto"]:
                self._finish_auto_frame()
            else:
                self._present_masks()
                if value["count"]:
                    message = f"{value['count']} picks placed."
                elif planning.get("raw_grid_points") == 0:
                    message = f"No grid sites at spacing {planning['spacing']}. Lower spacing and Detect again."
                elif planning.get("blocked_grid_points") == planning.get("raw_grid_points") and planning:
                    message = "No new picks: this contour is already covered by earlier picks."
                else:
                    message = "No new pick positions in this contour."
                if retired:
                    message += f" {retired} region{'s' if retired != 1 else ''} without new positions hidden."
                if self.candidates:
                    message += f" {len(self.candidates)} contour{'s' if len(self.candidates) != 1 else ''} ready."
                else:
                    message += " No selectable contours remain. Use Next micrograph or Detect."
                self.status.set(message)
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
            is_cryolo = getattr(self.args, "backend", "sam") == "cryolo"
            self.ttk.Label(frame, text="crYOLO weights" if is_cryolo else "SAM checkpoint").grid(row=0, column=0, sticky="w")
            self.ttk.Entry(frame, textvariable=self.checkpoint, width=48).grid(row=1, column=0, columnspan=2, sticky="ew")
            def browse():
                filetypes = [("crYOLO weights", "*.h5")] if is_cryolo else [("SAM checkpoint", "*.pth")]
                path = self.filedialog.askopenfilename(parent=window, filetypes=filetypes)
                if path:
                    self.checkpoint.set(path)
                    if not is_cryolo:
                        for model_type in ("vit_b", "vit_l", "vit_h"):
                            if model_type in Path(path).name:
                                self.model_type.set(model_type)
                                break
            self.ttk.Button(frame, text="Browse…", command=browse).grid(row=1, column=2, padx=4)
            combos = [("Device", self.device, ("auto", "cpu", "cuda"))]
            if not is_cryolo:
                combos.append(("Model type", self.model_type, ("vit_b", "vit_l", "vit_h")))
            combos.append(("Detection profile", self.automatic_profile, ("capsid", "capsid-learned", "generic")))
            index = 2
            for label, variable, values in combos:
                self.ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
                self.ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=18 if label == "Detection profile" else 10).grid(row=index, column=1)
                index += 1
            if is_cryolo:
                for label, variable in (("crYOLO configuration (.json)", self.cryolo_config),
                                        ("crYOLO Python executable", self.cryolo_python),
                                        ("Detection confidence threshold", self.cryolo_threshold)):
                    self.ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
                    self.ttk.Entry(frame, textvariable=variable, width=32).grid(row=index, column=1, columnspan=2, sticky="ew")
                    index += 1
            sliders = [("Click delay (all modes; 0 = maximum)", self.click_delay, 0, 2, .01),
                       ("Minimum capsid area (screen px²)", self.min_capsid_area, 0, 1000000, 500)]
            if not is_cryolo:
                sliders.insert(0, ("Automatic grid / side", self.grid_side, 4, 32, 1))
            for label, variable, low, high, step in sliders:
                self.ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=3)
                self.ttk.Spinbox(frame, textvariable=variable, from_=low, to=high, increment=step, width=10).grid(row=index, column=1)
                index += 1
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
            self.ttk.Label(frame, text="capsid-learned requires its installed appearance profile.", wraplength=settings_wrap).grid(
                row=index, column=0, columnspan=3, sticky="w", pady=(6, 0))
            size_note = "Minimum area applies to both capsid modes; 0 allows all sizes.\nZoom changes apparent size. Manual and Box prompts are unchanged."
            if is_cryolo:
                size_note = ("Minimum area applies to automatic capsid contours; 0 allows all sizes.\n"
                             "Select by box filters crYOLO detections; it does not prompt a segmentation model.")
            self.ttk.Label(frame, text=size_note, wraplength=settings_wrap).grid(
                row=index + 1, column=0, columnspan=3, sticky="w", pady=(6, 0))
            self.ttk.Label(frame, text="Click delay is in seconds. 0 removes the added wait; increase it if cryoSPARC misses picks.",
                           wraplength=settings_wrap).grid(row=index + 2, column=0, columnspan=3, sticky="w", pady=(6, 0))
            self.ttk.Button(frame, text="Done", command=close_settings).grid(row=index + 3, column=0, columnspan=3, pady=(10, 0))
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


# ---------------- app.py ----------------

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



APP_LOG = logging.getLogger("saru_picker")
APP_VERSION = "2.9.0"
DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"
DEFAULT_CRYOLO_CHECKPOINT = "cryolo-capsids.h5"
ACTIVE_BACKEND = "sam"


def user_state_directory() -> Path:
    """Writable application data, independent of the installation directory."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data and Path(local_app_data).is_absolute():
            return Path(local_app_data) / ("SARU_Picker_crYOLO" if ACTIVE_BACKEND == "cryolo" else "SARU_Picker")
        return Path.home() / "AppData" / "Local" / ("SARU_Picker_crYOLO" if ACTIVE_BACKEND == "cryolo" else "SARU_Picker")
    return Path.home() / (".saru_picker_cryolo" if ACTIVE_BACKEND == "cryolo" else ".saru_picker")


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
            self.frames = [np.repeat(np.random.default_rng(210+index).integers(
                60, 190, (geometry.height, geometry.width), dtype=np.uint8)[..., None], 3, axis=2)
                for index in range(4)]
            self.reset()
        def reset(self, index=0):
            self.index = index
            self.keys, self.clicks, self.thread_ids = [], [], set()
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
            return self.frames[self.index].copy()
        def click(self, x, y):
            self._layers_hidden()
            if not geometry.contains(x, y):
                raise RuntimeError("A generated cycle test pick escaped the selected area.")
            self.thread_ids.add(threading.get_ident())
            self.clicks.append((self.index, x, y))
        def press_next(self, key):
            self._layers_hidden()
            if key != "down":
                raise RuntimeError(f"Unexpected navigation key: {key!r}")
            self.thread_ids.add(threading.get_ident())
            self.keys.append(key)
            self.index = min(self.index + 1, len(self.frames)-1)

    desktop = MemoryDesktop()
    candidates = []
    for index in range(4):
        left = 15 + index * 30
        mask = polygon_mask([(left+9, 30), (left+43, 30), (left+54, 48),
                             (left+49, 94), (left+10, 101), (left, 73)],
                            (geometry.height, geometry.width))
        candidates.append(make_candidate(mask, source="synthetic micrograph-cycle check"))

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
            return [] if index in self.empty_frames else [candidates[index]]

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
        auto = {"continuous_setting": 0, "detected_frames": list(model.calls),
                "picked_frames": sorted({index for index, _, _ in desktop.clicks}),
                "empty_frame_skipped": True, "down_keys": len(desktop.keys),
                "unique_test_picks": len(desktop.clicks), "stop_during_navigation": True,
                "no_detection_or_picks_after_stop": True}

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
    report = diagnostic_report(log_path, Path(config.checkpoint))
    report.update(backend="cryolo", cryolo_config=config.cryolo_config,
                  cryolo_config_exists=bool(config.cryolo_config and Path(config.cryolo_config).is_file()),
                  cryolo_python=config.cryolo_python,
                  note="Metadata inspection only. crYOLO runs in its separate Python environment; use --smoke-test to check real inference.")
    report["dependencies"] = {name: version for name, version in report["dependencies"].items()
                              if name not in {"torch", "torchvision", "segment-anything"}}
    return report


def run_cryolo_smoke_test(log_path: Path, config: RecognitionConfig) -> dict[str, Any]:
    """Exercise the selected crYOLO process without sending desktop events."""
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
        root.title("SARU Picker crYOLO — review masks, then pick" if is_cryolo else "SARU Picker 2 — review masks, then pick")
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
        model = ttk.LabelFrame(controls, text="2. crYOLO model" if is_cryolo else "2. SAM model", padding=7)
        model.pack(fill="x", pady=4)
        checkpoint = str(resolve_checkpoint(self.args.checkpoint, backend="cryolo") if is_cryolo
                         else resolve_checkpoint(self.args.checkpoint))
        self.checkpoint = tk.StringVar(value=checkpoint)
        ttk.Entry(model, textvariable=self.checkpoint, width=36).pack(fill="x")
        row = ttk.Frame(model)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Browse…", command=self.browse_model).pack(side="left")
        self.device = tk.StringVar(value=self.args.device)
        ttk.Combobox(row, textvariable=self.device, values=("auto", "cpu", "cuda"),
                     state="readonly", width=7).pack(side="left", padx=5)
        self.model_type = tk.StringVar(value=self.args.model_type)
        if not is_cryolo:
            ttk.Combobox(row, textvariable=self.model_type, values=("vit_b", "vit_l", "vit_h"),
                         state="readonly", width=6).pack(side="left")
        self.grid_side = tk.IntVar(value=8)
        self.cryolo_config = tk.StringVar(value=getattr(self.args, "cryolo_config", None) or "")
        self.cryolo_python = tk.StringVar(value=getattr(self.args, "cryolo_python", None) or "")
        self.cryolo_threshold = tk.StringVar(value=str(getattr(self.args, "cryolo_threshold", .3)))
        if is_cryolo:
            for label, variable in (("crYOLO configuration (.json)", self.cryolo_config),
                                    ("crYOLO Python executable", self.cryolo_python),
                                    ("Detection confidence threshold", self.cryolo_threshold)):
                ttk.Label(model, text=label).pack(anchor="w")
                ttk.Entry(model, textvariable=variable, width=36).pack(fill="x")
        else:
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
        ttk.Button(row, text="Select detections" if is_cryolo else "Segment prompt", command=self.segment_prompt).pack(side="left")
        ttk.Button(row, text="Clear prompt", command=self.clear_prompt).pack(side="left", padx=4)
        row = ttk.Frame(tools_frame)
        row.pack(fill="x", pady=3)
        ttk.Button(row, text="Next alternative", command=self.cycle_alternative).pack(side="left")
        ttk.Button(row, text="Accept mask", command=self.accept_alternative).pack(side="left", padx=4)
        ttk.Button(tools_frame, text="Generate automatic proposals", command=self.auto_proposals).pack(fill="x")
        ttk.Label(tools_frame, text=("Yellow: proposal / preview. Green: selected.\nBoxes and points select crYOLO detections; manual contours remain available."
                                    if is_cryolo else "Yellow: proposal / preview. Green: selected.\nSAM scores estimate mask quality, not target identity."),
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
        is_cryolo = getattr(self.args, "backend", "sam") == "cryolo"
        path = self.filedialog.askopenfilename(filetypes=([("crYOLO weights", "*.h5")] if is_cryolo else
                                                        [("PyTorch checkpoint", "*.pth *.pt")]) + [("All files", "*")])
        if path:
            self.checkpoint.set(path)
            for kind in ("vit_b", "vit_l", "vit_h"):
                if kind in Path(path).name:
                    self.model_type.set(kind)
                    break

    def _model_config(self):
        runtime = model_runtime_options(self.args)
        if runtime["backend"] == "cryolo":
            runtime.update(cryolo_config=self.cryolo_config.get().strip() or None,
                           cryolo_python=self.cryolo_python.get().strip() or None,
                           cryolo_threshold=float(self.cryolo_threshold.get()))
        config = RecognitionConfig(checkpoint=self.checkpoint.get(), model_type=self.model_type.get(),
                                 device=self.device.get(), points_per_side=int(self.grid_side.get()),
                                 automatic_profile=getattr(self.args, "automatic_profile", "capsid"),
                                 min_capsid_area_px=getattr(self.args, "min_capsid_area", DEFAULT_MIN_CAPSID_AREA_PX), **runtime)
        save_cryolo_settings(config, getattr(self.args, "cryolo_settings_path", None))
        return config

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
    parser.add_argument("--backend", choices=("sam", "cryolo"), default="sam")
    parser.add_argument("--checkpoint", help="Model weights: SAM .pth or crYOLO .h5; otherwise use the selected backend's bundled model.")
    parser.add_argument("--cryolo-config", help="crYOLO JSON configuration; defaults to cryolo-config.json beside its weights.")
    parser.add_argument("--cryolo-python", help="Python executable in the separate crYOLO environment.")
    parser.add_argument("--cryolo-threshold", type=float, default=.3, help="crYOLO detector confidence threshold (0 to 1).")
    parser.add_argument("--cryolo-timeout", type=float, default=180.0, help="Maximum seconds for a crYOLO prediction process.")
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--segment-image", help="Run headless recognition on a clean image, without any desktop events.")
    modes.add_argument("--diagnostics", action="store_true", help="Report installed dependency metadata without loading SAM.")
    modes.add_argument("--smoke-test", action="store_true", help="Noninteractive native-library, Tk, and CPU model check; never sends desktop input.")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--point", nargs=3, action="append", type=float, metavar=("X", "Y", "LABEL"), help="Repeatable foreground(1)/background(0) prompt.")
    parser.add_argument("--points-per-side", type=int, default=8)
    parser.add_argument("--automatic-profile", choices=("capsid", "capsid-learned", "generic"), default="capsid",
                        help="capsid keeps the existing filter; capsid-learned selects the optional experimental profile; generic returns unclassified SAM proposals.")
    parser.add_argument("--min-capsid-area", type=int, default=DEFAULT_MIN_CAPSID_AREA_PX, metavar="PIXELS",
                        help="Minimum connected capsid area in native screenshot pixels for automatic detection (default: 3500; 0 allows all sizes). Manual, explicit prompts, and generic detection are unchanged.")
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
        checkpoint = (resolve_checkpoint(args.checkpoint, backend="cryolo") if is_cryolo
                      else resolve_checkpoint(args.checkpoint))
        if is_cryolo and not args.cryolo_config:
            args.cryolo_config = str(checkpoint.with_name("cryolo-config.json"))
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
                          "candidates": len(candidates), "grid_points": len(points),
                          "elapsed_seconds": elapsed, "output": str(output)}
                if args.report:
                    write_json_report(args.report, report)
                print(json.dumps(report))
            finally:
                recognizer.close()
            return 0
        enable_dpi_awareness()
        if is_cryolo:
            configure_application_identity("SARU.Picker.crYOLO")
        else:
            configure_application_identity()
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
    global ACTIVE_BACKEND
    console_available = sys.stderr is not None
    with windowed_console_streams():
        parser = build_parser()
        if default_backend == "cryolo":
            parser.set_defaults(backend="cryolo", automatic_profile="capsid-learned")
        args = parser.parse_args(argv)
        previous_backend, ACTIVE_BACKEND = ACTIVE_BACKEND, args.backend
        try:
            if args.backend == "cryolo":
                settings_path = user_state_directory() / "cryolo-settings.json"
                allowed = {"checkpoint", "device", "cryolo_config", "cryolo_python", "cryolo_threshold",
                           "cryolo_timeout", "automatic_profile", "min_capsid_area"}
                if settings_path.is_file():
                    try:
                        settings = json.loads(settings_path.read_text(encoding="utf-8"))
                        if not isinstance(settings, dict):
                            raise ValueError("Settings must be a JSON object.")
                        settings = {key: value for key, value in settings.items() if key in allowed}
                        probe = parser.parse_args([])
                        for key, value in settings.items():
                            setattr(probe, key, value)
                        RecognitionConfig(probe.checkpoint or "cryolo-capsids.h5", device=probe.device,
                                          backend="cryolo", cryolo_config=probe.cryolo_config,
                                          cryolo_python=probe.cryolo_python,
                                          cryolo_threshold=probe.cryolo_threshold,
                                          cryolo_timeout=probe.cryolo_timeout,
                                          automatic_profile=probe.automatic_profile,
                                          min_capsid_area_px=probe.min_capsid_area)
                        parser.set_defaults(**settings)
                        args = parser.parse_args(argv)  # Explicit CLI values take precedence.
                    except (OSError, ValueError, TypeError) as exc:
                        APP_LOG.warning("Ignoring invalid crYOLO settings: %s", exc)
                args.cryolo_settings_path = str(settings_path)
            return _run_application(args, console_available=console_available)
        finally:
            ACTIVE_BACKEND = previous_backend


if __name__ == "__main__":
    raise SystemExit(main())

