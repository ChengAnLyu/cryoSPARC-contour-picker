#!/usr/bin/env python3
"""SARU Picker 2: responsive SAM mask review and cancellable desktop picking.
Generated from the companion tested modules. Run python main.py --help.
"""
from __future__ import annotations


# ---------------- recognition.py ----------------

from dataclasses import dataclass
import gc
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

CancelCheck = Callable[[], bool] | None


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

    def __post_init__(self) -> None:
        if self.model_type not in {"vit_b", "vit_l", "vit_h"}:
            raise ValueError("model_type must be vit_b, vit_l, or vit_h.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
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
                      cancel: CancelCheck = None) -> list[Candidate]:
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
        self._automatic_result: list[Candidate] | None = None
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
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
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

    def automatic(self, rgb: np.ndarray | Image.Image,
                  cancel: CancelCheck = None) -> list[Candidate]:
        """Generate unclassified region proposals on a sparse grid for review."""
        rgb = validate_rgb(rgb)
        _check_cancel(cancel)
        key = _image_key(rgb)
        if key == self._automatic_image_key and self._automatic_result is not None:
            return list(self._automatic_result)
        self._ensure_model(cancel)
        self._active_cancel = cancel
        try:
            with self._torch.inference_mode():
                raw = self._generator.generate(rgb)
            _check_cancel(cancel)
            candidates = []
            for item in raw:
                _check_cancel(cancel)
                mask = np.asarray(item["segmentation"])
                if mask.shape != rgb.shape[:2]:
                    raise RecognitionError("SAM returned a mask with the wrong image dimensions.")
                if np.any(mask):
                    candidates.append(make_candidate(
                        mask, predicted_iou=item.get("predicted_iou"),
                        stability_score=item.get("stability_score"),
                        source="automatic proposal", image_key=key))
            result = filter_candidates(candidates, self.config, cancel)
            _check_cancel(cancel)
            self._automatic_image_key, self._automatic_result = key, result
            return list(result)
        except (Cancelled, RecognitionError):
            raise
        except Exception as exc:
            raise RecognitionError(
                f"SAM automatic proposals failed: {exc}. For memory errors, reduce "
                "points_per_batch or select CPU; for missing targets, use a box/point prompt.") from exc
        finally:
            self._active_cancel = None
            # Automatic generator resets its own predictor. It does not replace
            # the independent prompted predictor's cached image embedding.

    def close(self) -> None:
        """Release model/cache resources when no inference is running."""
        torch, device = self._torch, self.device
        self._model = self._predictor = self._generator = None
        self._predictor_image_key = self._automatic_image_key = None
        self._automatic_result = None
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


# ---------------- execution.py ----------------


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
            self._busy = True
            self._jobs.put_nowait((token, fn))
            return token

    def poll_results(self) -> list[TaskResult]:
        """Collect completed work; call this from the GUI's event loop."""
        results = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                if self._active_token is result.token:
                    self._active_token = None
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
        return not self._thread.is_alive()

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            item = self._jobs.get()
            if item is None:
                return
            token, fn = item
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
                result = TaskResult(token=token, error=exc, traceback_text=trace)
            self._results.put_nowait(result)


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


# ---------------- desktop.py ----------------


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
APP_VERSION = "2.0.0"
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


def _smoke_windows_runtime(log_path: Path) -> dict[str, Any]:
    # Importing desktop libraries is deliberate; no listener is started and no
    # screenshot, mouse action, or key is sent in this packaging check.
    enable_dpi_awareness()
    tkinter = importlib.import_module("tkinter")
    window = tkinter.Tk()
    picker = None
    try:
        window.withdraw()
        args = argparse.Namespace(checkpoint=None, device="cpu", model_type="vit_b", image=None)
        picker = PickerApp(window, args, log_path, start_hotkeys=False)
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[:] = (85, 105, 125)
        picker._set_image(rgb, description="Packaging smoke image")
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[12:52, 20:76] = True
        picker._add_candidate(make_candidate(mask, source="smoke-test"), select=True)
        window.update_idletasks()
        picker.render()
        window.update_idletasks()
        if (not picker.canvas.find_all() or not hasattr(picker, "photo")
                or picker.photo.width() < 1 or picker.photo.height() < 1
                or len(picker.table.get_children()) != 1 or picker.hotkeys is not None):
            raise RuntimeError("The packaged application did not construct and render its review interface.")
        tk_version = str(window.tk.call("info", "patchlevel"))
    finally:
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
    parser = argparse.ArgumentParser(description="SARU Picker: prompted SAM and reviewed mask sampling.")
    parser.add_argument("--image", help="Open an image in the GUI.")
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
        PickerApp(root, args, log_path)
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

