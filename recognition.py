"""Headless SAM proposals and bitmap-based sampling for SARU Picker.

SAM's predicted IoU is an estimate of mask quality, not the probability that a
mask is a viral capsid.  Automatic results therefore require review.  This module
does not send desktop events and can be imported without PyTorch or a display.
"""
from __future__ import annotations

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

from capsid import filter_capsid_proposals, capsid_guided_prompts, filter_capsid_with_profile
from capsid_prior import resolve_profile_path, load_profile, score_image
from carbon_support import detect_carbon_edges, guard_proposals

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
    automatic_profile: str = "capsid"
    capsid_prior_path: str | None = None

    def __post_init__(self) -> None:
        if self.model_type not in {"vit_b", "vit_l", "vit_h"}:
            raise ValueError("model_type must be vit_b, vit_l, or vit_h.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.automatic_profile not in {"capsid", "capsid-learned", "generic"}:
            raise ValueError("automatic_profile must be capsid, capsid-learned, or generic.")
        if self.capsid_prior_path is not None and (not isinstance(self.capsid_prior_path, str) or not self.capsid_prior_path.strip()):
            raise ValueError("capsid_prior_path must be a nonempty path string or None.")
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
        if key == self._automatic_image_key and self._automatic_result is not None:
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
            if is_capsid:
                if profile is not None:
                    raw = filter_capsid_with_profile(
                        rgb, raw, appearance_scores, check_cancel=lambda: _check_cancel(cancel),
                        min_area_ratio=self.config.min_area_ratio,
                        max_area_ratio=min(self.config.max_area_ratio, .40))
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
                "timing_seconds": {"appearance": feature_seconds, "guided_proposals": proposal_seconds,
                                   "sam": sam_seconds, "carbon_guard": carbon_seconds,
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
