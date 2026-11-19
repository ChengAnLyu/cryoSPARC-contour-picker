"""Conservative dense-core proposals for perforated VLP screen images.

This is an image-contrast heuristic on top of SAM object proposals, not a
trained capsid classifier. SAM's predicted IoU estimates mask quality, not the
probability of a capsid. Consequently, broad background masks and thin
membranes are scored using the clean screenshot's local texture before any
nested-mask suppression is performed.

No annotation images, coordinates, scientific labels, network services, or
model downloads are used here. The output can miss faint capsids and fragments
and must remain reviewable. Polygonality and a closed envelope are deliberately
not mandatory: the examples include irregular and frame-truncated targets.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
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
    return selected

