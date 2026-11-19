"""The original manual contour's screen-coordinate sampling, without a model.

The bounding-box phase, integer row height, alternating integer offset, and
boundary rule are deliberately preserved from the original picking script.
Model masks use the same lattice with actual bitmap membership, preserving
holes. No screen input is sent here; callers must receive the complete result
before starting playback.
"""
from __future__ import annotations

import math

import numpy as np

from execution import CancelledOperation


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
