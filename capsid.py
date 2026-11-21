"""Conservative dense-core proposals for perforated VLP screen images.

This is an image-contrast heuristic on top of SAM object proposals, not a
trained capsid classifier. SAM's predicted IoU estimates mask quality, not the
probability of a capsid. Consequently, broad background masks and thin
membranes are scored using the clean screenshot's local texture before any
nested-mask suppression is performed.

No annotation images, coordinates, scientific labels, network services, or
model downloads are used here. The output can miss faint capsids and fragments
and must remain reviewable. Dense cores receive simplified polygonal outer
boundaries, without forcing a fixed number of sides or a convex hull. Interior
exclusions and frame-truncated targets remain supported.
"""
from __future__ import annotations

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
        distances = ndi.distance_transform_edt(mask)
        cy, cx = np.unravel_index(np.argmax(distances * scores), scores.shape)
        rank = float(np.mean(scores[mask]) * math.sqrt(mask.sum()))
        proposals.append({'box': box, 'point': [int(cx), int(cy)], 'rank': rank,
                          'support_area': int(mask.sum()), 'shape': shape})
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
    return (np.count_nonzero(first['segmentation'] & second['segmentation'])
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
            annulus = (ndi.binary_dilation(component, iterations=20)
                       & ~ndi.binary_dilation(component, iterations=3) & valid)
            inside = float(np.median(texture[component]))
            outside = float(np.median(texture[annulus])) if np.any(annulus) else global_median
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
                            'bbox': _capsid_bbox(component), 'predicted_iou': None,
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
