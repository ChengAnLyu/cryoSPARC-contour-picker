"""Conservative screen-context guard for contours following carbon-film edges.

This is a geometric veto, not a trained carbon classifier.  A long coherent
edge is found once in the clean screenshot; individual final proposals must
then follow that edge for a substantial fraction of their boundary and have
an elongated footprint along it.  Proximity, darkness, an upstream SAM parent,
and an unmarked region are never sufficient grounds for rejection.

The guard intentionally fails open for ambiguous broad regions, curved support,
short straight capsid facets, and compact objects wholly inside a support strip.
It does not modify the screenshot or masks, and does not delete a whole side of
an edge.  Call it only for the two automatic capsid profiles, after refinement.
"""
from __future__ import annotations

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
