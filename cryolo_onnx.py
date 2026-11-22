"""Offline execution of SARU's exported crYOLO network.

The decoder independently implements single-anchor, single-class YOLO box
mathematics. It does not import or redistribute the legacy crYOLO package.
ONNX Runtime is imported only by the isolated prediction worker.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ONNX_FORMAT = "saru-cryolo-onnx-v1"
STANDARD_H5_SHA256 = "a883be0bf39f0a93b0e2c36441c062fac5ab3672c044a7cf0703dc01853d3761"


def resolve_onnx_checkpoint(checkpoint) -> Path | None:
    """Locate the standard export without loading, hashing or importing ORT.

    Older installations saved an H5 path, sometimes outside the installation.
    Prefer its adjacent export, then look in this application's model locations.
    Keep the original selection in the caller's config: the worker must still
    validate the export's standard source identity and any existing selected H5.
    Explicit ONNX choices and custom H5 filenames are never substituted.
    """
    path = Path(checkpoint).expanduser()
    if path.suffix.lower() == ".onnx":
        return path
    if path.name.lower() != "cryolo-capsids.h5":
        return None
    candidates = [path.with_suffix(".onnx")]
    directories = []
    if getattr(sys, "frozen", False):
        if getattr(sys, "_MEIPASS", None):
            bundle = Path(sys._MEIPASS)
            directories.extend((bundle / "models", bundle))
        executable_directory = Path(sys.executable).resolve().parent
        directories.extend((executable_directory / "models", executable_directory))
    source_directory = Path(__file__).resolve().parent
    directories.extend((source_directory / "models", source_directory))
    candidates.extend(directory / "cryolo-capsids.onnx" for directory in directories)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_metadata(raw, *, expected_source_sha256=None, source_h5=None):
    """Accept only the explicit, tested full-frame capsid export contract."""
    if not isinstance(raw, str) or len(raw) > 65536:
        raise ValueError("The ONNX model is missing valid SARU detector metadata.")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("The ONNX detector metadata must be a JSON object.")
    required = {
        "format": ONNX_FORMAT, "architecture": "PhosaurusNet",
        "input_size": [320, 320], "input_layout": "NHWC", "output_layout": "NHWAC",
        "normalization": "STANDARD", "labels": ["particle"], "filament_model": False,
        "nms_threshold": .3, "estimated_size_rounding": "truncate",
        "extraction_box_size": 1, "preprocessing": "pil-grayscale-bilinear-standard-v1",
    }
    for name, expected in required.items():
        if data.get(name) != expected:
            raise ValueError(f"Unsupported ONNX detector metadata: {name}.")
    for name in ("input_name", "output_name"):
        if not isinstance(data.get(name), str) or not data[name]:
            raise ValueError(f"Missing ONNX tensor name: {name}.")
    if not isinstance(data.get("opset"), int) or not 13 <= data["opset"] <= 18:
        raise ValueError("Unsupported ONNX export version.")
    anchors = data.get("anchors")
    if (not isinstance(anchors, list) or len(anchors) != 2
            or any(isinstance(x, bool) or not isinstance(x, (float, int))
                   or not math.isfinite(x) or not 0 < x < 100 for x in anchors)):
        raise ValueError("The ONNX model requires two finite learned anchor values.")
    source = data.get("source_h5_sha256")
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdef" for char in source)):
        raise ValueError("The ONNX model has no valid source checkpoint identity.")
    if expected_source_sha256 is not None and source != expected_source_sha256:
        raise ValueError("The ONNX model does not match the saved standard crYOLO checkpoint. Install the matching offline model bundle.")
    if source_h5 is not None and file_sha256(source_h5) != source:
        raise ValueError("The ONNX model does not match the selected H5 weights. Select the matching ONNX model explicitly.")
    return data


def preprocess_image(rgb):
    """Pillow grayscale/bilinear preprocessing and the trained STANDARD scale."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("L")
    image = image.resize((320, 320), resample=Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32)
    denominator = np.float32(6 * float(values.std()) + 1e-6)
    normalized = (values - values.mean()) / denominator
    return np.ascontiguousarray(normalized[None, ..., None])


def decode_detections(raw, image_shape, anchors, threshold=.3, nms_threshold=.3):
    """Decode raw network outputs to the detector ROIs used by SARU contours."""
    logits = np.asarray(raw, dtype=np.float32)
    if logits.shape != (1, 20, 20, 1, 6) or not np.isfinite(logits).all():
        raise ValueError("The detector returned an invalid raw tensor.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between zero and one.")
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("Input image dimensions must be positive.")
    grid_h, grid_w = logits.shape[1:3]
    values = logits[0, :, :, 0]
    with np.errstate(over="ignore"):
        activated = np.float32(1) / (np.float32(1) + np.exp(-values[..., [0, 1, 4]]))
        extent = np.exp(values[..., 2:4])
    if not np.isfinite(extent).all():
        raise ValueError("The detector returned non-finite box dimensions.")
    xnorm = ((np.arange(grid_w)[None, :] + activated[..., 0]) / grid_w).astype(np.float32)
    ynorm = ((np.arange(grid_h)[:, None] + activated[..., 1]) / grid_h).astype(np.float32)
    wnorm = extent[..., 0] * np.float32(anchors[0]) / np.float32(grid_w)
    hnorm = extent[..., 1] * np.float32(anchors[1]) / np.float32(grid_h)
    # Preserve the float32 full-tile coordinate round trip before converting
    # to pixels. Legacy NumPy used float64 for scalar pixel multiplication.
    xnorm = xnorm * np.float32(width) / np.float32(width)
    ynorm = ynorm * np.float32(height) / np.float32(height)
    wnorm = wnorm * np.float32(width) / np.float32(width)
    hnorm = hnorm * np.float32(height) / np.float32(height)
    confidence = activated[..., 2].reshape(-1)
    regions = np.stack((xnorm, ynorm, wnorm, hnorm), axis=-1).reshape(-1, 4).astype(np.float64)
    eligible = np.flatnonzero(confidence > min(.1, threshold))
    if not len(eligible):
        return []
    order = eligible[np.argsort(confidence[eligible], kind="quicksort")[::-1]]
    lower = regions[:, :2] - regions[:, 2:] / 2
    upper = regions[:, :2] + regions[:, 2:] / 2
    areas = np.prod(regions[:, 2:], axis=1)
    retained = []
    while len(order):
        chosen, others = order[0], order[1:]
        retained.append(int(chosen))
        overlap = np.maximum(0, np.minimum(upper[chosen], upper[others]) - np.maximum(lower[chosen], lower[others]))
        intersection = np.prod(overlap, axis=1)
        iou = intersection / (areas[chosen] + areas[others] - intersection)
        order = others[iou < nms_threshold]
    result = []
    for index in retained:
        score = float(confidence[index])
        if score < threshold:
            continue
        cx, cy, bw, bh = regions[index] * np.array([width, height, width, height], dtype=np.float64)
        # Match the 1px extraction-box border rule. Learned extents, including
        # partially clipped targets, remain independent of that extraction box.
        if not (.5 <= cx < width - .5 and .5 < cy <= height - .5):
            continue
        bw, bh = int(bw), int(bh)
        if bw <= 0 or bh <= 0:
            raise ValueError("The detector returned nonpositive estimated dimensions.")
        result.append({"x0": cx-bw/2, "y0": cy-bh/2, "x1": cx+bw/2,
                       "y1": cy+bh/2, "confidence": score})
    return sorted(result, key=lambda box: (-box["confidence"], box["y0"], box["x0"]))


def run_prediction(model, image_path, *, threshold=.3, cpu_threads=4,
                   expected_source_sha256=None, source_h5=None):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    if not 1 <= cpu_threads <= 64:
        raise ValueError("CPU thread count must be between 1 and 64.")
    started = time.monotonic()
    print("Loading offline crYOLO model", flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = cpu_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = validate_metadata(session.get_modelmeta().custom_metadata_map.get("saru_cryolo"),
                                 expected_source_sha256=expected_source_sha256, source_h5=source_h5)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != metadata["input_name"] or inputs[0].type != "tensor(float)"
            or inputs[0].shape[1:] != [320, 320, 1]
            or len(outputs) != 1 or outputs[0].name != metadata["output_name"]
            or outputs[0].type != "tensor(float)" or outputs[0].shape[1:] != [20, 20, 1, 6]):
        raise ValueError("ONNX tensor signatures do not match the exported detector metadata.")
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    print("Running offline crYOLO prediction", flush=True)
    raw = session.run([metadata["output_name"]], {metadata["input_name"]: preprocess_image(rgb)})[0]
    detections = decode_detections(raw, rgb.shape[:2], metadata["anchors"], threshold, metadata["nms_threshold"])
    return {"format": "saru-cryolo-detections-v1", "runtime_format": "onnx",
            "runtime": "onnxruntime-cpu", "runtime_version": ort.__version__,
            "source_h5_sha256": metadata["source_h5_sha256"], "model_sha256": file_sha256(model),
            "image_shape": list(rgb.shape[:2]), "detections": detections,
            "elapsed_seconds": time.monotonic()-started}


def _execute_worker(args):
    try:
        result = run_prediction(args.model, args.input, threshold=args.threshold, cpu_threads=args.cpu_threads,
                                source_h5=args.source_h5, expected_source_sha256=args.expected_source_sha256)
        args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        print(f"Offline crYOLO completed: {len(result['detections'])} detections", flush=True)
        return 0
    except Exception as exc:
        print(f"Offline crYOLO prediction failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def onnx_worker_main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated offline crYOLO prediction worker")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=.3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--source-h5", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    if args.log is not None:
        # Frozen console=False workers have sys.stdout/stderr=None even when
        # OS handles are redirected. An explicit stream retains useful errors.
        with args.log.open("a", encoding="utf-8", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                return _execute_worker(args)
    return _execute_worker(args)


main = onnx_worker_main


if __name__ == "__main__":
    raise SystemExit(onnx_worker_main())
