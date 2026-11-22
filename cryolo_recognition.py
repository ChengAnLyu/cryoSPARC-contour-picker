"""Real crYOLO detection with the existing SARU contour refinement.

crYOLO is a box detector, not a segmenter. Its objectness is kept separate from
SAM mask-IoU fields. The selected capsid appearance/texture filter traces each
box's image-supported interior; no SAM checkpoint or runtime is loaded here.
The legacy TensorFlow runtime is isolated behind ``python -m cryolo.predict``.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import logging
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

from recognition import (Cancelled, RecognitionConfig, RecognitionError,
                         _check_cancel, _filter_capsid_area, _image_key,
                         filter_candidates, make_candidate, validate_rgb)
from capsid import filter_capsid_proposals, filter_capsid_with_profile
from capsid_prior import load_profile, resolve_profile_path, score_image
from carbon_support import detect_carbon_edges, guard_proposals
from runtime_management import (ensure_cryolo_runtime, RuntimeSetupError, RuntimeSetupCancelled,
                                external_process_environment)


CRYOLO_LOG = logging.getLogger("saru.cryolo")


def _report_progress(progress, message):
    """Report from the worker; GUI callers enqueue this instead of touching Tk."""
    CRYOLO_LOG.info(message)
    if progress is not None:
        progress(message)


def _child_process_options():
    # A windowed frozen parent otherwise gives python.exe a visible console,
    # which can cover the micrograph or steal focus during live recognition.
    return ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
            else {"start_new_session": True})


def _prediction_log_tail(path):
    try:
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 8000))
            return log.read(8000).decode("utf-8", errors="replace").strip()
    except OSError:
        return "Prediction log was unavailable."


def _prediction_log_directory():
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "SARU_Picker"
    return Path.home() / ".saru_picker"


def _retain_prediction_log(path):
    """Replace one bounded diagnostic log; never keep screenshots or weights."""
    temporary = None
    try:
        if not path.is_file():
            return None
        directory = _prediction_log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "cryolo-prediction.log"
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 256 * 1024))
            tail = source.read(256 * 1024)
        with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".cryolo-log-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(tail)
        os.replace(temporary, destination)
        return destination
    except OSError:
        CRYOLO_LOG.warning("Could not retain the crYOLO prediction log", exc_info=True)
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                CRYOLO_LOG.warning("Could not remove an incomplete crYOLO diagnostic log")


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
            with external_process_environment() as child_env:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=2, check=False,
                               env=child_env, **_child_process_options())
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
        self.last_prediction_log = None

    def _paths(self, cancel=None, progress=None):
        _report_progress(progress, "Checking crYOLO model files")
        weights = Path(self.config.checkpoint).expanduser().resolve()
        config_value = getattr(self.config, "cryolo_config", None)
        config_path = (Path(config_value).expanduser().resolve() if config_value
                       else weights.with_name("cryolo-config.json"))
        # Check the chosen model before spending time preparing its runtime.
        for label, path in (("weights", weights), ("configuration", config_path)):
            if not path.is_file():
                raise RecognitionError(f"crYOLO {label} not found: {path}")
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("model"), dict):
                raise ValueError("Missing model configuration.")
        except (OSError, ValueError) as exc:
            raise RecognitionError(f"Invalid crYOLO configuration: {exc}") from exc
        python_value = getattr(self.config, "cryolo_python", None) or os.environ.get("SARU_CRYOLO_PYTHON")
        if not python_value:
            try:
                python_value = str(ensure_cryolo_runtime(
                    cancel=cancel, progress=lambda message: _report_progress(progress, message)))
            except RuntimeSetupCancelled as exc:
                raise Cancelled(str(exc)) from exc
            except (RuntimeSetupError, OSError) as exc:
                raise RecognitionError(f"Could not prepare the crYOLO runtime: {exc}") from exc
        _check_cancel(cancel)
        executable = shutil.which(str(python_value))
        if executable is None and Path(python_value).expanduser().is_file():
            executable = os.path.abspath(os.path.expanduser(python_value))
        if not executable:
            raise RecognitionError(f"The configured crYOLO runtime was not found: {python_value}. Run SARU Picker Setup to repair the runtime, or correct the explicit --cryolo-python override.")
        # Keep the venv executable spelling: resolving its symlink would
        # bypass pyvenv.cfg and lose the separately installed crYOLO packages.
        executable = os.path.abspath(executable)
        return executable, weights, config_path

    def _predict(self, rgb, cancel, progress=None):
        _check_cancel(cancel)
        executable, weights, config_path = self._paths(cancel, progress)
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in (weights, config_path))
        key = _image_key(rgb)
        if key == self._key and signature == self._artifact_signature and self._boxes is not None:
            _report_progress(progress, "Using cached crYOLO detections")
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
            _report_progress(progress, "Loading crYOLO model and predicting")
            try:
                with log_path.open("wb") as log:
                    with external_process_environment(env) as child_env:
                        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                         stdout=log, stderr=subprocess.STDOUT,
                                                         cwd=folder, env=child_env,
                                                         **_child_process_options())
                    while self._process.poll() is None:
                        _check_cancel(cancel)
                        if time.monotonic() - started > getattr(self.config, "cryolo_timeout", 180):
                            tail = _prediction_log_tail(log_path)
                            raise RecognitionError(
                                f"crYOLO prediction timed out after {time.monotonic() - started:.1f}s. "
                                f"The prediction process was stopped. Last runtime output:\n{tail}")
                        time.sleep(.05)
                    code = self._process.returncode
                _check_cancel(cancel)
                if code:
                    tail = _prediction_log_tail(log_path)
                    raise RecognitionError(f"crYOLO prediction failed (exit {code}).\n{tail}")
                _report_progress(progress, "Reading crYOLO detections")
                try:
                    boxes = read_cbox(output / "CBOX" / "screen.cbox", rgb.shape[:2], threshold)
                except RecognitionError as exc:
                    raise RecognitionError(f"{exc}\nLast runtime output:\n{_prediction_log_tail(log_path)}") from exc
                CRYOLO_LOG.info("crYOLO prediction completed in %.2fs: %d detections", time.monotonic()-started, len(boxes))
            except Cancelled:
                CRYOLO_LOG.info("crYOLO prediction cancelled")
                raise
            except RecognitionError as exc:
                retained = _retain_prediction_log(log_path)
                CRYOLO_LOG.exception("crYOLO prediction did not complete")
                if retained is not None:
                    raise RecognitionError(f"{exc}\nPrediction log: {retained}") from exc
                raise
            except OSError as exc:
                raise RecognitionError(f"Could not start the crYOLO Python runtime: {exc}") from exc
            finally:
                if self._process is not None:
                    _stop_process(self._process)
                    self._process = None
                self.last_prediction_log = _retain_prediction_log(log_path)
        _check_cancel(cancel)
        self._key, self._artifact_signature, self._boxes = key, signature, list(boxes)
        return boxes, False

    def _refine(self, rgb, boxes, cancel, *, prompt=False, progress=None):
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
            _report_progress(progress, "Scoring capsid appearance")
            scores = score_image(rgb, profile, check)
            _report_progress(progress, "Refining contours")
            raw = filter_capsid_with_profile(rgb, raw, scores, check_cancel=check,
                                             min_area_ratio=0 if prompt else self.config.min_area_ratio,
                                             max_area_ratio=1 if prompt else min(self.config.max_area_ratio, .4),
                                             min_capsid_area_px=minimum)
        else:
            _report_progress(progress, "Refining contours")
            raw = filter_capsid_proposals(rgb, raw, check_cancel=check,
                                         min_area_ratio=0 if prompt else self.config.min_area_ratio,
                                         max_area_ratio=1 if prompt else min(self.config.max_area_ratio, .4))
        _report_progress(progress, "Checking carbon support")
        context = detect_carbon_edges(rgb, check_cancel=check)
        raw, carbon = guard_proposals(context, raw, check_cancel=check)
        carbon["enabled"] = True
        raw, area = _filter_capsid_area(raw, minimum, cancel)
        _report_progress(progress, "Finishing contours")
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

    def automatic(self, rgb, cancel=None, progress=None):
        rgb = validate_rgb(rgb)
        started = time.perf_counter()
        boxes, cached = self._predict(rgb, cancel, progress)
        key = _image_key(rgb)
        if cached and self._automatic_key == key and self._automatic_result is not None:
            self.last_detection = copy.deepcopy(self._automatic_report)
            self.last_detection["detections_cached"] = True
            return list(self._automatic_result)
        candidates = self.refine_detections(rgb, boxes, cancel=cancel, progress=progress)
        self.last_detection["detections_cached"] = cached
        self.last_detection["detections_supplied"] = False
        self.last_detection["timing_seconds"]["total_including_initial_model_load"] = time.perf_counter() - started
        self._automatic_key, self._automatic_result = key, list(candidates)
        self._automatic_report = copy.deepcopy(self.last_detection)
        return candidates

    def refine_detections(self, rgb, boxes, cancel=None, *, prompt=False, progress=None):
        """Run the exact production contour tail on parsed batch CBOX detections.

        This supports evaluation with one crYOLO process for a whole dataset.
        It does not perform detection or imply that external boxes are held out.
        """
        rgb = validate_rgb(rgb)
        boxes = list(boxes)
        if any(not isinstance(box, CryoloBox) for box in boxes):
            raise ValueError("Detections must be CryoloBox instances from read_cbox.")
        started = time.perf_counter()
        candidates, evidence, carbon, area, appearance = self._refine(rgb, boxes, cancel, prompt=prompt, progress=progress)
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
               labels: Sequence[int] | None = None, cancel=None, progress=None):
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
        boxes, cached = self._predict(rgb, cancel, progress)
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
        candidates, evidence, carbon, area, appearance = self._refine(rgb, selected, cancel, prompt=True, progress=progress)
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
