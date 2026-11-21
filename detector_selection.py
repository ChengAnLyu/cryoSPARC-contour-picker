"""Four detection profiles and backend-specific persistent settings.

This module deliberately has no GUI or detector imports.  A detection profile
chooses both a backend and its contour behavior; checkpoint paths remain
separate so changing that choice cannot reuse another backend's weights.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import tempfile


DETECTION_PROFILES = ("capsid", "capsid-learned", "generic", "crYOLO")
_SAM_PROFILES = DETECTION_PROFILES[:3]
_SETTINGS_KEYS = frozenset({
    "detection_profile", "sam_checkpoint", "sam_model_type", "cryolo_checkpoint",
    "cryolo_config", "cryolo_threshold", "cryolo_timeout", "device",
    "points_per_side", "min_capsid_area",
})


def canonical_detection_profile(selection: str) -> str:
    """Return a displayed profile name; lowercase ``cryolo`` is its CLI alias."""
    if selection == "cryolo":
        return "crYOLO"
    if not isinstance(selection, str) or selection not in DETECTION_PROFILES:
        raise ValueError("Detection profile must be capsid, capsid-learned, generic, or crYOLO.")
    return selection


def resolve_detection_profile(selection: str) -> tuple[str, str]:
    """Map the four user choices to a backend and its internal contour profile."""
    selection = canonical_detection_profile(selection)
    if selection == "crYOLO":
        return "cryolo", "capsid-learned"
    return "sam", selection


def profile_from_backend(backend: str, automatic_profile: str) -> str:
    """Reverse the supported mapping without inventing a fifth profile."""
    if backend == "sam" and automatic_profile in _SAM_PROFILES:
        return automatic_profile
    if backend == "cryolo" and automatic_profile == "capsid-learned":
        return "crYOLO"
    raise ValueError("The selected backend and contour profile do not match one of the four detection profiles.")


def _optional_path(value, key: str, suffixes: tuple[str, ...] | None = None):
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"{key} must be a path string or None.")
    value = value.strip()
    if not value:
        return None
    if suffixes is not None and Path(value).suffix.lower() not in suffixes:
        raise ValueError(f"{key} must use {' or '.join(suffixes)} weights.")
    return value


def _finite_number(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number.")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{key} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number.")
    return value


def validate_detector_settings(payload) -> dict:
    """Validate known fields, ignoring foreign and legacy shared-path fields.

    Missing fields are left absent so callers can apply their current defaults.
    No model files are opened and no interpreter path is persisted here.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Detector settings must be a JSON object.")
    result = {}
    for key, value in payload.items():
        if key not in _SETTINGS_KEYS:
            continue
        if key == "detection_profile":
            result[key] = canonical_detection_profile(value)
        elif key == "sam_checkpoint":
            result[key] = _optional_path(value, key, (".pth", ".pt"))
        elif key == "cryolo_checkpoint":
            result[key] = _optional_path(value, key, (".h5",))
        elif key == "cryolo_config":
            result[key] = _optional_path(value, key)
        elif key == "sam_model_type":
            if not isinstance(value, str) or value not in {"vit_b", "vit_l", "vit_h"}:
                raise ValueError("sam_model_type must be vit_b, vit_l, or vit_h.")
            result[key] = value
        elif key == "device":
            if not isinstance(value, str) or value not in {"auto", "cpu", "cuda"}:
                raise ValueError("device must be auto, cpu, or cuda.")
            result[key] = value
        elif key == "cryolo_threshold":
            value = _finite_number(value, key)
            if not 0 <= value <= 1:
                raise ValueError("cryolo_threshold must be between 0 and 1.")
            result[key] = value
        elif key == "cryolo_timeout":
            value = _finite_number(value, key)
            if value <= 0:
                raise ValueError("cryolo_timeout must be greater than zero.")
            result[key] = value
        elif key in {"points_per_side", "min_capsid_area"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer.")
            if key == "points_per_side" and not 1 <= value <= 64:
                raise ValueError("points_per_side must be between 1 and 64.")
            if key == "min_capsid_area" and value < 0:
                raise ValueError("min_capsid_area must be nonnegative.")
            result[key] = value
    return result


def load_detector_settings(path: str | Path) -> dict:
    """Read this version's settings; missing files keep the caller's defaults."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Detector settings contain invalid JSON.") from exc
    return validate_detector_settings(payload)


def save_detector_settings(path: str | Path, payload) -> None:
    """Atomically save validated fields without altering older settings files."""
    clean = validate_detector_settings(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".detector-settings-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(clean, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
