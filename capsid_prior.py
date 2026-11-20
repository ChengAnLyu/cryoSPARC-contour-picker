"""Optional, locally trained positive-unlabeled capsid appearance profile.

The numeric JSON profile is independent of SAM and uses NumPy/SciPy inference.
Its bounded neural score is NOT a calibrated capsid probability: selected
positive examples are incomplete and confidence biased. No training images,
annotations, network access, or executable serialization are used at runtime.
"""
from __future__ import annotations

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
