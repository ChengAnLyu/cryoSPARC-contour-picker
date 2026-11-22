"""Frozen-runtime checks of positive, dense and cancelled crYOLO refinement.

Detector boxes and the constant numeric appearance fixture are deliberately
synthetic. These checks execute the production contour path; the separate H5
smoke check executes real crYOLO prediction. Neither measures capsid accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from capsid_prior import FEATURE_NAMES, PROFILE_FORMAT
from cryolo_recognition import CryoloBox, CryoloRecognizer
from recognition import Cancelled, RecognitionConfig


REFINEMENT_LIMIT_SECONDS = 120.0


def make_fixture(directory: Path):
    """Create a known textured interior and a nonprivate constant-score MLP."""
    profile = {
        'format': PROFILE_FORMAT, 'feature_names': list(FEATURE_NAMES),
        'mean': [0.] * len(FEATURE_NAMES), 'scale': [1.] * len(FEATURE_NAMES),
        'layers': [
            {'weight': np.zeros(shape).tolist(),
             'bias': [1.] if shape[0] == 1 else [0.] * shape[0]}
            for shape in ((32, len(FEATURE_NAMES)), (16, 32), (1, 16))
        ],
        'training': {'purpose': 'Synthetic constant model for packaging checks; not trained.'},
    }
    (directory / 'capsid-profile.json').write_text(json.dumps(profile), encoding='utf-8')
    rng = np.random.default_rng(48)
    yy, xx = np.mgrid[:512, :768]
    interior = ((xx - 300) / 84) ** 2 + ((yy - 256) / 96) ** 2 < 1
    gray = 125 + rng.normal(0, 3, interior.shape)
    gray[interior] = 125 + rng.normal(0, 42, int(interior.sum()))
    rgb = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    boxes = [CryoloBox(200 + i % 8, 140 + i // 8 % 8,
                      400 + i % 8, 380 + i // 8 % 8, .99 - i * .001)
             for i in range(128)]
    config = RecognitionConfig(str(directory / 'synthetic-not-loaded.h5'),
                               backend='cryolo', automatic_profile='capsid-learned',
                               device='cpu', min_capsid_area_px=3500)
    return rgb, interior, boxes, config


def _assert_positive(candidates, model, interior, supplied_count):
    if not candidates:
        raise AssertionError('Nonempty synthetic detections did not reach positive contour refinement.')
    report = model.last_detection
    assert report['detections_supplied'] is True
    assert report['cryolo_detections'] == supplied_count
    assert report['appearance_model']['status'] == 'loaded'
    assert report['carbon_guard']['enabled'] is True
    assert report['capsid_area_filter']['enabled'] is True
    expected_score = 1 / (1 + np.exp(-1.))
    for candidate in candidates:
        assert candidate.mask.shape == interior.shape and candidate.mask.dtype == np.bool_
        assert candidate.area >= 3500
        assert candidate.predicted_iou is None and candidate.stability_score is None
        assert np.count_nonzero(candidate.mask & interior) > .5 * np.count_nonzero(interior)
        evidence = report['capsid_evidence'][candidate.id]
        assert np.isclose(evidence['appearance_score_mean'], expected_score, atol=1e-5)
        assert evidence['cryolo_confidence'] is not None
        assert evidence['confidence_is_mask_iou'] is False


def verify():
    checks = []
    with tempfile.TemporaryDirectory(prefix='saru-refinement-check-') as folder:
        rgb, interior, boxes, config = make_fixture(Path(folder))
        model = CryoloRecognizer(config)
        try:
            for label, supplied in (('positive', boxes[:1]), ('dense', boxes)):
                started = time.monotonic()
                deadline = started + REFINEMENT_LIMIT_SECONDS
                candidates = model.refine_detections(rgb, supplied,
                                                     cancel=lambda: time.monotonic() >= deadline)
                elapsed = time.monotonic() - started
                _assert_positive(candidates, model, interior, len(supplied))
                if elapsed >= REFINEMENT_LIMIT_SECONDS:
                    raise AssertionError(f'{label} refinement exceeded its completion deadline.')
                checks.append({'name': label, 'status': 'ok', 'boxes': len(supplied),
                               'contours': len(candidates), 'seconds': elapsed,
                               'areas': [candidate.area for candidate in candidates],
                               'appearance_scores_computed': True, 'carbon_guard_executed': True})
            started = time.monotonic()
            assert model.refine_detections(np.full_like(rgb, 125), boxes[:1],
                                          cancel=lambda: time.monotonic() - started >= REFINEMENT_LIMIT_SECONDS) == []
            checks.append({'name': 'blank_with_box', 'status': 'ok', 'boxes': 1, 'contours': 0,
                           'seconds': time.monotonic() - started})
            calls = 0
            def cancel_during_refinement():
                nonlocal calls
                calls += 1
                return calls >= 140
            # More than128 checks permits the box-mask preparation to finish
            # and reaches the actual appearance/refinement work before stopping.
            started = time.monotonic()
            try:
                model.refine_detections(rgb, boxes, cancel=cancel_during_refinement)
            except Cancelled:
                elapsed = time.monotonic() - started
                assert calls >= 140
                if elapsed >= 10:
                    raise AssertionError('Refinement cancellation took more than10 seconds.')
                checks.append({'name': 'cancellation', 'status': 'ok', 'checks': calls,
                               'seconds': elapsed, 'partial_result_returned': False})
            else:
                raise AssertionError('Cancellation during dense refinement was not propagated.')
        finally:
            model.close()
    return {'status': 'ok', 'mode': 'cryolo-refinement-smoke', 'backend': 'cryolo',
            'frozen': bool(getattr(sys, 'frozen', False)), 'platform': sys.platform,
            'image_shape': list(rgb.shape), 'deadline_seconds': REFINEMENT_LIMIT_SECONDS,
            'checks': checks, 'actual_crYOLO_prediction': False,
            'note': 'Production refinement on deterministic supplied boxes and a generated constant appearance model. Real prediction is verified separately; no trained-model accuracy claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify()
    except Exception as error:
        report = {'status': 'error', 'mode': 'cryolo-refinement-smoke',
                  'error': str(error), 'error_type': type(error).__name__,
                  'frozen': bool(getattr(sys, 'frozen', False))}
    from app import APP_VERSION
    report['version'] = APP_VERSION
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
