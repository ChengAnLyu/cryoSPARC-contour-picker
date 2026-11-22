"""Public synthetic ONNX inference and contour checks; no private weights.

The tiny Conv graph emits one known box around a textured ellipse. It tests the
real CPU engine, decoder, worker boundary and positive refinement, not accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image


def make_fixture(directory: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from verify_refinement import make_fixture as make_refinement_fixture
    from cryolo_onnx import STANDARD_H5_SHA256

    directory.mkdir(parents=True, exist_ok=True)
    rgb, _, _, _ = make_refinement_fixture(directory)
    Image.fromarray(rgb).save(directory / 'synthetic.png')
    raw = np.zeros((1, 20, 20, 6), dtype=np.float32)
    raw[..., 4] = -20
    # x=300,y=260,w=200,h=240 in the 768x512 fixture, before pixel truncation.
    raw[0, 10, 7] = [np.log(.8125 / .1875), np.log(.15625 / .84375),
                      np.log(200 * 20 / 768), np.log(240 * 20 / 512), 5, 0]
    weights = np.full((6, 1, 16, 16), 1e-6, dtype=np.float32)
    graph = helper.make_graph([
        helper.make_node('Transpose', ['image'], ['nchw'], perm=[0, 3, 1, 2]),
        helper.make_node('Conv', ['nchw', 'kernel'], ['features'], strides=[16, 16]),
        helper.make_node('Transpose', ['features'], ['nhwc'], perm=[0, 2, 3, 1]),
        helper.make_node('Add', ['nhwc', 'fixture_logits'], ['logits']),
        helper.make_node('Reshape', ['logits', 'output_shape'], ['detections']),
    ], 'synthetic_packaging_fixture',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 320, 320, 1])],
        [helper.make_tensor_value_info('detections', TensorProto.FLOAT, [1, 20, 20, 1, 6])],
        [numpy_helper.from_array(weights, 'kernel'),
         numpy_helper.from_array(raw, 'fixture_logits'),
         numpy_helper.from_array(np.array([1, 20, 20, 1, 6], dtype=np.int64), 'output_shape')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=10)
    metadata = {
        'format': 'saru-cryolo-onnx-v1', 'architecture': 'PhosaurusNet',
        # Exercise the production migration identity contract without shipping
        # any private checkpoint. The graph remains explicitly synthetic.
        'source_h5_sha256': STANDARD_H5_SHA256,
        'input_name': 'image', 'output_name': 'detections',
        'input_size': [320, 320], 'input_layout': 'NHWC', 'output_layout': 'NHWAC',
        'anchors': [1., 1.], 'normalization': 'STANDARD', 'labels': ['particle'],
        'filament_model': False, 'opset': 13, 'nms_threshold': .3,
        'estimated_size_rounding': 'truncate', 'extraction_box_size': 1,
        'preprocessing': 'pil-grayscale-bilinear-standard-v1',
    }
    helper.set_model_props(model, {'saru_cryolo': json.dumps(metadata),
                                  'saru_fixture': 'Synthetic Conv graph; not a trained model.'})
    onnx.checker.check_model(model)
    onnx.save(model, directory / 'cryolo-capsids.onnx')
    config = {'model': {'architecture': 'PhosaurusNet', 'input_size': [320, 320],
                        'anchors': [1., 1.], 'norm': 'STANDARD', 'max_box_per_image': 32,
                        'labels': ['particle']}}
    (directory / 'cryolo-config.json').write_text(json.dumps(config), encoding='utf-8')


def verify(directory: Path) -> dict:
    from cryolo_recognition import CryoloRecognizer
    from recognition import RecognitionConfig, RecognitionError

    rgb = np.asarray(Image.open(directory / 'synthetic.png').convert('RGB'))
    config = RecognitionConfig(str(directory / 'cryolo-capsids.onnx'), backend='cryolo',
                               cryolo_config=str(directory / 'cryolo-config.json'),
                               automatic_profile='capsid-learned', device='cpu',
                               cryolo_threshold=.3, cryolo_timeout=90, min_capsid_area_px=3500)
    started = time.monotonic()
    model = CryoloRecognizer(config)
    try:
        result = model.automatic(rgb)
        details = model.last_detection
        assert result, 'Actual ONNX inference did not produce a refined positive contour.'
        assert details['cryolo_detections'] == 1, details
        assert details['appearance_model']['status'] == 'loaded'
        assert details['carbon_guard']['enabled'] is True
        assert details['capsid_area_filter']['enabled'] is True
        assert all(candidate.area >= 3500 for candidate in result)
        assert all(candidate.mask.shape == rgb.shape[:2] for candidate in result)
        serialized = json.dumps(details)
        assert 'onnxruntime-cpu' in serialized and 'onnx' in serialized, details
        # A windowed frozen worker has no Python stdout/stderr by default.
        # An actual bad model must preserve its cause through the worker log,
        # rather than surface a generic exit status or hang in runtime repair.
        with tempfile.TemporaryDirectory(prefix='saru-invalid-onnx-') as invalid_directory:
            invalid_model = Path(invalid_directory) / 'invalid.onnx'
            invalid_model.write_bytes(b'This intentionally is not an ONNX protobuf model.')
            invalid = CryoloRecognizer(replace(config, checkpoint=str(invalid_model)))
            try:
                try:
                    invalid.automatic(rgb)
                except RecognitionError as error:
                    assert 'Offline crYOLO prediction failed:' in str(error), str(error)
                else:
                    raise AssertionError('Malformed ONNX model unexpectedly succeeded.')
            finally:
                invalid.close()
        return {'status': 'ok', 'backend': 'cryolo', 'runtime_format': 'onnx',
                'runtime': 'onnxruntime-cpu', 'frozen': bool(getattr(sys, 'frozen', False)),
                'offline_guard': os.environ.get('SARU_VERIFY_OFFLINE') == '1',
                'external_python_required': False, 'actual_onnx_inference': True,
                'failure_diagnostics_preserved': True,
                'positive_contours': len(result), 'detections': details['cryolo_detections'],
                'seconds': time.monotonic() - started, 'details': details,
                'note': 'Synthetic Conv inference through the production worker, decoder and positive contour path. No trained-model accuracy claim.'}
    finally:
        model.close()


def verify_saved_preferences(directory: Path) -> dict:
    """Run normal settings loading with a relocated, absent legacy H5 path.

    The installed ONNX must be found by the production default search. No
    checkpoint argument, resolver override or adjacent ONNX is supplied here.
    Only the user-state directory is isolated to avoid altering real settings.
    """
    import app
    from cryolo_onnx import resolve_onnx_checkpoint, STANDARD_H5_SHA256
    from detector_selection import save_detector_settings

    with tempfile.TemporaryDirectory(prefix='saru-upgrade-preferences-') as temporary:
        workspace = Path(temporary)
        state = workspace / 'user-state'
        stale_h5 = workspace / 'removed-old-download' / 'cryolo-capsids.h5'
        assert not stale_h5.exists() and not stale_h5.with_suffix('.onnx').exists()
        installed_onnx = (directory / 'cryolo-capsids.onnx').resolve()
        effective = resolve_onnx_checkpoint(stale_h5)
        assert effective is not None and effective.resolve() == installed_onnx, effective
        settings = state / 'detector-settings.json'
        save_detector_settings(settings, {
            'detection_profile': 'crYOLO', 'cryolo_checkpoint': str(stale_h5),
            'cryolo_config': str(stale_h5.with_name('cryolo-config.json')),
            'cryolo_threshold': .3, 'cryolo_timeout': 90, 'device': 'cpu',
            'min_capsid_area': 3500,
        })
        original_state_directory = app.user_state_directory
        logger = logging.getLogger()
        original_level = logger.level
        original_saru_handlers = [handler for handler in logger.handlers
                                  if getattr(handler, '_saru_file', False)
                                  or getattr(handler, '_saru_console', False)]
        for handler in original_saru_handlers:
            logger.removeHandler(handler)
        started = time.monotonic()
        try:
            app.user_state_directory = lambda: state
            output = workspace / 'recognition'
            report_path = workspace / 'recognition-report.json'
            status = app.main(['--segment-image', str(directory / 'synthetic.png'),
                               '--output', str(output), '--report', str(report_path)])
            operation = json.loads(report_path.read_text(encoding='utf-8'))
            assert status == 0 and operation['status'] == 'ok', operation
            assert operation['detection_profile'] == 'crYOLO', operation
            assert operation['candidates'] >= 1 and operation['grid_points'] > 0, operation
            review = json.loads((output / 'review.json').read_text(encoding='utf-8'))
            config = review['settings']['config']
            recognition = review['settings']['recognition']
            assert config['cryolo_python'] is None, config
            assert recognition['cryolo_detections'] == 1, recognition
            assert recognition['appearance_model']['status'] == 'loaded', recognition
            assert 'onnxruntime-cpu' in json.dumps(recognition), recognition
        finally:
            app.user_state_directory = original_state_directory
            # Windows cannot remove the temporary state directory while the
            # application log is still open. Preserve the caller's handlers.
            for handler in list(logger.handlers):
                if getattr(handler, '_saru_file', False) or getattr(handler, '_saru_console', False):
                    logger.removeHandler(handler)
                    handler.close()
            for handler in original_saru_handlers:
                logger.addHandler(handler)
            logger.setLevel(original_level)
        return {'status': 'ok', 'frozen': bool(getattr(sys, 'frozen', False)),
                'offline_guard': os.environ.get('SARU_VERIFY_OFFLINE') == '1',
                'saved_preference_loaded': True, 'legacy_h5_missing': True,
                'adjacent_onnx_missing': True, 'installed_onnx_found': True,
                'expected_source_sha256': STANDARD_H5_SHA256,
                'external_python_required': False, 'actual_onnx_inference': True,
                'positive_contours': operation['candidates'],
                'grid_points': operation['grid_points'],
                'seconds': time.monotonic() - started, 'details': recognition,
                'note': 'Synthetic migration contract check through normal saved settings and actual offline prediction. No private weights or trained-model accuracy claim.'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--make-fixture', action='store_true')
    parser.add_argument('--saved-preferences', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.make_fixture:
        make_fixture(args.fixture)
        return 0
    if args.report is None:
        parser.error('--report is required for verification')
    try:
        report = (verify_saved_preferences(args.fixture) if args.saved_preferences
                  else verify(args.fixture))
    except Exception as error:
        report = {'status': 'error', 'error': str(error), 'error_type': type(error).__name__,
                  'frozen': bool(getattr(sys, 'frozen', False))}
    from app import APP_VERSION
    report['version'] = APP_VERSION
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
