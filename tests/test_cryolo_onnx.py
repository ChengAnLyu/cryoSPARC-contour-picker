"""Offline model identity, decoding, worker isolation and bounded execution."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryolo_onnx import (STANDARD_H5_SHA256, decode_detections, file_sha256,
                         onnx_worker_main, preprocess_image, resolve_onnx_checkpoint,
                         validate_metadata)
from cryolo_recognition import CryoloRecognizer, read_onnx_detections
from recognition import Cancelled, RecognitionConfig, RecognitionError


def metadata():
    return dict(format='saru-cryolo-onnx-v1', architecture='PhosaurusNet',
                source_h5_sha256=STANDARD_H5_SHA256, input_name='image', output_name='head',
                input_size=[320, 320], input_layout='NHWC', output_layout='NHWAC',
                anchors=[2., 2.], normalization='STANDARD', labels=['particle'],
                filament_model=False, opset=13, nms_threshold=.3,
                estimated_size_rounding='truncate', extraction_box_size=1,
                preprocessing='pil-grayscale-bilinear-standard-v1')


def test_only_standard_h5_migrates_and_explicit_onnx_paths_are_preserved(tmp_path):
    standard = tmp_path / 'cryolo-capsids.h5'
    exported = standard.with_suffix('.onnx')
    assert resolve_onnx_checkpoint(standard) is None
    exported.write_bytes(b'fixture')
    assert resolve_onnx_checkpoint(standard) == exported
    custom = tmp_path / 'custom.h5'
    custom.with_suffix('.onnx').write_bytes(b'fixture')
    assert resolve_onnx_checkpoint(custom) is None
    assert resolve_onnx_checkpoint(tmp_path / 'missing.onnx') == tmp_path / 'missing.onnx'


def test_metadata_validates_source_identity_in_worker(tmp_path):
    source = tmp_path / 'weights.h5'
    source.write_bytes(b'original-trained-weights')
    info = metadata()
    info['source_h5_sha256'] = file_sha256(source)
    assert validate_metadata(json.dumps(info), source_h5=source) == info
    with pytest.raises(ValueError, match='saved standard'):
        validate_metadata(json.dumps(info), expected_source_sha256=STANDARD_H5_SHA256)
    source.write_bytes(b'different-weights')
    with pytest.raises(ValueError, match='selected H5'):
        validate_metadata(json.dumps(info), source_h5=source)


@pytest.mark.parametrize('key,value', [('anchors', [0, 2]), ('anchors', [111, 107]),
    ('normalization', 'GMM'), ('input_size', [640, 640]), ('filament_model', True),
    ('source_h5_sha256', 'not-a-hash'), ('input_name', ''), ('format', 'unknown')])
def test_unsupported_model_metadata_is_rejected(key, value):
    info = metadata(); info[key] = value
    with pytest.raises(ValueError):
        validate_metadata(json.dumps(info))


def test_preprocess_constant_and_rectangular_grayscale_without_flip():
    constant = preprocess_image(np.full((40, 80, 3), 120, np.uint8))
    assert constant.shape == (1, 320, 320, 1) and constant.dtype == np.float32
    assert not constant.any()
    ramp = np.repeat(np.linspace(0, 255, 40).astype(np.uint8)[:, None, None], 80, axis=1)
    ramp = np.repeat(ramp, 3, axis=2)
    normalized = preprocess_image(ramp)
    assert normalized[0, 0, 0, 0] < normalized[0, -1, 0, 0]
    assert abs(float(normalized.mean())) < 1e-6


def test_decoder_suppression_threshold_and_pixel_extent():
    raw = np.zeros((1, 20, 20, 1, 6), np.float32)
    raw[..., 4] = -20
    raw[0, 8, 10, 0, 4] = np.log(.9/.1)
    raw[0, 8, 9, 0, 4] = np.log(.8/.2)  # IoU1/3: suppressed by the stronger box.
    raw[0, 2, 2, 0, 4] = np.log(.2/.8)
    boxes = decode_detections(raw, (320, 320), [2., 2.], .3)
    assert len(boxes) == 1
    assert [boxes[0][name] for name in ('x0', 'y0', 'x1', 'y1')] == pytest.approx([152, 120, 184, 152])
    assert boxes[0]['confidence'] == pytest.approx(.9)
    assert len(decode_detections(raw, (320, 320), [2., 2.], .1)) == 2
    raw[0, 0, 0, 0, :2] = -100
    raw[0, 0, 0, 0, 4] = 20
    assert len(decode_detections(raw, (320, 320), [2., 2.], .3)) == 1


def test_invalid_worker_protocol_is_not_a_blank(tmp_path):
    path = tmp_path / 'boxes.json'
    for data in ({}, {'format':'saru-cryolo-detections-v1', 'runtime_format':'onnx',
                      'image_shape':[80, 120], 'detections':[{'x0':float('nan')}]}):
        path.write_text(json.dumps(data))
        with pytest.raises(RecognitionError, match='Invalid offline'):
            read_onnx_detections(path, (80,120))


def test_windowed_worker_retains_error_when_standard_streams_are_none(tmp_path):
    log = tmp_path / 'worker.log'; output = tmp_path / 'result.json'
    with patch('cryolo_onnx.run_prediction', side_effect=ValueError('invalid model fixture')):
        with patch.object(sys, 'stdout', None), patch.object(sys, 'stderr', None):
            code = onnx_worker_main(['--model', str(tmp_path/'x.onnx'), '--input', str(tmp_path/'x.png'),
                                     '--output', str(output), '--log', str(log)])
    assert code == 1 and not output.exists()
    assert 'invalid model fixture' in log.read_text()


@pytest.fixture
def offline_config(tmp_path, monkeypatch):
    model = tmp_path / 'cryolo-capsids.onnx'; model.write_bytes(b'fixture-model')
    monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    monkeypatch.setattr('cryolo_recognition._prediction_log_directory', lambda: tmp_path/'diagnostics')
    return RecognitionConfig(str(model), backend='cryolo', device='cpu', automatic_profile='capsid',
                             cryolo_config='/stale/legacy/config.json', cryolo_timeout=5)


def test_offline_adapter_ignores_legacy_config_and_setup(offline_config):
    def start(command, **kwargs):
        assert command[1:3] == ['-m', 'cryolo_onnx']
        assert '--log' in command and '--model' in command
        assert 'cryolo.predict' not in command
        Path(command[command.index('--output')+1]).write_text(json.dumps({
            'format':'saru-cryolo-detections-v1', 'runtime_format':'onnx',
            'image_shape':[80,120], 'detections':[]}))
        class Completed:
            returncode = 0
            def poll(self): return 0
        return Completed()
    with patch('runtime_management.ensure_cryolo_runtime', side_effect=AssertionError('No setup')):
        with patch('cryolo_recognition.subprocess.Popen', side_effect=start):
            model = CryoloRecognizer(offline_config)
            assert model.automatic(np.zeros((80,120,3), np.uint8)) == []
    assert model.last_detection['runtime_format'] == 'onnx'


def test_explicit_legacy_runtime_wins_over_standard_h5_migration(offline_config):
    h5 = Path(offline_config.checkpoint).with_suffix('.h5'); h5.write_bytes(b'legacy')
    model = CryoloRecognizer(replace(offline_config, checkpoint=str(h5), cryolo_python=sys.executable))
    assert model._selected_onnx_checkpoint() is None


@pytest.mark.parametrize('cancel', [False, True])
def test_offline_child_timeout_and_cancel_reap_process(offline_config, cancel):
    original = subprocess.Popen
    def slow_child(command, **kwargs):
        assert command[1:3] == ['-m', 'cryolo_onnx']
        return original([sys.executable, '-c', 'import time;print("offline worker started",flush=True);time.sleep(20)'], **kwargs)
    model = CryoloRecognizer(replace(offline_config, cryolo_timeout=.3))
    started = time.monotonic()
    with patch('cryolo_recognition.subprocess.Popen', side_effect=slow_child):
        with pytest.raises(Cancelled if cancel else RecognitionError):
            model.automatic(np.zeros((80,120,3), np.uint8),
                            cancel=(lambda: time.monotonic()-started > .15) if cancel else None)
    assert time.monotonic()-started < 5
    assert model._process is None and model._boxes is None


def test_standalone_worker_dispatch_precedes_gui_imports(tmp_path):
    from build_standalone import build
    script = build(tmp_path/'single-file.py')
    result = subprocess.run([sys.executable, str(script), '--cryolo-onnx-worker', '--help'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'Isolated offline crYOLO prediction worker' in result.stdout
    assert '--sam-checkpoint' not in result.stdout
