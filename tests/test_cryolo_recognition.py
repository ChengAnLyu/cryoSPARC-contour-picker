"""crYOLO protocol/coordinate tests; synthetic child runtime, never model claims."""
from dataclasses import replace
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition import RecognitionConfig, RecognitionError, Cancelled
from cryolo_recognition import CryoloBox, CryoloRecognizer, read_cbox
from runtime_management import RuntimeSetupError, RuntimeSetupCancelled


HEADER = '''data_global
_cbox_format_version 1.0

data_cryolo
loop_
_CoordinateX #1
_CoordinateY #2
_CoordinateZ #3
_Width #4
_Height #5
_Depth #6
_EstWidth #7
_EstHeight #8
_Confidence #9
_NumBoxes #10
'''


def cbox(tmp_path, body):
    path = tmp_path / "screen.cbox"
    path.write_text(body)
    return path


def test_cbox_bottom_left_flip_and_estimated_extent(tmp_path):
    # Export box center (40, 70) bottom-left becomes (40, 30) top-left.
    path = cbox(tmp_path, HEADER + '30 50 <NA> 20 40 <NA> 12 16 .8 <NA>\n')
    assert read_cbox(path, (100, 200), .3) == [CryoloBox(34, 22, 46, 38, .8)]


def test_cbox_columns_are_named_and_other_blocks_are_ignored(tmp_path):
    text = '''data_global
_cbox_format_version 1.0

data_cryolo
loop_
_Confidence #1
_Height #2
_CoordinateY #3
_CoordinateX #4
_Width #5
.9 10 70 20 30

data_cryolo_include
loop_
_slice_index #1
0
'''
    assert read_cbox(cbox(tmp_path, text), (100, 200), .3) == [CryoloBox(20, 20, 50, 30, .9)]


def test_cbox_threshold_applied_and_order_is_objectness(tmp_path):
    path = cbox(tmp_path, HEADER + '\n'.join([
        '10 60 <NA> 10 10 <NA> <NA> <NA> .31 <NA>',
        '30 60 <NA> 10 10 <NA> <NA> <NA> .29 <NA>',
        '50 60 <NA> 10 10 <NA> <NA> <NA> .99 <NA>',
    ]))
    assert [b.confidence for b in read_cbox(path, (100, 200), .3)] == [.99, .31]


def test_present_empty_and_header_only_cbox_are_valid_blank(tmp_path):
    assert read_cbox(cbox(tmp_path, ''), (100, 200), .3) == []
    assert read_cbox(cbox(tmp_path, HEADER), (100, 200), .3) == []


@pytest.mark.parametrize('body', [
    'garbage',
    HEADER + '1 2 3',
    HEADER + 'nan 10 <NA> 20 20 <NA> <NA> <NA> .8 <NA>',
    HEADER + '10 10 <NA> -20 20 <NA> <NA> <NA> .8 <NA>',
    HEADER + '10 10 <NA> 20 20 <NA> <NA> <NA> 1.8 <NA>',
    HEADER + '1000 10 <NA> 20 20 <NA> <NA> <NA> .8 <NA>',
    HEADER.replace('_Confidence', '_Width'),
])
def test_malformed_output_is_not_a_blank(body, tmp_path):
    with pytest.raises(RecognitionError, match='Malformed'):
        read_cbox(cbox(tmp_path, body), (100, 200), .3)


def test_missing_output_is_not_a_blank(tmp_path):
    with pytest.raises(RecognitionError, match='did not produce'):
        read_cbox(tmp_path / 'missing.cbox', (100, 200), .3)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    module = tmp_path / 'cryolo'
    module.mkdir()
    (module / '__init__.py').write_text('')
    weights = tmp_path / 'weights.h5'
    weights.write_bytes(b'fixture, not a real model')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model': {'architecture': 'PhosaurusNet'}}))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    # The synthetic module lives only on this test's PYTHONPATH. Production
    # sanitization is tested independently; this seam proves the adapter uses
    # the external-process context and keeps subprocess execution real.
    @contextmanager
    def synthetic_external_environment(env):
        child = dict(env)
        child['SARU_TEST_EXTERNAL_CONTEXT'] = '1'
        yield child
    monkeypatch.setattr('cryolo_recognition.external_process_environment', synthetic_external_environment)
    cfg = RecognitionConfig(str(weights), backend='cryolo', cryolo_config=str(config),
                            cryolo_python=sys.executable, device='cpu', cryolo_timeout=5,
                            automatic_profile='capsid', min_capsid_area_px=0)
    return tmp_path, module / 'predict.py', cfg


def blank_program(output=True):
    return '''import sys, os
from pathlib import Path
from PIL import Image
args=sys.argv[1:]
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
assert os.environ['SARU_TEST_EXTERNAL_CONTEXT']=='1'
assert '-g' not in args
assert '--write_empty' in args
image = Image.open(Path(args[args.index('-i')+1])/'screen.png')
assert image.mode == 'L'
assert image.size == (120, 80)
assert image.getpixel((0, 0)) == 222
assert image.getpixel((0, 79)) == 11
''' + ('''output=Path(args[args.index('-o')+1])/'CBOX'
output.mkdir(parents=True)
(output/'screen.cbox').write_text('')
''' if output else '')


def test_actual_subprocess_protocol_blank_and_cache(runtime):
    tmp_path, script, config = runtime
    script.write_text(blank_program())
    image = np.zeros((80, 120, 3), np.uint8)
    image[0] = 222
    image[-1] = 11
    model = CryoloRecognizer(config)
    assert model.automatic(image) == []
    assert model.last_detection['cryolo_detections'] == 0
    assert model.last_detection['backend'] == 'cryolo'
    # A repeat uses cached detector results, proven by breaking the child.
    script.write_text('raise RuntimeError("should not run twice")')
    assert model.automatic(image) == []
    assert model._process is None
    model.close()
    assert model._boxes is None


def test_missing_cbox_from_successful_process_raises(runtime):
    _, script, config = runtime
    script.write_text('pass')
    with pytest.raises(RecognitionError, match='did not produce'):
        CryoloRecognizer(config).automatic(np.zeros((80, 120, 3), np.uint8))


def test_nonzero_child_exit_preserves_actionable_error(runtime):
    _, script, config = runtime
    script.write_text('import sys\nprint("bad architecture")\nsys.exit(7)')
    with pytest.raises(RecognitionError, match='exit 7.*bad architecture'):
        # DOTALL is supplied by the inline flag below via outer pytest wrapper.
        try:
            CryoloRecognizer(config).automatic(np.zeros((80, 120, 3), np.uint8))
        except RecognitionError as exc:
            raise RecognitionError(str(exc).replace('\n', ' '))


def test_timeout_reaps_child_and_does_not_cache(runtime):
    _, script, config = runtime
    script.write_text('import time\ntime.sleep(20)')
    model = CryoloRecognizer(replace(config, cryolo_timeout=.15))
    started = time.monotonic()
    with pytest.raises(RecognitionError, match='timed out'):
        model.automatic(np.zeros((80, 120, 3), np.uint8))
    assert time.monotonic() - started < 5
    assert model._process is None and model._boxes is None


def test_cancellation_reaps_child_and_does_not_cache(runtime):
    _, script, config = runtime
    script.write_text('import time\ntime.sleep(20)')
    model = CryoloRecognizer(config)
    started = time.monotonic()
    with pytest.raises(Cancelled):
        model.automatic(np.zeros((80, 120, 3), np.uint8), cancel=lambda: time.monotonic()-started > .15)
    assert model._process is None and model._boxes is None


def test_prompt_selects_boxes_and_ignores_automatic_min_area(runtime):
    _, _, config = runtime
    model = CryoloRecognizer(replace(config, min_capsid_area_px=3500))
    boxes = [CryoloBox(10, 10, 40, 40, .8), CryoloBox(60, 10, 100, 40, .9)]
    image = np.zeros((80, 120, 3), np.uint8)
    with patch.object(model, '_predict', return_value=(boxes, False)), patch.object(model, '_refine', return_value=([], {}, {}, {}, {})) as refine:
        model.prompt(image, box=(0, 0, 110, 50), points=[(20, 20), (80, 20)], labels=[1, 0])
        assert refine.call_args.args[1] == boxes[:1]
        assert refine.call_args.kwargs['prompt'] is True
    assert model.last_detection['mode'] == 'prompt_selection'


def test_detector_confidence_never_becomes_mask_iou(runtime):
    _, _, config = runtime
    model = CryoloRecognizer(config)
    image = np.zeros((80, 120, 3), np.uint8)
    mask = np.zeros((80, 120), bool)
    mask[15:35, 15:35] = True
    record = {'segmentation': mask, 'source': 'fixture', 'predicted_iou': .999,
              'stability_score': .999, 'capsid_evidence': {}}
    with patch('cryolo_recognition.filter_capsid_proposals', return_value=[record]), \
         patch('cryolo_recognition.detect_carbon_edges', return_value=None), \
         patch('cryolo_recognition.guard_proposals', side_effect=lambda context, raw, **kwargs: (raw, {})):
        result, evidence, _, _, _ = model._refine(image, [CryoloBox(10, 10, 40, 40, .8)], None)
    assert len(result) == 1
    assert result[0].predicted_iou is None and result[0].stability_score is None
    assert evidence[result[0].id]['cryolo_confidence'] == .8
    assert not evidence[result[0].id]['confidence_is_mask_iou']


def test_uniform_detected_box_is_not_returned_as_segmentation(runtime):
    _, _, config = runtime
    model = CryoloRecognizer(config)
    result, _, _, _, _ = model._refine(np.full((80, 120, 3), 100, np.uint8),
                                     [CryoloBox(10, 10, 70, 60, .99)], None)
    assert result == []


def test_missing_selected_profile_never_silently_falls_back(runtime):
    _, _, config = runtime
    model = CryoloRecognizer(replace(config, automatic_profile='capsid-learned'))
    with pytest.raises(RecognitionError, match='appearance profile'):
        model.refine_detections(np.zeros((80, 120, 3), np.uint8), [])


def test_automatic_cache_restores_its_report_after_prompt(runtime):
    _, script, config = runtime
    script.write_text(blank_program())
    image = np.zeros((80, 120, 3), np.uint8)
    image[0] = 222
    image[-1] = 11
    model = CryoloRecognizer(config)
    assert model.automatic(image) == []
    assert model.prompt(image, box=(1, 1, 60, 60)) == []
    assert model.last_detection['mode'] == 'prompt_selection'
    assert model.automatic(image) == []
    assert 'mode' not in model.last_detection
    assert model.last_detection['detections_cached'] is True


def test_relative_runtime_is_resolved_before_changing_child_directory(runtime, monkeypatch):
    tmp_path, script, config = runtime
    script.write_text(blank_program())
    # Windows runners can put the interpreter and temp files on different
    # drives, between which a relative path cannot exist.
    monkeypatch.chdir(Path(sys.executable).parent)
    relative_python = os.path.join('.', Path(sys.executable).name)
    model = CryoloRecognizer(replace(config, cryolo_python=relative_python))
    assert Path(model._paths()[0]).is_absolute()
    image = np.zeros((80, 120, 3), np.uint8)
    image[0] = 222
    image[-1] = 11
    assert model.automatic(image) == []


def test_runtime_setup_error_never_requests_interpreter_selection(runtime, monkeypatch):
    _, _, config = runtime
    monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    model = CryoloRecognizer(replace(config, cryolo_python=None))
    with patch('cryolo_recognition.ensure_cryolo_runtime', side_effect=RuntimeSetupError('Run SARU Picker Setup to repair it.')):
        with pytest.raises(RecognitionError, match='Run SARU Picker Setup') as error:
            model._paths()
    assert 'Select the Python' not in str(error.value)


def test_implicit_runtime_is_prepared_automatically(runtime, monkeypatch):
    _, _, config = runtime
    monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    model = CryoloRecognizer(replace(config, cryolo_python=None))
    cancel = lambda: False
    with patch('cryolo_recognition.ensure_cryolo_runtime', return_value=Path(sys.executable)) as setup:
        executable, _, _ = model._paths(cancel)
    setup.assert_called_once_with(cancel=cancel)
    assert executable == os.path.abspath(sys.executable)


@pytest.mark.parametrize('use_environment_override', [False, True])
def test_explicit_runtime_override_bypasses_managed_setup(runtime, monkeypatch, use_environment_override):
    _, _, config = runtime
    if use_environment_override:
        monkeypatch.setenv('SARU_CRYOLO_PYTHON', sys.executable)
        config = replace(config, cryolo_python=None)
    else:
        monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    with patch('cryolo_recognition.ensure_cryolo_runtime', side_effect=AssertionError('Must not install an overridden runtime')):
        assert CryoloRecognizer(config)._paths()[0] == os.path.abspath(sys.executable)


def test_setup_cancellation_uses_standard_cancelled_exception(runtime, monkeypatch):
    _, _, config = runtime
    monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    model = CryoloRecognizer(replace(config, cryolo_python=None))
    with patch('cryolo_recognition.ensure_cryolo_runtime', side_effect=RuntimeSetupCancelled('cancelled by user')):
        with pytest.raises(Cancelled, match='cancelled by user'):
            model._paths()


def test_missing_model_does_not_start_runtime_installation(runtime, monkeypatch):
    _, _, config = runtime
    monkeypatch.delenv('SARU_CRYOLO_PYTHON', raising=False)
    Path(config.checkpoint).unlink()
    model = CryoloRecognizer(replace(config, cryolo_python=None))
    with patch('cryolo_recognition.ensure_cryolo_runtime', side_effect=AssertionError('Unnecessary install')):
        with pytest.raises(RecognitionError, match='weights not found'):
            model._paths()
