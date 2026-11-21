"""One four-profile application, with backend-specific weights and safe Apply."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

import app
import main_cryolo
import screen_app
from detector_selection import DETECTION_PROFILES, save_detector_settings
from recognition import RecognitionConfig


class Variable:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def model_fields(owner, choice="capsid"):
    owner.args = SimpleNamespace(backend="sam", automatic_profile="capsid", cryolo_timeout=180,
                                 cryolo_python=None, detector_settings_path=None)
    owner.tk = SimpleNamespace(TclError=RuntimeError)
    for name, value in {"detection_profile": choice, "sam_checkpoint": "sam.pth",
                        "cryolo_checkpoint": "cryolo.h5", "cryolo_config": "cryolo.json",
                        "cryolo_threshold": ".3", "model_type": "vit_b", "device": "cpu",
                        "grid_side": 8, "min_capsid_area": "3500"}.items():
        setattr(owner, name, Variable(value))
    owner.checkpoint = owner.sam_checkpoint
    owner.automatic_profile = owner.detection_profile
    owner.status = Variable("")
    return owner


def screen_fixture():
    picker = model_fields(screen_app.ScreenPickerApp.__new__(screen_app.ScreenPickerApp))
    picker.runner = SimpleNamespace(busy=False)
    picker.transition = picker.drawing = False
    picker.auto_remaining = 0
    picker._close_input = Mock()
    picker.candidates, picker.alternatives = [object()], [object()]
    picker.alternative_index = 1
    picker.frame_id = 4
    picker.overlay = Mock()
    picker.picked_points = {(11, 12), (21, 22)}
    picker.prompt_button = Mock()
    picker.recognizer = picker.recognizer_config = None
    return picker


def review_fixture():
    picker = model_fields(app.PickerApp.__new__(app.PickerApp))
    picker._applied_model_config = screen_app.detector_config_from_fields(picker)
    picker.runner = SimpleNamespace(busy=False)
    picker.playback_pending = picker.capture_pending = False
    picker.batch_remaining = 0
    picker.rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    picker.candidates, picker.alternatives = [object()], [object()]
    picker.selected, picker.picked = {"proposal"}, {"already-picked"}
    picker.picked_points = {(1, 2)}
    picker.refresh_table = Mock()
    picker.render = Mock()
    picker._error = Mock()
    picker.prompt_button = Mock()
    return picker


def test_primary_and_compatibility_launchers_restore_sam_capsid_default():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        for launch in (app.main, main_cryolo.main):
            assert launch([]) == 0
            args = run.call_args.args[0]
            assert (args.detection_profile, args.backend, args.automatic_profile) == ("capsid", "sam", "capsid")
            assert args.min_capsid_area == 3500
        assert app.APP_VERSION == "3.0.0"


@pytest.mark.parametrize("choice,backend,profile", [
    ("capsid", "sam", "capsid"), ("capsid-learned", "sam", "capsid-learned"),
    ("generic", "sam", "generic"), ("crYOLO", "cryolo", "capsid-learned")])
def test_four_profiles_route_identically_in_cli_live_and_review(choice, backend, profile):
    args = app.build_parser().parse_args(["--detection-profile", choice])
    live = screen_fixture()
    live.detection_profile.set(choice)
    config = live._model_config()
    assert (args.backend, args.automatic_profile) == (backend, profile)
    assert (config.backend, config.automatic_profile) == (backend, profile)
    assert config.checkpoint == ("cryolo.h5" if backend == "cryolo" else "sam.pth")
    review = review_fixture()
    review.detection_profile.set(choice)
    review._apply_model_settings()
    review._error.assert_not_called()
    assert review._model_config() == config


def test_explicit_backend_cli_remains_supported_without_silent_conflicts():
    parser = app.build_parser()
    assert parser.parse_args(["--backend", "cryolo"]).detection_profile == "crYOLO"
    assert parser.parse_args(["--automatic-profile", "cryolo"]).detection_profile == "crYOLO"
    assert parser.parse_args(["--backend", "cryolo", "--automatic-profile", "capsid-learned"]).detection_profile == "crYOLO"
    assert parser.parse_args(["--backend", "sam"]).automatic_profile == "capsid"
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "sam", "--detection-profile", "crYOLO"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "cryolo", "--detection-profile", "generic"])


def test_unified_saved_choice_and_separate_paths_yield_to_explicit_cli():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        save_detector_settings(Path(directory) / "detector-settings.json", {
            "detection_profile": "crYOLO", "sam_checkpoint": "chosen-sam.pth",
            "cryolo_checkpoint": "chosen-cryolo.h5", "cryolo_threshold": .18})
        app.main([])
        saved = run.call_args.args[0]
        assert saved.backend == "cryolo"
        assert saved.sam_checkpoint == "chosen-sam.pth"
        assert saved.cryolo_checkpoint == "chosen-cryolo.h5"
        assert saved.cryolo_python is None
        app.main(["--backend", "sam"])
        assert run.call_args.args[0].detection_profile == "capsid"
        app.main(["--automatic-profile", "generic"])
        assert run.call_args.args[0].detection_profile == "generic"
        app.main(["--cryolo-threshold", ".42"])
        assert run.call_args.args[0].cryolo_threshold == .42


def test_old_cryolo_interpreter_setting_is_not_restored():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        (Path(directory) / "cryolo-settings.json").write_text('{"cryolo_python":"obsolete-python"}')
        app.main([])
        assert run.call_args.args[0].cryolo_python is None
        assert run.call_args.args[0].detection_profile == "capsid"


def test_invalid_unified_settings_fall_back_to_original_capsid():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        (Path(directory) / "detector-settings.json").write_text('{"cryolo_threshold": -9}')
        app.main([])
        assert run.call_args.args[0].detection_profile == "capsid"


def test_live_apply_retires_proposals_preserves_picks_and_separate_model_paths():
    picker = screen_fixture()
    previous = picker._model_config()
    picker.detection_profile.set("crYOLO")
    assert picker._apply_model_settings(previous)
    assert picker.candidates == picker.alternatives == []
    assert picker.picked_points == {(11, 12), (21, 22)}
    assert picker.sam_checkpoint.get() == "sam.pth"
    assert picker._model_config().checkpoint == "cryolo.h5"
    assert picker.args.backend == "cryolo"
    picker.detection_profile.set("generic")
    picker._apply_model_settings(picker._model_config().__class__("cryolo.h5", backend="cryolo", automatic_profile="capsid-learned"))
    assert picker._model_config().checkpoint == "sam.pth"
    assert picker._model_config().automatic_profile == "generic"


@pytest.mark.parametrize("field,value", [("busy", True), ("transition", True), ("auto_remaining", -1),
                                         ("auto_running", True), ("drawing", True)])
def test_live_apply_is_blocked_during_workers_and_between_auto_jobs(field, value):
    picker = screen_fixture()
    previous = picker._model_config()
    picker.detection_profile.set("crYOLO")
    if field == "busy":
        picker.runner.busy = value
    else:
        setattr(picker, field, value)
    with pytest.raises(RuntimeError, match="active operation"):
        picker._apply_model_settings(previous)
    assert len(picker.candidates) == 1
    assert picker.args.backend == "sam"


def test_review_pending_change_blocks_old_proposal_selection_until_apply():
    picker = review_fixture()
    previous = picker._model_config()
    picker.detection_profile.set("crYOLO")
    assert picker._model_config() is previous
    assert not picker._editable()
    with pytest.raises(RuntimeError, match="Apply detection settings"):
        picker._ensure_idle()
    assert picker._apply_model_settings()
    assert picker._model_config().backend == "cryolo"
    assert picker.candidates == picker.alternatives == []
    assert picker.selected == set()
    assert picker.picked_points == {(1, 2)}
    assert picker.picked == {"already-picked"}


@pytest.mark.parametrize("field,value", [("busy", True), ("batch_remaining", 2), ("auto_running", True)])
def test_review_rejects_profile_change_and_apply_while_running(field, value):
    picker = review_fixture()
    if field == "busy":
        picker.runner.busy = value
    else:
        setattr(picker, field, value)
    picker.detection_profile.set("crYOLO")
    picker._profile_draft_changed()
    assert picker.detection_profile.get() == "capsid"
    picker.detection_profile.set("crYOLO")
    assert not picker._apply_model_settings()
    assert picker._model_config().backend == "sam"
    assert len(picker.candidates) == 1
    picker._error.assert_called_once()


def test_review_cross_backend_weights_are_rejected_without_changing_applied_config():
    picker = review_fixture()
    before = picker._model_config()
    picker.sam_checkpoint.set("wrong.h5")
    assert not picker._apply_model_settings()
    picker._error.assert_called_once()
    assert picker._model_config() is before
    assert not picker._editable()


def test_screen_recognizer_switch_never_reuses_other_backend_instance():
    picker = screen_fixture()
    sam_config = picker._model_config()
    with patch.object(screen_app, "SamRecognizer") as sam, patch.object(screen_app, "create_recognizer") as factory:
        picker._get_recognizer(sam_config)
        sam.assert_called_once_with(sam_config)
        picker.detection_profile.set("crYOLO")
        cryolo_config = picker._model_config()
        picker._get_recognizer(cryolo_config)
        sam.return_value.close.assert_called_once()
        factory.assert_called_once_with(cryolo_config)


def test_live_settings_blocks_queued_clicks_on_previous_proposals():
    picker = screen_fixture()
    picker.mode = "semi-auto"
    picker.disabled = Variable(False)
    picker.advanced = object()
    picker._start_model_picks = Mock()
    picker._error = Mock()
    with patch.object(screen_app, "screen_mask_at") as hit_test:
        picker._semi_click(10, 20)
    hit_test.assert_not_called()
    picker._start_model_picks.assert_not_called()


def test_cryolo_smoke_failure_is_error_and_never_falls_back_to_sam():
    config = RecognitionConfig("model.h5", backend="cryolo")
    detector = Mock()
    detector.automatic.side_effect = RuntimeError("crYOLO runtime is missing")
    with patch.object(app, "create_recognizer", return_value=detector), \
         patch.object(app, "SamRecognizer") as sam, patch.object(app, "_smoke_image_processing"):
        report = app.run_cryolo_smoke_test(Path("test.log"), config)
    assert report["status"] == "error"
    detector.close.assert_called_once()
    sam.assert_not_called()


def test_headless_cryolo_routes_to_its_own_checkpoint_and_exports_backend():
    args = app.build_parser().parse_args(["--detection-profile", "crYOLO", "--segment-image", "input.png",
                                         "--sam-checkpoint", "sam.pth", "--cryolo-checkpoint", "model.h5"])
    detector = Mock()
    detector.automatic.return_value = []
    detector.last_detection = {"backend": "cryolo"}
    with tempfile.TemporaryDirectory() as directory:
        args.output = Path(directory)
        args.report = Path(directory) / "report.json"
        with patch.object(app, "setup_logging", return_value=Path(directory) / "test.log"), \
             patch.object(app, "load_rgb_file", return_value=np.zeros((64, 64, 3), dtype=np.uint8)), \
             patch.object(app, "create_recognizer", return_value=detector) as factory, \
             patch.object(app, "SamRecognizer") as sam, patch.object(app, "export_review"):
            assert app._run_application(args, console_available=True) == 0
        config = factory.call_args.args[0]
        assert config.backend == "cryolo" and Path(config.checkpoint).name == "model.h5"
        assert config.automatic_profile == "capsid-learned"
        assert json.loads(args.report.read_text())["detection_profile"] == "crYOLO"
        sam.assert_not_called()
