"""Parallel entry point keeps the screen workflow and selects only crYOLO."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
from unittest.mock import Mock, patch

import numpy as np

import app
import main_cryolo
import screen_app
from recognition import RecognitionConfig


def test_parallel_launcher_defaults_preserve_existing_sam_launcher():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        assert main_cryolo.main([]) == 0
        options = run.call_args.args[0]
        assert (options.backend, options.automatic_profile, options.min_capsid_area) == (
            "cryolo", "capsid-learned", 3500)
        assert options.review is False
        app.main([])
        assert run.call_args.args[0].backend == "sam"
        assert run.call_args.args[0].automatic_profile == "capsid"


def test_parallel_screen_launch_has_distinct_taskbar_identity():
    root = Mock()
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "setup_logging", return_value=Path(directory) / "test.log"), \
         patch.dict(sys.modules, {"tkinter": SimpleNamespace(Tk=Mock(return_value=root))}), \
         patch.object(app, "enable_dpi_awareness"), \
         patch.object(app, "configure_application_identity") as identity, \
         patch.object(app, "ScreenPickerApp") as screen, patch.object(app, "PickerApp") as review:
        assert main_cryolo.main([]) == 0
        identity.assert_called_once_with("SARU.Picker.crYOLO")
        screen.assert_called_once()
        review.assert_not_called()
        assert screen.call_args.args[1].backend == "cryolo"


def test_saved_settings_apply_only_to_cryolo_and_cli_wins():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        path = Path(directory) / "cryolo-settings.json"
        path.write_text(json.dumps({"cryolo_python": "chosen-python", "cryolo_threshold": .18,
                                    "min_capsid_area": 4100, "backend": "sam"}))
        main_cryolo.main(["--cryolo-threshold", ".42"])
        options = run.call_args.args[0]
        assert options.backend == "cryolo"
        assert options.cryolo_python == "chosen-python"
        assert options.cryolo_threshold == .42
        assert options.min_capsid_area == 4100
        app.main([])
        assert run.call_args.args[0].cryolo_python is None


def test_invalid_persisted_settings_do_not_break_launch():
    with tempfile.TemporaryDirectory() as directory, \
         patch.object(app, "user_state_directory", return_value=Path(directory)), \
         patch.object(app, "_run_application", return_value=0) as run:
        (Path(directory) / "cryolo-settings.json").write_text('{"cryolo_threshold": -9}')
        main_cryolo.main([])
        assert run.call_args.args[0].cryolo_threshold == .3


def test_parallel_checkpoint_does_not_select_sam_weights():
    with tempfile.TemporaryDirectory() as directory, patch.object(app, "__file__", str(Path(directory) / "app.py")):
        models = Path(directory) / "models"
        models.mkdir()
        (models / app.DEFAULT_CHECKPOINT).write_bytes(b"sam")
        assert app.resolve_checkpoint(backend="cryolo") == models / "cryolo-capsids.h5"
        (models / "cryolo-capsids.h5").write_bytes(b"cryolo")
        assert app.resolve_checkpoint(backend="cryolo").read_bytes() == b"cryolo"
        explicit = Path(directory) / "missing.h5"
        assert app.resolve_checkpoint(explicit, backend="cryolo") == explicit


def test_screen_settings_reach_cryolo_configuration():
    picker = screen_app.ScreenPickerApp.__new__(screen_app.ScreenPickerApp)
    picker.args = SimpleNamespace(backend="cryolo", cryolo_timeout=60)
    values = {"checkpoint": "model.h5", "model_type": "vit_b", "device": "cpu",
              "grid_side": 8, "automatic_profile": "capsid-learned", "min_capsid_area": "3500",
              "cryolo_config": "config.json", "cryolo_python": "python37", "cryolo_threshold": ".2"}
    for name, value in values.items():
        setattr(picker, name, SimpleNamespace(get=lambda value=value: value))
    config = picker._model_config()
    assert config.backend == "cryolo"
    assert config.cryolo_config == "config.json"
    assert config.cryolo_python == "python37"
    assert config.cryolo_threshold == .2
    assert config.cryolo_timeout == 60
    assert config.min_capsid_area_px == 3500


def test_screen_factory_never_uses_sam_for_cryolo():
    picker = screen_app.ScreenPickerApp.__new__(screen_app.ScreenPickerApp)
    picker.recognizer = picker.recognizer_config = None
    config = RecognitionConfig("model.h5", backend="cryolo")
    with patch.object(screen_app, "SamRecognizer") as sam, \
         patch.object(screen_app, "create_recognizer") as factory:
        assert picker._get_recognizer(config) is factory.return_value
        picker._get_recognizer(config)
        factory.assert_called_once_with(config)
        sam.assert_not_called()


def test_saved_settings_remain_in_separate_backend_file():
    config = RecognitionConfig("model.h5", backend="cryolo", cryolo_python="python37",
                               cryolo_threshold=.2, min_capsid_area_px=4200)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "cryolo-settings.json"
        screen_app.save_cryolo_settings(config, path)
        data = json.loads(path.read_text())
        assert data["checkpoint"] == "model.h5"
        assert data["cryolo_python"] == "python37"
        assert data["min_capsid_area"] == 4200
        original = path.read_bytes()
        screen_app.save_cryolo_settings(RecognitionConfig("sam.pth"), path)
        assert path.read_bytes() == original


def test_cryolo_diagnostics_do_not_claim_runtime_inference():
    config = RecognitionConfig("model.h5", backend="cryolo", cryolo_config="absent.json")
    report = app.cryolo_diagnostic_report(Path("test.log"), config)
    assert report["backend"] == "cryolo"
    assert report["cryolo_config_exists"] is False
    assert "segment-anything" not in report["dependencies"]
    assert "Metadata inspection only" in report["note"]


def test_cryolo_smoke_failure_cannot_report_success():
    config = RecognitionConfig("model.h5", backend="cryolo")
    detector = Mock()
    detector.automatic.side_effect = RuntimeError("crYOLO runtime is missing")
    with patch.object(app, "create_recognizer", return_value=detector), \
         patch.object(app, "SamRecognizer") as sam, patch.object(app, "_smoke_image_processing"):
        report = app.run_cryolo_smoke_test(Path("test.log"), config)
    assert report["status"] == "error"
    assert "missing" in report["error"]
    detector.close.assert_called_once()
    sam.assert_not_called()


def test_headless_cryolo_path_passes_runtime_and_exports_results():
    args = app.build_parser().parse_args(["--backend", "cryolo", "--segment-image", "input.png",
                                         "--checkpoint", "model.h5", "--cryolo-python", "python37"])
    detector = Mock()
    detector.automatic.return_value = []
    detector.last_detection = {"backend": "cryolo"}
    with tempfile.TemporaryDirectory() as directory:
        args.output = Path(directory)
        with patch.object(app, "setup_logging", return_value=Path(directory) / "test.log"), \
             patch.object(app, "load_rgb_file", return_value=np.zeros((64, 64, 3), dtype=np.uint8)), \
             patch.object(app, "create_recognizer", return_value=detector) as factory, \
             patch.object(app, "SamRecognizer") as sam, patch.object(app, "export_review") as export:
            assert app._run_application(args, console_available=True) == 0
        config = factory.call_args.args[0]
        assert config.backend == "cryolo"
        assert config.cryolo_python == "python37"
        assert Path(config.cryolo_config).name == "cryolo-config.json"
        assert config.min_capsid_area_px == 3500
        export.assert_called_once()
        sam.assert_not_called()
        detector.close.assert_called_once()
