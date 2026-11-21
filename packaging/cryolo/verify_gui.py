"""Native Windows checks of the four-profile selector and shared controls."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import app
from detector_selection import DETECTION_PROFILES
from screen_app import ScreenPickerApp


def verify(destination):
    if sys.platform != "win32":
        raise RuntimeError("Native GUI verification requires Windows.")
    assert app.APP_VERSION == "3.0.0", app.APP_VERSION
    log_path = app.setup_logging()
    shared = app._smoke_windows_runtime(log_path, screenshot_dir=destination / "ui")
    import tkinter as tk
    args = app.build_parser().parse_args([])
    assert args.backend == "sam" and args.automatic_profile == "capsid"
    root = tk.Tk()
    picker = None
    try:
        picker = ScreenPickerApp(root, args, log_path, app.resolve_checkpoint(backend="sam"), start_listeners=False)
        root.update()
        assert root.title() == "SARU PICKER 3.0.0", root.title()
        picker.open_advanced()
        root.update()
        assert picker.advanced is not None
        assert tuple(picker.profile_selector.cget("values")) == tuple(DETECTION_PROFILES)
        assert tuple(DETECTION_PROFILES) == ("capsid", "capsid-learned", "generic", "crYOLO")
        sam_path = picker.sam_checkpoint.get()
        cryolo_path = picker.cryolo_checkpoint.get()
        assert sam_path != cryolo_path
        routing = []
        for profile in DETECTION_PROFILES:
            picker.detection_profile.set(profile)
            root.update()
            config = picker._model_config()
            expected_backend = "cryolo" if profile == "crYOLO" else "sam"
            expected_profile = "capsid-learned" if profile == "crYOLO" else profile
            assert config.backend == expected_backend, (profile, config.backend)
            assert config.automatic_profile == expected_profile
            assert config.checkpoint == (cryolo_path if profile == "crYOLO" else sam_path)
            assert config.min_capsid_area_px == 3500
            assert config.cryolo_python is None
            routing.append({"selection": profile, "backend": config.backend,
                            "contour_profile": config.automatic_profile,
                            "checkpoint": config.checkpoint})
        picker.detection_profile.set("capsid")
        assert picker._listeners_requested is False
        report = {"status": "ok", "version": app.APP_VERSION, "default_backend": args.backend,
                "default_profile": "capsid", "shared_controls": shared,
                "panel": {"title": root.title(), "profiles": routing,
                          "separate_checkpoint_paths": True, "advanced_opened": True,
                          "runtime_selection_required": False},
                "note": "Actual Windows Tk selector/controller checks with synthetic input sinks. Model inference is checked separately."}
    finally:
        if picker is not None:
            picker.close()
        elif root.winfo_exists():
            root.destroy()
    review_root = tk.Tk()
    review = None
    try:
        review = app.PickerApp(review_root, app.build_parser().parse_args(["--review"]),
                               log_path, start_hotkeys=False)
        review_root.update()
        assert tuple(review.profile_selector.cget("values")) == tuple(DETECTION_PROFILES)
        paths = {"sam": review.sam_checkpoint.get(), "cryolo": review.cryolo_checkpoint.get()}
        assert paths["sam"] != paths["cryolo"]
        routing = []
        for profile in DETECTION_PROFILES:
            review.detection_profile.set(profile)
            review._apply_model_settings()
            review_root.update()
            config = review._model_config()
            expected_backend = "cryolo" if profile == "crYOLO" else "sam"
            assert config.backend == expected_backend
            assert config.automatic_profile == ("capsid-learned" if profile == "crYOLO" else profile)
            assert config.checkpoint == paths[expected_backend]
            assert config.cryolo_python is None
            routing.append({"selection": profile, "backend": config.backend,
                            "contour_profile": config.automatic_profile})
        report["review_panel"] = {"profiles": routing, "runtime_selection_required": False}
    finally:
        if review is not None:
            review.close()
        elif review_root.winfo_exists():
            review_root.destroy()
    return report


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    report = verify(target)
    (target / "source-ui-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
