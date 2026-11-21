"""Native Windows checks of shared controls and the crYOLO settings panel."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import app
from screen_app import ScreenPickerApp


def verify(destination):
    if sys.platform != "win32":
        raise RuntimeError("Native GUI verification requires Windows.")
    assert app.APP_VERSION == "3.0.0", app.APP_VERSION
    app.ACTIVE_BACKEND = "cryolo"
    log_path = app.setup_logging()
    shared = app._smoke_windows_runtime(log_path, screenshot_dir=destination / "ui")
    import tkinter as tk
    args = app.build_parser().parse_args([])
    assert args.backend == "cryolo" and args.automatic_profile == "capsid-learned"
    root = tk.Tk()
    picker = None
    try:
        picker = ScreenPickerApp(root, args, log_path, app.resolve_checkpoint(backend="cryolo"), start_listeners=False)
        root.update()
        assert "crYOLO" in root.title() and "3.0.0" in root.title(), root.title()
        config = picker._model_config()
        assert config.backend == "cryolo" and config.min_capsid_area_px == 3500
        assert config.automatic_profile == "capsid-learned"
        picker.open_advanced()
        root.update()
        assert picker.advanced is not None
        assert picker._listeners_requested is False
        return {"status": "ok", "version": app.APP_VERSION, "default_backend": args.backend,
                "shared_controls": shared,
                "cryolo_panel": {"title": root.title(), "minimum_contour_area": 3500,
                                 "profile": "capsid-learned", "advanced_opened": True},
                "note": "Native Tk/shared-controller checks with synthetic input sinks; no trained-model inference or live picking."}
    finally:
        if picker is not None:
            picker.close()
        elif root.winfo_exists():
            root.destroy()


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    report = verify(target)
    (target / "source-ui-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
