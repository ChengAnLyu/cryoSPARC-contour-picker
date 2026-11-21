"""Windowed entry point for the separately installed crYOLO edition."""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback


def launch() -> int:
    directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SARU_Picker_crYOLO"
    directory.mkdir(parents=True, exist_ok=True)
    fault_stream = (directory / "native_fault.log").open("a", encoding="utf-8")
    try:
        for name in ("stdout", "stderr"):
            if getattr(sys, name) is None:
                setattr(sys, name, (directory / "startup.log").open("a", encoding="utf-8", buffering=1))
        faulthandler.enable(file=fault_stream, all_threads=True)
        import main_cryolo
        return int(main_cryolo.main() or 0)
    except Exception as exc:
        detail = traceback.format_exc()
        (directory / "startup-error.txt").write_text(detail, encoding="utf-8")
        headless = any(arg.split("=", 1)[0] in {"--diagnostics", "--smoke-test", "--segment-image"}
                       for arg in sys.argv[1:])
        if not headless and sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                f"SARU Picker crYOLO could not start.\n\n{exc}\n\nDetails: {directory}",
                "SARU Picker crYOLO", 0x10)
        else:
            print(detail, file=sys.stderr)
        return 1
    finally:
        faulthandler.disable()
        fault_stream.close()


if __name__ == "__main__":
    raise SystemExit(launch())
