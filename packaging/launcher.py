"""Windowed entry point: preserve startup diagnostics before importing the app."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def _startup_directory() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "SARU_Picker" if local else Path.home() / ".saru_picker"


def _startup_log() -> Path | None:
    preferred = _startup_directory()
    for directory in (preferred, Path(tempfile.gettempdir()) / "SARU_Picker"):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "startup.log"
            with target.open("a", encoding="utf-8"):
                pass
            return target
        except OSError:
            continue
    return None


def _report_argument() -> Path | None:
    for index, argument in enumerate(sys.argv[1:], 1):
        if argument.startswith("--report="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
        if argument == "--report" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve()
    return None


def launch() -> int:
    startup_log = None
    try:
        startup_log = _startup_log()
        # GUI bootloaders have no console. Some native packages still print
        # warnings while importing, so provide streams before importing them.
        destination = startup_log if startup_log is not None else Path(os.devnull)
        if sys.stdout is None:
            sys.stdout = destination.open("a", encoding="utf-8", buffering=1)
        if sys.stderr is None:
            sys.stderr = destination.open("a", encoding="utf-8", buffering=1)
        import main as saru_app  # Static import so PyInstaller analyzes the app.
        return int(saru_app.main() or 0)
    except Exception as exc:
        detail = traceback.format_exc()
        if startup_log is not None:
            try:
                with startup_log.open("a", encoding="utf-8") as handle:
                    handle.write(detail + "\n")
            except OSError:
                pass
        headless = any(argument.split("=", 1)[0] in {
            "--smoke-test", "--diagnostics", "--segment-image",
        } for argument in sys.argv[1:])
        if headless:
            try:
                report = _report_argument()
                if report is not None:
                    report.parent.mkdir(parents=True, exist_ok=True)
                    report.write_text(json.dumps({
                        "status": "error", "stage": "startup",
                        "error": str(exc),
                        "log_path": str(startup_log) if startup_log else None,
                    }, indent=2), encoding="utf-8")
            except Exception:
                if sys.stderr is not None:
                    traceback.print_exc(file=sys.stderr)
            return 1
        if sys.platform == "win32":
            try:
                log_note = (f"Details were saved to:\n{startup_log}" if startup_log
                            else "A startup log could not be written.")
                ctypes.windll.user32.MessageBoxW(
                    None,
                    f"SARU Picker could not start.\n\n{exc}\n\n{log_note}",
                    "SARU Picker — startup error", 0x10,
                )
            except Exception:
                if sys.stderr is not None:
                    traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(launch())
