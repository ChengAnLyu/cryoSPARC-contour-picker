"""Windowed entry point for SARU 3.0 with four detection profiles."""
from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback


def enforce_offline_verification() -> None:
    """CI guard, inherited by the frozen worker; normal operation is unchanged."""
    if os.environ.get('SARU_VERIFY_OFFLINE') != '1':
        return
    import socket
    import subprocess

    def deny_network(*args, **kwargs):
        raise RuntimeError('Offline verification forbids network access.')

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    original_popen = subprocess.Popen

    def local_worker_only(command, *args, **kwargs):
        if (not isinstance(command, (list, tuple)) or len(command) < 2
                or Path(command[0]).resolve() != Path(sys.executable).resolve()
                or command[1] != '--cryolo-onnx-worker'):
            raise RuntimeError('Offline verification forbids external runtimes and setup processes.')
        return original_popen(command, *args, **kwargs)

    subprocess.Popen = local_worker_only


def launch() -> int:
    directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SARU_Picker_crYOLO"
    directory.mkdir(parents=True, exist_ok=True)
    fault_stream = (directory / "native_fault.log").open("a", encoding="utf-8")
    try:
        for name in ("stdout", "stderr"):
            if getattr(sys, name) is None:
                setattr(sys, name, (directory / "startup.log").open("a", encoding="utf-8", buffering=1))
        faulthandler.enable(file=fault_stream, all_threads=True)
        enforce_offline_verification()
        if '--cryolo-onnx-worker' in sys.argv[1:]:
            from cryolo_onnx import main as onnx_worker
            return onnx_worker([arg for arg in sys.argv[1:] if arg != '--cryolo-onnx-worker'])
        if '--cryolo-onnx-smoke' in sys.argv[1:]:
            from verify_onnx import main as verify_onnx
            return verify_onnx([arg for arg in sys.argv[1:] if arg != '--cryolo-onnx-smoke'])
        if '--cryolo-refinement-smoke' in sys.argv[1:]:
            from verify_refinement import main as verify_refinement
            return verify_refinement([arg for arg in sys.argv[1:] if arg != '--cryolo-refinement-smoke'])
        import app
        return int(app.main() or 0)
    except Exception as exc:
        detail = traceback.format_exc()
        (directory / "startup-error.txt").write_text(detail, encoding="utf-8")
        headless = any(arg.split("=", 1)[0] in {"--diagnostics", "--smoke-test", "--segment-image", "--cryolo-refinement-smoke", "--cryolo-onnx-worker", "--cryolo-onnx-smoke"}
                       for arg in sys.argv[1:])
        if not headless and sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                f"SARU Picker 3.0 could not start.\n\n{exc}\n\nDetails: {directory}",
                "SARU Picker 3.0", 0x10)
        else:
            print(detail, file=sys.stderr)
        return 1
    finally:
        faulthandler.disable()
        fault_stream.close()


if __name__ == "__main__":
    raise SystemExit(launch())
