"""Automatic private crYOLO runtime preparation, independent of the GUI toolkit."""
from __future__ import annotations

from contextlib import contextmanager
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


class RuntimeSetupError(RuntimeError):
    """The managed runtime could not be prepared."""


class RuntimeSetupCancelled(RuntimeSetupError):
    """The user cancelled runtime preparation."""


_DLL_LOCK = threading.RLock()
_SETUP_LOCK = threading.Lock()
_HEALTHY: set[tuple] = set()


def default_runtime_python() -> Path:
    base = os.environ.get('LOCALAPPDATA')
    if not base:
        raise RuntimeSetupError('The Windows user application folder is unavailable.')
    return Path(base) / 'SARU_Picker_crYOLO' / 'cryolo-env' / 'Scripts' / 'python.exe'


def _resource(name: str) -> Path:
    roots = []
    if getattr(sys, 'frozen', False):
        roots.append(Path(sys.executable).resolve().parent)
        if getattr(sys, '_MEIPASS', None):
            roots.append(Path(sys._MEIPASS))
    roots.append(Path(__file__).resolve().parent / 'packaging' / 'cryolo')
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise RuntimeSetupError('The application is missing its crYOLO setup files. Reinstall SARU Picker.')


@contextmanager
def external_process_environment(env=None):
    """Isolate external Python from PyInstaller's bundled DLLs and Python paths.

    Keep this context around Popen only. The child's library search state is
    inherited at creation; the GUI state is restored immediately afterwards.
    See PyInstaller's official common-issues-and-pitfalls external-program guide.
    """
    child = dict(os.environ if env is None else env)
    child.pop('PYTHONHOME', None)
    child.pop('PYTHONPATH', None)
    child['PYTHONNOUSERSITE'] = '1'
    frozen = bool(getattr(sys, 'frozen', False))
    if frozen and sys.platform == 'win32':
        root = getattr(sys, '_MEIPASS', None)
        if root:
            root = ntpath.normcase(ntpath.abspath(str(root))).rstrip('\\/')
            child['PATH'] = ';'.join(part for part in child.get('PATH', '').split(';')
                if not (ntpath.normcase(ntpath.abspath(part.strip('"'))).rstrip('\\/') == root
                        or ntpath.normcase(ntpath.abspath(part.strip('"'))).startswith(root + '\\')))
        import ctypes
        from ctypes import wintypes
        with _DLL_LOCK:
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            get_directory = kernel.GetDllDirectoryW
            get_directory.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
            get_directory.restype = wintypes.DWORD
            set_directory = kernel.SetDllDirectoryW
            set_directory.argtypes = [wintypes.LPCWSTR]
            set_directory.restype = wintypes.BOOL
            size = get_directory(0, None)
            buffer = ctypes.create_unicode_buffer(size + 1)
            if size:
                get_directory(len(buffer), buffer)
            previous = buffer.value if size else None
            if not set_directory(None):
                raise OSError(ctypes.get_last_error(), 'Cannot isolate the external runtime DLL search path.')
            try:
                yield child
            finally:
                if not set_directory(previous):
                    raise OSError(ctypes.get_last_error(), 'Cannot restore the application DLL search path.')
    else:
        if frozen and sys.platform.startswith('linux'):
            if 'LD_LIBRARY_PATH_ORIG' in child:
                child['LD_LIBRARY_PATH'] = child['LD_LIBRARY_PATH_ORIG']
            else:
                child.pop('LD_LIBRARY_PATH', None)
        yield child


def _cancelled(cancel) -> bool:
    return bool(cancel and (cancel.is_set() if hasattr(cancel, 'is_set') else cancel()))


def _stop(process):
    if process.poll() is not None:
        return
    if sys.platform == 'win32':
        try:
            with external_process_environment() as env:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run(command, *, cancel=None, timeout=120, log_path=None):
    """Capture without pipes, so verbose pip output cannot block cancellation."""
    if _cancelled(cancel):
        raise RuntimeSetupCancelled('crYOLO setup was cancelled.')
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        with external_process_environment() as env:
            process = subprocess.Popen([str(part) for part in command], stdin=subprocess.DEVNULL,
                stdout=output, stderr=errors, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if _cancelled(cancel):
                    raise RuntimeSetupCancelled('crYOLO setup was cancelled. It will resume automatically next time.')
                if time.monotonic() - started > timeout:
                    raise RuntimeSetupError('crYOLO setup timed out. Check the internet connection and retry.')
                time.sleep(0.1)
        except BaseException:
            _stop(process)
            raise
        finally:
            if log_path:
                output.seek(0)
                errors.seek(0)
                Path(log_path).write_bytes(output.read() + b'\n' + errors.read())
        output.seek(0)
        errors.seek(0)
        return subprocess.CompletedProcess(command, process.returncode,
            output.read().decode('utf-8', errors='replace'), errors.read().decode('utf-8', errors='replace'))


def _health_key(python, probe):
    paths = [python, probe, _resource('runtime-requirements-windows.txt'),
             python.parent.parent / 'saru-cryolo-runtime.json']
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 if path.exists() else (str(path), None, None) for path in paths)


def _probe(python, *, cancel=None):
    if not python.is_file():
        return {'status': 'missing', 'python': str(python)}
    probe = _resource('runtime_health.py')
    key = _health_key(python, probe)
    if key in _HEALTHY:
        return {'status': 'ok', 'python': str(python), 'cached': True}
    try:
        result = _run([python, probe], cancel=cancel, timeout=120)
        report = json.loads(result.stdout)
        if result.returncode == 0 and report.get('status') == 'ok':
            _HEALTHY.add(key)
            return report
        return {'status': 'error', 'python': str(python),
                'error': report.get('error', 'The runtime health check failed.')}
    except RuntimeSetupCancelled:
        raise
    except (OSError, ValueError, RuntimeSetupError) as exc:
        return {'status': 'error', 'python': str(python), 'error': str(exc)}


def runtime_status() -> dict:
    """Read-only discovery and CPU health check; never starts installation."""
    try:
        return _probe(default_runtime_python())
    except RuntimeSetupError as exc:
        return {'status': 'unavailable', 'error': str(exc)}


def ensure_cryolo_runtime(*, cancel=None, progress=None, force_private=False) -> Path:
    """Prepare the managed runtime in a worker; return only after verified ready."""
    if sys.platform != 'win32':
        raise RuntimeSetupError('Automatic crYOLO setup is available on Windows. Configure a crYOLO runtime for this platform.')
    python = default_runtime_python()
    while not _SETUP_LOCK.acquire(timeout=0.1):
        if _cancelled(cancel):
            raise RuntimeSetupCancelled('crYOLO setup was cancelled.')
    try:
        if progress:
            progress('Checking the crYOLO runtime…')
        if _probe(python, cancel=cancel).get('status') == 'ok':
            return python
        setup = _resource('setup-runtime.ps1')
        parent = python.parent.parent.parent
        parent.mkdir(parents=True, exist_ok=True)
        report_path = parent / 'runtime-install-report.json'
        log_path = parent / 'runtime-setup.log'
        if progress:
            progress('Setting up crYOLO automatically. The first setup may take several minutes…')
        system_root = os.environ.get('SystemRoot', r'C:\Windows')
        powershell = Path(system_root) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
        command = [powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', setup,
                   '-RuntimeDirectory', python.parent.parent, '-Report', report_path]
        if force_private:
            command.append('-ForcePrivatePython')
        result = _run(command, cancel=cancel, timeout=2700, log_path=log_path)
        if result.returncode:
            error = 'Automatic setup could not finish.'
            try:
                report = json.loads(report_path.read_text(encoding='utf-8-sig'))
                error = report.get('error', error)
            except (OSError, ValueError):
                pass
            raise RuntimeSetupError(f'{error} Retry crYOLO detection after correcting the problem. Setup log: {log_path}')
        if _probe(python, cancel=cancel).get('status') != 'ok':
            raise RuntimeSetupError(f'crYOLO setup finished but verification failed. Setup log: {log_path}')
        if progress:
            progress('crYOLO is ready.')
        return python
    finally:
        _SETUP_LOCK.release()
