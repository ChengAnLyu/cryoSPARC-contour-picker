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


class RuntimeSetupTimeout(RuntimeSetupError):
    """A bounded runtime check or installation exceeded its deadline."""


class RuntimeSetupCancelled(RuntimeSetupError):
    """The user cancelled runtime preparation."""


_DLL_LOCK = threading.RLock()
_SETUP_LOCK = threading.Lock()
_HEALTHY: set[tuple] = set()
_HEALTH_TIMEOUT = 120
_SETUP_TIMEOUT = 2700


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


def _read_report(path):
    try:
        report = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return report if isinstance(report, dict) else {}
    except (OSError, ValueError):
        return {}


def _run(command, *, cancel=None, timeout=120, log_path=None, progress=None,
         description='Preparing crYOLO', status_path=None):
    """Capture without pipes, so verbose pip output cannot block cancellation."""
    if _cancelled(cancel):
        raise RuntimeSetupCancelled('crYOLO setup was cancelled.')
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        with external_process_environment() as env:
            process = subprocess.Popen([str(part) for part in command], stdin=subprocess.DEVNULL,
                stdout=output, stderr=errors, env=env,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0)
        started = time.monotonic()
        last_progress = started - 5
        try:
            while process.poll() is None:
                if _cancelled(cancel):
                    raise RuntimeSetupCancelled('crYOLO setup was cancelled. It will resume automatically next time.')
                if time.monotonic() - started > timeout:
                    raise RuntimeSetupTimeout(f'{description} exceeded {timeout:g} seconds.')
                if progress and time.monotonic() - last_progress >= 5:
                    last_progress = time.monotonic()
                    stage = _read_report(status_path).get('stage') if status_path else None
                    detail = f' — {stage}' if stage else ''
                    progress(f'{description}{detail} ({last_progress - started:.0f}s)…')
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


def _diagnostic_directory(python):
    # Managed interpreters have <app>/cryolo-env/Scripts/python.exe. The fallback
    # keeps probes of test/nonstandard interpreters beside that interpreter.
    return python.parent.parent.parent if python.parent.name == 'Scripts' else python.parent


def _probe(python, *, cancel=None, progress=None):
    if not python.is_file():
        return {'status': 'missing', 'python': str(python)}
    probe = _resource('runtime_health.py')
    key = _health_key(python, probe)
    if key in _HEALTHY:
        return {'status': 'ok', 'python': str(python), 'cached': True}
    directory = _diagnostic_directory(python)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / 'runtime-probe.log'
    report_path = directory / 'runtime-probe.json'
    status_path = directory / 'runtime-probe-progress.json'
    status_path.write_text(json.dumps({'stage': 'starting runtime check'}), encoding='utf-8')
    started = time.monotonic()
    report = {'status': 'error', 'python': str(python), 'log': str(log_path)}
    try:
        result = _run([python, probe, '--progress-report', status_path], cancel=cancel,
                      timeout=_HEALTH_TIMEOUT, log_path=log_path, progress=progress,
                      description='Checking crYOLO runtime', status_path=status_path)
        report['returncode'] = result.returncode
        if sys.platform == 'win32' and result.returncode:
            report['native_exit_code'] = f'0x{result.returncode & 0xffffffff:08X}'
        try:
            native_report = json.loads(result.stdout)
            if not isinstance(native_report, dict):
                raise ValueError('The runtime check returned an invalid report.')
        except (ValueError, TypeError):
            native_report = {}
        if result.returncode == 0 and native_report.get('status') == 'ok':
            report.update(native_report)
            _HEALTHY.add(key)
        else:
            report['status'] = 'error'
            code = report.get('native_exit_code', str(result.returncode))
            report['error'] = native_report.get('error') or f'The runtime check stopped without a valid report (exit {code}).'
            if native_report.get('error_type'):
                report['error_type'] = native_report['error_type']
            if result.stderr.strip():
                report['stderr_tail'] = result.stderr[-4000:]
    except RuntimeSetupCancelled:
        report.update(status='cancelled', error='Runtime check cancelled.')
        raise
    except RuntimeSetupTimeout as exc:
        report.update(status='timeout', error=str(exc))
    except (OSError, ValueError, RuntimeSetupError) as exc:
        report.update(status='error', error=str(exc))
    finally:
        report['stage'] = _read_report(status_path).get('stage', 'runtime check')
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def runtime_status() -> dict:
    """Read-only discovery and CPU health check; never starts installation."""
    try:
        return _probe(default_runtime_python())
    except RuntimeSetupError as exc:
        return {'status': 'unavailable', 'error': str(exc)}


def _prepare_cryolo_runtime(*, cancel=None, progress=None, force_private=False, repair=False) -> Path:
    if sys.platform != 'win32':
        raise RuntimeSetupError('Automatic crYOLO setup is available on Windows. Configure a crYOLO runtime for this platform.')
    python = default_runtime_python()
    while not _SETUP_LOCK.acquire(timeout=0.1):
        if _cancelled(cancel):
            raise RuntimeSetupCancelled('crYOLO setup was cancelled.')
    try:
        if progress:
            progress('Preparing crYOLO runtime repair…' if repair else 'Checking the crYOLO runtime…')
        if repair:
            _HEALTHY.clear()
        health = {'status': 'missing'} if repair else _probe(python, cancel=cancel, progress=progress)
        if health.get('status') == 'ok':
            return python
        if health.get('status') != 'missing':
            detail = health.get('error', 'The runtime health check failed.')
            stage = health.get('stage', 'runtime check')
            log = health.get('log', str(_diagnostic_directory(python) / 'runtime-probe.log'))
            raise RuntimeSetupError(f'crYOLO runtime check failed during {stage}: {detail} '
                f'Open Advanced and choose Repair crYOLO runtime. Diagnostic log: {log}')
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
        if repair:
            command.append('-Repair')
        report_path.write_text(json.dumps({'status': 'running', 'stage': 'starting repair' if repair else 'starting setup'}), encoding='utf-8')
        result = _run(command, cancel=cancel, timeout=_SETUP_TIMEOUT, log_path=log_path,
                      progress=progress, description='Repairing crYOLO runtime' if repair else 'Setting up crYOLO runtime',
                      status_path=report_path)
        if result.returncode:
            error = 'Automatic setup could not finish.'
            try:
                report = json.loads(report_path.read_text(encoding='utf-8-sig'))
                error = report.get('error', error)
            except (OSError, ValueError):
                pass
            raise RuntimeSetupError(f'{error} Retry crYOLO detection after correcting the problem. Setup log: {log_path}')
        if _probe(python, cancel=cancel, progress=progress).get('status') != 'ok':
            raise RuntimeSetupError(f'crYOLO setup finished but verification failed. Setup log: {log_path}')
        if progress:
            progress('crYOLO is ready.')
        return python
    finally:
        _SETUP_LOCK.release()


def ensure_cryolo_runtime(*, cancel=None, progress=None, force_private=False) -> Path:
    """Verify a managed runtime, or install one when absent.

    Existing interpreters that time out, crash or fail their health check are
    reported without silently starting a package reinstall during recognition.
    """
    return _prepare_cryolo_runtime(cancel=cancel, progress=progress, force_private=force_private)


def repair_cryolo_runtime(*, cancel=None, progress=None) -> Path:
    """Explicitly rebuild the application-owned private runtime in a worker.

    Used by Advanced > Repair crYOLO runtime. Package reinstall is forced, the
    health cache is cleared, and cancellation/progress are shared with setup.
    Ownership checks in the setup script protect unrelated installations.
    """
    return _prepare_cryolo_runtime(cancel=cancel, progress=progress, force_private=True, repair=True)
