import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import runtime_management as runtime


@pytest.fixture(autouse=True)
def clean_health_cache():
    runtime._HEALTHY.clear()
    yield
    runtime._HEALTHY.clear()


def test_default_path_and_missing_status(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    expected = tmp_path / 'SARU_Picker_crYOLO' / 'cryolo-env' / 'Scripts' / 'python.exe'
    assert runtime.default_runtime_python() == expected
    assert runtime.runtime_status()['status'] == 'missing'


def test_health_cache_rechecks_changed_marker(tmp_path, monkeypatch):
    python = tmp_path / 'env' / 'Scripts' / 'python.exe'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'python')
    marker = python.parent.parent / 'saru-cryolo-runtime.json'
    marker.write_text('{}')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"status":"ok"}', '')
    monkeypatch.setattr(runtime, '_run', run)
    assert runtime._probe(python)['status'] == 'ok'
    assert runtime._probe(python)['cached']
    marker.write_text('{"updated":true}')
    assert runtime._probe(python)['status'] == 'ok'
    assert len(calls) == 2


def test_auto_setup_has_no_python_selection_and_checks_result(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    results = iter([{'status': 'missing'}, {'status': 'ok'}])
    monkeypatch.setattr(runtime, '_probe', lambda *a, **kw: next(results))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(runtime, '_run', run)
    assert runtime.ensure_cryolo_runtime(force_private=True) == runtime.default_runtime_python()
    args = [str(p) for p in commands[0]]
    assert '-ForcePrivatePython' in args
    assert '-Python37' not in args
    assert '-RuntimeDirectory' in args


def test_healthy_runtime_never_invokes_setup(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(runtime, '_probe', lambda *a, **kw: {'status': 'ok'})
    monkeypatch.setattr(runtime, '_run', lambda *a, **kw: pytest.fail('healthy setup must remain offline'))
    assert runtime.ensure_cryolo_runtime() == runtime.default_runtime_python()


def test_setup_failure_reports_stage_error(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(runtime, '_probe', lambda *a, **kw: {'status': 'error'})
    def run(command, **kwargs):
        report = runtime.default_runtime_python().parent.parent.parent / 'runtime-install-report.json'
        report.write_text(json.dumps({'status': 'error', 'stage': 'packages', 'error': 'Download unavailable'}), encoding='utf-8-sig')
        return subprocess.CompletedProcess(command, 1, '', '')
    monkeypatch.setattr(runtime, '_run', run)
    with pytest.raises(runtime.RuntimeSetupError, match='Download unavailable'):
        runtime.ensure_cryolo_runtime()


def test_cancelled_probe_is_not_converted_to_install_attempt(tmp_path, monkeypatch):
    python = tmp_path / 'python.exe'
    python.touch()
    def run(*args, **kwargs):
        raise runtime.RuntimeSetupCancelled('cancelled')
    monkeypatch.setattr(runtime, '_run', run)
    with pytest.raises(runtime.RuntimeSetupCancelled):
        runtime._probe(python)


def test_cancellation_reaps_real_child():
    cancel = threading.Event()
    timer = threading.Timer(.2, cancel.set)
    timer.start()
    try:
        with pytest.raises(runtime.RuntimeSetupCancelled):
            runtime._run([sys.executable, '-c', 'import time; time.sleep(30)'], cancel=cancel)
    finally:
        timer.cancel()


def test_external_environment_is_copied_and_python_paths_removed(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'frozen', False, raising=False)
    source = {'PATH': '/a:/b', 'PYTHONHOME': '/wrong', 'PYTHONPATH': '/wrong', 'OMP_NUM_THREADS': '4'}
    with runtime.external_process_environment(source) as child:
        assert 'PYTHONHOME' not in child and 'PYTHONPATH' not in child
        assert child['PYTHONNOUSERSITE'] == '1'
        assert child['OMP_NUM_THREADS'] == '4'
    assert source['PYTHONHOME'] == '/wrong'


def test_frozen_windows_dll_directory_restored_even_on_error(monkeypatch):
    import ctypes
    class Function:
        def __init__(self, fn): self.fn = fn
        def __call__(self, *args): return self.fn(*args)
    changes = []
    previous = r'C:\SARU\_internal'
    def get(size, buffer):
        if buffer is not None: buffer.value = previous
        return len(previous)
    class Kernel: pass
    kernel = Kernel()
    kernel.GetDllDirectoryW = Function(get)
    kernel.SetDllDirectoryW = Function(lambda directory: changes.append(directory) or 1)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(runtime.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(runtime.sys, 'platform', 'win32')
    monkeypatch.setattr(runtime.sys, '_MEIPASS', previous, raising=False)
    with pytest.raises(ValueError):
        with runtime.external_process_environment({'PATH': previous + ';' + previous + r'\torch\lib;C:\Windows'}) as child:
            assert child['PATH'] == r'C:\Windows'
            assert changes == [None]
            raise ValueError('child spawn failed')
    assert changes == [None, previous]


def test_probe_rejects_failed_exit_even_with_ok_json(tmp_path, monkeypatch):
    python = tmp_path / 'python.exe'
    python.touch()
    monkeypatch.setattr(runtime, '_run', lambda *a, **kw: subprocess.CompletedProcess([], 1, '{"status":"ok"}', 'bad'))
    assert runtime._probe(python)['status'] == 'error'
    assert not runtime._HEALTHY
