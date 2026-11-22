"""Executable dispatch and offline release-boundary regression tests."""
from pathlib import Path
import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / 'packaging' / 'cryolo'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_dispatch_precedes_application_import(tmp_path, monkeypatch):
    launcher = load_module('offline_launcher_test', 'launcher.py')
    worker = Mock(return_value=7)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.delenv('SARU_VERIFY_OFFLINE', raising=False)
    monkeypatch.setattr(sys, 'argv', ['picker.exe', '--cryolo-onnx-worker', '--model', 'fixture.onnx'])
    monkeypatch.setitem(sys.modules, 'cryolo_onnx', SimpleNamespace(main=worker))
    monkeypatch.setitem(sys.modules, 'app', SimpleNamespace(main=Mock(side_effect=AssertionError('GUI import path must not run'))))
    assert launcher.launch() == 7
    worker.assert_called_once_with(['--model', 'fixture.onnx'])


def test_offline_guard_denies_network_and_external_runtime_but_permits_own_worker(monkeypatch):
    launcher = load_module('offline_guard_test', 'launcher.py')
    monkeypatch.setenv('SARU_VERIFY_OFFLINE', '1')
    hooks = []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    original_popen = subprocess.Popen
    launcher.enforce_offline_verification()
    assert subprocess.Popen is original_popen
    class TorchStylePopenSubclass(subprocess.Popen):
        pass
    assert issubclass(TorchStylePopenSubclass, original_popen)
    guard, = hooks
    for event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        with pytest.raises(RuntimeError, match='network'):
            guard(event, ())
    for command in [['powershell.exe', '-File', 'setup.ps1'], ['python.exe', '-m', 'cryolo.predict'],
                    [sys.executable, '--diagnostics'], 'python.exe setup.py']:
        with pytest.raises(RuntimeError, match='external runtimes'):
            guard('subprocess.Popen', (command[0], command, None, None))
    command = [sys.executable, '--cryolo-onnx-worker', '--model', 'fixture.onnx']
    guard('subprocess.Popen', (sys.executable, command, None, None))
    guard('subprocess.Popen', (None, subprocess.list2cmdline(command), None, None))
    with pytest.raises(RuntimeError, match='external runtimes'):
        guard('subprocess.Popen', ('different.exe', subprocess.list2cmdline(command), None, None))


def test_real_audit_hook_blocks_network_and_processes_in_isolated_child():
    # Audit hooks cannot be removed, so exercise real events in a child only.
    code = '''
import importlib.util, os, socket, subprocess, sys
spec = importlib.util.spec_from_file_location('verification_launcher', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
os.environ['SARU_VERIFY_OFFLINE'] = '1'
original = subprocess.Popen
module.enforce_offline_verification()
class ImportCompatiblePopen(subprocess.Popen): pass
assert subprocess.Popen is original
for operation in [lambda: socket.create_connection(('127.0.0.1', 9)),
                  lambda: subprocess.Popen([sys.executable, '-c', 'pass'])]:
    try: operation()
    except RuntimeError as error: assert 'Offline verification forbids' in str(error)
    else: raise AssertionError('Real audited operation unexpectedly succeeded')
'''
    subprocess.run([sys.executable, '-c', code, str(DIRECTORY / 'launcher.py')],
                   check=True, timeout=15, capture_output=True, text=True)


def test_installer_has_no_bootstrap_and_removes_only_obsolete_owned_helpers():
    installer = (DIRECTORY / 'installer.iss').read_text()
    files = installer.split('[Files]', 1)[1].split('[Icons]', 1)[0]
    assert 'cryolo-capsids.onnx' in files
    assert 'cryolo-capsids.h5' not in files
    assert '[Code]' not in installer
    assert 'RuntimeIsReady' not in installer
    run = installer.split('[Run]', 1)[1]
    commands = [line for line in run.splitlines() if line.startswith('Filename:')]
    assert len(commands) == 1 and 'SARU Picker crYOLO.exe' in commands[0]
    delete = installer.split('[InstallDelete]', 1)[1].split('[Run]', 1)[0]
    for name in ('setup-runtime.ps1', 'runtime_health.py', 'runtime-requirements-windows.txt',
                 'Install-crYOLO-Runtime.cmd', 'python.3.7.9.nupkg'):
        assert name not in files
        assert f'Name: "{{app}}\\{name}"' in delete
    assert 'filesandordirs' not in delete
    assert 'cryolo-env' not in delete


def test_public_conv_fixture_executes_real_positive_inference(tmp_path, monkeypatch):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    monkeypatch.syspath_prepend(str(DIRECTORY))
    verification = load_module('offline_fixture_test', 'verify_onnx.py')
    verification.make_fixture(tmp_path)
    report = verification.verify(tmp_path)
    assert report['actual_onnx_inference'] is True
    assert report['detections'] == 1 and report['positive_contours'] >= 1
