"""PowerShell control-flow tests; real Windows installation is covered by CI.

On Linux, set SARU_POWERSHELL to a pwsh executable to run these with harmless
interpreter doubles. They verify ownership, offline reuse and partial recovery.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def powershell():
    executable = os.environ.get('SARU_POWERSHELL') or shutil.which('pwsh')
    if not executable or sys.platform == 'win32':
        pytest.skip('These shell-interpreter doubles run in optional Linux PowerShell tests.')
    return executable


def run_setup(powershell, root, *args):
    script = Path(__file__).resolve().parents[1] / 'packaging' / 'cryolo' / 'setup-runtime.ps1'
    command = [powershell, '-NoProfile', '-File', str(script), '-RuntimeDirectory', str(root / 'cryolo-env'), *args]
    return subprocess.run(command, env=dict(os.environ, OS='Windows_NT', LOCALAPPDATA=str(root)),
                          text=True, capture_output=True, timeout=30)


def fake_interpreter(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nprintf \'{"status":"ok","cpu_execution":true}\\n\'\n')
    path.chmod(0o755)


def test_healthy_reuse_is_offline(powershell, tmp_path):
    fake_interpreter(tmp_path / 'cryolo-env' / 'Scripts' / 'python.exe')
    result = run_setup(powershell, tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
    assert report['status'] == 'ok' and report['reused'] is True
    assert not (tmp_path / 'downloads').exists()


def test_unowned_partial_directory_is_not_changed(powershell, tmp_path):
    env = tmp_path / 'cryolo-env'
    env.mkdir()
    sentinel = env / 'user-data.txt'
    sentinel.write_text('untouched')
    result = run_setup(powershell, tmp_path)
    assert result.returncode == 1
    report = json.loads((tmp_path / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
    assert 'unrelated installation' in report['error']
    assert sentinel.read_text() == 'untouched'
    assert not (env / 'saru-cryolo-runtime.json').exists()


def test_owned_partial_environment_enters_repair(powershell, tmp_path):
    env = tmp_path / 'cryolo-env'
    env.mkdir()
    (env / 'saru-cryolo-runtime.json').write_text('{"owner":"SARU Picker crYOLO"}')
    sentinel = env / 'keep.txt'
    sentinel.write_text('preserved')
    # A failing explicitly provided interpreter stops before downloads. The report
    # proves the owned partial folder reaches repair instead of the refusal branch.
    base = tmp_path / 'python37'
    base.write_text('#!/bin/sh\nexit 7\n')
    base.chmod(0o755)
    result = run_setup(powershell, tmp_path, '-Python37', str(base))
    assert result.returncode == 1
    report = json.loads((tmp_path / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
    assert report['stage'] == 'python' and 'exit code 7' in report['error']
    assert sentinel.read_text() == 'preserved'


def test_bootstrap_uses_host_modules_with_incompatible_inherited_path(powershell, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'packaging' / 'cryolo' / 'setup-runtime.ps1'
    script = tmp_path / 'setup-runtime.ps1'
    shutil.copyfile(source, script)
    # A deliberately invalid package makes setup exercise Get-FileHash without
    # downloading anything or installing a runtime. The old Windows host resolved
    # Utility against an incompatible inherited PS7 path and failed command lookup.
    (tmp_path / 'python.3.7.9.nupkg').write_bytes(b'checksum-regression-test')
    incompatible = tmp_path / 'foreign-powershell-modules'
    incompatible.mkdir()
    restore_report = tmp_path / 'restored-module-path.txt'
    quoted = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = (f'$env:PSModulePath = {quoted(incompatible)}; '
               f'try {{ & {quoted(script)} -RuntimeDirectory {quoted(tmp_path / "owned" / "cryolo-env")} }} '
               f'finally {{ [IO.File]::WriteAllText({quoted(restore_report)}, $env:PSModulePath) }}; exit $LASTEXITCODE')
    result = subprocess.run([powershell, '-NoProfile', '-Command', command],
        env=dict(os.environ, OS='Windows_NT', LOCALAPPDATA=str(tmp_path)),
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    report = json.loads((tmp_path / 'owned' / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
    assert 'bundled Python package failed its checksum' in report['error']
    assert Path(report['powershell_module_root']) == Path(powershell).resolve().parent / 'Modules'
    assert restore_report.read_text() == str(incompatible)
