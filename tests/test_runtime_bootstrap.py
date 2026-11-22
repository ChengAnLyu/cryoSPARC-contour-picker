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


@pytest.mark.parametrize(('limits', 'expected_error'), [
    (['-CommandTimeoutSeconds', '1'], 'exceeded 1 seconds'),
    (['-CommandTimeoutSeconds', '20', '-SetupTimeoutSeconds', '2'], 'setup exceeded its 2-second limit'),
])
def test_native_command_timeout_keeps_live_diagnostics(powershell, tmp_path, limits, expected_error):
    import time
    env = tmp_path / 'cryolo-env'
    env.mkdir()
    (env / 'saru-cryolo-runtime.json').write_text('{"owner":"SARU Picker crYOLO"}')
    base = tmp_path / 'slow-python'
    base.write_text(f'#!/bin/sh\nexec "{sys.executable}" -c "import sys,time; print(\'native startup began\', file=sys.stderr, flush=True); time.sleep(20)"\n')
    base.chmod(0o755)
    started = time.monotonic()
    result = run_setup(powershell, tmp_path, '-Python37', str(base), *limits)
    assert result.returncode == 1
    assert time.monotonic() - started < 10
    report = json.loads((tmp_path / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
    assert expected_error in report['error']
    assert 'native startup began' in Path(report['command']['stderr_log']).read_text()
    assert not (tmp_path / 'runtime-setup-owner.json').exists()


def test_second_setup_fails_quickly_and_preserves_lock_owner(powershell, tmp_path):
    import time
    lock = tmp_path / 'runtime-setup.lock'
    ready = tmp_path / 'holder-ready'
    owner = tmp_path / 'runtime-setup-owner.json'
    owner.write_text('{"pid":12345,"started_utc":"test owner"}')
    quoted = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = (f'$f = [IO.File]::Open({quoted(lock)}, "OpenOrCreate", "ReadWrite", "None"); '
               f'[IO.File]::WriteAllText({quoted(ready)}, "ready"); Start-Sleep -Seconds 20')
    holder = subprocess.Popen([powershell, '-NoProfile', '-Command', command])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists()
        started = time.monotonic()
        result = run_setup(powershell, tmp_path, '-CheckOnly', '-LockTimeoutSeconds', '1')
        assert result.returncode == 1
        assert time.monotonic() - started < 10
        report = json.loads((tmp_path / 'runtime-install-report.json').read_text(encoding='utf-8-sig'))
        assert 'Another crYOLO setup is still running' in report['error']
        assert '12345' in report['error']
        assert owner.exists()
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_bounded_download_aborts_stalled_local_server(powershell, tmp_path):
    import http.server
    import threading
    import time
    class SlowServer(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '1000000')
            self.end_headers()
            self.wfile.write(b'x')
            self.wfile.flush()
            time.sleep(4)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), SlowServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = Path(__file__).resolve().parents[1] / 'packaging' / 'cryolo' / 'setup-runtime.ps1'
    quoted = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = (f'$tokens=$null; $errors=$null; $ast=[System.Management.Automation.Language.Parser]::ParseFile({quoted(source)},[ref]$tokens,[ref]$errors); '
        '$functions=$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false); '
        'foreach($f in $functions) { if($f.Name -in @("Assert-SetupTime","Save-Report","Get-BoundedDownload")) { Invoke-Expression $f.Extent.Text } }; '
        f'$ParentDirectory={quoted(tmp_path)}; $Report={quoted(tmp_path / "download-report.json")}; '
        '$SetupTimeoutSeconds=5; $script:SetupDeadline=[DateTime]::UtcNow.AddSeconds(5); $InstallReport=@{stage="download-test"}; '
        f'try {{ Get-BoundedDownload "http://127.0.0.1:{server.server_port}/slow" {quoted(tmp_path / "partial.download")} -TimeoutSeconds 1; exit 0 }} '
        'catch { Write-Output $_.Exception.Message; exit 1 }')
    try:
        started = time.monotonic()
        result = subprocess.run([powershell, '-NoProfile', '-Command', command], capture_output=True, text=True, timeout=10)
        assert result.returncode == 1
        assert 'Download exceeded 1 seconds' in result.stdout
        assert time.monotonic() - started < 8
        report = json.loads((tmp_path / 'download-report.json').read_text(encoding='utf-8-sig'))
        assert report['download']['url'].endswith('/slow')
    finally:
        server.shutdown()
        server.server_close()
