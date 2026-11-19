"""Checks that the Windows installer gate cannot accept the wrong icon."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "saru_icon_installation_check", PROJECT / "packaging" / "verify_icon_installation.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class IconInstallationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.fixtures = self.base / "fixture"
        self.install = self.base / "installed"
        assets = self.install / "_internal" / "assets"
        assets.mkdir(parents=True)
        self.fixture_report = check.make_fixture(self.fixtures, PROJECT / "assets")
        for suffix in ("png", "ico"):
            shutil.copy2(self.fixtures / f"app-icon.{suffix}", assets)
        self.target = self.install / "SARU Picker.exe"
        self.target.write_bytes(b"stand-in executable bytes for gate checks")
        (self.install / "SARU Icon Helper.exe").write_bytes(b"helper")
        native_helper = check._helper_module()
        self.frames = native_helper.parse_ico((self.fixtures / "app-icon.ico").read_bytes())
        self.helper = SimpleNamespace(
            parse_ico=native_helper.parse_ico, serialize_ico=native_helper.serialize_ico,
            read_executable_icon=Mock(return_value=self.frames))
        self.helper_report = self.base / "helper.json"
        self.installation = {
            "status": "ok", "overlay_preserved": True,
            "source_ico_sha256": self.fixture_report["asset_sha256"]["ico"],
            "icon_sha256": hashlib.sha256(native_helper.serialize_ico(self.frames)).hexdigest(),
            "target": str(self.target.resolve()),
            "target_sha256": hashlib.sha256(self.target.read_bytes()).hexdigest(),
            "frame_count": len(self.frames), "verified_languages": [0, 1033],
            "overlay_bytes": 100, "overlay_sha256": "test overlay digest",
        }
        self.helper_report.write_text(json.dumps(self.installation))
        self.runtime_report = self.base / "runtime.json"
        self.runtime = {"status": "ok", "mode": "smoke-test", "checks": [
            {"name": "windows_runtime", "status": "ok", "details": {
                "application_icon": {"small_icon": True, "large_icon": True}}}]}
        self.runtime_report.write_text(json.dumps(self.runtime))

    def verify(self):
        with patch.object(check, "_helper_module", return_value=self.helper):
            return check.verify_installation(self.install, self.fixtures,
                                             self.helper_report, self.runtime_report)

    def test_fixture_is_different_from_the_bundled_generic_artwork(self):
        self.assertTrue(self.fixture_report["differs_from_bundled"])
        self.assertNotEqual(self.fixture_report["asset_sha256"]["png"],
                            self.fixture_report["baseline_asset_sha256"]["png"])
        self.assertNotEqual(self.fixture_report["asset_sha256"]["ico"],
                            self.fixture_report["baseline_asset_sha256"]["ico"])
        self.assertEqual([(frame.width, frame.height) for frame in self.frames],
                         [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])

    def test_both_resource_languages_and_patched_application_are_required(self):
        result = self.verify()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["runtime_smoke_passed_after_resource_update"])
        self.helper.read_executable_icon.assert_has_calls([
            call(self.target.resolve(), language=0),
            call(self.target.resolve(), language=1033)])
        self.assertEqual(len(result["resource_checks"]), 2)

    def test_success_report_cannot_hide_the_wrong_native_icon(self):
        self.helper.read_executable_icon.return_value = self.helper.parse_ico(
            (PROJECT / "assets" / "app-icon.ico").read_bytes())
        with self.assertRaisesRegex(ValueError, "PE icon resources do not match"):
            self.verify()

    def test_report_for_a_stale_or_changed_executable_is_rejected(self):
        self.target.write_bytes(b"different executable after reported update")
        with self.assertRaisesRegex(ValueError, "changed after"):
            self.verify()

    def test_missing_sidecar_override_is_rejected(self):
        shutil.copy2(PROJECT / "assets" / "app-icon.png",
                     self.install / "_internal" / "assets" / "app-icon.png")
        with self.assertRaisesRegex(ValueError, "installed PNG"):
            self.verify()

    def test_native_window_without_the_large_icon_does_not_pass(self):
        self.runtime["checks"][0]["details"]["application_icon"]["large_icon"] = False
        self.runtime_report.write_text(json.dumps(self.runtime))
        with self.assertRaisesRegex(ValueError, "both native icon handles"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
