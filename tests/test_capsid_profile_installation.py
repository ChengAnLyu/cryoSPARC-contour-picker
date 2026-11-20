"""Reject sidecar installs that copied bytes but did not execute that model."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import app


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "saru_capsid_profile_installation_check",
    PROJECT / "packaging" / "verify_capsid_profile_installation.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class CapsidProfileInstallationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.fixture = self.base / "fixture"
        self.install = self.base / "installed"
        models = self.install / "_internal" / "models"
        models.mkdir(parents=True)
        self.fixture_report = check.make_fixture(self.fixture)
        self.profile = models / check.PROFILE_NAME
        shutil.copy2(self.fixture / check.PROFILE_NAME, self.profile)
        score = 1 / (1 + math.exp(-1))
        self.details = {
            "profile_status": "loaded", "fallback_used": False,
            "profile_inference_executed": True,
            "profile_path": str(self.profile.resolve()),
            "profile_sha256": self.fixture_report["profile_sha256"],
            "format": check.PROFILE_FORMAT,
            "inference": {"input_shape": [96, 128, 3], "score_shape": [96, 128],
                          "scored_pixels": 96 * 128, "minimum": score,
                          "maximum": score, "mean": score},
        }
        self.runtime = {"status": "ok", "mode": "smoke-test", "checks": [
            {"name": "capsid_profile", "status": "ok", "details": self.details}]}
        self.runtime_path = self.base / "runtime.json"

    def verify(self):
        self.runtime_path.write_text(json.dumps(self.runtime), encoding="utf-8")
        return check.verify_installation(self.install, self.fixture, self.runtime_path)

    def test_matching_sidecar_and_actual_fixture_scores_are_required(self):
        result = self.verify()
        self.assertTrue(result["runtime_loaded_same_profile"])
        self.assertTrue(result["runtime_profile_inference_executed"])
        self.assertEqual(result["installed_profile_sha256"], self.fixture_report["profile_sha256"])
        profile = json.loads((self.fixture / check.PROFILE_NAME).read_text())
        self.assertEqual(profile["layers"][-1]["bias"], [1.0])
        self.assertTrue(all(weight == 0 for layer in profile["layers"]
                            for row in layer["weight"] for weight in row))
        self.assertFalse(self.fixture_report["contains_user_training_data"])

    def test_same_filename_with_different_installed_weights_is_rejected(self):
        self.profile.write_text('{"different":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.verify()

    def test_file_existence_and_loaded_status_without_inference_do_not_pass(self):
        self.details["profile_inference_executed"] = False
        with self.assertRaisesRegex(ValueError, "execute"):
            self.verify()

    def test_runtime_using_different_profile_path_or_digest_is_rejected(self):
        original = self.details.copy()
        for name, value in (("profile_path", str(self.fixture / check.PROFILE_NAME)),
                            ("profile_sha256", "0" * 64)):
            with self.subTest(name=name):
                self.details.clear()
                self.details.update(original, **{name: value})
                with self.assertRaisesRegex(ValueError, "different capsid profile"):
                    self.verify()

    def test_success_report_with_wrong_scores_or_incomplete_map_is_rejected(self):
        for field, value in (("mean", 0.5), ("minimum", float("nan")),
                             ("maximum", True), ("scored_pixels", 0)):
            with self.subTest(field=field):
                original = self.details["inference"][field]
                self.details["inference"][field] = value
                with self.assertRaises(ValueError):
                    self.verify()
                self.details["inference"][field] = original

    def test_installer_profile_is_optional_external_and_installed_beside_sam(self):
        installer = (PROJECT / "packaging" / "installer.iss").read_text(encoding="utf-8")
        files = installer.split("[Files]", 1)[1].split("[Icons]", 1)[0]
        line = next(line for line in files.splitlines()
                    if line.startswith('Source: "{src}\\capsid-profile.json"'))
        self.assertIn('DestDir: "{app}\\_internal\\models"', line)
        flags = set(line.split("Flags:", 1)[1].strip().split())
        self.assertTrue({"external", "ignoreversion", "skipifsourcedoesntexist"} <= flags)


class CapsidProfileRuntimeTests(unittest.TestCase):
    def test_missing_optional_profile_retains_fallback_without_inference(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "load_profile") as load, \
             patch.object(app, "score_image") as score:
            details = app._smoke_capsid_profile(Path(directory) / "model.pth")
        self.assertEqual(details["profile_status"], "missing")
        self.assertTrue(details["fallback_used"])
        self.assertFalse(details["profile_inference_executed"])
        load.assert_not_called()
        score.assert_not_called()

    def test_installed_runtime_scores_the_sidecar_using_real_numpy_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "installed" / "_internal" / "models"
            fixture = check.make_fixture(models)
            details = app._smoke_capsid_profile(models / "model.pth")
            self.assertEqual(details["profile_path"], str((models / check.PROFILE_NAME).resolve()))
            self.assertEqual(details["profile_sha256"], fixture["profile_sha256"])
            self.assertTrue(details["profile_inference_executed"])
            self.assertFalse(details["fallback_used"])
            self.assertEqual(details["inference"]["score_shape"], [96, 128])
            self.assertEqual(details["inference"]["scored_pixels"], 96 * 128)
            for name in ("minimum", "maximum", "mean"):
                self.assertAlmostEqual(details["inference"][name], fixture["expected_score"], places=6)

    def test_corrupt_present_profile_fails_smoke_instead_of_silent_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            (checkpoint.parent / check.PROFILE_NAME).write_text('{"invalid":', encoding="utf-8")
            with patch.object(app.sys, "platform", "linux"), \
                 patch.object(app, "_smoke_image_processing", return_value={}), \
                 patch.object(app, "_smoke_torchvision", return_value={}), \
                 patch.object(app, "_smoke_sam", return_value={}), \
                 self.assertLogs("saru_picker", level="ERROR"):
                report = app.run_smoke_test(checkpoint, Path(directory) / "app.log")
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["checks"][-1]["name"], "capsid_profile")
        self.assertEqual(report["checks"][-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
