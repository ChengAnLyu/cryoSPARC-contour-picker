"""Build-only checks for the optional private capsid-profile installer sidecar.

The generated fixture has zero learned weights and one constant output bias.
It contains no micrographs, user labels, or user-trained model parameters.
The independent verification requires actual installed-runtime inference;
finding a JSON file or a successful startup alone cannot pass this gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


PROFILE_NAME = "capsid-profile.json"
PROFILE_FORMAT = "saru-capsid-appearance-v1"
INFERENCE_SHAPE = (96, 128)
FIXTURE_BIAS = 1.0


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_report(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in report: {path}")
    return value


def _write_report(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_fixture(directory: Path) -> dict:
    """Write a deterministic schema-valid fixture whose scores are known."""
    project = Path(__file__).resolve().parents[1]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from capsid_prior import FEATURE_NAMES

    count = len(FEATURE_NAMES)
    profile = {
        "format": PROFILE_FORMAT,
        "feature_names": list(FEATURE_NAMES),
        "mean": [0.0] * count,
        "scale": [1.0] * count,
        "layers": [
            {"weight": [[0.0] * count for _ in range(32)], "bias": [0.0] * 32},
            {"weight": [[0.0] * 32 for _ in range(16)], "bias": [0.0] * 16},
            {"weight": [[0.0] * 16], "bias": [FIXTURE_BIAS]},
        ],
        "training": {"purpose": "Nonprivate deterministic installer inference fixture; not a trained detector."},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROFILE_NAME
    data = (json.dumps(profile, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"status": "ok", "purpose": "Nonprivate capsid profile sidecar integration test",
            "profile_sha256": _digest(data), "profile_bytes": len(data),
            "format": PROFILE_FORMAT, "expected_score": 1.0 / (1.0 + math.exp(-FIXTURE_BIAS)),
            "contains_user_training_data": False}


def verify_installation(install_directory: Path, fixture_directory: Path,
                        runtime_report_path: Path) -> dict:
    installed = (install_directory / "_internal" / "models" / PROFILE_NAME).resolve()
    fixture_bytes = (fixture_directory / PROFILE_NAME).read_bytes()
    installed_bytes = installed.read_bytes()
    if installed_bytes != fixture_bytes:
        raise ValueError("The installed capsid profile does not match the external sidecar.")
    digest = _digest(installed_bytes)
    runtime = _read_report(runtime_report_path)
    if runtime.get("status") != "ok" or runtime.get("mode") != "smoke-test":
        raise ValueError("The installed application did not complete its runtime smoke test.")
    checks = [item for item in runtime.get("checks", [])
              if item.get("name") == "capsid_profile" and item.get("status") == "ok"]
    if len(checks) != 1:
        raise ValueError("The installed runtime report lacks its capsid profile inference check.")
    details = checks[0].get("details", {})
    if (details.get("profile_status") != "loaded" or details.get("fallback_used") is not False
            or details.get("profile_inference_executed") is not True):
        raise ValueError("The installed runtime did not execute the learned profile inference path.")
    if (details.get("profile_sha256") != digest or details.get("format") != PROFILE_FORMAT
            or Path(details.get("profile_path", "")).resolve() != installed):
        raise ValueError("The runtime loaded a different capsid profile or path.")
    inference = details.get("inference", {})
    if (inference.get("score_shape") != list(INFERENCE_SHAPE)
            or inference.get("scored_pixels") != math.prod(INFERENCE_SHAPE)
            or inference.get("input_shape") != [*INFERENCE_SHAPE, 3]):
        raise ValueError("The runtime did not score the complete synthetic profile fixture image.")
    expected = 1.0 / (1.0 + math.exp(-FIXTURE_BIAS))
    for field in ("minimum", "maximum", "mean"):
        value = inference.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, expected, abs_tol=1e-6)):
            raise ValueError("Runtime capsid scores do not match inference through the fixture weights.")
    return {"status": "ok", "mode": "installed-capsid-profile-verification",
            "nonprivate_sidecar_override_tested": True,
            "installed_profile_sha256": digest, "profile_path": str(installed),
            "runtime_loaded_same_profile": True, "runtime_profile_inference_executed": True,
            "expected_constant_score": expected, "verified_inference": inference,
            "runtime_report": str(runtime_report_path.resolve()),
            "note": "Packaging and inference-path validation with synthetic constant weights; not capsid recognition accuracy."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("make-fixture")
    fixture.add_argument("--directory", type=Path, required=True)
    fixture.add_argument("--report", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--install-directory", type=Path, required=True)
    verify.add_argument("--fixture-directory", type=Path, required=True)
    verify.add_argument("--runtime-report", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (make_fixture(args.directory) if args.command == "make-fixture" else
                  verify_installation(args.install_directory, args.fixture_directory, args.runtime_report))
        _write_report(args.report, result)
        return 0
    except Exception as exc:
        _write_report(args.report, {"status": "error", "mode": args.command,
                                   "error_type": type(exc).__name__, "error": str(exc)})
        print(f"Capsid profile installation verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
