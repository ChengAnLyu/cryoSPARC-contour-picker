"""Build-only integration checks for optional installer icon sidecars.

The fixture is generated from simple geometric shapes, not user artwork.
The installed EXE is read as PE resource data without executing arbitrary code.
The separately supplied runtime report proves that the patched application
also started, constructed its native windows, and completed its smoke checks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_report(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _read_report(path: Path) -> dict:
    # PowerShell's UTF-8 writer can include a BOM; Python reports do not.
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in report: {path}")
    return value


def _helper_module():
    path = Path(__file__).resolve().with_name("install_icon.py")
    name = "saru_icon_install_verification_helper"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def make_fixture(directory: Path, baseline_directory: Path) -> dict:
    """Create deterministic, deliberately different nonprivate test artwork."""
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (256, 256), (21, 49, 91, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 231, 231), fill=(211, 47, 130, 255))
    draw.polygon([(128, 43), (218, 207), (38, 207)], fill=(255, 214, 61, 255))
    draw.ellipse((106, 122, 150, 166), fill=(21, 49, 91, 255))
    png = directory / "app-icon.png"
    ico = directory / "app-icon.ico"
    image.save(png, format="PNG")
    sizes = [(n, n) for n in (16, 24, 32, 48, 64, 128, 256)]
    image.save(ico, format="ICO", sizes=sizes)
    hashes = {path.suffix[1:]: _digest(path.read_bytes()) for path in (png, ico)}
    baseline = {kind: _digest((baseline_directory / f"app-icon.{kind}").read_bytes())
                for kind in hashes}
    if any(hashes[kind] == baseline[kind] for kind in hashes):
        raise ValueError("The integration fixture must differ from both bundled icon assets.")
    return {"status": "ok", "purpose": "Nonprivate installer icon override integration test",
            "directory": str(directory.resolve()), "asset_sha256": hashes,
            "baseline_asset_sha256": baseline, "differs_from_bundled": True,
            "sizes": [list(size) for size in sizes]}


def verify_installation(install_directory: Path, fixture_directory: Path,
                        helper_report_path: Path, runtime_report_path: Path) -> dict:
    helper = _helper_module()
    target = (install_directory / "SARU Picker.exe").resolve()
    assets = install_directory / "_internal" / "assets"
    executable_helper = install_directory / "SARU Icon Helper.exe"
    if not target.is_file() or not executable_helper.is_file():
        raise ValueError("The installed picker and independent helper must both exist.")

    asset_hashes = {}
    for suffix in ("png", "ico"):
        expected = (fixture_directory / f"app-icon.{suffix}").read_bytes()
        actual = (assets / f"app-icon.{suffix}").read_bytes()
        if actual != expected:
            raise ValueError(f"The installed {suffix.upper()} does not match the external sidecar.")
        asset_hashes[suffix] = _digest(actual)

    expected_frames = helper.parse_ico((assets / "app-icon.ico").read_bytes())
    canonical_icon = helper.serialize_ico(expected_frames)
    canonical_hash = _digest(canonical_icon)
    installation = _read_report(helper_report_path)
    if installation.get("status") != "ok" or installation.get("overlay_preserved") is not True:
        raise ValueError("The installer helper did not report a successful archive-preserving update.")
    if installation.get("source_ico_sha256") != asset_hashes["ico"]:
        raise ValueError("The installer helper used a different ICO from the sidecar.")
    if installation.get("icon_sha256") != canonical_hash:
        raise ValueError("The installer helper's verified icon differs from the expected icon.")
    if Path(installation.get("target", "")).resolve() != target:
        raise ValueError("The installer helper report refers to a different executable.")
    if installation.get("target_sha256") != _digest(target.read_bytes()):
        raise ValueError("The installed executable changed after the helper verified it.")
    if installation.get("frame_count") != len(expected_frames):
        raise ValueError("The installer helper reported an incorrect icon frame count.")

    languages = (0, 0x0409)
    if sorted(installation.get("verified_languages", [])) != list(languages):
        raise ValueError("The installer helper did not verify both icon resource languages.")
    resource_checks = []
    for language in languages:
        actual_frames = helper.read_executable_icon(target, language=language)
        actual_bytes = helper.serialize_ico(actual_frames)
        if actual_bytes != canonical_icon:
            raise ValueError(f"Installed PE icon resources do not match sidecar (language {language}).")
        resource_checks.append({"language": language, "frame_count": len(actual_frames),
                                "canonical_ico_sha256": _digest(actual_bytes),
                                "matches_sidecar_frame_bytes": True})

    runtime = _read_report(runtime_report_path)
    if runtime.get("status") != "ok" or runtime.get("mode") != "smoke-test":
        raise ValueError("The patched installed application did not complete its runtime smoke test.")
    native = [check for check in runtime.get("checks", [])
              if check.get("name") == "windows_runtime" and check.get("status") == "ok"]
    if len(native) != 1:
        raise ValueError("The installed runtime report lacks its native Windows check.")
    icons = native[0].get("details", {}).get("application_icon", {})
    if not icons.get("small_icon") or not icons.get("large_icon"):
        raise ValueError("The installed application did not expose both native icon handles.")

    return {"status": "ok", "mode": "installed-icon-verification", "target": str(target),
            "nonprivate_sidecar_override_tested": True, "installed_asset_sha256": asset_hashes,
            "helper_report": str(helper_report_path.resolve()),
            "pyinstaller_archive_preserved": True,
            "overlay_bytes": installation.get("overlay_bytes"),
            "overlay_sha256": installation.get("overlay_sha256"),
            "resource_checks": resource_checks,
            "runtime_smoke_passed_after_resource_update": True,
            "native_window_icon_handles": icons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("make-fixture")
    fixture.add_argument("--directory", type=Path, required=True)
    fixture.add_argument("--baseline-directory", type=Path, required=True)
    fixture.add_argument("--report", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--install-directory", type=Path, required=True)
    verify.add_argument("--fixture-directory", type=Path, required=True)
    verify.add_argument("--helper-report", type=Path, required=True)
    verify.add_argument("--runtime-report", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make-fixture":
            result = make_fixture(args.directory, args.baseline_directory)
        else:
            result = verify_installation(args.install_directory, args.fixture_directory,
                                         args.helper_report, args.runtime_report)
        _write_report(args.report, result)
        return 0
    except Exception as exc:
        _write_report(args.report, {"status": "error", "mode": args.command,
                                   "error_type": type(exc).__name__, "error": str(exc)})
        print(f"Icon installation verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
