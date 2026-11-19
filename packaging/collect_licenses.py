"""Collect installed distribution notices alongside a frozen application.

This copies text files identified as license/notice/copying/copyright material
from package metadata and records distribution versions. It does not assign a
license to the user's own application code.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys


def collect(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for distribution in sorted(importlib.metadata.distributions(),
                               key=lambda d: d.metadata.get("Name", "").casefold()):
        name = distribution.metadata.get("Name", "unknown")
        destination = output / re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        copied = []
        for member in distribution.files or []:
            text_path = str(member).replace("\\", "/")
            if not any(re.search(r"(?i)(license|licence|notice|copying|copyright)", part)
                       for part in Path(text_path).parts):
                continue
            source = Path(distribution.locate_file(member)).resolve()
            if not source.is_file() or source.suffix.lower() in {".pyc", ".dll", ".so", ".pyd"}:
                continue
            relative = Path(*[part for part in Path(text_path).parts if part not in {"..", "."}])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(output)))
        records.append({"name": name, "version": distribution.version,
                        "license_metadata": distribution.metadata.get("License-Expression")
                            or distribution.metadata.get("License") or "See included notices or upstream",
                        "project_urls": distribution.metadata.get_all("Project-URL") or [],
                        "notice_files": copied})
    # CPython and Tcl/Tk are not pip distributions but are also bundled.
    base = Path(sys.base_prefix)
    runtime_files = [base / "LICENSE.txt", base / "LICENSE"]
    for candidate in (base / "tcl").glob("*/license.terms") if (base / "tcl").exists() else []:
        runtime_files.append(candidate)
    for source in runtime_files:
        if source.is_file():
            target = output / "Python-runtime" / source.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (output / "distributions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    overview = ["THIRD-PARTY NOTICES — SARU Picker Windows distribution", "",
                "This folder records the dependency versions and copies the license/notice",
                "files distributed with the build environment. Some entries are build tools.",
                "No license for the user's application code is assigned by this report.", "",
                "The bundled original SAM ViT-B model is distributed under Apache-2.0.",
                "Model source: https://github.com/facebookresearch/segment-anything", ""]
    overview.extend(f"{record['name']} {record['version']}" for record in records)
    (output / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(overview) + "\n", encoding="utf-8")
    return {"distributions": len(records), "destination": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(collect(parser.parse_args().output)))
