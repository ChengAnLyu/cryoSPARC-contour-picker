"""Build the readable single-file distribution from tested source modules.

No dynamic import tricks or embedded source strings are used. Each original
module remains a labeled section in main.py; only intra-project imports and
repeated future imports/docstrings are removed.
"""
from __future__ import annotations

import ast
from pathlib import Path


def build(destination: Path | None = None) -> Path:
    directory = Path(__file__).resolve().parent
    destination = destination or directory / "main.py"
    names = ("application_icon", "capsid_prior", "carbon_support", "capsid", "recognition", "cryolo_recognition", "execution", "desktop", "manual_sampling", "micrograph_navigation", "screen_overlay", "screen_app", "app")
    sections = [
        '#!/usr/bin/env python3\n',
        '"""SARU Picker 2.9.0: direct screen contours and model-assisted desktop picking.\n',
        'Generated from the companion tested modules. Run python main.py --help.\n',
        '"""\nfrom __future__ import annotations\n\n',
    ]
    for name in names:
        path = directory / f"{name}.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        removed = set()
        for node in ast.walk(tree):
            skip = isinstance(node, ast.ImportFrom) and (node.module in names or node.module == "__future__")
            skip = skip or (node is tree.body[0] and isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
            if skip:
                removed.update(range(node.lineno, node.end_lineno + 1))
        body = "".join(line for number, line in enumerate(source.splitlines(keepends=True), 1)
                       if number not in removed)
        sections.extend([f"\n# ---------------- {name}.py ----------------\n", body, "\n"])
    text = "".join(sections)
    compile(text, str(destination), "exec")
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


if __name__ == "__main__":
    print(build())
