# -*- mode: python ; coding: utf-8 -*-
"""Windows x64 one-directory bundle. Run through packaging/build.ps1."""
from pathlib import Path
import hashlib
import json
import os
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

if sys.platform != "win32":
    raise SystemExit("Build this Windows application on Windows; PyInstaller does not cross-compile.")

project = Path(SPECPATH).resolve().parent
manifest_path = project / "packaging" / "model-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
model = Path(os.environ.get("SARU_MODEL_FILE", project / "models" / manifest["filename"]))
if not model.is_file() or model.stat().st_size != manifest["size_bytes"]:
    raise SystemExit("A complete, verified SAM ViT-B checkpoint is required for the installer.")
with model.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
if digest != manifest["sha256"]:
    raise SystemExit("SAM checkpoint checksum mismatch; refusing to bundle different weights.")

notices = project / "build" / "third-party-notices"
if not (notices / "THIRD_PARTY_NOTICES.txt").is_file():
    raise SystemExit("Run collect_licenses.py before creating the frozen distribution.")

datas = [
    (str(model), "models"),
    (str(manifest_path), "models"),
    (str(notices), "licenses"),
    (str(project / "packaging" / "QUICK_START.txt"), "."),
]
for distribution in ("numpy", "Pillow", "scipy", "torch", "torchvision",
                     "segment-anything", "PyAutoGUI", "pynput"):
    datas += copy_metadata(distribution)

# Keep the standard torch/torchvision hooks. They collect native libraries and
# preserve Python sources needed by TorchScript; do not replace them with an
# archive-only hook or exclude torch submodules to reduce size.
hiddenimports = collect_submodules("segment_anything") + [
    "pyautogui", "pyautogui._pyautogui_win", "pyscreeze", "pygetwindow",
    "pymsgbox", "pytweening", "mouseinfo", "pyperclip", "pyrect",
    "pynput", "pynput.keyboard", "pynput.mouse",
    "pynput.keyboard._win32", "pynput.mouse._win32", "pynput._util.win32",
    "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox", "tkinter.font",
    "PIL.ImageTk", "PIL._imagingtk", "torch", "torchvision", "torchvision._C",
    "scipy.ndimage", "scipy.spatial",
]

a = Analysis(
    [str(project / "packaging" / "launcher.py")],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SARU Picker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    version=str(project / "packaging" / "windows-version.txt"),
    uac_admin=False,
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SARU_Picker")
