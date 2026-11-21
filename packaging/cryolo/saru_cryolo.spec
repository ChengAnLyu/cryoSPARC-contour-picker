# -*- mode: python ; coding: utf-8 -*-
"""SARU 3.0: bundled SAM runtime and a separately installed stock crYOLO runtime."""
from pathlib import Path
import hashlib
import json
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

if sys.platform != "win32":
    raise SystemExit("Build the Windows edition on Windows.")
project = Path(SPECPATH).resolve().parents[1]
icon = project / "assets" / "app-icon.ico"
notices = project / "build" / "cryolo" / "third-party-notices"
manifest_path = project / "packaging" / "model-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
sam_model = Path(os.environ.get("SARU_MODEL_FILE", str(project / "models" / manifest["filename"])))
if not sam_model.is_file() or sam_model.stat().st_size != manifest["size_bytes"]:
    raise SystemExit("A complete verified SAM ViT-B model is required for all three SAM profiles.")
with sam_model.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
        raise SystemExit("SAM ViT-B checksum mismatch.")
datas = [(str(icon), "assets"),
         (str(project / "assets" / "app-icon.png"), "assets"),
         (str(sam_model), "models"), (str(manifest_path), "models"),
         (str(notices), "licenses"),
         (str(project / "packaging" / "cryolo" / "QUICK_START.txt"), ".")]
for name in ("numpy", "Pillow", "scipy", "PyAutoGUI", "pynput", "torch", "torchvision", "segment-anything"):
    datas += copy_metadata(name)
hiddenimports = collect_submodules("segment_anything") + [
    "cryolo_recognition", "runtime_management", "pyautogui", "pyautogui._pyautogui_win", "pyscreeze", "pygetwindow",
    "pymsgbox", "pytweening", "mouseinfo", "pyperclip", "pyrect",
    "pynput", "pynput.keyboard", "pynput.mouse", "pynput.keyboard._win32", "pynput.mouse._win32", "pynput._util.win32",
    "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox", "tkinter.font",
    "PIL.ImageTk", "PIL._imagingtk", "scipy.ndimage", "scipy.spatial",
    "torch", "torchvision", "torchvision._C",
]
a = Analysis([str(project / "packaging" / "cryolo" / "launcher.py")],
             pathex=[str(project)], binaries=[], datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=["tensorflow", "keras", "cryolo"], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
# Retain the owned filename so this installer upgrades the previous 3.0 edition.
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SARU Picker crYOLO",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(icon), uac_admin=False,
          version=str(project / "packaging" / "cryolo" / "windows-version.txt"),
          contents_directory="_internal")
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SARU_Picker_crYOLO")
