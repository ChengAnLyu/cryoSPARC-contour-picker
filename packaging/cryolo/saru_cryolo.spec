# -*- mode: python ; coding: utf-8 -*-
"""Separate Windows GUI; proprietary crYOLO remains an external installation."""
from pathlib import Path
import sys
from PyInstaller.utils.hooks import copy_metadata

if sys.platform != "win32":
    raise SystemExit("Build the Windows edition on Windows.")
project = Path(SPECPATH).resolve().parents[1]
icon = project / "assets" / "app-icon.ico"
notices = project / "build" / "cryolo" / "third-party-notices"
datas = [(str(icon), "assets"),
         (str(project / "assets" / "app-icon.png"), "assets"),
         (str(notices), "licenses"),
         (str(project / "packaging" / "cryolo" / "QUICK_START.txt"), ".")]
for name in ("numpy", "Pillow", "scipy", "PyAutoGUI", "pynput"):
    datas += copy_metadata(name)
hiddenimports = ["cryolo_recognition", "pyautogui._pyautogui_win", "pyscreeze", "pygetwindow",
                 "pymsgbox", "pytweening", "mouseinfo", "pyperclip", "pyrect",
                 "pynput.keyboard._win32", "pynput.mouse._win32", "pynput._util.win32",
                 "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox", "tkinter.font",
                 "PIL.ImageTk", "PIL._imagingtk", "scipy.ndimage", "scipy.spatial"]
a = Analysis([str(project / "packaging" / "cryolo" / "launcher.py")],
             pathex=[str(project)], binaries=[], datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=["torch", "torchvision", "segment_anything", "tensorflow", "keras", "cryolo"],
             noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SARU Picker crYOLO",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(icon), uac_admin=False,
          contents_directory="_internal")
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SARU_Picker_crYOLO")
