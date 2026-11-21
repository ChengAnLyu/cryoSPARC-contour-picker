"""Application identity and artwork shared by the source and frozen GUIs.

Call these functions on the GUI thread. Windows uses the bundled multi-size
ICO directly, so setting an icon does not retain a Python/Tk image object.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
import sys


APPLICATION_ICON_LOG = logging.getLogger(__name__)
APPLICATION_USER_MODEL_ID = "SARU.Picker.Desktop"


def application_icon_paths() -> tuple[Path, Path]:
    """Locate artwork alongside source code or in PyInstaller's data folder."""
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return base / "assets" / "app-icon.ico", base / "assets" / "app-icon.png"


def configure_application_identity(application_id: str = APPLICATION_USER_MODEL_ID) -> bool:
    """Give source and installed Windows launches the same taskbar identity.

    Invoke before creating the first Tk window. Failure to set this optional
    Windows hint must not prevent particle picking from starting.
    """
    if sys.platform != "win32":
        return False
    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        setter = shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        result = setter(application_id)
        if result != 0:
            raise OSError(f"SetCurrentProcessExplicitAppUserModelID returned {result:#x}.")
        return True
    except Exception:
        APPLICATION_ICON_LOG.warning("Could not set the Windows taskbar identity.", exc_info=True)
        return False


def apply_application_icon(root) -> bool:
    """Set this window's icon and the default for its future child windows.

    A plain copied standalone script may not include assets. That remains a
    supported launch: missing artwork does not interrupt the application.
    The Windows packaging contract, in contrast, requires both icon assets.
    """
    ico, png = application_icon_paths()
    # A plain string records successful custom-artwork setup without retaining
    # a Tcl image. Native smoke checks must not accept an unrelated class icon.
    root._saru_application_icon_path = None
    if sys.platform == "win32" and ico.is_file():
        try:
            # Tk's -default path changes the native window *class* icons.
            # Set this root explicitly first (WM_SETICON); this also creates
            # its wrapper before setting the defaults for future toplevels.
            root.iconbitmap(str(ico))
            root.iconbitmap(default=str(ico))
            root._saru_application_icon_path = str(ico)
            return True
        except Exception:
            APPLICATION_ICON_LOG.warning("Could not set the ICO window icon; trying PNG.", exc_info=True)
    if not png.is_file():
        APPLICATION_ICON_LOG.warning("Application icon assets are unavailable in %s.", png.parent)
        return False
    try:
        import tkinter as tk

        photo = tk.PhotoImage(master=root, file=str(png))
        try:
            # Tk snapshots the pixels here; no retained PhotoImage is needed.
            # Its destructor therefore runs now, on this same GUI thread.
            root.iconphoto(True, photo)
        finally:
            del photo
        root._saru_application_icon_path = str(png)
        return True
    except Exception:
        APPLICATION_ICON_LOG.warning("Could not set the application window icon.", exc_info=True)
        return False


def windows_application_icon_status(root) -> dict[str, bool]:
    """Verify custom setup and real titlebar/taskbar icons on Windows.

    This intentionally checks the native top-level wrapper rather than Tk's
    child drawing window. WM_GETICON returns zero for class-only icons, so
    follow the documented GetClassLongPtr fallback (also used by Tk itself).
    Both outcomes require successful custom-artwork setup on this root; an
    untouched Python/Tk class icon is not evidence of application artwork.
    No icon is changed and no window is activated.
    """
    if sys.platform != "win32":
        raise RuntimeError("Native application icon inspection requires Windows.")
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    class_long = getattr(user32, "GetClassLongPtrW", None)
    if class_long is None:
        class_long = user32.GetClassLongW
    class_long.argtypes = [wintypes.HWND, ctypes.c_int]
    class_long.restype = ctypes.c_size_t
    child = root.winfo_id()
    handle = user32.GetAncestor(child, 2) or child  # GA_ROOT
    expected_paths = {str(path) for path in application_icon_paths() if path.is_file()}
    custom_applied = getattr(root, "_saru_application_icon_path", None) in expected_paths
    # WM_GETICON: ICON_SMALL=0, ICON_BIG=1, ICON_SMALL2=2. The last
    # fallbacks read GCLP_HICONSM=-34 and GCLP_HICON=-14 respectively.
    small = (user32.SendMessageW(handle, 0x007F, 0, 0)
             or user32.SendMessageW(handle, 0x007F, 2, 0)
             or class_long(handle, -34))
    large = user32.SendMessageW(handle, 0x007F, 1, 0) or class_long(handle, -14)
    return {
        "small_icon": bool(custom_applied and small),
        "large_icon": bool(custom_applied and large),
    }
