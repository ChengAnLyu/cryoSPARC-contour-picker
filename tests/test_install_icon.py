"""Bounded icon parsing and atomic resource updates using neutral fixtures."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib


SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "install_icon.py"
SPEC = importlib.util.spec_from_file_location("saru_install_icon", SCRIPT)
icon_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = icon_tool
SPEC.loader.exec_module(icon_tool)


def png(size=16, color=(45, 125, 180, 200)):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind+payload) & 0xffffffff)
    rows = (b"\0" + bytes(color)*size)*size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def neutral_ico(sizes=(16, 32, 256)):
    return icon_tool.serialize_ico(tuple(icon_tool.IconFrame(size, size, 0, 0, 32, png(size)) for size in sizes))


def synthetic_pe(overlay=b"PYINSTALLER-ARCHIVE\0exact\xffbytes", *, raw_size=512, signed=False):
    # One bounded raw section; all fixtures are inert and never executable.
    data = bytearray(512+raw_size)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<HH", data, 132, 0x8664, 1)
    struct.pack_into("<H", data, 148, 240)
    optional = 152
    struct.pack_into("<H", data, optional, 0x20b)
    struct.pack_into("<I", data, optional+60, 512)
    struct.pack_into("<I", data, optional+108, 16)
    if signed:
        struct.pack_into("<II", data, optional+112+4*8, 1024, 16)
    section = optional+240
    data[section:section+8] = b".rsrc\0\0\0"
    struct.pack_into("<II", data, section+16, raw_size, 512)
    data[512:524] = b"OTHER-BYTES!"
    return bytes(data)+overlay


class ResourceAPI:
    def __init__(self):
        self.resources = {(16, 1, 0): b"existing version metadata"}
        self.calls = []
        self.fail_update = self.fail_commit = self.bad_read = False
        self.source_path = None
        self.after_commit = None

    def begin(self, path):
        self.source_path = Path(path)
        self.calls.append(("begin", str(path)))
        self.pending = dict(self.resources)
        return 987

    def update(self, handle, kind, identifier, language, data):
        self.calls.append(("update", kind, identifier, language))
        if self.fail_update:
            raise OSError("simulated UpdateResourceW failure")
        self.pending[kind, identifier, language] = data

    def end(self, handle, discard):
        self.calls.append(("end", discard))
        if not discard:
            if self.fail_commit:
                raise OSError("simulated EndUpdateResourceW failure")
            self.resources = self.pending
            # Simulate native resource expansion and loss of the old overlay.
            self.source_path.write_bytes(synthetic_pe(b"WRONG-NEW-OVERLAY", raw_size=1024))
            if self.after_commit:
                self.after_commit()

    def load(self, path):
        self.calls.append(("load", str(path)))
        return 123

    def read(self, handle, kind, identifier, language):
        result = self.resources[kind, identifier, language]
        return result[:-1] if self.bad_read and kind == icon_tool.RT_ICON else result

    def free(self, handle):
        self.calls.append(("free", handle))


class IconParsingTests(unittest.TestCase):
    def test_png_alpha_frames_survive_canonical_round_trip_byte_for_byte(self):
        original = neutral_ico()
        frames = icon_tool.parse_ico(original)
        self.assertEqual([frame.width for frame in frames], [16, 32, 256])
        self.assertEqual(frames[-1].data, png(256))
        self.assertEqual(icon_tool.serialize_ico(frames), original)
        self.assertEqual(struct.unpack_from("<B", original, 6+16*2)[0], 0)

    def test_dib_frame_is_supported_without_conversion(self):
        size = 16
        image = struct.pack("<IiiHHIIIIII", 40, size, size*2, 1, 32, 0, size*size*4, 0, 0, 0, 0)
        image += b"\x40\x80\xc0\xff"*(size*size)
        frame = icon_tool.IconFrame(size, size, 0, 1, 32, image)
        self.assertEqual(icon_tool.parse_ico(icon_tool.serialize_ico([frame])), (frame,))

    def test_bad_counts_offsets_sizes_and_overlaps_are_rejected(self):
        for change in (lambda b: struct.pack_into("<H", b, 4, 0),
                       lambda b: struct.pack_into("<H", b, 4, 65),
                       lambda b: struct.pack_into("<I", b, 18, 0),
                       lambda b: struct.pack_into("<I", b, 14, 0),
                       lambda b: struct.pack_into("<I", b, 14, 0xffffffff),
                       lambda b: struct.pack_into("<I", b, 18, len(b)),
                       lambda b: struct.pack_into("<I", b, 34, struct.unpack_from("<I", b, 18)[0])):
            bad = bytearray(neutral_ico())
            change(bad)
            with self.assertRaises(ValueError):
                icon_tool.parse_ico(bytes(bad))

    def test_png_checksum_and_directory_dimension_mismatch_are_rejected(self):
        data = bytearray(neutral_ico((16,)))
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            icon_tool.parse_ico(bytes(data))
        data = bytearray(neutral_ico((16,)))
        data[6] = 17
        with self.assertRaisesRegex(ValueError, "dimensions"):
            icon_tool.parse_ico(bytes(data))


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.target = self.directory / "Picker.exe"
        self.icon = self.directory / "neutral.ico"
        self.target.write_bytes(synthetic_pe())
        self.icon.write_bytes(neutral_ico())
        self.api = ResourceAPI()

    def test_success_preserves_archive_and_other_resources_and_verifies_both_languages(self):
        original = self.target.read_bytes()
        report = icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        layout = icon_tool.read_pe_layout(self.target)
        self.assertEqual(layout.section_end, 1536)
        self.assertEqual(self.target.read_bytes()[layout.section_end:], original[1024:])
        self.assertEqual(self.target.read_bytes()[layout.section_end:], b"PYINSTALLER-ARCHIVE\0exact\xffbytes")
        self.assertEqual(self.api.resources[16, 1, 0], b"existing version metadata")
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["overlay_preserved"])
        self.assertEqual(report["verified_languages"], [0, 1033])
        self.assertEqual(report["source_ico_sha256"], hashlib.sha256(self.icon.read_bytes()).hexdigest())
        self.assertEqual(report["target_sha256"], hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.assertEqual(report["sizes"], [[16, 16], [32, 32], [256, 256]])
        for language in (0, 1033):
            read_back = icon_tool.read_executable_icon(self.target, language, api=self.api)
            self.assertEqual(icon_tool.serialize_ico(read_back), self.icon.read_bytes())
        self.assertEqual(list(self.directory.glob(".saru-icon-*")), [])
        self.assertNotEqual(Path(self.api.calls[0][1]), self.target)

    def test_update_and_commit_failures_discard_staging_without_changing_target(self):
        original = self.target.read_bytes()
        for failure in ("fail_update", "fail_commit"):
            with self.subTest(failure=failure):
                self.api = ResourceAPI()
                setattr(self.api, failure, True)
                with self.assertRaises(OSError):
                    icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
                self.assertIn(("end", True), self.api.calls)
                self.assertEqual(self.target.read_bytes(), original)
                self.assertEqual(list(self.directory.glob(".saru-icon-*")), [])

    def test_verification_failure_and_replace_failure_keep_original_executable(self):
        original = self.target.read_bytes()
        self.api.bad_read = True
        with self.assertRaisesRegex(ValueError, "size"):
            icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.assertEqual(self.target.read_bytes(), original)
        self.api = ResourceAPI()
        with patch.object(icon_tool.os, "replace", side_effect=PermissionError("target running")):
            with self.assertRaises(PermissionError):
                icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(list(self.directory.glob(".saru-icon-*")), [])

    def test_external_target_change_is_not_overwritten(self):
        changed = synthetic_pe(b"external replacement")
        self.api.after_commit = lambda: self.target.write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.assertEqual(self.target.read_bytes(), changed)

    def test_signed_executable_is_rejected_before_any_update(self):
        self.target.write_bytes(synthetic_pe(signed=True))
        with self.assertRaisesRegex(ValueError, "Signed"):
            icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.assertEqual(self.api.calls, [])

    def test_self_target_and_bad_pe_are_rejected(self):
        with patch.object(icon_tool.sys, "executable", str(self.target)):
            with self.assertRaisesRegex(ValueError, "own running"):
                icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.target.write_bytes(b"not an executable")
        with self.assertRaisesRegex(ValueError, "PE executable"):
            icon_tool.update_executable_icon(self.target, self.icon, api=self.api)
        self.assertEqual(self.api.calls, [])

    def test_pe_section_bounds_are_validated(self):
        malformed = bytearray(synthetic_pe())
        struct.pack_into("<I", malformed, 152+240+20, 500000)
        self.target.write_bytes(malformed)
        with self.assertRaisesRegex(ValueError, "section exceeds"):
            icon_tool.read_pe_layout(self.target)

    def test_cli_reports_success_and_error_without_loading_ui(self):
        report_path = self.directory / "report.json"
        arguments = ["--target", str(self.target), "--icon", str(self.icon), "--report", str(report_path)]
        expected = {"status": "ok", "overlay_preserved": True}
        with patch.object(icon_tool, "update_executable_icon", return_value=expected), patch("builtins.print"):
            self.assertEqual(icon_tool.main(arguments), 0)
        self.assertEqual(json.loads(report_path.read_text()), expected)
        with patch.object(icon_tool, "update_executable_icon", side_effect=OSError("Windows failure")), patch("builtins.print"):
            self.assertEqual(icon_tool.main(arguments), 1)
        result = json.loads(report_path.read_text())
        self.assertEqual(result["status"], "error")
        self.assertIn("Windows failure", result["error"])

    def test_report_hardlink_cannot_overwrite_target(self):
        report = self.directory / "aliased-report.json"
        try:
            report.hardlink_to(self.target)
        except OSError as exc:
            self.skipTest(f"Hardlinks unavailable: {exc}")
        original = self.target.read_bytes()
        with patch.object(icon_tool, "update_executable_icon") as update, patch("builtins.print"):
            code = icon_tool.main(["--target", str(self.target), "--icon", str(self.icon), "--report", str(report)])
        self.assertEqual(code, 1)
        update.assert_not_called()
        self.assertEqual(self.target.read_bytes(), original)


class NativeWrapperTests(unittest.TestCase):
    def test_native_resource_reader_preserves_binary_payload_and_limits_size(self):
        payload = b"\0\xff\0PNG-or-DIB-bytes"
        buffer = ctypes.create_string_buffer(payload)
        kernel = Mock()
        kernel.FindResourceExW.return_value = 72
        kernel.SizeofResource.return_value = len(payload)
        kernel.LoadResource.return_value = 99
        kernel.LockResource.return_value = ctypes.addressof(buffer)
        api = icon_tool._Win32Resources(kernel)
        self.assertEqual(api.read(12, 3, 101, 1033), payload)
        kernel.FindResourceExW.assert_called_once_with(12, 3, 101, 1033)
        kernel.SizeofResource.return_value = icon_tool.MAX_FRAME_BYTES+1
        with self.assertRaisesRegex(ValueError, "invalid size"):
            api.read(12, 3, 101, 1033)
        self.assertEqual(kernel.LockResource.call_count, 1)

    def test_native_signatures_preserve_resources_and_map_without_execution(self):
        kernel = Mock()
        kernel.BeginUpdateResourceW.return_value = 987
        kernel.UpdateResourceW.return_value = 1
        kernel.EndUpdateResourceW.return_value = 1
        kernel.LoadLibraryExW.return_value = 123
        kernel.FreeLibrary.return_value = 1
        api = icon_tool._Win32Resources(kernel)
        target = Path("C:/Picker.exe")
        handle = api.begin(target)
        kernel.BeginUpdateResourceW.assert_called_once_with(str(target), False)
        payload = b"\0icon bytes\xff"
        api.update(handle, 3, 101, 1033, payload)
        arguments = kernel.UpdateResourceW.call_args.args
        self.assertEqual(arguments[:4], (987, 3, 101, 1033))
        self.assertEqual(ctypes.string_at(arguments[4], arguments[5]), payload)
        api.end(handle, False)
        kernel.EndUpdateResourceW.assert_called_once_with(987, False)
        api.load(target)
        kernel.LoadLibraryExW.assert_called_once_with(str(target), None, 0x60)
        self.assertIs(kernel.BeginUpdateResourceW.restype, ctypes.c_void_p)

    def test_native_errors_do_not_silently_succeed(self):
        kernel = Mock()
        api = icon_tool._Win32Resources(kernel)
        for name, operation in (
            ("BeginUpdateResourceW", lambda: api.begin("test.exe")),
            ("UpdateResourceW", lambda: api.update(1, 3, 101, 0, b"payload")),
            ("EndUpdateResourceW", lambda: api.end(1, False)),
            ("LoadLibraryExW", lambda: api.load("test.exe")),
            ("FreeLibrary", lambda: api.free(1)),
        ):
            with self.subTest(name=name):
                getattr(kernel, name).return_value = 0
                with self.assertRaisesRegex(OSError, name):
                    operation()


if __name__ == "__main__":
    unittest.main()
