"""Install an ICO into an unsigned Windows executable without running it.

This small stdlib-only helper runs after installation, separately from the
picker. Resource edits happen on a same-directory temporary copy. The original
PE overlay (including PyInstaller's appended archive) is copied back verbatim,
then resources and the overlay are verified before atomic replacement.

Win32 contracts: BeginUpdateResourceW(False) preserves other resources;
UpdateResourceW queues changes; EndUpdateResourceW commits or discards them.
RT_GROUP_ICON uses 14-byte entries with WORD image IDs instead of ICO offsets.
Reference: https://learn.microsoft.com/windows/win32/menurc/resource-file-formats
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib


MAX_ICON_BYTES = 16 * 1024 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_ICON_FRAMES = 64
RESOURCE_LANGUAGES = (0, 0x0409)
RT_ICON = 3
RT_GROUP_ICON = 14
GROUP_ID = 1
FIRST_IMAGE_ID = 101


@dataclass(frozen=True)
class IconFrame:
    width: int
    height: int
    color_count: int
    planes: int
    bit_count: int
    data: bytes


@dataclass(frozen=True)
class PELayout:
    section_end: int
    file_size: int
    checksum_offset: int
    signed: bool


def _validate_image(data: bytes, width: int, height: int) -> None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        offset, saw_header, saw_data, saw_end = 8, False, False, False
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("Truncated PNG chunk in ICO frame.")
            size = struct.unpack_from(">I", data, offset)[0]
            end = offset + 12 + size
            if end > len(data):
                raise ValueError("PNG chunk exceeds the ICO frame.")
            kind = data[offset+4:offset+8]
            payload = data[offset+8:offset+8+size]
            expected_crc = struct.unpack_from(">I", data, offset+8+size)[0]
            if zlib.crc32(kind + payload) & 0xffffffff != expected_crc:
                raise ValueError("Invalid PNG checksum in ICO frame.")
            if not saw_header:
                if kind != b"IHDR" or size != 13:
                    raise ValueError("An ICO PNG frame must begin with IHDR.")
                w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
                legal_depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
                if ((w, h) != (width, height) or depth not in legal_depths.get(color, ())
                        or compression or filtering or interlace not in (0, 1)):
                    raise ValueError("Invalid or mismatched PNG dimensions/format in ICO frame.")
                saw_header = True
            elif kind == b"IHDR":
                raise ValueError("Duplicate PNG header in ICO frame.")
            if kind == b"IDAT":
                saw_data = True
            if kind == b"IEND":
                if size or end != len(data):
                    raise ValueError("Invalid PNG end chunk in ICO frame.")
                saw_end = True
            offset = end
        if not (saw_header and saw_data and saw_end):
            raise ValueError("Incomplete PNG image in ICO frame.")
        return
    if len(data) < 40:
        raise ValueError("ICO frame must contain a PNG or Windows DIB image.")
    header, w, h, planes, bits, compression = struct.unpack_from("<IiiHHI", data)
    if (header not in (40, 108, 124) or header > len(data) or (w, h) != (width, height*2)
            or planes != 1 or bits not in (1, 4, 8, 16, 24, 32) or compression not in (0, 3)):
        raise ValueError("Invalid or unsupported DIB image in ICO frame.")
    colors_used = struct.unpack_from("<I", data, 32)[0]
    palette = colors_used or (1 << bits if bits <= 8 else 0)
    if palette > 256:
        raise ValueError("DIB icon palette is too large.")
    masks = 12 if compression == 3 and header == 40 else 0
    minimum = header + masks + 4*palette + ((width*bits+31)//32)*4*height
    if bits != 32:
        minimum += ((width+31)//32)*4*height
    if len(data) < minimum:
        raise ValueError("Truncated DIB image in ICO frame.")


def parse_ico(data: bytes) -> tuple[IconFrame, ...]:
    """Parse bounded ICO directory entries, preserving every image byte."""
    if not isinstance(data, bytes) or not 6 <= len(data) <= MAX_ICON_BYTES:
        raise ValueError("ICO file must contain 6 bytes to 16 MiB.")
    reserved, kind, count = struct.unpack_from("<HHH", data)
    directory_end = 6 + 16*count
    if reserved or kind != 1 or not 1 <= count <= MAX_ICON_FRAMES or directory_end > len(data):
        raise ValueError("Invalid ICO header or image count.")
    result, ranges = [], []
    for index in range(count):
        width, height, colors, reserved, planes, bits, size, offset = struct.unpack_from("<BBBBHHII", data, 6+16*index)
        if (reserved or planes not in (0, 1) or bits not in (0, 1, 4, 8, 16, 24, 32)
                or not 1 <= size <= MAX_FRAME_BYTES or offset < directory_end or offset+size > len(data)):
            raise ValueError("Invalid ICO image entry or image bounds.")
        if any(offset < end and start < offset+size for start, end in ranges):
            raise ValueError("ICO image payloads overlap.")
        ranges.append((offset, offset+size))
        image = data[offset:offset+size]
        width, height = width or 256, height or 256
        _validate_image(image, width, height)
        result.append(IconFrame(width, height, colors, planes, bits, image))
    return tuple(result)


def serialize_ico(frames) -> bytes:
    """Reconstruct canonical ICO bytes for resource/source comparisons."""
    frames = tuple(frames)
    if not 1 <= len(frames) <= MAX_ICON_FRAMES:
        raise ValueError("Invalid number of icon frames.")
    directory, payloads = [struct.pack("<HHH", 0, 1, len(frames))], []
    offset = 6 + 16*len(frames)
    for frame in frames:
        if not (1 <= frame.width <= 256 and 1 <= frame.height <= 256):
            raise ValueError("ICO dimensions must be between 1 and 256 pixels.")
        directory.append(struct.pack("<BBBBHHII", frame.width % 256, frame.height % 256,
                                     frame.color_count, 0, frame.planes, frame.bit_count,
                                     len(frame.data), offset))
        payloads.append(frame.data)
        offset += len(frame.data)
    data = b"".join(directory + payloads)
    parse_ico(data)
    return data


def _group_icon(frames) -> bytes:
    entries = [struct.pack("<HHH", 0, 1, len(frames))]
    for index, frame in enumerate(frames):
        entries.append(struct.pack("<BBBBHHIH", frame.width % 256, frame.height % 256,
                                   frame.color_count, 0, frame.planes, frame.bit_count,
                                   len(frame.data), FIRST_IMAGE_ID+index))
    return b"".join(entries)


def read_pe_layout(path) -> PELayout:
    """Locate raw PE sections and the appended overlay without loading code."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("Target is not a Windows PE executable.")
        pe_offset = struct.unpack_from("<I", header, 60)[0]
        if pe_offset < 64 or pe_offset > min(size-24, 16*1024*1024):
            raise ValueError("Invalid PE header offset.")
        stream.seek(pe_offset)
        coff = stream.read(24)
        if len(coff) != 24 or coff[:4] != b"PE\0\0":
            raise ValueError("Invalid PE signature.")
        sections = struct.unpack_from("<H", coff, 6)[0]
        optional_size = struct.unpack_from("<H", coff, 20)[0]
        if not 1 <= sections <= 96 or not 96 <= optional_size <= 4096:
            raise ValueError("Invalid PE section count or optional header size.")
        optional = stream.read(optional_size)
        if len(optional) != optional_size:
            raise ValueError("Truncated PE optional header.")
        magic = struct.unpack_from("<H", optional)[0]
        directory_offset = 96 if magic == 0x10b else 112 if magic == 0x20b else 0
        if not directory_offset or optional_size < directory_offset:
            raise ValueError("Unsupported PE optional header.")
        directory_count = struct.unpack_from("<I", optional, directory_offset-4)[0]
        if directory_count > (optional_size-directory_offset)//8:
            raise ValueError("PE data directories exceed the optional header.")
        signed = False
        if directory_count > 4:
            security_offset, security_size = struct.unpack_from("<II", optional, directory_offset+4*8)
            signed = bool(security_offset or security_size)
        section_table_end = pe_offset+24+optional_size+40*sections
        section_end = struct.unpack_from("<I", optional, 60)[0]
        if not section_table_end <= section_end <= size:
            raise ValueError("Invalid PE header/section table bounds.")
        for _ in range(sections):
            section = stream.read(40)
            if len(section) != 40:
                raise ValueError("Truncated PE section table.")
            raw_size, raw_offset = struct.unpack_from("<II", section, 16)
            if raw_size:
                if raw_offset < section_table_end or raw_offset+raw_size > size:
                    raise ValueError("PE section exceeds the executable file.")
                section_end = max(section_end, raw_offset+raw_size)
    return PELayout(section_end, size, pe_offset+24+64, signed)


def _hash_file(path, offset=0) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Win32Resources:
    """Pointer-safe resource APIs; construction alone requires Windows."""
    def __init__(self, kernel32=None):
        if kernel32 is None:
            if os.name != "nt":
                raise OSError("Executable icon installation requires Windows.")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32 = kernel32
        p, u32, u16, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_int
        signatures = {
            "BeginUpdateResourceW": ([ctypes.c_wchar_p, boolean], p),
            "UpdateResourceW": ([p, p, p, u16, p, u32], boolean),
            "EndUpdateResourceW": ([p, boolean], boolean),
            "LoadLibraryExW": ([ctypes.c_wchar_p, p, u32], p),
            "FindResourceExW": ([p, p, p, u16], p),
            "SizeofResource": ([p, p], u32),
            "LoadResource": ([p, p], p),
            "LockResource": ([p], p),
            "FreeLibrary": ([p], boolean),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(kernel32, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def _failed(operation):
        code = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
        raise OSError(code, f"{operation} failed (Windows error {code}).")

    def begin(self, path):
        handle = self.kernel32.BeginUpdateResourceW(str(path), False)
        if not handle:
            self._failed("BeginUpdateResourceW")
        return handle

    def update(self, handle, kind, identifier, language, data):
        buffer = ctypes.create_string_buffer(data)
        if not self.kernel32.UpdateResourceW(handle, kind, identifier, language, buffer, len(data)):
            self._failed("UpdateResourceW")

    def end(self, handle, discard):
        if not self.kernel32.EndUpdateResourceW(handle, bool(discard)):
            self._failed("EndUpdateResourceW")

    def load(self, path):
        # These flags map only resources: no entry point or imports execute.
        handle = self.kernel32.LoadLibraryExW(str(path), None, 0x40 | 0x20)
        if not handle:
            self._failed("LoadLibraryExW")
        return handle

    def read(self, handle, kind, identifier, language):
        info = self.kernel32.FindResourceExW(handle, kind, identifier, language)
        if not info:
            self._failed("FindResourceExW")
        size = self.kernel32.SizeofResource(handle, info)
        if not 1 <= size <= MAX_FRAME_BYTES:
            raise ValueError("Executable icon resource has an invalid size.")
        loaded = self.kernel32.LoadResource(handle, info)
        if not loaded:
            self._failed("LoadResource")
        address = self.kernel32.LockResource(loaded)
        if not address:
            self._failed("LockResource")
        return ctypes.string_at(address, size)

    def free(self, handle):
        if not self.kernel32.FreeLibrary(handle):
            self._failed("FreeLibrary")


def read_executable_icon(target, language=0, *, api=None) -> tuple[IconFrame, ...]:
    """Read icon group 1 as data, never execute or initialize the EXE."""
    target = Path(target).resolve(strict=True)
    if not target.is_file() or target.suffix.lower() != ".exe":
        raise ValueError("Icon target must be an existing .exe file.")
    if isinstance(language, bool) or not isinstance(language, int) or not 0 <= language <= 65535:
        raise ValueError("Invalid resource language.")
    api = api or _Win32Resources()
    handle = api.load(target)
    try:
        group = api.read(handle, RT_GROUP_ICON, GROUP_ID, language)
        if len(group) < 6:
            raise ValueError("Truncated executable icon group.")
        reserved, kind, count = struct.unpack_from("<HHH", group)
        if reserved or kind != 1 or not 1 <= count <= MAX_ICON_FRAMES or len(group) != 6+14*count:
            raise ValueError("Invalid executable icon group.")
        frames = []
        for index in range(count):
            w, h, colors, reserved, planes, bits, size, identifier = struct.unpack_from("<BBBBHHIH", group, 6+14*index)
            if reserved or not identifier or not 1 <= size <= MAX_FRAME_BYTES:
                raise ValueError("Invalid executable icon group entry.")
            payload = api.read(handle, RT_ICON, identifier, language)
            if len(payload) != size:
                raise ValueError("Executable icon image size does not match its group entry.")
            frames.append(IconFrame(w or 256, h or 256, colors, planes, bits, payload))
        frames = tuple(frames)
        serialize_ico(frames)  # Validate payloads and directory metadata too.
        return frames
    finally:
        api.free(handle)


def _update_resources(path, frames, api):
    handle = api.begin(path)
    committed = False
    try:
        group = _group_icon(frames)
        for language in RESOURCE_LANGUAGES:
            for index, frame in enumerate(frames):
                api.update(handle, RT_ICON, FIRST_IMAGE_ID+index, language, frame.data)
            api.update(handle, RT_GROUP_ICON, GROUP_ID, language, group)
        api.end(handle, False)
        committed = True
    finally:
        if not committed:
            try:
                api.end(handle, True)
            except OSError:
                # The original target is untouched; failed staging is removed.
                pass


def update_executable_icon(target, icon, *, api=None) -> dict:
    """Update only a staged copy, validate it, then replace the original atomically."""
    target, icon = Path(target).resolve(strict=True), Path(icon).resolve(strict=True)
    if not target.is_file() or target.suffix.lower() != ".exe":
        raise ValueError("Icon target must be an existing .exe file.")
    if target.samefile(Path(sys.executable)):
        raise ValueError("The icon helper cannot modify its own running executable.")
    if not icon.is_file() or icon.suffix.lower() != ".ico" or not 6 <= icon.stat().st_size <= MAX_ICON_BYTES:
        raise ValueError("Icon source must be an existing ICO file of at most 16 MiB.")
    icon_bytes = icon.read_bytes()
    frames = parse_ico(icon_bytes)
    canonical_icon = serialize_ico(frames)
    original = read_pe_layout(target)
    if original.signed:
        raise ValueError("Signed executables cannot be modified without invalidating their signature.")
    original_digest = _hash_file(target)
    overlay_digest = _hash_file(target, original.section_end)
    api = api or _Win32Resources()
    descriptor, temporary = tempfile.mkstemp(prefix=".saru-icon-", suffix=".exe", dir=str(target.parent))
    os.close(descriptor)
    staged = Path(temporary)
    try:
        shutil.copy2(target, staged)
        _update_resources(staged, frames, api)
        updated = read_pe_layout(staged)
        if _hash_file(target) != original_digest:
            raise RuntimeError("Target executable changed during icon installation; original file was not replaced.")
        # Win32 resource rewriting may move/drop the appended CArchive. Its
        # original exact bytes belong after the updated last raw PE section.
        with staged.open("r+b") as output, target.open("rb") as source:
            output.truncate(updated.section_end)
            output.seek(updated.section_end)
            source.seek(original.section_end)
            shutil.copyfileobj(source, output, length=1024*1024)
            output.seek(updated.checksum_offset)
            output.write(b"\0\0\0\0")  # Unsigned application; old checksum is stale.
            output.flush()
            os.fsync(output.fileno())
        verified = read_pe_layout(staged)
        if (verified.file_size-verified.section_end != original.file_size-original.section_end
                or _hash_file(staged, verified.section_end) != overlay_digest):
            raise RuntimeError("Executable archive overlay was not preserved.")
        for language in RESOURCE_LANGUAGES:
            if serialize_ico(read_executable_icon(staged, language, api=api)) != canonical_icon:
                raise RuntimeError(f"Installed icon verification failed for resource language {language}.")
        if _hash_file(target) != original_digest:
            raise RuntimeError("Target executable changed during icon installation; original file was not replaced.")
        staged_digest = _hash_file(staged)
        os.replace(staged, target)
        return {
            "status": "ok", "target": str(target), "icon": str(icon),
            "source_ico_sha256": hashlib.sha256(icon_bytes).hexdigest(),
            "icon_sha256": hashlib.sha256(canonical_icon).hexdigest(),
            "frame_count": len(frames), "sizes": [[frame.width, frame.height] for frame in frames],
            "resource_languages": list(RESOURCE_LANGUAGES), "verified_languages": list(RESOURCE_LANGUAGES),
            "overlay_preserved": True, "overlay_bytes": original.file_size-original.section_end,
            "overlay_sha256": overlay_digest, "target_sha256": staged_digest,
        }
    finally:
        staged.unlink(missing_ok=True)


def _report_path(path, target, icon):
    path = Path(path).resolve()
    protected = (Path(target).resolve(), Path(icon).resolve(), Path(sys.executable).resolve())
    if (path.suffix.lower() != ".json" or path in protected
            or (path.exists() and any(other.exists() and path.samefile(other) for other in protected))):
        raise ValueError("Report must be a separate JSON file.")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install the SARU executable icon without launching the picker.")
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--icon", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report_path = _report_path(args.report, args.target, args.icon)
        report = update_executable_icon(args.target, args.icon)
        code = 0
    except Exception as exc:
        report = {"status": "error", "error": str(exc), "error_type": type(exc).__name__,
                  "target": str(args.target), "icon": str(args.icon)}
        code = 1
    try:
        # Never overwrite a target/icon on a malformed --report argument.
        report_path = _report_path(args.report, args.target, args.icon)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    except Exception as exc:
        report["report_error"] = str(exc)
        code = 1
    if sys.stdout is not None:
        print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
