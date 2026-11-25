#!/usr/bin/env python3
"""
Drives the PyInstaller onedir bundle's --self-check and --smoke modes
(packaging/entry_frozen.py) and reports their results. Build the bundle
first:

  .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
  .venv/bin/python3 tools/verify_frozen_build.py

--self-check exists because missing bundled data is the dominant freeze bug
class, and it fails silently otherwise. --smoke goes further and actually
constructs a QApplication + ViewerWindow, catching a missing/unloadable Qt
platform plugin that --self-check can't see. Both run headless: --self-check
touches no Qt object at all, and --smoke forces QT_QPA_PLATFORM=offscreen.

On Windows it also reads DEscape.exe's own icon resources (the first
RT_GROUP_ICON, the one Explorer and the taskbar show) and requires its
256x256 frame's pixels to equal descape/app_icon.png's. PyInstaller's default
icon has a 256 px frame too, so a size check alone would pass a build that
silently kept the default.
"""

from __future__ import annotations

import io
import struct
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
APP_ICON = ROOT / "descape" / "app_icon.png"

RT_ICON = 3
RT_GROUP_ICON = 14
_GRPICONDIR = struct.Struct("<HHH")
_GRPICONDIRENTRY_SIZE = 14  # BYTE x4, WORD planes, WORD bits, DWORD size, WORD nId
_ICONDIRENTRY_SIZE = 16


def _bundle_executable() -> Path:
    if sys.platform == "win32":
        return ROOT / "dist" / "DEscape" / "DEscape.exe"
    return ROOT / "dist" / "DEscape" / "DEscape"


def _run_mode(exe: Path, mode: str) -> int:
    print(f"=== {mode} ===")
    # check=False: this function reports the failure and returns the exit code.
    result = subprocess.run([str(exe), mode], capture_output=True, text=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"verify_frozen_build: {mode} FAILED (exit {result.returncode})")
    else:
        print(f"verify_frozen_build: {mode} PASSED")
    return result.returncode


def expected_icon_rgba(png_path: Path = APP_ICON) -> Image.Image:
    """The 256 px frame Pillow's ICO writer (PyInstaller's PNG conversion) makes from png_path, as RGBA."""
    with Image.open(png_path) as src:
        im = src.convert("RGBA") if src.mode == "P" and "transparency" in src.info else src.copy()
    if im.size != (256, 256):
        im.thumbnail((256, 256), Image.Resampling.LANCZOS, reducing_gap=None)
    return im.convert("RGBA")


def _decode_icon_frame(entry: bytes, data: bytes) -> Image.Image:
    # A one-image ICO around the resource, so Pillow decodes PNG and headerless DIB frames alike.
    ico = _GRPICONDIR.pack(0, 1, 1) + entry[:12] + struct.pack("<I", 6 + _ICONDIRENTRY_SIZE) + data
    with Image.open(io.BytesIO(ico)) as im:
        im.load()
        return im.convert("RGBA")


def check_icon_resources(group: bytes | None, icons: Mapping[int, bytes],
                         expected: Image.Image) -> list[str]:
    """Failures for one RT_GROUP_ICON directory plus its RT_ICON data by id; empty means it passed."""
    if group is None:
        return ["no RT_GROUP_ICON resource: the exe has no icon"]
    if len(group) < _GRPICONDIR.size:
        return [f"RT_GROUP_ICON is {len(group)} bytes, too short for its header"]
    _reserved, kind, count = _GRPICONDIR.unpack_from(group)
    if kind != 1 or count == 0:
        return [f"RT_GROUP_ICON header has type {kind}, {count} entries (want type 1, 1 or more)"]
    if len(group) < _GRPICONDIR.size + count * _GRPICONDIRENTRY_SIZE:
        return [f"RT_GROUP_ICON is {len(group)} bytes, too short for {count} entries"]
    failures: list[str] = []
    frames: list[tuple[int, Image.Image]] = []
    for i in range(count):
        start = _GRPICONDIR.size + i * _GRPICONDIRENTRY_SIZE
        entry = group[start:start + _GRPICONDIRENTRY_SIZE]
        (icon_id,) = struct.unpack_from("<H", entry, 12)
        data = icons.get(icon_id)
        if data is None:
            failures.append(f"RT_ICON {icon_id} named by the group is missing")
            continue
        try:
            frames.append((icon_id, _decode_icon_frame(entry, data)))
        except Exception as exc:
            failures.append(f"RT_ICON {icon_id} does not decode: {exc}")
    # By decoded size, not the entry's width byte: every Pillow-written frame is a PNG.
    big = [(icon_id, im) for icon_id, im in frames if im.size == (256, 256)]
    if not big:
        sizes = sorted({im.size for _, im in frames})
        failures.append(f"no 256x256 icon frame (decoded sizes: {sizes})")
    want = expected.convert("RGBA").tobytes()
    for icon_id, im in big:
        if im.tobytes() != want:
            failures.append(f"RT_ICON {icon_id} (256x256) pixels differ from {APP_ICON.name}: "
                            "the exe kept another icon, maybe PyInstaller's default")
    return failures


def _load_exe_icon_resources(exe: Path) -> tuple[bytes | None, dict[int, bytes]]:
    """The exe's first RT_GROUP_ICON directory and every integer-named RT_ICON. Windows only."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    k32.LoadLibraryExW.restype = wintypes.HMODULE
    k32.FreeLibrary.argtypes = [wintypes.HMODULE]
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMODULE, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_ssize_t)
    k32.EnumResourceNamesW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, enum_proc, ctypes.c_ssize_t]
    k32.EnumResourceNamesW.restype = wintypes.BOOL
    k32.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    k32.FindResourceW.restype = ctypes.c_void_p
    k32.SizeofResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
    k32.SizeofResource.restype = wintypes.DWORD
    k32.LoadResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
    k32.LoadResource.restype = ctypes.c_void_p
    k32.LockResource.argtypes = [ctypes.c_void_p]
    k32.LockResource.restype = ctypes.c_void_p

    load_as_datafile = 0x2 | 0x20  # LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE
    module = k32.LoadLibraryExW(str(exe), None, load_as_datafile)
    if not module:
        raise OSError(f"LoadLibraryExW({exe}) failed: WinError {ctypes.get_last_error()}")

    def names(kind: int) -> list[int | str]:
        found: list[int | str] = []

        def collect(_module, _kind, name, _param):
            # IS_INTRESOURCE: an id fits in the low word, anything else points at a string.
            found.append(name if name >> 16 == 0 else ctypes.wstring_at(name))
            return True

        # FALSE with ERROR_RESOURCE_TYPE_NOT_FOUND (1813) just means none of this type.
        if not k32.EnumResourceNamesW(module, kind, enum_proc(collect), 0) and not found:
            error = ctypes.get_last_error()
            if error != 1813:
                raise OSError(f"EnumResourceNamesW(type {kind}) failed: WinError {error}")
        return found

    def load(kind: int, name: int | str) -> bytes | None:
        key = ctypes.c_wchar_p(name) if isinstance(name, str) else ctypes.c_void_p(name)
        info = k32.FindResourceW(module, key, kind)
        handle = k32.LoadResource(module, info) if info else None
        data = k32.LockResource(handle) if handle else None
        return ctypes.string_at(data, k32.SizeofResource(module, info)) if data else None

    try:
        groups = names(RT_GROUP_ICON)
        group = load(RT_GROUP_ICON, groups[0]) if groups else None
        icons = {}
        for name in names(RT_ICON):
            data = load(RT_ICON, name) if isinstance(name, int) else None
            if data is not None:
                icons[name] = data
        return group, icons
    finally:
        k32.FreeLibrary(module)


def _check_exe_icon(exe: Path) -> int:
    print("=== exe icon ===")
    try:
        group, icons = _load_exe_icon_resources(exe)
        failures = check_icon_resources(group, icons, expected_icon_rgba())
    except Exception as exc:
        failures = [f"could not read the icon resources: {exc}"]
    for failure in failures:
        print(f"  {failure}")
    if failures:
        print("verify_frozen_build: exe icon FAILED")
        return 1
    print(f"verify_frozen_build: exe icon PASSED (256x256 frame matches {APP_ICON.name})")
    return 0


def main() -> int:
    exe = _bundle_executable()
    if not exe.is_file():
        print(f"No frozen build found at {exe} -- build it first:")
        print("  .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm")
        return 1

    # Run every check even if one fails, same report-every-failure stance
    # as _self_check() itself, so a red CI step shows all results at once.
    self_check_rc = _run_mode(exe, "--self-check")
    smoke_rc = _run_mode(exe, "--smoke")
    icon_rc = _check_exe_icon(exe) if sys.platform == "win32" else 0

    if self_check_rc != 0 or smoke_rc != 0 or icon_rc != 0:
        return 1

    total_size = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file())
    print(f"Bundle size: {total_size / (1024 * 1024):.1f} MiB ({exe.parent})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
