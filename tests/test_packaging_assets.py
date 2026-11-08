"""Default-tier guard for the packaging assets committed for the Linux
AppImage build. Deliberately does NOT copy tests/test_config_example.py's byte-identity
drift-guard pattern: asserting the committed PNG equals
tools/gen_app_icon.py's output would make the follow-up "swap the
placeholder for real artwork" fail this test. Assert properties only, so
real art can drop in without touching anything else.
"""

from __future__ import annotations

import configparser
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def test_app_icon_is_256x256_rgba_png() -> None:
    icon_path = ROOT / "descape" / "app_icon.png"
    assert icon_path.is_file(), f"{icon_path} is missing -- run tools/gen_app_icon.py"
    with Image.open(icon_path) as img:
        img.load()
        assert img.format == "PNG"
        assert img.size == (256, 256)
        assert img.mode == "RGBA"


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_frozen_exe_icon_is_windows_only(monkeypatch, platform):
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "platform", platform)
    exe_args = {}

    def exe(*args, **kwargs):
        exe_args.update(kwargs)

    runpy.run_path(str(ROOT / "packaging" / "descape.spec"), init_globals={
        "SPECPATH": str(ROOT / "packaging"),
        "Analysis": lambda *args, **kwargs: SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]),
        "PYZ": lambda *args: None,
        "EXE": exe,
        "COLLECT": lambda *args, **kwargs: None,
    })
    assert exe_args["icon"] == (str(ROOT / "descape" / "app_icon.png") if platform == "win32" else None)


def test_windows_icon_conversion_preserves_pixels_and_sizes(tmp_path):
    # PyInstaller's PNG conversion is Pillow's default ICO save, with no size options.
    icon_path = tmp_path / "app_icon.ico"
    with Image.open(ROOT / "descape" / "app_icon.png") as source:
        source.save(icon_path)
        with Image.open(icon_path) as icon:
            assert icon.ico.sizes() == {(n, n) for n in (16, 24, 32, 48, 64, 128, 256)}
            assert icon.convert("RGBA").tobytes() == source.convert("RGBA").tobytes()


def test_desktop_file_has_required_keys() -> None:
    desktop_path = ROOT / "packaging" / "DEscape.desktop"
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.read(desktop_path)
    assert "Desktop Entry" in parser
    section = parser["Desktop Entry"]
    for key in ("Type", "Name", "Exec", "Icon", "Categories"):
        assert section.get(key, "").strip(), f"DEscape.desktop missing key {key!r}"
