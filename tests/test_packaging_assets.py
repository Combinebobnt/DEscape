"""Default-tier guard for the packaging assets committed for the Linux
AppImage build. Deliberately does NOT copy tests/test_config_example.py's byte-identity
drift-guard pattern: asserting the committed PNG equals
tools/gen_app_icon.py's output would make the follow-up "swap the
placeholder for real artwork" fail this test. Assert properties only, so
real art can drop in without touching anything else.
"""

from __future__ import annotations

import configparser
from pathlib import Path

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


def test_desktop_file_has_required_keys() -> None:
    desktop_path = ROOT / "packaging" / "DEscape.desktop"
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.read(desktop_path)
    assert "Desktop Entry" in parser
    section = parser["Desktop Entry"]
    for key in ("Type", "Name", "Exec", "Icon", "Categories"):
        assert section.get(key, "").strip(), f"DEscape.desktop missing key {key!r}"
