"""tools/verify_frozen_build.py's Windows exe-icon check, on fixtures shaped
like PE icon resources: an RT_GROUP_ICON directory (GRPICONDIRENTRY rows
carrying an RT_ICON id where an .ico file has an offset) plus the RT_ICON
data by id, built in memory from Pillow-written ICO files. Only the ctypes
resource loading is Windows-only; everything it hands on is tested here.
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

from tools import verify_frozen_build as vfb

ROOT = Path(__file__).resolve().parent.parent


def _resources(ico: bytes, first_id: int = 1) -> tuple[bytes, dict[int, bytes]]:
    # The way PyInstaller's CopyIcons_FromIco writes an .ico into the exe.
    reserved, kind, count = struct.unpack_from("<HHH", ico)
    group = struct.pack("<HHH", reserved, kind, count)
    icons = {}
    for i in range(count):
        fields = struct.unpack_from("<BBBBHHII", ico, 6 + 16 * i)
        size, offset = fields[6], fields[7]
        group += struct.pack("<BBBBHHIH", *fields[:7], first_id + i)
        icons[first_id + i] = ico[offset:offset + size]
    return group, icons


def _ico(im: Image.Image, **params) -> bytes:
    out = io.BytesIO()
    im.save(out, format="ICO", **params)
    return out.getvalue()


@pytest.fixture
def app_icon() -> Image.Image:
    with Image.open(ROOT / "descape" / "app_icon.png") as im:
        im.load()
        return im


def test_expected_icon_is_the_app_png_as_rgba(app_icon):
    expected = vfb.expected_icon_rgba()
    assert expected.size == (256, 256)
    assert expected.tobytes() == app_icon.convert("RGBA").tobytes()


def test_expected_icon_scales_a_larger_png_like_the_ico_writer(tmp_path, app_icon):
    big = app_icon.resize((512, 512), Image.Resampling.NEAREST)
    big.save(tmp_path / "big.png")
    group, icons = _resources(_ico(big))
    assert vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba(tmp_path / "big.png")) == []


def test_app_icon_resources_pass(app_icon):
    # PyInstaller's PNG conversion is Pillow's default ICO save: seven PNG frames.
    group, icons = _resources(_ico(app_icon), first_id=5)
    assert len(icons) == 7
    assert vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba()) == []


def test_bmp_dib_frames_decode_and_pass(app_icon):
    group, icons = _resources(_ico(app_icon, bitmap_format="bmp"))
    assert not any(data.startswith(b"\x89PNG") for data in icons.values())
    assert vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba()) == []


def test_no_group_resource_fails():
    failures = vfb.check_icon_resources(None, {}, vfb.expected_icon_rgba())
    assert failures == ["no RT_GROUP_ICON resource: the exe has no icon"]


def test_group_naming_missing_icon_resources_fails(app_icon):
    group, _icons = _resources(_ico(app_icon))
    failures = vfb.check_icon_resources(group, {}, vfb.expected_icon_rgba())
    assert [f for f in failures if "named by the group is missing" in f] == failures[:7]
    assert failures[7].startswith("no 256x256 icon frame")


def test_truncated_group_fails(app_icon):
    group, icons = _resources(_ico(app_icon))
    failures = vfb.check_icon_resources(group[:20], icons, vfb.expected_icon_rgba())
    assert failures == ["RT_GROUP_ICON is 20 bytes, too short for 7 entries"]


def test_undecodable_icon_fails(app_icon):
    group, icons = _resources(_ico(app_icon))
    icons[7] = b"\x89PNG\r\n\x1a\n not really"
    failures = vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba())
    assert len(failures) == 2
    assert failures[0].startswith("RT_ICON 7 does not decode")
    assert failures[1].startswith("no 256x256 icon frame")


@pytest.mark.parametrize("other", ["solid", "flipped"])
def test_other_256px_icon_fails(app_icon, other):
    # Stand-ins for PyInstaller's default icon, which the Linux wheel does not ship.
    if other == "solid":
        im = Image.new("RGBA", (256, 256), (40, 90, 160, 255))
    else:
        im = app_icon.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    group, icons = _resources(_ico(im))
    failures = vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba())
    assert len(failures) == 1
    assert "(256x256) pixels differ from app_icon.png" in failures[0]


def test_icon_without_a_256px_frame_fails(app_icon):
    group, icons = _resources(_ico(app_icon, sizes=[(16, 16), (32, 32), (48, 48), (128, 128)]))
    failures = vfb.check_icon_resources(group, icons, vfb.expected_icon_rgba())
    assert failures == ["no 256x256 icon frame (decoded sizes: [(16, 16), (32, 32), (48, 48), (128, 128)])"]


@pytest.mark.parametrize(("platform", "icon_rc", "want_rc", "checked"), [
    ("linux", 1, 0, False),
    ("darwin", 1, 0, False),
    ("win32", 0, 0, True),
    ("win32", 1, 1, True),
])
def test_main_checks_the_icon_on_windows_only(monkeypatch, tmp_path, capsys,
                                              platform, icon_rc, want_rc, checked):
    exe = tmp_path / "DEscape.exe"
    exe.write_bytes(b"")
    calls = []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(vfb, "_bundle_executable", lambda: exe)
    monkeypatch.setattr(vfb, "_run_mode", lambda _exe, _mode: 0)
    monkeypatch.setattr(vfb, "_check_exe_icon", lambda path: calls.append(path) or icon_rc)
    assert vfb.main() == want_rc
    assert calls == ([exe] if checked else [])
    capsys.readouterr()
