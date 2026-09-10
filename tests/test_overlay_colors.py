"""settings.py coverage for Settings > Appearance's per-element tool overlay
colors and the Ruler label font size -- both plain settings.py round-trip
gaps, same shape tests/test_settings.py already covers for keybinds/distance
ticks, but kept in their own file since OVERLAY_COLORS is a whole new family
rather than one more getter/setter pair.

Every test here relies on conftest.py's autouse _isolated_settings fixture
(redirects descape.settings.CONFIG_PATH to this test's own tmp_path/
config.yaml) exactly as test_settings.py's own module docstring describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import settings


def _write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "config.yaml").write_text(text)


# --- overlay colors ---------------------------------------------------------


def test_missing_config_returns_every_declared_default(tmp_path: Path) -> None:
    for color_id, _label, default in settings.OVERLAY_COLORS:
        assert settings.get_overlay_color(color_id) == default
        assert settings.get_default_overlay_color(color_id) == default


def test_a_malformed_persisted_hex_falls_back_to_default_others_unaffected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "overlay_colors:\n  ruler_line: 'not-a-color'\n  unit_hover: '#123abc'\n",
    )
    assert settings.get_overlay_color("ruler_line") == settings.get_default_overlay_color("ruler_line")
    assert settings.get_overlay_color("unit_hover") == "#123abc"


@pytest.mark.parametrize("bad", ["not-a-color", "#12345", "#gggggg", "123456", "", None, 42])
def test_set_overlay_color_with_a_bad_value_raises_and_writes_nothing(tmp_path: Path, bad) -> None:
    with pytest.raises(ValueError):
        settings.set_overlay_color("ruler_line", bad)
    assert not (tmp_path / "config.yaml").exists()


def test_set_overlay_color_accepts_uppercase_and_normalizes_to_lowercase(tmp_path: Path, monkeypatch) -> None:
    settings.set_overlay_color("ruler_line", "#ABCDEF")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    assert settings.get_overlay_color("ruler_line") == "#abcdef"


def test_overlay_color_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    settings.set_overlay_color("region_fill", "#123456")
    monkeypatch.setattr(settings, "_overlay_colors", None)
    assert settings.get_overlay_color("region_fill") == "#123456"
    # Every other id must still read back at its own default -- set_
    # overlay_color must update the persisted dict, not replace it.
    for color_id, _label, default in settings.OVERLAY_COLORS:
        if color_id != "region_fill":
            assert settings.get_overlay_color(color_id) == default


# The "every OVERLAY_COLORS prefix has a section title" reflective check
# lives in test_overlay_colors_viewer.py instead of here: _OVERLAY_SECTION_
# TITLES is a SettingsDialog class attribute (descape/viewer.py), which needs
# Qt to import, and this file is deliberately Qt-free like test_settings.py.


# --- ruler label font size ---------------------------------------------------


def test_missing_font_size_falls_back_to_default(tmp_path: Path) -> None:
    assert settings.get_ruler_label_font_px() == settings.RULER_LABEL_FONT_PX_DEFAULT


@pytest.mark.parametrize("raw", [0, 4, 7, 33, 200, -5, "18", None, [18]])
def test_an_out_of_range_or_malformed_font_size_falls_back_to_default(tmp_path: Path, raw) -> None:
    _write_config(tmp_path, f"ruler_label_font_px: {raw!r}\n")
    assert settings.get_ruler_label_font_px() == settings.RULER_LABEL_FONT_PX_DEFAULT


@pytest.mark.parametrize("bad", [settings.RULER_LABEL_FONT_PX_MIN - 1, settings.RULER_LABEL_FONT_PX_MAX + 1])
def test_setting_an_out_of_range_font_size_raises_and_writes_nothing(tmp_path: Path, bad: int) -> None:
    with pytest.raises(ValueError):
        settings.set_ruler_label_font_px(bad)
    assert not (tmp_path / "config.yaml").exists()


def test_font_size_round_trips_through_the_file(tmp_path: Path, monkeypatch) -> None:
    other = settings.RULER_LABEL_FONT_PX_DEFAULT + 1
    settings.set_ruler_label_font_px(other)
    monkeypatch.setattr(settings, "_ruler_label_font_px", None)
    assert settings.get_ruler_label_font_px() == other
