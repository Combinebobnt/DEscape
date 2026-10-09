"""App-chrome themes: descape/themes.py's preset data, viewer_dialogs.
apply_theme()'s palette build, and Settings > Appearance's Theme combo and
Chrome colours group.

apply_theme() sets the one shared QApplication's style and palette, and
viewer_dialogs._LIGHT_PALETTE is cached on its first call, so the autouse
fixture restores all three; a theme left applied would recolour every later
test's widgets.
"""

from __future__ import annotations

import pytest

from descape import settings, themes

import conftest

_GROUP_NAMES = ("Active", "Inactive", "Disabled")
_TEXT_PAIRS = (("window_text", "window"), ("text", "base"), ("button_text", "button"))
_NON_LEGACY = [pid for pid in themes.PRESETS if pid not in themes.LEGACY_PRESETS]


@pytest.fixture
def qt_app():
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    conftest.ensure_qapp()
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape import viewer_dialogs

    app = QApplication.instance()
    saved_palette = QPalette(app.palette())
    saved_style = app.style().objectName()
    saved_light = viewer_dialogs._LIGHT_PALETTE
    viewer_dialogs._LIGHT_PALETTE = None
    yield app
    viewer_dialogs._LIGHT_PALETTE = saved_light
    app.setStyle(saved_style)
    app.setPalette(saved_palette)


def _all_colors(palette) -> dict[tuple[int, int], int]:
    from PyQt5.QtGui import QPalette

    return {
        (group, role): palette.color(getattr(QPalette, group), role).rgba()
        for group in _GROUP_NAMES
        for role in range(QPalette.NColorRoles)
    }


def _role(role_id: str):
    from PyQt5.QtGui import QPalette

    return getattr(QPalette, themes.ROLE_QT_NAMES[role_id])


def _applied(app, preset_id: str, overrides: dict[str, str] | None = None):
    from PyQt5.QtGui import QPalette

    from descape.viewer_dialogs import apply_theme

    apply_theme(app, preset_id, overrides)
    return QPalette(app.palette())


def _legacy_dark_recipe(base):
    """The pre-theme _build_dark_palette(), verbatim apart from starting from
    `base`: the oracle the `dark` preset must still reproduce exactly."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QPalette

    palette = QPalette(base)
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))
    return palette


# --- preset data (Qt-free) ---------------------------------------------------


def test_light_is_empty_and_only_dark_carries_the_legacy_extra_keys() -> None:
    assert themes.PRESETS["light"][1] == {}
    assert set(themes.PRESETS["dark"][1]) == set(themes.ROLE_IDS) | set(themes.LEGACY_EXTRA_KEYS)
    for preset_id in _NON_LEGACY:
        assert not set(themes.PRESETS[preset_id][1]) & set(themes.LEGACY_EXTRA_KEYS), preset_id


@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_every_non_legacy_preset_sets_every_role_with_valid_hex(preset_id: str) -> None:
    colors = themes.PRESETS[preset_id][1]
    assert set(colors) == set(themes.ROLE_IDS)
    for role_id, value in colors.items():
        assert themes.normalize_hex(value) == value, (role_id, value)


@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_every_non_legacy_preset_is_legible(preset_id: str) -> None:
    colors = themes.PRESETS[preset_id][1]
    floor = 7.0 if preset_id in themes.HIGH_CONTRAST_PRESETS else 4.5
    for fg, bg in _TEXT_PAIRS:
        ratio = themes.contrast_ratio(colors[fg], colors[bg])
        assert ratio >= floor, f"{preset_id} {fg}/{bg} = {ratio:.2f} < {floor}"
    # UI-component level: Fusion's own light pair, #ffffff on #308cc6, is 3.69.
    ratio = themes.contrast_ratio(colors["highlighted_text"], colors["highlight"])
    assert ratio >= 3.0, f"{preset_id} highlighted_text/highlight = {ratio:.2f}"


def test_contrast_ratio_matches_known_wcag_values() -> None:
    assert themes.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert themes.contrast_ratio("#ffffff", "#308cc6") == pytest.approx(3.69, abs=0.01)
    assert themes.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


def test_blend_and_normalize_hex() -> None:
    assert themes.blend("#000000", "#ffffff", 0.5) == "#808080"
    assert themes.blend("#102030", "#102030", 0.6) == "#102030"
    assert themes.normalize_hex("#ABCDEF") == "#abcdef"
    for bad in ("abcdef", "#abc", "#abcdeg", 123, None):
        with pytest.raises(ValueError):
            themes.normalize_hex(bad)


# --- palette build -----------------------------------------------------------


def test_light_without_overrides_is_fusion_standard_for_every_role_and_group(qt_app) -> None:
    palette = _applied(qt_app, "light")
    assert qt_app.style().objectName().lower() == "fusion"
    assert _all_colors(palette) == _all_colors(qt_app.style().standardPalette())


def test_dark_without_overrides_is_the_legacy_recipe_for_every_role_and_group(qt_app) -> None:
    palette = _applied(qt_app, "dark")
    assert _all_colors(palette) == _all_colors(_legacy_dark_recipe(qt_app.style().standardPalette()))


def test_dark_is_built_on_the_given_base_not_the_live_app_palette(qt_app) -> None:
    """The old recipe started from QPalette(), a copy of the live app palette.
    apply_theme()'s setStyle() happens to reset that first, so the builder is
    checked directly with a foreign palette live."""
    from PyQt5.QtGui import QColor, QPalette

    from descape.viewer_dialogs import build_theme_palette

    standard = qt_app.style().standardPalette()
    qt_app.setPalette(QPalette(QColor("#8b140d")))
    built = build_theme_palette(standard, "dark", {})
    assert _all_colors(built) == _all_colors(_legacy_dark_recipe(standard))


def test_dark_is_unchanged_after_another_theme(qt_app) -> None:
    first = _all_colors(_applied(qt_app, "dark"))
    _applied(qt_app, "solarized_light", {"button": "#8b140d"})
    assert _all_colors(_applied(qt_app, "dark")) == first


def test_an_unknown_preset_builds_light(qt_app) -> None:
    assert _all_colors(_applied(qt_app, "no_such_preset")) == _all_colors(_applied(qt_app, "light"))


@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_a_preset_sets_every_role_in_active_and_inactive(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QPalette

    palette = _applied(qt_app, preset_id)
    for role_id, value in themes.PRESETS[preset_id][1].items():
        for group in (QPalette.Active, QPalette.Inactive):
            assert palette.color(group, _role(role_id)).name() == value, (preset_id, role_id)


def test_an_override_wins_over_the_preset(qt_app) -> None:
    from PyQt5.QtGui import QPalette

    palette = _applied(qt_app, "solarized_dark", {"window": "#123456", "text": "#fedcba"})
    assert palette.color(QPalette.Active, QPalette.Window).name() == "#123456"
    assert palette.color(QPalette.Inactive, QPalette.Text).name() == "#fedcba"
    assert palette.color(QPalette.Active, QPalette.Base).name() == themes.PRESETS["solarized_dark"][1]["base"]


def test_overriding_text_changes_disabled_text(qt_app) -> None:
    from PyQt5.QtGui import QPalette

    legacy = _applied(qt_app, "dark")
    overridden = _applied(qt_app, "dark", {"text": "#ffcc00"})
    disabled = overridden.color(QPalette.Disabled, QPalette.Text).name()
    assert disabled != legacy.color(QPalette.Disabled, QPalette.Text).name()
    assert disabled == themes.blend("#ffcc00", themes.PRESETS["dark"][1]["base"], 0.5)
    # Only the overridden role is re-derived on a legacy preset.
    for role in (QPalette.WindowText, QPalette.ButtonText, QPalette.Highlight, QPalette.HighlightedText):
        assert overridden.color(QPalette.Disabled, role) == legacy.color(QPalette.Disabled, role)


def test_overriding_text_on_light_derives_its_disabled_colour_too(qt_app) -> None:
    from PyQt5.QtGui import QPalette

    light = _applied(qt_app, "light")
    overridden = _applied(qt_app, "light", {"text": "#aa0000"})
    base = light.color(QPalette.Active, QPalette.Base).name()
    assert overridden.color(QPalette.Disabled, QPalette.Text).name() == themes.blend("#aa0000", base, 0.5)
    assert overridden.color(QPalette.Disabled, QPalette.WindowText) == light.color(QPalette.Disabled, QPalette.WindowText)


@pytest.mark.parametrize("preset_id", list(themes.PRESETS))
def test_disabled_differs_from_active_for_every_preset(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QPalette

    palette = _applied(qt_app, preset_id)
    roles = ["window_text", "text", "button_text", "highlight"]
    if preset_id not in themes.LEGACY_PRESETS:
        roles.append("highlighted_text")
    for role_id in roles:
        role = _role(role_id)
        assert palette.color(QPalette.Disabled, role) != palette.color(QPalette.Active, role), (preset_id, role_id)


@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_non_legacy_disabled_colours_follow_the_blend_rule(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QPalette

    colors = themes.PRESETS[preset_id][1]
    palette = _applied(qt_app, preset_id)
    for role_id, background, amount in themes.DISABLED_BLENDS:
        expected = themes.blend(colors[role_id], colors[background], amount)
        assert palette.color(QPalette.Disabled, _role(role_id)).name() == expected, (preset_id, role_id)
    assert palette.color(QPalette.Disabled, QPalette.HighlightedText) == palette.color(QPalette.Disabled, QPalette.Text)


def _bevels(palette):
    from PyQt5.QtGui import QPalette

    roles = (QPalette.Light, QPalette.Midlight, QPalette.Mid, QPalette.Dark, QPalette.Shadow)
    return [palette.color(group, role).rgba() for group in (QPalette.Active, QPalette.Disabled) for role in roles]


@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_bevels_follow_the_button_colour(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QColor, QPalette

    palette = _applied(qt_app, preset_id)
    assert _bevels(palette) == _bevels(QPalette(QColor(themes.PRESETS[preset_id][1]["button"])))


def test_a_button_override_rederives_legacy_bevels_and_nothing_else_does(qt_app) -> None:
    from PyQt5.QtGui import QColor, QPalette

    fusion = _bevels(qt_app.style().standardPalette())
    assert _bevels(_applied(qt_app, "light", {"text": "#aa0000"})) == fusion
    assert _bevels(_applied(qt_app, "dark", {"button": "#8b140d"})) == _bevels(QPalette(QColor("#8b140d")))


# --- Settings > Appearance ---------------------------------------------------


@pytest.fixture
def dialog(qt_app):
    dialog, window = conftest.dialog_and_window()
    yield dialog
    dialog.close()
    conftest.close_window(window)


def _choose(dialog, preset_id: str) -> None:
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData(preset_id))


@pytest.mark.gui
def test_theme_combo_lists_every_preset_in_order_and_opens_on_the_persisted_one(tmp_path, qt_app) -> None:
    (tmp_path / "config.yaml").write_text("theme: solarized_dark\n")
    dialog, window = conftest.dialog_and_window()
    try:
        combo = dialog.theme_combo
        assert [combo.itemData(i) for i in range(combo.count())] == list(themes.PRESETS)
        assert combo.currentData() == "solarized_dark"
        assert set(dialog._theme_swatches) == set(themes.ROLE_IDS)
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.gui
def test_choosing_a_preset_persists_and_applies_it_live(qt_app, dialog) -> None:
    from PyQt5.QtGui import QPalette

    _choose(dialog, "high_contrast_light")
    assert settings.get_theme() == "high_contrast_light"
    settings._theme = None
    assert settings.get_theme() == "high_contrast_light"
    window_hex = themes.PRESETS["high_contrast_light"][1]["window"]
    assert qt_app.palette().color(QPalette.Active, QPalette.Window).name() == window_hex
    assert dialog._theme_swatches["window"].toolTip() == window_hex


@pytest.mark.gui
def test_picking_a_role_colour_overrides_it_live(qt_app, dialog, monkeypatch) -> None:
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtWidgets import QColorDialog

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *_args, **_kw: QColor("#336699")))
    dialog._theme_swatches["highlight"].click()
    assert settings.get_theme_colors() == {"highlight": "#336699"}
    assert qt_app.palette().color(QPalette.Active, QPalette.Highlight).name() == "#336699"
    assert dialog._theme_swatches["highlight"].toolTip() == "#336699"


@pytest.mark.gui
def test_switching_preset_with_overrides_asks_and_no_keeps_everything(qt_app, dialog, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    settings.set_theme_color("window", "#101010")
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.No))
    _choose(dialog, "dim")
    assert len(asked) == 1
    assert settings.get_theme() == "light"
    assert settings.get_theme_colors() == {"window": "#101010"}
    assert dialog.theme_combo.currentData() == "light"


@pytest.mark.gui
def test_switching_preset_with_overrides_and_yes_clears_them(qt_app, dialog, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    settings.set_theme_color("window", "#101010")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    _choose(dialog, "dim")
    assert settings.get_theme() == "dim"
    assert settings.get_theme_colors() == {}


@pytest.mark.gui
def test_switching_preset_without_overrides_does_not_ask(qt_app, dialog, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    def _fail(*_args, **_kw):
        raise AssertionError("asked with no overrides to lose")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_fail))
    _choose(dialog, "dim")
    assert settings.get_theme() == "dim"


@pytest.mark.gui
def test_row_default_and_reset_all_clear_overrides(qt_app, dialog) -> None:
    from PyQt5.QtGui import QPalette

    _choose(dialog, "solarized_light")
    settings.set_theme_color("window", "#101010")
    settings.set_theme_color("text", "#202020")
    dialog._reset_theme_color("window")
    assert settings.get_theme_colors() == {"text": "#202020"}
    dialog.theme_reset_button.click()
    assert settings.get_theme_colors() == {}
    assert settings.get_theme() == "solarized_light"
    text_hex = themes.PRESETS["solarized_light"][1]["text"]
    assert qt_app.palette().color(QPalette.Active, QPalette.Text).name() == text_hex
