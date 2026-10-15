"""App-chrome themes: descape/themes.py's preset data, viewer_dialogs.
apply_theme()'s two-palette build, the theme-aware status text colours, and
Settings > Appearance's Theme combo and Chrome colours group.

apply_theme() sets the one shared QApplication's style, palette and
QPushButton class palette, and viewer_dialogs._LIGHT_PALETTE is cached on its
first call, so the qt_app fixture restores all four through
conftest.restoring_app_theme(); a theme left applied would recolour every
later test's widgets.

Every pixel read here is PyQt 5.15, Fusion, offscreen QPA.
"""

from __future__ import annotations

import pytest

from descape import settings, themes

import conftest

_GROUP_NAMES = ("Active", "Inactive", "Disabled")
_TEXT_PAIRS = (
    ("window_text", "window"),
    ("text", "base"),
    ("button_text", "button"),
    ("surface_text", "surface"),
    # Menu bar and toolbar text, which Fusion draws in ButtonText on Window.
    ("surface_text", "window"),
)
_NON_LEGACY = [pid for pid in themes.PRESETS if pid not in themes.LEGACY_PRESETS]
_PLAIN_PRESETS = [pid for pid in themes.PRESETS if pid not in themes.EDITOR_PRESETS]
# apply_theme()'s two palettes. "app" takes the surface pair as Button/ButtonText, "push_button" the button pair.
_PALETTES = ("app", "push_button")
_ONE_PALETTE_ROLES = [role_id for role_id in themes.ROLE_IDS if role_id not in themes.SURFACE_ROLES]
_OVERRIDE_PRESETS = ("light", "dark", "dim", "editor_dark_wood_blue")
# A fixed button wider than its label, sampled 3 px in on three rows: no glyph antialiasing reaches that column.
_BUTTON_SIZE = (160, 28)
_BUTTON_ROWS = (4, 14, 23)


@pytest.fixture
def qt_app():
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    with conftest.restoring_app_theme() as app:
        yield app


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


def _applied_pair(app, preset_id: str, overrides: dict[str, str] | None = None):
    """Copies of (app palette, QPushButton class palette) after apply_theme()."""
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import apply_theme

    apply_theme(app, preset_id, overrides)
    return QPalette(app.palette()), QPalette(QApplication.palette("QPushButton"))


def _applied(app, preset_id: str, overrides: dict[str, str] | None = None, which: str = "app"):
    return _applied_pair(app, preset_id, overrides)[_PALETTES.index(which)]


def _seen_by(which: str, colors: dict[str, str]) -> dict[str, str]:
    """`colors` with `button`/`button_text` holding what that palette's Button
    and ButtonText get: the surface pair in the app palette."""
    if which == "app":
        return {**colors, "button": colors["surface"], "button_text": colors["surface_text"]}
    return colors


def _use(app, preset_id: str, overrides: dict[str, str] | None = None) -> None:
    """Sets the theme the way the app does: persisted first (the status floor
    reads settings.get_theme()), then applied."""
    from descape.viewer_dialogs import apply_theme

    settings.set_theme(preset_id)
    for role_id, value in (overrides or {}).items():
        settings.set_theme_color(role_id, value)
    apply_theme(app, settings.get_theme(), settings.get_theme_colors())


def _pixel(widget, x: int, y: int) -> str:
    from PyQt5.QtGui import QColor

    return QColor(widget.grab().toImage().pixel(x, y)).name()


def _tab_page_pixel(app) -> str:
    """What a real, shown QTabWidget fills its page with under the live theme."""
    from PyQt5.QtWidgets import QLabel, QTabWidget

    tabs = QTabWidget()
    tabs.addTab(QLabel("body"), "Tab A")
    tabs.addTab(QLabel("other"), "Tab B")
    tabs.resize(240, 120)
    tabs.show()
    app.processEvents()
    try:
        return _pixel(tabs, 6, tabs.height() - 6)
    finally:
        tabs.close()


def _button_rows(app, button, *, hover: bool, default: bool) -> list[str]:
    """CE_PushButton painted from the shown button's own option, so its
    palette is whatever the class palette resolved to, in a pointer state
    no offscreen test can put a real widget in."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor, QImage, QPainter, QPalette
    from PyQt5.QtWidgets import QStyle, QStyleOptionButton

    option = QStyleOptionButton()
    button.initStyleOption(option)
    option.rect = QRect(0, 0, *_BUTTON_SIZE)
    option.state = QStyle.State_Enabled | QStyle.State_Raised
    if hover:
        option.state |= QStyle.State_MouseOver
    if default:
        option.features |= QStyleOptionButton.DefaultButton
    image = QImage(*_BUTTON_SIZE, QImage.Format_RGB32)
    image.fill(option.palette.color(QPalette.Window))
    painter = QPainter(image)
    app.style().drawControl(QStyle.CE_PushButton, option, painter, button)
    painter.end()
    return [QColor(image.pixel(3, y)).name() for y in _BUTTON_ROWS]


def _painted(app, preset_id: str) -> dict:
    """Pixel colours of real, shown widgets under `preset_id`: every surface
    Fusion paints from Button, plus the push button's rows in four states."""
    from PyQt5.QtWidgets import QComboBox, QLabel, QPushButton, QTableWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget

    _use(app, preset_id)
    host = QWidget()
    layout = QVBoxLayout(host)
    push = QPushButton("Push")
    push.setFixedSize(*_BUTTON_SIZE)
    combo = QComboBox()
    combo.addItems(["Combo", "Other"])
    tool = QToolButton()
    tool.setText("Tool")
    tool.setFixedSize(*_BUTTON_SIZE)
    table = QTableWidget(2, 2)
    table.setHorizontalHeaderLabels(["Head A", "Head B"])
    table.setFixedHeight(80)
    tabs = QTabWidget()
    tabs.addTab(QLabel("body"), "Tab A")
    tabs.addTab(QLabel("other"), "Tab B")
    for widget in (push, combo, tool, table, tabs):
        layout.addWidget(widget)
    host.resize(420, 420)
    host.show()
    app.processEvents()
    try:
        bar = tabs.tabBar()
        selected, unselected = bar.tabRect(0), bar.tabRect(1)
        header = table.horizontalHeader()
        return {
            "tab_page": _pixel(tabs, 6, tabs.height() - 6),
            "selected_tab": _pixel(bar, selected.left() + 4, selected.center().y()),
            "unselected_tab": _pixel(bar, unselected.left() + 4, unselected.center().y()),
            "combo": _pixel(combo, 5, combo.height() // 2),
            "header": _pixel(header, 4, header.height() // 2),
            "tool_button": _pixel(tool, 3, tool.height() // 2),
            "push_grabbed": [_pixel(push, 3, y) for y in _BUTTON_ROWS],
            "push_states": {
                "resting": _button_rows(app, push, hover=False, default=False),
                "hover": _button_rows(app, push, hover=True, default=False),
                "default": _button_rows(app, push, hover=False, default=True),
                "default with hover": _button_rows(app, push, hover=True, default=True),
            },
        }
    finally:
        host.close()


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


def test_the_surface_pair_follows_the_button_pair_and_shares_its_qt_roles() -> None:
    assert len(themes.ROLE_IDS) == 14
    at = themes.ROLE_IDS.index("button")
    assert themes.ROLE_IDS[at : at + 4] == (*themes.PUSH_BUTTON_ROLES, *themes.SURFACE_ROLES)
    assert [themes.ROLE_QT_NAMES[role_id] for role_id in themes.SURFACE_ROLES] == ["Button", "ButtonText"]
    assert [themes.ROLE_QT_NAMES[role_id] for role_id in themes.PUSH_BUTTON_ROLES] == ["Button", "ButtonText"]
    assert themes.role_label("surface") == "Tabs and drop-downs"
    assert themes.role_label("surface_text") == "Drop-down and menu text"
    assert themes.role_label("button") == "Button"


def test_the_editor_family_is_two_bases_by_four_accents() -> None:
    expected = tuple(f"editor_{base}_{accent}" for base in ("parchment", "dark_wood") for accent in ("red", "blue", "green", "purple"))
    assert expected == themes.EDITOR_PRESETS
    assert tuple(themes.PRESETS)[-8:] == expected
    assert themes.preset_label("editor_parchment_red") == "Wood (red)"
    assert themes.preset_label("editor_dark_wood_red") == "Dark Wood (red)"
    for preset_id in expected:
        colors = themes.PRESETS[preset_id][1]
        assert colors["button"] != colors["surface"], preset_id


def test_only_the_editor_presets_declare_a_surface_apart_from_their_button() -> None:
    for preset_id in _PLAIN_PRESETS:
        colors = themes.PRESETS[preset_id][1]
        if colors:
            assert (colors["surface"], colors["surface_text"]) == (colors["button"], colors["button_text"]), preset_id


def test_legible_leaves_a_colour_that_already_reaches_the_floor() -> None:
    assert themes.contrast_ratio("#c62828", "#fbfbfb") >= 4.5
    assert themes.legible("#c62828", "#fbfbfb", 4.5) == "#c62828"


@pytest.mark.parametrize(
    ("color", "background", "minimum", "expected"),
    [
        ("#4caf50", "#fbfbfb", 4.5, "#39833c"),  # light background: toward black
        ("#e05252", "#434343", 4.5, "#eea0a0"),  # dark background: toward white
        ("#4caf50", "#f1f1f1", 7.0, "#265828"),
        ("#e05252", "#313131", 7.0, "#f1b1b1"),
    ],
)
def test_legible_takes_the_first_step_that_reaches_the_floor(color: str, background: str, minimum: float, expected: str) -> None:
    assert themes.contrast_ratio(color, background) < minimum
    shade = themes.legible(color, background, minimum)
    assert shade == expected
    assert themes.contrast_ratio(shade, background) >= minimum
    pole = "#000000" if themes.relative_luminance(shade) < themes.relative_luminance(color) else "#ffffff"
    steps = [themes.blend(color, pole, step / 20) for step in range(1, 21)]
    assert all(themes.contrast_ratio(earlier, background) < minimum for earlier in steps[: steps.index(shade)])


def test_legible_returns_the_pole_when_no_step_reaches_the_floor() -> None:
    # Black is the better pole on mid grey, at 4.69:1, and still short of 7.
    assert themes.contrast_ratio("#000000", "#777777") > themes.contrast_ratio("#ffffff", "#777777")
    assert themes.contrast_ratio("#000000", "#777777") < 7.0
    assert themes.legible("#4caf50", "#777777", 7.0) == "#000000"


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
    app_palette, push_palette = _applied_pair(qt_app, "dark")
    assert _all_colors(app_palette) == first
    assert _all_colors(push_palette) == first


def test_an_unknown_preset_builds_the_default(qt_app) -> None:
    assert _all_colors(_applied(qt_app, "no_such_preset")) == _all_colors(_applied(qt_app, themes.THEME_DEFAULT))


@pytest.mark.parametrize("which", _PALETTES)
@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_a_preset_sets_every_role_in_active_and_inactive(qt_app, preset_id: str, which: str) -> None:
    from PyQt5.QtGui import QPalette

    palette = _applied(qt_app, preset_id, which=which)
    colors = _seen_by(which, themes.PRESETS[preset_id][1])
    for role_id in _ONE_PALETTE_ROLES:
        for group in (QPalette.Active, QPalette.Inactive):
            assert palette.color(group, _role(role_id)).name() == colors[role_id], (preset_id, which, role_id)


# --- two palettes: the accent reaches push buttons only -------------------------


@pytest.mark.parametrize("preset_id", _PLAIN_PRESETS)
def test_a_preset_with_no_accent_gives_push_buttons_the_app_palette(qt_app, preset_id: str) -> None:
    app_palette, push_palette = _applied_pair(qt_app, preset_id)
    assert _all_colors(push_palette) == _all_colors(app_palette)


@pytest.mark.parametrize("preset_id", themes.EDITOR_PRESETS)
def test_an_editor_preset_puts_its_accent_in_the_push_button_palette_only(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QPalette

    colors = themes.PRESETS[preset_id][1]
    app_palette, push_palette = _applied_pair(qt_app, preset_id)
    for group in (QPalette.Active, QPalette.Inactive):
        assert app_palette.color(group, QPalette.Button).name() == colors["surface"]
        assert app_palette.color(group, QPalette.ButtonText).name() == colors["surface_text"]
        assert push_palette.color(group, QPalette.Button).name() == colors["button"]
        assert push_palette.color(group, QPalette.ButtonText).name() == colors["button_text"]
        assert push_palette.color(group, QPalette.Highlight).name() == colors["highlight"]
        assert app_palette.color(group, QPalette.Highlight).name() == colors["highlight"]


@pytest.mark.parametrize("preset_id", _OVERRIDE_PRESETS)
def test_a_button_override_reaches_the_push_button_palette_only(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QColor, QPalette

    plain_app = _all_colors(_applied(qt_app, preset_id))
    app_palette, push_palette = _applied_pair(qt_app, preset_id, {"button": "#123456"})
    assert push_palette.color(QPalette.Active, QPalette.Button).name() == "#123456"
    assert _bevels(push_palette) == _bevels(QPalette(QColor("#123456")))
    assert _all_colors(app_palette) == plain_app


@pytest.mark.parametrize("preset_id", _OVERRIDE_PRESETS)
def test_a_surface_override_reaches_the_app_palette_only(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QColor, QPalette

    plain_app, plain_push = _applied_pair(qt_app, preset_id)
    app_palette, push_palette = _applied_pair(qt_app, preset_id, {"surface": "#123456"})
    assert app_palette.color(QPalette.Active, QPalette.Button).name() == "#123456"
    assert _bevels(app_palette) == _bevels(QPalette(QColor("#123456")))
    disabled_text = app_palette.color(QPalette.Disabled, QPalette.ButtonText)
    if preset_id in themes.LEGACY_PRESETS:
        # The legacy rule: only a role that is itself overridden gets its Disabled colour derived.
        assert disabled_text == plain_app.color(QPalette.Disabled, QPalette.ButtonText)
    else:
        text = app_palette.color(QPalette.Active, QPalette.ButtonText).name()
        assert disabled_text.name() == themes.blend(text, "#123456", 0.5)
        assert disabled_text != plain_app.color(QPalette.Disabled, QPalette.ButtonText)
    assert _all_colors(push_palette) == _all_colors(plain_push)


def test_a_surface_text_override_on_a_legacy_preset_derives_its_disabled_colour(qt_app) -> None:
    from PyQt5.QtGui import QPalette

    plain_push = _all_colors(_applied(qt_app, "dark", which="push_button"))
    app_palette, push_palette = _applied_pair(qt_app, "dark", {"surface_text": "#ffcc00"})
    surface = themes.PRESETS["dark"][1]["surface"]
    assert app_palette.color(QPalette.Disabled, QPalette.ButtonText).name() == themes.blend("#ffcc00", surface, 0.5)
    assert _all_colors(push_palette) == plain_push


def test_light_after_an_editor_preset_leaves_no_accent_behind(qt_app) -> None:
    _applied(qt_app, "editor_parchment_red")
    app_palette, push_palette = _applied_pair(qt_app, "light")
    standard = _all_colors(qt_app.style().standardPalette())
    assert _all_colors(app_palette) == standard
    assert _all_colors(push_palette) == standard


def test_dropping_a_button_override_restores_the_presets_button(qt_app) -> None:
    """The trap: both app palettes are equal, and Qt clears a class palette
    only when the app palette it is given differs from the live one."""
    from PyQt5.QtGui import QPalette

    overridden_app, overridden_push = _applied_pair(qt_app, "dim", {"button": "#8b140d"})
    assert overridden_push.color(QPalette.Active, QPalette.Button).name() == "#8b140d"
    app_palette, push_palette = _applied_pair(qt_app, "dim")
    assert _all_colors(app_palette) == _all_colors(overridden_app)
    assert push_palette.color(QPalette.Active, QPalette.Button).name() == themes.PRESETS["dim"][1]["button"]
    assert _all_colors(push_palette) == _all_colors(app_palette)


def test_the_restoring_helper_puts_back_a_class_palette_qt_would_leave_stale(qt_app) -> None:
    """The same trap in the fixture's own teardown. From a pristine app palette
    it cannot bite, since that palette's resolve mask differs from any
    apply_theme() sets, so the helper is entered here with a theme applied."""
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    themed = _all_colors(_applied(qt_app, "light"))
    with conftest.restoring_app_theme():
        _applied(qt_app, "light", {"button": "#8b140d"})
        assert QApplication.palette("QPushButton").color(QPalette.Active, QPalette.Button).name() == "#8b140d"
    assert _all_colors(qt_app.palette()) == themed
    assert _all_colors(QApplication.palette("QPushButton")) == themed


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


@pytest.mark.parametrize("which", _PALETTES)
@pytest.mark.parametrize("preset_id", list(themes.PRESETS))
def test_disabled_differs_from_active_for_every_preset(qt_app, preset_id: str, which: str) -> None:
    from PyQt5.QtGui import QPalette

    palette = _applied(qt_app, preset_id, which=which)
    roles = ["window_text", "text", "button_text", "highlight"]
    if preset_id not in themes.LEGACY_PRESETS:
        roles.append("highlighted_text")
    for role_id in roles:
        role = _role(role_id)
        assert palette.color(QPalette.Disabled, role) != palette.color(QPalette.Active, role), (preset_id, which, role_id)


@pytest.mark.parametrize("which", _PALETTES)
@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_non_legacy_disabled_colours_follow_the_blend_rule(qt_app, preset_id: str, which: str) -> None:
    from PyQt5.QtGui import QPalette

    colors = _seen_by(which, themes.PRESETS[preset_id][1])
    palette = _applied(qt_app, preset_id, which=which)
    for role_id, background, amount in themes.DISABLED_BLENDS:
        expected = themes.blend(colors[role_id], colors[background], amount)
        assert palette.color(QPalette.Disabled, _role(role_id)).name() == expected, (preset_id, which, role_id)
    assert palette.color(QPalette.Disabled, QPalette.HighlightedText) == palette.color(QPalette.Disabled, QPalette.Text)


def _bevels(palette):
    from PyQt5.QtGui import QPalette

    roles = (QPalette.Light, QPalette.Midlight, QPalette.Mid, QPalette.Dark, QPalette.Shadow)
    return [palette.color(group, role).rgba() for group in (QPalette.Active, QPalette.Disabled) for role in roles]


@pytest.mark.parametrize("which", _PALETTES)
@pytest.mark.parametrize("preset_id", _NON_LEGACY)
def test_bevels_follow_the_button_colour(qt_app, preset_id: str, which: str) -> None:
    from PyQt5.QtGui import QColor, QPalette

    palette = _applied(qt_app, preset_id, which=which)
    button = _seen_by(which, themes.PRESETS[preset_id][1])["button"]
    assert _bevels(palette) == _bevels(QPalette(QColor(button)))


def test_a_button_override_rederives_legacy_bevels_and_nothing_else_does(qt_app) -> None:
    from PyQt5.QtGui import QColor, QPalette

    fusion = _bevels(qt_app.style().standardPalette())
    for which in _PALETTES:
        assert _bevels(_applied(qt_app, "light", {"text": "#aa0000"}, which=which)) == fusion
    app_palette, push_palette = _applied_pair(qt_app, "dark", {"button": "#8b140d"})
    # Push buttons only: the app palette is still the legacy recipe, bevels and all.
    assert _all_colors(app_palette) == _all_colors(_legacy_dark_recipe(qt_app.style().standardPalette()))
    assert _bevels(push_palette) == _bevels(QPalette(QColor("#8b140d")))


# --- painted: what Fusion makes of the editor presets ---------------------------


@pytest.mark.parametrize("preset_id", themes.EDITOR_PRESETS)
def test_an_editor_preset_paints_legible_text_on_every_surface(qt_app, preset_id: str) -> None:
    """Fusion lightens a dark Button before painting it, so the declared
    colours cannot prove this. Each pair is the role Fusion draws that label
    with: tab bar text in WindowText, combo, header and tool button text in
    ButtonText."""
    colors = themes.PRESETS[preset_id][1]
    painted = _painted(qt_app, preset_id)
    pairs = (
        ("window_text", "tab_page"),
        ("window_text", "selected_tab"),
        ("window_text", "unselected_tab"),
        ("surface_text", "combo"),
        ("surface_text", "header"),
        ("surface_text", "tool_button"),
    )
    for role_id, surface in pairs:
        ratio = themes.contrast_ratio(colors[role_id], painted[surface])
        assert ratio >= 4.5, f"{preset_id} {role_id} on {surface} {painted[surface]} = {ratio:.2f}"


@pytest.mark.parametrize("preset_id", themes.EDITOR_PRESETS)
def test_an_editor_accent_keeps_button_text_legible_in_every_state(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QColor

    colors = themes.PRESETS[preset_id][1]
    painted = _painted(qt_app, preset_id)
    for state, rows in painted["push_states"].items():
        ratio = min(themes.contrast_ratio(colors["button_text"], row) for row in rows)
        assert ratio >= 4.5, f"{preset_id} button_text on a {state} push button {rows} = {ratio:.2f}"
    # The style-painted resting state is what the shown widget itself paints.
    assert painted["push_grabbed"] == painted["push_states"]["resting"]
    # A combo painted from the same Button colour still differs, by up to 9 levels a channel: ask for more.
    push_rgb, combo_rgb = (QColor(name).getRgb()[:3] for name in (painted["push_grabbed"][1], painted["combo"]))
    assert max(abs(a - b) for a, b in zip(push_rgb, combo_rgb, strict=True)) >= 32, (preset_id, painted["push_grabbed"], painted["combo"])


@pytest.mark.parametrize("base", ["parchment", "dark_wood"])
def test_the_accent_never_reaches_the_tab_page(qt_app, base: str) -> None:
    pages = {accent: _painted(qt_app, f"editor_{base}_{accent}")["tab_page"] for accent in ("red", "blue", "green", "purple")}
    assert len(set(pages.values())) == 1, pages


def test_a_custom_theme_can_flip_its_presets_polarity_with_the_surface_swatch(qt_app) -> None:
    """Light with a dark window: without `surface` the tab page stays Fusion's
    near-white under the white window text."""
    flipped = {"window": "#202020", "window_text": "#ffffff"}
    _use(qt_app, "light", flipped)
    assert themes.contrast_ratio("#ffffff", _tab_page_pixel(qt_app)) < 1.5
    _use(qt_app, "light", {**flipped, "surface": "#303030"})
    assert themes.contrast_ratio("#ffffff", _tab_page_pixel(qt_app)) >= 4.5


# --- status text colours ---------------------------------------------------------


@pytest.mark.parametrize("preset_id", list(themes.PRESETS))
def test_painted_tab_page_is_a_real_tab_widgets_page_colour(qt_app, preset_id: str) -> None:
    from descape.viewer_dialogs import painted_tab_page

    _use(qt_app, preset_id)
    assert painted_tab_page(qt_app.palette()) == _tab_page_pixel(qt_app)


def _find_error_color() -> str:
    from descape import find_dialog

    return find_dialog._error_style().removeprefix("color: ").removesuffix(";")


@pytest.mark.parametrize("preset_id", list(themes.PRESETS))
def test_status_text_reaches_the_floor_on_its_own_background(qt_app, preset_id: str) -> None:
    from PyQt5.QtGui import QPalette

    from descape.viewer import status_color

    _use(qt_app, preset_id)
    floor = 7.0 if preset_id in themes.HIGH_CONTRAST_PRESETS else 4.5
    page = _tab_page_pixel(qt_app)
    window = qt_app.palette().color(QPalette.Active, QPalette.Window).name()
    samples = (
        ("ok on a tab page", status_color(True, on_tab_page=True), page),
        ("error on a tab page", status_color(False, on_tab_page=True), page),
        ("Find's pattern error", _find_error_color(), page),
        ("ok on the window", status_color(True, on_tab_page=False), window),
        ("error on the window", status_color(False, on_tab_page=False), window),
    )
    for name, color, background in samples:
        ratio = themes.contrast_ratio(color, background)
        assert ratio >= floor, f"{preset_id} {name}: {color} on {background} = {ratio:.2f} < {floor}"


def test_status_text_follows_a_window_override_that_flips_polarity(qt_app) -> None:
    """Dark with a light window: the label on the window is judged against
    that window alone, not against the still-dark tab page as well."""
    from descape.viewer import status_color

    _use(qt_app, "dark", {"window": "#f0f0f0"})
    color = status_color(False, on_tab_page=False)
    assert themes.contrast_ratio(color, "#f0f0f0") >= 4.5
    assert themes.contrast_ratio(status_color(False, on_tab_page=True), _tab_page_pixel(qt_app)) >= 4.5


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
    assert settings.get_theme() == themes.THEME_DEFAULT
    assert settings.get_theme_colors() == {"window": "#101010"}
    assert dialog.theme_combo.currentData() == themes.THEME_DEFAULT


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


@pytest.mark.gui
def test_the_button_and_surface_swatches_each_show_and_change_their_own_palette(qt_app, dialog, monkeypatch) -> None:
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtWidgets import QApplication, QColorDialog

    swatches = dialog._theme_swatches
    assert list(swatches) == list(themes.ROLE_IDS) and len(swatches) == 14
    _choose(dialog, "light")  # away from the default first, so the next choice is a real switch
    _choose(dialog, "editor_parchment_red")
    colors = themes.PRESETS["editor_parchment_red"][1]
    assert swatches["button"].toolTip() == colors["button"]
    assert swatches["button_text"].toolTip() == colors["button_text"]
    assert swatches["surface"].toolTip() == colors["surface"]
    assert swatches["surface_text"].toolTip() == colors["surface_text"]

    app_before = _all_colors(qt_app.palette())
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *_args, **_kw: QColor("#336699")))
    swatches["button"].click()
    assert settings.get_theme_colors() == {"button": "#336699"}
    assert QApplication.palette("QPushButton").color(QPalette.Active, QPalette.Button).name() == "#336699"
    assert _all_colors(qt_app.palette()) == app_before
    assert (swatches["button"].toolTip(), swatches["surface"].toolTip()) == ("#336699", colors["surface"])

    push_before = _all_colors(QApplication.palette("QPushButton"))
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *_args, **_kw: QColor("#996633")))
    swatches["surface"].click()
    assert settings.get_theme_colors() == {"button": "#336699", "surface": "#996633"}
    assert qt_app.palette().color(QPalette.Active, QPalette.Button).name() == "#996633"
    assert _all_colors(QApplication.palette("QPushButton")) == push_before
    assert (swatches["button"].toolTip(), swatches["surface"].toolTip()) == ("#336699", "#996633")


@pytest.mark.gui
def test_reset_all_to_preset_takes_a_custom_accent_off_the_push_buttons(qt_app, dialog) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    _choose(dialog, "dim")
    settings.set_theme_color("button", "#8b140d")
    dialog._apply_chrome_theme()
    assert QApplication.palette("QPushButton").color(QPalette.Active, QPalette.Button).name() == "#8b140d"
    dialog.theme_reset_button.click()
    dim_button = themes.PRESETS["dim"][1]["button"]
    assert QApplication.palette("QPushButton").color(QPalette.Active, QPalette.Button).name() == dim_button
    assert dialog._theme_swatches["button"].toolTip() == dim_button


@pytest.mark.gui
def test_a_theme_change_restyles_a_set_status_label_and_skips_one_never_set(qt_app) -> None:
    """Light to parchment on purpose. Dark and Solarized dark derive the same
    status colours, so a change between those two passes with no restyle."""
    from descape.viewer import status_color

    _use(qt_app, "light")
    dialog, window = conftest.dialog_and_window()
    try:
        ok = dialog._install_status_ok
        assert ok is not None and dialog._autosave_dir_status_ok is None
        before = dialog.install_status_label.styleSheet()
        assert before == f"color: {status_color(ok, on_tab_page=True)};"
        assert dialog.autosave_dir_status_label.styleSheet() == ""

        _choose(dialog, "editor_parchment_red")
        after = dialog.install_status_label.styleSheet()
        assert after != before
        assert after == f"color: {status_color(ok, on_tab_page=True)};"
        assert dialog.autosave_dir_status_label.styleSheet() == ""
    finally:
        dialog.close()
        conftest.close_window(window)


@pytest.mark.gui
def test_mirrors_blocked_summary_is_coloured_for_the_window_not_a_tab_page(qt_app, monkeypatch) -> None:
    from descape.mirror_tools import MirrorPlan
    from descape.viewer import MirrorDialog, status_color

    # Dim, where the window and the tab page derive different error shades.
    _use(qt_app, "dim")
    assert status_color(False, on_tab_page=False) != status_color(False, on_tab_page=True)
    blocked = MirrorPlan(changes=[(0, (5, 0, -1))], source_indices=frozenset(), elevation_violations=[(0, 1)])
    monkeypatch.setattr(MirrorDialog, "_compute_plan", lambda _self: blocked)
    window = conftest.terrain_edit_window()
    try:
        dialog = MirrorDialog(window)
        try:
            assert "BLOCKED" in dialog.summary_label.text()
            assert dialog.summary_label.styleSheet() == f"color: {status_color(False, on_tab_page=False)};"
        finally:
            dialog.close()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_theme_change_restyles_both_of_finds_pattern_error_labels(qt_app) -> None:
    from descape import find_dialog

    _use(qt_app, "light")
    dialog = find_dialog.FindDialog()
    try:
        labels = (dialog.objects_tab.pattern_error, dialog.triggers_tab.pattern_error)
        before = find_dialog._error_style()
        assert [label.styleSheet() for label in labels] == [before, before]

        _use(qt_app, "editor_parchment_red")
        after = find_dialog._error_style()
        assert after != before
        assert [label.styleSheet() for label in labels] == [after, after]
    finally:
        dialog.close()
