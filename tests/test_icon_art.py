"""The drawn icon set: it renders, it is a set (no two alike), its neutrals follow
the ink, and every accent clears 3:1 on every preset once adapted. Whether an icon
reads as its tool is a judgement for eyes, not for these."""

from __future__ import annotations

import hashlib

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    conftest.ensure_qapp()


def _black():
    from PyQt5.QtGui import QColor

    return QColor("#000000")


def _white():
    from PyQt5.QtGui import QColor

    return QColor("#ffffff")


def _pixels(pixmap):
    # pixelColor, not pixel(): a premultiplied QRgb reads alpha 255 on a half-transparent pixel.
    image = pixmap.toImage()
    return [image.pixelColor(x, y) for y in range(image.height()) for x in range(image.width())]


def _fingerprint(name, ink, size=24, backdrops=()):
    from descape import icon_art

    image = icon_art.paint_icon(name, size, ink, backdrops=backdrops).toImage()
    return hashlib.sha256(image.bits().asstring(image.byteCount())).hexdigest()


def _names():
    from descape import icon_art

    return sorted(icon_art.ICON_SHAPES)


def test_the_painter_scales_the_grid_to_the_size_asked_for():
    from descape.icon_art import Role, paint_shapes

    for size in (16, 24, 32):
        image = paint_shapes((("rect", Role.INK, (0, 0, 24, 24)),), size, _black()).toImage()
        assert (image.width(), image.height()) == (size, size)
        assert image.pixelColor(size // 2, size // 2) == _black()
        assert image.pixelColor(0, 0).alpha() > 0


def test_the_painter_honours_the_device_pixel_ratio():
    from descape.icon_art import Role, paint_shapes

    pixmap = paint_shapes((("rect", Role.INK, (0, 0, 24, 24)),), 24, _black(), dpr=2.0)
    assert pixmap.width() == 48
    assert pixmap.devicePixelRatio() == 2.0


def test_a_2x_render_covers_the_same_part_of_the_grid():
    """A full-grid fill cannot catch a doubled painter scale; a quarter does."""
    from descape.icon_art import Role, paint_shapes

    for dpr in (1.0, 2.0):
        image = paint_shapes((("rect", Role.INK, (0, 0, 12, 12)),), 24, _black(), dpr=dpr).toImage()
        half = image.width() // 2
        assert image.pixelColor(half - 1, half - 1).alpha() == 255, dpr
        assert image.pixelColor(half, half).alpha() == 0, dpr
        assert image.pixelColor(half, 0).alpha() == 0, dpr


def test_ink_soft_is_the_ink_at_reduced_alpha():
    from PyQt5.QtGui import QColor

    from descape import icon_art
    from descape.icon_art import Role, paint_shapes

    pixel = paint_shapes((("rect", Role.INK_SOFT, (0, 0, 24, 24)),), 24, QColor("#112233")).toImage().pixelColor(12, 12)
    assert pixel.alpha() == icon_art.SOFT_ALPHA
    for got, wanted in zip(pixel.getRgb()[:3], (0x11, 0x22, 0x33), strict=True):
        assert abs(got - wanted) <= 1  # one premultiply round trip


def test_an_unknown_primitive_is_refused():
    from descape.icon_art import Role, paint_shapes

    with pytest.raises(ValueError, match="not a known icon primitive"):
        paint_shapes((("squiggle", Role.INK, (0, 0, 1, 1)),), 24, _black())


def test_a_name_with_no_artwork_is_refused():
    from descape import icon_art, ui_icons

    with pytest.raises(KeyError, match="has no artwork"):
        icon_art.paint_icon("not-an-icon", 24, _black())
    with pytest.raises(KeyError, match="has no artwork"):
        ui_icons.icon("not-an-icon")


def test_an_empty_icon_paints_nothing():
    from descape.icon_art import paint_shapes

    assert all(pixel.alpha() == 0 for pixel in _pixels(paint_shapes((), 24, _black())))


def test_every_literal_colour_is_a_named_accent():
    """The contrast test walks ACCENTS, so a colour written straight into a shape would escape it."""
    from descape import icon_art
    from descape.icon_art import Role

    allowed = {accent.rgb() for accent in icon_art.ACCENTS}
    strays = [
        (name, paint.name())
        for name, shapes in icon_art.ICON_SHAPES.items()
        for _kind, paint, _points in shapes
        if not isinstance(paint, Role) and paint.rgb() not in allowed
    ]
    assert not strays, strays


def test_every_icon_has_an_ink_part():
    from descape import icon_art
    from descape.icon_art import Role

    inkless = [n for n, shapes in icon_art.ICON_SHAPES.items() if not any(isinstance(p, Role) for _k, p, _pts in shapes)]
    assert not inkless, inkless


@pytest.mark.parametrize("size", (16, 24, 32))
def test_every_icon_renders_something(size):
    from descape import icon_art

    blank = [n for n in _names() if not any(p.alpha() > 0 for p in _pixels(icon_art.paint_icon(n, size, _black())))]
    assert not blank, f"painted nothing at {size}px: {blank}"


@pytest.mark.parametrize("size", (16, 24))
def test_no_two_icons_render_alike(size):
    seen: dict[str, str] = {}
    for name in _names():
        digest = _fingerprint(name, _black(), size)
        assert digest not in seen, f"{name} renders identically to {seen[digest]} at {size}px"
        seen[digest] = name


def test_every_icon_changes_with_the_ink():
    """Catches an icon drawn entirely in accents: it would stay put while the rest re-inked."""
    same = [n for n in _names() if _fingerprint(n, _black()) == _fingerprint(n, _white())]
    assert not same, same


def test_every_table_name_has_artwork():
    """TOOL_ICONS is generated from settings.TOOLS, so this is also "every tool has an icon"."""
    from descape import icon_art, settings, ui_icons

    assert set(ui_icons.TOOL_ICONS) == {tool.tool_id for tool in settings.TOOLS}
    named = {
        *ui_icons.TOOL_ICONS.values(),
        *ui_icons.ACTION_ICONS.values(),
        *ui_icons.WINDOW_ICONS.values(),
        ui_icons.CLEAR_RECENT_ICON,
        ui_icons.NEW_SIZE_ICON,
    }
    assert named - set(icon_art.ICON_SHAPES) == set()
    # And the reverse: artwork nothing asks for is a drawing no one sees.
    assert set(icon_art.ICON_SHAPES) - named == set()


def test_every_action_icon_key_is_a_rebindable_action():
    from descape import settings, ui_icons

    rebindable = {action_id for action_id, _label, _default in settings.REBINDABLE_ACTIONS}
    assert set(ui_icons.ACTION_ICONS) - rebindable == set()


def _preset_backdrops():
    """(preset id, backdrops) as apply_theme() builds each preset's app palette."""
    from PyQt5.QtWidgets import QStyleFactory

    from descape import icon_art, themes
    from descape.viewer_dialogs import build_theme_palette

    base = QStyleFactory.create("Fusion").standardPalette()
    for preset_id in themes.PRESETS:
        yield preset_id, icon_art.backdrops_of(build_theme_palette(base, preset_id, {}))


def test_the_backdrops_include_the_menu_panel_fusion_paints():
    """Measured: a shown QMenu's panel pixel is Base.lighter(108), not Window."""
    from PyQt5.QtGui import QColor, QPalette
    from PyQt5.QtWidgets import QMenu, QStyleFactory

    from descape import icon_art

    palette = QStyleFactory.create("Fusion").standardPalette()
    palette.setColor(QPalette.Base, QColor("#d6b597"))
    palette.setColor(QPalette.Window, QColor("#c7a07d"))
    menu = QMenu()
    menu.setStyle(QStyleFactory.create("Fusion"))
    menu.setPalette(palette)
    menu.addAction("A menu entry wide enough to sample")
    menu.adjustSize()
    panel = menu.grab().toImage()
    pixel = panel.pixelColor(panel.width() - 6, panel.height() // 2)
    assert pixel.rgb() in {c.rgb() for c in icon_art.backdrops_of(palette)}
    assert pixel.rgb() != palette.color(QPalette.Window).rgb()


def test_every_accent_clears_three_to_one_on_every_preset_once_adapted():
    from descape import icon_art, themes

    failures = []
    for preset_id, backdrops in _preset_backdrops():
        for accent in icon_art.ACCENTS:
            shade = icon_art.adapted_accent(accent, backdrops)
            for backdrop in backdrops:
                ratio = themes.contrast_ratio(shade.name(), backdrop.name())
                if ratio < icon_art.ACCENT_CONTRAST:
                    failures.append(f"{accent.name()}->{shade.name()} on {preset_id} {backdrop.name()}: {ratio:.2f}")
    assert not failures, "\n".join(failures)


def test_the_ink_clears_eight_to_one_on_every_preset_and_keeps_strong_inks():
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QStyleFactory

    from descape import icon_art, themes
    from descape.viewer_dialogs import build_theme_palette

    base = QStyleFactory.create("Fusion").standardPalette()
    failures = []
    deepened = []
    for preset_id in themes.PRESETS:
        palette = build_theme_palette(base, preset_id, {})
        backdrops = icon_art.backdrops_of(palette)
        own = palette.color(QPalette.ButtonText)
        ink = icon_art.adapted_colour(own, backdrops, icon_art.INK_CONTRAST)
        worst = min(themes.contrast_ratio(ink.name(), b.name()) for b in backdrops)
        # Pure black or white is as far as it can go (Dim's white reaches 7.28).
        if worst < icon_art.INK_CONTRAST and ink.name() not in ("#000000", "#ffffff"):
            failures.append(f"{preset_id}: {ink.name()} {worst:.2f}")
        if ink.rgb() != own.rgb():
            deepened.append(preset_id)
    assert not failures, "\n".join(failures)
    assert "editor_parchment_red" in deepened
    assert "light" not in deepened and "dark" not in deepened


def test_ink_colour_reads_the_adapted_ink():
    from PyQt5.QtGui import QPalette

    from descape import icon_art
    from descape.viewer_dialogs import apply_theme

    with conftest.restoring_app_theme() as app:
        apply_theme(app, "editor_parchment_red", {})
        own = app.palette().color(QPalette.ButtonText)
        ink = icon_art.ink_colour()
        assert ink.rgb() != own.rgb()
        assert ink.lightness() < own.lightness()


def test_an_accent_that_already_reads_is_left_alone():
    from PyQt5.QtGui import QColor

    from descape import icon_art

    # The base accents were tuned against these two; each clears 3:1 on both unchanged.
    backdrops = (QColor("#efefef"), QColor("#353535"))
    for accent in icon_art.ACCENTS:
        assert icon_art.adapted_accent(accent, backdrops).rgb() == accent.rgb(), accent.name()


def test_the_accents_change_with_the_backdrop():
    """The same icon on the Wood window and on the Dark window: its accents differ, its ink is the same."""
    from PyQt5.QtGui import QColor

    wood = (QColor("#c7a07d"), QColor("#c7a07d"))
    dark = (QColor("#353535"), QColor("#353535"))
    assert _fingerprint("tool-fill", _black(), backdrops=wood) != _fingerprint("tool-fill", _black(), backdrops=dark)


def test_icon_is_cached_per_colour_set():
    from descape import ui_icons

    first = ui_icons.icon("file-open")
    assert ui_icons.icon("file-open").cacheKey() == first.cacheKey()
    assert not first.pixmap(24, 24).isNull()
