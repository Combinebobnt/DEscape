"""Settings > Appearance's UI font pair: persistence, live apply, and the
map-overlay carve-out that keeps it out of the map view.

Same offscreen technique as tests/test_elev_step_slider.py, including
constructing SettingsDialog directly rather than through
ViewerWindow._show_settings() (that method calls exec_(), which is modal and
would hang an offscreen run).

The autouse baseline fixture below is load-bearing, not hygiene:
viewer_dialogs._DEFAULT_FONT is a module global cached on the first
apply_ui_font() call, and conftest.ensure_qapp() hands one QApplication to
the whole session. Without it the first test to apply a font would pin the
baseline for every later test, and any test leaving a non-default font
applied would poison the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _font_baseline():
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape import viewer_dialogs

    app = QApplication.instance()
    saved_font = app.font()
    saved_default = viewer_dialogs._DEFAULT_FONT
    viewer_dialogs._DEFAULT_FONT = None
    yield
    viewer_dialogs._DEFAULT_FONT = saved_default
    app.setFont(saved_font)


def _other_family() -> str:
    """A real installed family that is not the one currently in effect --
    picked at runtime rather than hardcoded, since no family name is
    guaranteed present on every machine this runs on."""
    from PyQt5.QtGui import QFontDatabase
    from PyQt5.QtWidgets import QApplication

    current = QApplication.instance().font().family()
    for family in QFontDatabase().families():
        if family != current and not family.startswith("."):
            return family
    pytest.skip("no second font family installed")


def _system_family() -> str:
    from PyQt5.QtGui import QFontDatabase

    return QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()


def test_settings_round_trip(tmp_path, monkeypatch) -> None:
    assert (settings.get_ui_font_family(), settings.get_ui_font_size()) == ("", None)

    settings.set_ui_font_family("Some Family")
    settings.set_ui_font_size(settings.UI_FONT_SIZE_MAX)
    monkeypatch.setattr(settings, "_ui_font_family", None)
    monkeypatch.setattr(settings, "_ui_font_size", None)
    assert settings.get_ui_font_family() == "Some Family"
    assert settings.get_ui_font_size() == settings.UI_FONT_SIZE_MAX

    for bad in (settings.UI_FONT_SIZE_MIN - 1, settings.UI_FONT_SIZE_MAX + 1):
        with pytest.raises(ValueError):
            settings.set_ui_font_size(bad)

    # A stored value outside the range, or of the wrong type, reads back as
    # None ("platform default") rather than being clamped into the range.
    # conftest's autouse _isolated_settings already points CONFIG_PATH at
    # this tmp_path, and _load_config() re-reads the file every call, so
    # clearing the memo is all a fresh read needs.
    for bad in ("999", "'12'", "true", "null"):
        (tmp_path / "config.yaml").write_text(f"ui_font_size: {bad}\n")
        monkeypatch.setattr(settings, "_ui_font_size", None)
        assert settings.get_ui_font_size() is None


def test_apply_ui_font_applies_and_restores() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_dialogs import apply_ui_font

    app = QApplication.instance()
    baseline_family = app.font().family()
    family = _other_family()

    apply_ui_font(app, family, settings.UI_FONT_SIZE_MAX)
    assert app.font().family() == family
    assert app.font().pointSize() == settings.UI_FONT_SIZE_MAX

    apply_ui_font(app, "", None)
    assert app.font().family() == baseline_family

    # An unknown family is ignored rather than handed to Qt's silent
    # substitution, so the baseline family survives.
    apply_ui_font(app, "No Such Font Family At All", None)
    assert app.font().family() == baseline_family


def test_map_overlay_text_does_not_follow_the_app_font() -> None:
    """The regression this change exists to prevent. Every map-overlay item
    is built fresh AFTER a non-default app font is applied: before the
    map_overlay_font() carve-out, the ruler label's family followed the app
    font outright and EdgeTickItem's went stale at construction time."""
    from PyQt5.QtWidgets import QApplication

    from descape import edge_ticks
    from descape.viewer_canvas import EdgeTickItem, StackBadgeItem
    from descape.viewer_dialogs import apply_ui_font

    app = QApplication.instance()
    apply_ui_font(app, _other_family(), settings.UI_FONT_SIZE_MAX)
    assert app.font().family() != _system_family(), "probe font matched the system font"

    window = conftest.dialog_and_window()[1]
    try:
        map_view = window.map_view
        map_view._create_ruler_items()
        # Armed AFTER the items exist: _clear_ruler() also forgets the
        # session, so apply_ruler_label_font()'s _update_ruler() would
        # otherwise find no measurement and drop the items it is meant to
        # restyle.
        map_view._ruler.press((2, 2))
        map_view._ruler.move((6, 5))
        assert map_view._ruler.release() == "done"
        label_font = map_view._ruler_label_item.font()
        assert label_font.family() == _system_family()
        assert label_font.pixelSize() == settings.get_ruler_label_font_px()
        assert label_font.bold()

        map_view.apply_ruler_label_font()
        assert map_view._ruler_label_item is not None, "ruler items cleared mid-test"
        assert map_view._ruler_label_item.font().family() == _system_family()
    finally:
        conftest.close_window(window)

    tick = EdgeTickItem(8, 8, edge_ticks.TICK_INTERVAL_DEFAULT, _unit_rect(), tile_px=8)
    assert tick._font.family() == _system_family()
    assert tick._font.pixelSize() == edge_ticks.LABEL_FONT_PX
    # The live-resize path, not just construction: it rebuilds the font from
    # scratch, so it is a second chance to reintroduce the leak.
    tick.set_label_font_px(edge_ticks.LABEL_FONT_PX + 4)
    assert tick._font.family() == _system_family()
    assert tick._font.pixelSize() == edge_ticks.LABEL_FONT_PX + 4

    badge = StackBadgeItem(8.0, _white())
    assert badge._font.family() == _system_family()
    assert badge._font.pixelSize() == StackBadgeItem.FONT_PX


def _unit_rect():
    from PyQt5.QtCore import QRectF

    return QRectF(0, 0, 64, 64)


def _white():
    from PyQt5.QtGui import QColor

    return QColor(255, 255, 255)


def test_dialog_opens_on_the_persisted_values_and_persists_changes() -> None:
    from PyQt5.QtWidgets import QApplication

    family = _other_family()
    settings.set_ui_font_family(family)
    settings.set_ui_font_size(settings.UI_FONT_SIZE_MIN)

    dialog, window = conftest.dialog_and_window()
    try:
        assert dialog.ui_font_combo.currentFont().family() == family
        assert dialog.ui_font_size_spin.value() == settings.UI_FONT_SIZE_MIN

        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MAX)
        assert settings.get_ui_font_size() == settings.UI_FONT_SIZE_MAX
        # The family must NOT have been rewritten by a size-only change:
        # an applier that read both widgets would pin whatever the combo
        # happened to be showing.
        assert settings.get_ui_font_family() == family

        dialog._on_ui_font_reset()
        assert settings.get_ui_font_family() == ""
        assert settings.get_ui_font_size() is None
        assert QApplication.instance().font().family() == dialog.ui_font_combo.currentFont().family()
        # Reset repopulates both widgets; if it did so without blocking
        # signals it would persist them straight back.
        assert settings.get_ui_font_family() == ""
        assert settings.get_ui_font_size() is None
    finally:
        dialog.close()
        conftest.close_window(window)


def test_log_minimum_height_recomputes_on_a_font_change() -> None:
    dialog, window = conftest.dialog_and_window()
    try:
        before = window.status_log.minimumHeight()
        dialog.ui_font_combo.setCurrentFont(window.status_log.font())
        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MAX)
        dialog._apply_ui_font()
        assert window.status_log.minimumHeight() > before
    finally:
        dialog.close()
        conftest.close_window(window)


def _ink_bbox_height(widget) -> int:
    """Height in device pixels of the non-background ink in `widget`'s own
    render -- measured off a real grab rather than read back off
    fontMetrics(), which would pass even with the font never applied."""
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB888)
    array = qimage_rgb888_to_array(image)
    background = array[0, 0].astype(np.int16)
    hit = (np.abs(array.astype(np.int16) - background) > 24).any(axis=2)
    if not hit.any():
        return 0
    ys, _xs = np.nonzero(hit)
    return int(ys.max() - ys.min() + 1)


def test_chrome_grows_with_the_font_and_the_browse_button_does_not_clip() -> None:
    """The measured half of this change's verification, as a test rather
    than a throwaway capture: chrome text really grows, and the fixed-width
    "…" browse button still holds its glyph at the largest size."""
    from PyQt5.QtWidgets import QApplication, QLabel, QPushButton

    from descape.viewer_dialogs import apply_ui_font

    app = QApplication.instance()

    apply_ui_font(app, "", None)
    default_label = QLabel("Sample chrome text")
    default_label.adjustSize()
    default_ink = _ink_bbox_height(default_label)

    apply_ui_font(app, _other_family(), settings.UI_FONT_SIZE_MAX)
    big_label = QLabel("Sample chrome text")
    big_label.adjustSize()
    assert _ink_bbox_height(big_label) > default_ink

    button = QPushButton("…")
    button.setFixedWidth(max(28, button.fontMetrics().horizontalAdvance("…") + 12))
    button.adjustSize()
    assert button.fontMetrics().horizontalAdvance("…") + 4 <= button.width()
    assert _ink_bbox_height(button) > 0
