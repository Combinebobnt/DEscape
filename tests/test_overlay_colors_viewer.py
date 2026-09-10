"""Settings > Appearance's per-element tool overlay color grid --
descape.viewer.SettingsDialog's swatch rows and MapView.apply_overlay_colors()'s
live-push path. Same offscreen-ViewerWindow technique tests/test_fill_tool.py
documents: QT_QPA_PLATFORM=offscreen, one shared QApplication via
conftest.ensure_qapp(), and every window edit_history.mark_saved()'d before
close() so closeEvent's discard prompt can't block forever offscreen.

SettingsDialog is constructed directly rather than through
ViewerWindow._show_settings() (that method calls exec_(), which is modal and
would hang an offscreen run) -- same precedent as tests/test_keybinds.py and
tests/test_elev_step_slider.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import settings
from descape.scenario_io import BLANK_TEMPLATE_PATH
from test_ruler_viewer import _drag, _ruler_window

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _dialog_and_window():
    from descape.viewer import SettingsDialog, ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    return SettingsDialog(window), window


def test_every_overlay_color_prefix_has_a_section_title() -> None:
    """Reflective, mirroring test_settings.py's memoized-globals check: an
    unlisted prefix still renders (falls back to section.title()), so
    nothing would fail loudly if this decayed on its own -- the Appearance
    tab would just grow an ungrouped, oddly-cased header."""
    from descape.viewer import SettingsDialog

    prefixes = {color_id.split("_", 1)[0] for color_id, _label, _default in settings.OVERLAY_COLORS}
    assert prefixes <= set(SettingsDialog._OVERLAY_SECTION_TITLES)


def test_every_overlay_color_has_a_matching_swatch_row() -> None:
    """Both directions, mirroring test_keybinds.py's test_every_rebindable_
    action_has_a_matching_keybind_action: a missing row leaves that color
    permanently stuck at whatever it happens to be, and a stray row would be
    dead UI for an id nothing else recognizes."""
    dialog, window = _dialog_and_window()
    try:
        declared_ids = {color_id for color_id, _label, _default in settings.OVERLAY_COLORS}
        assert declared_ids <= set(dialog._overlay_swatches)
        assert set(dialog._overlay_swatches) <= declared_ids
    finally:
        dialog.close()
        window.edit_history.mark_saved()
        window.close()


def test_picking_a_ruler_color_paints_the_live_measurement(tmp_path, monkeypatch) -> None:
    """The real proof, not just that settings.py round-trips: pin ruler_line
    to a distinctive magenta before the window opens, draw a measurement on
    uniform terrain, and check the actual rendered pixels -- both that the
    configured color shows up AND that the old default doesn't linger
    anywhere (e.g. a class-constant draw site that got missed)."""
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    (tmp_path / "config.yaml").write_text("overlay_colors:\n  ruler_line: '#ff00ff'\n")
    monkeypatch.setattr(settings, "_overlay_colors", None)

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        pixmap = map_view.viewport().grab()
        array = qimage_rgb888_to_array(pixmap.toImage().convertToFormat(QImage.Format_RGB888))

        magenta = (255, 0, 255)
        default_orange = tuple(
            int(settings.get_default_overlay_color("ruler_line")[i : i + 2], 16) for i in (1, 3, 5)
        )
        hit_magenta = (np.abs(array.astype(np.int16) - np.array(magenta, dtype=np.int16)) <= 8).all(axis=2)
        hit_default = (np.abs(array.astype(np.int16) - np.array(default_orange, dtype=np.int16)) <= 8).all(
            axis=2
        )
        assert hit_magenta.any(), "configured ruler color never appeared in the capture"
        assert not hit_default.any(), "default ruler color still showing despite the override"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_live_apply_repaints_without_replacing_the_item(tmp_path, monkeypatch) -> None:
    """apply_overlay_colors() must update the existing scene item's pen/brush
    in place, not remove-and-recreate it -- a recreate would drop whatever
    else (z-order, event filters) the original item carried."""
    from PyQt5.QtGui import QColor

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        item_id_before = id(map_view._ruler_line_item)

        settings.set_overlay_color("ruler_line", "#ff00ff")
        map_view.apply_overlay_colors()

        assert id(map_view._ruler_line_item) == item_id_before
        assert map_view._ruler_line_item.pen().color() == QColor("#ff00ff")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_apply_overlay_colors_clears_the_stale_edit_highlight() -> None:
    """_clear_highlight() runs as part of apply_overlay_colors() (the edit
    highlight has no in-place update path) -- confirm the highlight actually
    goes away rather than keep showing the old color until the next hover."""
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    try:
        map_view = window.map_view
        window._on_tool_selected("draw")
        map_view._update_highlight(5, 5)
        assert map_view._highlight_outline_item is not None

        settings.set_overlay_color("highlight_outline", "#123456")
        map_view.apply_overlay_colors()

        assert map_view._highlight_outline_item is None
        assert map_view._highlight_key is None
    finally:
        window.edit_history.mark_saved()
        window.close()
