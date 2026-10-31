"""TASK-019 step 3: raising Settings > Appearance's UI font size widens the
Map Options and Triggers panes on the next mode switch.

Drives the real dialog spin box and its applier on a shown viewer, then enters
each mode from a 150 px pane, so the scaled floor is read through the same
FontScaledWidth the viewer uses at runtime. Steps 1, 2 and 4 run at 144/168 DPI
in tests/test_font_dpi_sweep.py.
"""

from __future__ import annotations

import math

import pytest
from test_gh142_panel_widths import _NARROW_PX, _blank_window, _enter, check_terrain_fits_its_drag_floor
from test_unit_edit_viewer import _no_install_env  # noqa: F401 -- autouse fixture

from descape import settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture(autouse=True)
def _font_baseline():
    """Puts the shared QApplication's font and apply_ui_font()'s cached default back
    (tests/test_ui_font.py's fixture: a font left applied poisons the worker)."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    from descape import gc_hold, viewer_dialogs

    app = QApplication.instance()
    saved_font = app.font()
    saved_default = viewer_dialogs._DEFAULT_FONT
    viewer_dialogs._DEFAULT_FONT = None
    yield
    viewer_dialogs._DEFAULT_FONT = saved_default
    with gc_hold.paused():
        app.setFont(saved_font)


def _scaled(base_px: int) -> int:
    from PyQt5.QtGui import QFontMetricsF
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_common import _BASELINE_AVG_CHAR_PX

    ratio = QFontMetricsF(QApplication.font()).averageCharWidth() / _BASELINE_AVG_CHAR_PX
    return max(base_px, math.ceil(base_px * ratio))


@pytest.mark.parametrize(("mode", "base_px"), [("Map Options", 300), ("Triggers", 340)])
def test_raising_the_ui_font_to_max_widens_the_panel_pane_to_the_scaled_floor(mode: str, base_px: int) -> None:
    """Pass conditions, after the Appearance spin box is set to UI_FONT_SIZE_MAX and applied:
    - the panel's MIN_USEFUL_WIDTH is the font-scaled floor, and wider than its 96 DPI base;
    - entering the mode from a 150 px pane leaves the pane at exactly that scaled floor;
    - after the font goes back to UI_FONT_SIZE_MIN, re-entering the mode from that widened
      pane leaves it as it was (the floor never shrinks a pane).
    """
    from PyQt5.QtWidgets import QApplication

    from descape.map_options_panel import MapOptionsPanel
    from descape.trigger_panel import TriggerPanel
    from descape.viewer import SettingsDialog

    panel_cls = {"Map Options": MapOptionsPanel, "Triggers": TriggerPanel}[mode]
    window = _blank_window()
    dialog = SettingsDialog(window)
    try:
        assert base_px == panel_cls.MIN_USEFUL_WIDTH, "the pinned test font should give the base floor"

        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MAX)
        dialog._ui_font_apply_timer.stop()
        dialog._apply_ui_font()
        QApplication.processEvents()
        assert QApplication.font().pointSize() == settings.UI_FONT_SIZE_MAX

        floor = _scaled(base_px)
        assert floor > base_px, f"UI_FONT_SIZE_MAX should scale the {base_px} px floor up"
        assert floor == panel_cls.MIN_USEFUL_WIDTH
        widened = _enter(window, mode, from_px=_NARROW_PX)
        assert widened == floor, f"{mode} pane {widened} px, scaled floor {floor} px"

        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MIN)
        dialog._ui_font_apply_timer.stop()
        dialog._apply_ui_font()
        QApplication.processEvents()
        assert base_px == panel_cls.MIN_USEFUL_WIDTH
        assert _enter(window, mode, from_px=widened) == widened
    finally:
        dialog.close()
        conftest.close_window(window)


def test_raising_the_ui_font_to_max_raises_terrains_drag_floor_and_it_still_fits() -> None:
    """TASK-551. Pass conditions, after the Appearance spin box is set to UI_FONT_SIZE_MAX and
    applied while in Terrain mode, with no mode switch: Terrain's drag floor is wider than at the pinned test font, and at it
    test_gh142_panel_widths.check_terrain_fits_its_drag_floor's conditions all hold."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import SettingsDialog

    window = _blank_window()
    dialog = SettingsDialog(window)
    try:
        window.mode_combo.setCurrentText("Terrain")  # already there: the font change alone must re-measure
        base_floor = window.terrain_panel.drag_floor()
        dialog.ui_font_size_spin.setValue(settings.UI_FONT_SIZE_MAX)
        dialog._ui_font_apply_timer.stop()
        dialog._apply_ui_font()
        QApplication.processEvents()
        assert QApplication.font().pointSize() == settings.UI_FONT_SIZE_MAX

        floor = check_terrain_fits_its_drag_floor(window)
        assert floor > base_floor, f"the bigger font left the floor at {floor} px (was {base_floor} px)"
    finally:
        dialog.close()
        conftest.close_window(window)
