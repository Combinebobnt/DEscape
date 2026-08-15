"""Tool-param toolbar widgets (Terrain type / Level / Brush) show only for the
tool that reads them, driven through a real offscreen ViewerWindow -- same
technique and default-tier rationale as tests/test_fill_tool.py.

Asserts on the captured QAction handles (terrain_param_combo_action,
level_param_spin_action, brush_size_spin_action, ...), never on the widgets'
own isVisible(): under QT_QPA_PLATFORM=offscreen with a never-shown window, a
widget's isVisible() is False unconditionally, which would make a
widget-level assertion fail in one direction and pass vacuously in the
other. QAction.isVisible() has no such dependency on the window ever being
shown.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Which of the two single-valued params (if either) is expected to be
# visible for each tool once a square map is loaded and Edit mode is active.
_EXPECTED_PARAM = {
    "pan": None,
    "terrain": "terrain",
    "fill": "terrain",
    "elevation": None,
    "set_level": "level",
}

# Brush size/shape is orthogonal to _EXPECTED_PARAM above -- Set Elevation
# shows both a level AND a brush, Elevate shows a brush with no
# _EXPECTED_PARAM entry at all. See settings.ToolDef.supports_brush.
_EXPECTED_BRUSH = {
    "pan": False,
    "terrain": True,
    "fill": False,
    "elevation": True,
    "set_level": True,
}


def _edit_window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Edit")
    return window


@pytest.mark.parametrize("tool", list(_EXPECTED_PARAM))
def test_tool_param_visibility_matches_active_tool(tool: str) -> None:
    window = _edit_window()
    try:
        window._on_tool_selected(tool)
        expected = _EXPECTED_PARAM[tool]
        expected_brush = _EXPECTED_BRUSH[tool]

        assert window.terrain_param_combo_action.isVisible() == (expected == "terrain")
        assert window.terrain_param_label_action.isVisible() == (expected == "terrain")
        assert window.level_param_spin_action.isVisible() == (expected == "level")
        assert window.level_param_label_action.isVisible() == (expected == "level")
        assert window.brush_size_spin_action.isVisible() == expected_brush
        assert window.brush_shape_combo_action.isVisible() == expected_brush
        assert window.brush_param_label_action.isVisible() == expected_brush
        # Elevate has no _EXPECTED_PARAM entry (expected is None) but DOES
        # have a brush -- the separator must track either group, not just
        # the single-valued param.
        assert window.tool_param_separator_action.isVisible() == (expected is not None or expected_brush)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_level_spin_dead_while_terrain_selected() -> None:
    """The discriminating check: gating the Level spinbox on its own
    enabled state alone (orthogonal to visibility) would leave it live
    while hidden whenever Terrain is active. Terrain's ]/[ keys are alive
    for a different reason now -- see test_brush_size_owns_keys_over_level
    below -- so they are not asserted dead here."""
    window = _edit_window()
    try:
        window._on_tool_selected("terrain")
        assert not window.level_param_spin_action.isVisible()
        assert not window.elevation_level_spin.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_brush_size_owns_keys_over_level() -> None:
    """]/[ (tool_value_inc/dec_action) drive brush size for any
    supports_brush tool, falling back to Set Elevation's Level only when
    the active tool has no brush at all. Set Elevation is the one tool
    with both a level AND a brush -- brush wins there, which is the
    overlap this test pins."""
    window = _edit_window()
    try:
        window._on_tool_selected("terrain")
        assert window.tool_value_inc_action.isEnabled()
        before = window.brush_size_spin.value()
        window.tool_value_inc_action.trigger()
        assert window.brush_size_spin.value() == before + 1
        assert window.elevation_level_spin.value() == 0  # untouched

        window._on_tool_selected("set_level")
        window.brush_size_spin.setValue(3)
        level_before = window.elevation_level_spin.value()
        window.tool_value_inc_action.trigger()
        assert window.brush_size_spin.value() == 4
        assert window.elevation_level_spin.value() == level_before  # brush wins, level untouched

        window._on_tool_selected("pan")
        assert not window.tool_value_inc_action.isEnabled()
        assert not window.tool_value_dec_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_tool_params_hidden_before_map_loaded() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.terrain_param_combo_action.isVisible()
        assert not window.level_param_spin_action.isVisible()
        assert not window.brush_size_spin_action.isVisible()
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_tool_params_hidden_in_view_mode() -> None:
    """View mode forces the active tool back to Pan (on_mode_changed), so
    Terrain's own params stay hidden even though they were showing moments
    before the mode switch."""
    window = _edit_window()
    try:
        window._on_tool_selected("terrain")
        assert window.terrain_param_combo_action.isVisible()
        assert window.brush_size_spin_action.isVisible()

        window.mode_combo.setCurrentText("View")
        assert not window.terrain_param_combo_action.isVisible()
        assert not window.brush_size_spin_action.isVisible()
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()
