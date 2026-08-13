"""Tool-param toolbar widgets (Terrain type / Level) show only for the tool
that reads them, driven through a real offscreen ViewerWindow -- same
technique and default-tier rationale as tests/test_fill_tool.py.

Asserts on the captured QAction handles (terrain_param_combo_action,
level_param_spin_action, ...), never on the widgets' own isVisible(): under
QT_QPA_PLATFORM=offscreen with a never-shown window, a widget's isVisible()
is False unconditionally, which would make a widget-level assertion fail in
one direction and pass vacuously in the other. QAction.isVisible() has no
such dependency on the window ever being shown.

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

# Which of the two params (if either) is expected to be visible for each
# tool once a square map is loaded and Edit mode is active.
_EXPECTED_PARAM = {
    "pan": None,
    "terrain": "terrain",
    "fill": "terrain",
    "elevation": None,
    "set_level": "level",
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

        assert window.terrain_param_combo_action.isVisible() == (expected == "terrain")
        assert window.terrain_param_label_action.isVisible() == (expected == "terrain")
        assert window.level_param_spin_action.isVisible() == (expected == "level")
        assert window.level_param_label_action.isVisible() == (expected == "level")
        assert window.tool_param_separator_action.isVisible() == (expected is not None)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_level_keybinds_dead_while_terrain_selected() -> None:
    """The discriminating check: gating the ]/[ step-value keybinds on the
    spinbox's enabled state alone (orthogonal to visibility) would leave
    them live against a hidden Level spinbox whenever Terrain is active."""
    window = _edit_window()
    try:
        window._on_tool_selected("terrain")
        assert not window.level_param_spin_action.isVisible()
        assert not window.elevation_level_spin.isEnabled()
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
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_tool_params_hidden_in_view_mode() -> None:
    """View mode forces the active tool back to Pan (on_mode_changed), so
    Terrain's own param stays hidden even though it was showing moments
    before the mode switch."""
    window = _edit_window()
    try:
        window._on_tool_selected("terrain")
        assert window.terrain_param_combo_action.isVisible()

        window.mode_combo.setCurrentText("View")
        assert not window.terrain_param_combo_action.isVisible()
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()
