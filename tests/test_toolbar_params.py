"""Tool-param toolbar widgets (Trees / Eye candy / Level / Brush) show only for the
tool that reads them, and mode-inapplicable tool buttons hide rather than
grey out (ToolDef.modes) -- driven through a real offscreen ViewerWindow, same
technique and default-tier rationale as tests/test_fill_tool.py.

Asserts on the captured QAction handles (paint_trees_param_action,
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

from descape import settings, viewer_common

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Derived from settings.TOOLS rather than hand-listed, so a new tool (or a
# param_widget/supports_brush change to an existing one) can't silently drift
# out of sync -- this file's own dicts used to omit Ruler for exactly that
# reason. Every Units-mode-only tool (Place Unit and Convert, both
# modes == ("units",)) is excluded: this file's fixture is Terrain mode
# throughout, where either action stays both hidden AND disabled, which
# would fail a param- or brush-visibility assertion for a reason that has
# nothing to do with what this file tests. See tests/test_unit_edit_viewer.py
# for Units-mode tools' own coverage. Eyedropper is also excluded: it has
# param_widget == "" like Elevate, but deliberately shows BOTH single-valued
# params rather than neither -- see test_eyedropper.py's own visibility test,
# and _update_tool_enabled()'s eyedropper special-case comment. Select is
# excluded for the same reason: param_widget == "" like Elevate, but shows
# its own param group (the three paste-filter checkboxes, gated on
# _current_tool == "select" rather than on ToolDef.param_widget) -- see
# tests/test_region_select.py's own visibility test.
# Triggers-only tools (Create Objects) are excluded for the same reason as
# the Units-only ones; tests/test_create_objects_tool.py covers them.
_TERRAIN_MODE_TOOLS = [
    t
    for t in settings.TOOLS
    if viewer_common.tool_applicable(t.tool_id, "terrain") and t.tool_id not in ("eyedropper", "select")
]

# Which of the two single-valued params (if either) is expected to be
# visible for each tool once a square map is loaded and Terrain mode is active.
_EXPECTED_PARAM = {t.tool_id: (t.param_widget or None) for t in _TERRAIN_MODE_TOOLS}

# Brush size/shape is orthogonal to _EXPECTED_PARAM above -- Set Elevation
# shows both a level AND a brush, Elevate shows a brush with no
# _EXPECTED_PARAM entry at all. See settings.ToolDef.supports_brush.
#
# Read through viewer_common.brush_applicable() rather than off
# ToolDef.supports_brush directly: Draw Rectangle supports a brush but hides
# it in Filled mode, and that carve-out has exactly one home. Taking the
# window's own live Fill/Outline state keeps this generated rather than
# pinned to whichever default the combo happens to ship with.
def _expected_brush(tool_id: str, window) -> bool:
    return viewer_common.brush_applicable(tool_id, rect_filled=window._rect_filled())

# Free placement's own orthogonal group (D2's toggle). Derived, not asserted
# flat-False, so that if a Terrain-mode tool ever sets supports_free_place
# this file starts checking it instead of silently contradicting it.
_EXPECTED_FREE_PLACE = {t.tool_id: t.supports_free_place for t in _TERRAIN_MODE_TOOLS}


def test_tool_visibility_matches_mode_applicability() -> None:
    """Closes the "Tools that don't apply to the current mode should be
    hidden, not just grayed out" TODO item. Checked generically against
    settings.TOOLS/ToolDef.modes rather than hardcoding which tools are
    Terrain-only, so a future mode-gated tool is covered for free."""
    window = conftest.blank_window()
    try:
        for mode_text, mode_id in [
            ("View", "view"), ("Terrain", "terrain"), ("Units", "units"), ("Triggers", "triggers")
        ]:
            window.mode_combo.setCurrentText(mode_text)
            for tool in settings.TOOLS:
                action = getattr(window, f"{tool.tool_id}_action")
                assert action.isVisible() == viewer_common.tool_applicable(tool.tool_id, mode_id), (
                    tool.tool_id,
                    mode_text,
                )
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("tool", list(_EXPECTED_PARAM))
def test_tool_param_visibility_matches_active_tool(tool: str) -> None:
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected(tool)
        expected = _EXPECTED_PARAM[tool]
        expected_brush = _expected_brush(tool, window)
        expected_free = _EXPECTED_FREE_PLACE[tool]

        # The terrain itself is picked in the sidebar (GH #56); Trees/Eye candy carry its gate.
        assert window.paint_trees_param_action.isVisible() == (expected == "terrain")
        assert window.paint_eye_candy_param_action.isVisible() == (expected == "terrain")
        assert window.level_param_spin_action.isVisible() == (expected == "level")
        assert window.level_param_label_action.isVisible() == (expected == "level")
        assert window.brush_size_spin_action.isVisible() == expected_brush
        assert window.brush_shape_combo_action.isVisible() == expected_brush
        assert window.brush_param_label_action.isVisible() == expected_brush
        assert window.free_place_param_action.isVisible() == expected_free
        # Elevate has no _EXPECTED_PARAM entry (expected is None) but DOES
        # have a brush -- the separator must track EVERY group, not just the
        # single-valued param. A new group that forgets to join the roll-up
        # `or` chain shows its own widget beside a hidden separator.
        assert window.tool_param_separator_action.isVisible() == (
            expected is not None or expected_brush or expected_free
        )
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_level_spin_dead_while_draw_selected() -> None:
    """The discriminating check: gating the Level spinbox on its own
    enabled state alone (orthogonal to visibility) would leave it live
    while hidden whenever Draw is active. Draw's ]/[ keys are alive
    for a different reason now -- see test_brush_size_owns_keys_over_level
    below -- so they are not asserted dead here."""
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
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
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
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
        assert not window.paint_trees_param_action.isVisible()
        assert not window.level_param_spin_action.isVisible()
        assert not window.brush_size_spin_action.isVisible()
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_tool_params_hidden_in_view_mode() -> None:
    """View mode forces the active tool back to Pan (on_mode_changed), so
    Draw's own params stay hidden even though they were showing moments
    before the mode switch."""
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
        assert window.paint_trees_param_action.isVisible()
        assert window.brush_size_spin_action.isVisible()

        window.mode_combo.setCurrentText("View")
        assert not window.paint_trees_param_action.isVisible()
        assert not window.brush_size_spin_action.isVisible()
        assert not window.tool_param_separator_action.isVisible()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rectangle_fill_toggle_shows_only_for_draw_rectangle() -> None:
    """Not covered by the parametrization above, which asserts on the two
    single-valued params and the brush group only."""
    window = conftest.terrain_edit_window()
    try:
        for tool in _EXPECTED_PARAM:
            window._on_tool_selected(tool)
            assert window.rect_fill_param_action.isVisible() == (tool == "draw_rect"), tool
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_rectangle_brush_params_follow_the_fill_toggle() -> None:
    """Flag A of the Draw Line/Rectangle plan, as a test: a filled rectangle
    would be dilated past its own previewed bounds by a brush, so the brush
    params hide in Filled mode and return in Outline mode. The discriminating
    half is that MapView's brush is reset alongside them -- without that the
    hover preview would keep dilating while the widgets were hidden."""
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw_rect")
        window.brush_size_spin.setValue(5)

        window.rect_fill_combo.setCurrentText("Outline")
        assert window.brush_size_spin_action.isVisible()
        assert window.brush_shape_combo_action.isVisible()
        assert window.map_view._brush_size == 5

        window.rect_fill_combo.setCurrentText("Filled")
        assert not window.brush_size_spin_action.isVisible()
        assert not window.brush_shape_combo_action.isVisible()
        assert window.map_view._brush_size == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- auto beach (2026-08-31 water/beach plan, Stage 3) ------------------------

_WATER_DEEP = 22
_GRASS_1 = 0


def _select_terrain(window, terrain_id: int) -> None:
    window.terrain_panel.set_terrain(terrain_id)


def test_auto_beach_shows_only_for_draw() -> None:
    """Paint Can shares param_widget == "terrain", so without the
    `== "draw"` term in the gate the checkbox would appear for Fill and do
    nothing. The shape tools are excluded for the same reason: their commit
    path is on_shape_commit, not the Draw stroke branch the ring hangs off."""
    window = conftest.terrain_edit_window()
    try:
        _select_terrain(window, _WATER_DEEP)
        for tool in _EXPECTED_PARAM:
            window._on_tool_selected(tool)
            assert window.auto_beach_param_action.isVisible() == (tool == "draw"), tool
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_auto_beach_is_live_only_for_a_water_terrain() -> None:
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
        _select_terrain(window, _WATER_DEEP)
        assert window.auto_beach_check.isEnabled()
        _select_terrain(window, _GRASS_1)
        assert not window.auto_beach_check.isEnabled()
        assert "water terrain" in window.auto_beach_check.toolTip()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_switching_the_terrain_regates_the_checkbox() -> None:
    """The terrain picker had no change signal wired at all before this feature;
    without one the checkbox would keep whatever state the last tool switch
    left it in."""
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
        _select_terrain(window, _GRASS_1)
        assert not window.auto_beach_check.isEnabled()
        _select_terrain(window, _WATER_DEEP)
        assert window.auto_beach_check.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_beach_combo_and_width_follow_the_checkbox() -> None:
    window = conftest.terrain_edit_window()
    try:
        window._on_tool_selected("draw")
        _select_terrain(window, _WATER_DEEP)
        assert not window.beach_param_action.isVisible()
        assert not window.beach_width_param_action.isVisible()
        window.auto_beach_check.setChecked(True)
        assert window.beach_param_action.isVisible()
        assert window.beach_width_param_action.isVisible()
        assert window.beach_combo.isEnabled()
        assert window.beach_width_spin.isEnabled()
        # Ticked but then switched to land: the whole group goes away again,
        # so a hidden setting can never sit live.
        _select_terrain(window, _GRASS_1)
        assert not window.beach_param_action.isVisible()
        assert not window.beach_combo.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_beach_combo_offers_auto_plus_every_beach_terrain() -> None:
    from descape import terrain_classes

    window = conftest.terrain_edit_window()
    try:
        assert window.beach_combo.itemData(0) is None  # "Auto"
        offered = [window.beach_combo.itemData(i) for i in range(1, window.beach_combo.count())]
        assert offered == list(terrain_classes.beach_terrains())
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_beach_width_defaults_to_one() -> None:
    """The perf gate is stated on the default, not the worst case: width 1
    projects to ~1.1x a plain Draw stroke, width 3 to ~2.4x."""
    window = conftest.terrain_edit_window()
    try:
        assert window.beach_width_spin.value() == 1
        assert window.beach_width_spin.minimum() == 1
        assert window.beach_width_spin.maximum() == 3
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_elevation_view_combo_explains_every_style() -> None:
    """GH #55: the Elevation View combo carries a tooltip naming each style."""
    from descape.terrain_style import STYLE_LABELS
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        tip = window.terrain_style_combo.toolTip()
        for label in STYLE_LABELS:
            assert f"{label}:" in tip, label
    finally:
        window.edit_history.mark_saved()
        window.close()
