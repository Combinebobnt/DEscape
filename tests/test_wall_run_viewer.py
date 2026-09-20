"""The Wall Run tool driven through a real offscreen ViewerWindow
(2026-09-19 wall-runs plan) -- the wiring, not the geometry.

Modelled on test_cliff_tool_viewer.py, with the one dispatch difference that
matters: Wall Run is a drag_shape tool, so MapView commits it through
on_shape_commit(tiles) rather than the on_edit_stroke_* trio. These tests
call that method directly rather than synthesizing Qt events, for the same
reason that file gives: what's under test is the commit, and MapView's own
rubber-band/tile-set half is covered headless in test_shape_tools.py.

The planner itself (variant derivation, skips, junction rewrites) is tested
headless in test_wall_run.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import unit_sprites

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

WALL = 117  # Stone Wall
GAIA_PLAYER_ID = 0
PLAYER = 1


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    assert window.units_panel.select_owner(PLAYER)
    index = window.wall_family_combo.findData(WALL)
    assert index != -1
    window.wall_family_combo.setCurrentIndex(index)
    window.wall_run_action.setChecked(True)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _unit_count(window) -> int:
    return sum(len(u) for u in window.scenario.unit_manager.units)


def test_the_button_only_shows_in_units_mode():
    """QAction.isVisible(), never the widget's -- an offscreen unshown
    widget reports False unconditionally and would pass vacuously."""
    window = _window()
    try:
        assert window.wall_run_action.isVisible()
        window.mode_combo.setCurrentText("Terrain")
        assert not window.wall_run_action.isVisible()
        window.mode_combo.setCurrentText("View")
        assert not window.wall_run_action.isVisible()
    finally:
        _close(window)


def test_the_family_combo_offers_the_eight_wall_consts():
    window = _window()
    try:
        offered = [
            window.wall_family_combo.itemData(i)
            for i in range(window.wall_family_combo.count())
        ]
        assert offered == list(unit_sprites.wall_family_consts())
        assert window.wall_family_param_action.isVisible()
    finally:
        _close(window)


def test_a_many_piece_run_is_one_undo_record():
    window = _window()
    try:
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        window.on_shape_commit([(x, 40) for x in range(20, 32)])

        assert _unit_count(window) == before + 12
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        placed = window.scenario.unit_manager.units[PLAYER][-12:]
        assert all(u.unit_const == WALL for u in placed)
        # Measured: all 8193 corpus walls store 0 here, unlike cliffs.
        assert all(u.initial_animation_frame == 0 for u in placed)
        # A literal index, never a radian re-encoding.
        assert all(u.rotation in (0.0, 1.0, 2.0, 3.0, 4.0) for u in placed)
        assert [u.rotation for u in placed][1:-1] == [0.0] * 10

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_a_single_click_places_one_wall_at_the_fallback_index():
    """A press-release without moving. 2, not place_unit's rotation=0.0
    default: that is what 24 of 24 isolated corpus walls store, and the game
    re-derives either value anyway."""
    window = _window()
    try:
        before = _unit_count(window)
        window.on_shape_commit([(50, 50)])
        assert _unit_count(window) == before + 1
        assert len(window.edit_history.records) == 1
        unit = window.scenario.unit_manager.units[PLAYER][-1]
        assert unit.rotation == 2.0
        assert (unit.x, unit.y) == (50.5, 50.5)
    finally:
        _close(window)


def test_no_terrain_stroke_opens_alongside_the_unit_record():
    """Wall Run's tile set never reaches EditHistory's terrain path -- the
    check that a Units-mode drag_shape tool didn't inherit the terrain
    branch on_shape_commit() was written for."""
    window = _window()
    try:
        window.on_shape_commit([(x, 40) for x in range(20, 26)])
        assert [r.kind for r in window.edit_history.records] == ["unit"]
    finally:
        _close(window)


def test_a_gaia_run_is_refused_rather_than_drawn_flat():
    """render.stored_rotation() forces a GAIA unit's rotation to 0.0, so a
    GAIA run would draw as all index 0 whatever is written."""
    window = _window()
    try:
        assert window.units_panel.select_owner(GAIA_PLAYER_ID)
        before = _unit_count(window)
        window.on_shape_commit([(x, 40) for x in range(20, 26)])
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
    finally:
        _close(window)


def test_no_family_chosen_is_a_no_op():
    window = _window()
    try:
        window.wall_family_combo.clear()  # currentData() now None
        before = _unit_count(window)
        window.on_shape_commit([(20, 40)])
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
    finally:
        _close(window)


def test_a_junction_rewrite_rides_in_the_same_undo_record():
    """The new AGENTS.md write-path exception, end to end: the pre-existing
    wall's own stored index changes inside the one record, and undo puts it
    back."""
    from descape.render import unit_tile_bounds

    window = _window()
    try:
        mm = window.scenario.map_manager
        wall = next(
            u
            for units in window.scenario.unit_manager.units
            for u in units
            if u.unit_const == WALL
        )
        bounds = unit_tile_bounds(wall, mm.map_width, mm.map_height)
        wx, wy = bounds[0], bounds[2]
        before_rotation = wall.rotation
        # Flanking both sides turns an isolated piece into a run along x, so
        # the stored index really has to change.
        window.on_shape_commit([(wx - 1, wy), (wx + 1, wy)])

        assert len(window.edit_history.records) == 1
        assert wall.rotation == 0.0
        window.undo()
        assert wall.rotation == before_rotation
    finally:
        _close(window)


def test_a_run_over_an_existing_wall_skips_that_tile():
    from descape.render import unit_tile_bounds

    window = _window()
    try:
        mm = window.scenario.map_manager
        wall = next(
            u
            for units in window.scenario.unit_manager.units
            for u in units
            if u.unit_const == WALL
        )
        bounds = unit_tile_bounds(wall, mm.map_width, mm.map_height)
        wx, wy = bounds[0], bounds[2]
        before = _unit_count(window)
        window.on_shape_commit([(wx + dx, wy) for dx in range(-2, 3)])
        # Five path tiles, one already occupied.
        assert _unit_count(window) == before + 4
    finally:
        _close(window)
