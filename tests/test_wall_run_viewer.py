"""Place Unit's wall branch driven through a real offscreen ViewerWindow
(GH #98, which folded the 2026-09-19 Wall Run tool into Place Unit) -- the
wiring, not the geometry.

Picking one of the 9 wall-family consts (8 walls plus Aqueduct, GH #110) in
the Units catalog turns Place Unit into a drag_shape path: MapView commits
it through on_shape_commit(tiles) rather than on_unit_place(). Most tests below drive MapView's own press/move/release
handlers with real QMouseEvents (the technique tests/test_shape_tools_viewer.py
documents), because the routing decision itself -- wall const or not, latched
at press -- is what changed.

The planner itself (variant derivation, skips, junction rewrites) is tested
headless in test_wall_run.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

WALL = 117  # Stone Wall
PALISADE = 72  # Palisade Wall, another wall const
GATE = 64  # Stone Gate, one orientation: not a wall-run const
AQUEDUCT = 231  # class 27 with 5 variants, a full wall since GH #110
VILLAGER = 83
GAIA_PLAYER_ID = 0
PLAYER = 1


def _window(const: int = WALL):
    """A shown window, Units mode, Place Unit armed with `const` picked.
    Caller must _close() it."""
    from PyQt5.QtWidgets import QApplication

    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    assert window.units_panel.select_owner(PLAYER)
    _pick(window, const)
    window.place_unit_action.setChecked(True)
    window.show()
    QApplication.processEvents()
    return window


def _pick(window, const: int) -> None:
    window.units_panel.select_object(const)
    # Without this every test could pass vacuously on a None selection.
    assert window.units_panel.selected_object_const() == const


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _unit_count(window) -> int:
    return sum(len(u) for u in window.scenario.unit_manager.units)


def _mouse_event(kind, pos, button, buttons, modifiers=None):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier if modifiers is None else modifiers)


def _press(map_view, tile, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mousePressEvent(
        _mouse_event(
            QEvent.MouseButtonPress, conftest.polygon_viewport_pos(map_view, *tile),
            Qt.LeftButton, Qt.LeftButton, modifiers,
        )
    )


def _move(map_view, tile, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseMoveEvent(
        _mouse_event(
            QEvent.MouseMove, conftest.polygon_viewport_pos(map_view, *tile),
            Qt.NoButton, Qt.LeftButton, modifiers,
        )
    )


def _release(map_view, tile) -> None:
    from PyQt5.QtCore import QEvent, Qt

    map_view.mouseReleaseEvent(
        _mouse_event(
            QEvent.MouseButtonRelease, conftest.polygon_viewport_pos(map_view, *tile),
            Qt.LeftButton, Qt.NoButton,
        )
    )


def _drag(map_view, a, b, modifiers=None) -> None:
    _press(map_view, a, modifiers)
    _move(map_view, b, modifiers)
    _release(map_view, b)


def _gaia_wall(window):
    """The fixture's one wall: GAIA-owned, radian-encoded (index 2)."""
    return next(u for u in window.scenario.unit_manager.units[GAIA_PLAYER_ID] if u.unit_const == WALL)


def test_the_wall_run_tool_is_gone():
    window = _window()
    try:
        assert not hasattr(window, "wall_run_action")
        assert not hasattr(window, "wall_family_combo")
        assert "wall_run" not in {t.tool_id for t in settings.TOOLS}
        assert "tool_wall_run" not in {row[0] for row in settings.REBINDABLE_ACTIONS}
        assert window.place_unit_action.isVisible()
    finally:
        _close(window)


def test_a_drag_places_a_run_as_one_undo_record():
    window = _window()
    try:
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        _drag(window.map_view, (20, 40), (31, 40))

        assert _unit_count(window) == before + 12
        assert len(window.edit_history.records) == before_records + 1
        assert window.edit_history.peek_undo().kind == "unit"
        placed = window.scenario.unit_manager.units[PLAYER][-12:]
        assert all(u.unit_const == WALL for u in placed)
        # Measured: all 8193 corpus walls store 0 here, unlike cliffs.
        assert all(u.initial_animation_frame == 0 for u in placed)
        # A literal index, never a radian re-encoding.
        assert [u.rotation for u in placed] == [2.0] + [0.0] * 10 + [2.0]
        assert window.map_view._shape_anchor is None

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_a_single_click_is_a_one_tile_run_at_the_fallback_index():
    """Not the old model.add() at rotation 0.0: 2 is what 24 of 24 isolated
    corpus walls store. Tile-centred, and the selection is left alone."""
    window = _window()
    try:
        window._selection = []
        before = _unit_count(window)
        _press(window.map_view, (50, 50))
        _release(window.map_view, (50, 50))
        assert _unit_count(window) == before + 1
        assert len(window.edit_history.records) == 1
        unit = window.scenario.unit_manager.units[PLAYER][-1]
        assert unit.unit_const == WALL
        assert unit.rotation == 2.0
        assert (unit.x, unit.y) == (50.5, 50.5)
        assert window._selection == []
    finally:
        _close(window)


def test_a_single_click_next_to_a_wall_joins_and_reshapes_it():
    """The fixture's GAIA wall at tile (7, 5) is isolated (index 2). One click
    west of it makes it a run end (still 2); a second click east flanks it,
    so that click's own record reshapes it to a run along x."""
    window = _window()
    try:
        wall = _gaia_wall(window)
        before_rotation = wall.rotation
        for tile in ((6, 5), (8, 5)):
            _press(window.map_view, tile)
            _release(window.map_view, tile)
        assert wall.rotation == 0.0
        assert window.scenario.unit_manager.units[PLAYER][-1].rotation == 2.0
        assert len(window.edit_history.records) == 2
        window.undo()
        assert wall.rotation == before_rotation
    finally:
        _close(window)


def test_a_wall_beside_an_aqueduct_joins_it():
    """GH #110: Aqueduct (231) is a full wall. The wall between the Aqueduct
    and a second wall becomes a mid-run (0), and the Aqueduct's own index is
    re-derived as the run end it now is (2)."""
    window = _window()
    try:
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place aqueduct", [PLAYER]):
            aqueduct = model.add(PLAYER, AQUEDUCT, 50.5, 50.5, rotation=3.0)
        for tile in ((49, 50), (48, 50)):
            _press(window.map_view, tile)
            _release(window.map_view, tile)
        inner = next(u for u in window.scenario.unit_manager.units[PLAYER] if (u.x, u.y) == (49.5, 50.5))
        assert inner.unit_const == WALL
        assert inner.rotation == 0.0
        assert aqueduct.rotation == 2.0
    finally:
        _close(window)


def test_an_aqueduct_drag_places_the_same_run_as_a_stone_wall():
    window = _window(AQUEDUCT)
    try:
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (25, 40))
        assert _unit_count(window) == before + 6
        placed = window.scenario.unit_manager.units[PLAYER][-6:]
        assert all(u.unit_const == AQUEDUCT for u in placed)
        assert [u.rotation for u in placed] == [2.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        assert all(u.initial_animation_frame == 0 for u in placed)
    finally:
        _close(window)


def test_a_single_click_aqueduct_joins_its_neighbour():
    """Three clicks along x: the third flanks the first-clicked Aqueduct, so
    that click's own record reshapes it to a mid-run (0)."""
    window = _window(AQUEDUCT)
    try:
        for tile in ((50, 50), (51, 50), (49, 50)):
            _press(window.map_view, tile)
            _release(window.map_view, tile)
        units = window.scenario.unit_manager.units[PLAYER]
        middle = next(u for u in units if (u.x, u.y) == (50.5, 50.5))
        assert middle.unit_const == AQUEDUCT
        assert middle.rotation == 0.0
        assert units[-1].rotation == 2.0
        assert len(window.edit_history.records) == 3
    finally:
        _close(window)


def test_a_gaia_run_places_with_derived_indices():
    """Replaces the old GAIA refusal, whose premise was stale:
    render.stored_rotation() keeps a GAIA wall's variant index."""
    window = _window()
    try:
        assert window.units_panel.select_owner(GAIA_PLAYER_ID)
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (25, 40))
        assert _unit_count(window) == before + 6
        placed = window.scenario.unit_manager.units[GAIA_PLAYER_ID][-6:]
        assert all(u.unit_const == WALL for u in placed)
        assert [u.rotation for u in placed] == [2.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        assert "for GAIA" in window.status_log.toPlainText()
    finally:
        _close(window)


@pytest.mark.parametrize("const", [VILLAGER, GATE])
def test_a_non_wall_const_still_click_places_one_unit(const):
    window = _window(const)
    try:
        before = _unit_count(window)
        _press(window.map_view, (50, 50))
        assert window.map_view._shape_anchor is None
        _move(window.map_view, (55, 50))
        _release(window.map_view, (55, 50))
        assert _unit_count(window) == before + 1
        assert window.scenario.unit_manager.units[PLAYER][-1].unit_const == const
    finally:
        _close(window)


def _click(window, tile) -> None:
    _press(window.map_view, tile)
    _release(window.map_view, tile)


def _walls_on(window, tiles) -> list:
    wall_consts = {WALL, PALISADE, AQUEDUCT}
    return [
        u for units in window.scenario.unit_manager.units for u in units
        if u.unit_const in wall_consts and (int(u.x), int(u.y)) in tiles
    ]


def test_a_gate_clicked_onto_a_run_replaces_the_walls_under_it():
    """GH #159. The run from (3, 5) to (11, 5) passes through the fixture's
    GAIA wall at (7, 5), so the gate's footprint (6..9, 5) holds walls of
    two owners. One record; undo brings all four back, redo removes them."""
    from descape import unit_pick

    window = _window()
    try:
        window.on_shape_commit([(x, 5) for x in range(3, 12)])
        footprint = {(x, 5) for x in range(6, 10)}
        under = _walls_on(window, footprint)
        assert len(under) == 4
        rotations = {id(u): u.rotation for u in under}
        before = _unit_count(window)
        records = len(window.edit_history.records)

        _pick(window, GATE)
        _click(window, (7, 5))

        assert _walls_on(window, footprint) == []
        assert _unit_count(window) == before - 3
        gate = window.scenario.unit_manager.units[PLAYER][-1]
        assert gate.unit_const == GATE and (gate.x, gate.y) == (7.5, 5.5)
        assert len(window.edit_history.records) == records + 1
        assert window._selection == [unit_pick.unit_key(PLAYER, gate)]
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "replacing 4 walls" in last
        assert "reshaping" not in last

        window.undo()
        restored = _walls_on(window, footprint)
        assert {id(u) for u in restored} == set(rotations)
        assert all(u.rotation == rotations[id(u)] for u in restored)
        assert _unit_count(window) == before

        window.redo()
        assert _walls_on(window, footprint) == []
        assert _unit_count(window) == before - 3
    finally:
        _close(window)


def test_a_gate_across_a_runs_end_reshapes_the_wall_it_now_touches():
    """The x run ends at the GAIA wall (7, 5); the gate takes (6, 5) and
    (7, 5) and its far end (9, 5) lands on a y run's tower end, which
    becomes a run along y inside the same record."""
    window = _window()
    try:
        window.on_shape_commit([(x, 5) for x in range(2, 7)])
        window.on_shape_commit([(9, y) for y in range(6, 10)])
        top = next(u for u in window.scenario.unit_manager.units[PLAYER] if (u.x, u.y) == (9.5, 6.5))
        assert top.rotation == 2.0

        _pick(window, GATE)
        _click(window, (7, 5))

        assert top.rotation == 1.0
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "replacing 2 walls, reshaping 1 adjacent" in last
        window.undo()
        assert top.rotation == 2.0
    finally:
        _close(window)


def test_free_placement_does_not_apply_to_walls():
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.free_place_check.setChecked(True)
        _press(window.map_view, (50, 50), Qt.AltModifier)
        _release(window.map_view, (50, 50))
        unit = window.scenario.unit_manager.units[PLAYER][-1]
        assert unit.unit_const == WALL
        assert (unit.x, unit.y) == (50.5, 50.5)
    finally:
        _close(window)


def test_show_walls_off_says_the_run_is_hidden():
    window = _window()
    try:
        window.show_walls_action.setChecked(False)
        _drag(window.map_view, (20, 40), (23, 40))
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "Placed 4" in last
        assert "a Filters toggle is hiding it" in last
    finally:
        _close(window)


def test_show_walls_on_carries_no_filter_clause():
    window = _window()
    try:
        _drag(window.map_view, (20, 40), (23, 40))
        assert "Filters toggle" not in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_shape_is_latched_at_press():
    """A mid-drag switch to a non-wall const keeps the rubber band (the drag
    stays a wall drag) and the commit then refuses rather than writing a
    non-wall const with a wall index."""
    window = _window()
    try:
        map_view = window.map_view
        before = _unit_count(window)
        _press(map_view, (20, 40))
        _pick(window, VILLAGER)
        _move(map_view, (26, 40))
        assert map_view._shape_anchor is not None
        assert map_view._highlight_outline_item is not None
        _release(map_view, (26, 40))
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
        assert "pick a wall" in window.status_log.toPlainText()
        assert map_view._shape_anchor is None
    finally:
        _close(window)


def test_a_mid_drag_switch_to_another_wall_places_that_wall():
    window = _window()
    try:
        _press(window.map_view, (20, 40))
        _pick(window, PALISADE)
        _move(window.map_view, (24, 40))
        _release(window.map_view, (24, 40))
        placed = window.scenario.unit_manager.units[PLAYER][-5:]
        assert [u.unit_const for u in placed] == [PALISADE] * 5
    finally:
        _close(window)


def test_no_terrain_stroke_opens_alongside_the_unit_record():
    """The wall tile set never reaches EditHistory's terrain path -- the
    check that the Units-mode shape commit didn't inherit the terrain branch
    on_shape_commit() was written for."""
    window = _window()
    try:
        _drag(window.map_view, (20, 40), (25, 40))
        assert [r.kind for r in window.edit_history.records] == ["unit"]
    finally:
        _close(window)


def test_a_junction_rewrite_rides_in_the_same_undo_record():
    """AGENTS.md's wall-run write-path exception, end to end: the
    pre-existing wall's own stored index changes inside the one record, and
    undo puts it back."""
    window = _window()
    try:
        wall = _gaia_wall(window)
        before_rotation = wall.rotation
        window.on_shape_commit([(6, 5), (8, 5)])

        assert len(window.edit_history.records) == 1
        assert wall.rotation == 0.0
        window.undo()
        assert wall.rotation == before_rotation
    finally:
        _close(window)


def test_a_run_over_an_existing_wall_skips_that_tile():
    window = _window()
    try:
        before = _unit_count(window)
        window.on_shape_commit([(7 + dx, 5) for dx in range(-2, 3)])
        # Five path tiles, one already occupied.
        assert _unit_count(window) == before + 4
    finally:
        _close(window)


@pytest.mark.parametrize("owner", [PLAYER, GAIA_PLAYER_ID])
def test_a_committed_run_agrees_with_the_render_sides_derivation(owner):
    """The fixture's one wall is radian-encoded, so the file classifies as
    radian and render resolves every wall with a neighbour through the
    override path: no override may disagree with what was stored."""
    from descape import render, unit_sprites

    window = _window()
    try:
        assert window.units_panel.select_owner(owner)
        window.on_shape_commit([(x, 5) for x in range(8, 14)] + [(13, y) for y in range(6, 11)])
        window.on_shape_commit([(x, 30) for x in range(30, 36)])
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        assert overrides, "the radian wall should force the override path"
        units = window.scenario.unit_manager.units
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            # variant_index(): the untouched fixture wall stores the radian form.
            stored = unit_sprites.variant_index(unit.rotation, 5)
            assert derived == stored, (player_id, unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


# --- Wall Rectangle (2026-09-21 wall enclosure plan) --------------------


def _rect_window(const: int = WALL):
    window = _window(const)
    window.wall_rect_action.setChecked(True)
    assert window._current_tool == "wall_rect"
    return window


def _ring(x0, y0, x1, y1):
    from descape import shape_tools

    return shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, 120, 120)


def test_the_wall_rectangle_button_only_shows_in_units_mode():
    """QAction.isVisible(), never the widget's -- an offscreen unshown
    widget reports False unconditionally and would pass vacuously."""
    window = _rect_window()
    try:
        assert window.wall_rect_action.isVisible()
        window.mode_combo.setCurrentText("Terrain")
        assert not window.wall_rect_action.isVisible()
        window.mode_combo.setCurrentText("View")
        assert not window.wall_rect_action.isVisible()
    finally:
        _close(window)


def test_a_ring_drag_is_one_undo_record_even_with_draw_rectangle_filled():
    """Pins the fallthrough trap: Draw Rectangle's Filled state must not
    reach wall_rect's preview or commit."""
    window = _rect_window()
    try:
        map_view = window.map_view
        map_view.set_rect_filled(True)
        before = _unit_count(window)
        _press(map_view, (20, 40))
        _move(map_view, (28, 44))
        assert sorted(map_view._shape_tiles(preview=True)) == sorted(_ring(20, 40, 28, 44))
        _release(map_view, (28, 44))
        assert _unit_count(window) == before + 24
        assert [r.kind for r in window.edit_history.records] == ["unit"]
        placed = window.scenario.unit_manager.units[PLAYER][-24:]
        assert {(int(u.x), int(u.y)) for u in placed} == set(_ring(20, 40, 28, 44))
        by_tile = {(int(u.x), int(u.y)): u.rotation for u in placed}
        assert by_tile[(20, 40)] == by_tile[(28, 44)] == 2.0
        assert by_tile[(24, 40)] == 0.0
        assert by_tile[(20, 42)] == 1.0
        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_shift_squares_the_ring():
    from PyQt5.QtCore import Qt

    window = _rect_window()
    try:
        map_view = window.map_view
        _press(map_view, (20, 40), Qt.ShiftModifier)
        _move(map_view, (28, 44), Qt.ShiftModifier)
        tiles = map_view._shape_tiles(preview=False)
        xs = {x for x, _y in tiles}
        ys = {y for _x, y in tiles}
        assert max(xs) - min(xs) == max(ys) - min(ys)
        assert sorted(tiles) == sorted(_ring(min(xs), min(ys), max(xs), max(ys)))
        _release(map_view, (28, 44))
    finally:
        _close(window)


def test_a_non_wall_const_is_refused_with_a_status_line():
    window = _rect_window(VILLAGER)
    try:
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (28, 44))
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
        assert "Wall Rectangle: pick a wall in the catalog" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_gaia_ring_is_placed_like_any_other():
    window = _rect_window()
    try:
        assert window.units_panel.select_owner(GAIA_PLAYER_ID)
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (28, 44))
        assert _unit_count(window) == before + 24
    finally:
        _close(window)


@pytest.mark.parametrize("owner", [PLAYER, GAIA_PLAYER_ID])
def test_a_committed_ring_agrees_with_the_render_sides_derivation(owner):
    """The main automated check of the enclosure plan: every ring node has
    orthogonal neighbours, so render resolves each one, and none may
    disagree with what was stored. The ring's top edge crosses the fixture's
    radian wall at (7, 5), which also forces the override path."""
    from descape import render, unit_sprites

    window = _rect_window()
    try:
        assert window.units_panel.select_owner(owner)
        window.on_shape_commit(_ring(3, 5, 11, 9))
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        units = window.scenario.unit_manager.units
        ring_nodes = [u for u in units[owner] if u.unit_const == WALL and (int(u.x), int(u.y)) in set(_ring(3, 5, 11, 9))]
        assert len(ring_nodes) >= 23
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            stored = unit_sprites.variant_index(unit.rotation, 5)
            assert derived == stored, (player_id, unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


# --- Walls skip occupied tiles (GH #124) --------------------------------

TREE = 349  # Tree (Oak): occupied, so it blocks with the toggle on


def _plant(window, tile, const: int = TREE):
    """A GAIA object at `tile`, in its own undo record."""
    model = window._ensure_unit_edits()
    assert model is not None
    with window._unit_edit(model, "Plant", [GAIA_PLAYER_ID]):
        return model.add(GAIA_PLAYER_ID, const, tile[0] + 0.5, tile[1] + 0.5)


def _walls_by_tile(window, owner: int = PLAYER) -> dict[tuple[int, int], float]:
    return {
        (int(u.x), int(u.y)): u.rotation
        for u in window.scenario.unit_manager.units[owner]
        if u.unit_const == WALL
    }


def test_the_skip_occupied_checkbox_shows_for_place_unit_and_wall_rectangle_only():
    window = _window()
    try:
        assert not window.skip_occupied_check.isChecked()
        for tool in ("place_unit", "wall_rect"):
            window._on_tool_selected(tool)
            assert window.skip_occupied_param_action.isVisible(), tool
            assert window.skip_occupied_check.isEnabled(), tool
            assert window.tool_param_separator_action.isVisible(), tool
        window._on_tool_selected("convert")
        assert not window.skip_occupied_param_action.isVisible()
        assert not window.skip_occupied_check.isEnabled()
    finally:
        _close(window)


def test_wall_rectangle_alone_still_shows_the_tool_param_separator():
    """Wall Rectangle has no other param group, so only the skip-occupied
    term in the separator roll-up can show the separator for it."""
    window = _rect_window()
    try:
        assert window.tool_param_separator_action.isVisible()
        assert not window.free_place_param_action.isVisible()
    finally:
        _close(window)


def test_toggle_off_a_run_through_a_tree_places_on_every_tile():
    window = _window()
    try:
        _plant(window, (25, 40))
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (31, 40))
        assert _unit_count(window) == before + 12
        assert (25, 40) in _walls_by_tile(window)
    finally:
        _close(window)


def test_toggle_on_a_run_through_a_tree_leaves_a_gap_with_ends_either_side():
    window = _window()
    try:
        _plant(window, (25, 40))
        window.skip_occupied_check.setChecked(True)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)

        _drag(window.map_view, (20, 40), (31, 40))

        assert _unit_count(window) == before + 11
        assert len(window.edit_history.records) == before_records + 1
        walls = _walls_by_tile(window)
        assert (25, 40) not in walls
        assert walls[(24, 40)] == walls[(26, 40)] == 2.0
        assert walls[(22, 40)] == 0.0
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "Placed 11" in last
        assert "(1 occupied tiles skipped)" in last
        assert "already held a wall" not in last

        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_toggle_on_eye_candy_does_not_block():
    from descape import unit_kind

    window = _window()
    try:
        assert 143 in unit_kind.eye_candy_consts()
        _plant(window, (25, 40), const=143)
        window.skip_occupied_check.setChecked(True)
        _drag(window.map_view, (20, 40), (31, 40))
        assert (25, 40) in _walls_by_tile(window)
    finally:
        _close(window)


def test_toggle_on_a_single_click_on_a_tree_places_nothing_and_says_occupied():
    window = _window()
    try:
        _plant(window, (50, 50))
        window.skip_occupied_check.setChecked(True)
        before = _unit_count(window)
        before_records = len(window.edit_history.records)
        _press(window.map_view, (50, 50))
        _release(window.map_view, (50, 50))
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == before_records
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "occupied" in last
        assert "already hold a wall" not in last
    finally:
        _close(window)


def test_toggle_on_wall_rectangle_leaves_the_gap():
    window = _rect_window()
    try:
        _plant(window, (24, 40))
        window.skip_occupied_check.setChecked(True)
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (28, 44))
        assert _unit_count(window) == before + 23
        walls = _walls_by_tile(window)
        assert (24, 40) not in walls
        assert walls[(23, 40)] == walls[(25, 40)] == 2.0
    finally:
        _close(window)


def test_the_preview_drops_blocked_tiles_and_the_commit_set_keeps_them():
    window = _window()
    try:
        _plant(window, (25, 40))
        window.skip_occupied_check.setChecked(True)
        map_view = window.map_view
        _press(map_view, (20, 40))
        _move(map_view, (31, 40))
        preview = map_view._shape_tiles(preview=True)
        assert (25, 40) not in preview
        assert preview == [(x, 40) for x in range(20, 32) if x != 25]
        assert (25, 40) in map_view._shape_tiles(preview=False)
        _release(map_view, (31, 40))
        assert map_view._drag_blocked == frozenset()
    finally:
        _close(window)


def test_toggle_off_the_preview_keeps_every_tile():
    window = _window()
    try:
        _plant(window, (25, 40))
        map_view = window.map_view
        _press(map_view, (20, 40))
        _move(map_view, (31, 40))
        assert (25, 40) in map_view._shape_tiles(preview=True)
        _release(map_view, (31, 40))
    finally:
        _close(window)


def test_a_gapped_run_agrees_with_the_render_sides_derivation():
    from descape import render, unit_sprites

    window = _window()
    try:
        for tile in ((10, 5), (13, 8)):
            _plant(window, tile)
        window.skip_occupied_check.setChecked(True)
        window.on_shape_commit([(x, 5) for x in range(8, 14)] + [(13, y) for y in range(6, 11)])
        walls = _walls_by_tile(window)
        assert (10, 5) not in walls and (13, 8) not in walls
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        assert overrides, "the radian wall should force the override path"
        units = window.scenario.unit_manager.units
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            stored = unit_sprites.variant_index(unit.rotation, 5)
            assert derived == stored, (player_id, unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


# --- The wall button (GH #125 Part A) ------------------------------------


def _pan_window(const: int = VILLAGER):
    """Units mode with Pan active and a non-wall picked, so the button's
    pick and tool switch are both observable."""
    window = _window(const)
    window.pan_action.setChecked(True)
    assert window._current_tool == "pan"
    return window


def _menu_action(window, const: int):
    return next(a for a in window.wall_pick_button.menu().actions() if a.data() == const)


def test_the_wall_button_only_shows_in_units_mode():
    """QAction.isVisible(), never the widget's, for the reason the Wall
    Rectangle test gives. The separator must follow it even under Pan."""
    window = _pan_window()
    try:
        assert window.wall_pick_param_action.isVisible()
        assert window.wall_pick_button.isEnabled()
        assert window.tool_param_separator_action.isVisible()
        for mode in ("Terrain", "View", "Triggers"):
            window.mode_combo.setCurrentText(mode)
            assert not window.wall_pick_param_action.isVisible(), mode
    finally:
        _close(window)


def test_the_wall_menu_lists_every_wall_family_const():
    from descape import object_catalog, unit_sprites

    window = _pan_window()
    try:
        actions = window.wall_pick_button.menu().actions()
        assert [a.data() for a in actions] == list(unit_sprites.wall_family_consts())
        assert [a.text() for a in actions] == [
            object_catalog.display_name(c) for c in unit_sprites.wall_family_consts()
        ]
    finally:
        _close(window)


def test_a_click_with_nothing_placed_yet_arms_place_unit_with_stone_wall():
    window = _pan_window()
    try:
        assert window.wall_pick_button.text() == "Wall: Stone Wall"
        window.wall_pick_button.click()
        assert window._current_tool == "place_unit"
        assert window.place_unit_action.isChecked()
        assert window.units_panel.selected_object_const() == WALL
    finally:
        _close(window)


def test_after_a_palisade_run_a_click_arms_palisade():
    window = _window(PALISADE)
    try:
        _drag(window.map_view, (20, 40), (26, 40))
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
        _pick(window, VILLAGER)
        window.pan_action.setChecked(True)
        window.wall_pick_button.click()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == PALISADE
    finally:
        _close(window)


def test_a_refused_run_does_not_change_the_last_used_wall():
    window = _window(PALISADE)
    try:
        _drag(window.map_view, (20, 40), (26, 40))
        _pick(window, WALL)
        # Every tile already holds a wall, so the plan is empty and nothing commits.
        _drag(window.map_view, (20, 40), (26, 40))
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
    finally:
        _close(window)


def test_a_menu_pick_while_wall_rectangle_is_active_keeps_wall_rectangle():
    window = _rect_window()
    try:
        _menu_action(window, PALISADE).trigger()
        assert window._current_tool == "wall_rect"
        assert window.wall_rect_action.isChecked()
        assert window.units_panel.selected_object_const() == PALISADE
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
        # And the click now re-arms the menu's pick, not Stone Wall.
        _pick(window, VILLAGER)
        window.wall_pick_button.click()
        assert window.units_panel.selected_object_const() == PALISADE
        assert window._current_tool == "wall_rect"
    finally:
        _close(window)


def test_a_menu_pick_from_pan_arms_place_unit():
    window = _pan_window()
    try:
        _menu_action(window, AQUEDUCT).trigger()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == AQUEDUCT
    finally:
        _close(window)


def test_a_catalog_filter_that_hides_the_wall_row_still_lands_the_pick():
    """select_object() alone is a silent no-op on a filtered-out row."""
    window = _pan_window()
    try:
        filter_edit = window.units_panel.catalog_view.filter_edit
        filter_edit.setText("villager")
        window.wall_pick_button.click()
        assert window.units_panel.selected_object_const() == WALL
        assert window._current_tool == "place_unit"
    finally:
        _close(window)


def test_a_pick_that_cannot_land_says_so_and_changes_nothing():
    window = _pan_window()
    try:
        window._arm_wall(-12345)
        assert window._current_tool == "pan"
        assert window.units_panel.selected_object_const() == VILLAGER
        assert window.wall_pick_button.text() == "Wall: Stone Wall"
        assert "Wall: could not pick" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_the_keybind_action_arms_the_last_used_wall():
    window = _pan_window()
    try:
        assert window._keybind_actions["unit_wall_pick"] is window.wall_pick_action
        window.wall_pick_action.trigger()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == WALL
    finally:
        _close(window)


def test_the_wall_button_is_disabled_without_a_unit_editable_map():
    from PyQt5.QtWidgets import QApplication

    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.mode_combo.setCurrentText("Units")
        QApplication.processEvents()
        assert window.mode == "units"
        assert not window.wall_pick_button.isEnabled()
        assert not window.wall_pick_action.isEnabled()
        window.wall_pick_action.trigger()
        assert window._current_tool == "pan"
    finally:
        _close(window)
