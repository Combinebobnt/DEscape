"""Place Unit's wall branch driven through a real offscreen ViewerWindow
(GH #98, which folded the 2026-09-19 Wall Run tool into Place Unit) -- the
wiring, not the geometry.

Picking one of the 8 wall consts in the Units catalog turns Place Unit into
a drag_shape path: MapView commits it through on_shape_commit(tiles) rather
than on_unit_place(). Most tests below drive MapView's own press/move/release
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
AQUEDUCT = 231  # class 27 with 5 variants, but not a wall
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


def test_a_wall_beside_an_aqueduct_neither_joins_nor_rewrites_it():
    """GH #52's Aqueduct step, DEscape half. Aqueduct (231) is class 27 and
    has 5 angles like a wall, but is no wall connector. The wall between the
    Aqueduct and a second wall stays a run end (2), not a mid-run (0)."""
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
        assert inner.rotation == 2.0
        assert aqueduct.rotation == 3.0
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
