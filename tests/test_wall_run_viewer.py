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

# MapView.leaveEvent reads QApplication.mouseButtons() and a synthetic press never sets it (TASK-406).
_HELD = {"value": False}


@pytest.fixture(autouse=True)
def _hold_the_synthetic_button(monkeypatch):
    if conftest.PYQT5_AVAILABLE:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        monkeypatch.setattr(
            QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if _HELD["value"] else Qt.NoButton)
        )
    _HELD["value"] = False
    yield
    _HELD["value"] = False


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

    _HELD["value"] = True
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
    _HELD["value"] = False


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


def _l_of_two_drags(const: int) -> dict[tuple[int, int], tuple[int, float]]:
    """An L laid by two Place Unit drags, (20, 40)-(25, 40) then (25, 40)-(25, 45):
    one drag cannot turn a right angle, since wall_path_tiles bends at 45
    degrees. The second starts on the first's end tile, which it skips."""
    window = _window(const)
    try:
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (25, 40))
        _drag(window.map_view, (25, 40), (25, 45))
        assert _unit_count(window) == before + 11
        assert len(window.edit_history.records) == 2
        return {
            (int(u.x), int(u.y)): (u.unit_const, u.rotation)
            for u in window.scenario.unit_manager.units[PLAYER]
            if 40 <= u.y < 46 and 20 <= u.x < 26
        }
    finally:
        _close(window)


def test_an_aqueduct_l_of_two_drags_turns_its_corner_like_a_stone_wall():
    """GH #110, the corner. Pass conditions:
    - the Aqueduct L is all Aqueduct, 11 pieces, one undo record per drag;
    - the corner (25, 40) reads 2, the x leg 0, the y leg 1, the two ends 2;
      the second drag's first piece (25, 41) is 1, so it joined the corner
      rather than standing as an end of its own;
    - the same two drags with Stone Wall store the same index on every tile."""
    aqueduct = _l_of_two_drags(AQUEDUCT)
    assert {const for const, _r in aqueduct.values()} == {AQUEDUCT}
    rotations = {tile: r for tile, (_c, r) in aqueduct.items()}
    expected = {(20, 40): 2.0, (25, 40): 2.0, (25, 45): 2.0}
    expected |= {(x, 40): 0.0 for x in range(21, 25)}
    expected |= {(25, y): 1.0 for y in range(41, 45)}
    assert rotations == expected
    stone = _l_of_two_drags(WALL)
    assert {tile: r for tile, (_c, r) in stone.items()} == rotations


def test_a_stone_wall_drag_ending_against_an_aqueduct_run_joins_it():
    """GH #110, wall against Aqueduct. An Aqueduct run (20..25, 40) stored as
    a run stores it; a Stone Wall drag from (30, 40) back to (26, 40) ends
    against its east end. Pass conditions:
    - the Aqueduct end piece (25, 40) is rewritten 2 -> 0 (now a mid-run);
    - the wall's own end at (26, 40) joins (0), its free end (30, 40) is 2;
    - the rewrite rides in the drag's one record, and undo puts the 2 back."""
    window = _window(WALL)
    try:
        model = window._ensure_unit_edits()
        assert model is not None
        with window._unit_edit(model, "Place aqueducts", [PLAYER]):
            run = [
                model.add(PLAYER, AQUEDUCT, x + 0.5, 40.5, rotation=r)
                for x, r in zip(range(20, 26), [2.0, 0.0, 0.0, 0.0, 0.0, 2.0], strict=True)
            ]
        records = len(window.edit_history.records)
        _drag(window.map_view, (30, 40), (26, 40))
        assert len(window.edit_history.records) == records + 1
        assert run[-1].rotation == 0.0
        assert [u.rotation for u in run[:-1]] == [2.0, 0.0, 0.0, 0.0, 0.0]
        assert _walls_by_tile(window) == {(26, 40): 0.0, (27, 40): 0.0, (28, 40): 0.0, (29, 40): 0.0, (30, 40): 2.0}
        window.undo()
        assert run[-1].rotation == 2.0
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


def _click_point(window, x: float, y: float) -> None:
    """A click at a continuous Flat map point, off a tile centre."""
    from PyQt5.QtCore import QEvent, QPointF, Qt

    view = window.map_view
    tp = view._tile_pixels
    pos = QPointF(view.mapFromScene(QPointF(x * tp, y * tp)))
    _HELD["value"] = True
    try:
        view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
    finally:
        _HELD["value"] = False


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
        window.unrestricted_check.setChecked(True)  # the fixture's pine at (6.5, 5.5) is under the gate
        window.on_shape_commit([(x, 5) for x in range(3, 12)])
        footprint = {(x, 5) for x in range(6, 10)}
        under = _walls_on(window, footprint)
        assert len(under) == 4
        rotations = {id(u): u.rotation for u in under}
        before = _unit_count(window)
        records = len(window.edit_history.records)

        _pick(window, GATE)
        # Off the tile centre, which is the snapped anchor's tie on the 4-long
        # axis: the whole number nearest x = 7.7 is 8, so tiles 6..9 (GH #206).
        _click_point(window, 7.7, 5.4)

        assert _walls_on(window, footprint) == []
        assert _unit_count(window) == before - 3
        gate = window.scenario.unit_manager.units[PLAYER][-1]
        assert gate.unit_const == GATE and (gate.x, gate.y) == (8.0, 5.5)
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


def test_a_gate_clicked_in_a_tiles_low_half_anchors_one_tile_lower():
    """GH #206: x = 7.3 rounds to 7, so the 4-long axis covers tiles 5..8,
    a removal set the old 7.5 anchor (tiles 6..9) never produced."""
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the fixture's trees at (5.5, 5.5) and (6.5, 5.5) are under the gate
        window.on_shape_commit([(x, 5) for x in range(3, 12)])
        footprint = {(x, 5) for x in range(5, 9)}
        assert len(_walls_on(window, footprint)) == 4

        _pick(window, GATE)
        _click_point(window, 7.3, 5.4)

        gate = window.scenario.unit_manager.units[PLAYER][-1]
        assert gate.unit_const == GATE and (gate.x, gate.y) == (7.0, 5.5)
        assert _walls_on(window, footprint) == []
        assert len(_walls_on(window, {(9, 5)})) == 1
    finally:
        _close(window)


def test_a_gate_across_a_runs_end_reshapes_the_wall_it_now_touches():
    """The x run ends at the GAIA wall (7, 5); the gate takes (6, 5) and
    (7, 5) and its far end (9, 5) lands on a y run's tower end, which
    becomes a run along y inside the same record."""
    window = _window()
    try:
        window.unrestricted_check.setChecked(True)  # the fixture's pine at (6.5, 5.5) is under the gate
        window.on_shape_commit([(x, 5) for x in range(2, 7)])
        window.on_shape_commit([(9, y) for y in range(6, 10)])
        top = next(u for u in window.scenario.unit_manager.units[PLAYER] if (u.x, u.y) == (9.5, 6.5))
        assert top.rotation == 2.0

        _pick(window, GATE)
        _click_point(window, 7.7, 5.4)  # gate x = 8.0, tiles 6..9, off the tie

        assert top.rotation == 1.0
        last = window.status_log.toPlainText().splitlines()[-1]
        assert "replacing 2 walls, reshaping 1 adjacent" in last
        window.undo()
        assert top.rotation == 2.0
    finally:
        _close(window)


def test_off_grid_does_not_apply_to_walls():
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        window.off_grid_check.setChecked(True)
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


def test_a_leave_mid_drag_with_the_button_held_keeps_the_run():
    """The held-button half of test_shape_tools_viewer's no-button Leave
    cancel: the cursor running past the viewport edge mid-drag keeps the run."""
    from PyQt5.QtCore import QEvent

    window = _window()
    try:
        map_view = window.map_view
        before = _unit_count(window)
        before_records = len(window.edit_history.records)
        _press(map_view, (20, 40))
        map_view.leaveEvent(QEvent(QEvent.Leave))
        _move(map_view, (24, 40))
        _release(map_view, (24, 40))
        assert _unit_count(window) == before + 5
        assert len(window.edit_history.records) == before_records + 1
        placed = window.scenario.unit_manager.units[PLAYER][-5:]
        assert [u.unit_const for u in placed] == [WALL] * 5
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


def _band_polygons(map_view) -> list[tuple]:
    """The rubber band's drawn subpaths, each as its point tuple, sorted."""
    path = map_view._highlight_outline_item.path()
    return sorted(tuple((round(p.x(), 6), round(p.y(), 6)) for p in poly) for poly in path.toSubpathPolygons())


def _tile_polygons(map_view, tiles) -> list[tuple]:
    out = []
    for tx, ty in tiles:
        polygon = map_view._tile_polygon(tx, ty, coarse=True)
        polygon.append(polygon[0])  # closeSubpath() repeats the first point
        out.append(tuple((round(p.x(), 6), round(p.y(), 6)) for p in polygon))
    return sorted(out)


def test_the_ring_preview_band_is_built_mid_drag_before_release():
    """TASK-074: the band the user sees while dragging, not just the tile
    list behind it. Pass conditions:
    - the press builds the band at once, on the anchor tile alone;
    - a move to (28, 44) rebuilds it, still before any release: the outline
      and fill items are in the scene, share one path, and that path is
      exactly the 24 ring tiles' polygons, one closed subpath per tile;
    - nothing is placed until the release."""
    window = _rect_window()
    try:
        map_view = window.map_view
        before = _unit_count(window)
        _press(map_view, (20, 40))
        assert map_view._highlight_outline_item is not None
        assert _band_polygons(map_view) == _tile_polygons(map_view, [(20, 40)])
        _move(map_view, (28, 44))
        outline, fill = map_view._highlight_outline_item, map_view._highlight_fill_item
        assert outline.scene() is map_view.scene() and fill.scene() is map_view.scene()
        assert fill.path() == outline.path()
        ring = _ring(20, 40, 28, 44)
        assert len(ring) == 24
        assert _band_polygons(map_view) == _tile_polygons(map_view, ring)
        assert _unit_count(window) == before
        _release(map_view, (28, 44))
        assert _unit_count(window) == before + 24
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
    disagree with what was stored. The fixture's radian wall at (7, 5) keeps
    the file radian, so render resolves them through the override path. The
    ring stays clear of that wall, as the Wall Circle twin does: a junction
    there would rewrite it to a literal integer and render would resolve
    nothing (TASK-469; the junction itself is test_wall_run.py's)."""
    from descape import render, unit_sprites

    window = _rect_window()
    try:
        assert window.units_panel.select_owner(owner)
        path = _ring(20, 40, 28, 44)
        ring = set(path)
        window.on_shape_commit(path)
        assert not unit_sprites.is_literal_variant_index(_gaia_wall(window).rotation, 5)
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        units = window.scenario.unit_manager.units
        ring_nodes = [u for u in units[owner] if u.unit_const == WALL and (int(u.x), int(u.y)) in ring]
        assert len(ring_nodes) == len(ring) == 24
        on_ring = [key for key in overrides if (int(units[key[0]][key[1]].x), int(units[key[0]][key[1]].y)) in ring]
        # Not vacuous: render resolves the ring's nodes rather than none at all.
        assert len(on_ring) >= 20, len(on_ring)
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            stored = unit_sprites.variant_index(unit.rotation, 5)
            assert derived == stored, (player_id, unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


# --- Wall Circle (GH #125 Part B) ---------------------------------------

# A 15x11 drag box: big enough that its ellipse has diagonal stretches (3 and 4), not just towers and runs.
_CIRCLE_A, _CIRCLE_B = (20, 40), (34, 50)


def _circle_window(const: int = WALL):
    window = _window(const)
    window.wall_circle_action.setChecked(True)
    assert window._current_tool == "wall_circle"
    return window


def _ellipse(x0, y0, x1, y1):
    from descape import shape_tools

    return shape_tools.ellipse_perimeter_tiles(x0, y0, x1, y1, 120, 120)


def test_the_wall_circle_button_only_shows_in_units_mode():
    window = _circle_window()
    try:
        assert window.wall_circle_action.isVisible()
        assert window.wall_circle_action.isEnabled()
        window.mode_combo.setCurrentText("Terrain")
        assert not window.wall_circle_action.isVisible()
        window.mode_combo.setCurrentText("View")
        assert not window.wall_circle_action.isVisible()
    finally:
        _close(window)


def test_the_wall_circle_tool_is_registered_like_wall_rectangle():
    """Unbound, Units-only, its own drag shape, the skip-occupied toggle,
    and a tooltip that names its terms."""
    tool = next(t for t in settings.TOOLS if t.tool_id == "wall_circle")
    assert (tool.label, tool.drag_shape, tool.modes, tool.default_key) == (
        "Wall Circle", "wall_circle", ("units",), ""
    )
    assert tool.supports_skip_occupied and not tool.supports_brush and not tool.supports_off_grid
    window = _circle_window()
    try:
        tip = window.wall_circle_action.toolTip()
        for words in ("outline", "one undo step per drag", "Shift makes it a circle", "Wall button"):
            assert words in tip, words
    finally:
        _close(window)


def test_a_circle_drag_is_one_ring_and_one_undo_record_even_with_draw_rectangle_filled():
    """Pins the fallthrough trap for Wall Circle: Draw Rectangle's Filled
    state must not reach wall_circle's preview or commit."""
    from descape import unit_sprites

    window = _circle_window()
    try:
        map_view = window.map_view
        map_view.set_rect_filled(True)
        ring = _ellipse(*_CIRCLE_A, *_CIRCLE_B)
        assert len(ring) == 32
        before = _unit_count(window)
        _press(map_view, _CIRCLE_A)
        _move(map_view, _CIRCLE_B)
        assert map_view._shape_tiles(preview=True) == ring
        assert map_view._shape_tiles(preview=False) == ring
        _release(map_view, _CIRCLE_B)
        assert _unit_count(window) == before + len(ring)
        assert [r.kind for r in window.edit_history.records] == ["unit"]
        placed = window.scenario.unit_manager.units[PLAYER][-len(ring):]
        by_tile = {(int(u.x), int(u.y)): u.rotation for u in placed}
        assert set(by_tile) == set(ring)
        ring_set = set(ring)
        for (tx, ty), rotation in by_tile.items():
            mask = unit_sprites.neighbour_mask(tx, ty, ring_set, diagonals=True)
            assert rotation == float(unit_sprites.wall_variant_from_neighbours8(mask)), (tx, ty)
        assert set(by_tile.values()) == {0.0, 1.0, 2.0, 3.0, 4.0}
        window.undo()
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_shift_makes_the_circle_a_circle():
    from PyQt5.QtCore import Qt

    window = _circle_window()
    try:
        map_view = window.map_view
        _press(map_view, _CIRCLE_A, Qt.ShiftModifier)
        _move(map_view, _CIRCLE_B, Qt.ShiftModifier)
        tiles = map_view._shape_tiles(preview=False)
        xs = {x for x, _y in tiles}
        ys = {y for _x, y in tiles}
        assert max(xs) - min(xs) == max(ys) - min(ys) == 14
        assert tiles == _ellipse(min(xs), min(ys), max(xs), max(ys))
        # A circle: the ring is its own transpose about the box's diagonal.
        assert {(y - min(ys), x - min(xs)) for x, y in tiles} == {(x - min(xs), y - min(ys)) for x, y in tiles}
        _release(map_view, _CIRCLE_B)
    finally:
        _close(window)


def test_a_non_wall_const_is_refused_with_a_wall_circle_status_line():
    window = _circle_window(VILLAGER)
    try:
        before = _unit_count(window)
        _drag(window.map_view, _CIRCLE_A, _CIRCLE_B)
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
        assert "Wall Circle: pick a wall in the catalog" in window.status_log.toPlainText()
    finally:
        _close(window)


def test_a_gaia_circle_is_placed_like_any_other():
    window = _circle_window()
    try:
        assert window.units_panel.select_owner(GAIA_PLAYER_ID)
        before = _unit_count(window)
        _drag(window.map_view, _CIRCLE_A, _CIRCLE_B)
        assert _unit_count(window) == before + 32
        assert "for GAIA" in window.status_log.toPlainText().splitlines()[-1]
    finally:
        _close(window)


@pytest.mark.parametrize("owner", [PLAYER, GAIA_PLAYER_ID])
def test_a_committed_circle_agrees_with_the_render_sides_derivation(owner):
    """The fixture's radian wall at (7, 5) classifies the file as radian, so
    render resolves every circle node with an orthogonal neighbour through
    the override path; none may disagree with what was stored. The circle
    stays clear of that wall: a junction would rewrite it to a literal
    integer, the file would read as integer-only, and render would resolve
    nothing at all (the junction itself is test_wall_run.py's)."""
    from descape import render, unit_sprites

    window = _circle_window()
    try:
        assert window.units_panel.select_owner(owner)
        ring = _ellipse(*_CIRCLE_A, *_CIRCLE_B)
        window.on_shape_commit(ring)
        assert not unit_sprites.is_literal_variant_index(_gaia_wall(window).rotation, 5)
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        units = window.scenario.unit_manager.units
        on_ring = [
            key for key in overrides
            if (int(units[key[0]][key[1]].x), int(units[key[0]][key[1]].y)) in set(ring)
        ]
        # Not vacuous: every ring node with an orthogonal neighbour (all but the 3/4 pieces) is resolved.
        assert len(on_ring) >= len(ring) - 12
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            stored = unit_sprites.variant_index(unit.rotation, 5)
            assert derived == stored, (player_id, unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


def test_the_wall_button_shows_for_wall_circle_and_a_menu_pick_keeps_it():
    """TASK-299 shows the wall button only beside _WALL_TOOLS; Wall Circle is one of them."""
    window = _circle_window()
    try:
        assert window.wall_pick_param_action.isVisible()
        assert window.wall_pick_button.isEnabled()
        _menu_action(window, PALISADE).trigger()
        assert window._current_tool == "wall_circle"
        assert window.wall_circle_action.isChecked()
        assert window.units_panel.selected_object_const() == PALISADE
        before = _unit_count(window)
        _drag(window.map_view, _CIRCLE_A, _CIRCLE_B)
        assert _unit_count(window) == before + 32
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
    finally:
        _close(window)


def test_toggle_on_wall_circle_leaves_the_gap():
    window = _circle_window()
    try:
        _plant(window, (20, 45))  # the ellipse's left apex
        window.skip_occupied_check.setChecked(True)
        before = _unit_count(window)
        _drag(window.map_view, _CIRCLE_A, _CIRCLE_B)
        assert _unit_count(window) == before + 31
        walls = _walls_by_tile(window)
        assert (20, 45) not in walls
        assert walls[(20, 44)] == walls[(20, 46)] == 2.0
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


def test_the_skip_occupied_checkbox_shows_for_the_wall_tools_only():
    window = _window()
    try:
        assert not window.skip_occupied_check.isChecked()
        for tool in ("place_unit", "wall_rect", "wall_circle"):
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
        assert not window.off_grid_param_action.isVisible()
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


def test_the_skip_occupied_checkbox_shows_under_no_tool_but_the_wall_tools():
    """GH #124 step 1, every tool in every mode it runs in. Pass conditions:
    - with a wall picked, the box starts unticked;
    - Place Unit, Wall Rectangle and Wall Circle (GH #125) show it and enable it;
    - every other tool, Pan and the Ruler included, hides and disables it;
    - the walk really reached every registered tool (no vacuous pass)."""
    window = _window()
    try:
        assert not window.skip_occupied_check.isChecked()
        shown = set()
        reached = set()
        for mode in ("View", "Terrain", "Units", "Triggers"):
            window.mode_combo.setCurrentText(mode)
            for tool in settings.TOOLS:
                if tool.modes and mode.lower() not in tool.modes:
                    continue
                window._on_tool_selected(tool.tool_id)
                if window._current_tool != tool.tool_id:
                    continue
                reached.add(tool.tool_id)
                visible = window.skip_occupied_param_action.isVisible()
                assert visible == window.skip_occupied_check.isEnabled(), (mode, tool.tool_id)
                if visible:
                    shown.add(tool.tool_id)
        assert reached == {tool.tool_id for tool in settings.TOOLS}
        assert shown == {"place_unit", "wall_rect", "wall_circle"}
        assert not window.skip_occupied_check.isChecked()
    finally:
        _close(window)


@pytest.mark.parametrize("const", [334, 859, 1366], ids=["flowers-1", "flower-bed", "plant-flowers"])
def test_toggle_on_flowers_do_not_block(const):
    """GH #124 step 4's flowers. Pass condition: with the box ticked, a run
    dragged across a flower places a wall on its tile like any other."""
    from descape import unit_kind

    window = _window()
    try:
        assert const in unit_kind.passable_eye_candy_consts()
        _plant(window, (25, 40), const=const)
        window.skip_occupied_check.setChecked(True)
        before = _unit_count(window)
        _drag(window.map_view, (20, 40), (31, 40))
        assert (25, 40) in _walls_by_tile(window)
        assert _unit_count(window) == before + 12
    finally:
        _close(window)


# --- The wall button (GH #125 Part A) ------------------------------------


def _pan_window(const: int = VILLAGER):
    """Units mode with Pan active and a non-wall picked, so the keybind's
    pick and tool switch are both observable."""
    window = _window(const)
    window.pan_action.setChecked(True)
    assert window._current_tool == "pan"
    return window


def _menu_action(window, const: int):
    return next(a for a in window.wall_pick_button.menu().actions() if a.data() == const)


def test_the_wall_button_only_shows_for_the_wall_tools():
    """GH #190. QAction.isVisible(), never the widget's, for the reason the
    Wall Rectangle test gives. Every other Units-mode tool hides it, Convert included."""
    from descape.viewer import _WALL_TOOLS
    from descape.viewer_common import tool_applicable

    window = _window(VILLAGER)
    try:
        for tool_id in _WALL_TOOLS:
            getattr(window, f"{tool_id}_action").setChecked(True)
            assert window._current_tool == tool_id
            assert window.wall_pick_param_action.isVisible(), tool_id
            assert window.wall_pick_button.isEnabled(), tool_id
            assert window.tool_param_separator_action.isVisible(), tool_id
        others = [
            t.tool_id for t in settings.TOOLS
            if t.tool_id not in _WALL_TOOLS and tool_applicable(t.tool_id, "units")
        ]
        assert "convert" in others and "pan" in others
        for tool_id in others:
            action = getattr(window, f"{tool_id}_action")
            assert action.isEnabled(), tool_id
            action.setChecked(True)
            assert window._current_tool == tool_id
            assert not window.wall_pick_param_action.isVisible(), tool_id
            assert not window.wall_pick_button.isEnabled(), tool_id
            # The keybind is not the button: it still arms a wall from any Units tool.
            assert window.wall_pick_action.isEnabled(), tool_id
        window.place_unit_action.setChecked(True)
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


def test_a_click_with_nothing_placed_yet_picks_stone_wall():
    window = _window(VILLAGER)
    try:
        assert window.wall_pick_button.text() == "Wall: Stone Wall"
        window.wall_pick_button.click()
        assert window._current_tool == "place_unit"
        assert window.place_unit_action.isChecked()
        assert window.units_panel.selected_object_const() == WALL
    finally:
        _close(window)


def test_after_a_palisade_run_a_click_picks_palisade():
    window = _window(PALISADE)
    try:
        _drag(window.map_view, (20, 40), (26, 40))
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
        _pick(window, VILLAGER)
        window.wall_pick_button.click()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == PALISADE
        # And the keybind arms the same wall from a tool that hides the button.
        _pick(window, VILLAGER)
        window.pan_action.setChecked(True)
        window.wall_pick_action.trigger()
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


def test_a_menu_pick_while_place_unit_is_active_keeps_place_unit():
    window = _window(VILLAGER)
    try:
        _menu_action(window, AQUEDUCT).trigger()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == AQUEDUCT
        assert window.wall_pick_button.text() == "Wall: Aqueduct"
    finally:
        _close(window)


def test_a_catalog_filter_that_hides_the_wall_row_still_lands_the_pick():
    """select_object() alone is a silent no-op on a filtered-out row."""
    window = _window(VILLAGER)
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
        assert not window.wall_pick_param_action.isVisible()
        window.wall_pick_action.trigger()
        assert window._current_tool == "place_unit"
        assert window.units_panel.selected_object_const() == WALL
        assert window.wall_pick_param_action.isVisible()
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


# --- Moving a gate updates walls (GH #194) --------------------------------

_ROW = 44
# Laid from x=20 to x=40, so a +5 move still lands the gate on four walls.
_RUN = [(x, _ROW) for x in range(20, 41)]
_OLD = {(x, _ROW) for x in range(25, 29)}
_NEW = {(x, _ROW) for x in range(30, 34)}
# A cursor point inside the gate, off the 4-long axis' rounding tie (GH #206).
_GRAB = (26.75, _ROW + 0.4)


def _point_pos(window, x: float, y: float):
    """Viewport position of a continuous Flat map point."""
    from PyQt5.QtCore import QPointF

    view = window.map_view
    tp = view._tile_pixels
    return QPointF(view.mapFromScene(QPointF(x * tp, y * tp)))


def _drag_point(window, a, b) -> None:
    """A real press-move-release between two continuous map points."""
    from PyQt5.QtCore import QEvent, Qt

    view = window.map_view
    start, end = _point_pos(window, *a), _point_pos(window, *b)
    _HELD["value"] = True
    try:
        view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
    finally:
        _HELD["value"] = False


def _drag_gate(window, dx: float, dy: float) -> None:
    _drag_point(window, _GRAB, (_GRAB[0] + dx, _GRAB[1] + dy))


def _gate_window(box: bool = True, free: bool = False):
    """(window, gate): the run on _ROW with a Stone Gate clicked onto tiles
    25..28 (GH #159's path), then Move. `box` ticks "Moving a gate updates
    walls", which alone lets the gate land on walls; `free` ticks Free
    placement, which a box-off drop onto walls still needs."""
    window = _window()
    window.on_shape_commit(_RUN)
    _pick(window, GATE)
    _click_point(window, *_GRAB)
    gate = window.scenario.unit_manager.units[PLAYER][-1]
    assert gate.unit_const == GATE and (gate.x, gate.y) == (27.0, _ROW + 0.5)
    assert _walls_on(window, _OLD) == [] and len(_walls_on(window, _NEW)) == 4
    window.move_action.setChecked(True)
    assert window._current_tool == "move"
    window.unrestricted_check.setChecked(free)
    window.gate_walls_check.setChecked(box)
    return window, gate


def _last_status(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _moved_gate_status(window) -> str:
    """The last status line past the gate's catalog name, which depends on the install."""
    last = _last_status(window)
    assert last.startswith("Moved ") and " to (" in last, last
    return last[last.index(" to (") + 1:]


def _is_field_delta(window) -> bool:
    """Whether the newest undo record is today's fields-only move, not a wholesale one."""
    return window.edit_history.peek_undo().unit_field_entries is not None


def _row_indices(window) -> dict[int, float]:
    return {int(u.x): u.rotation for u in _walls_on(window, set(_RUN))}


def _add_unit(window, const: int, x: float, y: float, garrisoned_in_id: int = -1):
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [PLAYER]):
        return model.add(PLAYER, const, x, y, garrisoned_in_id=garrisoned_in_id)


def _select_units(window, units) -> list:
    from descape import unit_pick

    keys = [unit_pick.unit_key(PLAYER, u) for u in units]
    window._selection = list(keys)
    window._refresh_selection_view()
    return keys


def _arrow(window, key, modifiers=None) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window.map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier if modifiers is None else modifiers))


def test_the_gate_walls_checkbox_shows_under_its_tool_in_units_mode_only():
    """QAction.isVisible(), for the reason the Wall Rectangle test gives. Its
    tool is Move, the tool that drags units (TASK-012), and Move is Units-only,
    so leaving Units mode lands on Pan with the box hidden."""
    from descape.viewer_common import GATE_WALLS_TOOLS, tool_applicable

    assert sorted(GATE_WALLS_TOOLS) == ["move"]
    window = _window(VILLAGER)
    try:
        assert not window.gate_walls_check.isChecked()
        assert not window.gate_walls_param_action.isVisible()
        for tool_id in GATE_WALLS_TOOLS:
            for mode in ("Terrain", "View", "Triggers", "Units"):
                window.mode_combo.setCurrentText("Units")
                getattr(window, f"{tool_id}_action").setChecked(True)
                assert window._current_tool == tool_id
                assert window.gate_walls_param_action.isVisible(), tool_id
                assert window.gate_walls_check.isEnabled(), tool_id
                assert window.tool_param_separator_action.isVisible(), tool_id
                assert window.gate_walls_check.text() == "Moving a gate updates walls"
                window.mode_combo.setCurrentText(mode)
                assert window._current_tool == (tool_id if mode == "Units" else "pan"), mode
                assert window.gate_walls_param_action.isVisible() == (mode == "Units"), mode
                assert window.gate_walls_check.isEnabled() == (mode == "Units"), mode
        others = [
            t.tool_id for t in settings.TOOLS
            if t.tool_id not in GATE_WALLS_TOOLS and tool_applicable(t.tool_id, "units")
        ]
        assert "place_unit" in others
        for tool_id in others:
            getattr(window, f"{tool_id}_action").setChecked(True)
            assert window._current_tool == tool_id
            assert not window.gate_walls_param_action.isVisible(), tool_id
            assert not window.gate_walls_check.isEnabled(), tool_id
    finally:
        _close(window)


def test_box_off_a_gate_dragged_along_its_run_is_todays_plain_move():
    window, gate = _gate_window(box=False, free=True)
    try:
        before = _unit_count(window)
        indices = _row_indices(window)
        records = len(window.edit_history.records)

        _drag_gate(window, 5, 0)

        assert (gate.x, gate.y) == (32.0, _ROW + 0.5)
        assert _unit_count(window) == before
        assert len(_walls_on(window, _NEW)) == 4
        assert _walls_on(window, _OLD) == []
        assert _row_indices(window) == indices
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        assert _moved_gate_status(window) == f"to (32, {_ROW + 0.5:g})"
    finally:
        _close(window)


def test_box_on_a_gate_dragged_along_its_run_swaps_the_walls_in_one_record():
    """The GH #194 screenshots. The walls under the new footprint go, the four
    tiles the gate left hold the run's wall again, and undo restores the
    removed walls by identity with the rotations they had. The box alone is
    enough: the walls the gate replaces never block it, Free placement off."""
    from descape import unit_pick

    window, gate = _gate_window()
    try:
        assert not window.unrestricted_check.isChecked()
        under = _walls_on(window, _NEW)
        rotations = {id(u): u.rotation for u in under}
        before = _unit_count(window)
        records = len(window.edit_history.records)

        _drag_gate(window, 5, 0)

        assert (gate.x, gate.y) == (32.0, _ROW + 0.5)
        assert _walls_on(window, _NEW) == []
        filled = _walls_on(window, _OLD)
        assert sorted(int(u.x) for u in filled) == [25, 26, 27, 28]
        assert all(u.unit_const == WALL and u.rotation == 0.0 and u.initial_animation_frame == 0 for u in filled)
        assert all(any(u is own for own in window.scenario.unit_manager.units[PLAYER]) for u in filled)
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert window._selection == [unit_pick.unit_key(PLAYER, gate)]
        assert _moved_gate_status(window) == f"to (32, {_ROW + 0.5:g}), replacing 4 walls, filling 4"
        # An integer-only read of the row: ends are towers, the rest runs along x.
        assert _row_indices(window) == {x: (2.0 if x in (20, 40) else 0.0) for x in range(20, 41) if not 30 <= x <= 33}

        window.undo()
        restored = _walls_on(window, _NEW)
        assert {id(u) for u in restored} == set(rotations)
        assert all(u.rotation == rotations[id(u)] for u in restored)
        assert _walls_on(window, _OLD) == []
        assert _unit_count(window) == before
        moved = next(u for u in window.scenario.unit_manager.units[PLAYER] if u.unit_const == GATE)
        assert (moved.x, moved.y) == (27.0, _ROW + 0.5)

        window.redo()
        assert _walls_on(window, _NEW) == []
        assert len(_walls_on(window, _OLD)) == 4
        assert _unit_count(window) == before
    finally:
        _close(window)


def test_box_on_a_gate_dragged_off_its_run_fills_without_free_placement():
    """Nothing stands where it lands, so placement_rules has nothing to refuse."""
    window, gate = _gate_window()
    try:
        before = _unit_count(window)
        _drag_gate(window, 0, 3)
        assert (gate.x, gate.y) == (27.0, _ROW + 3.5), _last_status(window)
        assert len(_walls_on(window, _OLD)) == 4
        assert _unit_count(window) == before + 4
        assert _moved_gate_status(window) == f"to (27, {_ROW + 3.5:g}), filling 4"
    finally:
        _close(window)


def test_box_on_a_shift_arrow_nudge_swaps_one_wall_and_a_plain_one_touches_none():
    from PyQt5.QtCore import Qt

    window, gate = _gate_window()
    try:
        _select_units(window, [gate])
        before = _unit_count(window)
        indices = _row_indices(window)
        records = len(window.edit_history.records)

        # 0.1 tile keeps the footprint, so this is a field edit and no wall is re-derived.
        _arrow(window, Qt.Key_Right)
        assert gate.x == pytest.approx(27.1)
        assert _row_indices(window) == indices
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        assert "filling" not in _last_status(window) and "replacing" not in _last_status(window)

        window.undo()
        assert gate.x == 27.0
        _arrow(window, Qt.Key_Right, Qt.ShiftModifier)

        assert gate.x == 28.0
        assert _walls_on(window, {(29, _ROW)}) == []
        assert [u.rotation for u in _walls_on(window, {(25, _ROW)})] == [0.0]
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert _moved_gate_status(window) == f"to (28, {_ROW + 0.5:g}), replacing 1 wall, filling 1"

        window.undo()
        assert gate.x == 27.0
        assert len(_walls_on(window, {(29, _ROW)})) == 1 and _walls_on(window, {(25, _ROW)}) == []
    finally:
        _close(window)


def test_the_box_still_applies_while_another_units_tool_hides_it():
    from PyQt5.QtCore import Qt

    window, gate = _gate_window()
    try:
        window.place_unit_action.setChecked(True)
        assert window._current_tool == "place_unit"
        assert not window.gate_walls_param_action.isVisible()
        _select_units(window, [gate])
        _arrow(window, Qt.Key_Down, Qt.ShiftModifier)
        assert gate.y == _ROW + 1.5
        assert len(_walls_on(window, _OLD)) == 4
    finally:
        _close(window)


def test_a_unit_garrisoned_in_the_gate_does_not_block_the_fill_under_it():
    """The occupant stands on the gate's anchor point, a tile of the old line,
    and rides along: it is no blocker where it stood."""
    from descape import wall_run

    window, gate = _gate_window()
    try:
        guard = _add_unit(window, VILLAGER, gate.x, gate.y, garrisoned_in_id=gate.reference_id)
        mm = window.scenario.map_manager
        assert (27, _ROW) in wall_run.blocked_tiles(window.scenario, mm.map_width, mm.map_height)

        _drag_gate(window, 5, 0)

        assert (guard.x, guard.y) == (gate.x, gate.y) == (32.0, _ROW + 0.5)
        assert guard.garrisoned_in_id == gate.reference_id
        assert sorted(int(u.x) for u in _walls_on(window, _OLD)) == [25, 26, 27, 28]
        assert _last_status(window).endswith("replacing 4 walls, filling 4")
    finally:
        _close(window)


def test_something_left_standing_in_the_gateway_keeps_its_tile_open():
    window, gate = _gate_window()
    try:
        # On tile 27, clear of the tile the drag grabs the gate by.
        _add_unit(window, VILLAGER, 27.5, _ROW + 0.5)
        _drag_gate(window, 5, 0)
        assert (gate.x, gate.y) == (32.0, _ROW + 0.5)
        assert sorted(int(u.x) for u in _walls_on(window, _OLD)) == [25, 26, 28]
        # The walls either side of the gap are run ends.
        assert _row_indices(window)[26] == _row_indices(window)[28] == 2.0
        assert _last_status(window).endswith("replacing 4 walls, filling 3 (1 occupied tiles left open)")
    finally:
        _close(window)


def test_a_group_drag_of_the_gate_with_its_whole_run_changes_no_wall():
    """Every wall travelling in the selection is re-derived where it lands
    (GH #266), and a run that already stores its shapes keeps them; nothing
    stays behind to plan."""
    window, gate = _gate_window()
    try:
        walls = _walls_on(window, set(_RUN))
        assert len(walls) == 17
        rotations = {id(u): u.rotation for u in walls}
        _select_units(window, [gate, *walls])
        before = _unit_count(window)

        _drag_gate(window, 0, 3)

        assert (gate.x, gate.y) == (27.0, _ROW + 3.5), _last_status(window)
        assert _walls_on(window, set(_RUN)) == []
        assert all((u.y, u.rotation) == (_ROW + 3.5, rotations[id(u)]) for u in walls)
        assert _unit_count(window) == before
        assert _is_field_delta(window)
        assert _last_status(window) == "Moved 18 units by (0, 3)"
    finally:
        _close(window)


def test_a_group_drag_of_the_gate_with_an_unrelated_unit_plans_like_a_single_move():
    """Onto walls with Free placement off: a group's gate is not blocked by the walls it replaces."""
    window, gate = _gate_window()
    try:
        assert not window.unrestricted_check.isChecked()
        other = _add_unit(window, VILLAGER, 60.5, 60.5)
        _select_units(window, [gate, other])
        before = _unit_count(window)
        records = len(window.edit_history.records)

        _drag_gate(window, 5, 0)

        assert (gate.x, gate.y) == (32.0, _ROW + 0.5), _last_status(window)
        assert (other.x, other.y) == (65.5, 60.5)
        assert _walls_on(window, _NEW) == []
        assert len(_walls_on(window, _OLD)) == 4
        assert _unit_count(window) == before
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert _last_status(window) == "Moved 2 units by (5, 0), replacing 4 walls, filling 4"

        window.undo()
        assert (gate.x, other.x) == (27.0, 60.5)
        assert len(_walls_on(window, _NEW)) == 4 and _walls_on(window, _OLD) == []
    finally:
        _close(window)


def test_a_group_drag_holding_a_gate_plans_the_gate_move_once(monkeypatch):
    """TASK-352: the refusal check and the move share one plan, not a scene walk each."""
    from descape import wall_run

    window, gate = _gate_window()
    try:
        assert not window.unrestricted_check.isChecked()
        other = _add_unit(window, VILLAGER, 60.5, 60.5)
        _select_units(window, [gate, other])
        calls = []
        real = wall_run.scenario_move_plan
        monkeypatch.setattr(wall_run, "scenario_move_plan", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

        _drag_gate(window, 5, 0)

        assert _last_status(window) == "Moved 2 units by (5, 0), replacing 4 walls, filling 4"
        assert len(calls) == 1
    finally:
        _close(window)


_MOVE_HINT = "Tick Free placement to move it anyway."


def _gate_site_state(window, gate) -> tuple:
    """Everything a refused gate move must leave alone."""
    return (
        (gate.x, gate.y),
        _unit_count(window),
        len(window.edit_history.records),
        sorted((id(u), u.x, u.rotation) for u in _walls_on(window, set(_RUN))),
    )


def test_box_off_a_gate_dragged_onto_walls_is_refused_as_before():
    from descape import object_catalog

    window, gate = _gate_window(box=False)
    try:
        state = _gate_site_state(window, gate)
        _drag_gate(window, 5, 0)
        assert _gate_site_state(window, gate) == state
        wall = object_catalog.display_name(WALL)
        assert _last_status(window) == f"Move: blocked by {wall} (Player 1) at (30.5, {_ROW + 0.5:g}). {_MOVE_HINT}"
    finally:
        _close(window)


def test_box_on_something_that_is_not_a_wall_under_the_new_footprint_still_refuses():
    """Only the walls the plan removes stop blocking. The villager stands on
    one of them, and no wall is removed before the refusal is known."""
    from descape import object_catalog

    window, gate = _gate_window()
    try:
        _add_unit(window, VILLAGER, 31.5, _ROW + 0.5)
        state = _gate_site_state(window, gate)
        _drag_gate(window, 5, 0)
        assert _gate_site_state(window, gate) == state
        villager = object_catalog.display_name(VILLAGER)
        assert _last_status(window) == f"Move: blocked by {villager} (Player 1) at (31.5, {_ROW + 0.5:g}). {_MOVE_HINT}"

        # The same in a group: all or nothing.
        other = _add_unit(window, VILLAGER, 60.5, 60.5)
        _select_units(window, [gate, other])
        state = _gate_site_state(window, gate)
        _drag_gate(window, 5, 0)
        assert _gate_site_state(window, gate) == state
        assert (other.x, other.y) == (60.5, 60.5)
        assert _last_status(window).startswith(f"Move: 1 of 2 units can't go there (first: blocked by {villager} ")
    finally:
        _close(window)


def test_box_on_a_wall_the_model_would_not_remove_still_refuses():
    """A unit garrisoned in the wall makes remove_many() refuse it, so the
    plan keeps it, and a wall that stays is a blocker like any other."""
    from descape import object_catalog

    window, gate = _gate_window()
    try:
        (kept,) = _walls_on(window, {(31, _ROW)})
        _add_unit(window, VILLAGER, kept.x, kept.y, garrisoned_in_id=kept.reference_id)
        assert window._ensure_unit_edits().referencing(kept)
        state = _gate_site_state(window, gate)

        _drag_gate(window, 5, 0)

        assert _gate_site_state(window, gate) == state
        wall = object_catalog.display_name(WALL)
        assert _last_status(window) == f"Move: blocked by {wall} (Player 1) at (31.5, {_ROW + 0.5:g}). {_MOVE_HINT}"
    finally:
        _close(window)


def test_box_on_a_move_with_no_gate_in_it_keeps_the_splice_path(monkeypatch):
    """Checked before the scene walk: the planner is never asked, and the
    refresh is the per-unit splice, not the wholesale one."""
    from descape import wall_run

    window, _gate = _gate_window()
    try:
        villager = _add_unit(window, VILLAGER, 60.5, 60.5)

        def never(*_args, **_kwargs):
            raise AssertionError("a move with no gate or wall in it must not plan")

        monkeypatch.setattr(wall_run, "scenario_move_plan", never)
        refreshes = []
        real = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
        )

        _drag_point(window, (60.5, 60.5), (63.5, 60.5))

        assert (villager.x, villager.y) == (63.5, 60.5), _last_status(window)
        assert _is_field_delta(window)
        assert len(refreshes) == 1 and refreshes[0], "expected one spliced refresh"
    finally:
        _close(window)


@pytest.mark.parametrize("encoding", ["radian", "integer"])
def test_a_moved_gates_walls_agree_with_the_render_sides_derivation(encoding):
    """Fills (along the run, then beside a gate stepped off it) and rewrites
    (the gate stepping further away). The fixture's GAIA wall is radian, which
    forces render's override path: no override may disagree with what is
    stored. Without that wall the file is integer-only and render derives
    nothing, so there the stored indices are read directly."""
    from PyQt5.QtCore import Qt

    from descape import render, unit_sprites

    window, gate = _gate_window()
    try:
        if encoding == "integer":
            model = window._ensure_unit_edits()
            with window._unit_edit(model, "Drop the radian wall", [GAIA_PLAYER_ID]):
                model.remove(_gaia_wall(window))
        _drag_gate(window, 5, 0)
        _select_units(window, [gate])
        steps = []
        deltas = []
        for _step in range(2):
            _arrow(window, Qt.Key_Down, Qt.ShiftModifier)
            steps.append(_last_status(window))
            deltas.append(_is_field_delta(window))
            overrides = render.wall_variant_rotation_overrides(window.scenario)
            assert bool(overrides) == (encoding == "radian")
            units = window.scenario.unit_manager.units
            for (player_id, index), derived in overrides.items():
                unit = units[player_id][index]
                assert derived == unit_sprites.variant_index(unit.rotation, 5), (unit.x, unit.y, derived, unit.rotation)
        assert steps[0].endswith(", filling 4")
        assert steps[1].endswith(", reshaping 4 adjacent")
        # GH #266: a plan with rewrites only is a fields-only splice edit; fills keep the wholesale record.
        assert deltas == [False, True]
        # The gate is two rows away again, so the whole row reads as one plain run.
        assert _row_indices(window) == {x: (2.0 if x in (20, 40) else 0.0) for x in range(20, 41)}
    finally:
        _close(window)


@pytest.mark.parametrize("box", [True, False], ids=["box_on", "box_off"])
def test_a_gate_drag_ghost_onto_walls_is_refused_only_where_the_drop_is(box):
    """TASK-327 beside GH #194: with the box ticked the drop replaces the
    walls, so the ghost is not refused; with it off the walls block, so it is."""
    from PyQt5.QtCore import QEvent, Qt

    window, gate = _gate_window(box=box)
    try:
        view = window.map_view
        start, end = _point_pos(window, *_GRAB), _point_pos(window, _GRAB[0] + 5, _GRAB[1])
        _HELD["value"] = True
        try:
            view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
            view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
            assert view._unit_ghost_item is not None
            assert view._unit_ghost_item.refused is not box
            view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        finally:
            _HELD["value"] = False
        assert ((gate.x, gate.y) != (27.0, _ROW + 0.5)) is box, _last_status(window)
    finally:
        _close(window)


# --- A moved wall takes the shape of its new place (GH #266) ----------------
#
# The integer cases store the fixture's one radian wall as the literal 2.0
# before laying the run, so the file is integer-encoded and DEscape draws each
# stored index as is. Laying the run bumps unit_gen past that bare write.


def _moved_wall_window(integer: bool = True):
    """(window, wall, west, east): _RUN laid on _ROW with Move armed; `wall`
    is the piece on (30, _ROW), `west` and `east` its run neighbours, all
    storing the run index 0."""
    window = _window()
    if integer:
        _gaia_wall(window).rotation = 2.0
    window.on_shape_commit(_RUN)
    window.move_action.setChecked(True)
    (west,), (wall,), (east,) = (_walls_on(window, {(x, _ROW)}) for x in (29, 30, 31))
    assert (west.rotation, wall.rotation, east.rotation) == (0.0, 0.0, 0.0)
    return window, wall, west, east


def _drag_wall_down(window, rows: int) -> None:
    _drag_point(window, (30.5, _ROW + 0.5), (30.5, _ROW + 0.5 + rows))


def _shapes(*walls) -> tuple:
    return tuple(u.rotation for u in walls)


def _derived_around(window, tiles) -> int:
    """Asserts every wall on or beside `tiles` stores the index its neighbours
    derive (2 with none) and returns how many it checked."""
    from descape import unit_sprites, wall_run

    mm = window.scenario.map_manager
    connectors, walls = wall_run.wall_scene(window.scenario, mm.map_width, mm.map_height)
    near = {(tx + dx, ty + dy) for tx, ty in tiles for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    checked = 0
    for wall in walls:
        if (wall.tx, wall.ty) not in near:
            continue
        mask = unit_sprites.neighbour_mask(wall.tx, wall.ty, connectors, diagonals=True)
        derived = unit_sprites.wall_variant_from_neighbours8(mask)
        assert unit_sprites.variant_index(wall.unit.rotation, 5) == (2 if derived is None else derived), (wall.tx, wall.ty)
        checked += 1
    return checked


def _file_is_radian(window) -> bool:
    from descape import render, unit_sprites

    return unit_sprites.file_is_radian(
        [
            render.stored_rotation(player_id, u)
            for player_id, units in enumerate(window.scenario.unit_manager.units)
            for u in units
            if unit_sprites.rotation_variant_eligible(u.unit_const)
        ],
        5,
    )


def test_a_wall_dragged_off_its_run_is_a_tower_and_so_are_its_old_neighbours():
    """The GH #266 screenshot case, as one fields-only record: undo puts back
    the position and all three indices, redo reapplies them."""
    window, wall, west, east = _moved_wall_window()
    try:
        records = len(window.edit_history.records)

        _drag_wall_down(window, 3)

        assert (wall.x, wall.y) == (30.5, _ROW + 3.5), _last_status(window)
        assert _shapes(wall, west, east) == (2.0, 2.0, 2.0)
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        assert _last_status(window).endswith(", reshaping 2 adjacent")
        assert _derived_around(window, [(30, _ROW), (30, _ROW + 3)]) == 3
        assert not _file_is_radian(window)

        window.undo()
        assert (wall.x, wall.y) == (30.5, _ROW + 0.5)
        assert _shapes(wall, west, east) == (0.0, 0.0, 0.0)
        window.redo()
        assert (wall.y, *_shapes(wall, west, east)) == (_ROW + 3.5, 2.0, 2.0, 2.0)
    finally:
        _close(window)


def test_a_shift_nudge_carries_a_wall_off_its_run_and_a_plain_one_keeps_its_tile(monkeypatch):
    """A plain arrow moves 0.1 inside the tile: nothing reshapes and the one
    splice is the wall's own. Shift moves it a tile, diagonal to both old
    neighbours: a tower, each of them a run end, one splice per reshaped wall."""
    from PyQt5.QtCore import Qt

    window, wall, west, east = _moved_wall_window()
    try:
        _select_units(window, [wall])
        refreshes = []
        real = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
        )

        _arrow(window, Qt.Key_Down)
        assert wall.y == pytest.approx(_ROW + 0.6)
        assert _shapes(wall, west, east) == (0.0, 0.0, 0.0)
        assert _is_field_delta(window) and "reshaping" not in _last_status(window)
        assert [s.unit for s in refreshes[-1]] == [wall]

        window.undo()
        _arrow(window, Qt.Key_Down, Qt.ShiftModifier)
        assert wall.y == _ROW + 1.5
        assert _shapes(wall, west, east) == (2.0, 2.0, 2.0)
        assert _is_field_delta(window)
        assert _last_status(window).endswith(", reshaping 2 adjacent")
        assert sorted(int(s.unit.x) for s in refreshes[-1]) == [29, 30, 31]
    finally:
        _close(window)


@pytest.mark.parametrize("how", ["group_drag", "y_step"])
def test_two_walls_carried_out_of_a_run_together_reshape_the_ends_they_leave(how):
    """A group drag and the Inspector's Y step on a two-wall selection (GH
    #154) both go through _move_units: the pair lands as two run ends, and
    the walls either side of the gap become run ends."""
    window, wall, west, east = _moved_wall_window()
    try:
        (beyond,) = _walls_on(window, {(32, _ROW)})
        _select_units(window, [wall, east])
        records = len(window.edit_history.records)

        if how == "group_drag":
            _drag_wall_down(window, 3)
        else:
            window.units_panel.unit_field_editors["y"].stepBy(3)

        assert (wall.y, east.y) == (_ROW + 3.5, _ROW + 3.5), _last_status(window)
        assert _shapes(wall, east, west, beyond) == (2.0, 2.0, 2.0, 2.0)
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        moved = "Moved 2 units by (0, 3)" if how == "group_drag" else "Moved 2 units by +3 on Y"
        assert _last_status(window) == f"{moved}, reshaping 2 adjacent"
    finally:
        _close(window)


def test_typing_x_on_a_wall_carries_it_off_its_run():
    window, wall, west, east = _moved_wall_window()
    try:
        _select_units(window, [wall])
        records = len(window.edit_history.records)

        window.units_panel.unit_field_editors["x"].setValue(50.5)

        assert (wall.x, wall.y) == (50.5, _ROW + 0.5)
        assert _shapes(wall, west, east) == (2.0, 2.0, 2.0)
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        window.undo()
        assert (wall.x, *_shapes(wall, west, east)) == (30.5, 0.0, 0.0, 0.0)
    finally:
        _close(window)


def test_scatter_move_mode_carries_a_wall_off_its_run():
    from descape.scatter_dialog import ScatterParams

    window, wall, west, east = _moved_wall_window()
    try:
        keys = _select_units(window, [wall])
        entries = [window.map_view._unit_index.entry_for_key(key) for key in keys]
        area = [(x, y) for x in range(50, 54) for y in range(50, 54)]
        params = ScatterParams(
            tiles=area, unit_const=None, player=PLAYER, count=None, density=None, seed=7, jitter=0.0,
            min_spacing=0, move_selected=True,
        )
        records = len(window.edit_history.records)

        window._scatter_selected_units(entries, params)

        assert (int(wall.x), int(wall.y)) in area, _last_status(window)
        assert _shapes(wall, west, east) == (2.0, 2.0, 2.0)
        assert len(window.edit_history.records) == records + 1
        assert _is_field_delta(window)
        assert _last_status(window) == "Scattered 1 selected units across the area, reshaping 2 adjacent"
    finally:
        _close(window)


def test_in_a_radian_file_the_drag_writes_literal_shapes_the_render_agrees_with():
    """The fixture as loaded: its radian GAIA wall keeps the file radian. One
    run neighbour stores radian noise (index 1); left a run end, it stores
    the literal 2.0, and no override disagrees with a stored index."""
    from descape import render, unit_sprites

    window, wall, west, east = _moved_wall_window(integer=False)
    try:
        west.rotation = 1.256637
        window._after_unit_mutation()

        _drag_wall_down(window, 3)

        assert _shapes(wall, west, east) == (2.0, 2.0, 2.0), _last_status(window)
        assert _file_is_radian(window)
        assert _is_field_delta(window)
        overrides = render.wall_variant_rotation_overrides(window.scenario)
        assert overrides, "the radian wall should force the override path"
        units = window.scenario.unit_manager.units
        for (player_id, index), derived in overrides.items():
            unit = units[player_id][index]
            assert derived == unit_sprites.variant_index(unit.rotation, 5), (unit.x, unit.y, derived, unit.rotation)
    finally:
        _close(window)


def test_the_status_counts_no_rewrite_of_a_wall_the_user_moved():
    """A wall moved within its own tile can be re-derived as a staying wall,
    but it is one of the movers, so it is not reported as reshaped."""
    from descape import wall_run
    from descape.viewer import ViewerWindow

    mover, other = object(), object()
    plan = wall_run.MovePlan(removals=[], fills=[], rewrites=[(1, mover, 2), (1, other, 2)], moved=[(1, object(), 2)])
    assert ViewerWindow._gate_move_clauses(plan, [mover]) == ", reshaping 1 adjacent"
    assert ViewerWindow._gate_move_clauses(plan) == ", reshaping 2 adjacent"
