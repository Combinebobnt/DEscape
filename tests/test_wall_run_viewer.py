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

import json
import threading
from pathlib import Path

import pytest

from descape import game_dataset, settings, unit_sprites

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


# TASK-444: these tests flaked as if Palisade and Aqueduct had stopped being walls. The guard below
# fails such a test with where the wall set came from, instead of with a wrong unit count.
_WALL_SOURCES = {
    name: getattr(unit_sprites, name)
    for name in (
        "graphic_map",
        "rotation_variant_eligible",
        "wall_family_consts",
        "rotation_variant_const_set",
        "_ROTATION_VARIANT_CONSTS",
    )
}
_WALL_FIELDS = ("angle_count", "file_name")
_WALL_ENTRIES = {
    const: {field: entry[field] for field in _WALL_FIELDS}
    for const, entry in (
        (const, json.loads(Path(unit_sprites.GRAPHIC_MAP_PATH).read_text())["graphics"][str(const)])
        for const in (PALISADE, WALL, AQUEDUCT)
    )
}


def _describe(obj) -> str:
    code = getattr(obj, "__code__", None)
    where = f" at {code.co_filename}:{code.co_firstlineno}" if code is not None else ""
    name = getattr(obj, "__qualname__", type(obj).__name__)
    return f"{getattr(obj, '__module__', '?')}.{name}{where}"


def _wall_state() -> str | None:
    """None while Place Unit sees the real wall set, else where it came from.
    Reads viewer._wall_consts(), which is cached per dataset, and the live set
    beside it: a cache filled under another test's patch outlives the patch."""
    from descape import viewer

    walls = {PALISADE, WALL, AQUEDUCT}
    cached, live = viewer._wall_consts(), frozenset(unit_sprites.wall_family_consts())
    if walls <= cached and walls <= live and game_dataset.cache_key() == game_dataset.AOE2:
        return None
    parts = [f"viewer._wall_consts() = {sorted(cached)}, unit_sprites.wall_family_consts() = {sorted(live)}"]
    for name, original in _WALL_SOURCES.items():
        now = getattr(unit_sprites, name)
        if now is not original:
            parts.append(f"unit_sprites.{name} is patched: {_describe(now)}")
    parts.append(f"cache_key() = {game_dataset.cache_key()!r}")
    table = unit_sprites.graphic_map()
    parts.append(f"len(graphic_map()) = {len(table)}")
    for const, want in _WALL_ENTRIES.items():
        entry = table.get(const)
        got = None if entry is None else {field: entry.get(field) for field in _WALL_FIELDS}
        if got != want:
            parts.append(f"graphic_map()[{const}] = {got}, the JSON has {want}")
    parts.append(f"threads = {[t.name for t in threading.enumerate()]}")
    return "; ".join(parts)


@pytest.fixture(autouse=True)
def _wall_consts_intact():
    state = _wall_state()
    assert state is None, f"wall set wrong before the test: {state}"
    yield
    state = _wall_state()
    assert state is None, f"wall set changed during the test: {state}"


def _trace(view, what: str, tile=None) -> None:
    """One routing line per event, shown only when a test fails (TASK-444)."""
    print(
        f"{what} {tile}: tool={view._tool!r} shape_query={view._place_shape_query()!r} "
        f"shape_tool={view._is_shape_tool()} anchor={view._shape_anchor!r}"
    )


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
    _trace(map_view, "press", tile)


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
    _trace(map_view, "release", tile)


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
        _trace(view, "click press", (x, y))
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
        _trace(view, "click release", (x, y))
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


def _shape_window(const: int, kind: str):
    """Unit Shape (GH #365, which replaced Wall Rectangle and Wall Circle) with `kind` in its Shape combo."""
    from descape.tool_option_bindings import find_data

    window = _window(const)
    window.unit_shape_action.setChecked(True)
    assert window._current_tool == "unit_shape"
    window.unit_shape_kind_combo.setCurrentIndex(find_data(window.unit_shape_kind_combo, kind))
    return window


def _rect_window(const: int = WALL):
    return _shape_window(const, "rectangle")


def _ring(x0, y0, x1, y1):
    from descape import shape_tools

    return shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, 120, 120)


@pytest.mark.parametrize("tool", ["unit_line", "unit_shape"])
def test_the_unit_line_and_shape_buttons_only_show_in_units_mode(tool):
    """QAction.isVisible(), never the widget's -- an offscreen unshown
    widget reports False unconditionally and would pass vacuously."""
    window = _rect_window()
    try:
        action = getattr(window, f"{tool}_action")
        assert action.isVisible() and action.isEnabled()
        window.mode_combo.setCurrentText("Terrain")
        assert not action.isVisible()
        window.mode_combo.setCurrentText("View")
        assert not action.isVisible()
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


def test_a_gate_pick_starts_no_drag_and_says_why():
    window = _rect_window(GATE)
    try:
        before = _unit_count(window)
        _press(window.map_view, (20, 40))
        assert window.map_view._shape_anchor is None and window.map_view._shape_latch is None
        _move(window.map_view, (28, 44))
        _release(window.map_view, (28, 44))
        assert _unit_count(window) == before
        assert not window.edit_history.can_undo
        assert "Unit Shape: gates are placed one at a time" in window.status_log.toPlainText()
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
    there would rewrite it to a literal integer, the file would read as
    integer-only, and render would derive only lone walls and run ends
    (TASK-469; the junction itself is test_wall_run.py's)."""
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
    return _shape_window(const, "ellipse")


def _ellipse(x0, y0, x1, y1):
    from descape import shape_tools

    return shape_tools.ellipse_perimeter_tiles(x0, y0, x1, y1, 120, 120)


@pytest.mark.parametrize(("tool_id", "label"), [("unit_line", "Unit Line"), ("unit_shape", "Unit Shape")])
def test_unit_line_and_shape_are_registered_unbound_and_units_only(tool_id, label):
    tool = next(t for t in settings.TOOLS if t.tool_id == tool_id)
    assert (tool.label, tool.drag_shape, tool.modes, tool.default_key) == (label, tool_id, ("units",), "")
    assert tool.supports_skip_occupied and tool.supports_unrestricted
    assert not tool.supports_brush and not tool.supports_off_grid
    assert not any(t.tool_id in ("wall_rect", "wall_circle") for t in settings.TOOLS)


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
        assert sorted(map_view._shape_tiles(preview=True)) == sorted(ring)
        assert sorted(map_view._shape_tiles(preview=False)) == sorted(ring)
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
        assert sorted(tiles) == sorted(_ellipse(min(xs), min(ys), max(xs), max(ys)))
        # A circle: the ring is its own transpose about the box's diagonal.
        assert {(y - min(ys), x - min(xs)) for x, y in tiles} == {(x - min(xs), y - min(ys)) for x, y in tiles}
        _release(map_view, _CIRCLE_B)
    finally:
        _close(window)


def test_a_villager_ellipse_places_units_not_walls():
    window = _circle_window(VILLAGER)
    try:
        before = _unit_count(window)
        _drag(window.map_view, _CIRCLE_A, _CIRCLE_B)
        assert _unit_count(window) > before
        assert [r.kind for r in window.edit_history.records] == ["unit"]
        assert "Placed " in window.status_log.toPlainText().splitlines()[-1]
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
    integer, the file would read as integer-only, and render would derive
    only lone walls and run ends (the junction itself is test_wall_run.py's)."""
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
        assert window._current_tool == "unit_shape"
        assert window.unit_shape_action.isChecked()
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
        for tool in ("place_unit", "unit_line", "unit_shape"):
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
        assert shown == {"place_unit", "unit_line", "unit_shape"}
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
        assert window._current_tool == "unit_shape"
        assert window.unit_shape_action.isChecked()
        assert window.units_panel.selected_object_const() == PALISADE
        assert window.wall_pick_button.text() == "Wall: Palisade Wall"
        # And the click now re-arms the menu's pick, not Stone Wall.
        _pick(window, VILLAGER)
        window.wall_pick_button.click()
        assert window.units_panel.selected_object_const() == PALISADE
        assert window._current_tool == "unit_shape"
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
        _trace(view, "drag press", a)
        view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
        _trace(view, "drag release", b)
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
    only lone walls and run ends, so there the stored indices are read directly."""
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
        assert _last_status(window) == "Set unit X, reshaping 2 adjacent"
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


# --- GH #280: a gate turned during its drag ---------------------------------------------


def _turn_drag(window, a, b, cw: int = 0, ccw: int = 0) -> None:
    """_drag_point(), with the Rotate actions fired while the button is down."""
    from PyQt5.QtCore import QEvent, Qt

    view = window.map_view
    start, end = _point_pos(window, *a), _point_pos(window, *b)
    _HELD["value"] = True
    try:
        view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton))
        view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton))
        for _ in range(cw):
            window.rotate_cw_action.trigger()
        for _ in range(ccw):
            window.rotate_ccw_action.trigger()
        view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton))
    finally:
        _HELD["value"] = False


_GATE_Y = 88  # the (1, 4) sibling of GATE, two steps along


def _player_gate(window):
    return next(u for u in window.scenario.unit_manager.units[PLAYER] if u.unit_const in (GATE, _GATE_Y))


def test_a_gate_turned_then_moved_out_of_its_run_fills_the_gap_and_undo_restores_it_all():
    """GH #280's reported case: read after a turn at the source, the gap reads as a line across the wall."""
    window, gate = _gate_window()
    try:
        before = (_unit_count(window), gate.x, gate.y)
        records = len(window.edit_history.records)

        _turn_drag(window, _GRAB, (50.5, _ROW + 10.1), cw=2)

        assert gate.unit_const == _GATE_Y
        assert (gate.x, gate.y) == (50.5, _ROW + 10.0)
        filled = _walls_on(window, _OLD)
        assert sorted(int(u.x) for u in filled) == [25, 26, 27, 28]
        assert all(u.unit_const == WALL for u in filled)
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert window.edit_history.peek_undo().label == "Move and rotate unit"
        assert "filling 4" in _last_status(window)

        window.undo()
        restored = _player_gate(window)
        assert (_unit_count(window), restored.x, restored.y) == before
        assert restored.unit_const == GATE
        assert _walls_on(window, _OLD) == []
    finally:
        _close(window)


def test_a_gate_turned_onto_a_run_replaces_the_walls_under_its_turned_footprint_only():
    window, gate = _gate_window()
    try:
        _turn_drag(window, _GRAB, (36.5, _ROW + 0.1), cw=2)
        assert gate.unit_const == _GATE_Y
        assert (gate.x, gate.y) == (36.5, float(_ROW))
        assert _walls_on(window, {(36, _ROW)}) == []
        assert len(_walls_on(window, {(35, _ROW), (37, _ROW)})) == 2, "a (4, 1) footprint would take them"
        assert "replacing 1 wall," in _last_status(window)
    finally:
        _close(window)


def test_a_gate_turned_back_in_the_drag_plans_as_an_unturned_move():
    window, gate = _gate_window()
    try:
        _turn_drag(window, _GRAB, (_GRAB[0] + 5, _GRAB[1]), cw=1, ccw=1)
        assert (gate.unit_const, gate.x, gate.y) == (GATE, 32.0, _ROW + 0.5)
        assert _walls_on(window, _NEW) == []
        assert len(_walls_on(window, _OLD)) == 4
        assert window.edit_history.peek_undo().label == "Move unit"
        assert _moved_gate_status(window) == f"to (32, {_ROW + 0.5:g}), replacing 4 walls, filling 4"
    finally:
        _close(window)


def test_a_gate_turned_on_its_own_low_corner_plans_with_the_box_as_a_rotate():
    """Its in-place drop still reads the plan: the rest of the line fills, wholesale, with no position write."""
    window, gate = _gate_window()
    try:
        # The (1, 4) sibling's anchor on the low corner (25, _ROW), and off the tie on the even axis.
        _turn_drag(window, _GRAB, (25.5, _ROW + 2.1), cw=2)
        assert (gate.unit_const, gate.x, gate.y) == (_GATE_Y, 25.5, _ROW + 2.0)
        assert sorted(int(u.x) for u in _walls_on(window, _OLD)) == [26, 27, 28]
        assert window.edit_history.peek_undo().label == "Rotate unit"
    finally:
        _close(window)


def test_a_group_holding_a_turned_gate_plans_it_once_by_its_turned_span(monkeypatch):
    from descape import wall_run

    window, gate = _gate_window()
    try:
        villager = _add_unit(window, VILLAGER, 27.5, _ROW + 6.5)
        _select_units(window, [gate, villager])
        plans: list = []
        real = wall_run.scenario_move_plan
        monkeypatch.setattr(
            wall_run, "scenario_move_plan", lambda *a, **k: (plans.append(k.get("consts")), real(*a, **k))[1]
        )
        _turn_drag(window, _GRAB, (_GRAB[0] + 24, _GRAB[1] + 10), cw=2)
        assert len(plans) == 1
        assert plans[0][id(gate)] == _GATE_Y
        assert gate.unit_const == _GATE_Y
        # Rotate, then the delta: (1, 4) on the low corner (25, _ROW) is (25.5, _ROW + 2).
        assert (gate.x, gate.y) == (49.5, _ROW + 12.0)
        assert sorted(int(u.x) for u in _walls_on(window, _OLD)) == [25, 26, 27, 28]
    finally:
        _close(window)


# --- GH #339: a gate's orientation cycle clears the walls under its new footprint ----
#
# A y run on x = _COL, and a Stone Gate clicked across it with its low corner
# on (_COL, _ROW) (GH #159's path: one wall removed, three tiles sticking out
# along x). Turned two steps it lies along the run, on _TURNED.

_COL = 50
_TURNED = {(_COL, y) for y in range(_ROW, _ROW + 4)}
_ARCHER = 4


def _crossed_gate_window(run_owner: int = PLAYER):
    """(window, gate): the y run (owned by `run_owner`) with the x gate
    across it, the gate selected under Move so no Place ghost turns instead."""
    window = _window()
    window.unrestricted_check.setChecked(True)
    assert window.units_panel.select_owner(run_owner)
    window.on_shape_commit([(_COL, y) for y in range(_ROW - 4, _ROW + 7)])
    assert window.units_panel.select_owner(PLAYER)
    _pick(window, GATE)
    _click_point(window, _COL + 1.7, _ROW + 0.4)
    gate = window.scenario.unit_manager.units[PLAYER][-1]
    assert gate.unit_const == GATE and (gate.x, gate.y) == (_COL + 2.0, _ROW + 0.5)
    assert _walls_on(window, {(_COL, _ROW)}) == [] and len(_walls_on(window, _TURNED)) == 3
    window.move_action.setChecked(True)
    _select_units(window, [gate])
    return window, gate


def _footprint(window, unit, const=None) -> set:
    """`unit`'s sparse tiles, or as `const` re-anchored the way a cycle re-anchors it."""
    from descape import render
    from descape.unit_model import reanchored

    mm = window.scenario.map_manager
    if const is None:
        return set(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
    return set(render.occupied_tiles_for(const, *reanchored(unit, const), mm.map_width, mm.map_height))


def test_rotate_turns_a_gate_onto_a_crossing_run_and_clears_the_walls_under_it():
    window, gate = _crossed_gate_window()
    try:
        under = _walls_on(window, _TURNED)
        rotations = {id(u): u.rotation for u in under}
        start = (gate.unit_const, gate.x, gate.y)
        before = _unit_count(window)
        records = len(window.edit_history.records)

        window.on_unit_rotate_coarse(1)

        assert gate.unit_const == _GATE_Y and _footprint(window, gate) == _TURNED
        assert _walls_on(window, _TURNED) == []
        assert _unit_count(window) == before - 3
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert "replacing 3 walls" in _last_status(window)

        window.undo()
        restored = _walls_on(window, _TURNED)
        assert {id(u) for u in restored} == set(rotations)
        assert all(u.rotation == rotations[id(u)] for u in restored)
        assert (gate.unit_const, gate.x, gate.y) == start

        window.redo()
        assert _walls_on(window, _TURNED) == [] and gate.unit_const == _GATE_Y
    finally:
        _close(window)


def test_a_gate_and_an_archer_rotate_in_the_one_wholesale_record():
    from descape import unit_pick, unit_rotation

    window, gate = _crossed_gate_window()
    try:
        assert unit_rotation.rotation_is_angle(_ARCHER)
        archer = _add_unit(window, _ARCHER, 70.5, 70.5)
        _select_units(window, [gate, archer])
        facing, start = archer.rotation, (gate.unit_const, gate.x, gate.y)
        records = len(window.edit_history.records)

        window.on_unit_rotate_coarse(1)

        assert gate.unit_const == _GATE_Y and archer.rotation != facing
        assert _walls_on(window, _TURNED) == []
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert len(_walls_on(window, _TURNED)) == 3
        assert archer.rotation == facing and (gate.unit_const, gate.x, gate.y) == start
        assert window._selection == [unit_pick.unit_key(PLAYER, gate), unit_pick.unit_key(PLAYER, archer)]
    finally:
        _close(window)


def test_a_tree_in_a_turned_gates_selection_still_cycles_its_variant():
    from descape import unit_pick, unit_variant

    window, gate = _crossed_gate_window()
    try:
        assert unit_variant.is_cyclable(TREE)
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [GAIA_PLAYER_ID]):
            tree = model.add(GAIA_PLAYER_ID, TREE, 72.5, 72.5)
        window._selection = [unit_pick.unit_key(PLAYER, gate), unit_pick.unit_key(GAIA_PLAYER_ID, tree)]
        window._refresh_selection_view()
        variant = tree.rotation
        records = len(window.edit_history.records)

        window.on_unit_rotate_coarse(1)

        assert gate.unit_const == _GATE_Y and _walls_on(window, _TURNED) == []
        assert tree.rotation != variant
        assert len(window.edit_history.records) == records + 1
        window.undo()
        assert tree.rotation == variant and len(_walls_on(window, _TURNED)) == 3
    finally:
        _close(window)


def test_the_rotation_field_typed_to_the_runs_orientation_clears_the_same_walls():
    window, gate = _crossed_gate_window()
    try:
        before = _unit_count(window)
        records = len(window.edit_history.records)

        window.units_panel.unit_field_editors["rotation"].setValue(2)

        assert gate.unit_const == _GATE_Y and _walls_on(window, _TURNED) == []
        assert _unit_count(window) == before - 3
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Set unit Rotation"
        assert _last_status(window) == "Set unit Rotation, replacing 3 walls"
        window.undo()
        assert gate.unit_const == GATE and len(_walls_on(window, _TURNED)) == 3
    finally:
        _close(window)


def test_a_wall_an_intermediate_orientation_cleared_stays_cleared_after_the_next_press():
    """The plan's decision A: each Rotate press is its own record, and a wall
    the diagonal passed over is not put back when the next press moves on."""
    window, gate = _crossed_gate_window()
    try:
        diagonal = _footprint(window, gate, 659)
        assert len(diagonal) == 6
        tile = min(diagonal - _footprint(window, gate) - _TURNED)
        assert tile[0] != _COL
        _pick(window, WALL)
        window.place_unit_action.setChecked(True)
        window.on_shape_commit([tile])
        assert len(_walls_on(window, {tile})) == 1
        window.move_action.setChecked(True)
        _select_units(window, [gate])

        window.on_unit_rotate(1)
        assert gate.unit_const == 659 and _walls_on(window, diagonal) == []
        window.on_unit_rotate(1)

        assert gate.unit_const == _GATE_Y
        assert _walls_on(window, {tile}) == [], "decision A: the diagonal's removal stands"
        assert _walls_on(window, _TURNED) == []
    finally:
        _close(window)


def test_a_free_standing_gates_cycle_keeps_the_fields_only_splice_record(monkeypatch):
    window, _gate = _crossed_gate_window()
    try:
        _pick(window, GATE)
        window.place_unit_action.setChecked(True)
        _click_point(window, 81.7, 90.4)
        lone = window.scenario.unit_manager.units[PLAYER][-1]
        assert lone.unit_const == GATE and not _walls_on(window, _footprint(window, lone, 659) | _footprint(window, lone))
        window.move_action.setChecked(True)
        _select_units(window, [lone])
        before = _unit_count(window)
        refreshes = []
        real = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
        )

        window.on_unit_rotate(1)

        assert lone.unit_const == 659 and _unit_count(window) == before
        assert _is_field_delta(window)
        assert len(refreshes) == 1 and refreshes[0], "expected one spliced refresh"
        assert "replacing" not in _last_status(window) and "reshaping" not in _last_status(window)
    finally:
        _close(window)


def test_a_referenced_wall_under_the_turned_gate_stays():
    window, gate = _crossed_gate_window()
    try:
        (held,) = _walls_on(window, {(_COL, _ROW + 2)})
        _add_unit(window, VILLAGER, _COL + 0.5, _ROW + 2.5, garrisoned_in_id=held.reference_id)
        _select_units(window, [gate])

        window.on_unit_rotate_coarse(1)

        assert gate.unit_const == _GATE_Y
        assert _walls_on(window, _TURNED) == [held]
        assert "replacing 2 walls" in _last_status(window)
    finally:
        _close(window)


def test_undo_restores_walls_another_player_owns_under_the_turned_gate():
    """gate_move_players() reaches the snapshot: the run is player 2's, the gate player 1's."""
    window, gate = _crossed_gate_window(run_owner=2)
    try:
        assert gate.unit_const == GATE
        under = _walls_on(window, _TURNED)
        assert {id(u) for u in under} <= {id(u) for u in window.scenario.unit_manager.units[2]}

        window.on_unit_rotate_coarse(1)
        assert _walls_on(window, _TURNED) == []

        window.undo()
        assert {id(u) for u in _walls_on(window, _TURNED)} == {id(u) for u in under}
        assert all(any(u is w for w in window.scenario.unit_manager.units[2]) for u in under)
    finally:
        _close(window)


def test_a_notch_on_a_mixed_gate_group_clears_the_walls_under_each_new_footprint():
    """The Rotation field's step over "(mixed)" gates (_step_rotation_field) is a cycle too."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    window, gate = _crossed_gate_window()
    try:
        other = _add_unit(window, _GATE_Y, 90.5, 92.0)
        landing = _footprint(window, gate, 659)
        _pick(window, WALL)
        window.place_unit_action.setChecked(True)
        window.on_shape_commit(sorted(landing - _footprint(window, gate) - {(_COL, y) for y in range(_ROW, _ROW + 7)}))
        under = _walls_on(window, landing)
        assert len(under) >= 2
        window.move_action.setChecked(True)
        _select_units(window, [gate, other])
        spin = window.units_panel.unit_field_editors["rotation"]
        assert spin.text() == "(mixed)"
        records = len(window.edit_history.records)

        centre = QPointF(spin.rect().center())
        QApplication.sendEvent(
            spin, QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        )

        assert gate.unit_const == 659 and _walls_on(window, landing) == []
        assert len(window.edit_history.records) == records + 1
        assert not _is_field_delta(window)
        assert _last_status(window) == f"Turned 2 units by +1, replacing {len(under)} walls"
        window.undo()
        assert {id(u) for u in _walls_on(window, landing)} == {id(u) for u in under}
    finally:
        _close(window)


# --- GH #340: a removed wall or gate reshapes the walls left beside it -------
#
# The integer cases make the fixture's one (radian) wall literal before laying
# the run, so the file is integer-only: render memoizes on unit_gen, which a
# direct field write does not bump and the run's add_many() does.

_DEL_ROW = 30
_DEL_RUN = [(x, _DEL_ROW) for x in range(30, 41)]


def _run_window(integer: bool = True):
    """(window, {tile x: wall}): a Stone Wall run along _DEL_ROW, Place Unit still armed."""
    window = _window()
    if integer:
        _gaia_wall(window).rotation = 2.0
    window.on_shape_commit(_DEL_RUN)
    return window, {int(u.x): u for u in _walls_on(window, set(_DEL_RUN))}


def _delete(window, units) -> None:
    _select_units(window, units)
    window.on_unit_delete(None)


def test_deleting_a_mid_run_wall_turns_its_two_neighbours_into_run_ends():
    from descape import render, unit_sprites

    window, row = _run_window()
    try:
        assert (row[34].rotation, row[36].rotation) == (0.0, 0.0)
        before = {x: u.rotation for x, u in row.items()}
        records = len(window.edit_history.records)

        _delete(window, [row[35]])

        assert (row[34].rotation, row[36].rotation) == (2.0, 2.0)
        assert not any(u is row[35] for u in window.scenario.unit_manager.units[PLAYER])
        assert len(window.edit_history.records) == records + 1
        assert _last_status(window).endswith(", reshaping 2 adjacent")
        tiles = {(int(u.x), int(u.y)) for u in _walls_on(window, set(_DEL_RUN))}
        for x in range(33, 38):
            if x == 35:
                continue
            derived = unit_sprites.wall_variant_from_neighbours8(unit_sprites.neighbour_mask(x, _DEL_ROW, tiles, diagonals=True))
            assert row[x].rotation == float(2 if derived is None else derived), x
        assert render.wall_variant_rotation_overrides(window.scenario) == {}, "the file stays integer-only"

        window.undo()
        assert any(u is row[35] for u in window.scenario.unit_manager.units[PLAYER])
        assert {x: u.rotation for x, u in row.items()} == before
        window.redo()
        assert (row[34].rotation, row[36].rotation) == (2.0, 2.0)
    finally:
        _close(window)


def test_cut_units_and_finds_delete_reshape_the_same_way():
    for how in ("cut", "find"):
        window, row = _run_window()
        try:
            if how == "cut":
                _select_units(window, [row[35], row[36]])
                window.cut_units()
            else:
                index = window.map_view._unit_index
                entries = [index.entry_for_key((PLAYER, row[x].reference_id)) for x in (35, 36)]
                assert window._delete_units(entries) == (2, 0, 0)
            assert (row[34].rotation, row[37].rotation) == (2.0, 2.0), how
            assert _last_status(window).endswith(", reshaping 2 adjacent"), how
        finally:
            _close(window)


def test_deleting_both_ends_of_a_three_wall_run_leaves_the_middle_a_tower():
    window = _window()
    try:
        _gaia_wall(window).rotation = 2.0
        window.on_shape_commit([(x, _DEL_ROW) for x in range(50, 53)])
        west, middle, east = sorted(_walls_on(window, {(x, _DEL_ROW) for x in range(50, 53)}), key=lambda u: u.x)
        assert middle.rotation == 0.0

        _delete(window, [west, east])

        assert middle.rotation == 2.0
        assert _last_status(window).endswith(", reshaping 1 adjacent")
    finally:
        _close(window)


@pytest.mark.parametrize("what", ["villager", "lone wall"])
def test_a_delete_that_reshapes_no_wall_keeps_the_single_unit_splice_path(monkeypatch, what):
    window = _window()
    try:
        if what == "villager":
            unit = _add_unit(window, VILLAGER, 60.5, 60.5)
        else:
            window.on_shape_commit([(60, 60)])
            (unit,) = _walls_on(window, {(60, 60)})
        refreshes = []
        real = window._after_unit_mutation
        monkeypatch.setattr(
            window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
        )

        _delete(window, [unit])

        assert not any(u is unit for u in window.scenario.unit_manager.units[PLAYER])
        assert len(refreshes) == 1 and refreshes[0], "expected one spliced refresh"
        assert "reshaping" not in _last_status(window)
    finally:
        _close(window)


def test_a_radian_files_neighbour_is_rewritten_and_the_file_stays_radian():
    """Decision 1: both file classes. The run sits clear of the radian GAIA
    wall at (7, 5), and one neighbour is seeded with radian index 1."""
    from descape import render

    window, row = _run_window(integer=False)
    try:
        row[34].rotation = 1.256637
        assert render.wall_variant_rotation_overrides(window.scenario)

        _delete(window, [row[35]])

        assert row[34].rotation == 2.0 and row[36].rotation == 2.0
        assert render.wall_variant_rotation_overrides(window.scenario), "the GAIA wall keeps the file radian"
    finally:
        _close(window)


# --- TASK-585: a paste reshapes the walls beside pasted walls and gates ----------


def _paste_units_at(window, tile) -> None:
    window.on_hover(tile)
    window.paste_region()


def _record_refreshes(window, monkeypatch) -> list:
    refreshes = []
    real = window._after_unit_mutation
    monkeypatch.setattr(
        window, "_after_unit_mutation", lambda changed=None, **kw: (refreshes.append(changed), real(changed, **kw))[1]
    )
    return refreshes


def test_a_wall_cut_from_a_run_and_pasted_back_joins_it_again():
    """The task's regression. Wall 34 belongs to another player, so a missing
    owner in the paste's undo snapshot would show on undo."""
    window, row = _run_window()
    try:
        index = window.map_view._unit_index
        window._reassign_units([index.entry_for_key((PLAYER, row[34].reference_id))], 2)
        assert any(u is row[34] for u in window.scenario.unit_manager.units[2])
        _select_units(window, [row[35]])
        window.cut_units()
        assert (row[34].rotation, row[36].rotation) == (2.0, 2.0)
        records = len(window.edit_history.records)

        _paste_units_at(window, (35, _DEL_ROW))

        (pasted,) = _walls_on(window, {(35, _DEL_ROW)})
        assert (row[34].rotation, pasted.rotation, row[36].rotation) == (0.0, 0.0, 0.0)
        assert len(window.edit_history.records) == records + 1
        assert window.edit_history.peek_undo().label == "Paste Units"
        assert _last_status(window) == f"Pasted 1 unit at (35, {_DEL_ROW}), reshaping 2 adjacent"
        window.undo()
        assert (row[34].rotation, row[36].rotation) == (2.0, 2.0)
        assert _walls_on(window, {(35, _DEL_ROW)}) == []
        window.redo()
        assert (row[34].rotation, row[36].rotation) == (0.0, 0.0)
    finally:
        _close(window)


def _region_window():
    """The blank template (no walls, so integer-only), Select active: two x
    runs on 30..34 at y 30 and y 33, and a two-wall run at (50..51, 40) copied
    as a region. Returns (window, {tile: wall})."""
    from test_region_select import _select_window

    window = _select_window()
    model = window._ensure_unit_edits()
    tiles = {(x, y): 2.0 if x in (30, 34) else 0.0 for y in (30, 33) for x in range(30, 35)}
    tiles |= {(50, 40): 2.0, (51, 40): 2.0}
    with window._unit_edit(model, "Add", [PLAYER]):
        walls = {t: model.add(PLAYER, WALL, t[0] + 0.5, t[1] + 0.5, rotation=r) for t, r in tiles.items()}
    # Terrain of its own, so a paste writes a tile record too, not only units.
    mm = window.scenario.map_manager
    for x in (50, 51):
        tile = mm.terrain[40 * mm.map_width + x]
        tile.terrain_id = (tile.terrain_id + 1) % 10
    window.on_region_selected((50, 40, 52, 41))
    window.copy_region()
    return window, walls


def test_paste_region_against_a_runs_end_joins_it_in_one_composite_record():
    from descape.edit_history import CompositeDiffRecord

    window, walls = _region_window()
    try:
        assert window.paste_terrain_check.isChecked()
        _paste_units_at(window, (35, 30))

        (near,), (far,) = _walls_on(window, {(35, 30)}), _walls_on(window, {(36, 30)})
        assert (walls[(34, 30)].rotation, near.rotation, far.rotation) == (0.0, 0.0, 2.0)
        assert isinstance(window.edit_history.peek_undo(), CompositeDiffRecord)
        assert _last_status(window).endswith("at (35, 30), reshaping 1 adjacent")
        window.undo()
        assert walls[(34, 30)].rotation == 2.0
        assert _walls_on(window, {(35, 30), (36, 30)}) == []
    finally:
        _close(window)


def test_moving_the_pasted_region_restores_the_first_site_and_joins_the_second():
    window, walls = _region_window()
    try:
        _paste_units_at(window, (35, 30))
        assert walls[(34, 30)].rotation == 0.0

        window.on_region_move(0, 3)

        assert walls[(34, 30)].rotation == 2.0
        assert walls[(34, 33)].rotation == 0.0
        assert [u.rotation for u in _walls_on(window, {(35, 33)})] == [0.0]
        assert _walls_on(window, {(35, 30), (36, 30)}) == []
        assert _last_status(window) == "Moved pasted region to (35, 33), reshaping 1 adjacent"
    finally:
        _close(window)


def _stepped(window) -> None:
    window.iso_action.setChecked(True)
    window.terrain_style_combo.setCurrentText("Stepped")


def test_a_paste_units_that_reshapes_walls_refreshes_wholesale_on_stepped(monkeypatch):
    """Rewritten walls carry no splice and can flip a radian file's verdict."""
    window, row = _run_window()
    try:
        _stepped(window)
        _select_units(window, [row[35]])
        window.cut_units()
        refreshes = _record_refreshes(window, monkeypatch)

        _paste_units_at(window, (35, _DEL_ROW))

        assert row[34].rotation == 0.0
        assert refreshes == [None]
    finally:
        _close(window)


@pytest.mark.parametrize("what", ["lone wall", "villager"])
def test_a_paste_units_that_reshapes_nothing_keeps_its_splices_on_stepped(monkeypatch, what):
    window = _window()
    try:
        _stepped(window)
        if what == "villager":
            unit = _add_unit(window, VILLAGER, 60.5, 60.5)
        else:
            window.on_shape_commit([(60, 60)])
            (unit,) = _walls_on(window, {(60, 60)})
        _select_units(window, [unit])
        assert window.copy_units()
        refreshes = _record_refreshes(window, monkeypatch)

        _paste_units_at(window, (70, 70))

        assert len(refreshes) == 1 and refreshes[0], "expected one spliced refresh"
    finally:
        _close(window)


def test_a_paste_beside_a_radian_neighbour_writes_the_literal_and_the_file_stays_radian():
    """The fixture as loaded is radian (its GAIA wall). (60, 60) stores radian
    index 1 and a lone wall pasted east of it makes it a run end: it takes the
    literal 2."""
    from descape import render

    window = _window()
    try:
        model = window._ensure_unit_edits()
        with window._unit_edit(model, "Add", [PLAYER]):
            neighbour = model.add(PLAYER, WALL, 60.5, 60.5, rotation=1.256637)
            source = model.add(PLAYER, WALL, 80.5, 80.5, rotation=2.0)
        assert render.wall_variant_rotation_overrides(window.scenario)
        _select_units(window, [source])
        assert window.copy_units()

        _paste_units_at(window, (61, 60))

        assert neighbour.rotation == 2.0
        assert render.wall_variant_rotation_overrides(window.scenario), "the GAIA wall keeps the file radian"
    finally:
        _close(window)


# --- the TASK-444 guard itself -------------------------------------------------


def test_the_wall_guard_names_a_patched_table(monkeypatch) -> None:
    stone_only = {WALL: dict(unit_sprites.graphic_map()[WALL])}
    with monkeypatch.context() as m:
        m.setattr(unit_sprites, "graphic_map", lambda: stone_only)
        state = _wall_state()
    assert state is not None
    assert f"unit_sprites.graphic_map is patched: {__name__}.test_the_wall_guard_names_a_patched_table.<locals>.<lambda>" in state
    assert f"{__file__}:" in state, "the patching test's file and line"


def test_the_wall_guard_names_a_changed_entry(monkeypatch) -> None:
    real = unit_sprites.graphic_map()

    def changed():
        table = dict(real)
        table[PALISADE] = {**real[PALISADE], "angle_count": 1}
        return table

    with monkeypatch.context() as m:
        m.setattr(unit_sprites, "graphic_map", changed)
        state = _wall_state()
    assert state is not None
    assert "graphic_map()[72] = {'angle_count': 1," in state and "the JSON has {'angle_count': 5," in state
    assert real[PALISADE]["angle_count"] == 5, "the real cached table was mutated"


def test_the_wall_guard_names_another_dataset() -> None:
    try:
        game_dataset.set_active(game_dataset.ROR)
        state = _wall_state()
    finally:
        game_dataset.set_active(game_dataset.AOE2)
    assert state is not None and "cache_key() = 'ror'" in state
