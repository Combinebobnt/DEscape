"""Place Unit and Move against descape/placement_rules.py (GH #176/#177,
GH #174), through a real offscreen ViewerWindow.

The rules themselves are tested headless in test_placement_rules.py. This is
the wiring: which paths ask (a Place Unit click, a single drag, a group
drag), what a refusal changes (nothing, and no undo record), what Free
placement and Off-grid each lift, and which paths never ask (a wall run, an
arrow nudge).

Flat throughout, on the units_120x120 fixture. Its terrain is all Grass, so
a test that needs water writes the terrain id straight onto the tile: the
rule reads the live map.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt5.QtCore import QPointF, Qt

from descape import object_catalog, placement_rules

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

VILLAGER = 83
ARCHER = 4
STONE_WALL = 117
GATE = 64  # Stone Gate, 4x1
GATE_DIAGONAL = 659  # Stone Gate, 4x4 bounds over six tiles
WATER_DEEP = 22
PLAYER = 1

# The fixture's own units.
_TREE_OAK = (0, 100)  # GAIA, at (5.5, 5.5)
_HOUSE = (1, 200)  # at (10.5, 10.5), holding _OCCUPANT
_OCCUPANT = (1, 203)  # a Villager stored at (12.5, 10.5), garrisoned in _HOUSE
_ARCHER_P2 = (2, 300)  # at (20.5, 20.5)
_VILLAGER_P2 = (2, 301)

_PLACE_HINT = "Tick Free placement to place it anyway."
_MOVE_HINT = "Tick Free placement to move it anyway."
_GROUP_HINT = "Tick Free placement to move them anyway."


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """Status lines are asserted by name, and AOE2DE_INSTALL_PATH would swap in the install's."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _window(show_garrisoned: bool = False):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show_garrisoned_action.setChecked(show_garrisoned)
    window.mode_combo.setCurrentText("Units")
    assert not window.unrestricted_check.isChecked() and not window.off_grid_check.isChecked()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _arm(window, const: int) -> None:
    """Place Unit with `const` picked for PLAYER."""
    window.units_panel.select_object(const)
    assert window.units_panel.selected_object_const() == const
    assert window.units_panel.select_owner(PLAYER)
    window.place_unit_action.setChecked(True)


def _point(window, x: float, y: float) -> QPointF:
    """The scene position of a continuous Flat map point."""
    tp = window.map_view._tile_pixels
    return QPointF(x * tp, y * tp)


def _click(window, tile: tuple[int, int], modifiers=Qt.NoModifier) -> None:
    window.on_unit_place(_point(window, tile[0] + 0.5, tile[1] + 0.5), modifiers)


def _set_terrain(window, tiles, terrain_id: int) -> None:
    mm = window.scenario.map_manager
    for x, y in tiles:
        mm.terrain[y * mm.map_width + x].terrain_id = terrain_id


def _unit_count(window) -> int:
    return sum(len(units) for units in window.scenario.unit_manager.units)


def _records(window) -> int:
    return len(window.edit_history.records)


def _last(window) -> str:
    return window.status_log.toPlainText().splitlines()[-1]


def _live(window, key: tuple[int, int]):
    return next(u for u in window.scenario.unit_manager.units[key[0]] if u.reference_id == key[1])


def _position(window, key: tuple[int, int]) -> tuple[float, float]:
    unit = _live(window, key)
    return unit.x, unit.y


def _add_units(window, placements) -> list[tuple[int, int]]:
    """[(const, x, y)] added for PLAYER through the model, which never asks the rules."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add", [PLAYER]):
        units = [model.add(PLAYER, const, x, y) for const, x, y in placements]
    return [(PLAYER, unit.reference_id) for unit in units]


def _select(window, keys) -> None:
    window._selection = list(keys)
    window._refresh_selection_view()


def _name(const: int) -> str:
    return object_catalog.display_name(const)


# --- Place Unit -------------------------------------------------------------------


def test_a_click_on_terrain_the_unit_cannot_stand_on_places_nothing() -> None:
    window = _window()
    try:
        _set_terrain(window, [(70, 70)], WATER_DEEP)
        _arm(window, VILLAGER)
        before = _unit_count(window)

        _click(window, (70, 70))

        assert _unit_count(window) == before
        assert _records(window) == 0 and not window.edit_history.can_undo
        assert _last(window) == f"Place Unit: Villager Male can't stand on Water Deep. {_PLACE_HINT}"

        _click(window, (71, 70))
        assert _unit_count(window) == before + 1
        assert _last(window).startswith("Placed Villager Male for Player 1 at (71.5, 70.5)")
    finally:
        _close(window)


def test_a_click_on_an_occupied_point_places_nothing_and_names_the_blocker() -> None:
    window = _window()
    try:
        _arm(window, VILLAGER)
        _click(window, (70, 70))
        before, records = _unit_count(window), _records(window)
        assert records == 1

        _click(window, (70, 70))

        assert _unit_count(window) == before
        assert _records(window) == records
        assert _last(window) == f"Place Unit: blocked by Villager Male (Player 1) at (70.5, 70.5). {_PLACE_HINT}"
    finally:
        _close(window)


def test_free_placement_places_what_the_rules_refuse_and_still_snaps() -> None:
    window = _window()
    try:
        _set_terrain(window, [(70, 70)], WATER_DEEP)
        _arm(window, VILLAGER)
        before = _unit_count(window)
        window.unrestricted_check.setChecked(True)

        _click(window, (70, 70))
        _click(window, (70, 70))

        assert _unit_count(window) == before + 2
        assert _records(window) == 2
        assert _last(window).startswith("Placed Villager Male for Player 1 at (70.5, 70.5)")
        placed = window.scenario.unit_manager.units[PLAYER][-2:]
        assert [(u.x, u.y) for u in placed] == [(70.5, 70.5), (70.5, 70.5)]

        window.unrestricted_check.setChecked(False)
        _click(window, (70, 70))
        assert _unit_count(window) == before + 2, "unticking the box did not bring the rules back"
    finally:
        _close(window)


def test_off_grid_alone_places_beside_a_unit_in_its_tile_and_still_refuses_an_overlap(monkeypatch) -> None:
    """Clearance boxes, not tile occupancy: two villagers half a tile apart
    share tile (70, 70). Off-grid lifts the snap and nothing else."""
    window = _window()
    try:
        _arm(window, VILLAGER)
        window.off_grid_check.setChecked(True)
        target = {}
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: target["point"])
        before = _unit_count(window)

        for point in [(70.25, 70.5), (70.75, 70.5)]:
            target["point"] = point
            _click(window, (70, 70))
        assert _unit_count(window) == before + 2
        placed = window.scenario.unit_manager.units[PLAYER][-2:]
        assert [(u.x, u.y) for u in placed] == [(70.25, 70.5), (70.75, 70.5)]

        target["point"] = (70.625, 70.5)
        _click(window, (70, 70))
        assert _unit_count(window) == before + 2
        assert _last(window) == f"Place Unit: blocked by Villager Male (Player 1) at (70.25, 70.5). {_PLACE_HINT}"
    finally:
        _close(window)


def test_held_alt_is_still_the_off_grid_one_off(monkeypatch) -> None:
    """A snapped click in the tile lands 0.25 from the standing villager and is
    refused; the same click with Alt lands under the cursor, half a tile away."""
    window = _window()
    try:
        _add_units(window, [(VILLAGER, 70.25, 70.5)])
        _arm(window, VILLAGER)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: (70.75, 70.5))
        before = _unit_count(window)

        _click(window, (70, 70))
        assert _unit_count(window) == before
        assert _last(window).startswith("Place Unit: blocked by Villager Male")

        _click(window, (70, 70), Qt.AltModifier)
        assert _unit_count(window) == before + 1
        placed = window.scenario.unit_manager.units[PLAYER][-1]
        assert (placed.x, placed.y) == (70.75, 70.5)
        assert not window.off_grid_check.isChecked()
    finally:
        _close(window)


def test_a_unit_a_filter_hides_still_blocks_and_is_named() -> None:
    window = _window()
    try:
        window.show_trees_action.setChecked(False)
        assert window.map_view._unit_index.entry_for_key(_TREE_OAK) is None, "the tree is still shown"
        _arm(window, VILLAGER)
        before = _unit_count(window)

        _click(window, (5, 5))

        assert _unit_count(window) == before
        assert _last(window) == f"Place Unit: blocked by {_name(349)} (GAIA) at (5.5, 5.5). {_PLACE_HINT}"
    finally:
        _close(window)


def _walls_on(window, tiles) -> list:
    return [
        unit
        for units in window.scenario.unit_manager.units
        for unit in units
        if unit.unit_const == STONE_WALL and (int(unit.x), int(unit.y)) in tiles
    ]


def _garrison_into(window, occupant_key, host) -> None:
    """Makes `host` a unit remove_many() refuses: something is garrisoned in it."""
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Garrison units", [occupant_key[0]]):
        model.set_garrisoned_in(_live(window, occupant_key), host.reference_id)
    assert model.referencing(host)


def test_a_gate_still_replaces_the_walls_under_it_and_sits_over_a_referenced_one() -> None:
    """GH #159 is unchanged by the rules: every wall under the footprint is
    excluded, the one the plan has to keep included."""
    window = _window()
    try:
        _arm(window, STONE_WALL)
        window.on_shape_commit([(x, 50) for x in range(44, 56)])
        footprint = {(x, 50) for x in range(48, 52)}
        (kept,) = _walls_on(window, {(49, 50)})
        _garrison_into(window, _VILLAGER_P2, kept)
        before = _unit_count(window)

        _arm(window, GATE)
        # x = 50.2 rounds to the whole number 50, so the 4-long axis covers tiles 48..51.
        window.on_unit_place(_point(window, 50.2, 50.5), Qt.NoModifier)

        gate = window.scenario.unit_manager.units[PLAYER][-1]
        assert gate.unit_const == GATE and (gate.x, gate.y) == (50.0, 50.5), _last(window)
        assert _walls_on(window, footprint) == [kept]
        assert _unit_count(window) == before - 3 + 1
        assert "replacing 3 walls" in _last(window)
    finally:
        _close(window)


def test_a_diagonal_gate_still_replaces_its_six_walls_beside_one_in_its_empty_corner() -> None:
    window = _window()
    try:
        mm = window.scenario.map_manager
        from descape.render import occupied_tiles_for

        gate_tiles = set(occupied_tiles_for(GATE_DIAGONAL, 60, 60, mm.map_width, mm.map_height))
        assert len(gate_tiles) == 6
        corner = (58, 61)
        assert corner not in gate_tiles
        _arm(window, STONE_WALL)
        window.on_shape_commit([*sorted(gate_tiles), corner])
        (kept,) = _walls_on(window, {(59, 59)})
        _garrison_into(window, _VILLAGER_P2, kept)
        before = _unit_count(window)

        _arm(window, GATE_DIAGONAL)
        window.on_unit_place(_point(window, 60.2, 60.2), Qt.NoModifier)

        gate = window.scenario.unit_manager.units[PLAYER][-1]
        assert gate.unit_const == GATE_DIAGONAL and (gate.x, gate.y) == (60.0, 60.0), _last(window)
        assert _walls_on(window, gate_tiles) == [kept]
        assert len(_walls_on(window, {corner})) == 1
        assert _unit_count(window) == before - 5 + 1
    finally:
        _close(window)


def test_a_gate_is_still_blocked_by_what_is_not_a_wall_under_it() -> None:
    """Only the walls are excluded: a villager standing in the run blocks the gate."""
    window = _window()
    try:
        _arm(window, STONE_WALL)
        window.on_shape_commit([(x, 50) for x in range(44, 56)])
        _add_units(window, [(VILLAGER, 50.5, 50.5)])
        before = _unit_count(window)

        _arm(window, GATE)
        window.on_unit_place(_point(window, 50.2, 50.5), Qt.NoModifier)

        assert _unit_count(window) == before
        assert len(_walls_on(window, {(x, 50) for x in range(48, 52)})) == 4
        assert _last(window) == f"Place Unit: blocked by Villager Male (Player 1) at (50.5, 50.5). {_PLACE_HINT}"
    finally:
        _close(window)


def test_a_wall_run_never_asks_the_rules() -> None:
    """Out of scope by decision: a run crosses water and a standing unit as it did before."""
    window = _window()
    try:
        run = [(x, 70) for x in range(60, 66)]
        _set_terrain(window, run, WATER_DEEP)
        _add_units(window, [(VILLAGER, 63.5, 70.5)])
        assert placement_rules.refusal(window.scenario, STONE_WALL, 63.5, 70.5) is not None
        _arm(window, STONE_WALL)
        before = _unit_count(window)

        window.on_shape_commit(run)

        assert _unit_count(window) == before + len(run)
        assert _last(window).startswith("Placed 6 x Stone Wall")
    finally:
        _close(window)


# --- Move -------------------------------------------------------------------------


def test_a_refused_single_move_leaves_the_unit_and_the_undo_history_alone() -> None:
    window = _window()
    try:
        (key,) = _add_units(window, [(VILLAGER, 70.5, 70.5)])
        _set_terrain(window, [(72, 70)], WATER_DEEP)
        records = _records(window)

        window.on_unit_move(key, _point(window, 72.5, 70.5), Qt.NoModifier)
        assert _position(window, key) == (70.5, 70.5)
        assert _records(window) == records
        assert _last(window) == f"Move: Villager Male can't stand on Water Deep. {_MOVE_HINT}"

        # An archer holds no garrison, so this drop is a plain move, not a garrison.
        window.on_unit_move(key, _point(window, 20.5, 20.5), Qt.NoModifier)
        assert _position(window, key) == (70.5, 70.5)
        assert _records(window) == records
        assert _last(window) == f"Move: blocked by Archer (Player 2) at (20.5, 20.5). {_MOVE_HINT}"

        window.on_unit_move(key, _point(window, 73.5, 70.5), Qt.NoModifier)
        assert _position(window, key) == (73.5, 70.5)
        assert _records(window) == records + 1
    finally:
        _close(window)


def test_free_placement_lets_a_single_move_through() -> None:
    window = _window()
    try:
        (key,) = _add_units(window, [(VILLAGER, 70.5, 70.5)])
        _set_terrain(window, [(72, 70)], WATER_DEEP)
        window.unrestricted_check.setChecked(True)

        window.on_unit_move(key, _point(window, 72.5, 70.5), Qt.NoModifier)
        assert _position(window, key) == (72.5, 70.5)
        window.on_unit_move(key, _point(window, 20.5, 20.5), Qt.NoModifier)
        assert _position(window, key) == (20.5, 20.5)
    finally:
        _close(window)


def test_a_unit_never_blocks_its_own_move(monkeypatch) -> None:
    """An off-grid drop a quarter tile away overlaps where the unit stands now."""
    window = _window()
    try:
        (key,) = _add_units(window, [(VILLAGER, 70.5, 70.5)])
        window.off_grid_check.setChecked(True)
        monkeypatch.setattr(window.map_view, "_pick_map_point", lambda pos: (70.75, 70.5))

        window.on_unit_move(key, _point(window, 70.75, 70.5), Qt.NoModifier)

        assert _position(window, key) == (70.75, 70.5), _last(window)
    finally:
        _close(window)


def test_a_group_move_is_all_or_nothing() -> None:
    window = _window()
    try:
        keys = _add_units(window, [(VILLAGER, 62.5, 62.5), (VILLAGER, 64.5, 62.5)])
        _add_units(window, [(ARCHER, 72.5, 62.5)])
        _set_terrain(window, [(68, 62)], WATER_DEEP)
        _select(window, keys)
        records = _records(window)
        start = [_position(window, key) for key in keys]

        # +4: only the second member's destination, (68.5, 62.5), is water.
        window.on_unit_move(keys[0], _point(window, 66.5, 62.5), Qt.NoModifier)
        assert [_position(window, key) for key in keys] == start
        assert _records(window) == records
        assert _last(window) == (
            f"Move: 1 of 2 units can't go there (first: Villager Male can't stand on Water Deep). {_GROUP_HINT}"
        )

        # +8: the second member lands on the archer.
        window.on_unit_move(keys[0], _point(window, 70.5, 62.5), Qt.NoModifier)
        assert [_position(window, key) for key in keys] == start
        assert _last(window) == (
            f"Move: 1 of 2 units can't go there (first: blocked by Archer (Player 1) at (72.5, 62.5)). {_GROUP_HINT}"
        )

        window.unrestricted_check.setChecked(True)
        window.on_unit_move(keys[0], _point(window, 66.5, 62.5), Qt.NoModifier)
        assert [_position(window, key) for key in keys] == [(66.5, 62.5), (68.5, 62.5)]
        assert _records(window) == records + 1
    finally:
        _close(window)


def test_group_members_never_block_each_other() -> None:
    """+2 puts the first member exactly where the second stands now."""
    window = _window()
    try:
        keys = _add_units(window, [(VILLAGER, 62.5, 62.5), (VILLAGER, 64.5, 62.5)])
        _select(window, keys)

        window.on_unit_move(keys[0], _point(window, 64.5, 62.5), Qt.NoModifier)

        assert [_position(window, key) for key in keys] == [(64.5, 62.5), (66.5, 62.5)], _last(window)
    finally:
        _close(window)


def test_a_host_dragged_with_its_selected_occupant_moves() -> None:
    """With Show Garrisoned on the occupant is a group member. It is exempt:
    its destination here is water with an archer standing on it."""
    window = _window(show_garrisoned=True)
    try:
        assert _live(window, _OCCUPANT).garrisoned_in_id == _HOUSE[1]
        _set_terrain(window, [(42, 40)], WATER_DEEP)
        _add_units(window, [(ARCHER, 42.5, 40.5)])
        _select(window, [_HOUSE, _OCCUPANT])

        window.on_unit_move(_HOUSE, _point(window, 40.5, 40.5), Qt.NoModifier)

        assert _position(window, _HOUSE) == (40.5, 40.5), _last(window)
        assert _position(window, _OCCUPANT) == (42.5, 40.5)
    finally:
        _close(window)


def test_a_lone_occupant_drag_is_not_blocked_by_its_host() -> None:
    """Shift keeps the drop a move instead of a garrison. The destination is
    inside the host's own box, which would block any unit standing on the map."""
    window = _window(show_garrisoned=True)
    try:
        host = _live(window, _HOUSE)
        assert placement_rules.refusal(window.scenario, VILLAGER, host.x, host.y) is not None

        window.on_unit_move(_OCCUPANT, _point(window, host.x, host.y), Qt.ShiftModifier)

        assert _position(window, _OCCUPANT) == (host.x, host.y), _last(window)
    finally:
        _close(window)


def test_an_arrow_nudge_never_asks_the_rules() -> None:
    """Out of scope by decision: a nudge moves onto water and into another unit."""
    window = _window()
    try:
        (key,) = _add_units(window, [(VILLAGER, 70.5, 70.5)])
        _add_units(window, [(ARCHER, 71.5, 70.5)])
        _set_terrain(window, [(71, 70)], WATER_DEEP)
        assert placement_rules.refusal(window.scenario, VILLAGER, 71.5, 70.5) is not None
        _select(window, [key])

        assert window.on_unit_nudge(1, 0, Qt.ShiftModifier) is True

        assert _position(window, key) == (71.5, 70.5)
    finally:
        _close(window)
