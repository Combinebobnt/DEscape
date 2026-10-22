"""GH #230: descape/train_locations.py, the file's location effects and what
they do to a building's panel.

Qt-free. Builds effects on the real triggers fixture.
"""

from __future__ import annotations

from pathlib import Path

from descape import train_locations
from descape.command_buttons import ECONOMIC, MILITARY, TECH, UNIT
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.train_locations import ADD_TRAIN, CHANGE_RESEARCH, CHANGE_TRAIN, LocationEdit

TRIGGERS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_BARRACKS, _STABLE, _BUILDER = 12, 101, 118
_MILITIA, _KNIGHT, _SCOUT, _HOUSE, _CASTLE, _LOOM = 74, 38, 448, 70, 82, 22


def _manager():
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    first = manager.add_trigger("gh230 a")
    first.new_effect.change_train_location(
        object_list_unit_id=_KNIGHT, source_player=1, object_list_unit_id_2=_BARRACKS, button_location=9
    )
    first.new_effect.send_chat(source_player=1, message="not a location effect")
    first.new_effect.add_train_location(
        object_list_unit_id=_SCOUT, source_player=2, object_list_unit_id_2=_BARRACKS, button_location=10
    )
    second = manager.add_trigger("gh230 b")
    second.new_effect.change_technology_location(
        technology=_LOOM, source_player=-1, object_list_unit_id_2=_BARRACKS, button_location=11
    )
    return loaded, manager, first.trigger_id, second.trigger_id


def test_location_edits_lists_the_three_types_in_file_order() -> None:
    _loaded, manager, first, second = _manager()
    edits = [edit for edit in train_locations.location_edits(manager) if edit.trigger_id in (first, second)]
    assert [(e.trigger_id, e.index, e.effect) for e in edits] == [
        (first, 0, CHANGE_TRAIN),
        (first, 2, ADD_TRAIN),
        (second, 0, CHANGE_RESEARCH),
    ]
    knight, _scout, loom = edits
    assert (knight.player, knight.mover, knight.building, knight.value) == (1, _KNIGHT, _BARRACKS, 9)
    assert knight.item == (_KNIGHT, UNIT) and loom.item == (_LOOM, TECH)
    assert knight.label == f'Trigger {first} "gh230 a": Change Train Location'
    assert train_locations.for_player(edits, 1) == (knight, loom), "-1 is every player"
    assert train_locations.location_edits(None) == ()


def test_player_context_leaves_out_the_edited_effect() -> None:
    loaded, manager, first, second = _manager()
    context = train_locations.player_context(manager, loaded, 1, exclude=(first, 0))
    assert [edit.item for edit in context.edits if edit.trigger_id in (first, second)] == [(_LOOM, TECH)]
    assert context.civ is None and context.civ_text == "Random", "the fixture's players are random"
    assert context.keep(1808, UNIT), "unfiltered"
    assert train_locations.player_context(manager, loaded, -1).edits == ()


def _edit(effect: str, mover: int, building: int, value: int, player: int = 1) -> LocationEdit:
    return LocationEdit(0, "t", 0, effect, player, mover, building, value)


def test_apply_moves_adds_and_marks() -> None:
    cells = {1: [(_MILITIA, UNIT)], 6: [(222, TECH)]}
    applied = train_locations.apply(
        cells,
        _BARRACKS,
        None,
        [
            _edit(CHANGE_TRAIN, _MILITIA, _STABLE, 4),
            _edit(ADD_TRAIN, _KNIGHT, _BARRACKS, 6),
            _edit(CHANGE_RESEARCH, _LOOM, _BARRACKS, 0),
        ],
    )
    assert 1 not in applied.cells, "Change took the Militia away"
    assert applied.cells[6] == [(_KNIGHT, UNIT), (222, TECH)], "units first: they cover techs"
    assert [e.mover for e in applied.removed[1]] == [_MILITIA]
    assert [e.mover for e in applied.placed[6]] == [_KNIGHT]
    assert cells == {1: [(_MILITIA, UNIT)], 6: [(222, TECH)]}, "the input is left alone"


def test_apply_skips_what_keep_rejects_and_follows_the_builder_page() -> None:
    applied = train_locations.apply({}, _BARRACKS, None, [_edit(ADD_TRAIN, 1808, _BARRACKS, 7)], lambda _i, _k: False)
    assert applied.cells == {} and applied.placed == {}
    assert [e.mover for e in applied.unavailable[7]] == [1808]
    house = _edit(CHANGE_TRAIN, _HOUSE, _BUILDER, 3)
    castle = _edit(CHANGE_TRAIN, _CASTLE, _BUILDER, 4)
    economic = train_locations.apply({}, _BUILDER, ECONOMIC, [house, castle])
    military = train_locations.apply({}, _BUILDER, MILITARY, [house, castle])
    assert economic.cells == {3: [(_HOUSE, UNIT)]}
    assert military.cells == {4: [(_CASTLE, UNIT)]}
