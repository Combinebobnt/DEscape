"""Compares tests/fixtures/test_map_240x240.aoe2scenario with
tests/fixtures/test_map_game_resave.aoe2scenario, the same map opened and
saved once in the AoE2:DE 1.59 editor (2026-10-17).

The resave is never regenerated. Every difference below was measured on it
and is listed explicitly, so a generator change that moves one of them fails
here: update the expected drift, or refresh the resave with a new in-game
save. Each rewrite the game makes is a triage item (filed as TASK-402), not
a tolerance.

The per-file tests also run on the resave, through conftest's
`scenario_path` hook (param id `test_map_game_resave`).
"""

from __future__ import annotations

import functools
from collections import Counter
from pathlib import Path

import pytest
from test_trigger_write_path import _object_fields

from descape import disables_fields, player_fields
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.scenario_io import load_map_and_units, parse_triggers

import conftest

pytestmark = pytest.mark.test_map

RESAVE_PATH = conftest.TEST_MAP_RESAVE_PATH
MAP_PATH = conftest.TEST_MAP_PATH

# Dat-only and editor-hidden terrains: the editor stores each as a listed
# terrain plus a layer, whatever layer the generated tile had.
TERRAIN_CONVERSIONS = {
    33: (6, 32), 34: (0, 32), 36: (6, 32), 38: (25, 32), 39: (6, 75),
    43: (14, 25), 44: (0, 76), 61: (75, 60), 62: (77, 60), 103: (6, 78),
}
# Terrain the editor paints under a farm-type building, dropping the layer.
FOUNDATION_TERRAINS = {7, 63}

# (owner, const) placements the editor deletes. Player-owned wild animals
# and birds (but not bears or Wild Camel), relics, some eye candy and blockers.
DROPPED = {
    0: {332},
    1: {
        48, 65, 89, 96, 126, 158, 202, 285, 303, 345, 810, 812, 816, 822, 835, 862, 1019, 1026, 1028, 1029,
        1031, 1056, 1135, 1137, 1139, 1239, 1241, 1247, 1301, 1305, 1608, 1609, 1610, 1776, 1796, 1896, 1924,
        1955, 1956, 1958, 1965, 1973, 2026, 2084, 2086, 2088, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098,
        2099, 2100, 2340, 2423, 2424, 2490, 2537, 2578, 2589, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2615,
        2735, 2736, 2737, 2739, 2742,
    },
    2: {818, 1366},
    3: {818, 1358},
}
# (owner, const) the editor adds (walls, gates and 2422 MOLE CONSTRUCTED); cause open.
ADDED = {(0, 2422), (0, 2444), (1, 63), (1, 64), (1, 789), (1, 2422), (1, 2444)}
# (const, generated rotation, resaved rotation): buildings get 2.0, and three
# const 444 placements change; cause open.
ROTATION_REWRITES = {
    *((const, 0.0, 2.0) for const in (
        72, 79, 117, 119, 155, 190, 231, 234, 235, 236, 370, 684, 685, 785, 788, 1062, 1102, 1665, 2678,
    )),
    (444, 0.0, 1.5707963705062866), (444, 8.0, 0.0), (444, 15.0, 1.9634954929351807),
}
CONST_REWRITES = {(768, 420)}
# A unit on the map's edge line is pulled 0.21 tiles inside it.
EDGE_CLAMP = {0.0: 0.21000000834465027, 239.99000549316406: 239.79000854492188}
# Fields the editor recomputes: z from stacking and elevation.
RECOMPUTED_FIELDS = ("z",)
# initial_animation_frame changes only to equal the rotation (variants and
# scenery), except on these consts. Cliff frames never change.
FRAME_REWRITE_CONSTS = {1869}
# Text the editor replaces with the string table's entry for the slot's id.
MESSAGE_REWRITES = {"hints": "Tent A", "scouts": "Tent B"}


@functools.cache
def _loaded(path: Path):
    return load_map_and_units(path)


def _units(path: Path) -> dict[int, tuple[int, object]]:
    scenario = _loaded(path)._scenario
    return {u.reference_id: (owner, u) for owner, units in enumerate(scenario.unit_manager.units) for u in units}


def test_the_resave_is_the_same_scenario() -> None:
    generated, resaved = _loaded(MAP_PATH), _loaded(RESAVE_PATH)
    assert resaved.scenario_version == generated.scenario_version == "1.59"
    assert resaved._scenario.map_manager.map_size == generated._scenario.map_manager.map_size == 240


def test_terrain_differs_only_by_the_editors_conversions() -> None:
    generated = _loaded(MAP_PATH)._scenario.map_manager.terrain
    resaved = _loaded(RESAVE_PATH)._scenario.map_manager.terrain
    unexplained, seen = [], Counter()
    for index, (a, b) in enumerate(zip(generated, resaved, strict=True)):
        assert a.elevation == b.elevation, f"tile {index}"
        if (a.terrain_id, a.layer) == (b.terrain_id, b.layer):
            continue
        if TERRAIN_CONVERSIONS.get(a.terrain_id) == (b.terrain_id, b.layer):
            seen["conversion"] += 1
        elif b.terrain_id in FOUNDATION_TERRAINS and b.layer == -1:
            seen["foundation"] += 1
        else:
            unexplained.append((index, a.terrain_id, a.layer, b.terrain_id, b.layer))
    assert not unexplained, unexplained[:10]
    assert seen["conversion"] and seen["foundation"], seen


_STALE_RESAVE = pytest.mark.xfail(
    strict=True,
    reason="STALE RESAVE: the generator moved after the 2026-10-17 in-game save (TASK-358 lays Short Marble "
    "2190 runs flush, TASK-396 types create_decision's options as text). Refresh the resave in AoE2:DE.",
)


@_STALE_RESAVE
def test_units_differ_only_by_the_measured_rewrites() -> None:
    generated, resaved = _units(MAP_PATH), _units(RESAVE_PATH)
    dropped: dict[int, set[int]] = {}
    for ref, (owner, unit) in generated.items():
        if ref not in resaved:
            dropped.setdefault(owner, set()).add(unit.unit_const)
    assert dropped == DROPPED
    assert {(owner, u.unit_const) for ref, (owner, u) in resaved.items() if ref not in generated} == ADDED
    unexplained = []
    for ref in generated.keys() & resaved.keys():
        (owner_a, a), (owner_b, b) = generated[ref], resaved[ref]
        if owner_a != owner_b:
            unexplained.append((ref, "owner", owner_a, owner_b))
        if a.unit_const != b.unit_const and (a.unit_const, b.unit_const) not in CONST_REWRITES:
            unexplained.append((ref, "unit_const", a.unit_const, b.unit_const))
        if a.rotation != b.rotation and (a.unit_const, a.rotation, b.rotation) not in ROTATION_REWRITES:
            unexplained.append((ref, "rotation", a.unit_const, a.rotation, b.rotation))
        for axis in ("x", "y"):
            va, vb = getattr(a, axis), getattr(b, axis)
            # A garrisoned unit is moved onto its host.
            if va != vb and EDGE_CLAMP.get(va) != vb and a.garrisoned_in_id == -1:
                unexplained.append((ref, axis, va, vb))
        frame_a, frame_b = a.initial_animation_frame, b.initial_animation_frame
        if frame_a != frame_b and frame_b != b.rotation and a.unit_const not in FRAME_REWRITE_CONSTS:
            unexplained.append((ref, "initial_animation_frame", a.unit_const, frame_a, frame_b))
        unexplained.extend(
            (ref, name, getattr(a, name), getattr(b, name))
            for name in ("status", "garrisoned_in_id", "caption_string_id")
            if getattr(a, name) != getattr(b, name)
        )
    assert not unexplained, unexplained[:10]


@pytest.mark.parametrize("field", RECOMPUTED_FIELDS)
def test_each_recomputed_field_really_changes(field: str) -> None:
    """Keeps RECOMPUTED_FIELDS honest: drop a field once the generator writes
    what the editor would."""
    generated, resaved = _units(MAP_PATH), _units(RESAVE_PATH)
    assert any(getattr(generated[ref][1], field) != getattr(resaved[ref][1], field) for ref in generated.keys() & resaved.keys())


@_STALE_RESAVE
def test_triggers_and_variables_survive_unchanged() -> None:
    generated = parse_triggers(_loaded(MAP_PATH))
    resaved = parse_triggers(_loaded(RESAVE_PATH))
    assert len(generated.triggers) == len(resaved.triggers) == 600
    assert [(v.variable_id, v.name) for v in generated.variables] == [(v.variable_id, v.name) for v in resaved.variables]
    for a, b in zip(generated.triggers, resaved.triggers, strict=True):
        assert _object_fields(a) == _object_fields(b), a.name
        for kind in ("conditions", "effects"):
            assert [_object_fields(e) for e in getattr(a, kind)] == [_object_fields(e) for e in getattr(b, kind)], a.name


def _option_field_ids(model: OptionsEditModel, loaded) -> list[str]:
    ids = [spec.field_id for spec in model.specs]
    ids += list(player_fields.write_targets(loaded))
    ids += list(disables_fields.all_field_ids())
    ids += [f"stance:{row}:{col}" for row in range(1, 9) for col in range(1, 9) if row != col]
    ids += [f"allied_victory:{player}" for player in range(1, 9)]
    return ids


def test_options_players_diplomacy_and_disables_survive_unchanged() -> None:
    generated, resaved = _loaded(MAP_PATH), _loaded(RESAVE_PATH)
    a, b = OptionsEditModel(generated), OptionsEditModel(resaved)
    ids = _option_field_ids(a, generated)
    assert len(ids) > 200
    assert [(i, a.current_value(i)) for i in ids] == [(i, b.current_value(i)) for i in ids]


def test_messages_differ_only_where_a_string_id_wins() -> None:
    a, b = MessagesEditModel(_loaded(MAP_PATH)), MessagesEditModel(_loaded(RESAVE_PATH))
    for field_id in ("instructions", "hints", "victory", "loss", "history", "scouts"):
        expected = MESSAGE_REWRITES.get(field_id, a.current_value(field_id))
        assert b.current_value(field_id) == expected, field_id
        assert b.current_value(f"{field_id}_id") == a.current_value(f"{field_id}_id"), field_id
