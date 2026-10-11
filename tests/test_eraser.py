"""descape/eraser.py, headless (GH #184): which units each Eraser target
matches, the cliff veto, and the hit test over unit_pick's index.
The viewer's stroke is tests/test_eraser_viewer.py."""

from __future__ import annotations

from dataclasses import fields

import pytest

from descape import eraser, unit_kind
from descape.eraser import EraseTargets, collect_hits, unit_matches
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter
from descape.unit_pick import build_index
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

VILLAGER = 83
RELIC = 285
GOLD_MINE = 66
PALISADE = 72
TREE = min(TREE_UNIT_IDS)
CLIFF = min(unit_kind.cliff_consts())
INVISIBLE = next(c for c in sorted(unit_kind.invisible_consts()) if c not in unit_kind.building_consts())
BUILDING_4X4 = next(c for c in sorted(unit_kind.building_consts()) if BUILDING_TILE_SPANS.get(c) == (4, 4))

NONE = EraseTargets(units=False, buildings=False)


def _only(name: str) -> EraseTargets:
    return EraseTargets(**{f.name: f.name == name for f in fields(EraseTargets)})


@pytest.mark.parametrize(
    ("target", "player_id", "const"),
    [
        ("units", 1, VILLAGER),
        ("buildings", 1, BUILDING_4X4),
        ("buildings", 1, PALISADE),
        ("trees", GAIA_PLAYER_ID, TREE),
        ("trees", 2, TREE),
        ("invisible", 1, INVISIBLE),
        ("gaia", GAIA_PLAYER_ID, GOLD_MINE),
        ("gaia", GAIA_PLAYER_ID, VILLAGER),
        ("cliffs", GAIA_PLAYER_ID, CLIFF),
    ],
)
def test_each_target_alone_matches_its_own_category_and_nothing_else_does(target, player_id, const) -> None:
    assert unit_matches(_only(target), player_id, const)
    others = [f.name for f in fields(EraseTargets) if f.name != target]
    # A GAIA villager is a Unit and a Gaia object both; every other row has one owner target.
    expected_also = {"units"} if (target == "gaia" and const == VILLAGER) else set()
    assert {name for name in others if unit_matches(_only(name), player_id, const)} == expected_also


def test_no_target_matches_nothing() -> None:
    for player_id, const in [(1, VILLAGER), (GAIA_PLAYER_ID, TREE), (GAIA_PLAYER_ID, GOLD_MINE)]:
        assert not unit_matches(NONE, player_id, const)


def test_a_unit_matching_any_ticked_target_is_erased() -> None:
    targets = EraseTargets(units=True, buildings=False, trees=True)
    assert unit_matches(targets, 1, VILLAGER)
    assert unit_matches(targets, GAIA_PLAYER_ID, TREE)
    assert not unit_matches(targets, 1, BUILDING_4X4)


def test_relics_and_resources_are_not_units() -> None:
    assert not unit_matches(_only("units"), GAIA_PLAYER_ID, RELIC)
    assert not unit_matches(_only("units"), GAIA_PLAYER_ID, GOLD_MINE)
    assert unit_matches(_only("gaia"), GAIA_PLAYER_ID, RELIC)


def test_gaia_objects_exclude_trees() -> None:
    assert not unit_matches(_only("gaia"), GAIA_PLAYER_ID, TREE)
    assert not unit_matches(_only("gaia"), 1, GOLD_MINE)


def test_a_cliff_is_vetoed_unless_cliffs_is_ticked() -> None:
    assert not unit_matches(_only("gaia"), GAIA_PLAYER_ID, CLIFF)
    everything = eraser.everything(EraseTargets())
    assert not everything.cliffs
    assert not unit_matches(everything, GAIA_PLAYER_ID, CLIFF)
    assert unit_matches(eraser.everything(EraseTargets(cliffs=True)), GAIA_PLAYER_ID, CLIFF)


def test_everything_ticks_the_six_gh_184_targets() -> None:
    everything = eraser.everything(NONE)
    assert {f.name for f in fields(EraseTargets) if getattr(everything, f.name)} == {
        "units", "buildings", "trees", "invisible", "gaia", "overlay",
    }


def test_defaults_are_units_and_buildings_only() -> None:
    defaults = EraseTargets()
    assert {f.name for f in fields(EraseTargets) if getattr(defaults, f.name)} == {"units", "buildings"}
    assert not EraseTargets(units=False, buildings=False, overlay=True).any_units
    assert EraseTargets(units=False, buildings=False, cliffs=True).any_units


def test_every_menu_target_is_a_field() -> None:
    assert [name for name, _label in eraser.TARGET_LABELS] == [
        "units", "buildings", "trees", "overlay", "invisible", "gaia", "cliffs",
    ]
    assert {name for name, _label in eraser.TARGET_LABELS} == {f.name for f in fields(EraseTargets)}


# --- collect_hits over a real unit_pick index -------------------------------

MAP = 20


def _scenario(units_by_player) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(MAP) for x in range(MAP)]
    return FakeScenario(MAP, MAP, tiles, units_by_player)


def _players(**by_player):
    units = [[] for _ in range(9)]
    for key, value in by_player.items():
        units[int(key[1:])] = value
    return units


def test_a_multi_tile_building_is_hit_by_a_brush_touching_only_its_corner() -> None:
    house = SyntheticUnit(x=10.0, y=10.0, unit_const=BUILDING_4X4, reference_id=1)
    index = build_index(_scenario(_players(p1=[house])))
    covered = sorted(t for t, orders in index.by_tile.items() if orders)
    assert len(covered) == 16
    corner = covered[-1]
    hits = collect_hits(EraseTargets(), index, [corner], set())
    assert [h.unit for h in hits] == [house]


def test_a_unit_is_hit_once_however_many_of_its_tiles_the_drag_crosses() -> None:
    house = SyntheticUnit(x=10.0, y=10.0, unit_const=BUILDING_4X4, reference_id=1)
    index = build_index(_scenario(_players(p1=[house])))
    done: set[int] = set()
    tiles = list(index.by_tile)
    assert [h.unit for h in collect_hits(EraseTargets(), index, tiles[:2], done)] == [house]
    assert collect_hits(EraseTargets(), index, tiles[2:], done) == []


def test_a_filter_hidden_unit_is_skipped() -> None:
    tree = SyntheticUnit(x=3.5, y=3.5, unit_const=TREE, reference_id=1)
    villager = SyntheticUnit(x=3.5, y=3.5, unit_const=VILLAGER, reference_id=2)
    scenario = _scenario(_players(p0=[tree], p1=[villager]))
    targets = EraseTargets(trees=True)
    shown = collect_hits(targets, build_index(scenario), [(3, 3)], set())
    assert {id(h.unit) for h in shown} == {id(tree), id(villager)}
    hidden = collect_hits(targets, build_index(scenario, UnitFilter(show_trees=False)), [(3, 3)], set())
    assert [h.unit for h in hidden] == [villager]


def test_an_off_map_unit_is_skipped() -> None:
    off_map = SyntheticUnit(x=MAP + 5.5, y=3.5, unit_const=VILLAGER, reference_id=1)
    index = build_index(_scenario(_players(p1=[off_map])))
    every_tile = [(x, y) for y in range(MAP) for x in range(MAP)]
    assert collect_hits(eraser.everything(EraseTargets()), index, every_tile, set()) == []


def test_the_default_targets_leave_a_gaia_resource_on_a_shared_tile() -> None:
    gold = SyntheticUnit(x=5.5, y=5.5, unit_const=GOLD_MINE, reference_id=1)
    villager = SyntheticUnit(x=5.5, y=5.5, unit_const=VILLAGER, reference_id=2)
    index = build_index(_scenario(_players(p0=[gold], p1=[villager])))
    hits = collect_hits(EraseTargets(), index, [(5, 5)], set())
    assert [h.unit for h in hits] == [villager]
