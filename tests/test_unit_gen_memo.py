"""render.anchor_tiles()/wall_variant_rotation_overrides()'s unit_gen memo
(Batch D's D1b) -- both are pure functions of scenario.unit_manager.units,
memoized against scenario.unit_gen so a splice batch's own per-edit cost
doesn't pay for a full re-derivation on every call.

A real LoadedScenario (via UnitEditModel, which is what actually bumps
unit_gen) proves the memo hits and invalidates; a duck-typed scenario with
no unit_gen at all (the shape every other render test module keeps its own
copy of -- test_wall_connectivity.py, test_sprite_chunks.py) proves the
fallback never touches either WeakKeyDictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from descape import render
from descape.scenario_io import load_map_and_units
from descape.unit_model import UnitEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_REF_VILLAGER_P1 = 203


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded)


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


# -- real scenario: hit and invalidation --------------------------------------


def test_anchor_tiles_memo_hit_returns_the_identical_object() -> None:
    loaded, _model = _open()
    first = render.anchor_tiles(loaded)
    second = render.anchor_tiles(loaded)
    assert first is second


def test_anchor_tiles_memo_drops_on_any_mutator() -> None:
    loaded, model = _open()
    first = render.anchor_tiles(loaded)
    model.set_position(_unit(loaded, _REF_VILLAGER_P1), 5.5, 5.5, 0.0)
    second = render.anchor_tiles(loaded)
    assert second is not first
    assert (5, 5) in second


def test_wall_overrides_memo_hit_returns_the_identical_object() -> None:
    loaded, _model = _open()
    first = render.wall_variant_rotation_overrides(loaded)
    second = render.wall_variant_rotation_overrides(loaded)
    assert first is second


def test_wall_overrides_memo_drops_on_any_mutator() -> None:
    loaded, model = _open()
    first = render.wall_variant_rotation_overrides(loaded)
    model.set_position(_unit(loaded, _REF_VILLAGER_P1), 5.5, 5.5, 0.0)
    second = render.wall_variant_rotation_overrides(loaded)
    assert second is not first


# -- duck-typed scenario: never touches the dict -------------------------------


@dataclass
class _Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width = self.map_height = 4
        self.terrain = tiles


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _DuckScenario:
    """No unit_gen attribute at all -- the shape a test fixture that appends
    straight to unit_manager.units produces, and the case both memoized
    functions must always compute fresh for rather than raising."""

    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)


def _duck_scenario():
    tiles = [_Tile(x, y, 0) for y in range(4) for x in range(4)]
    units = [[_Unit(1.5, 1.5, 83)], [], [], [], [], [], [], [], []]
    return _DuckScenario(tiles, units)


def test_a_duck_typed_scenario_computes_fresh_and_never_touches_the_memo() -> None:
    scenario = _duck_scenario()
    before = len(render._ANCHOR_TILES_MEMO)

    first = render.anchor_tiles(scenario)
    second = render.anchor_tiles(scenario)

    assert first == second  # same value
    assert first is not second  # but never cached -- a fresh set each call
    assert len(render._ANCHOR_TILES_MEMO) == before


def test_a_duck_typed_scenario_never_touches_the_wall_overrides_memo() -> None:
    scenario = _duck_scenario()
    before = len(render._WALL_OVERRIDES_MEMO)

    render.wall_variant_rotation_overrides(scenario)
    render.wall_variant_rotation_overrides(scenario)

    assert len(render._WALL_OVERRIDES_MEMO) == before
