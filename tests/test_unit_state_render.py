"""A unit whose Unit State is not Alive paints its coloured mark mixed towards
grey (unit_state.ghost_mark()), in Flat, Stepped and Sloped alike.

Every mark painter reads render_units._unit_color(), so the check runs each
style's chunk cache over a synthetic map (no install, sprites off) and counts
footprint-coloured pixels. An invisible object's box is an editor marker and
keeps its colour; an all-Alive map takes no new branch.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_unit_filter import _BUILDING_CONST, MAP_H, MAP_W, _flat_cache, _iso_cache, _sloped_cache, _whole_canvas

from descape import render_units, unit_kind, unit_state
from descape.terrain_palette import PLAYER_COLORS
from descape.unit_filter import UnitFilter
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit

_PLAYER = 1
_MAKERS = {"flat": _flat_cache, "stepped": _iso_cache, "sloped": _sloped_cache}


def _scenario(status: int, unit_const: int = _BUILDING_CONST) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(MAP_H) for x in range(MAP_W)]
    units = [[] for _ in range(9)]
    unit = SyntheticUnit(x=8.0, y=8.0, unit_const=unit_const)
    unit.status = status
    units[_PLAYER] = [unit]
    return FakeScenario(MAP_W, MAP_H, tiles, units)


def _count(canvas: np.ndarray, color) -> int:
    return int(np.count_nonzero(np.all(canvas == np.array(color, dtype=np.uint8), axis=2)))


def _player_color():
    return PLAYER_COLORS[_PLAYER % len(PLAYER_COLORS)]


@pytest.mark.parametrize("style", list(_MAKERS))
def test_a_ghosted_mark_paints_the_mixed_colour_and_never_its_own(style: str) -> None:
    own = _player_color()
    mixed = unit_state.ghost_mark(own)
    alive = _whole_canvas(_MAKERS[style](_scenario(unit_state.ALIVE), UnitFilter()))
    ghost = _whole_canvas(_MAKERS[style](_scenario(7), UnitFilter()))
    painted = _count(alive, own)
    assert painted > 0, "the alive mark painted nothing"
    assert _count(alive, mixed) == 0
    assert _count(ghost, own) == 0
    assert _count(ghost, mixed) == painted, "the ghost must paint the same footprint, in the mixed colour"


@pytest.mark.parametrize("style", list(_MAKERS))
def test_an_unlabelled_state_paints_as_alive(style: str) -> None:
    alive = _whole_canvas(_MAKERS[style](_scenario(unit_state.ALIVE), UnitFilter()))
    raw = _whole_canvas(_MAKERS[style](_scenario(42), UnitFilter()))
    assert np.array_equal(alive, raw)


def test_an_invisible_objects_box_keeps_its_colour() -> None:
    marker = min(unit_kind.invisible_consts())
    alive, ghost = _scenario(unit_state.ALIVE, marker), _scenario(0, marker)
    color = render_units._unit_color(alive.unit_manager.units[_PLAYER][0], _player_color())
    assert render_units._unit_color(ghost.unit_manager.units[_PLAYER][0], _player_color()) == color
    for make in _MAKERS.values():
        assert np.array_equal(_whole_canvas(make(alive, UnitFilter())), _whole_canvas(make(ghost, UnitFilter())))


def test_the_kind_colour_ignores_the_state() -> None:
    unit = _scenario(7).unit_manager.units[_PLAYER][0]
    assert render_units.kind_mark_color(unit.unit_const, _player_color()) == _player_color()
    assert render_units.unit_mark_color(unit, _player_color()) == unit_state.ghost_mark(_player_color())
