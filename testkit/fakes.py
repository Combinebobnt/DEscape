"""Synthetic scenario fixtures for tests that build a tile grid by hand.

Duck-types just enough of LoadedScenario (map_manager.map_width/map_height/
terrain/get_tile, unit_manager.units) for the render, pick and filter code
paths, without loading a real file.

Scope note: these cover the four default-tier `tests/` modules only.
`tools/verify_iso_render.py` keeps its own variants on purpose -- its
FakeUnitManager takes no arguments at all ("no units, on purpose", per its
own docstring) and it is a `verify_*.py` scheduled for deletion, whose
`check_*` functions `tests/test_legacy_adapter.py` reaches by name.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape.terrain_palette import PLAYER_COLORS


@dataclass
class SyntheticTile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int
    # Defaulted because most callers place units that no assertion
    # identifies. tests/test_unit_pick.py is the exception -- there
    # reference_id IS the pick identity, so every unit it builds passes one
    # explicitly, and a new test in that file should keep doing so.
    reference_id: int = 1


class FakeMapManager:
    """MapManager's read surface for synthetic grids: the two dimensions,
    the flat row-major terrain list, and get_tile(). Raises KeyError rather
    than returning None off-map -- there is no get_tile_safe() here because
    no caller of the original five copies needed one."""

    def __init__(self, w: int, h: int, tiles: list):
        self.map_width = w
        self.map_height = h
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x: int, y: int):
        return self._by_xy[(x, y)]


class FakeUnitManager:
    def __init__(self, units_by_player: list[list[SyntheticUnit]]):
        self.units = units_by_player


class FakeScenario:
    def __init__(self, w: int, h: int, tiles: list[SyntheticTile], units_by_player):
        self.map_manager = FakeMapManager(w, h, tiles)
        self.unit_manager = FakeUnitManager(units_by_player)
        # Identity default -- no synthetic test here builds a stored color
        # override, so player_id and TEAM_COLORS index coincide, same as
        # PLAYER_COLORS itself.
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))
