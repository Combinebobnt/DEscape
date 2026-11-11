"""GH #369: render.created_unit_point(), where a trigger-created unit's ghost
stands. The expected points come from an in-game check on 1.59, not from the
code: a castle (Place Foundation), house, barracks and a (4, 1) gate each
spawned inside a palisade ring round the footprint these points give."""

from __future__ import annotations

import pytest

from descape import render
from descape.terrain_palette import BUILDING_TILE_SPANS, FISH_TILE_SPANS

TILE = (70, 80)
_CASTLE = 82
_HOUSE = 70
_BARRACKS = 12
_WONDER = 276
_GATE = 63
_ARCHER = 4


def _two_by_two_fish() -> int:
    return next(c for c, span in sorted(FISH_TILE_SPANS.items()) if span == (2, 2))


@pytest.mark.parametrize(
    ("const", "span", "point"),
    [
        (_CASTLE, (4, 4), (70.0, 80.0)),
        (_HOUSE, (2, 2), (70.0, 80.0)),
        (_BARRACKS, (3, 3), (70.5, 80.5)),
        (_WONDER, (5, 5), (70.5, 80.5)),
        (_GATE, (4, 1), (70.0, 80.5)),
    ],
)
def test_a_building_stands_where_the_game_spawns_it(const, span, point) -> None:
    assert BUILDING_TILE_SPANS[const] == span
    assert render.created_unit_point(const, *TILE) == point


def test_a_unit_stands_on_the_tile_centre() -> None:
    assert _ARCHER not in BUILDING_TILE_SPANS
    assert render.created_unit_point(_ARCHER, *TILE) == (70.5, 80.5)


def test_a_two_by_two_non_building_keeps_the_tile_centre() -> None:
    fish = _two_by_two_fish()
    assert fish not in BUILDING_TILE_SPANS
    assert render.created_unit_point(fish, *TILE) == (70.5, 80.5)


def test_the_castle_footprint_covers_the_tiles_the_game_uses() -> None:
    x, y = render.created_unit_point(_CASTLE, *TILE)
    assert render._span_start(x, 4) == 68
    assert render._span_start(y, 4) == 78
    # The old tile + 0.5 point drew it one tile high: 69..72.
    assert render._span_start(70.5, 4) == 69


@pytest.mark.parametrize("const", [_CASTLE, _HOUSE, _BARRACKS, _WONDER, _GATE, _ARCHER])
def test_the_location_tile_is_inside_the_footprint(const) -> None:
    span_x, span_y = render.tile_span(const, render.NON_BUILDING_SPAN)
    x, y = render.created_unit_point(const, *TILE)
    assert render._span_start(x, span_x) <= TILE[0] < render._span_start(x, span_x) + span_x
    assert render._span_start(y, span_y) <= TILE[1] < render._span_start(y, span_y) + span_y
