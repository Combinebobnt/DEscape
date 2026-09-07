"""Verifies building footprint sizing -- descape.render.unit_tile_bounds.

**Why this file exists separately.** Every footprint assertion already in the
suite is a consistency oracle built on unit_tile_bounds itself
(test_unit_filter.py, test_unit_pick.py): they check that the index, the
renderer and the picker all agree, which they did just as happily while every
even-footprint building was drawn a tile too large in each axis. A green suite
did not catch that bug and would not catch its return. Everything here asserts
against ground truth from outside the code instead -- real spans read off the
game's own unit table, and an invariant swept over inputs rather than derived
from the implementation.

Three separate failure modes are pinned:

- **Size.** The real per-const tile spans. This is the assertion that would
  have caught the original bug.
- **Size must not depend on position.** Off-grid placement is legal in the
  engine and the editor, so a House is 2x2 wherever it sits. A future
  "simplification" that derives the span from the coordinate reintroduces the
  original bug on exactly the ~1% of real placements most likely to expose it.
- **The own-tile invariant.** unit_tile_bounds' docstring states it as a
  precondition for three downstream consumers, so it is swept over arbitrary
  fractional coordinates, not just the two parities buildings happen to use.
  Sweeping parities alone passes while the span-1 anchor is broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import render
from descape.terrain_palette import (
    BUILDING_TILE_OFFSETS,
    BUILDING_TILE_SPANS,
    OBJECT_TILE_SPANS,
)

MAP_W = MAP_H = 64

# unit_const -> (span_x, span_y), read from empires2_x2_p1.dat via
# tools/gen_unit_render_data.py and cross-checked against
# AoE2ScenarioParser's own BuildingInfo/OtherInfo names. Hardcoded on purpose:
# a table derived from BUILDING_TILE_SPANS would just restate the data it is
# supposed to be checking.
REAL_SPANS = {
    70: ("House", (2, 2)),
    68: ("Mill", (2, 2)),
    50: ("Farm", (3, 3)),
    109: ("Town Center", (4, 4)),
    33: ("Fortress", (4, 4)),
    117: ("Stone Wall", (1, 1)),
    64: ("Gate (SW-NE)", (4, 1)),
    276: ("Wonder", (5, 5)),
    263: ("Colosseum", (8, 8)),
}


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int


def bounds(x: float, y: float, unit_const: int) -> tuple[int, int, int, int]:
    result = render.unit_tile_bounds(Unit(x=x, y=y, unit_const=unit_const), MAP_W, MAP_H)
    assert result is not None
    return result


def span_of(x: float, y: float, unit_const: int) -> tuple[int, int]:
    x0, x1, y0, y1 = bounds(x, y, unit_const)
    return x1 - x0, y1 - y0


def is_off_parity(coord: float, span: int) -> bool:
    """An even span "expects" a tile boundary (fractional part 0.0), an odd one
    a tile centre (0.5). Sitting on the other one is legal, not corruption."""
    return round(coord - int(coord), 6) != (0.0 if span % 2 == 0 else 0.5)


# -- size ----------------------------------------------------------------


@pytest.mark.parametrize("unit_const", sorted(REAL_SPANS))
def test_buildings_span_their_real_number_of_tiles(unit_const):
    """The table the original bug got wrong: clearance_size is a HALF-span, so
    every one of these rendered one tile too large per axis while it was
    consumed as a radius about a centre tile."""
    name, expected = REAL_SPANS[unit_const]
    assert BUILDING_TILE_SPANS[unit_const] == expected, name
    # Placed at the parity its own size implies: even spans sit on a tile
    # boundary, odd spans on a tile centre.
    x = 32.0 if expected[0] % 2 == 0 else 32.5
    y = 32.0 if expected[1] % 2 == 0 else 32.5
    assert span_of(x, y, unit_const) == expected, name


def test_a_unit_with_no_footprint_entry_is_a_single_tile():
    unknown = next(uid for uid in range(1, 10_000) if uid not in BUILDING_TILE_SPANS)
    assert span_of(32.5, 32.5, unknown) == (1, 1)


# -- size must not depend on position ------------------------------------


@pytest.mark.parametrize("unit_const", sorted(REAL_SPANS))
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_span_is_the_same_on_either_coordinate_parity(unit_const, offset):
    """Off-grid placement is legal -- the corpus holds 46 of 158 Mills and 32
    of 254 Castles on the parity their size does not 'expect'. Those are valid
    scenarios, so size must be parity-independent.

    This is the test that fails if anyone later derives the span from the
    position (e.g. floor(x - c) .. ceil(x + c) - 1), which reintroduces the
    original bug on exactly that data.
    """
    name, expected = REAL_SPANS[unit_const]
    assert span_of(32.0 + offset, 32.0 + offset, unit_const) == expected, name


def test_an_off_parity_building_shifts_but_does_not_grow():
    """A tile grid cannot represent a half-tile offset, so an off-grid
    building quantizes to the nearer tile. It moves; it does not change size."""
    on_grid = bounds(32.0, 32.0, 70)  # House, span 2
    off_grid = bounds(32.5, 32.5, 70)
    assert on_grid == (31, 33, 31, 33)
    assert off_grid == (32, 34, 32, 34)


# -- the own-tile invariant ----------------------------------------------


@pytest.mark.parametrize("frac", [round(0.05 * i, 2) for i in range(20)])
@pytest.mark.parametrize("span", range(1, 9))
def test_the_anchor_always_keeps_the_units_own_tile_inside_the_span(frac, span):
    """unit_tile_bounds' stated precondition, relied on by _unit_iso_footprint
    (which indexes elevations[py, px]), unit_pick's stepped gate, and
    _unit_screen_bbox_iso (which indexes tile_x1 - 1).

    Swept over arbitrary fractional parts, not just 0.0 and 0.5. Buildings
    only ever sit on those two, but span-1 units are everything else on the
    map -- 15,370 distinct fractional parts across the example corpus -- and a
    parity-only sweep passes while the span-1 anchor is a whole tile off.

    Against the anchor function rather than through unit_tile_bounds, because
    the invariant is a property of the model over every span, and the real
    data has no square building at span 6 or 7 to drive it with.
    """
    coord = 32.0 + frac
    start = render._span_start(coord, span)
    assert start <= int(coord) < start + span, (span, frac, start)


@pytest.mark.parametrize("frac", [0.0, 0.5])
@pytest.mark.parametrize("span", sorted({s[0] for s in BUILDING_TILE_SPANS.values() if s[0] == s[1]}))
def test_the_invariant_holds_end_to_end_for_every_real_square_span(frac, span):
    """The same invariant through the public entry point, for the square spans
    real buildings actually use (1, 2, 3, 4, 5, 8)."""
    unit_const = next(uid for uid, s in BUILDING_TILE_SPANS.items() if s == (span, span))
    x = y = 32.0 + frac
    x0, x1, y0, y1 = bounds(x, y, unit_const)
    assert x0 <= int(x) < x1, (span, frac, (x0, x1))
    assert y0 <= int(y) < y1, (span, frac, (y0, y1))
    assert (x1 - x0, y1 - y0) == (span, span)


def test_asymmetric_spans_take_the_anchor_branch_per_axis():
    """A gate segment is (4, 1): its long axis uses the half-tile model and its
    short axis anchors on int(coord). Applying one rule to both axes is the
    mistake this pins."""
    assert BUILDING_TILE_SPANS[64] == (4, 1)
    x0, x1, y0, y1 = bounds(32.0, 32.9, 64)
    assert (x1 - x0, y1 - y0) == (4, 1)
    assert (y0, y1) == (32, 33)  # int(32.9), not the half-tile anchor's 33
    assert x0 <= 32 < x1


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.75, 0.76, 0.9, 0.99])
def test_a_single_tile_unit_sits_on_exactly_its_own_tile(frac):
    """Span-1 units carry arbitrary float coordinates, so their anchor is
    int(coord) and nothing else. Routing them through the half-tile model used
    for buildings moves every unit with a fractional part above 0.75 -- 5,671
    real corpus values -- which is a visible regression across the GAIA
    clutter that makes up most of a typical file.
    """
    unknown = next(uid for uid in range(1, 10_000) if uid not in BUILDING_TILE_SPANS)
    x = y = 32.0 + frac
    assert bounds(x, y, unknown) == (32, 33, 32, 33)


# -- sparse footprints (Part 2, diagonal gates) --------------------------

# The two shapes tools/gen_unit_render_data.py's building_tiles docs claim,
# hardcoded independently rather than derived from BUILDING_TILE_OFFSETS --
# a table built FROM that dict would just restate what it's checking.
_E_GATE_SHAPE = frozenset({(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)})
_N_GATE_SHAPE = frozenset({(0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (3, 0)})


def test_exactly_36_consts_are_sparse():
    assert len(BUILDING_TILE_OFFSETS) == 36


def test_every_sparse_const_has_a_4x4_bbox():
    for unit_const in BUILDING_TILE_OFFSETS:
        assert BUILDING_TILE_SPANS[unit_const] == (4, 4), unit_const


def test_every_sparse_const_is_one_of_exactly_two_shapes():
    for unit_const, offsets in BUILDING_TILE_OFFSETS.items():
        assert offsets in (_E_GATE_SHAPE, _N_GATE_SHAPE), (unit_const, sorted(offsets))


def test_the_two_shapes_are_each_used_18_times():
    e_count = sum(1 for offs in BUILDING_TILE_OFFSETS.values() if offs == _E_GATE_SHAPE)
    n_count = sum(1 for offs in BUILDING_TILE_OFFSETS.values() if offs == _N_GATE_SHAPE)
    assert (e_count, n_count) == (18, 18)


@pytest.mark.parametrize("unit_const", sorted(BUILDING_TILE_OFFSETS))
def test_the_own_tile_is_always_inside_the_sparse_occupied_set(unit_const):
    # Local bbox offset (2, 2) is the own-tile invariant unit_tile_bounds()
    # documents; the sparse occupied set must keep it too, at all four real
    # coordinate parities a building can legally sit at.
    for frac in (0.0, 0.5):
        x = y = 32.0 + frac
        tiles = render.unit_occupied_tiles(Unit(x=x, y=y, unit_const=unit_const), MAP_W, MAP_H)
        assert (int(x), int(y)) in tiles, (unit_const, frac)


@pytest.mark.parametrize("unit_const", sorted(BUILDING_TILE_OFFSETS))
def test_a_sparse_gate_occupies_exactly_6_of_its_16_bbox_tiles(unit_const):
    x = y = 32.0
    unit = Unit(x=x, y=y, unit_const=unit_const)
    tiles = render.unit_occupied_tiles(unit, MAP_W, MAP_H)
    x0, x1, y0, y1 = bounds(x, y, unit_const)
    assert len(tiles) == 6
    assert set(tiles).issubset({(tx, ty) for tx in range(x0, x1) for ty in range(y0, y1)})


@pytest.mark.parametrize("unit_const", sorted(REAL_SPANS))
def test_non_gate_buildings_stay_fully_occupied(unit_const):
    """The full rect, not a sparse subset -- every REAL_SPANS const (walls,
    House, Farm, Town Center, Wonder, Colosseum, an axis-aligned gate
    segment) is untouched by Part 2's sparsity, including the axis-aligned
    gate whose collision_size also narrows but whose real footprint is
    genuinely its whole strip (see the building_tiles generator docs)."""
    assert unit_const not in BUILDING_TILE_OFFSETS
    name, expected = REAL_SPANS[unit_const]
    x = 32.0 if expected[0] % 2 == 0 else 32.5
    y = 32.0 if expected[1] % 2 == 0 else 32.5
    unit = Unit(x=x, y=y, unit_const=unit_const)
    tiles = render.unit_occupied_tiles(unit, MAP_W, MAP_H)
    x0, x1, y0, y1 = bounds(x, y, unit_const)
    assert set(tiles) == {(tx, ty) for tx in range(x0, x1) for ty in range(y0, y1)}, name


def test_unit_occupies_tile_agrees_with_unit_occupied_tiles():
    for unit_const in sorted(BUILDING_TILE_OFFSETS)[:5]:
        unit = Unit(x=32.0, y=32.0, unit_const=unit_const)
        tiles = set(render.unit_occupied_tiles(unit, MAP_W, MAP_H))
        x0, x1, y0, y1 = bounds(32.0, 32.0, unit_const)
        for tx in range(x0, x1):
            for ty in range(y0, y1):
                assert render.unit_occupies_tile(unit, tx, ty) == ((tx, ty) in tiles)


# -- clamping ------------------------------------------------------------


def test_a_footprint_reaching_off_the_map_is_clamped_not_wrapped():
    x0, x1, y0, y1 = bounds(0.0, 0.0, 263)  # Colosseum, span 8, at the corner
    assert (x0, y0) == (0, 0)
    assert x0 <= 0 < x1 and y0 <= 0 < y1


def test_a_unit_whose_own_tile_is_off_the_map_has_no_bounds():
    assert render.unit_tile_bounds(Unit(x=-1.0, y=5.0, unit_const=70), MAP_W, MAP_H) is None
    assert render.unit_tile_bounds(Unit(x=5.0, y=MAP_H + 1.0, unit_const=70), MAP_W, MAP_H) is None


# -- against real placement data -----------------------------------------


@pytest.mark.corpus
def test_real_buildings_span_their_real_footprints(scenario_path):
    """The synthetic checks above all place units by hand. This one takes the
    placements the game itself wrote, so the fix is proven against real data
    including the legally off-grid buildings that motivated the anchor rule."""
    from descape.scenario_io import load_map_and_units

    scenario = load_map_and_units(str(scenario_path))
    width = scenario.map_manager.map_width
    height = scenario.map_manager.map_height

    seen = 0
    off_parity = 0
    for player_units in scenario.unit_manager.units:
        for unit in player_units:
            expected = REAL_SPANS.get(unit.unit_const)
            if expected is None:
                continue
            name, span = expected
            result = render.unit_tile_bounds(unit, width, height)
            if result is None:
                continue
            x0, x1, y0, y1 = result
            # Only assert the full span where the map hasn't clamped it.
            if 0 < x0 and x1 < width and 0 < y0 and y1 < height:
                assert (x1 - x0, y1 - y0) == span, f"{name} at ({unit.x}, {unit.y})"
                seen += 1
            assert x0 <= int(unit.x) < x1, f"{name} at ({unit.x}, {unit.y})"
            assert y0 <= int(unit.y) < y1, f"{name} at ({unit.x}, {unit.y})"
            off_parity += sum(is_off_parity(axis, s) for axis, s in ((unit.x, span[0]), (unit.y, span[1])))

    if seen == 0:
        pytest.skip(f"{scenario_path.name} has none of the buildings in REAL_SPANS")
    print(f"\n{scenario_path.name}: {seen} known buildings, {off_parity} off-parity axis values")


@pytest.mark.corpus
def test_the_corpus_actually_exercises_off_grid_placement(corpus_files):
    """Guards the guard. The off-grid case is the whole reason size is derived
    from clearance rather than position, and the test above asserts it only if
    the corpus contains such placements at all -- otherwise it passes without
    ever exercising the decision it exists to protect.

    Measured over the full 20-file corpus: 310 of 27,320 building axis values
    are off-parity (1.1%), concentrated in span 2 (46) and span 4 (32).
    """
    from descape.scenario_io import load_map_and_units

    off_parity = 0
    for path in corpus_files:
        scenario = load_map_and_units(str(path))
        for player_units in scenario.unit_manager.units:
            for unit in player_units:
                span = BUILDING_TILE_SPANS.get(unit.unit_const)
                if span is None:
                    continue
                off_parity += sum(is_off_parity(axis, s) for axis, s in ((unit.x, span[0]), (unit.y, span[1])))

    assert off_parity > 0, (
        "no off-parity building placement in this corpus -- "
        "test_real_buildings_span_their_real_footprints is not covering the off-grid case"
    )


# -- cliffs (2026-09-05 cliffs plan, Track A4) ---------------------------

# The four distinct shapes the Default family's nine pieces take, with the
# sub-tile coordinate parity each one's span implies. Hardcoded from the .dat's
# `collision_size` for the same reason REAL_SPANS above is: a table derived
# from OBJECT_TILE_SPANS would restate the data it is meant to check.
#
# Suffixes 04/05, 06/07 and 08/09 pair up -- same span, different face
# direction -- which is why only one of each pair is listed.
REAL_CLIFF_SPANS = {
    264: ("Cliff (Default) 01", (3, 3)),
    265: ("Cliff (Default) 02", (1, 3)),
    266: ("Cliff (Default) 03", (3, 1)),
    267: ("Cliff (Default) 04", (2, 3)),
    269: ("Cliff (Default) 06", (3, 2)),
    271: ("Cliff (Default) 08", (2, 2)),
}


@pytest.mark.parametrize("unit_const", sorted(REAL_CLIFF_SPANS))
def test_cliffs_span_their_real_number_of_tiles(unit_const):
    """Cliffs are not buildings (`unit.building is None`, type 10), so they got
    no BUILDING_TILE_SPANS entry and fell back to (1, 1) -- which made
    _span_start re-snap their already-exact coordinate to int(x) + 0.5."""
    name, expected = REAL_CLIFF_SPANS[unit_const]
    assert unit_const not in BUILDING_TILE_SPANS, f"{name} must stay out of the is-a-building test"
    assert OBJECT_TILE_SPANS[unit_const] == expected, name
    x = 32.0 if expected[0] % 2 == 0 else 32.5
    y = 32.0 if expected[1] % 2 == 0 else 32.5
    assert span_of(x, y, unit_const) == expected, name


@pytest.mark.parametrize(
    ("unit_const", "x", "y", "expected_centre"),
    [
        (264, 103.5, 61.5, (103.5, 61.5)),  # 3x3, already correct before the fix
        (267, 124.0, 98.5, (124.0, 98.5)),  # 2x3, was (124.5, 98.5)
        (269, 100.5, 26.0, (100.5, 26.0)),  # 3x2, was (100.5, 26.5)
        (271, 124.0, 98.0, (124.0, 98.0)),  # 2x2, was (124.5, 98.5)
    ],
)
def test_a_cliffs_footprint_centre_lands_on_its_own_coordinate(unit_const, x, y, expected_centre):
    """The half-tile displacement, in the exact terms the sprite path uses it:
    every cliff SLD frame's hotspot is the (200, 200) centre of its 400x400
    canvas, so the art is authored around the unit's true (x, y) and any
    footprint-centre error moves the sprite one-for-one.

    That is what made a cliff RUN visibly fail to join up: neighbouring nodes
    of different pieces were displaced in different directions.

    The three failing rows are real corpus placements, and each expects the
    coordinate back unchanged -- the fix is that the renderer stops moving it.
    """
    # A 200-tile map, because these are real corpus coordinates and MAP_W is 64.
    unit = Unit(x=x, y=y, unit_const=unit_const)
    result = render.unit_tile_bounds(unit, 200, 200)
    assert result is not None
    x0, x1, y0, y1 = result
    assert (x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2) == expected_centre


@pytest.mark.parametrize("frac", [0.0, 0.5])
def test_a_cliffs_own_tile_stays_inside_its_bounds(frac):
    """unit_tile_bounds' downstream precondition, re-checked for the even spans
    cliffs introduce to non-buildings -- _unit_iso_footprint reads
    elevations[int(y), int(x)] for the whole slab, so an anchor that excluded
    the own tile would index outside it."""
    for unit_const in sorted(REAL_CLIFF_SPANS):
        x0, x1, y0, y1 = bounds(32.0 + frac, 32.0 + frac, unit_const)
        assert x0 <= int(32.0 + frac) < x1
        assert y0 <= int(32.0 + frac) < y1


def test_the_cliff_span_table_and_the_cliff_variant_set_cover_the_same_consts():
    """Both are the .dat's `class_ == 34`, reached two different ways:
    OBJECT_TILE_SPANS is generated from it, unit_sprites._CLIFF_VARIANT_CONSTS
    is hand-written ranges. A const in one and not the other is a real
    disagreement -- and it is exactly how the six-const CLF01..CLF08 partial
    family (1339-1346), which the plan's own ranges missed, gets caught.
    """
    from descape import unit_sprites

    assert set(OBJECT_TILE_SPANS) == set(unit_sprites._CLIFF_VARIANT_CONSTS)
    assert len(OBJECT_TILE_SPANS) == 96
