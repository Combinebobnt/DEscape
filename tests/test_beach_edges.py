"""Algorithm-level coverage for descape/beach_edges.py's automatic shoreline.

Uses the same fake-tile/fake-map-manager protocol tests/test_fill_tools.py
does, and for the same reason: beach_edges only ever reads
mm.map_width/map_height/terrain and each tile's terrain_id/layer, so staying
duck-typed keeps these fast and independent of AoE2ScenarioParser. The fakes
are duplicated here rather than imported, matching that module's own
deliberately self-contained style.

Real terrain ids throughout, not arbitrary integers -- the whole point of
this module is that it consults the game's own classification, so a test on
made-up ids would exercise nothing.
"""

from __future__ import annotations

import pytest

from descape.beach_edges import PROTECTED_TERRAIN_IDS, apply_beach_ring, ring_tiles

_GRASS_1 = 0
_BEACH = 2
_BEACH_WET = 107
_BEACH_ICE = 37
_SHALLOWS = 4
_WATER_DEEP = 22
_ICE_NAVIGABLE = 26
_ICE = 35
_DIRT_3 = 3


class FakeTile:
    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


class FakeMapManager:
    def __init__(self, terrain: list[FakeTile], map_width: int, map_height: int):
        self.terrain = terrain
        self.map_width = map_width
        self.map_height = map_height


def _map(width: int, height: int, terrain_id: int = _GRASS_1) -> FakeMapManager:
    return FakeMapManager(
        [FakeTile(terrain_id=terrain_id) for _ in range(width * height)], width, height
    )


def _at(mm: FakeMapManager, x: int, y: int) -> FakeTile:
    return mm.terrain[y * mm.map_width + x]


def _ids(mm: FakeMapManager) -> list[list[int]]:
    return [
        [_at(mm, x, y).terrain_id for x in range(mm.map_width)]
        for y in range(mm.map_height)
    ]


def _paint(mm: FakeMapManager, core, water_id: int = _WATER_DEEP) -> None:
    """The caller's own job in the real flow: the core is applied BEFORE the
    ring, which is what keeps a tile this touch just made water from being
    beached."""
    for x, y in core:
        _at(mm, x, y).terrain_id = water_id


# -- the ring's shape ---------------------------------------------------------


def test_width_1_around_one_tile_is_the_eight_neighbours():
    assert sorted(ring_tiles([(5, 5)], 1, 20, 20)) == sorted(
        [
            (4, 4), (5, 4), (6, 4),
            (4, 5), (6, 5),
            (4, 6), (5, 6), (6, 6),
        ]
    )


def test_width_2_includes_the_far_corner():
    """Chebyshev, not Euclidean: this is the case that distinguishes them,
    and an orthogonal-only ring would leave diagonal shorelines with land
    touching water at a corner."""
    tiles = set(ring_tiles([(5, 5)], 2, 20, 20))
    assert (7, 7) in tiles
    assert (3, 3) in tiles
    assert len(tiles) == 5 * 5 - 1


def test_width_3_is_a_seven_square_minus_the_core():
    assert len(ring_tiles([(5, 5)], 3, 20, 20)) == 7 * 7 - 1


def test_width_0_produces_no_ring_at_all():
    assert ring_tiles([(5, 5)], 0, 20, 20) == []
    assert ring_tiles([(5, 5)], -1, 20, 20) == []


def test_the_core_is_never_in_its_own_ring():
    core = [(4, 4), (5, 4), (4, 5), (5, 5)]
    assert not (set(ring_tiles(core, 2, 20, 20)) & set(core))


def test_the_ring_has_no_duplicates():
    core = [(5, 5), (6, 5), (7, 5)]
    tiles = ring_tiles(core, 2, 20, 20)
    assert len(tiles) == len(set(tiles))


def test_clipping_at_an_edge_never_wraps_a_row():
    """Core at (0, y) on a 7-wide map must not touch (6, y-1) -- the
    flat-index wraparound bug fill_tools guards against."""
    tiles = set(ring_tiles([(0, 3)], 1, 7, 7))
    assert (6, 2) not in tiles
    assert (6, 3) not in tiles
    assert (6, 4) not in tiles
    assert tiles == {(0, 2), (1, 2), (1, 3), (0, 4), (1, 4)}


def test_a_non_square_map_is_fine():
    mm = _map(7, 3)
    _paint(mm, [(3, 1)])
    changed = apply_beach_ring(mm, [(3, 1)], _WATER_DEEP, _BEACH, 1)
    assert changed  # did not raise, and did something
    assert _at(mm, 3, 0).terrain_id == _BEACH
    assert _at(mm, 3, 2).terrain_id == _BEACH


# -- the overwrite policy -----------------------------------------------------


def test_a_plain_land_ring_becomes_beach():
    mm = _map(9, 9)
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 4, 4).terrain_id == _WATER_DEEP
    for x, y in ring_tiles([(4, 4)], 1, 9, 9):
        assert _at(mm, x, y).terrain_id == _BEACH
    # Untouched beyond the ring.
    assert _at(mm, 4, 2).terrain_id == _GRASS_1


def test_existing_water_and_shallows_are_never_overwritten():
    mm = _map(9, 9)
    _at(mm, 3, 4).terrain_id = _SHALLOWS
    _at(mm, 5, 4).terrain_id = _WATER_DEEP
    _at(mm, 3, 3).terrain_id = _ICE_NAVIGABLE
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 3, 4).terrain_id == _SHALLOWS
    assert _at(mm, 5, 4).terrain_id == _WATER_DEEP
    assert _at(mm, 3, 3).terrain_id == _ICE_NAVIGABLE


def test_existing_beach_is_normalized_to_the_current_beach():
    """The settled policy: the ring always reads as one uniform shoreline,
    so an old beach of a different kind is rewritten, not left alone."""
    mm = _map(9, 9)
    _at(mm, 3, 4).terrain_id = _BEACH_WET
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 3, 4).terrain_id == _BEACH


@pytest.mark.parametrize("terrain_id", sorted(PROTECTED_TERRAIN_IDS))
def test_the_non_navigable_beaches_are_carved_out(terrain_id: int):
    """79-82 are family `land` in the .dat, so the overwrite policy would
    convert them -- and that turns a shore ships deliberately cannot dock at
    into one they can, which is a gameplay change rather than a cosmetic
    one."""
    mm = _map(9, 9)
    _at(mm, 3, 4).terrain_id = terrain_id
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 3, 4).terrain_id == terrain_id


def test_layer_is_cleared_on_ring_tiles_too():
    """Roughly 1 in 4 tiles in the real examples/ files carry a genuine
    second terrain via layer; a beach left holding a stale blend renders
    fine here and wrong in-game."""
    mm = _map(9, 9)
    for tile in mm.terrain:
        tile.layer = 7
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    for x, y in ring_tiles([(4, 4)], 1, 9, 9):
        assert _at(mm, x, y).layer == -1
    # Beyond the ring, untouched -- including its layer.
    assert _at(mm, 4, 2).layer == 7


def test_width_0_writes_no_beach_at_all():
    mm = _map(9, 9)
    _paint(mm, [(4, 4)])
    before = _ids(mm)
    assert apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 0) == []
    assert _ids(mm) == before


def test_a_second_identical_call_changes_nothing():
    mm = _map(9, 9)
    _paint(mm, [(4, 4)])
    first = apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert first
    assert apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1) == []


def test_the_returned_indices_are_the_tiles_that_actually_changed():
    mm = _map(9, 9)
    _at(mm, 3, 4).terrain_id = _BEACH  # already the target, so no change
    _paint(mm, [(4, 4)])
    changed = apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert (4 * 9 + 3) not in changed
    assert len(changed) == 7
    assert len(set(changed)) == len(changed)


# -- the Auto rule, per tile --------------------------------------------------


def test_auto_picks_a_different_beach_on_each_side_of_one_call():
    """Keyed per tile, not per stroke: one call spanning ice and grass gets
    BEACH_ICE on the ice side and BEACH on the grass side."""
    mm = _map(9, 9)
    for y in range(9):
        for x in range(4):
            _at(mm, x, y).terrain_id = _ICE
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, None, 1)
    assert _at(mm, 3, 4).terrain_id == _BEACH_ICE  # the ice side
    assert _at(mm, 5, 4).terrain_id == _BEACH  # the grass side


def test_an_explicit_beach_overrides_auto_everywhere():
    mm = _map(9, 9)
    for y in range(9):
        for x in range(4):
            _at(mm, x, y).terrain_id = _ICE
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH_WET, 1)
    assert _at(mm, 3, 4).terrain_id == _BEACH_WET
    assert _at(mm, 5, 4).terrain_id == _BEACH_WET


# -- the drag shape, which is what hazard 1 is about --------------------------


def test_a_straight_drag_leaves_no_beach_inside_the_water():
    """The highest-value case in the plan: applying core-then-ring per touch
    along a drag must leave a clean 1-wide water line inside a 1-wide beach
    border, with no beach stranded in the middle where an earlier touch's
    ring was."""
    mm = _map(12, 12)
    cores = [(3, 5), (4, 5), (5, 5), (6, 5)]
    for tile in cores:
        _paint(mm, [tile])
        apply_beach_ring(mm, [tile], _WATER_DEEP, _BEACH, 1)
    for tile in cores:
        assert _at(mm, *tile).terrain_id == _WATER_DEEP, tile
    # The border above and below is beach all along the run.
    for x, _y in cores:
        assert _at(mm, x, 4).terrain_id == _BEACH
        assert _at(mm, x, 6).terrain_id == _BEACH


def test_a_diagonal_drag_leaves_the_notch_as_beach():
    """An accepted behaviour, pinned so it stays a decision: (6, 5) sits in
    the notch between two diagonally-adjacent water tiles and is correctly
    beach, not water."""
    mm = _map(12, 12)
    for tile in [(5, 5), (6, 6)]:
        _paint(mm, [tile])
        apply_beach_ring(mm, [tile], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 5, 5).terrain_id == _WATER_DEEP
    assert _at(mm, 6, 6).terrain_id == _WATER_DEEP
    assert _at(mm, 6, 5).terrain_id == _BEACH


def test_a_multi_tile_brush_core_beaches_only_its_outside():
    mm = _map(12, 12)
    core = [(x, y) for y in range(4, 7) for x in range(4, 7)]
    _paint(mm, core)
    apply_beach_ring(mm, core, _WATER_DEEP, _BEACH, 1)
    for tile in core:
        assert _at(mm, *tile).terrain_id == _WATER_DEEP, tile
    assert _at(mm, 5, 3).terrain_id == _BEACH
    assert _at(mm, 3, 3).terrain_id == _BEACH
    assert _at(mm, 5, 2).terrain_id == _GRASS_1


def test_a_terrain_that_is_neither_water_nor_beach_still_gets_converted():
    mm = _map(9, 9, _DIRT_3)
    _paint(mm, [(4, 4)])
    apply_beach_ring(mm, [(4, 4)], _WATER_DEEP, _BEACH, 1)
    assert _at(mm, 3, 4).terrain_id == _BEACH
