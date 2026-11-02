"""Map mirroring Stage 2: mirror_tools.plan_mirror_units() -- no Qt, no
AoE2ScenarioParser. Units are duck-typed here the same way the map manager is
in tests/test_mirror_tools.py.

The claims under test: an image's footprint is the reflection of its source's,
orbit dedup keeps one image per distinct position, a wall's shape index is
permuted by wall_index_image() (axis runs under an axis-swapping element,
diagonal runs under r, r3, mx, my) and only in an integer-encoded
file, an ANGLE const's facing maps through facing_image() (derived from
TRANSFORMS' linear part), ownership rotates along the scenario's own player list, and the two
refusal sweeps (axis straddle, garrisoned destination unit) fire, the straddle one only under
the default overlap policy (TASK-250: skip and place go past it, never past a garrison).
"""

from __future__ import annotations

import math
import struct

import pytest
from test_wall_connectivity import WALL_CONST, real_gate_wall_install  # noqa: F401 (a pytest fixture, imported for its name)

from descape import render
from descape.mirror_tools import (
    AXIS_SWAPPING,
    MODE_BY_ID,
    OVERLAP_POLICIES,
    OVERLAPS_PLACE,
    OVERLAPS_REFUSE,
    OVERLAPS_SKIP,
    POSITION_TRANSFORMS,
    TRANSFORMS,
    AngularElement,
    _image_position,
    _linear_part,
    compose,
    doubled,
    facing_image,
    gate_orientation_map,
    invert,
    plan_mirror,
    plan_mirror_units,
    reorient_gate_const,
    swaps_axes,
    unit_image,
    wall_index_image,
    walls_are_radian,
)
from descape.terrain_palette import tile_span

import conftest

N = 64
_RADIAN_3 = 3 * 2 * math.pi / 5  # a radian-encoded wall index, so the file reads as radian
OAK = 349  # 1x1, variant-indexed
WALL = 117  # 1x1, angle_count 5, rotation is a run-direction index
HOUSE = 70  # 2x2
CASTLE = 82  # 4x4
GATE_NE = 64  # (4, 1) -- a non-square span
ARCHER = 4  # 1x1, rotation is a real angle


class FakeTile:
    def __init__(self):
        self.terrain_id = 0
        self.elevation = 0
        self.layer = -1


class FakeMapManager:
    def __init__(self, n: int = N):
        self.terrain = [FakeTile() for _ in range(n * n)]
        self.map_width = n
        self.map_height = n


class FakeUnit:
    def __init__(self, unit_const: int, x: float, y: float, player: int = 1, rotation: float = 0.0):
        self.unit_const = unit_const
        self.x = x
        self.y = y
        self.z = 0.0
        self.player = player
        self.rotation = rotation
        self.status = 2
        self.initial_animation_frame = 0
        self.caption_string_id = -1
        self.caption_string = ""


def _by_player(units):
    lists = [[] for _ in range(9)]
    for unit in units:
        lists[unit.player].append(unit)
    return lists


def _plan(units, mode_id=1, slice_index=0, **kwargs):
    mm = FakeMapManager()
    tiles = plan_mirror(mm, mode_id, slice_index, False, False)
    return plan_mirror_units(
        mm, mode_id, slice_index, _by_player(units), tiles.source_indices, **kwargs
    )


def _source_tiles(mode_id: int, slice_index: int = 0, margin: int = 0) -> list[tuple[int, int]]:
    """Tiles of this mode's source slice, away from the map edge and from the
    symmetry axes -- a source unit has to sit in the slice it is read from,
    and which quadrant/half that is differs per mode."""
    plan = plan_mirror(FakeMapManager(), mode_id, slice_index, False, False)
    inside = plan.source_indices
    return [
        (x, y)
        for x, y in ((idx % N, idx // N) for idx in sorted(inside))
        # Surrounded by source tiles out to `margin`, so a multi-tile
        # building sits wholly inside the slice and its image cannot overlap
        # the original.
        if all(
            6 <= x + dx < N - 6 and 6 <= y + dy < N - 6 and (y + dy) * N + (x + dx) in inside
            for dx in range(-margin, margin + 1)
            for dy in range(-margin, margin + 1)
        )
    ]


def _at(const: int, mode_id: int, nth: int = 0, **kwargs) -> FakeUnit:
    """A unit centred on the nth usable source tile of `mode_id`'s slice, with
    room for its own footprint."""
    span_x, span_y = tile_span(const, render.NON_BUILDING_SPAN)
    x, y = _source_tiles(mode_id, margin=max(span_x, span_y))[nth]
    return FakeUnit(const, *render.span_anchor(x, y, span_x, span_y), **kwargs)


def test_compose_and_invert_agree_with_the_transforms():
    for name, fn in TRANSFORMS.items():
        assert TRANSFORMS[invert(name)](*fn(3, 7, N), N) == (3, 7)
    assert compose("d", "a") == "r2"
    assert compose("id", "r") == "r"


@pytest.mark.parametrize("const", [OAK, WALL, HOUSE, CASTLE, ARCHER])
@pytest.mark.parametrize("mode_id", [1, 2, 3, 5, 6, 9])
def test_an_image_occupies_the_reflection_of_its_sources_tiles(const, mode_id):
    """The footprint parity rule, run through the real TRANSFORMS rather than
    a table re-typed here. Square spans only: a non-square span has no const
    to carry its swapped shape, which is what unsquare_spans reports."""
    source = _at(const, mode_id)
    plan = _plan([source], mode_id=mode_id)
    src_tiles = set(render.unit_occupied_tiles(source, N, N))
    assert plan.images, "nothing mirrored, so this proves nothing"
    for image in plan.images:
        tiles = set(render.occupied_tiles_for(image.unit_const, image.x, image.y, N, N))
        assert tiles == {TRANSFORMS[image.element](tx, ty, N) for tx, ty in src_tiles}


def test_a_unit_on_a_tile_boundary_still_lands_in_the_reflected_tile():
    """2.5% of 1x1 corpus placements sit on an exact integer coordinate, whose
    continuous reflection lands on the far boundary -- i.e. in the next tile
    unless it is snapped back."""
    source = FakeUnit(OAK, 10.0, 12.0)
    (image,) = _plan([source], mode_id=3).images  # mx: x -> n-1-x
    assert (int(image.x), int(image.y)) == TRANSFORMS["mx"](10, 12, N)


def test_a_unit_on_the_axis_produces_no_duplicate_image():
    """Its orbit is degenerate: the mirror maps it onto itself."""
    on_axis = FakeUnit(OAK, N / 2, N / 2)  # fixed by mode 1's 'a'
    assert _plan([on_axis], mode_id=1).images == []


def test_units_outside_the_source_slice_are_removed_and_recreated():
    source = _at(OAK, 1)
    stale = FakeUnit(OAK, *TRANSFORMS["a"](int(source.x), int(source.y), N))
    stale.x, stale.y = stale.x + 0.5, stale.y + 0.5
    plan = _plan([source, stale])
    assert plan.removals == [stale]
    assert [(i.x, i.y) for i in plan.images] == [(N - source.y, N - source.x)]


def test_a_gate_image_carries_the_reoriented_sibling_and_its_own_anchor():
    """Stage 2b: an ne gate (4, 1) mirrored under `d` becomes its se sibling
    (1, 4), re-anchored so it occupies the reflection of its source's tiles."""
    from descape.mirror_tools import reorient_gate_const

    x, y = _source_tiles(2, margin=4)[0]
    gate = FakeUnit(GATE_NE, *render.span_anchor(x, y, 4, 1))
    plan = _plan([gate], mode_id=2)
    assert plan.gates == [gate]
    assert plan.unsquare_spans == []  # a gate's four orientations ARE four consts
    (image,) = plan.images
    assert image.element == "d"
    assert image.unit_const == reorient_gate_const(GATE_NE, "d") != GATE_NE
    src_tiles = set(render.unit_occupied_tiles(gate, N, N))
    tiles = set(render.occupied_tiles_for(image.unit_const, image.x, image.y, N, N))
    assert tiles == {TRANSFORMS["d"](tx, ty, N) for tx, ty in src_tiles}


def test_a_gate_fixed_in_place_still_reorients_when_its_shape_changes():
    """A gate whose anchor is unmoved by the element is still a different
    const if its run direction turned -- the position dedup must not swallow
    it."""
    from descape.mirror_tools import reorient_gate_const

    x, y = _source_tiles(3, margin=4)[0]
    gate = FakeUnit(GATE_NE, *render.span_anchor(x, y, 4, 1))
    plan = _plan([gate], mode_id=3)
    assert [image.unit_const for image in plan.images] == [reorient_gate_const(GATE_NE, "mx")]


def test_a_straddling_building_is_refused_not_repaired():
    """A castle anchored off-axis whose image overlaps the original: orbit
    dedup cannot catch it (the two anchors genuinely differ)."""
    castle = FakeUnit(CASTLE, N / 2 - 1.0, N / 2 - 1.0)
    plan = _plan([castle], mode_id=1)
    assert plan.straddling == [castle]
    assert plan.blocked
    assert plan.images == []


GRASS = 1358  # "Grass, Green": passable eye candy


def test_passable_eye_candy_on_a_diagonal_axis_tile_is_mirrored_not_refused():
    """TASK-042: grass at a sub-tile offset on mode 1's axis row maps into its
    own tile. It has no collision, so its copy is placed; an oak there still refuses."""
    from descape import unit_kind

    assert GRASS in unit_kind.passable_eye_candy_consts() and OAK not in unit_kind.passable_eye_candy_consts()
    x, y = 20.3, N - 20 - 1 + 0.6  # tile (20, 43): x + y == n - 1
    grass = FakeUnit(GRASS, x, y, player=0)
    plan = _plan([grass], mode_id=1)
    assert not plan.blocked and plan.straddling == [] and plan.overlapping == []
    (image,) = plan.images
    assert (int(image.x), int(image.y)) == (20, 43) and (image.x, image.y) != (x, y)
    oak = _plan([FakeUnit(OAK, x, y, player=0)], mode_id=1)
    assert oak.blocked and oak.images == []


@pytest.mark.parametrize("mode_id", [1, 2])
def test_mirroring_on_axis_passable_eye_candy_again_adds_nothing(mode_id):
    """TASK-042: the copy lands in the source slice, so a second mirror sees
    original and copy as sources whose copies land on each other. Positions
    go through float32, as a save stores them."""
    import numpy as np

    x = 20.3
    y = (N - 20 - 1 + 0.6) if mode_id == 1 else 20.6  # tile (20, 43) on mode 1's axis, (20, 20) on mode 2's
    units = [FakeUnit(GRASS, x, y, player=0)]
    first = _plan(units, mode_id=mode_id)
    assert len(first.images) == 1 and not first.blocked
    units += [FakeUnit(i.unit_const, float(np.float32(i.x)), float(np.float32(i.y)), player=i.player) for i in first.images]
    second = _plan(units, mode_id=mode_id)
    assert second.images == [] and not second.blocked


def _mode_5_spiller() -> FakeUnit:
    """A castle in mode 5's source slice whose footprint spills into the
    destination: its r2 copy lands on source tile (21, 41), never on itself."""
    return FakeUnit(CASTLE, 40.5, 20.5)


@pytest.mark.parametrize("const", [GRASS, OAK])
def test_passable_eye_candy_is_neither_landed_on_nor_lands_on_a_building(const):
    """TASK-042, both directions: the castle's copy lands on the 1x1, and the
    1x1's copy lands inside the castle's spilled footprint. Only a collidable
    1x1 (the oak) makes either count."""
    castle = _mode_5_spiller()
    small = FakeUnit(const, 21.3, 41.6, player=0)
    plan = _plan([castle, small], mode_id=5)
    image_tiles = render.occupied_tiles_for(CASTLE, *_image_position(CASTLE, castle.x, castle.y, N, "r2"), N, N)
    assert (21, 41) in image_tiles
    assert TRANSFORMS["r2"](21, 41, N) in render.unit_occupied_tiles(castle, N, N)
    if const == GRASS:
        assert not plan.blocked and plan.straddling == []
        assert sorted(image.source is castle for image in plan.images) == [False, True]
    else:
        assert plan.blocked and plan.straddling == [small, castle]  # GAIA's list comes first


def test_a_tree_landing_on_a_spilling_building_blames_the_building():
    """TASK-043: the oak's copy lands on the castle's one tile past mode 5's
    boundary (and the castle's copy on the oak). Both straddle, but the
    refusal names the castle and that tile, never the oak."""
    castle = _mode_5_spiller()
    oak = FakeUnit(OAK, 21.3, 41.6, player=0)
    plan = _plan([castle, oak], mode_id=5)
    source = plan_mirror(FakeMapManager(), 5, 0, False, False).source_indices
    spilled = [(x, y) for x, y in render.unit_occupied_tiles(castle, N, N) if y * N + x not in source]
    assert spilled == [(42, 22)] == [TRANSFORMS["r2"](21, 41, N)]
    assert plan.blocked and plan.straddling == [oak, castle]
    assert plan.overlap_blockers == [(castle, (42, 22))]


def test_a_unit_straddling_a_diagonal_axis_names_itself():
    """Modes 1/2: the axis castle's own copy lands on it, so it is the one in the way."""
    castle = _axis_castle()
    plan = _plan([castle], mode_id=1)
    ((blocker, tile),) = plan.overlap_blockers
    source = plan_mirror(FakeMapManager(), 1, 0, False, False).source_indices
    assert blocker is castle
    assert tile in render.unit_occupied_tiles(castle, N, N) and tile[1] * N + tile[0] not in source
    oak = FakeUnit(OAK, 20.3, N - 20 - 1 + 0.6)  # on the axis row, inside the source: nothing spills
    assert _plan([oak], mode_id=1).overlap_blockers == [(oak, (20, 43))]


BLOCKER_1X3 = 2423  # (1, 3) since GH #121
BLOCKER_3X1 = 2424


def test_a_blocker_keeps_its_orientation_under_an_axis_swap_and_is_reported():
    """GH #121's named side effect: a blocker now takes the span > 1 path. It
    has no orientation sibling, so the image keeps 1x3, re-anchored from the
    reflected footprint's min corner, and unsquare_spans says so."""
    source = _at(BLOCKER_1X3, 1)
    plan = _plan([source], mode_id=1)
    assert plan.unsquare_spans == [source]
    (image,) = plan.images
    assert swaps_axes(image.element)
    assert image.unit_const == BLOCKER_1X3
    reflected = [TRANSFORMS[image.element](tx, ty, N) for tx, ty in render.unit_occupied_tiles(source, N, N)]
    low = (min(t[0] for t in reflected), min(t[1] for t in reflected))
    tiles = render.occupied_tiles_for(image.unit_const, image.x, image.y, N, N)
    assert set(tiles) == {(low[0], low[1] + k) for k in range(3)}


def test_a_blocker_under_a_non_swapping_element_occupies_its_reflection():
    source = _at(BLOCKER_3X1, 3)
    plan = _plan([source], mode_id=3)
    assert plan.unsquare_spans == []
    (image,) = plan.images
    src = render.unit_occupied_tiles(source, N, N)
    assert len(src) == 3
    tiles = set(render.occupied_tiles_for(image.unit_const, image.x, image.y, N, N))
    assert tiles == {TRANSFORMS[image.element](tx, ty, N) for tx, ty in src}


def test_a_blocker_straddling_the_axis_is_refused_like_a_building():
    """A 3x1 centred one tile inside mode 3's axis reaches across it, so its
    image overlaps the original."""
    blocker = FakeUnit(BLOCKER_3X1, N / 2 - 0.5, 20.5)
    plan = _plan([blocker], mode_id=3)
    assert plan.straddling == [blocker]
    assert plan.blocked
    assert plan.images == []


def test_a_garrisoned_destination_unit_blocks_the_plan():
    source = _at(OAK, 1, 0)
    holder = FakeUnit(HOUSE, N - source.x - 0.5, N - source.y - 0.5)
    rider = FakeUnit(ARCHER, N - source.y, N - source.x)
    plan = _plan([source, holder, rider], referencing=lambda u: [rider] if u is holder else [])
    assert plan.garrisoned_blockers == []  # the rider is itself being removed
    plan = _plan([source, holder], referencing=lambda u: [source] if u is holder else [])
    assert plan.garrisoned_blockers == [holder]
    assert plan.blocked


# --- TASK-250: what an overlapping copy does (refuse / skip / place) ----------


def _axis_castle() -> FakeUnit:
    """test_a_straddling_building_is_refused_not_repaired's castle: its `a` image overlaps it."""
    return FakeUnit(CASTLE, N / 2 - 1.0, N / 2 - 1.0)


def _mode_8_straddler() -> FakeUnit:
    """A castle reaching one column across mode 8's `mx` axis and clear of
    `my`'s: its mx image overlaps it, its my and r2 images do not."""
    return FakeUnit(CASTLE, *render.span_anchor(N // 2 - 1, 40, 4, 4))


def test_skip_applies_the_plan_without_the_straddlers_copies():
    castle = _axis_castle()
    plan = _plan([castle], mode_id=1, overlaps=OVERLAPS_SKIP)
    assert not plan.blocked
    assert plan.images == []
    assert plan.straddling == [castle]
    (image,) = plan.overlapping
    assert image.source is castle and image.element == "a"
    assert (image.x, image.y) == _image_position(CASTLE, castle.x, castle.y, N, "a")


def test_skip_keeps_a_straddlers_non_overlapping_copies():
    castle = _mode_8_straddler()
    plan = _plan([castle], mode_id=8, overlaps=OVERLAPS_SKIP)
    assert not plan.blocked
    assert plan.straddling == [castle]
    assert [image.element for image in plan.overlapping] == ["mx"]
    assert sorted(image.element for image in plan.images) == ["my", "r2"]
    src_tiles = set(render.unit_occupied_tiles(castle, N, N))
    for image in plan.images:
        assert not src_tiles & set(render.occupied_tiles_for(image.unit_const, image.x, image.y, N, N))


def test_refuse_is_the_default_and_unchanged():
    """No `overlaps=` is OVERLAPS_REFUSE: blocked, the same images, and
    `overlapping` filled under refuse too. Skip shares refuse's images, and
    place is those plus the overlapping copies."""
    castle = _mode_8_straddler()
    default = _plan([castle], mode_id=8)
    refuse = _plan([castle], mode_id=8, overlaps=OVERLAPS_REFUSE)
    skip = _plan([castle], mode_id=8, overlaps=OVERLAPS_SKIP)
    place = _plan([castle], mode_id=8, overlaps=OVERLAPS_PLACE)
    assert default.overlaps == OVERLAPS_REFUSE
    assert default.blocked and refuse.blocked and default.overlap_blocked
    assert default.images == refuse.images == skip.images
    assert default.overlapping == refuse.overlapping == skip.overlapping != []
    assert place.images == skip.images + skip.overlapping


@pytest.mark.parametrize("overlaps", OVERLAP_POLICIES)
def test_a_garrison_block_ignores_the_overlap_policy(overlaps):
    source = _at(OAK, 1, 0)
    holder = FakeUnit(HOUSE, N - source.x - 0.5, N - source.y - 0.5)
    castle = _axis_castle()
    plan = _plan([source, holder, castle], referencing=lambda u: [source] if u is holder else [], overlaps=overlaps)
    assert plan.garrisoned_blockers == [holder]
    assert plan.straddling == [castle]
    assert plan.blocked


def test_an_unknown_overlap_policy_raises():
    with pytest.raises(ValueError, match="overlap policy"):
        _plan([_axis_castle()], mode_id=1, overlaps="ignore")


def test_place_keeps_the_overlapping_copy():
    """Under place the withheld copy is applied, at the end of `images`
    (extended after the emit loop). TASK-427 settled on no image post-pass,
    so there is nothing else for the copy to be ordered against."""
    oak = _at(OAK, 1)
    castle = _axis_castle()
    plan = _plan([oak, castle], mode_id=1, overlaps=OVERLAPS_PLACE)
    assert not plan.blocked
    assert plan.straddling == [castle]
    (overlap,) = plan.overlapping
    assert overlap.source is castle
    assert [image.source for image in plan.images] == [oak, castle]
    assert plan.images[-1] is overlap


@pytest.mark.parametrize("element,expected", [("a", 1.0), ("d", 1.0), ("mx", 0.0), ("r2", 0.0)])
def test_a_walls_run_direction_swaps_only_under_an_axis_swapping_element(element, expected):
    """0 is a run along +-x, 1 along +-y. mx/my/r2 preserve the axes, so the
    stored index passes through."""
    mode_id = {"a": 1, "d": 2, "mx": 3, "r2": 5}[element]
    wall = _at(WALL, mode_id, rotation=0.0)
    (image,) = _plan([wall], mode_id=mode_id).images
    assert image.element == element
    assert image.rotation == expected
    assert (element in AXIS_SWAPPING) == (expected == 1.0)


def test_a_wall_index_in_a_radian_encoded_file_passes_through_verbatim():
    """There the stored value carries no shape information -- both DEscape and
    the game derive the shape from connectivity instead."""
    literal = _at(WALL, 1, 0, rotation=0.0)
    radian = _at(WALL, 1, 1, rotation=2.5132741228718345)  # 2 * 2pi/5
    plan = _plan([literal, radian], mode_id=1)
    assert [image.rotation for image in plan.images] == [0.0, radian.rotation]


# Measured for GH #197 by pushing canonical masks through
# wall_variant_from_neighbours8(): the image of wall index 0, 1, 3, 4 per element.
_WALL_INDEX_TABLE = {
    "id": (0, 1, 3, 4),
    "r2": (0, 1, 3, 4),
    "r": (1, 0, 4, 3),
    "r3": (1, 0, 4, 3),
    "mx": (0, 1, 4, 3),
    "my": (0, 1, 4, 3),
    "d": (1, 0, 3, 4),
    "a": (1, 0, 3, 4),
}


@pytest.mark.parametrize("element", list(_WALL_INDEX_TABLE))
def test_wall_index_image_matches_the_measured_table(element):
    images = [wall_index_image(float(index), element) for index in (0, 1, 3, 4)]
    assert images == [float(i) for i in _WALL_INDEX_TABLE[element]]
    assert wall_index_image(2.0, element) == 2.0
    for off in (7.0, 2.5, -1.0, 5.0):
        assert wall_index_image(off, element) == off


# Wall pieces on a 16x16 lattice, apart from each other: a run along x, a run
# along y, both diagonal runs, a corner and an L with a free end.
_WALL_TILES = (
    (2, 2), (3, 2), (4, 2), (5, 2),
    (12, 1), (12, 2), (12, 3),
    (2, 13), (3, 12), (4, 11), (5, 10),
    (9, 8), (10, 9), (11, 10), (12, 11),
    (8, 2), (8, 3), (9, 3),
    (1, 6), (2, 6), (3, 6), (3, 7),
)


@pytest.mark.parametrize("element", list(TRANSFORMS))
def test_a_transformed_wall_layout_still_agrees_with_its_neighbours(element):
    """Every wall stored with the index its neighbours give it must, after its
    position and its index both go through `element`, still store the index
    its new neighbours give it -- the shape the game would re-derive."""
    from descape import unit_sprites

    n = 16
    tiles = set(_WALL_TILES)

    def derived(tile, layout):
        mask = unit_sprites.neighbour_mask(*tile, layout, diagonals=True)
        return unit_sprites.wall_variant_from_neighbours8(mask)

    stored = {tile: derived(tile, tiles) for tile in tiles}
    assert set(stored.values()) == {0, 1, 2, 3, 4}, "the layout must exercise every index"
    fn = TRANSFORMS[element]
    image_tiles = {fn(x, y, n) for x, y in tiles}
    for tile, index in stored.items():
        image = fn(*tile, n)
        assert wall_index_image(float(index), element) == derived(image, image_tiles), (tile, index)


def test_a_diagonal_wall_mirrored_under_mx_stores_the_other_diagonal():
    wall = _at(WALL, 3, rotation=3.0)  # mode 3 is `mx`
    (image,) = _plan([wall], mode_id=3).images
    assert image.element == "mx"
    assert image.rotation == 4.0


def test_an_angular_element_leaves_a_diagonal_wall_verbatim():
    walls = [FakeUnit(WALL, *_wedge_point(10), rotation=3.0), FakeUnit(WALL, *_wedge_point(14), rotation=4.0)]
    plan, _ = _angular_plan(walls)
    assert plan.images
    assert {image.rotation for image in plan.images} == {3.0, 4.0}


def test_walls_are_radian_reads_the_wall_consts_only():
    assert not walls_are_radian([FakeUnit(WALL, 1.5, 1.5, rotation=3.0), FakeUnit(OAK, 2.5, 2.5, rotation=7.25)])
    assert walls_are_radian([FakeUnit(WALL, 1.5, 1.5, rotation=_RADIAN_3)])
    assert not walls_are_radian([])


@pytest.mark.parametrize("mode_id", [1, 2, 3, 4, 6])
def test_unit_image_is_the_plans_own_image(mode_id):
    """plan_mirror_units() and unit_image() resolve an image the same way."""
    units = [
        _at(ARCHER, mode_id, 0, rotation=1.25),
        _at(GATE_NE, mode_id, 1),
        _at(HOUSE, mode_id, 2),
        _at(WALL, mode_id, 3, rotation=3.0),
    ]
    plan = _plan(units, mode_id=mode_id)
    assert plan.images
    for image in plan.images:
        source = image.source
        got = unit_image(source, source.x, source.y, image.element, N, False)
        assert got == (image.unit_const, image.x, image.y, image.rotation)


def test_an_angle_consts_facing_is_mirrored():
    """Mode 1 is `a`, which sends a tile-space facing theta to 3pi/2 - theta."""
    archer = _at(ARCHER, 1, 0, rotation=1.25)
    (image,) = _plan([archer], mode_id=1).images
    assert image.element == "a"
    assert image.rotation == pytest.approx(3 * math.pi / 2 - 1.25, abs=1e-9)


def test_a_variant_consts_rotation_is_never_transformed():
    tree = _at(OAK, 1, 1, rotation=7.0)
    (image,) = _plan([tree], mode_id=1).images
    assert image.rotation == 7.0


def _angle_gap(a: float, b: float) -> float:
    turn = 2 * math.pi
    return min((a - b) % turn, (b - a) % turn)


_THETAS = (0.0, math.pi / 4, 1.25, math.pi, 7.0, 2 * math.pi - 1e-12)


@pytest.mark.parametrize("element", sorted(TRANSFORMS))
@pytest.mark.parametrize("theta", _THETAS)
def test_facing_image_is_the_linear_part_applied_to_the_facing_vector(element, theta):
    result = facing_image(theta, element)
    assert 0.0 <= result < 2 * math.pi
    lx, ly = _linear_part(element)(1, 0)
    mx, my = _linear_part(element)(0, 1)
    c, s = math.cos(theta), math.sin(theta)
    vx, vy = lx * c + mx * s, ly * c + my * s
    assert math.cos(result) == pytest.approx(vx, abs=1e-9)
    assert math.sin(result) == pytest.approx(vy, abs=1e-9)


_ORACLE = {
    "id": lambda t: t,
    "r": lambda t: t + math.pi / 2,
    "r2": lambda t: t + math.pi,
    "r3": lambda t: t - math.pi / 2,
    "mx": lambda t: math.pi - t,
    "my": lambda t: -t,
    "d": lambda t: math.pi / 2 - t,
    "a": lambda t: 3 * math.pi / 2 - t,
}


@pytest.mark.parametrize("element", sorted(TRANSFORMS))
@pytest.mark.parametrize("theta", _THETAS)
def test_facing_image_matches_the_closed_form(element, theta):
    assert _angle_gap(facing_image(theta, element), _ORACLE[element](theta)) < 1e-9


@pytest.mark.parametrize("g", sorted(TRANSFORMS))
@pytest.mark.parametrize("h", sorted(TRANSFORMS))
def test_facing_image_obeys_the_group_law(g, h):
    for theta in _THETAS:
        chained = facing_image(facing_image(theta, g), h)
        assert _angle_gap(chained, facing_image(theta, compose(h, g))) < 1e-9


@pytest.mark.parametrize("element", ["mx", "my", "d", "a", "r2"])
def test_every_involution_applied_twice_returns_theta(element):
    for theta in _THETAS:
        assert _angle_gap(facing_image(facing_image(theta, element), element), theta) < 1e-9


@pytest.mark.parametrize("angle_count", [8, 16])
@pytest.mark.parametrize("element", sorted(TRANSFORMS))
def test_every_stored_step_maps_exactly_onto_a_stored_step(angle_count, element):
    step = 2 * math.pi / angle_count
    grid = {k * step for k in range(angle_count)}
    images = {facing_image(k * step, element, angle_count) for k in range(angle_count)}
    assert images == grid


def test_ownership_rotation_is_off_by_default_and_steps_along_the_defined_list():
    source = _at(OAK, 6, 0, player=1)
    gaia = _at(OAK, 6, 1, player=0)
    plan = _plan([source, gaia], mode_id=6)
    assert {image.player for image in plan.images} == {0, 1}

    plan = _plan([source, gaia], mode_id=6, player_ids=[1, 2, 4, 5], ownership_steps=1)
    owners = [image.player for image in plan.images if image.source is source]
    assert owners == [2, 4, 5]  # one step per slice, along the defined list
    assert all(image.player == 0 for image in plan.images if image.source is gaia)


@pytest.mark.usefixtures("real_gate_wall_install")
def test_a_mirrored_gates_walls_derive_the_connected_shape():
    """GH #77 step 7: in a radian file the plan copies wall rotations verbatim,
    so the image run's shape comes from connectivity to the reoriented gate."""
    from types import SimpleNamespace

    y = 10
    walls = [FakeUnit(WALL_CONST, x + 0.5, y + 0.5, rotation=_RADIAN_3) for x in (14, 15, 20, 21)]
    gate = FakeUnit(GATE_NE, *render.span_anchor(16, y, 4, 1))  # tiles 16..19 along x
    mm = FakeMapManager()
    tiles = plan_mirror(mm, 2, 0, False, False)  # `d`: a run along x becomes a run along y
    plan = plan_mirror_units(mm, 2, 0, _by_player([*walls, gate]), tiles.source_indices)
    assert not plan.blocked and len(plan.images) == 5
    images = [FakeUnit(i.unit_const, i.x, i.y, player=i.player, rotation=i.rotation) for i in plan.images]
    (image_gate,) = [u for u in images if u.unit_const != WALL_CONST]
    assert image_gate.unit_const == reorient_gate_const(GATE_NE, "d")
    assert set(render.unit_occupied_tiles(image_gate, N, N)) == {(y, 16), (y, 17), (y, 18), (y, 19)}

    units = _by_player([*walls, gate, *images])
    scenario = SimpleNamespace(map_manager=mm, unit_manager=SimpleNamespace(units=units))
    overrides = render.wall_variant_rotation_overrides(scenario)
    shape = {(int(u.x), int(u.y)): overrides.get((1, i)) for i, u in enumerate(units[1]) if u.unit_const == WALL_CONST}
    # Next to the gate: a through-run, index 0 along x and 1 along y. Run ends: 2.
    assert shape == {
        (14, y): 2.0, (15, y): 0.0, (20, y): 0.0, (21, y): 2.0,
        (y, 14): 2.0, (y, 15): 1.0, (y, 20): 1.0, (y, 21): 2.0,
    }


def test_a_non_zero_source_slice_still_emits_the_sources_own_orbit():
    """The element carrying a source onto each destination is h . source^-1,
    not h, so slice 1 mirrors the same way slice 0 does."""
    unit = FakeUnit(OAK, N - 5.5, 5.5)
    plan = _plan([unit], mode_id=3, slice_index=1)
    assert MODE_BY_ID[3].group[1] == "mx"
    assert [(i.x, i.y) for i in plan.images] == [(5.5, 5.5)]


@pytest.mark.parametrize("mode_id", [1, 2, 3, 4, 5, 6, 9])
def test_an_off_parity_building_is_not_its_own_straddler(mode_id):
    """A 4x4 anchored half a tile off (corpus Mills and Castles) re-anchors
    through identity onto other tiles, so emitting identity refused every mirror."""
    source = _at(CASTLE, mode_id)
    source.x += 0.5
    source.y += 0.5
    plan = _plan([source], mode_id=mode_id)
    assert plan.straddling == []
    assert "id" not in {image.element for image in plan.images}
    assert len(plan.images) == len(MODE_BY_ID[mode_id].group) - 1


@pytest.mark.parametrize(("mode_id", "anchor"), [(2, 33.5), (5, 31.5)])
def test_an_off_parity_building_mapped_onto_its_own_footprint_is_deduped(mode_id, anchor):
    """On the axis, d (or r2 at the centre) maps the footprint onto itself; the
    image's whole-tile anchor differs from the off-parity source's raw one."""
    on_axis = FakeUnit(CASTLE, anchor, anchor)
    assert (int(anchor) * N + int(anchor)) in plan_mirror(FakeMapManager(), mode_id, 0, False, False).source_indices
    plan = _plan([on_axis], mode_id=mode_id)
    assert plan.straddling == []
    assert plan.images == []


@pytest.mark.parametrize(("mode_id", "slice_index", "owners"), [(1, 1, [2]), (6, 2, [4, 5, 2])])
def test_ownership_rotation_counts_slices_from_the_source_slice(mode_id, slice_index, owners):
    """Measured bug: a 2-way mirror from slice 1 kept the source owner, since
    steps were counted from slice 0. Images come in group order."""
    x, y = _source_tiles(mode_id, slice_index, margin=1)[0]
    source = FakeUnit(OAK, x + 0.5, y + 0.5, player=1)
    plan = _plan([source], mode_id=mode_id, slice_index=slice_index, player_ids=[1, 2, 4, 5], ownership_steps=1)
    assert [image.player for image in plan.images] == owners


# --- the real-file leg ---------------------------------------------------------


class _History:
    """The bare EditHistory surface commit_unit_edit() touches, for a
    model-level test with no undo stack of its own."""

    def push_unit_record(self, record):
        self.record = record


def _assert_integer_encoded_walls(loaded) -> None:
    """The file holds walls and stores their shapes as literal indices, so the mirror permutes them."""
    from descape import unit_sprites

    units = [u for units in loaded.unit_manager.units for u in units]
    walls = [u for u in units if unit_sprites.rotation_variant_eligible(u.unit_const)]
    assert walls, "no wall in the file, so its encoding proves nothing"
    assert not walls_are_radian(units), "the file's walls read as radian-encoded"


def _assert_permuted_walls(plan, region) -> None:
    """Some wall image of a source in `region` carries an index other than its source's: the remap ran."""
    from descape import unit_sprites

    walls = [
        image for image in plan.images
        if unit_sprites.rotation_variant_eligible(image.unit_const) and region.holds_tile(int(image.source.x), int(image.source.y))
    ]
    assert any(image.rotation != float(image.source.rotation) for image in walls), f"no wall index permuted among {len(walls)}"


@pytest.mark.corpus
def test_a_mirror_survives_write_and_reload(tmp_path):
    """Plan -> apply through UnitEditModel -> write_scenario -> reload, on a
    real integer-encoded file: every image has to read back at the position,
    rotation and owner the plan computed."""
    from pathlib import Path

    from descape.scenario_io import load_map_and_units

    # Mode 3 on this file is the one real-corpus combination with no
    # axis-straddling building, i.e. the case a user could actually apply.
    source = Path(__file__).resolve().parent.parent / "examples" / "2_Joan_coop_4_v0_13.aoe2scenario"
    _mirror_round_trip(load_map_and_units(source), tmp_path)


@pytest.mark.test_map
def test_a_mirror_survives_write_and_reload_on_the_test_map(tmp_path):
    """The corpus round trip on the generated map, whose walls are integer-encoded.
    Mode 3 slice 1 holds the walls region, so `mx` exchanges its diagonal runs (3, 4)."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(conftest.TEST_MAP_PATH)
    _assert_integer_encoded_walls(loaded)
    walls = conftest.load_verify_module("gen_test_map").REGIONS["walls"]
    plan = _mirror_round_trip(loaded, tmp_path, slice_index=1)
    _assert_permuted_walls(plan, walls)


def _mirror_round_trip(loaded, tmp_path, slice_index: int = 0):
    """Mode 3 applied, written and reloaded; every image must read back. Returns the plan."""
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    mm = loaded.map_manager
    model = UnitEditModel(loaded)
    tiles = plan_mirror(mm, 3, slice_index, False, False)
    plan = plan_mirror_units(
        mm, 3, slice_index, loaded.unit_manager.units, tiles.source_indices, referencing=model.referencing
    )
    assert plan.images, "nothing to mirror, so this proves nothing"
    assert not plan.blocked, f"this file no longer mirrors cleanly under mode 3 slice {slice_index}"

    model.begin_unit_edit(range(9))
    model.remove_many(plan.removals)
    for image in plan.images:
        model.add(
            player=image.player,
            unit_const=image.unit_const,
            x=image.x,
            y=image.y,
            z=image.source.z,
            rotation=image.rotation,
        )
    model.commit_unit_edit("Mirror Map", _History())

    out = tmp_path / "mirrored.aoe2scenario"
    write_scenario(loaded, str(out), units=model)
    reloaded = load_map_and_units(out)
    # Positions are compared to float32 resolution, not for exact equality:
    # the file stores them as float32, so a plan value computed in float64
    # comes back rounded (100.54535 -> 100.54535 at f32, which is a different
    # decimal string, not a different position).
    written: dict[tuple, list[tuple[float, float]]] = {}
    for u in reloaded.unit_manager.get_all_units():
        written.setdefault((int(u.player), u.unit_const, round(float(u.rotation), 4)), []).append(
            (float(u.x), float(u.y))
        )
    missing = [
        image
        for image in plan.images
        if not any(
            abs(x - image.x) < 1e-4 and abs(y - image.y) < 1e-4
            for x, y in written.get((int(image.player), image.unit_const, round(image.rotation, 4)), ())
        )
    ]
    assert not missing, (
        f"{len(missing)} image(s) did not survive the round trip: "
        f"{[(m.unit_const, m.x, m.y) for m in missing[:5]]}"
    )
    return plan


@pytest.mark.corpus
def test_real_walls_swap_their_run_direction_under_a_screen_mirror():
    """The remap on real data, not a fake unit: in an integer-encoded corpus
    file, every eligible wall image under `d` carries the swapped index."""
    from pathlib import Path

    from descape import unit_sprites
    from descape.scenario_io import load_map_and_units

    source = Path(__file__).resolve().parent.parent / "examples" / "2_Joan_coop_4_v0_13.aoe2scenario"
    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    tiles = plan_mirror(mm, 2, 0, False, False)  # mode 2's non-identity element is `d`
    plan = plan_mirror_units(mm, 2, 0, loaded.unit_manager.units, tiles.source_indices)
    swapped = [
        image
        for image in plan.images
        if unit_sprites.rotation_variant_eligible(image.unit_const)
        and float(image.source.rotation) in (0.0, 1.0)
    ]
    assert swapped, "no literal-index wall in the source slice, so this proves nothing"
    for image in swapped:
        assert image.rotation == 1.0 - float(image.source.rotation)


@pytest.mark.corpus
def test_real_diagonal_walls_exchange_their_index_under_mode_4():
    """The table above on real data (GH #197): mode 4 is `my`, which keeps the axis
    runs (0, 1) but exchanges the two diagonal runs (3, 4). Modes 2 and 3,
    the legs above, cannot see this: neither moves a diagonal wall."""
    from pathlib import Path

    from descape import unit_sprites
    from descape.scenario_io import load_map_and_units

    source = Path(__file__).resolve().parent.parent / "examples" / "2_Joan_coop_4_v0_13.aoe2scenario"
    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    assert not walls_are_radian(u for units in loaded.unit_manager.units for u in units)
    tiles = plan_mirror(mm, 4, 0, False, False)
    plan = plan_mirror_units(mm, 4, 0, loaded.unit_manager.units, tiles.source_indices)
    walls = [image for image in plan.images if unit_sprites.rotation_variant_eligible(image.unit_const)]
    diagonal = [image for image in walls if float(image.source.rotation) in (3.0, 4.0)]
    assert diagonal, "no diagonal wall in mode 4's source slice, so this proves nothing"
    for image in diagonal:
        assert image.element == "my"
        assert image.rotation == 7.0 - float(image.source.rotation)
    for image in walls:
        if float(image.source.rotation) in (0.0, 1.0, 2.0):
            assert image.rotation == float(image.source.rotation)


@pytest.mark.corpus
def test_the_swapped_wall_index_draws_a_different_frame():
    """What makes the remap load-bearing rather than cosmetic: index 0 and
    index 1 are different sprites, so passing a mirrored wall's rotation
    through verbatim would draw a run along the wrong axis."""
    from descape import asset_source, unit_sprites

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")
    along_x = unit_sprites.sprite_for(WALL, 0.0, 1, 32)
    along_y = unit_sprites.sprite_for(WALL, 1.0, 1, 32)
    assert along_x is not None and along_y is not None
    assert along_x.rgba.shape != along_y.rgba.shape or along_x.rgba.tobytes() != along_y.rgba.tobytes()


# --- angular modes (6-way / 3-fold) --------------------------------------------
#
# Elements are (sixths, reflect) = R^sixths . a^reflect. The expectations
# below are hand-computed from the geometry (run directions at 0/45/90/135
# degrees, a 60 degree turn landing 15 degrees from exactly one of them), not
# read back from the code.

R = [AngularElement(s, False) for s in range(6)]
A = AngularElement(0, True)
R2A = AngularElement(2, True)
R4A = AngularElement(4, True)

_GATE_MAP = {
    R[0]: (0, 1, 2, 3),
    R[1]: (1, 2, 3, 0),  # +60: every run direction steps one orientation on
    R[2]: (3, 0, 1, 2),  # +120 = -60 mod 180
    R[3]: (0, 1, 2, 3),  # 180 is the identity on lines
    R[4]: (1, 2, 3, 0),
    R[5]: (3, 0, 1, 2),
    A: (2, 1, 0, 3),  # `a` itself, the exact D4 element
    R2A: (1, 0, 3, 2),  # mirror axis at 15 degrees: theta -> 30 - theta
    R4A: (3, 2, 1, 0),  # mirror axis at 75 degrees: theta -> 150 - theta
}
_SWAPS = {R[0]: False, R[1]: True, R[2]: True, R[3]: False, R[4]: True, R[5]: True, A: True, R2A: False, R4A: False}


def _angular_plan(units, mode_id=11, slice_index=0, **kwargs):
    mm = FakeMapManager()
    tiles = plan_mirror(mm, mode_id, slice_index, False, False)
    plan = plan_mirror_units(
        mm, mode_id, slice_index, _by_player(units), tiles.source_indices, unreachable=tiles.unreachable, **kwargs
    )
    return plan, tiles


def _wedge_point(radius: float, mode_id: int = 11, slice_index: int = 0) -> tuple[float, float]:
    """A point `radius` tiles from the centre, on the source wedge's bisector."""
    spec = MODE_BY_ID[mode_id].angular
    bu, bv = spec.elements[slice_index].apply(*spec.bisector(1.0))
    return (N / 2 + radius * bu, N / 2 + radius * bv)


@pytest.mark.parametrize("element", list(_GATE_MAP))
def test_an_angular_gate_snaps_to_the_nearest_orientation(element):
    assert gate_orientation_map(element) == _GATE_MAP[element]


def test_the_exact_angular_elements_reorient_gates_exactly_like_d4():
    assert gate_orientation_map(R[0]) == gate_orientation_map("id")
    assert gate_orientation_map(R[3]) == gate_orientation_map("r2")
    assert gate_orientation_map(A) == gate_orientation_map("a")


def test_a_rotation_reorients_a_gate_the_way_cycle_const_does():
    """The plan's note: the rotational cases ARE cycle_const() steps (the
    reflections are not), which is an independent derivation of the map."""
    from descape import gate_orientation

    checked = 0
    for siblings in gate_orientation.groups().values():
        for const in siblings:
            for sixths, steps in ((1, 1), (2, -1), (4, 1), (5, -1)):
                if all(tile_span(c, render.NON_BUILDING_SPAN) == (1, 1) for c in siblings):
                    assert reorient_gate_const(const, R[sixths]) == const
                    continue
                assert reorient_gate_const(const, R[sixths]) == gate_orientation.cycle_const(const, steps)
                checked += 1
    assert checked == 18 * 4 * 4


def test_every_gate_orientation_has_an_angular_sibling_within_15_degrees():
    """The angular counterpart of the D4 block's 576-case check, with its own
    count: 18 non-corner groups x 4 orientations x 9 distinct elements."""
    from descape import gate_orientation
    from descape.mirror_tools import _RUN_DIRECTIONS

    elements = sorted({e for mode_id in (10, 11, 12) for e in MODE_BY_ID[mode_id].angular.elements})
    assert len(elements) == 9
    checked = 0
    for siblings in gate_orientation.groups().values():
        if all(tile_span(c, render.NON_BUILDING_SPAN) == (1, 1) for c in siblings):
            continue
        for index, const in enumerate(siblings):
            for element in elements:
                target = reorient_gate_const(const, element)
                assert target in siblings
                vx, vy = element.apply(*_RUN_DIRECTIONS[index])
                tx, ty = _RUN_DIRECTIONS[siblings.index(target)]
                gap = abs(math.degrees(math.atan2(vy, vx) - math.atan2(ty, tx))) % 180
                assert min(gap, 180 - gap) == pytest.approx(0 if element in (R[0], R[3], A) else 15, abs=1e-6)
                checked += 1
    assert checked == 18 * 4 * 9


@pytest.mark.parametrize("element", list(_SWAPS))
def test_a_walls_run_direction_swaps_under_an_angular_element_by_nearest_axis(element):
    assert swaps_axes(element) == _SWAPS[element]


def test_a_mirrored_wall_swaps_its_index_under_60_and_120_but_not_180():
    wall = FakeUnit(WALL, *_wedge_point(10), rotation=0.0)
    plan, _ = _angular_plan([wall])
    by_element = {image.element: image.rotation for image in plan.images}
    assert by_element == {R[1]: 1.0, R[2]: 1.0, R[3]: 0.0, R[4]: 1.0, R[5]: 1.0}


def test_a_building_image_keeps_its_axis_aligned_span_on_whole_tiles():
    x, y = _wedge_point(12)
    castle = FakeUnit(CASTLE, round(x), round(y))  # a 4x4 anchors on a whole tile
    plan, _ = _angular_plan([castle])
    assert len(plan.images) == 5
    for image in plan.images:
        assert image.unit_const == CASTLE
        assert (image.x - 2).is_integer() and (image.y - 2).is_integer()
        u, v = image.element.apply(2 * castle.x - N, 2 * castle.y - N)
        assert abs(image.x - (u + N) / 2) <= 0.5 and abs(image.y - (v + N) / 2) <= 0.5


def test_a_unit_rotated_off_the_square_is_skipped_and_reported():
    corner = FakeUnit(OAK, N - 0.5, N - 0.5)  # the far corner of wedge 0, 45 degrees out
    plan, _ = _angular_plan([corner])
    skipped = {element for unit, element in plan.off_map if unit is corner}
    assert R[1] in skipped  # 105 degrees at radius ~44.5: well past the square's edge
    assert not skipped & {image.element for image in plan.images}
    for image in plan.images:
        assert 0 <= image.x < N and 0 <= image.y < N


def test_a_unit_on_an_unreachable_tile_is_kept_not_removed():
    plan, tiles = _angular_plan([])
    idx = min(tiles.unreachable)
    kept = FakeUnit(OAK, idx % N + 0.5, idx // N + 0.5)
    plan, _ = _angular_plan([kept])
    assert plan.removals == [] and plan.images == []


@pytest.mark.parametrize("const", [GRASS, OAK])
def test_a_kept_passable_unit_does_not_block_an_angular_copy(const):
    """TASK-042 in the angular modes: the castle's 120-degree copy covers
    unreachable tile (15, 59), where a unit is kept. Only a collidable one skips it."""
    castle = FakeUnit(CASTLE, 62, 32)
    kept = FakeUnit(const, 15.5, 59.5, player=0)
    plan, tiles = _angular_plan([castle, kept], mode_id=10)
    assert 59 * N + 15 in tiles.unreachable
    skipped = {element for unit, element in plan.off_map if unit is castle}
    assert (R[2] in skipped) == (const == OAK)


def test_a_unit_at_a_fixed_point_produces_no_duplicate_images():
    """_key()'s 1e-6 rounding absorbs float-trig error at a genuine fixed
    point: the centre under every rotation, and a point on `a`'s axis."""
    centre = FakeUnit(OAK, N / 2, N / 2)
    plan, _ = _angular_plan([centre], mode_id=10)
    assert plan.images == []
    # Mode 11's r2 takes D4's exact path, whose boundary snap-back moves a
    # tile-corner unit into the reflected tile -- exactly what mode 5 does.
    plan, _ = _angular_plan([centre], mode_id=11)
    (image,) = plan.images
    assert image.element == R[3]
    from descape.mirror_tools import _image_position

    assert (image.x, image.y) == _image_position(OAK, centre.x, centre.y, N, "r2") == (N / 2 - 1, N / 2 - 1)

    spec = MODE_BY_ID[12].angular
    on_axis = FakeUnit(OAK, 20.3, N - 20.3)  # x + y == n: fixed by `a`
    slice_index = spec.sector(*doubled(20, 43, N))
    plan, _ = _angular_plan([on_axis], mode_id=12, slice_index=slice_index)
    assert len(plan.images) == 2  # a D3 orbit through a mirror axis has 3 points


def test_six_way_ownership_hands_the_wedges_consecutive_owners():
    unit = FakeUnit(OAK, *_wedge_point(8), player=1)
    plan, _ = _angular_plan([unit], player_ids=[1, 2, 3, 4, 5, 6], ownership_steps=1)
    assert [image.player for image in plan.images] == [2, 3, 4, 5, 6]


def test_six_way_ownership_counts_wedges_from_the_source_wedge():
    unit = FakeUnit(OAK, *_wedge_point(8, slice_index=2), player=1)
    plan, _ = _angular_plan([unit], slice_index=2, player_ids=[1, 2, 3, 4, 5, 6], ownership_steps=1)
    assert [image.player for image in plan.images] == [5, 6, 2, 3, 4]


def test_an_off_parity_building_is_not_its_own_straddler_in_an_angular_mode():
    x, y = _wedge_point(12)
    castle = FakeUnit(CASTLE, round(x) + 0.5, round(y) + 0.5)
    plan, _ = _angular_plan([castle])
    assert plan.straddling == []
    assert AngularElement(0, False) not in {image.element for image in plan.images}


def test_an_angle_consts_facing_turns_with_the_rotation():
    archer = FakeUnit(ARCHER, *_wedge_point(8), rotation=1.25)
    plan, _ = _angular_plan([archer])
    image = next(image for image in plan.images if image.element == R[1])
    assert image.rotation == pytest.approx(1.25 + math.pi / 3, abs=1e-9)


def test_the_exact_screen_mirror_inside_mode_12_places_units_like_mode_1():
    spec = MODE_BY_ID[12].angular
    unit = FakeUnit(OAK, *_wedge_point(9, 12, 0))
    plan, _ = _angular_plan([unit], mode_id=12)
    (image,) = [image for image in plan.images if image.element == A]
    assert spec.elements[0] == R[0]
    assert (image.x, image.y) == POSITION_TRANSFORMS["a"](unit.x, unit.y, N)


@pytest.mark.corpus
def test_a_six_way_mirror_survives_write_and_reload(tmp_path):
    """Verification step 4: mode 11 through the real write path on a real
    file. The source wedge and the unreachable corners read back
    byte-identical, every other tile reads back as planned, and every unit
    image survives."""
    from pathlib import Path

    from descape.scenario_io import load_map_and_units

    source = Path(__file__).resolve().parent.parent / "examples" / "atilla_1_scn_resaved.aoe2scenario"
    _six_way_round_trip(load_map_and_units(source), tmp_path)


@pytest.mark.test_map
def test_a_six_way_mirror_survives_write_and_reload_on_the_test_map(tmp_path):
    """The corpus round trip on the generated map: wedge 0 holds the walls
    region, whose integer-encoded axis runs (0, 1) swap under 60 and 120 degrees."""
    from descape.scenario_io import load_map_and_units

    loaded = load_map_and_units(conftest.TEST_MAP_PATH)
    _assert_integer_encoded_walls(loaded)
    from descape import unit_sprites

    walls = conftest.load_verify_module("gen_test_map").REGIONS["walls"]
    plan, reloaded = _six_way_round_trip(loaded, tmp_path)
    _assert_permuted_walls(plan, walls)
    # The shared position check ignores rotation, so the permuted wall indexes are read back here.
    stored = {
        (int(u.player), u.unit_const, round(float(u.x), 3), round(float(u.y), 3), round(float(u.rotation), 4))
        for u in reloaded.unit_manager.get_all_units() if unit_sprites.rotation_variant_eligible(u.unit_const)
    }
    for image in plan.images:
        if unit_sprites.rotation_variant_eligible(image.unit_const):
            key = (int(image.player), image.unit_const, round(_f32(image.x), 3), round(_f32(image.y), 3),
                   round(_f32(image.rotation), 4))
            assert key in stored, key


def _six_way_round_trip(loaded, tmp_path):
    """Mode 11 wedge 0, terrain and units, applied, written and reloaded. Returns the unit plan and the reload."""
    from descape.edit_history import tile_state
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    mm = loaded.map_manager
    before = [tile_state(tile) for tile in mm.terrain]
    model = UnitEditModel(loaded)
    tiles = plan_mirror(mm, 11, 0, True, False)
    plan = plan_mirror_units(
        mm, 11, 0, loaded.unit_manager.units, tiles.source_indices,
        referencing=model.referencing, unreachable=tiles.unreachable,
    )
    assert plan.images and tiles.changes
    assert not plan.blocked, "this file no longer mirrors cleanly under mode 11 wedge 0"

    for idx, (terrain_id, elevation, layer) in tiles.changes:
        tile = mm.terrain[idx]
        tile.terrain_id, tile.elevation, tile.layer = terrain_id, elevation, layer
    model.begin_unit_edit(range(9))
    model.remove_many(plan.removals)
    for image in plan.images:
        model.add(
            player=image.player, unit_const=image.unit_const, x=image.x, y=image.y,
            z=image.source.z, rotation=image.rotation,
        )
    model.commit_unit_edit("Mirror Map", _History())

    out = tmp_path / "mirrored.aoe2scenario"
    write_scenario(loaded, str(out), units=model)
    reloaded = load_map_and_units(out)
    after = [tile_state(tile) for tile in reloaded.map_manager.terrain]
    planned = dict(tiles.changes)
    for idx, state in enumerate(after):
        if idx in tiles.source_indices or idx in tiles.unreachable:
            assert state == before[idx], idx
        else:
            assert state == planned.get(idx, before[idx]), idx
    written = sorted(
        (int(u.player), u.unit_const, round(float(u.x), 3), round(float(u.y), 3))
        for u in reloaded.unit_manager.get_all_units()
    )
    for image in plan.images:
        # Rounded from float32, as the file stores it: 221.55350 rounds to .554 in float64 but .553 stored.
        key = (int(image.player), image.unit_const, round(_f32(image.x), 3), round(_f32(image.y), 3))
        assert key in written, key
    return plan, reloaded


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]
