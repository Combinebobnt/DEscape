"""Map mirroring Stage 2: mirror_tools.plan_mirror_units() -- no Qt, no
AoE2ScenarioParser. Units are duck-typed here the same way the map manager is
in tests/test_mirror_tools.py.

The claims under test: an image's footprint is the reflection of its source's,
orbit dedup keeps one image per distinct position, a wall's run-direction
index swaps only under an axis-swapping element and only in an integer-encoded
file, ownership rotates along the scenario's own player list, and the two
refusal sweeps (axis straddle, garrisoned destination unit) fire.
"""

from __future__ import annotations

import pytest

from descape import render
from descape.mirror_tools import (
    AXIS_SWAPPING,
    MODE_BY_ID,
    TRANSFORMS,
    compose,
    invert,
    plan_mirror,
    plan_mirror_units,
)
from descape.terrain_palette import tile_span

N = 64
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


def test_a_garrisoned_destination_unit_blocks_the_plan():
    source = _at(OAK, 1, 0)
    holder = FakeUnit(HOUSE, N - source.x - 0.5, N - source.y - 0.5)
    rider = FakeUnit(ARCHER, N - source.y, N - source.x)
    plan = _plan([source, holder, rider], referencing=lambda u: [rider] if u is holder else [])
    assert plan.garrisoned_blockers == []  # the rider is itself being removed
    plan = _plan([source, holder], referencing=lambda u: [source] if u is holder else [])
    assert plan.garrisoned_blockers == [holder]
    assert plan.blocked


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


def test_a_non_wall_rotation_is_never_transformed():
    archer = _at(ARCHER, 1, 0, rotation=1.25)
    tree = _at(OAK, 1, 1, rotation=7.0)
    plan = _plan([archer, tree], mode_id=1)
    assert [image.rotation for image in plan.images] == [1.25, 7.0]


def test_ownership_rotation_is_off_by_default_and_steps_along_the_defined_list():
    source = _at(OAK, 6, 0, player=1)
    gaia = _at(OAK, 6, 1, player=0)
    plan = _plan([source, gaia], mode_id=6)
    assert {image.player for image in plan.images} == {0, 1}

    plan = _plan([source, gaia], mode_id=6, player_ids=[1, 2, 4, 5], ownership_steps=1)
    owners = [image.player for image in plan.images if image.source is source]
    assert owners == [2, 4, 5]  # one step per slice, along the defined list
    assert all(image.player == 0 for image in plan.images if image.source is gaia)


def test_a_non_zero_source_slice_still_emits_the_sources_own_orbit():
    """The element carrying a source onto each destination is h . source^-1,
    not h, so slice 1 mirrors the same way slice 0 does."""
    unit = FakeUnit(OAK, N - 5.5, 5.5)
    plan = _plan([unit], mode_id=3, slice_index=1)
    assert MODE_BY_ID[3].group[1] == "mx"
    assert [(i.x, i.y) for i in plan.images] == [(5.5, 5.5)]


# --- the real-file leg ---------------------------------------------------------


class _History:
    """The bare EditHistory surface commit_unit_edit() touches, for a
    model-level test with no undo stack of its own."""

    def push_unit_record(self, record):
        self.record = record


@pytest.mark.corpus
def test_a_mirror_survives_write_and_reload(tmp_path):
    """Plan -> apply through UnitEditModel -> write_scenario -> reload, on a
    real integer-encoded file: every image has to read back at the position,
    rotation and owner the plan computed."""
    from pathlib import Path

    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    # Mode 3 on this file is the one real-corpus combination with no
    # axis-straddling building, i.e. the case a user could actually apply.
    source = Path(__file__).resolve().parent.parent / "examples" / "2_Joan_coop_4_v0_13.aoe2scenario"
    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    model = UnitEditModel(loaded)
    tiles = plan_mirror(mm, 3, 0, False, False)
    plan = plan_mirror_units(
        mm, 3, 0, loaded.unit_manager.units, tiles.source_indices, referencing=model.referencing
    )
    assert plan.images, "nothing to mirror, so this proves nothing"
    assert not plan.blocked, "this file no longer mirrors cleanly under mode 3"

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
