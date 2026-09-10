"""descape.terrain_units -- the pure planner behind Draw/Paint Can's Trees
and Eye candy checkboxes. All of it is Qt-free (stdlib + unit_rotation only),
so this whole file runs in the default tier. See that module's own docstring
for the measured placement model these tests pin.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from descape import terrain_units, unit_rotation

_TERRAIN_UNIT_MAP_PATH = Path(__file__).resolve().parent.parent / "descape" / "terrain_unit_map.json"
_TREE_UNIT_IDS_PATH = Path(__file__).resolve().parent.parent / "descape" / "tree_unit_ids.json"
_GRAPHIC_MAP_PATH = Path(__file__).resolve().parent.parent / "descape" / "unit_graphic_map.json"

FOREST_OAK = 10  # density 1000: 411 FORTR
FOREST_PINE = 19  # 302 BUSH 0, 1053 BUSH2 0, 350 FPIN 1000
FOREST_DRY_SOUTH_AMERICAN = 128  # six-species non-monotonic density list
GRASS_1 = 0  # 1358, density 60 -- doodad only, no tree const
GRASS_2 = 12  # 1358 density 80 (centering 0) + four density-4 doodads
DIRT_1 = 6  # no terrain_unit entry at all -- a safe "no units here" old terrain

FOREST_OAK_CONST = 411
FOREST_PINE_CONST = 350


class FakeUnit:
    """Duck-typed stand-in for AoE2ScenarioParser's Unit -- the planner only
    ever reads .unit_const, and the caller (viewer.py) is what appends the
    real Unit objects this module hands back in TerrainUnitPlan.removes."""

    def __init__(self, unit_const: int, x: float = 0.0, y: float = 0.0):
        self.unit_const = unit_const
        self.x = x
        self.y = y


def _changes(*transitions: tuple[int, int]) -> tuple[list, dict]:
    """Builds a synthetic TileDiffRecord.changes list and its tiles_by_index,
    one tile per (old_terrain_id, new_terrain_id) pair, laid out along a
    single row so index i is tile (i, 0)."""
    changes = [(i, (old, 0, -1), (new, 0, -1)) for i, (old, new) in enumerate(transitions)]
    tiles_by_index = {i: (i, 0) for i in range(len(transitions))}
    return changes, tiles_by_index


def test_every_terrain_unit_const_has_a_variant_graphic_entry():
    """The test that matters, per the plan: render.stored_rotation() returns
    0.0 for a GAIA unit whose const is not rotation_is_variant, so a doodad
    const missing this would render as variant 0 inside DEscape while the
    file on disk is correct. A miss here is a generator bug, not something
    to fall back on at runtime."""
    terrains = json.loads(_TERRAIN_UNIT_MAP_PATH.read_text())["terrains"]
    graphics = {int(k): v for k, v in json.loads(_GRAPHIC_MAP_PATH.read_text())["graphics"].items()}
    consts = {spec["id"] for units in terrains.values() for spec in units}
    for const in consts:
        entry = graphics.get(const)
        assert entry is not None, f"const {const} has no unit_graphic_map.json entry"
        assert entry["angle_count"] >= 1
        assert entry.get("rotation_is_variant") is True, f"const {const} is not rotation_is_variant"


def test_tree_doodad_partition_covers_every_terrain_unit_const():
    tree_ids = {int(uid) for uid in json.loads(_TREE_UNIT_IDS_PATH.read_text())["ids"]}
    terrains = json.loads(_TERRAIN_UNIT_MAP_PATH.read_text())["terrains"]
    consts = {spec["id"] for units in terrains.values() for spec in units}

    assert terrain_units.TREE_CONSTS <= tree_ids
    assert terrain_units.TREE_CONSTS | terrain_units.DOODAD_CONSTS == consts
    assert terrain_units.TREE_CONSTS & terrain_units.DOODAD_CONSTS == frozenset()


def test_is_tree_terrain_matches_terrain_ids_own_tree_terrains():
    from AoE2ScenarioParser.datasets.terrains import TerrainId

    terrains = json.loads(_TERRAIN_UNIT_MAP_PATH.read_text())["terrains"]
    tree_terrain_ids = {t.value for t in TerrainId.tree_terrains()}
    present = {int(tid) for tid in terrains}
    expected_true = tree_terrain_ids & present
    assert len(expected_true) == 24
    for tid in expected_true:
        assert terrain_units.is_tree_terrain(tid), tid

    assert not terrain_units.is_tree_terrain(GRASS_1)
    assert not terrain_units.is_tree_terrain(60)  # GRASS_JUNGLE -- doodads only


def test_density_places_at_the_measured_rate():
    rng = random.Random(0)
    placed = sum(
        1 for _ in range(10_000) if terrain_units.roll_unit(FOREST_OAK, trees=True, doodads=False, rng=rng)
    )
    assert placed == 10_000  # density 1000/1000 -- always succeeds

    rng = random.Random(0)
    placed = sum(
        1
        for _ in range(10_000)
        if terrain_units.roll_unit(GRASS_1, trees=False, doodads=True, rng=rng) is not None
    )
    assert 540 <= placed <= 660  # ~60/1000, wide tolerance against one seed's own noise

    rng = random.Random(0)
    seen = set()
    for _ in range(2000):
        rolled = terrain_units.roll_unit(FOREST_DRY_SOUTH_AMERICAN, trees=True, doodads=True, rng=rng)
        if rolled is not None:
            seen.add(rolled[0])
    assert seen == {1348, 1349, 2570, 1063, 2567, 1053}


def test_variant_is_in_range_and_varies():
    rng = random.Random(0)
    angle_count = unit_rotation.angle_count_for(FOREST_OAK_CONST)
    variants = {terrain_units.variant_for(FOREST_OAK_CONST, rng) for _ in range(100)}
    assert variants <= set(range(angle_count))
    assert len(variants) > 1


def test_plan_one_unit_per_tile_never_two():
    changes, tiles_by_index = _changes((DIRT_1, FOREST_OAK))
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, {}, trees=True, doodads=True, rng=rng
    )
    assert len(plan.adds) == 1
    add = plan.adds[0]
    assert add.unit_const == FOREST_OAK_CONST
    assert (add.x, add.y) == (0.5, 0.5)  # centering 1
    assert add.rotation == float(add.initial_animation_frame)
    assert 0 <= add.initial_animation_frame < unit_rotation.angle_count_for(FOREST_OAK_CONST)


def test_centering_zero_is_within_tile_and_off_center():
    # GRASS_2's own consts are ALL centering 0 -- every add this produces is
    # a doodad placement, never a tile-centered one.
    changes, tiles_by_index = _changes(*[(DIRT_1, GRASS_2)] * 500)
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, {}, trees=True, doodads=True, rng=rng
    )
    assert plan.adds, "expected at least one doodad placement over 500 rolls"
    for add in plan.adds:
        tile_x, tile_y = int(add.x), int(add.y)
        assert tile_x <= add.x < tile_x + 1
        assert tile_y <= add.y < tile_y + 1
        assert add.x != tile_x + 0.5 or add.y != tile_y + 0.5


def test_repainting_the_same_terrain_plans_nothing():
    changes, tiles_by_index = _changes((FOREST_OAK, FOREST_OAK))
    existing = {(0, 0): [FakeUnit(FOREST_OAK_CONST)]}
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=True, doodads=True, rng=rng
    )
    assert plan.adds == []
    assert plan.removes == []


def test_grass_over_forest_removes_the_tree_and_adds_nothing():
    """GRASS_1's only unit (1358) is a doodad, not a tree, so with doodads
    off the new terrain's own roll can never succeed -- deterministic
    regardless of rng, unlike a density-based outcome."""
    changes, tiles_by_index = _changes((FOREST_OAK, GRASS_1))
    tree = FakeUnit(FOREST_OAK_CONST)
    existing = {(0, 0): [tree]}
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=True, doodads=False, rng=rng
    )
    assert plan.removes == [tree]
    assert plan.adds == []
    assert existing[(0, 0)] == []  # the caller's index is kept in step


def test_forest_to_forest_plans_one_remove_and_one_add():
    """FOREST_PINE's list has two density-0 entries before its density-1000
    FPIN -- deterministic regardless of rng, same reasoning as the grass
    case above."""
    changes, tiles_by_index = _changes((FOREST_OAK, FOREST_PINE))
    tree = FakeUnit(FOREST_OAK_CONST)
    existing = {(0, 0): [tree]}
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=True, doodads=True, rng=rng
    )
    assert plan.removes == [tree]
    assert len(plan.adds) == 1
    assert plan.adds[0].unit_const == FOREST_PINE_CONST


def test_both_checkboxes_off_plans_nothing():
    changes, tiles_by_index = _changes((FOREST_OAK, FOREST_PINE))
    tree = FakeUnit(FOREST_OAK_CONST)
    existing = {(0, 0): [tree]}
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=False, doodads=False, rng=rng
    )
    assert not plan
    assert plan.adds == []
    assert plan.removes == []
    # Both off means nothing touches the caller's index either.
    assert existing[(0, 0)] == [tree]


def test_eye_candy_off_leaves_existing_grass_doodads_alone():
    """Removal is scoped to enabled categories: painting a tile whose
    pre-existing doodad isn't in the enabled set must not remove it, even
    though the tile's terrain is genuinely changing."""
    changes, tiles_by_index = _changes((GRASS_1, FOREST_OAK))
    doodad = FakeUnit(1358)
    existing = {(0, 0): [doodad]}
    rng = random.Random(0)
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=True, doodads=False, rng=rng
    )
    assert plan.removes == []
    assert existing[(0, 0)] == [doodad]
    assert len(plan.adds) == 1
    assert plan.adds[0].unit_const == FOREST_OAK_CONST
