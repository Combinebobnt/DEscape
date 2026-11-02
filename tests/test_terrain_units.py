"""descape.terrain_units -- the pure planner behind Draw/Paint Can's Trees
and Eye candy checkboxes. All of it is Qt-free (stdlib + unit_rotation only),
so this whole file runs in the default tier. See that module's own docstring
for the measured placement model these tests pin.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from descape import game_dataset, terrain_units, unit_rotation, unit_variant

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
    0.0 for a GAIA unit whose const is neither rotation_is_variant nor ANGLE, so a doodad
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

    assert tree_ids >= terrain_units.TREE_CONSTS
    assert consts == terrain_units.TREE_CONSTS | terrain_units.DOODAD_CONSTS
    assert frozenset() == terrain_units.TREE_CONSTS & terrain_units.DOODAD_CONSTS


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


@pytest.mark.parametrize("dataset", [game_dataset.AOE2, game_dataset.ROR])
def test_every_planted_const_rolls_inside_its_cycle_variant_range(dataset):
    # docs/WRITE_RULES.md's Terrain planting bullet: variant_for()'s range is unit_variant's modulus on every planted const.
    with game_dataset.activated(dataset):
        planted = terrain_units.TREE_CONSTS | terrain_units.DOODAD_CONSTS
        assert planted
        bad = {
            const: (unit_rotation.angle_count_for(const), unit_variant.variant_count_for(const))
            for const in sorted(planted)
            if unit_rotation.angle_count_for(const) != unit_variant.variant_count_for(const)
        }
        assert not bad, bad


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


# --- GH #183: no auto-placement on cliff tiles ---


def test_a_blocked_tile_gets_no_add_while_its_neighbour_does():
    changes, tiles_by_index = _changes((DIRT_1, FOREST_OAK), (DIRT_1, FOREST_OAK))
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, {}, trees=True, doodads=True, rng=random.Random(0), blocked={(0, 0)}
    )
    assert [(int(a.x), int(a.y)) for a in plan.adds] == [(1, 0)]
    assert plan.adds[0].unit_const == FOREST_OAK_CONST


def test_an_existing_tree_on_a_blocked_changed_tile_is_still_removed():
    changes, tiles_by_index = _changes((FOREST_PINE, FOREST_OAK))
    tree = FakeUnit(FOREST_PINE_CONST)
    existing = {(0, 0): [tree]}
    plan = terrain_units.plan_terrain_units(
        changes, tiles_by_index, existing, trees=True, doodads=True, rng=random.Random(0), blocked={(0, 0)}
    )
    assert plan.removes == [tree]
    assert plan.adds == []
    assert existing[(0, 0)] == []


def test_the_default_blocked_set_plans_what_an_explicit_empty_one_does():
    changes, tiles_by_index = _changes(*[(DIRT_1, GRASS_2)] * 200, (DIRT_1, FOREST_OAK))
    default = terrain_units.plan_terrain_units(
        changes, tiles_by_index, {}, trees=True, doodads=True, rng=random.Random(7)
    )
    explicit = terrain_units.plan_terrain_units(
        changes, tiles_by_index, {}, trees=True, doodads=True, rng=random.Random(7), blocked=frozenset()
    )
    assert default.adds == explicit.adds
    assert any(a.unit_const == FOREST_OAK_CONST for a in default.adds)


# --- TASK-456: the whole-map planner behind New Map's Trees and Eye candy ---

# Synthetic, so a regen of the real tables never moves the captured specs below.
_PARTIAL = 900  # a doodad, then two trees, all partial density
_FULL = 901  # one tree at density 1000
_SYNTHETIC_TERRAINS = {
    _PARTIAL: [
        {"id": 9001, "density": 300, "centering": 0},
        {"id": 9002, "density": 500, "centering": 1},
        {"id": 9003, "density": 200, "centering": 0},
    ],
    _FULL: [{"id": 9004, "density": 1000, "centering": 1}],
}
_SYNTHETIC_ANGLES = {9001: 3, 9002: 5, 9003: 7, 9004: 4}


@pytest.fixture
def synthetic_tables(monkeypatch):
    monkeypatch.setattr(terrain_units, "_TERRAIN_UNITS", _SYNTHETIC_TERRAINS)
    monkeypatch.setattr(terrain_units, "TREE_CONSTS", {9002, 9003, 9004})
    monkeypatch.setattr(terrain_units, "DOODAD_CONSTS", {9001})
    monkeypatch.setattr(unit_rotation, "angle_count_for", _SYNTHETIC_ANGLES.__getitem__)


def _synthetic_changes() -> tuple[list, dict]:
    changes = [(i, (DIRT_1, 0, -1), (_FULL if i % 3 == 0 else _PARTIAL, 0, -1)) for i in range(12)]
    return changes, {i: (i % 4, i // 4) for i in range(12)}


def _rows(adds) -> list[tuple]:
    return [(a.x, a.y, a.unit_const, a.rotation, a.initial_animation_frame) for a in adds]


# Captured from plan_terrain_units() before _spec_for_tile() was split out of it.
_CAPTURED = {
    (True, True): [
        (0.5, 0.5, 9004, 3.0, 3),
        (3.5, 0.5, 9004, 2.0, 2),
        (0.6566183237281052, 1.1337694064564727, 9003, 1.0, 1),
        (1.2480510814816745, 1.412369906876008, 9001, 2.0, 2),
        (2.5, 1.5, 9004, 3.0, 3),
        (3.5, 1.5, 9002, 4.0, 4),
        (0.7655625700113993, 2.6513489647193387, 9001, 1.0, 1),
        (1.5, 2.5, 9004, 3.0, 3),
        (2.7383055323211773, 2.2032412721496306, 9001, 1.0, 1),
        (3.609469415966534, 2.353825806767755, 9001, 0.0, 0),
    ],
    (True, False): [
        (0.5, 0.5, 9004, 3.0, 3),
        (1.5, 0.5, 9002, 0.0, 0),
        (3.5, 0.5, 9004, 0.0, 0),
        (0.5, 1.5, 9002, 0.0, 0),
        (1.5, 1.5, 9002, 3.0, 3),
        (2.5, 1.5, 9004, 1.0, 1),
        (3.5, 1.5, 9002, 1.0, 1),
        (1.5, 2.5, 9004, 3.0, 3),
        (2.52651958485626, 2.390349470567424, 9003, 5.0, 5),
    ],
}


@pytest.mark.parametrize(("trees", "doodads"), list(_CAPTURED))
def test_a_seeded_plan_is_the_captured_one(synthetic_tables, trees, doodads):
    """Pins the roll walk's draw order: the same seed plans the same units."""
    changes, tiles = _synthetic_changes()
    plan = terrain_units.plan_terrain_units(
        changes, tiles, {}, trees=trees, doodads=doodads, rng=random.Random(456)
    )
    assert _rows(plan.adds) == _CAPTURED[(trees, doodads)]


@pytest.mark.parametrize("terrain_id", (_PARTIAL, _FULL))
def test_plan_fill_plants_what_the_terrain_tool_plants_on_a_whole_map_fill(synthetic_tables, terrain_id):
    width, height = 9, 6
    changes = [(i, (DIRT_1, 0, -1), (terrain_id, 0, -1)) for i in range(width * height)]
    tiles = {i: (i % width, i // width) for i in range(width * height)}
    fill = terrain_units.plan_fill(width, height, terrain_id, trees=True, doodads=True, rng=random.Random(3))
    paint = terrain_units.plan_terrain_units(changes, tiles, {}, trees=True, doodads=True, rng=random.Random(3))
    assert fill == paint.adds
    assert fill


def test_plan_fill_puts_one_unit_on_every_tile_of_a_full_density_forest():
    width, height = 13, 7
    specs = terrain_units.plan_fill(width, height, FOREST_OAK, trees=True, doodads=False, rng=random.Random(0))
    assert sorted((int(s.x), int(s.y)) for s in specs) == [(x, y) for x in range(width) for y in range(height)]
    assert {s.unit_const for s in specs} == {FOREST_OAK_CONST}


def test_plan_fill_places_only_the_enabled_kinds(synthetic_tables):
    def consts(**kinds):
        return {s.unit_const for s in terrain_units.plan_fill(20, 20, _PARTIAL, rng=random.Random(1), **kinds)}

    assert consts(trees=True, doodads=False) == {9002, 9003}
    assert consts(trees=False, doodads=True) == {9001}
    assert consts(trees=True, doodads=True) == {9001, 9002, 9003}
    assert consts(trees=False, doodads=False) == set()
    assert terrain_units.plan_fill(20, 20, GRASS_1, trees=True, doodads=False, rng=random.Random(1)) == []


@pytest.mark.parametrize(("trees", "doodads", "rate"), ((True, True, 0.72), (True, False, 0.6), (False, True, 0.3)))
def test_expected_unit_count_matches_a_seeded_fill_on_a_partial_density_terrain(synthetic_tables, trees, doodads, rate):
    tiles = 100 * 100
    expected = terrain_units.expected_unit_count(_PARTIAL, tiles, trees=trees, doodads=doodads)
    assert expected == round(tiles * rate)
    planted = len(terrain_units.plan_fill(100, 100, _PARTIAL, trees=trees, doodads=doodads, rng=random.Random(5)))
    assert abs(planted - expected) <= 0.02 * tiles  # about four standard deviations


def test_expected_unit_count_is_exact_at_full_density():
    for tiles in (61 * 61, 480 * 480):
        assert terrain_units.expected_unit_count(FOREST_OAK, tiles, trees=True, doodads=False) == tiles
    assert len(terrain_units.plan_fill(61, 61, FOREST_OAK, trees=True, doodads=False, rng=random.Random(2))) == 61 * 61
    assert terrain_units.expected_unit_count(FOREST_OAK, 480 * 480, trees=False, doodads=True) == 0
    beach = 2  # no terrain_unit_map entry
    assert terrain_units.expected_unit_count(beach, 480 * 480, trees=True, doodads=True) == 0
