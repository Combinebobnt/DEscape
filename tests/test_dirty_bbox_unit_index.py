"""TASK-031.56: `_dirty_screen_bbox` finds the units an elevation edit
triggers through `render.unit_own_tile_index()` (`_units_owning_tiles`)
instead of walking every unit on every step.

The oracle is the walk itself, kept here as `_old_walk`. Every case runs the
bbox twice, once as shipped and once with the walk swapped in, and requires
the same rect, elevation_changed set and written elevations. The `checked`
spy also compares the two unit sets on every call, since a dropped 1x1 unit
can hide inside the dirty ring's own bbox and leave the rect unchanged.

The plan's three exactness claims each get a case: a unit a UnitFilter hides
still counts, every player slot (GAIA, 1, 8) counts, and the unit_gen memo is
neither rebuilt by an elevation edit nor stale after a unit edit.
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import numpy as np
import pytest
from test_invalidate_units_splice import MILL_CONST, WALL_CONST, Unit

from descape import render
from descape.elevation_tools import set_tiles_elevation
from descape.render import _canvas_pixel_dims, elevations_and_proj
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS
from descape.unit_filter import UnitFilter
from descape.unit_model import UnitEditModel

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CORPUS_FILES = (
    "old-allies-final-v2.aoe2scenario",
    "0_June_Event_Scenario.aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
)
BIG_CONST = 182
assert BUILDING_TILE_SPANS[BIG_CONST] == (5, 5)
ONE_TILE_CONST = 4242  # unknown to the span tables, so NON_BUILDING_SPAN
EDIT = (60, 60)
LOOP, ARRAY = 10**9, 0
# (unit_band_radius, sprite_band_radius, with_sprites): Stepped, Sloped, Sloped with sprites.
SHAPES = [(0, 0, False), (1, 0, False), (1, 1, True)]


def _old_walk(scenario, tiles) -> list:
    """The pre-TASK-031.56 scan: every unit of every player, kept by own tile."""
    return [u for units in scenario.unit_manager.units for u in units if (int(u.x), int(u.y)) in tiles]


@pytest.fixture
def checked(monkeypatch):
    """Spies on the shipped _units_owning_tiles: each call must return exactly the walk's units."""
    real = render._units_owning_tiles
    stats = {"calls": 0, "units": 0}

    def spy(scenario, tiles):
        got = real(scenario, tiles)
        want = _old_walk(scenario, tiles)
        assert sorted(map(id, got)) == sorted(map(id, want)), f"index found {len(got)} units, the walk {len(want)}"
        stats["calls"] += 1
        stats["units"] += len(want)
        return got

    monkeypatch.setattr(render, "_units_owning_tiles", spy)
    return stats


def _bbox(scenario, dirty, pre, proj, *, walk=False, threshold=LOOP, **kw):
    elevations = pre.copy()
    changed: set = set()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(render, "BBOX_ARRAY_PATH_MIN_TILES", threshold)
        if walk:
            mp.setattr(render, "_units_owning_tiles", _old_walk)
        rect = render._dirty_screen_bbox(
            scenario, dirty, elevations, proj, _canvas_pixel_dims(proj), elevation_changed=changed, **kw
        )
    return rect, changed, elevations


def _assert_matches_walk(scenario, dirty, pre, proj, tag, **kw):
    new = _bbox(scenario, dirty, pre, proj, **kw)
    old = _bbox(scenario, dirty, pre, proj, walk=True, **kw)
    assert new[0] == old[0], f"{tag}: rect {new[0]} vs the walk's {old[0]}"
    assert new[1] == old[1], f"{tag}: elevation_changed differs"
    assert np.array_equal(new[2], old[2]), f"{tag}: written elevations differ"
    return new


def _shape_kw(shape) -> dict:
    unit_r, sprite_r, sprites = shape
    return {"unit_band_radius": unit_r, "sprite_band_radius": sprite_r, "with_sprites": sprites}


def _raise_around(mm, cx, cy, brush, step, propagate=True) -> list[int]:
    """A brush-sized raise (or lower); returns the moved indices. Raw writes
    when not `propagate`: the library's recursion overflows on random noise."""
    before = [t.elevation for t in mm.terrain]
    half = brush // 2
    targets = [
        (x, y, min(max(mm.get_tile(x, y).elevation + step, 0), 15))
        for x in range(cx - half, cx + half + 1)
        for y in range(cy - half, cy + half + 1)
        if 0 <= x < mm.map_width and 0 <= y < mm.map_height
    ]
    if propagate:
        set_tiles_elevation(mm, targets)
    else:
        for x, y, e in targets:
            mm.get_tile(x, y).elevation = e
    return [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]


def _drive(scenario, rng, rounds, tag, propagate=True):
    """Random brush edits centred on units, each checked against the walk on
    every shape and on both the loop and the array path."""
    mm = scenario.map_manager
    pre, proj = elevations_and_proj(scenario)
    on_map = [
        u for units in scenario.unit_manager.units for u in units
        if 0 <= int(u.x) < mm.map_width and 0 <= int(u.y) < mm.map_height
    ]
    for r in range(rounds):
        seed = rng.choice(on_map)
        brush = rng.choice((1, 3, 9, 21))
        dirty = _raise_around(mm, int(seed.x), int(seed.y), brush, rng.choice((-2, -1, 1, 2, 3)), propagate)
        # Terrain-only tiles ride along, as a Draw stroke's would.
        dirty += [rng.randrange(mm.map_width * mm.map_height) for _ in range(5)]
        if not any(mm.terrain[i].elevation != pre[i // mm.map_width, i % mm.map_width] for i in dirty):
            continue
        last = None
        for shape in SHAPES:
            for threshold in (LOOP, ARRAY):
                last = _assert_matches_walk(
                    scenario, dirty, pre, proj, f"{tag} round {r} brush {brush} {shape} t={threshold}",
                    threshold=threshold, **_shape_kw(shape),
                )
        pre = last[2]


def _synthetic_scenario(seed: int):
    """Blank template, bumpy, with units on every player slot: four spans,
    stacked tiles, map edges, off-map units and a negative fractional x."""
    rng = random.Random(seed)
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for t in mm.terrain:
        t.elevation = rng.randrange(0, 8)
    w, h = mm.map_width, mm.map_height
    players = scenario.unit_manager.units
    consts = (ONE_TILE_CONST, MILL_CONST, WALL_CONST, BIG_CONST)
    for k in range(400):
        x, y = rng.randrange(0, w), rng.randrange(0, h)
        pid = k % 9
        players[pid].append(Unit(x + rng.choice((0.0, 0.5, 0.99)), y + 0.5, consts[k % 4], player=pid))
        if k % 7 == 0:
            players[(pid + 4) % 9].append(Unit(x + 0.5, y + 0.5, consts[(k + 1) % 4], player=(pid + 4) % 9))
    for x, y in ((0, 0), (w - 1, h - 1), (0, h - 1), (w - 1, 0)):
        players[2].append(Unit(x + 0.5, y + 0.5, BIG_CONST, player=2))
    # int() truncates toward zero: (-0.5, 10.5) owns on-map tile (0, 10) for both lookups.
    players[3].append(Unit(-0.5, 10.5, MILL_CONST, player=3))
    players[4].append(Unit(-3.5, 20.5, BIG_CONST, player=4))
    players[5].append(Unit(w + 1.5, 30.5, BIG_CONST, player=5))
    scenario.unit_gen += 1  # direct appends must bump it themselves (LoadedScenario.unit_gen)
    return scenario


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_synthetic_edits_match_the_walk(checked, seed):
    scenario = _synthetic_scenario(seed)
    _drive(scenario, random.Random(seed * 31), 25, f"seed {seed}", propagate=False)
    assert checked["calls"] and checked["units"], f"vacuous: {checked}"


@pytest.mark.corpus
@pytest.mark.parametrize("name", CORPUS_FILES)
def test_corpus_edits_match_the_walk(checked, name):
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not in examples/")
    scenario = load_map_and_units(path)
    _drive(scenario, random.Random(name), 12, name)
    assert checked["calls"] and checked["units"], f"vacuous: {checked}"


def _blank_with(player: int, **add_kw):
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = 2
    model = UnitEditModel(scenario)
    unit = model.add(player, BIG_CONST, EDIT[0] + 0.5, EDIT[1] + 0.5, **add_kw)
    return scenario, model, unit


def _raise_edit(scenario, xy=EDIT) -> list[int]:
    mm = scenario.map_manager
    mm.get_tile(*xy).elevation += 3
    return [xy[1] * mm.map_width + xy[0]]


def _covers_footprint(scenario, unit, kw):
    """The bbox as shipped equals the walk's and, sprites off, is wider than
    with units off: the 5x5 footprint (2 tiles past the edit) widened it.
    Sprite reach swallows the footprint, so there the spy is the evidence."""
    pre, proj = elevations_and_proj(scenario)
    dirty = _raise_edit(scenario)
    rect = _assert_matches_walk(scenario, dirty, pre, proj, "edge", **kw)[0]
    if not kw["with_sprites"]:
        bare = _bbox(scenario, dirty, pre, proj, with_units=False, **kw)[0]
        assert rect != bare
        assert rect[0] <= bare[0] and rect[1] <= bare[1] and rect[2] >= bare[2] and rect[3] >= bare[3]
    assert [id(u) for u in render._units_owning_tiles(scenario, {(int(unit.x), int(unit.y))})] == [id(unit)]


@pytest.mark.parametrize("shape", SHAPES, ids=["stepped", "sloped", "sloped_sprites"])
@pytest.mark.parametrize(
    "unit_filter, add_kw",
    [
        (UnitFilter(players=frozenset({1})), {}),
        (UnitFilter(show_garrisoned=False), {"garrisoned_in_id": 999_999}),
    ],
    ids=["player_hidden", "garrisoned_hidden"],
)
def test_a_unit_the_filter_hides_still_widens_the_bbox(checked, shape, unit_filter, add_kw):
    assert "unit_filter" not in inspect.signature(render._dirty_screen_bbox).parameters
    scenario, _, unit = _blank_with(3, **add_kw)
    assert not unit_filter.matches(3, unit)
    _covers_footprint(scenario, unit, _shape_kw(shape))
    assert checked["units"]


@pytest.mark.parametrize("shape", SHAPES, ids=["stepped", "sloped", "sloped_sprites"])
@pytest.mark.parametrize("player", [0, 1, 8], ids=["gaia", "p1", "p8"])
def test_every_player_slot_widens_the_bbox(checked, shape, player):
    scenario, _, unit = _blank_with(player)
    _covers_footprint(scenario, unit, _shape_kw(shape))
    assert checked["units"]


@pytest.mark.parametrize("shape", SHAPES, ids=["stepped", "sloped", "sloped_sprites"])
def test_the_memo_survives_elevation_edits_and_follows_unit_edits(checked, shape):
    kw = _shape_kw(shape)
    scenario, model, unit = _blank_with(2)
    mm = scenario.map_manager
    pre, proj = elevations_and_proj(scenario)
    pre = _assert_matches_walk(scenario, _raise_edit(scenario), pre, proj, "first", **kw)[2]
    index = render.unit_own_tile_index(scenario)
    gen = scenario.unit_gen

    # An elevation edit, propagation included, neither bumps unit_gen nor rebuilds the index.
    dirty = _raise_around(mm, *EDIT, 9, 2)
    pre = _assert_matches_walk(scenario, dirty, pre, proj, "elevation", **kw)[2]
    assert scenario.unit_gen == gen
    assert render.unit_own_tile_index(scenario) is index

    # A unit move bumps it, and the next lookup sees the new own tile, not the old one.
    new_xy = (EDIT[0] + 20, EDIT[1] + 15)
    model.set_position(unit, new_xy[0] + 0.5, new_xy[1] + 0.5, unit.z)
    assert scenario.unit_gen != gen
    assert render._units_owning_tiles(scenario, {EDIT}) == []
    assert [id(u) for u in render._units_owning_tiles(scenario, {new_xy})] == [id(unit)]
    before = checked["units"]
    pre = _assert_matches_walk(scenario, _raise_edit(scenario, EDIT), pre, proj, "old tile", **kw)[2]
    assert checked["units"] == before, "the moved unit was still found at its old tile"
    _assert_matches_walk(scenario, _raise_edit(scenario, new_xy), pre, proj, "new tile", **kw)
    assert checked["units"] > before
