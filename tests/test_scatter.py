"""descape/scatter.py: random unit distribution.

Placement maths runs against a duck-typed scenario and a fake model that
records add_many() specs; the write-path and undo claims run against the
real units_120x120 fixture through UnitEditModel. A corpus-marked test at
the bottom repeats the round trip on a real examples/ file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import batch_api, scatter
from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.unit_model import _NEXT_UNIT_ID_STRUCT, UnitEditModel

RNG_SEED = 20260917
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_FISH_SALMON = 456
_FISH_TUNA = 457


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list]] = []

    def add_many(self, player: int, specs):
        specs = list(specs)
        self.calls.append((player, specs))
        return specs


def _fake_scenario(width: int = 20, height: int = 20) -> SimpleNamespace:
    return SimpleNamespace(map_manager=SimpleNamespace(map_width=width, map_height=height))


def _all_tiles(width: int = 20, height: int = 20) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height) for x in range(width)]


def _scatter(tiles=None, **kwargs):
    model = _FakeModel()
    kwargs.setdefault("count", 10)
    kwargs.setdefault("seed", RNG_SEED)
    placed = scatter.scatter_units(
        _fake_scenario(), model, _all_tiles() if tiles is None else tiles, _FISH_SALMON, **kwargs
    )
    return placed, model


def _positions(placed) -> list[tuple[float, float]]:
    return [(u.x, u.y) for u in placed]


# -- determinism --------------------------------------------------------------


def test_same_seed_is_identical_for_list_set_and_generator() -> None:
    tiles = _all_tiles()
    as_list, _ = _scatter(list(tiles))
    as_set, _ = _scatter(set(tiles))
    as_gen, _ = _scatter(t for t in reversed(tiles))
    assert _positions(as_list) == _positions(as_set) == _positions(as_gen)


def test_different_seed_gives_different_placements() -> None:
    a, _ = _scatter(seed=1)
    b, _ = _scatter(seed=2)
    assert _positions(a) != _positions(b)


# -- placement invariants -----------------------------------------------------


def test_no_two_units_share_a_tile_and_all_are_in_the_tile_set() -> None:
    tiles = {(x, y) for x in range(3, 9) for y in range(4, 7)}
    placed, _ = _scatter(tiles, count=15)
    occupied = [(int(u.x), int(u.y)) for u in placed]
    assert len(placed) == 15
    assert len(set(occupied)) == len(occupied)
    assert set(occupied) <= tiles


def test_off_map_and_duplicate_tiles_are_filtered() -> None:
    tiles = [(-1, 0), (0, -1), (20, 0), (0, 20), (5, 5), (5, 5)]
    placed, _ = _scatter(tiles, count=10)
    assert _positions(placed) == [(5.5, 5.5)]


def test_terrain_tile_objects_are_accepted() -> None:
    tiles = [SimpleNamespace(x=2, y=3), SimpleNamespace(x=4, y=1)]
    placed, _ = _scatter(tiles, count=2)
    assert sorted(_positions(placed)) == [(2.5, 3.5), (4.5, 1.5)]


def test_zero_jitter_places_exactly_at_tile_centres() -> None:
    placed, _ = _scatter(count=40)
    assert all(u.x % 1 == 0.5 and u.y % 1 == 0.5 for u in placed)


def test_jitter_stays_inside_its_band_and_actually_moves_something() -> None:
    placed, _ = _scatter(count=40, jitter=0.4)
    for u in placed:
        for v in (u.x, u.y):
            frac = v - int(v)
            assert 0.1 - 1e-9 <= frac <= 0.9 + 1e-9
    assert any(u.x % 1 != 0.5 for u in placed)


def test_oversized_jitter_is_clamped_inside_the_tile() -> None:
    tiles = {(7, 7)}
    for seed in range(200):
        placed, _ = _scatter(tiles, count=1, jitter=5.0, seed=seed)
        assert (int(placed[0].x), int(placed[0].y)) == (7, 7)


def test_min_spacing_is_respected_and_may_place_fewer() -> None:
    tiles = {(x, y) for x in range(6) for y in range(6)}
    placed, _ = _scatter(tiles, count=36, min_spacing=2)
    occupied = [(int(u.x), int(u.y)) for u in placed]
    assert 0 < len(occupied) < 36
    for i, (ax, ay) in enumerate(occupied):
        for bx, by in occupied[i + 1 :]:
            assert max(abs(ax - bx), abs(ay - by)) >= 2


def test_requesting_more_than_eligible_places_every_eligible_tile() -> None:
    tiles = {(x, y) for x in range(5) for y in range(6)}
    placed, _ = _scatter(tiles, count=200)
    assert len(placed) == 30


def test_count_range_is_drawn_inside_the_range() -> None:
    counts = {len(_scatter(count=(3, 6), seed=seed)[0]) for seed in range(60)}
    assert counts <= {3, 4, 5, 6}
    assert len(counts) > 1


def test_density_rounds_up() -> None:
    tiles = {(x, 0) for x in range(10)}
    placed, _ = _scatter(tiles, count=None, density=0.25)
    assert len(placed) == 3


@pytest.mark.parametrize("kwargs", [{"count": 5, "density": 0.5}, {"count": None}])
def test_exactly_one_of_count_or_density(kwargs) -> None:
    with pytest.raises(ValueError):
        _scatter(**kwargs)


def test_nothing_eligible_never_calls_add_many() -> None:
    placed, model = _scatter([], count=5)
    assert placed == []
    assert model.calls == []


def test_weighted_mix_uses_only_the_given_consts() -> None:
    model = _FakeModel()
    placed = scatter.scatter_units(
        _fake_scenario(),
        model,
        _all_tiles(),
        [_FISH_SALMON, _FISH_TUNA],
        count=100,
        weights=[0.0, 1.0],
        seed=RNG_SEED,
    )
    assert {u.unit_const for u in placed} == {_FISH_TUNA}


def test_player_is_passed_to_add_many() -> None:
    _, model = _scatter(player=3)
    assert [player for player, _ in model.calls] == [3]


# -- pass-through (the rotation hard rule) ------------------------------------


def test_default_rotation_and_animation_frame_are_zero() -> None:
    placed, _ = _scatter(count=40)
    assert all(u.rotation == 0.0 for u in placed)
    assert all(u.initial_animation_frame == 0 for u in placed)


def test_rotation_choices_are_stored_verbatim() -> None:
    placed, _ = _scatter(count=40, rotation_choices=[7.0, 11.0], animation_frames=[2, 9])
    assert {u.rotation for u in placed} <= {7.0, 11.0}
    assert {u.initial_animation_frame for u in placed} <= {2, 9}
    assert len({u.rotation for u in placed}) == 2


# -- occupied tiles -----------------------------------------------------------


def test_avoid_occupied_skips_tiles_under_existing_units() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    covered = scatter.occupied_tiles(loaded)
    assert (5, 5) in covered  # GAIA const 349 at 5.5, 5.5
    assert (10, 10) in covered  # player 1 house at 10.5, 10.5

    model = UnitEditModel(loaded)
    region = {(x, y) for x in range(4, 14) for y in range(4, 12)}
    placed = scatter.scatter_units(loaded, model, region, _FISH_SALMON, count=500, avoid_occupied=True, seed=RNG_SEED)
    placed_tiles = {(int(u.x), int(u.y)) for u in placed}
    assert placed_tiles == region - covered


def test_occupied_tiles_exclude_skips_only_the_given_units() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    stray = next(u for u in loaded.unit_manager.units[0] if (int(u.x), int(u.y)) == (5, 5))
    covered = scatter.occupied_tiles(loaded, exclude=[stray])
    assert (5, 5) not in covered
    assert (10, 10) in covered


def test_occupied_tiles_multi_tile_only_keeps_just_the_buildings() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    buildings = scatter.occupied_tiles(loaded, multi_tile_only=True)
    assert (5, 5) not in buildings  # the 1x1 GAIA const 349
    assert (10, 10) in buildings  # the house
    assert buildings < scatter.occupied_tiles(loaded)


def _garrison_scenario(*units) -> SimpleNamespace:
    return SimpleNamespace(unit_manager=SimpleNamespace(units=[list(units)] + [[] for _ in range(8)]))


def _ref(reference_id: int, inside: int = -1) -> SimpleNamespace:
    return SimpleNamespace(reference_id=reference_id, garrisoned_in_id=inside)


def test_garrison_riders_walks_nested_garrisons_and_skips_the_hosts() -> None:
    host, rider, nested, other = _ref(7), _ref(8, inside=7), _ref(9, inside=8), _ref(10, inside=99)
    found = scatter.garrison_riders(_garrison_scenario(host, rider, nested, other), [host])
    assert {id(u) for u in found} == {id(rider), id(nested)}


def test_garrison_riders_a_minus_one_host_holds_nothing_and_self_reference_is_no_rider() -> None:
    free = [_ref(i) for i in range(3)]
    unset_host = _ref(-1)
    self_held = _ref(5, inside=5)
    scenario = _garrison_scenario(unset_host, self_held, *free)
    assert scatter.garrison_riders(scenario, [unset_host]) == []
    assert scatter.garrison_riders(scenario, [self_held]) == []


# -- scatter_existing_units (GH #105) -----------------------------------------

_HOUSE = 70  # 2x2
_TOWN_CENTER = 109  # 4x4


def _unit(const: int = _FISH_SALMON, x: float = 0.5, y: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(unit_const=const, x=x, y=y, rotation=7.0, z=1.25)


def _existing(units, tiles=None, **kwargs) -> scatter.ExistingScatter:
    kwargs.setdefault("seed", RNG_SEED)
    return scatter.scatter_existing_units(_fake_scenario(), units, _all_tiles() if tiles is None else tiles, **kwargs)


def test_existing_same_seed_gives_the_same_positions() -> None:
    units = [_unit() for _ in range(12)] + [_unit(_HOUSE, 1.0, 1.0)]
    tiles = _all_tiles()
    first = _existing(units, list(tiles))
    again = _existing(units, set(tiles))
    assert first == again
    assert _existing(units, seed=RNG_SEED + 1).positions != first.positions


def test_existing_one_per_tile_inside_the_region_and_nothing_is_mutated() -> None:
    tiles = {(x, y) for x in range(3, 9) for y in range(4, 8)}
    units = [_unit() for _ in range(20)]
    plan = _existing(units, tiles)
    assert plan.reused == 0
    cells = [(int(x), int(y)) for x, y in plan.positions]
    assert set(cells) <= tiles
    assert len(set(cells)) == 20
    assert all((u.x, u.y, u.rotation, u.z) == (0.5, 0.5, 7.0, 1.25) for u in units)


def test_existing_reuses_tiles_when_the_region_is_short_and_counts_them() -> None:
    tiles = {(x, 2) for x in range(5)}
    plan = _existing([_unit() for _ in range(8)], tiles)
    cells = [(int(x), int(y)) for x, y in plan.positions]
    assert set(cells) == tiles
    assert plan.reused == 3


def test_existing_jitter_stays_on_its_tile() -> None:
    tiles = {(4, 4), (9, 2)}
    for seed in range(100):
        plan = _existing([_unit(), _unit()], tiles, seed=seed, jitter=5.0)
        assert {(int(x), int(y)) for x, y in plan.positions} == tiles


def test_existing_buildings_snap_to_their_span_and_never_overlap() -> None:
    from descape.render import span_anchor

    tiles = {(x, y) for x in range(10) for y in range(10)}
    units = [_unit(_HOUSE) for _ in range(6)] + [_unit() for _ in range(30)]
    plan = _existing(units, tiles)
    covered: list[tuple[int, int]] = []
    for (x, y), unit in zip(plan.positions, units, strict=True):
        span = 2 if unit.unit_const == _HOUSE else 1
        low = (int(x - span / 2), int(y - span / 2))
        if span == 2:
            assert (x, y) == span_anchor(*low, 2, 2)
        covered += [(low[0] + dx, low[1] + dy) for dy in range(span) for dx in range(span)]
    assert set(covered) <= tiles
    assert len(covered) == len(set(covered)) == 6 * 4 + 30


def test_existing_a_building_that_cannot_fit_stays_put() -> None:
    tiles = {(x, y) for x in range(3) for y in range(3)}  # 3x3: no room for a 4x4
    # All start off-region: a stayer's old footprint inside it would block the others.
    units = [_unit(_TOWN_CENTER, 15.0, 15.0), _unit(_HOUSE, 10.0, 10.0), _unit(_HOUSE, 12.0, 10.0)]
    plan = _existing(units, tiles)
    assert plan.positions[0] is None
    assert plan.positions[1] is not None
    # A second 2x2 cannot fit beside the first in 3x3 without overlapping it.
    assert plan.positions[2] is None


def _final_tiles(unit, position) -> set[tuple[int, int]]:
    """Tiles `unit` covers after the plan: its new footprint, or its old one if it stays put."""
    from descape.render import unit_tile_bounds

    placed = unit if position is None else SimpleNamespace(unit_const=unit.unit_const, x=position[0], y=position[1])
    x0, x1, y0, y1 = unit_tile_bounds(placed, 20, 20)
    return {(x, y) for y in range(y0, y1) for x in range(x0, x1)}


def test_existing_a_building_that_stays_put_is_not_overlapped_by_another_building() -> None:
    tiles = {(x, y) for x in range(4) for y in range(2)}
    stayer_seeds = 0
    for seed in range(60):
        units = [_unit(_HOUSE, 1.0, 1.0), _unit(_HOUSE, 3.0, 1.0)]
        plan = _existing(units, tiles, seed=seed)
        stayer_seeds += any(pos is None for pos in plan.positions)
        first, second = (_final_tiles(u, p) for u, p in zip(units, plan.positions, strict=True))
        assert first.isdisjoint(second), seed
    assert stayer_seeds, "no seed hit the stays-put path"


def test_existing_a_small_unit_never_lands_on_a_building_that_stays_put() -> None:
    tiles = {(x, 5) for x in range(3)}  # a 3x1 strip: no room for a 2x2
    house = _unit(_HOUSE, 3.0, 6.0)  # covers x2-3, y5-6, so (2, 5) is inside the strip
    for seed in range(40):
        plan = _existing([house, _unit()], tiles, seed=seed)
        assert plan.positions[0] is None
        x, y = plan.positions[1]
        assert (int(x), int(y)) in tiles - {(2, 5)}, seed


def test_existing_small_units_with_no_free_tile_stay_put() -> None:
    tiles = {(x, y) for x in range(2) for y in range(2)}
    plan = _existing([_unit(_HOUSE), _unit()], tiles)
    assert plan.positions[0] == (1.0, 1.0)
    assert plan.positions[1] is None
    assert plan.reused == 0


# -- write path and undo (real UnitEditModel) ---------------------------------


def _pond(loaded) -> set[tuple[int, int]]:
    return {(x, y) for x in range(60, 70) for y in range(60, 70)}


def test_scatter_save_reload_round_trip(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    before_max = max(u.reference_id for units in loaded.unit_manager.units for u in units)
    model = UnitEditModel(loaded)
    placed = scatter.scatter_units(loaded, model, _pond(loaded), _FISH_SALMON, count=40, seed=RNG_SEED)
    assert model.has_added_units

    out = tmp_path / "scattered.aoe2scenario"
    batch_api.save(loaded, out, units=model)
    reloaded = load_map_and_units(out)

    gaia = [u for u in reloaded.unit_manager.units[0] if u.unit_const == _FISH_SALMON]
    assert sorted((u.x, u.y) for u in gaia) == sorted(_positions(placed))
    all_ids = [u.reference_id for units in reloaded.unit_manager.units for u in units]
    assert len(all_ids) == len(set(all_ids))
    assert all(u.reference_id > before_max for u in gaia)


def test_save_without_units_model_drops_the_scatter(tmp_path: Path) -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = UnitEditModel(loaded)
    scatter.scatter_units(loaded, model, _pond(loaded), _FISH_SALMON, count=40, seed=RNG_SEED)

    out = tmp_path / "terrain_only.aoe2scenario"
    batch_api.save(loaded, out)
    reloaded = load_map_and_units(out)
    assert not any(u.unit_const == _FISH_SALMON for u in reloaded.unit_manager.units[0])


def test_one_begin_commit_pair_is_one_undo_step() -> None:
    loaded = load_map_and_units(FIXTURE_PATH)
    model = UnitEditModel(loaded)
    history = EditHistory()
    gaia_before = list(loaded.unit_manager.units[0])
    next_id_before = model.next_unit_id

    model.begin_unit_edit([0])
    placed = scatter.scatter_units(loaded, model, _pond(loaded), _FISH_SALMON, count=40, seed=RNG_SEED)
    model.commit_unit_edit("Scatter fish", history)
    assert len(placed) == 40
    assert len(history.records) == 1
    assert model.next_unit_id > next_id_before

    history.undo([], None, None, model)

    assert loaded.unit_manager.units[0] == gaia_before
    assert model.next_unit_id == next_id_before
    # UnitSnapshot carries next_unit_id but not _has_added_units, so the flag
    # stays set and the save patches next_unit_id_to_place back to its
    # original value. Benign; pinned so a change to it is deliberate.
    assert model.has_added_units


# -- corpus -------------------------------------------------------------------


@pytest.mark.corpus
def test_scatter_round_trips_on_a_real_scenario(scenario_path: Path, tmp_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip("units not writable for this file")
    mm = loaded.map_manager
    water = [
        (i % mm.map_width, i // mm.map_width)
        for i, tile in enumerate(mm.terrain)
        if batch_api.is_water(tile.terrain_id)
    ]
    if not water:
        pytest.skip("no water tiles")
    before_max = max((u.reference_id for units in loaded.unit_manager.units for u in units), default=0)
    model = UnitEditModel(loaded)
    placed = scatter.scatter_units(loaded, model, water, _FISH_SALMON, count=25, seed=RNG_SEED)

    out = tmp_path / "scattered.aoe2scenario"
    batch_api.save(loaded, out, units=model)
    reloaded = load_map_and_units(out)

    placed_ids = {u.reference_id for u in placed}
    found = {u.reference_id for u in reloaded.unit_manager.units[0] if u.reference_id in placed_ids}
    assert found == placed_ids
    assert min(placed_ids) > before_max
    all_ids = [u.reference_id for units in reloaded.unit_manager.units for u in units]
    assert len(all_ids) == len(set(all_ids))
    assert _NEXT_UNIT_ID_STRUCT.unpack_from(reloaded.decompressed_body, 0)[0] > max(all_ids)
