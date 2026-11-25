"""The own-tile index (descape/own_tile_index.py) stays current through
UnitEditModel's deltas instead of being rebuilt after every unit edit.

Each wired mutator must leave the SAME index object current (spliced, not
rebuilt) and equal to a fresh walk (own_tile_index.check()); an unwired gen
move (whole-list restore(), a hand bump, a dropped model) must give a new one.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path

import pytest

from descape import own_tile_index, render_units, unit_model, unit_sprites
from descape.edit_history import EditHistory
from descape.own_tile_index import OwnTileIndex, own_tile
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_units import UnitAddSpec
from descape.unit_model import _REMOVE_BY_INDEX_MAX, UnitEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
MAP = 120
ARCHER, KNIGHT, HOUSE, CASTLE, OAK, STONE_WALL = 4, 38, 70, 82, 349, 117
GATE_NE, GATE_E = 64, 659
SHARED = [(20 + 3 * k, 40 + (k % 3)) for k in range(8)]


def _scene():
    """Two or three units on each SHARED tile across GAIA and two players, plus
    a gate, a house, a wall and a cliff that each share their own tile."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    model = UnitEditModel(loaded)
    for k, (tx, ty) in enumerate(SHARED):
        model.add(1, ARCHER, tx + 0.5, ty + 0.5)
        model.add(2, KNIGHT, tx + 0.25, ty + 0.75)
        if k % 2:
            model.add(0, OAK, tx + 0.5, ty + 0.5)
    named = {
        "gate": model.add(1, GATE_NE, 10.0, 5.5),
        "house": model.add(2, HOUSE, 30.0, 30.0),
        "wall": model.add(1, STONE_WALL, 50.5, 50.5),
        "cliff": model.add(0, min(unit_sprites.cliff_consts()), 60.5, 60.5),
    }
    for unit in named.values():
        model.add(2, ARCHER, unit.x, unit.y)
    return loaded, model, named


def _current(loaded) -> OwnTileIndex:
    index = render_units.unit_own_tile_index(loaded)
    assert own_tile_index.check(loaded) is index
    return index


def _spliced(loaded, index) -> None:
    assert render_units.unit_own_tile_index(loaded) is index, "rebuilt instead of spliced"
    assert own_tile_index.check(loaded) is index


def _shared_unit(loaded, player=1):
    return loaded.unit_manager.units[player][0]


def _first(loaded, player):
    return loaded.unit_manager.units[player][0]


def _bulk_specs(n: int) -> list[UnitAddSpec]:
    return [UnitAddSpec(5.5 + (k % 7), 90.5 + (k % 5), OAK, 0.0, 0) for k in range(n)]


MUTATIONS = {
    "set_position": lambda loaded, m, n: m.set_position(_shared_unit(loaded), *SHARED[3], 0.0),
    "set_position_same_tile": lambda loaded, m, n: m.set_position(_shared_unit(loaded), 20.9, 40.1, 0.0),
    "set_unit_const": lambda loaded, m, n: m.set_unit_const(n["gate"], GATE_E),
    "replace_type": lambda loaded, m, n: m.replace_type(n["house"], CASTLE, map_w=MAP, map_h=MAP),
    "add": lambda loaded, m, n: m.add(3, KNIGHT, SHARED[0][0] + 0.5, SHARED[0][1] + 0.5),
    "add_many": lambda loaded, m, n: m.add_many(0, [UnitAddSpec(*map(float, SHARED[1]), OAK, 0.0, 0), *_bulk_specs(5)]),
    "remove": lambda loaded, m, n: m.remove(_first(loaded, 2)),
    "remove_many_by_index": lambda loaded, m, n: m.remove_many([_first(loaded, 1), _first(loaded, 2), _first(loaded, 0)]),
    "reassign": lambda loaded, m, n: m.reassign(_first(loaded, 1), 5),
    "swap_players": lambda loaded, m, n: m.swap_players(1, 2),
    "set_status": lambda loaded, m, n: m.set_status(_first(loaded, 1), 1),
    "set_capture_flag": lambda loaded, m, n: m.set_capture_flag(_first(loaded, 1), 1),
    "set_rotation": lambda loaded, m, n: m.set_rotation(_first(loaded, 1), 1.0),
    "set_variant": lambda loaded, m, n: m.set_variant(_first(loaded, 0), 3.0),
    "set_wall_variant": lambda loaded, m, n: m.set_wall_variant(n["wall"], 2),
    "set_cliff_frame": lambda loaded, m, n: m.set_cliff_frame(n["cliff"], 3),
    "set_garrisoned_in": lambda loaded, m, n: m.set_garrisoned_in(_first(loaded, 1), n["house"].reference_id),
}

# The mutators whose delta changes a bucket: the test asserts it did, so a no-op scene can't pass vacuously.
MOVES_A_TILE = {"set_position", "set_unit_const", "replace_type", "add", "add_many", "remove", "remove_many_by_index"}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_each_wired_mutator_splices_the_same_index(name):
    loaded, model, named = _scene()
    index = _current(loaded)
    # Membership only: a reassign or swap reorders a bucket without moving anyone.
    before = {tile: {id(u) for _, _, u in index.get(tile)} for tile in index.tiles()}
    gen = loaded.unit_gen
    MUTATIONS[name](loaded, model, named)
    assert loaded.unit_gen == gen + 1
    _spliced(loaded, index)
    after = {tile: {id(u) for _, _, u in index.get(tile)} for tile in index.tiles()}
    assert (before != after) == (name in MOVES_A_TILE)


def test_remove_many_over_the_by_index_cap_splices():
    loaded, model, _named = _scene()
    bulk = model.add_many(0, _bulk_specs(_REMOVE_BY_INDEX_MAX + 20))
    index = _current(loaded)
    # A unit no list holds rides along: the filter pass skips it, and so must the delta.
    stray = unit_model.new_unit(loaded._scenario.uuid, 0, OAK, 1.5, 1.5, 10**6)
    model.remove_many([*bulk, _first(loaded, 1), stray])
    _spliced(loaded, index)
    assert index.get((1, 1)) == ()


def test_a_whole_list_restore_rebuilds_and_matches():
    loaded, model, _named = _scene()
    model.begin_unit_edit([1])
    model.add(1, KNIGHT, *SHARED[2])
    model.remove(_first(loaded, 1))
    record = model.commit_unit_edit("edit", EditHistory(), push=False)
    index = _current(loaded)
    model.restore(record.before)
    rebuilt = render_units.unit_own_tile_index(loaded)
    assert rebuilt is not index
    assert own_tile_index.check(loaded) is rebuilt


def test_a_field_delta_undo_and_redo_splice():
    loaded, model, named = _scene()
    unit = _shared_unit(loaded)
    model.begin_unit_edit([1], fields_only=True)
    model.set_position(unit, 70.5, 70.5, 0.0)
    model.set_unit_const(named["gate"], GATE_E)
    record = model.commit_unit_edit("move", EditHistory(), push=False)
    index = _current(loaded)
    model.restore(record.before)
    _spliced(loaded, index)
    assert own_tile(unit) == SHARED[0]
    model.restore(record.after)
    _spliced(loaded, index)
    assert [u for _, _, u in index.get((70, 70))] == [unit]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_random_edit_sequence_stays_equal_to_a_fresh_walk(seed):
    rng = random.Random(seed)
    loaded, model, named = _scene()
    index = _current(loaded)
    history = EditHistory()

    def others():
        # The gate stays, for the set_unit_const step.
        return [u for units in loaded.unit_manager.units for u in units if u is not named["gate"]]

    def anyone():
        return rng.choice(others())

    def tile():
        return rng.choice([*SHARED, (rng.randrange(MAP), rng.randrange(MAP))])

    def undo_redo(record):
        """Undo, check, and sometimes redo: each restore sees exactly the state its record left."""
        if rng.random() < 0.7:
            model.restore(record.before)
            own_tile_index.check(loaded)
            if rng.random() < 0.5:
                model.restore(record.after)

    def field_edit():
        model.begin_unit_edit([1], fields_only=True)
        for _ in range(rng.randrange(1, 4)):
            x, y = tile()
            model.set_position(anyone(), x + rng.random(), y + rng.random(), 0.0)
        if rng.random() < 0.3:
            model.set_unit_const(named["gate"], GATE_E if named["gate"].unit_const == GATE_NE else GATE_NE)
        undo_redo(model.commit_unit_edit("fields", history, push=False))

    def list_edit():
        model.begin_unit_edit(range(9))
        model.add(rng.randrange(9), KNIGHT, *(c + 0.5 for c in tile()))
        model.reassign(anyone(), rng.randrange(9))
        undo_redo(model.commit_unit_edit("lists", history, push=False))

    steps = [
        lambda: model.set_position(anyone(), *(c + rng.random() for c in tile()), 0.0),
        lambda: model.add(rng.randrange(9), ARCHER, *(c + 0.5 for c in tile())),
        lambda: model.add_many(0, [UnitAddSpec(*(c + 0.5 for c in tile()), OAK, 0.0, 0) for _ in range(rng.randrange(1, 4))]),
        lambda: model.remove(anyone()),
        lambda: model.remove_many(rng.sample(others(), 3)),
        lambda: model.reassign(anyone(), rng.randrange(9)),
        lambda: model.swap_players(1 + rng.randrange(4), 5 + rng.randrange(4)),
        lambda: model.set_unit_const(named["gate"], GATE_E if named["gate"].unit_const == GATE_NE else GATE_NE),
        field_edit,
        field_edit,
        list_edit,
    ]
    spliced = 0
    for step in range(200):
        rng.choice(steps)()
        live = own_tile_index.check(loaded)
        if live is None:
            index = _current(loaded)
        else:
            spliced += live is index
            assert live is index, f"step {step}: a new index without a rebuild"
    assert spliced > 100, f"vacuous: only {spliced} spliced steps"


def test_a_direct_append_and_hand_bump_rebuilds():
    loaded, _model, _named = _scene()
    index = _current(loaded)
    unit = unit_model.new_unit(loaded._scenario.uuid, 4, ARCHER, 77.5, 77.5, 10**6)
    loaded.unit_manager.units[4].append(unit)
    loaded.unit_gen += 1
    rebuilt = render_units.unit_own_tile_index(loaded)
    assert rebuilt is not index
    assert [u for _, _, u in rebuilt.get((77, 77))] == [unit]


def test_a_dropped_model_leaves_the_index_stale_not_raising():
    loaded, model, _named = _scene()
    index = _current(loaded)
    model.set_position(_shared_unit(loaded), 80.5, 80.5, 0.0)
    _spliced(loaded, index)
    del model
    gc.collect()
    assert not index.is_live()
    rebuilt = render_units.unit_own_tile_index(loaded)
    assert rebuilt is not index
    assert own_tile_index.check(loaded) is rebuilt
    assert len(rebuilt.get((80, 80))) == 1


def _two_models():
    """The units fixture under two models, both built before any edit (a model
    built after an add() would refuse: the file's bytes hold no blob for it)."""
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded), UnitEditModel(loaded)


def test_two_models_alternating_field_edits_stay_equal_to_a_fresh_walk():
    loaded, first, second = _two_models()
    index = _current(loaded)
    rng = random.Random(7)
    units = [u for us in loaded.unit_manager.units for u in us]
    for step in range(40):
        model = (first, second)[step % 2]
        x, y = rng.choice(SHARED)
        model.set_position(rng.choice(units), x + 0.5, y + 0.5, 0.0)
        _spliced(loaded, index)
    assert any(len(index.get(t)) > 1 for t in SHARED), "vacuous: no tile ended up shared"


def test_a_model_left_behind_by_another_models_reassign_raises_not_answers_stale():
    """Two models on one scenario stay aligned only through field edits: after
    the first model's reassign the second one's lists are stale, which its own
    alignment gate would refuse at save. The index must not serve its answer."""
    loaded, first, second = _two_models()
    index = _current(loaded)
    player = next(p for p, us in enumerate(loaded.unit_manager.units) if us)
    moved = _first(loaded, player)
    first.reassign(moved, 8 if player != 8 else 7)
    _spliced(loaded, index)
    second.set_status(moved, 1)
    with pytest.raises(RuntimeError, match="drift"):
        index.get(own_tile(moved))


def test_the_oracle_catches_a_direct_x_write_under_a_stamped_gen():
    loaded, model, _named = _scene()
    _current(loaded)
    unit = _shared_unit(loaded)
    unit.x += 30.0  # behind the model's back, so no delta
    model.set_status(_first(loaded, 2), 1)  # stamps the index current
    with pytest.raises(RuntimeError, match="differs from a fresh walk"):
        own_tile_index.check(loaded)


def test_get_returns_the_fresh_walks_order_and_default():
    loaded, model, _named = _scene()
    fresh = OwnTileIndex.build(loaded, loaded.unit_gen)
    index = _current(loaded)
    model.reassign(_first(loaded, 2), 0)
    model.reassign(_first(loaded, 1), 8)
    _spliced(loaded, index)
    walk: dict = {}
    for player_id, units in enumerate(loaded.unit_manager.units):
        for i, u in enumerate(units):
            walk.setdefault(own_tile(u), []).append((player_id, i, u))
    assert {t: [(p, i, id(u)) for p, i, u in e] for t, e in index.as_dict().items()} == {
        t: [(p, i, id(u)) for p, i, u in e] for t, e in walk.items()
    }
    assert index.get((0, 0)) == () and fresh.get((0, 0), None) is None


def test_a_duck_typed_scenario_builds_fresh_and_is_never_memoized():
    loaded, _model, _named = _scene()

    class Duck:
        unit_manager = loaded.unit_manager

    duck = Duck()
    first, second = render_units.unit_own_tile_index(duck), render_units.unit_own_tile_index(duck)
    assert first is not second
    assert render_units.live_own_tile_index(duck) is None
    assert len(first.get(SHARED[1])) == 3
