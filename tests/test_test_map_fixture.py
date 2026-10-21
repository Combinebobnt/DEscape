"""Guards tests/fixtures/test_map_240x240.aoe2scenario, the generated
feature-rich map (tools/gen_test_map.py).

The pin pair is the same shape as tests/test_units_fixture.py's: the
committed file equals a fresh build() on header plus decompressed body
always, and on the compressed bytes only on stock zlib. Every change to a
cliff, wall, trigger or unit serialization path regenerates this file, so
the failure says how.

The completeness check reads the live catalogs and vocabulary, so a
game-data regen that adds an object, terrain, condition or effect fails here
by name until the map is regenerated. The inventory check holds each region
and section to what its "why" promises, reading only the committed file: a
feature dropped from the generator fails by its region's name.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

from descape import (
    asset_source,
    cliff_catalog,
    cliff_connectivity,
    gate_orientation,
    library_compat,
    object_catalog,
    terrain_catalog,
    terrain_classes,
    trigger_fields,
    trigger_organize,
    unit_kind,
    unit_rotation,
    unit_sprites,
    unit_variant,
)
from descape.garrison import capacity
from descape.messages_model import MessagesEditModel
from descape.options_model import OptionsEditModel
from descape.render import unit_tile_bounds
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.terrain_palette import is_dat_only_terrain
from descape.trigger_model import exec_order_value
from testkit.zlib_runtime import BYTES_DIFFER, NOT_STOCK_REASON, STOCK_ZLIB, scenario_parts

import conftest

pytestmark = pytest.mark.test_map

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "test_map_240x240.aoe2scenario"
SIZE_CEILING = 1_000_000

# Vocabulary entries the app's trigger model cannot author. Empty: every 1.59
# condition and effect is authored (tools/gen_test_map.py's docstring lists the
# field-level workarounds).
EXPECTED_MISSING_CONDITIONS: frozenset[int] = frozenset()
EXPECTED_MISSING_EFFECTS: frozenset[int] = frozenset()

MAP = 240
CARDINALS = frozenset({(0, -1), (0, 1), (-1, 0), (1, 0)})
SHORT_MARBLE = 2190
MARBLE_CLIFF = 2178
# Real .sld frame counts of the generator's shaped cliff families (the .dat declares 25, Short Marble 23).
SHAPED_FAMILY_SLD_FRAMES = {264: 24, 1849: 24, 1858: 24, 2069: 24, 2190: 23, 2651: 24}


def _generator():
    return conftest.load_verify_module("gen_test_map")


REGENERATE = f"regenerate it with `{_generator().REGEN_COMMAND}` and commit the result"


@functools.cache
def _fresh() -> bytes:
    """build() with no install visible, as main() runs it; the caches are cleared either side."""
    gen = _generator()
    with mock.patch.dict(os.environ, {"AOE2DE_INSTALL_PATH": gen.NO_INSTALL}):
        asset_source.clear_install_caches()
        try:
            return gen.build()
        finally:
            asset_source.clear_install_caches()


def test_fixture_is_tracked_on_disk() -> None:
    assert FIXTURE_PATH.is_file(), f"{FIXTURE_PATH} is missing -- {REGENERATE}"


def test_committed_fixture_matches_its_generator() -> None:
    """Header plus decompressed body, on any zlib."""
    same = scenario_parts(FIXTURE_PATH.read_bytes()) == scenario_parts(_fresh())
    assert same, f"{FIXTURE_PATH.name} no longer matches tools/gen_test_map.py's build() -- {REGENERATE}"


@pytest.mark.skipif(not STOCK_ZLIB, reason=NOT_STOCK_REASON)
def test_committed_fixture_matches_its_generator_byte_identical() -> None:
    same = FIXTURE_PATH.read_bytes() == _fresh()
    assert same, f"{BYTES_DIFFER}; {REGENERATE}"


def test_fixture_stays_under_the_size_ceiling() -> None:
    size = FIXTURE_PATH.stat().st_size
    assert size < SIZE_CEILING, f"{FIXTURE_PATH.name} is {size} bytes, over the {SIZE_CEILING}-byte ceiling"


# -- reading the committed file -----------------------------------------------


@dataclass
class _Map:
    loaded: object
    units: list  # (player, unit)
    manager: object
    options: OptionsEditModel
    messages: MessagesEditModel

    def elevation(self, x: int, y: int) -> int:
        return self.loaded.map_manager.terrain[y * MAP + x].elevation

    def terrain(self, x: int, y: int) -> int:
        return self.loaded.map_manager.terrain[y * MAP + x].terrain_id

    def layer(self, x: int, y: int) -> int:
        return self.loaded.map_manager.terrain[y * MAP + x].layer

    def in_region(self, region) -> list:
        return [(p, u) for p, u in self.units if region.holds_tile(int(u.x), int(u.y))]


@functools.cache
def _load_committed() -> _Map:
    loaded = load_map_and_units(FIXTURE_PATH)
    units = [(player, unit) for player, player_units in enumerate(loaded.unit_manager.units) for unit in player_units]
    return _Map(loaded, units, parse_triggers(loaded), OptionsEditModel(loaded), MessagesEditModel(loaded))


def _committed() -> _Map:
    """The cached parse, depoisoned on every access: another test in this worker may
    have loaded a file whose version gated the library classes these objects read through."""
    m = _load_committed()
    parse_triggers(m.loaded)
    return m


def _regions() -> dict:
    return _generator().REGIONS


# -- completeness ---------------------------------------------------------------


def _vocabulary_gaps(m: _Map) -> list[str]:
    vocab = library_compat.load_vocabulary("1.59")
    conditions = {c.condition_type for t in m.manager.triggers for c in t.conditions}
    effects = {e.effect_type for t in m.manager.triggers for e in t.effects}
    gaps = [f"condition {t} ({d.name})" for t, d in sorted(vocab.conditions.items())
            if t not in conditions and t not in EXPECTED_MISSING_CONDITIONS]
    gaps += [f"effect {t} ({d.name})" for t, d in sorted(vocab.effects.items())
             if t not in effects and t not in EXPECTED_MISSING_EFFECTS]
    return gaps


def completeness_gaps(m: _Map) -> list[str]:
    """Every catalog object, terrain, condition and effect the committed file lacks."""
    consts = {u.unit_const for _p, u in m.units}
    gaps = [f"object {e.id} ({e.category})" for e in sorted(object_catalog.objects(), key=lambda e: e.id) if e.id not in consts]
    terrains = {t.terrain_id for t in m.loaded.map_manager.terrain}
    gaps += [f"terrain {t.id} ({t.name})" for t in terrain_catalog.terrains() if t.id not in terrains]
    return gaps + _vocabulary_gaps(m)


def test_every_catalog_entry_and_vocabulary_entry_is_on_the_map() -> None:
    gaps = completeness_gaps(_committed())
    assert not gaps, f"missing from {FIXTURE_PATH.name} ({REGENERATE}): {gaps}"


def test_the_expected_missing_vocabulary_is_really_missing() -> None:
    """An entry that becomes authorable must leave the expected-missing set."""
    m = _committed()
    conditions = {c.condition_type for t in m.manager.triggers for c in t.conditions}
    effects = {e.effect_type for t in m.manager.triggers for e in t.effects}
    assert not EXPECTED_MISSING_CONDITIONS & conditions
    assert not EXPECTED_MISSING_EFFECTS & effects


# -- inventory: each region and section holds what its "why" promises -----------


def _consts(pairs) -> set[int]:
    return {u.unit_const for _p, u in pairs}


def _tile(u) -> tuple[int, int]:
    return int(u.x), int(u.y)


def _bounds(u):
    return unit_tile_bounds(u, MAP, MAP)


def _corner(m: _Map, region, cx: int, cy: int) -> None:
    assert m.elevation(cx, cy) == 7
    edge = {0: 0.0, MAP - 1: 239.99}
    assert any(math.isclose(u.x, edge[cx], abs_tol=1e-3) and math.isclose(u.y, edge[cy], abs_tol=1e-3)
               for _p, u in m.in_region(region))


def _edge(m: _Map, region, axis: int, edge: int, point: float, building_flush, run_const) -> None:
    pairs = m.in_region(region)
    assert sum(math.isclose((u.x, u.y)[axis], point, abs_tol=1e-3) for _p, u in pairs) >= 3, "units on the edge line"
    assert any(u.unit_const in unit_kind.building_consts() and building_flush(_bounds(u)) for _p, u in pairs), "flush building"
    if run_const is not None:
        side = 2 * axis + (edge != 0)
        assert any(run_const(u) and _bounds(u)[side] == (edge + 1 if edge else 0) for _p, u in pairs), "run into the edge"


def _check_objects(category: str):
    def check(m: _Map, region) -> None:
        invisible = unit_kind.invisible_consts()
        expected = {e.id for e in object_catalog.objects() if e.category == category and not e.hidden and e.id not in invisible}
        pairs = m.in_region(region)
        missing = expected - _consts(pairs)
        assert not missing, f"missing {sorted(missing)}"
        if category == "Buildings":
            owners = {p for p, u in pairs if u.unit_const not in unit_kind.gaia_only_consts()}
            assert owners == {1}, f"building owners {owners}"
        layered = sum(m.layer(x, y) != -1 for x, y in region.tiles())
        assert layered > 0.8 * sum(1 for _ in region.tiles()), f"only {layered} layered tiles"

    return check


def _check_hidden(m: _Map, region) -> None:
    missing = {e.id for e in object_catalog.objects() if e.hidden} - _consts(m.in_region(region))
    assert not missing, f"missing {sorted(missing)}"


def _check_no_graphic(m: _Map, region) -> None:
    expected = {e.id for e in object_catalog.objects() if not e.hidden} & unit_kind.invisible_consts()
    assert len(expected) >= 10
    missing = expected - _consts(m.in_region(region))
    assert not missing, f"missing {sorted(missing)}"


def _check_cameras(m: _Map, region) -> None:
    views = Counter(
        (m.options.current_value(f"player:initial_view_x:{p}"), m.options.current_value(f"player:initial_view_y:{p}"))
        for p in range(1, 9)
    )
    (tile, count), = views.most_common(1)
    assert count >= 5 and region.holds_tile(*tile), views


def _blocks(m: _Map, region) -> dict[int, int]:
    """terrain id -> how many whole 4x4 blocks of it the region holds, on the generator's 5-tile grid."""
    found: Counter = Counter()
    for by in range(region.y0, region.y1 - 3, 5):
        for bx in range(region.x0, region.x1 - 3, 5):
            ids = {m.terrain(x, y) for x in range(bx, bx + 4) for y in range(by, by + 4)}
            if len(ids) == 1:
                found[ids.pop()] += 1
    return found


def _check_terrain_listed(m: _Map, region) -> None:
    listed = [t.id for t in terrain_catalog.terrains() if not t.hidden and not is_dat_only_terrain(t.id)]
    missing = set(listed) - set(_blocks(m, region))
    assert not missing, f"no 4x4 block of {sorted(missing)}"
    trees = [u for p, u in m.in_region(region) if p == 0]
    assert len(trees) > 100, f"forests auto-treed: {len(trees)} GAIA units"


def _check_terrain_hidden(m: _Map, region) -> None:
    hidden = [t.id for t in terrain_catalog.terrains() if t.hidden or is_dat_only_terrain(t.id)]
    missing = set(hidden) - set(_blocks(m, region))
    assert not missing, f"no 4x4 block of {sorted(missing)}"


def _check_mosaic(m: _Map, region) -> None:
    tiles = list(region.tiles())
    assert region.x1 - region.x0 == 60 and region.y1 - region.y0 == 60
    assert len({m.terrain(x, y) for x, y in tiles}) == 45
    assert sum(m.layer(x, y) != -1 for x, y in tiles) >= len(tiles) // 3


def _rotations_by_const(pairs) -> dict[int, set[float]]:
    out: dict[int, set[float]] = defaultdict(set)
    for _p, u in pairs:
        out[u.unit_const].add(u.rotation)
    return out


def _check_facings(m: _Map, region) -> None:
    by_const = _rotations_by_const(m.in_region(region))
    angle = [e.id for e in object_catalog.objects() if not e.hidden and unit_rotation.rotation_is_angle(e.id)]
    wrong = []
    for const in angle:
        count = unit_rotation.angle_count_for(const)
        facings = {unit_rotation.rotation_to_facing(r, count) for r in by_const.get(const, ())}
        if facings != set(_generator().middle_and_last(count)):
            wrong.append(const)
    assert not wrong, f"facings other than exactly the middle and last for {wrong[:20]}"


def _check_variants(m: _Map, region) -> None:
    by_const = _rotations_by_const(m.in_region(region))
    cyclable = [e.id for e in object_catalog.objects() if not e.hidden and unit_variant.is_cyclable(e.id)]
    middle_and_last = _generator().middle_and_last
    wrong = [c for c in cyclable
             if by_const.get(c) != {float(v) for v in middle_and_last(unit_variant.variant_count_for(c))}]
    assert not wrong, f"variants other than exactly the middle and last for {wrong[:20]}"


def _neighbours(x: int, y: int):
    return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _check_elevation(m: _Map, region) -> None:
    inner = [(x, y) for x, y in region.tiles() if region.holds_tile(x - 1, y - 1) and region.holds_tile(x + 1, y + 1)]
    heights = {m.elevation(x, y) for x, y in inner}
    assert heights == set(range(8)), f"heights {sorted(heights)}"
    peaks = [t for t in inner if all(m.elevation(*n) < m.elevation(*t) for n in _neighbours(*t))]
    pits = [t for t in inner if all(m.elevation(*n) > m.elevation(*t) for n in _neighbours(*t))]
    assert peaks and pits, f"peaks {peaks}, pits {pits}"
    def saddle(x: int, y: int) -> bool:
        h = m.elevation(x, y)
        return min(m.elevation(x, y - 2), m.elevation(x, y + 2)) > h > max(m.elevation(x - 2, y), m.elevation(x + 2, y))

    assert any(saddle(x, y) for x, y in inner if region.holds_tile(x - 2, y - 2) and region.holds_tile(x + 2, y + 2)), "no saddle"
    assert any(m.elevation(x, y) == m.elevation(x + 1, y + 1) > max(m.elevation(x + 1, y), m.elevation(x, y + 1)) for x, y in inner), (
        "no diagonal-only step"
    )
    pairs = m.in_region(region)
    footprints = [{m.elevation(x, y) for x in range(b[0], b[1]) for y in range(b[2], b[3])} for _p, u in pairs if (b := _bounds(u))]
    assert any(len(f) > 1 for f in footprints), "nothing on uneven ground"
    assert any(u.unit_const in unit_sprites.cliff_consts() and m.elevation(*_tile(u)) > 0 for _p, u in pairs), "no cliff on a hill"


def _check_stacks(m: _Map, region) -> None:
    per_tile = Counter(_tile(u) for _p, u in m.in_region(region))
    stacked = [t for t, n in per_tile.items() if n >= 2]
    assert len(stacked) >= 150, f"{len(stacked)} stacked tiles"
    deep = sorted(n for n in per_tile.values() if n >= 70)
    assert len(deep) == 3 and all(n <= 80 for n in deep), f"deep stacks (70+ units) on {len(deep)} tiles: {deep}"
    assert any(m.elevation(*t) > 0 for t in stacked), "no stack on a slope"
    kinds = {u.unit_const for _p, u in m.in_region(region)}
    assert kinds & unit_kind.passable_eye_candy_consts() and kinds & unit_kind.building_consts()
    points = Counter((u.x, u.y) for _p, u in m.in_region(region))
    assert sum(2 <= n < 70 for n in points.values()) >= 50, "few identical-point stacks"
    by_tile = defaultdict(set)
    for _p, u in m.in_region(region):
        by_tile[_tile(u)].add((u.x, u.y))
    assert sum(len(p) >= 2 for p in by_tile.values()) >= 20, "few same-tile, different-point stacks"


def _check_transitions(m: _Map, region) -> None:
    tiles = list(region.tiles())
    present = {m.terrain(x, y) for x, y in tiles}
    missing = set(terrain_classes.water_terrains()) - present
    assert not missing, f"water terrains missing {sorted(missing)}"
    assert present & set(terrain_classes.beach_terrains()), "no beach"

    def water(x, y):
        return terrain_classes.is_water_family(m.terrain(x, y))

    inner = [(x, y) for x, y in tiles if region.holds_tile(x - 1, y - 1) and region.holds_tile(x + 1, y + 1)]
    assert any(not water(*t) and all(water(*n) for n in _neighbours(*t)) for t in inner), "no 1-tile island"
    assert any(water(x, y) and not water(x - 1, y) and not water(x + 1, y) for x, y in inner), "no 1-tile channel"
    farms = [u for _p, u in m.in_region(region) if u.unit_const == 50]
    assert farms and all(m.terrain(*_tile(u)) == 7 for u in farms)


def _check_walls(m: _Map, region) -> None:
    shapes = {(u.unit_const, unit_sprites.variant_index(u.rotation, 5)) for _p, u in m.in_region(region)}
    missing = [(c, i) for c in unit_sprites.wall_family_consts() for i in range(5) if (c, i) not in shapes]
    assert not missing, f"wall (const, index) missing: {missing}"


def _check_gates(m: _Map, region) -> None:
    pairs = m.in_region(region)
    hidden = {e.id: e.hidden for e in object_catalog.objects()}
    visible = {c for g in gate_orientation.groups().values() if all(hidden.get(c) is False for c in g) for c in g}
    assert len(visible) == 24
    missing = visible - _consts(pairs)
    assert not missing, f"gates missing {sorted(missing)}"
    assert sum(u.unit_const in unit_sprites.wall_family_consts() for _p, u in pairs) >= 2 * len(visible)


def _cardinal_graph(pieces: list) -> list[frozenset]:
    """Per piece, the cardinal directions of the pieces sharing a footprint edge with it."""
    bounds = [_bounds(u) for u in pieces]
    arms = []
    for i, b in enumerate(bounds):
        dirs = {cliff_connectivity.adjacency_dir(b, other) for j, other in enumerate(bounds) if j != i}
        arms.append(frozenset(dirs & CARDINALS))
    return arms


def _components(pieces: list) -> list[list[int]]:
    bounds = [_bounds(u) for u in pieces]
    seen: set[int] = set()
    out = []
    for start in range(len(pieces)):
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            group.append(i)
            for j in range(len(pieces)):
                if j not in seen and cliff_connectivity.adjacency_dir(bounds[i], bounds[j]) in CARDINALS:
                    seen.add(j)
                    stack.append(j)
        out.append(group)
    return out


def _check_cliffs(m: _Map, region) -> None:
    pairs = m.in_region(region)
    missing = unit_sprites.cliff_consts() - _consts(pairs)
    assert not missing, f"cliff consts missing {sorted(missing)}"
    frames = {int(u.rotation) for _p, u in pairs}
    assert len(frames) >= 15, f"only frames {sorted(frames)}: runs, caps and corners missing"
    pieces = [u for _p, u in pairs if u.unit_const in unit_sprites.cliff_consts()]
    suffixes = cliff_catalog.suffixes()
    # Every joined piece shows the frame the table gives its realized neighbours at its own size.
    unjoined = set()
    for u, arms in zip(pieces, _cardinal_graph(pieces), strict=True):
        if arms:
            want = cliff_connectivity.rewrite_frames(
                arms, suffixes[u.unit_const], frame_limit=cliff_catalog.frame_count(u.unit_const)
            )
            if want is None or int(u.rotation) not in want:
                unjoined.add((u.unit_const, _bounds(u)))
    assert unjoined == set(), f"pieces not shaped for their neighbours: {sorted(unjoined)}"
    graph = _cardinal_graph(pieces)
    groups = _components(pieces)
    assert any(len(g) >= 8 and all(len(graph[i]) == 2 for i in g) for g in groups), "no closed loop"
    marble = {cliff_catalog.piece_for_suffix(SHORT_MARBLE, s) for s in range(1, 10)} - {None}
    runs = [len(g) for g in groups if all(pieces[i].unit_const in marble for i in g)]
    assert max(runs, default=0) >= 6, f"no Short Marble run continued off its end cap: runs {runs}"


def _check_scenery(m: _Map, region) -> None:
    pairs = m.in_region(region)
    hosts = [u for _p, u in pairs if u.unit_const in (310, 311, 744, 1837, 1148, 2530)]
    assert len({u.unit_const for u in hosts}) == 6
    for host in hosts:
        x0, x1, y0, y1 = _bounds(host)
        inside = [u for _p, u in pairs if u is not host and x0 <= u.x < x1 and y0 <= u.y < y1]
        assert any(u.unit_const == 349 for u in inside) and len(inside) >= 3, host.unit_const
    kinds = _consts(pairs)
    assert len(kinds & unit_kind.obstacle_consts()) >= 10 and len(kinds & unit_kind.passable_eye_candy_consts()) >= 10


def _check_trigger_targets(m: _Map, region) -> None:
    assert len({p for p, _u in m.in_region(region)}) >= 4


def _check_players(m: _Map, region) -> None:
    owned = defaultdict(set)
    for p, u in m.in_region(region):
        owned[p].add(u.unit_const)
    for player in range(1, 7):
        assert {109, 82, 70, 83, 74, 4, 448} <= owned[player], player


def _check_garrisons(m: _Map, region) -> None:
    by_ref = {u.reference_id: (p, u) for p, u in m.units}
    held = Counter(u.garrisoned_in_id for _p, u in m.in_region(region) if u.garrisoned_in_id != -1)
    hosts = [(by_ref[ref][1], n) for ref, n in held.items()]
    assert any(h.unit_const == 82 and n == capacity(82) for h, n in hosts), "no castle at capacity"
    assert any(h.unit_const == 545 and n > 0 for h, n in hosts), "no ship carrying units"
    assert any(not region.holds_tile(*_tile(h)) for h, _n in hosts), "no host outside the region"


def _check_field_spread(m: _Map, region) -> None:
    pairs = m.in_region(region)
    assert len({u.z for _p, u in pairs}) >= 106
    assert any(u.status == 0 for _p, u in pairs)
    widest = max(unit_variant.variant_count_for(u.unit_const) for _p, u in m.units if unit_variant.is_cyclable(u.unit_const))
    assert any(unit_variant.is_cyclable(u.unit_const) and u.rotation == widest - 1 for _p, u in pairs)


def _check_placement(m: _Map, region) -> None:
    pairs = m.in_region(region)
    points = Counter((u.x, u.y) for _p, u in pairs)
    assert max(points.values()) >= 4, "no stack on one point"
    assert any(abs(u.x % 1 - 0.5) > 0.3 for _p, u in pairs), "no off-centre point"
    ships = [u for _p, u in pairs if u.unit_const == 539]
    assert ships and all(terrain_classes.is_water_family(m.terrain(*_tile(u))) for u in ships)
    cliffs = [b for _p, u in pairs if u.unit_const in unit_sprites.cliff_consts() and (b := _bounds(u))]
    assert any(b[0] <= u.x < b[1] and b[2] <= u.y < b[3] for b in cliffs for _p, u in pairs
               if u.unit_const not in unit_sprites.cliff_consts()), "no unit on a cliff"
    wall_consts = set(unit_sprites.wall_family_consts())
    walls = [_bounds(u) for _p, u in pairs if u.unit_const in wall_consts]
    houses = [_bounds(u) for _p, u in pairs if u.unit_const in unit_kind.building_consts() and u.unit_const not in wall_consts]
    others = [u for _p, u in pairs if u.unit_const not in unit_kind.building_consts() | wall_consts]

    def under(u, b) -> bool:
        return b[0] <= u.x < b[1] and b[2] <= u.y < b[3]

    assert any(under(u, b) for u in others for b in houses), "no unit under a building"
    overlapping = any(a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3] for a, b in itertools.combinations(houses, 2))
    assert overlapping, "no overlapping footprints"
    assert any(under(u, b) for u in others for b in walls), "no unit on a wall"

    def water(x: int, y: int) -> bool:
        return terrain_classes.is_water_family(m.terrain(x, y))

    shore = [u for u in others if u.unit_const != 539 and not water(*_tile(u))
             and any(water(_tile(u)[0] + dx, _tile(u)[1] + dy) for dx, dy in CARDINALS)]
    assert shore, "no land unit on the shoreline"


def _check_player_eye_candy(m: _Map, region) -> None:
    pairs = m.in_region(region)
    assert {p for p, u in pairs if u.unit_const == 697} == {1}
    assert len({p for p, u in pairs if u.unit_const in unit_kind.eye_candy_consts() and p not in (0, 1)}) >= 2
    assert {1291, 837} <= _consts(pairs)


def _check_refs(m: _Map, region) -> None:
    refs = sorted(u.reference_id for _p, u in m.units)
    assert len(refs) == len(set(refs))
    gaps = [b for a, b in itertools.pairwise(refs) if b - a > 1]
    assert len(gaps) >= 2, "no reference_id gap"
    large = max(refs)
    assert 2**16 < large < 2**31 and region.holds_tile(*_tile(next(u for _p, u in m.units if u.reference_id == large)))
    next_id = struct.unpack_from("<I", m.loaded.decompressed_body, 0)[0]
    assert next_id > large
    named = {ref for t in m.manager.triggers for e in t.effects for ref in (e.selected_object_ids or [])}
    assert any(ref < next_id and ref not in set(refs) for ref in named), "no trigger names a deleted unit"
    owners = {p for p, u in m.units if u.reference_id in named}
    assert len(owners) >= 3, f"triggers name units of players {owners}"


def _check_strings(m: _Map, region) -> None:
    captions = {u.caption_string for _p, u in m.in_region(region)}
    assert "" in captions and any(len(c.encode()) >= 4000 for c in captions)
    assert any(any(ord(ch) > 0x7F for ch in c) for c in captions)
    assert any("\r\n" in c for c in captions) and any("\r" in c.replace("\r\n", "") for c in captions)
    assert any("\n" in c.replace("\r\n", "") for c in captions)


def _check_players_section(m: _Map) -> None:
    o = m.options
    assert o.current_value("player_count") == 8
    civs = {o.current_value(f"player:civilization:{p}") for p in range(1, 9)}
    colours = {o.current_value(f"player:color:{p}") for p in range(1, 9)}
    assert len(civs) == 8 and len(colours) == 8
    ai = o.current_personality(7)
    assert ai.stored_name == "TestMapAi.Ai" and ai.ai_type == 0
    assert b"TestMapLib.per" in m.loaded.decompressed_body, "the custom AI's included script is not embedded"
    assert o.current_value("player:human:2") == 0


def _check_diplomacy(m: _Map) -> None:
    o = m.options
    grid = {(r, c): o.current_value(f"stance:{r}:{c}") for r in range(1, 9) for c in range(1, 9) if r != c}
    assert set(grid.values()) == {0, 1, 3}
    assert any(grid[(r, c)] != grid[(c, r)] for r, c in grid)
    assert sum(o.current_value(f"allied_victory:{p}") for p in range(1, 9)) >= 2


def _check_map_options(m: _Map) -> None:
    o = m.options
    assert o.current_value("victory_condition") == 4
    assert o.current_value("custom_conquest") == 1 and o.current_value("custom_relics") > 0
    assert o.current_value("custom_explored_percent") > 0
    assert o.current_value("victory_score") == 0xFFFFFFFF, "no score sentinel"


def _check_messages(m: _Map) -> None:
    retrievers = m.loaded._scenario.sections["Messages"].retriever_map
    texts = [r.data for name, r in retrievers.items() if name.startswith("ascii_")]
    assert len(texts) == 6 and all(texts), texts
    assert "\r\r" in texts
    assert m.loaded._scenario.sections["FileHeader"].retriever_map["scenario_instructions"].data == "\r\r", "header copy"
    assert m.messages.current_value("hints_id") != 4294967294 and m.messages.current_value("scouts_id") != 4294967294


def _check_disables(m: _Map) -> None:
    for category in ("techs", "units", "buildings"):
        assert any(m.options.current_value(f"disabled:{category}:{p}") for p in range(1, 9)), category


def _check_trigger_vocab(m: _Map) -> None:
    assert not _vocabulary_gaps(m)
    vocab = library_compat.load_vocabulary("1.59")
    left = []
    for trigger in m.manager.triggers:
        for entries, defs, pmap, attr in (
            (trigger.conditions, vocab.conditions, vocab.condition_presentation, "condition_type"),
            (trigger.effects, vocab.effects, vocab.effect_presentation, "effect_type"),
        ):
            for entry in entries:
                definition = defs.get(getattr(entry, attr))
                if definition is None or not trigger.name.startswith("Vocab "):
                    continue
                specs = trigger_fields.field_specs(definition, pmap, attr)
                if attr == "effect_type":
                    specs = trigger_fields.apply_quantity_cluster_rule(specs, entry)
                for spec in specs:
                    value = getattr(entry, spec.attribute, None)
                    if not spec.read_only and (value == definition.default_attributes.get(spec.name) or value in (None, [], -1, "")):
                        left.append(f"{definition.name}.{spec.name}")
    assert not left, f"fields left at their default: {left}"
    assert any(e.effect_type == 91 for t in m.manager.triggers for e in t.effects), "no unknown effect type"


def _check_trigger_structure(m: _Map) -> None:
    triggers = m.manager.triggers
    names = [t.name for t in triggers]
    order = list(m.manager.trigger_display_order)
    assert len(triggers) == 600 and order != list(range(len(order)))
    assert exec_order_value(m.loaded) == 1
    dividers = [n for n in names if trigger_organize.is_divider(n)]
    assert any(" " not in n for n in dividers) and any(" " in n for n in dividers), dividers
    assert not trigger_organize.is_divider(names[order[0]]), "no leading run"
    assert sum(trigger_organize.parse_tag(n) is not None for n in names) >= 3
    assert any(not t.enabled for t in triggers) and any(t.looping for t in triggers)
    assert any(t.header and not t.effects for t in triggers), "no objective header without effects"
    areas = {(e.area_x1, e.area_y1, e.area_x2, e.area_y2) for t in triggers for e in t.effects
             if getattr(e, "area_x1", None) is not None}
    assert (0, 0, MAP - 1, MAP - 1) in areas, "no whole-map area"
    assert (MAP - 1,) * 4 in areas, "no corner-tile area"
    vocab = library_compat.load_vocabulary("1.59")
    (or_type,) = (t for t, d in vocab.conditions.items() if d.name == "or")
    shown = [[t.conditions[i] for i in t.condition_order] for t in triggers]
    assert any(any(c.condition_type == or_type for c in s[1:-1]) for s in shown), "no OR between displayed conditions"
    for kind in ("condition", "effect"):
        orders = [list(getattr(t, f"{kind}_order")) for t in triggers]
        assert any(o != sorted(o) for o in orders), f"no non-identity {kind} order"


def _check_odd_values(m: _Map) -> None:
    o = m.options
    food = {o.current_value(f"player:food:{p}") for p in range(1, 9)}
    assert {0, 2147483520} <= food
    assert o.current_value("player:starting_age:0") == 4294967295
    pops = {o.current_value(f"player:pop_limit:{p}") for p in range(1, 9)}
    assert 0 in pops
    quantities = [getattr(e, "quantity", None) for t in m.manager.triggers for e in t.effects]
    assert any(isinstance(q, int) and q < 0 and q != -1 for q in quantities), "no negative quantity"
    assert any(isinstance(q, float) and q != int(q) for q in quantities), "no fractional float quantity"


REGION_CHECKS = {
    "corner_nw": lambda m, r: _corner(m, r, 0, 0),
    "corner_ne": lambda m, r: _corner(m, r, MAP - 1, 0),
    "corner_sw": lambda m, r: _corner(m, r, 0, MAP - 1),
    "corner_se": lambda m, r: _corner(m, r, MAP - 1, MAP - 1),
    "edge_n": lambda m, r: _edge(m, r, 1, 0, 0.0, lambda b: b[2] == 0,
                                 lambda u: u.unit_const in unit_sprites.wall_family_consts()),
    "edge_s": lambda m, r: (
        _edge(m, r, 1, MAP - 1, 239.99, lambda b: b[3] == MAP, None),
        _assert(any(terrain_classes.is_water_family(m.terrain(x, MAP - 1)) for x in range(r.x0, r.x1)), "no water on the edge"),
    ),
    "edge_w": lambda m, r: _edge(m, r, 0, 0, 0.0, lambda b: b[0] == 0, lambda u: u.unit_const in unit_sprites.cliff_consts()),
    "edge_e": lambda m, r: _edge(m, r, 0, MAP - 1, 239.99, lambda b: b[1] == MAP,
                                 lambda u: u.unit_const in unit_sprites.wall_family_consts()),
    "objects_buildings": _check_objects("Buildings"),
    "objects_others": _check_objects("Others"),
    "objects_units": _check_objects("Units"),
    "objects_heroes": _check_objects("Heroes"),
    "objects_hidden": _check_hidden,
    "objects_no_graphic": _check_no_graphic,
    "cameras": _check_cameras,
    "terrain_listed": _check_terrain_listed,
    "mosaic": _check_mosaic,
    "facings": _check_facings,
    "variants": _check_variants,
    "elevation": _check_elevation,
    "stacks": _check_stacks,
    "transitions": _check_transitions,
    "walls": _check_walls,
    "gates": _check_gates,
    "cliffs": _check_cliffs,
    "scenery": _check_scenery,
    "terrain_hidden": _check_terrain_hidden,
    "trigger_targets": _check_trigger_targets,
    "players": _check_players,
    "garrisons": _check_garrisons,
    "field_spread": _check_field_spread,
    "placement": _check_placement,
    "player_eye_candy": _check_player_eye_candy,
    "refs": _check_refs,
    "strings": _check_strings,
}

SECTION_CHECKS = {
    "players": _check_players_section,
    "diplomacy": _check_diplomacy,
    "map_options": _check_map_options,
    "messages": _check_messages,
    "disables": _check_disables,
    "trigger_vocab": _check_trigger_vocab,
    "trigger_structure": _check_trigger_structure,
    "odd_values": _check_odd_values,
}


def _assert(condition: bool, message: str) -> None:
    assert condition, message


def test_every_region_and_section_has_an_inventory_check() -> None:
    gen = _generator()
    assert set(REGION_CHECKS) == set(gen.REGIONS)
    assert set(SECTION_CHECKS) == set(gen.SECTIONS)


@pytest.mark.parametrize("name", sorted(REGION_CHECKS))
def test_region_holds_what_its_why_promises(name: str) -> None:
    region = _regions()[name]
    REGION_CHECKS[name](_committed(), region)


@pytest.mark.parametrize("name", sorted(SECTION_CHECKS))
def test_section_holds_what_its_why_promises(name: str) -> None:
    SECTION_CHECKS[name](_committed())


def _family(const: int) -> set[int]:
    return {cliff_catalog.piece_for_suffix(const, s) for s in range(1, 10)} - {None}


def _dat_frame_limit(const: int) -> int:
    """cliff_catalog.frame_count()'s no-install fallback, whatever this run can see."""
    return max(1, int(unit_sprites.graphic_map()[const]["angle_count"]))


def _limit_changes_a_frame(family: int, real: int) -> bool:
    """Whether any shape or join the table holds gets other frames under the .dat limit than the real one."""
    tables = cliff_connectivity.shape_tables()
    allowed = frozenset(s for s in range(1, 10) if cliff_catalog.piece_for_suffix(family, s) is not None)
    for const in _family(family):
        dat = _dat_frame_limit(const)
        for arm_set in tables.rows:
            dirs = sorted(arm_set)
            for legs in itertools.product((True, False), repeat=len(dirs)):
                arms = dict(zip(dirs, legs, strict=True))
                kw = {"allowed_suffixes": allowed, "tables": tables}
                if cliff_connectivity.resolve_shape(arms, frame_limit=dat, **kw) != cliff_connectivity.resolve_shape(
                    arms, frame_limit=real, **kw
                ):
                    return True
            suffix = cliff_catalog.suffixes()[const]
            if cliff_connectivity.rewrite_frames(arm_set, suffix, frame_limit=dat) != cliff_connectivity.rewrite_frames(
                arm_set, suffix, frame_limit=real
            ):
                return True
    return False


def test_shaped_cliff_families_resolve_the_same_frames_with_or_without_an_install() -> None:
    """The generator's SHAPED_CLIFF_FAMILIES criterion: the .dat and .sld limits differ (24 against 25 for
    most), but the shape table caps a limit at 24, so no frame changes. Marble (23 real, 25 declared) does."""
    assert set(SHAPED_FAMILY_SLD_FRAMES) == set(_generator().SHAPED_CLIFF_FAMILIES)
    changed = [f for f, real in SHAPED_FAMILY_SLD_FRAMES.items() if _limit_changes_a_frame(f, real)]
    assert not changed, f"an install changes shaped-stroke frames for families {changed}"
    assert _limit_changes_a_frame(MARBLE_CLIFF, 23), "the check cannot tell a limit that matters"


def test_the_recorded_shaped_family_frame_counts_are_the_installs() -> None:
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")
    real = {f: {cliff_catalog.frame_count(c) for c in _family(f)} for f in SHAPED_FAMILY_SLD_FRAMES}
    assert real == {f: {n} for f, n in SHAPED_FAMILY_SLD_FRAMES.items()}


def test_regions_do_not_overlap_and_keep_their_gap() -> None:
    _generator().check_layout()


def test_only_elevation_regions_are_elevated() -> None:
    m = _committed()
    regions = _regions()
    raised = [
        (x, y) for y in range(MAP) for x in range(MAP)
        if m.elevation(x, y) and not any(r.elevated and r.holds_tile(x, y) for r in regions.values())
    ]
    assert not raised, f"{len(raised)} elevated tiles outside an elevation region, e.g. {raised[:5]}"
