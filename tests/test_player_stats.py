"""GH #5 and GH #145: descape/player_stats.py, the Qt-free counts behind View
mode's per-player readout. The viewer side is tests/test_player_stats_viewer.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import object_catalog, player_stats, unit_kind
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"


_PILES = ("gold_mines", "stone_mines", "berry_bushes", "fish", "other_piles")


def _top_level_tables() -> dict[str, frozenset[int]]:
    """The partition's non-unit buckets as const tables; a unit is in none."""
    sets = player_stats.bucket_sets()
    return {
        "buildings": frozenset(BUILDING_TILE_SPANS) - player_stats.TREBUCHET_CONSTS,
        "trees": TREE_UNIT_IDS,
        "eye_candy": unit_kind.eye_candy_consts(),
        **{name: sets[name] for name in (*_PILES, "animals", "cliffs", "relics")},
    }


def _assert_partition(loaded, label: str) -> None:
    """Each top-level bucket by its own membership test, so an overlap between
    the const tables shows up as a sum larger than the list; then the unit and
    building sub-partitions each sum to their totals."""
    tables = _top_level_tables()
    for player_id, units in enumerate(loaded.unit_manager.units):
        consts = [u.unit_const for u in units]
        independent = {name: sum(c in table for c in consts) for name, table in tables.items()}
        unit_count = sum(not any(c in table for table in tables.values()) for c in consts)
        assert sum(independent.values()) + unit_count == len(units), f"{label} P{player_id}"
        c = player_stats.counts_for(loaded, player_id)
        where = f"{label} P{player_id}"
        assert c.placements == c.partition_total() == len(units), where
        assert c.units == unit_count == c.economy_units + c.military_units + c.other_units, where
        assert c.buildings == independent["buildings"], where
        assert c.buildings == c.economy_buildings + c.military_buildings + c.walls + c.other_buildings, where
        for name in ("trees", "eye_candy", *_PILES, "animals", "cliffs", "relics"):
            assert getattr(c, name) == independent[name], (where, name)
        assert c.piles == sum(getattr(c, name) for name in _PILES), where
        assert max(c.water_units, c.unique_units, c.hero_units) <= c.units, where


def test_the_bucket_const_tables_are_disjoint_and_walls_are_buildings() -> None:
    tables = _top_level_tables()
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not tables[a] & tables[b], (a, b, sorted(tables[a] & tables[b])[:10])
    assert unit_kind.wall_consts() < tables["buildings"]
    sets = player_stats.bucket_sets()
    assert not sets["economy_buildings"] & sets["military_buildings"]
    assert not sets["economy_units"] & sets["military_units"]


def test_trebuchets_count_as_military_units_not_buildings() -> None:
    assert player_stats.TREBUCHET_CONSTS.issubset(BUILDING_TILE_SPANS)
    loaded = _synthetic({1: [42, 331, 12]})
    c = player_stats.counts_for(loaded, 1)
    assert (c.military_units, c.units, c.buildings, c.military_buildings) == (2, 2, 1, 1)


@pytest.mark.parametrize(
    ("bucket", "consts"),
    [
        ("military_buildings", (12, 498, 132, 20)),  # Barracks Age1..Age4
        ("economy_buildings", (109, 71, 141, 142)),  # Town Center, later-age consts
        ("military_buildings", (82, 2418)),  # Castle
        ("economy_buildings", (70, 463, 464, 465)),  # House
        ("military_buildings", (79, 234, 235, 236, 598)),  # watch/guard tower, keep, bombard tower, outpost
        ("economy_buildings", (50, 199, 1897, 1888)),  # farm, fish trap, pasture, pasture post
    ],
)
def test_age_variants_land_in_their_family(bucket: str, consts: tuple[int, ...]) -> None:
    family = player_stats.bucket_sets()[bucket]
    assert set(consts) <= family, sorted(set(consts) - family)
    for const in consts:
        assert const in BUILDING_TILE_SPANS, const


def _synthetic(owned: dict[int, list[int]]):
    units = [[SimpleNamespace(unit_const=c) for c in owned.get(p, [])] for p in range(player_stats.NUM_PLAYER_SLOTS)]
    return SimpleNamespace(unit_manager=SimpleNamespace(units=units))


def test_a_gold_mine_and_two_trees_sum_and_a_villager_adds_nothing() -> None:
    """GOLDM 66 holds 800 gold, a forest tree 100 wood; a villager's only
    storage is population (type 4), which is not a resource."""
    loaded = _synthetic({0: [66, 411, 411], 1: [83]})
    assert player_stats.resources_on_map(loaded) == player_stats.ResourceTotals(food=0, wood=200, gold=800, stone=0)
    assert player_stats.resources_on_map(_synthetic({1: [83]})) == player_stats.ResourceTotals(0, 0, 0, 0)


def test_every_catalog_resources_entry_has_only_the_four_keys_with_positive_values() -> None:
    objects = object_catalog.dat_objects()
    with_resources = {c: e["resources"] for c, e in objects.items() if "resources" in e}
    assert len(with_resources) == 155  # measured 2026-10-07
    for const, resources in with_resources.items():
        assert resources, const
        assert set(resources) <= {"food", "wood", "gold", "stone"}, const
        assert all(isinstance(v, int) and v > 0 for v in resources.values()), const
    assert with_resources[66] == {"gold": 800}
    assert with_resources[102] == {"stone": 350}
    assert with_resources[594] == {"food": 100}


def test_counts_partition_every_player_of_the_units_fixture() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _assert_partition(loaded, UNITS_FIXTURE.name)
    assert [player_stats.counts_for(loaded, p).placements for p in range(3)] == [3, 3, 2]


def test_trigger_counts_are_none_until_a_parse_has_happened() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    assert player_stats.trigger_counts(loaded) is None
    assert player_stats.triggers_without_player(loaded) is None
    assert player_stats.trigger_summary(loaded) is None
    # Asking must not be what pays the parse.
    assert loaded.trigger_read_supported is None


def test_trigger_counts_on_the_trigger_fixture() -> None:
    """tools/gen_trigger_fixture.py sets source_player=1 on the setup and
    armour triggers' effects; the references and variable triggers name no
    player."""
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    assert parse_triggers(loaded) is not None
    counts = player_stats.trigger_counts(loaded)
    assert counts == {0: 0, 1: 2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    assert player_stats.triggers_without_player(loaded) == 2
    assert player_stats.trigger_summary(loaded).total == 4


def test_trigger_counts_are_none_not_zero_when_the_parse_failed() -> None:
    loaded = load_map_and_units(TRIGGER_FIXTURE)
    loaded.trigger_read_supported = False
    assert player_stats.trigger_counts(loaded) is None
    assert player_stats.triggers_without_player(loaded) is None


@pytest.mark.corpus
def test_counts_partition_every_player_of_every_corpus_file(scenario_path) -> None:
    _assert_partition(load_map_and_units(scenario_path), scenario_path.name)


@pytest.mark.corpus
def test_trigger_counts_follow_the_parse_outcome(scenario_path) -> None:
    """None exactly when the Triggers section doesn't parse (the 1.54/3.9
    set), never zeros; otherwise every trigger is counted somewhere."""
    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    summary = player_stats.trigger_summary(loaded)
    if manager is None:
        assert summary is None
        return
    assert summary is not None
    assert summary.total == len(manager.triggers)
    assert summary.without_player <= summary.total
    assert all(0 <= n <= summary.total for n in summary.per_player.values())


def _example(name: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "examples" / name
    if not path.is_file():
        pytest.skip(f"{name} not in examples/")
    return path


@pytest.mark.corpus
def test_an_unparseable_triggers_file_reads_none() -> None:
    loaded = load_map_and_units(_example("2_Joan_coop_1_v0_13.aoe2scenario"))
    assert parse_triggers(loaded) is None
    assert loaded.trigger_read_supported is False
    assert player_stats.trigger_counts(loaded) is None


@pytest.mark.corpus
def test_old_allies_player_five_matches_the_measured_table() -> None:
    path = _example("old-allies-final-v2.aoe2scenario")
    loaded = load_map_and_units(path)
    counts = player_stats.counts_for(loaded, 5)
    # GH #145 moved P5's three trebuchets from Buildings (1042 under GH #5) to Military units.
    assert (counts.placements, counts.buildings, counts.walls) == (1910, 1039, 753)
    assert (counts.units, counts.economy_units, counts.military_units, counts.other_units) == (861, 47, 663, 151)
    assert (counts.economy_buildings, counts.military_buildings, counts.other_buildings) == (167, 109, 10)
    assert (counts.water_units, counts.unique_units, counts.hero_units) == (2, 466, 2)
    assert (counts.trees, counts.animals, counts.eye_candy) == (0, 4, 6)
    parse_triggers(loaded)
    summary = player_stats.trigger_summary(loaded)
    # Source-or-target rule; source-only would give 90.
    assert (summary.per_player[5], summary.total, summary.without_player) == (95, 590, 83)


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("name", "totals"),
    [
        # Measured 2026-09-26 (GH #145's plan); all owners, so old-allies adds P-owned livestock's 1,800 food.
        ("F7_2_Dos Pilas (648).aoe2scenario", player_stats.ResourceTotals(251_725, 2_466_900, 49_600, 23_100)),
        ("old-allies-final-v2.aoe2scenario", player_stats.ResourceTotals(23_665, 612_275, 117_600, 17_150)),
    ],
)
def test_resources_on_map_match_the_measured_totals(name: str, totals) -> None:
    assert player_stats.resources_on_map(load_map_and_units(_example(name))) == totals


# Every const the full examples/ corpus places that lands in Other units or
# Other buildings, measured 2026-10-07. A newly placed class or family fails
# here instead of landing in Other silently.
_OTHER_UNITS = frozenset({
    304, 499, 545, 600, 601, 602, 603, 604, 697, 709, 721, 722, 837, 838, 840, 896, 1150, 1153, 1270,
    1271, 1282, 1284, 1291, 1292, 1308, 1333, 1334, 1335, 1336, 1338, 1376, 1572, 1635, 1774, 1775,
    1776, 1956, 2253, 2254, 2255, 2258, 2353, 2355, 2381, 2423, 2424, 2520, 2605, 2607, 2635,
})
_OTHER_BUILDINGS = frozenset({
    103, 104, 209, 263, 276, 345, 445, 599, 605, 606, 607, 608, 609, 610, 624, 625, 626, 655, 712, 713,
    714, 715, 716, 717, 718, 738, 742, 743, 826, 872, 890, 904, 905, 906, 907, 908, 909, 1097, 1098,
    1099, 1100, 1101, 1196, 1197, 1198, 1199, 1200, 1218, 1220, 1264, 1309, 1310, 1311, 1312, 1313,
    1314, 1368, 1396, 1397, 2421,
})


def _remainders(loaded) -> tuple[set[int], set[int]]:
    """The consts each Other bucket took, found by diffing counts_for() per const."""
    other_units: set[int] = set()
    other_buildings: set[int] = set()
    for units in loaded.unit_manager.units:
        for const in {u.unit_const for u in units}:
            c = player_stats.counts_for(_synthetic({1: [const]}), 1)
            if c.other_units:
                other_units.add(const)
            if c.other_buildings:
                other_buildings.add(const)
    return other_units, other_buildings


@pytest.mark.corpus
def test_the_other_remainders_are_pinned(corpus_files, request) -> None:
    other_units: set[int] = set()
    other_buildings: set[int] = set()
    for path in corpus_files:
        units, buildings = _remainders(load_map_and_units(path))
        other_units |= units
        other_buildings |= buildings
    assert not other_units - _OTHER_UNITS, f"new Other units: {sorted(other_units - _OTHER_UNITS)}"
    assert not other_buildings - _OTHER_BUILDINGS, f"new Other buildings: {sorted(other_buildings - _OTHER_BUILDINGS)}"
    if request.config.getoption("--corpus-full"):
        assert other_units == _OTHER_UNITS
        assert other_buildings == _OTHER_BUILDINGS
