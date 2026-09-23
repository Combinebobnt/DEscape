"""GH #5: descape/player_stats.py, the Qt-free counts behind View mode's
per-player readout. The viewer side is tests/test_player_stats_viewer.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import player_stats, unit_kind
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGER_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"


def _assert_partition(loaded, label: str) -> None:
    """Each bucket by its own membership test, so an overlap between the
    const tables shows up as a sum larger than the list."""
    walls_set = unit_kind.wall_consts()
    eye_candy_set = unit_kind.eye_candy_consts()
    for player_id, units in enumerate(loaded.unit_manager.units):
        consts = [u.unit_const for u in units]
        buildings = sum(c in BUILDING_TILE_SPANS for c in consts)
        trees = sum(c in TREE_UNIT_IDS for c in consts)
        eye_candy = sum(c in eye_candy_set for c in consts)
        walls = sum(c in walls_set for c in consts)
        other = sum(
            c not in BUILDING_TILE_SPANS and c not in TREE_UNIT_IDS and c not in eye_candy_set for c in consts
        )
        assert buildings + trees + eye_candy + other == len(units), f"{label} P{player_id}"
        counts = player_stats.counts_for(loaded, player_id)
        assert counts == player_stats.PlayerCounts(len(units), other, buildings, walls, trees, eye_candy), (
            f"{label} P{player_id}"
        )
        assert counts.walls <= counts.buildings


def test_the_bucket_const_tables_are_disjoint_and_walls_are_buildings() -> None:
    eye_candy = unit_kind.eye_candy_consts()
    buildings = frozenset(BUILDING_TILE_SPANS)
    assert not buildings & TREE_UNIT_IDS
    assert not buildings & eye_candy
    assert not TREE_UNIT_IDS & eye_candy
    assert unit_kind.wall_consts() < buildings


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
    assert (counts.placements, counts.buildings, counts.walls) == (1910, 1042, 753)
    parse_triggers(loaded)
    summary = player_stats.trigger_summary(loaded)
    # Source-or-target rule; source-only would give 90.
    assert (summary.per_player[5], summary.total, summary.without_player) == (95, 590, 83)
