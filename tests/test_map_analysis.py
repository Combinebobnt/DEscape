"""descape/map_analysis.py: the Qt-free Map Analysis engine.

Each check runs against a real fixture mutated in memory, one condition at a
time. The corpus-marked test at the bottom is the false-positive baseline:
shipping campaign scenarios must produce no error-severity findings.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import map_analysis
from descape.fill_tools import connected_regions
from descape.messages_fields import STRING_ID_UNSET
from descape.scenario_io import load_map_and_units, parse_triggers

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNITS_FIXTURE = FIXTURES / "units_120x120.aoe2scenario"
TRIGGERS_FIXTURE = FIXTURES / "triggers_120x120.aoe2scenario"

_WATER = 1  # TerrainId.WATER


def _tile(loaded, x: int, y: int):
    mm = loaded.map_manager
    return mm.terrain[y * mm.map_width + x]


def _unit(loaded, reference_id: int):
    return next(u for units in loaded.unit_manager.units for u in units if u.reference_id == reference_id)


# -- report shape ---------------------------------------------------------------


def test_headline_counts_clean_findings_and_unavailable() -> None:
    finding = map_analysis.Finding("x", "warning")
    report = map_analysis.AnalysisReport(
        (
            map_analysis.CheckResult("a"),
            map_analysis.CheckResult("b", (finding, finding)),
            map_analysis.CheckResult("c", unavailable="nope"),
        )
    )
    assert report.finding_count == 2
    assert report.headline == "Clean on 1 of 3 checks: 2 findings, 1 unavailable"


def test_unavailable_is_not_clean() -> None:
    assert not map_analysis.CheckResult("c", unavailable="nope").is_clean


def test_findings_are_capped_with_a_summary_row() -> None:
    findings = [map_analysis.Finding(str(i), "error") for i in range(map_analysis.MAX_FINDINGS_PER_CHECK + 5)]
    result = map_analysis._capped("x", findings)
    assert len(result.findings) == map_analysis.MAX_FINDINGS_PER_CHECK + 1
    assert result.findings[-1].message == "...and 5 more not listed"


def test_the_untouched_units_fixture_has_no_errors_or_warnings() -> None:
    report = map_analysis.analyze(load_map_and_units(UNITS_FIXTURE))
    severities = {f.severity for r in report.results for f in r.findings}
    assert severities <= {"info"}, report


# -- off-map units --------------------------------------------------------------


def test_off_map_unit_is_reported_and_edge_unit_is_not() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 201).x = -3.0
    _unit(loaded, 203).x = 0.0
    result = map_analysis.check_off_map_units(loaded)
    assert [f.unit_key for f in result.findings] == [(1, 201)]


# -- elevation ------------------------------------------------------------------


@pytest.mark.parametrize("neighbour", [(11, 10), (10, 11), (11, 11), (9, 11)])
def test_elevation_step_of_two_is_an_error_in_all_four_directions(neighbour) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _tile(loaded, *neighbour).elevation = 2
    result = map_analysis.check_elevation(loaded)
    assert result.findings, neighbour
    assert {f.severity for f in result.findings} == {"error"}
    assert all(f"({neighbour[0]}, {neighbour[1]})" in f.message for f in result.findings)


def test_elevation_step_of_one_is_clean() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _tile(loaded, 10, 10).elevation = 1
    assert map_analysis.check_elevation(loaded).is_clean


def test_elevation_outside_range_is_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    for tile in loaded.map_manager.terrain:
        tile.elevation = 15
    _tile(loaded, 3, 4).elevation = 16
    messages = [f.message for f in map_analysis.check_elevation(loaded).findings]
    assert any("(3, 4) elevation 16 is outside" in m for m in messages)


def test_elevation_grid_flat_indexes_a_non_square_map() -> None:
    width, height = 6, 3
    terrain = [SimpleNamespace(elevation=i) for i in range(width * height)]
    fake = SimpleNamespace(map_manager=SimpleNamespace(map_width=width, map_height=height, terrain=terrain))
    grid = map_analysis._elevation_grid(fake)
    assert grid.shape == (3, 6)
    assert grid[2, 5] == 17


# -- triggers -------------------------------------------------------------------


def test_trigger_checks_are_unavailable_not_clean_when_triggers_do_not_parse() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    loaded.trigger_read_supported = False
    for check in (
        map_analysis.check_trigger_display_order,
        map_analysis.check_trigger_references,
        map_analysis.check_unit_references,
    ):
        result = check(loaded)
        assert result.unavailable
        assert not result.is_clean


def test_corrupt_display_order_is_reported() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    manager.trigger_display_order = [0, 0, 2, 3]
    result = map_analysis.check_trigger_display_order(loaded)
    assert [f.severity for f in result.findings] == ["error"]


def test_activate_trigger_pointing_at_a_missing_trigger_is_reported() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    assert map_analysis.check_trigger_references(loaded).is_clean
    manager.triggers[0].new_effect.activate_trigger(trigger_id=99)
    manager.triggers[1].new_effect.activate_trigger(trigger_id=2)
    result = map_analysis.check_trigger_references(loaded)
    assert len(result.findings) == 1
    assert "trigger 99" in result.findings[0].message


def test_unit_reference_to_an_unplaced_unit_is_reported_and_placed_one_is_not() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    manager = parse_triggers(loaded)
    trigger = manager.add_trigger("analysis test")
    trigger.new_condition.destroy_object(unit_object=200)  # placed: player 1's house
    trigger.new_condition.destroy_object(unit_object=99999)
    result = map_analysis.check_unit_references(loaded)
    assert len(result.findings) == 1
    assert "unit id 99999" in result.findings[0].message


# -- garrison links -------------------------------------------------------------


def test_dangling_garrison_link_is_reported_and_a_live_one_is_not() -> None:
    """GH #42: the fixture's villager sits inside its house, which is placed.
    Pointing it at an id nothing carries makes it invisible under the default
    filter, which is why this check exists."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_garrison_links(loaded).is_clean
    _unit(loaded, 203).garrisoned_in_id = 99999
    result = map_analysis.check_garrison_links(loaded)
    assert [f.unit_key for f in result.findings] == [(1, 203)]
    assert "unit id 99999" in result.findings[0].message
    assert result.findings[0].severity == "warning"


def test_a_self_referencing_garrison_link_is_not_dangling() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    unit = _unit(loaded, 203)
    unit.garrisoned_in_id = unit.reference_id
    assert map_analysis.check_garrison_links(loaded).is_clean


# -- players --------------------------------------------------------------------


def test_active_player_with_no_units_is_a_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_players_without_units(loaded).is_clean
    loaded.unit_manager.units[2].clear()
    result = map_analysis.check_players_without_units(loaded)
    assert [f.severity for f in result.findings] == ["warning"]
    assert "Player 2" in result.findings[0].message
    assert "runtime" in result.findings[0].message


# -- stranded units -------------------------------------------------------------


def _flood_with_water(loaded) -> None:
    for tile in loaded.map_manager.terrain:
        tile.terrain_id = _WATER


def _make_land(loaded, x0: int, x1: int, y0: int, y1: int) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            _tile(loaded, x, y).terrain_id = 0


def test_player_unit_alone_on_a_small_island_is_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _flood_with_water(loaded)
    _make_land(loaded, 0, 120, 60, 120)  # big southern land mass for the rest
    _make_land(loaded, 11, 13, 10, 12)  # 4-tile island under player 1's archer at (11.5, 10.5)
    for units in loaded.unit_manager.units[1:]:
        for unit in units:
            if unit.reference_id != 201:
                unit.y = 80.5
    result = map_analysis.check_stranded_units(loaded)
    assert [f.unit_key for f in result.findings] == [(1, 201)]
    assert result.findings[0].severity == "info"
    assert "4-tile" in result.findings[0].message


def test_gaia_alone_on_a_small_island_is_not_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _flood_with_water(loaded)
    _make_land(loaded, 0, 120, 60, 120)
    _make_land(loaded, 5, 7, 5, 7)  # GAIA tree at (5.5, 5.5)
    for units in loaded.unit_manager.units[1:]:
        for unit in units:
            unit.y = 80.5
    assert map_analysis.check_stranded_units(loaded).is_clean


def test_two_large_land_masses_with_different_players_are_not_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _flood_with_water(loaded)
    _make_land(loaded, 0, 50, 0, 50)
    _make_land(loaded, 60, 120, 0, 50)
    for unit in loaded.unit_manager.units[2]:
        unit.x += 50.0
    assert map_analysis.check_stranded_units(loaded).is_clean


def test_non_square_map_skips_reachability_and_says_so() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    loaded.map_is_square = False
    assert map_analysis.check_stranded_units(loaded).unavailable
    assert [f.severity for f in map_analysis.check_map_shape(loaded).findings] == ["info"]


# -- instructions ---------------------------------------------------------------


def _set_instructions(loaded, text: str, string_id: int) -> None:
    messages = loaded._scenario.sections["Messages"].retriever_map
    messages["ascii_instructions"].data = text
    messages["instructions"].data = string_id


def test_instructions_three_valued() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _set_instructions(loaded, "", STRING_ID_UNSET)
    result = map_analysis.check_instructions(loaded)
    assert [f.severity for f in result.findings] == ["info"]

    _set_instructions(loaded, "", 44708)
    result = map_analysis.check_instructions(loaded)
    assert result.unavailable and not result.findings

    _set_instructions(loaded, "Defend the castle.", STRING_ID_UNSET)
    assert map_analysis.check_instructions(loaded).is_clean


# -- fill_tools.connected_regions -------------------------------------------------


def test_connected_regions_splits_on_the_predicate_without_row_wraparound() -> None:
    rows = ["LLWLL", "LLWLL", "WWWWL"]
    width = len(rows[0])
    terrain = [SimpleNamespace(ch=ch) for row in rows for ch in row]
    mm = SimpleNamespace(map_width=width, map_height=len(rows), terrain=terrain)
    regions = connected_regions(mm, lambda i: terrain[i].ch == "L")
    assert sorted(sorted(r) for r in regions) == [[0, 1, 5, 6], [3, 4, 8, 9, 14]]


# -- map markers ------------------------------------------------------------------


def _marker_report(*groups):
    return map_analysis.AnalysisReport(
        tuple(map_analysis.CheckResult(f"check {i}", tuple(findings)) for i, findings in enumerate(groups))
    )


def test_marker_anchors_dedupe_per_tile_with_worst_severity_and_a_count() -> None:
    F = map_analysis.Finding
    report = _marker_report(
        [F("a", "info", tile=(3, 4)), F("b", "error", tile=(5, 5))],
        [F("c", "warning", tile=(3, 4)), F("d", "info", tile=(3, 4)), F("e", "warning", tile=(5, 5))],
    )
    assert map_analysis.marker_anchors(report, 10, 10) == [((3, 4), "warning", 3), ((5, 5), "error", 2)]


def test_marker_anchors_clamp_to_the_map_before_deduping() -> None:
    F = map_analysis.Finding
    report = _marker_report([F("a", "warning", tile=(-2, 30)), F("b", "info", tile=(0, 99)), F("c", "info", tile=(12, 3))])
    assert map_analysis.marker_anchors(report, 10, 20) == [((0, 19), "warning", 2), ((9, 3), "info", 1)]


def test_unlocated_and_unit_only_findings_get_no_marker() -> None:
    F = map_analysis.Finding
    report = _marker_report(
        [F("trigger", "warning"), F("off-map unit", "warning", unit_key=(1, 201)), F("...and 5 more", "error")],
        [F("stranded", "info", tile=(2, 2), unit_key=(1, 7))],
    )
    assert map_analysis.marker_anchors(report, 10, 10) == [((2, 2), "info", 1)]
    assert map_analysis.marker_tile(F("x", "info", unit_key=(1, 201)), 10, 10) is None


def test_marker_anchors_from_a_real_elevation_violation() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _tile(loaded, 10, 10).elevation = 2
    report = map_analysis.analyze(loaded)
    mm = loaded.map_manager
    anchors = map_analysis.marker_anchors(report, mm.map_width, mm.map_height)
    errors = {tile: count for tile, severity, count in anchors if severity == "error"}
    # Each finding sits on its pair's "a" tile: (10, 10) for four of the eight pairs, a neighbour for the rest.
    assert errors[(10, 10)] == 4
    assert sum(errors.values()) == len(map_analysis.check_elevation(loaded).findings) == 8


# -- corpus ---------------------------------------------------------------------


@pytest.mark.corpus
def test_shipping_scenarios_have_no_error_findings(scenario_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    report = map_analysis.analyze(loaded)
    errors = [f for r in report.results for f in r.findings if f.severity == "error"]
    assert errors == []
