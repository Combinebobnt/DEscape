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


# -- off-grid buildings and units -----------------------------------------------

_HOUSE, _BARRACKS, _WATCH_TOWER, _STONE_GATE_4X1, _TREBUCHET = 70, 12, 79, 64, 42
_DEER, _DECORATION, _TREE = 65, 1033, 349


def _place(loaded, reference_id: int, const: int, x: float, y: float) -> None:
    unit = _unit(loaded, reference_id)
    unit.unit_const, unit.x, unit.y = const, x, y


def _building_findings(loaded) -> dict[tuple[int, int], tuple[str, str]]:
    return {f.unit_key: (f.severity, f.message) for f in map_analysis.check_off_grid_buildings(loaded).findings}


def test_the_fixture_house_is_the_one_untouched_off_grid_building() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    result = map_analysis.check_off_grid_buildings(loaded)
    assert [(f.unit_key, f.severity, f.tile) for f in result.findings] == [((1, 200), "info", (10, 10))]
    assert result.findings[0].message == (
        "Player 1 building 70 (id 200) at (10.5, 10.5) is half a tile off the grid "
        "for a 2x2 footprint (x and y should end in .0)"
    )


@pytest.mark.parametrize(
    ("const", "x", "y"),
    [(_HOUSE, 10.0, 12.0), (_BARRACKS, 10.5, 12.5), (_WATCH_TOWER, 10.5, 12.5), (_STONE_GATE_4X1, 10.0, 12.5)],
)
def test_on_grid_buildings_are_clean(const, x, y) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 200, const, x, y)
    assert map_analysis.check_off_grid_buildings(loaded).is_clean


@pytest.mark.parametrize(
    ("const", "x", "y", "hint"),
    [
        (_BARRACKS, 10.0, 12.5, "for a 3x3 footprint (x should end in .5)"),
        (_WATCH_TOWER, 10.0, 12.0, "for a 1x1 footprint (x and y should end in .5)"),
        (_HOUSE, 10.0, 12.5, "for a 2x2 footprint (y should end in .0)"),
        (_STONE_GATE_4X1, 10.5, 12.0, "for a 4x1 footprint (x should end in .0, y in .5)"),
    ],
)
def test_wrong_parity_building_is_info(const, x, y, hint) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 200, const, x, y)
    severity, message = _building_findings(loaded)[(1, 200)]
    assert severity == "info"
    assert message.endswith(hint)


def test_building_off_the_half_tile_grid_is_one_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 200, _HOUSE, 10.3, 10.5)  # off the half grid on x, wrong parity on y
    result = map_analysis.check_off_grid_buildings(loaded)
    assert [(f.unit_key, f.severity) for f in result.findings] == [((1, 200), "warning")]
    assert "is off the half-tile grid; no game-made building sits there" in result.findings[0].message


def test_a_later_warning_survives_the_cap_ahead_of_info(monkeypatch) -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 201, _BARRACKS, 20.3, 20.5)
    monkeypatch.setattr(map_analysis, "MAX_FINDINGS_PER_CHECK", 1)
    result = map_analysis.check_off_grid_buildings(loaded)
    assert [(f.unit_key, f.severity) for f in result.findings] == [((1, 201), "warning"), (None, "warning")]


def test_trebuchet_and_off_map_buildings_are_not_building_findings() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 200, _TREBUCHET, 10.0, 10.0)
    _place(loaded, 102, _HOUSE, -3.3, 5.5)  # GAIA wall turned into an off-map, off-grid house
    assert map_analysis.check_off_grid_buildings(loaded).is_clean


def _unit_keys(check, loaded) -> list[tuple[int, int]]:
    return [f.unit_key for f in check(loaded).findings]


def test_the_untouched_fixture_has_no_off_grid_units() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_off_grid_player_units(loaded).is_clean
    assert map_analysis.check_off_grid_gaia_units(loaded).is_clean


def test_off_centre_player_unit_lands_only_in_the_player_check() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 300).y = 30.1
    result = map_analysis.check_off_grid_player_units(loaded)
    assert [(f.unit_key, f.severity, f.tile) for f in result.findings] == [((2, 300), "info", (20, 30))]
    assert result.findings[0].message == "Player 2 unit 4 (id 300) at (20.5, 30.1) is not at its tile's centre"
    assert map_analysis.check_off_grid_gaia_units(loaded).is_clean


def test_off_centre_gaia_animal_lands_only_in_the_gaia_check() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 100, _DEER, 5.0, 5.5)
    result = map_analysis.check_off_grid_gaia_units(loaded)
    assert [(f.unit_key, f.severity) for f in result.findings] == [((0, 100), "info")]
    assert result.findings[0].message.startswith("GAIA unit 65 (id 100) at (5, 5.5)")
    assert map_analysis.check_off_grid_player_units(loaded).is_clean


def test_centred_garrisoned_off_map_and_non_unit_objects_are_not_off_grid_units() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 100, _DEER, 5.5, 5.5)  # centred
    _place(loaded, 101, _DECORATION, 6.2, 5.7)
    _place(loaded, 102, _TREE, 7.1, 5.9)
    _unit(loaded, 203).x = 12.9  # garrisoned in house 200
    _unit(loaded, 201).x = -3.2  # off-map
    assert map_analysis.check_off_grid_player_units(loaded).is_clean
    assert map_analysis.check_off_grid_gaia_units(loaded).is_clean


# -- GAIA-only objects owned by a player ------------------------------------------

_FORAGE_BUSH, _FLARE4 = 59, 697


def test_player_owned_forage_bush_is_one_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 300, _FORAGE_BUSH, 20.5, 30.5)
    result = map_analysis.check_gaia_only_ownership(loaded)
    assert [(f.unit_key, f.severity, f.tile) for f in result.findings] == [((2, 300), "warning", (20, 30))]
    assert result.findings[0].message == (
        "Player 2 object 59 (id 300) at (20.5, 30.5) is GAIA-only; game-made files only give those to GAIA"
    )


def test_player_owned_tree_is_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 300, _TREE, 20.5, 30.5)
    assert _unit_keys(map_analysis.check_gaia_only_ownership, loaded) == [(2, 300)]


def test_player_owned_macaw_is_reported() -> None:
    """GH #32: AoE2's wild animals are GAIA-only too."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 300, 816, 20.5, 30.5)
    assert _unit_keys(map_analysis.check_gaia_only_ownership, loaded) == [(2, 300)]


def test_gaia_owned_forage_bush_is_clean() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 100, _FORAGE_BUSH, 5.5, 5.5)
    assert map_analysis.check_gaia_only_ownership(loaded).is_clean


def test_the_untouched_fixture_and_player_owned_eye_candy_are_clean() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_gaia_only_ownership(loaded).is_clean
    _place(loaded, 300, _FLARE4, 20.5, 30.5)
    assert map_analysis.check_gaia_only_ownership(loaded).is_clean


def test_off_map_player_owned_forage_bush_is_still_reported() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 300, _FORAGE_BUSH, -3.5, 30.5)
    assert _unit_keys(map_analysis.check_gaia_only_ownership, loaded) == [(2, 300)]


# -- objects in invalid positions (GH #173) ----------------------------------------

_FISHING_SHIP, _BRIDGE_MIDDLE, _MAP_REVEALER, _STONE_WALL = 13, 606, 837, 117
_WATER_DEEP, _BEACH = 22, 2


@pytest.fixture
def _no_install_env(monkeypatch):
    """Messages are asserted by name, and AOE2DE_INSTALL_PATH would swap in the install's."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _rows(check, loaded) -> list[tuple[tuple[int, int] | None, str, tuple[int, int] | None]]:
    return [(f.unit_key, f.severity, f.tile) for f in check(loaded).findings]


def _set_terrain(loaded, terrain_id: int, tiles) -> None:
    for x, y in tiles:
        _tile(loaded, x, y).terrain_id = terrain_id


@pytest.mark.usefixtures("_no_install_env")
def test_a_unit_on_deep_water_is_one_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean
    _set_terrain(loaded, _WATER_DEEP, [(20, 20)])
    result = map_analysis.check_units_on_wrong_terrain(loaded)
    assert _rows(map_analysis.check_units_on_wrong_terrain, loaded) == [((2, 300), "warning", (20, 20))]
    assert result.findings[0].message == "Player 2 unit 4 (id 300) at (20.5, 20.5): Archer can't stand on Water Deep"


@pytest.mark.usefixtures("_no_install_env")
def test_a_fishing_ship_is_reported_on_grass_and_clean_on_water() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 300, _FISHING_SHIP, 20.5, 20.5)
    result = map_analysis.check_units_on_wrong_terrain(loaded)
    assert [f.unit_key for f in result.findings] == [(2, 300)]
    assert result.findings[0].message.endswith(": Fishing Ship can't stand on Grass 1")
    _set_terrain(loaded, _WATER_DEEP, [(20, 20)])
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean


def test_garrisoned_and_off_map_units_are_not_on_wrong_terrain() -> None:
    """Villager 203 is inside House 200 at (12.5, 10.5): it stands wherever its host does."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _set_terrain(loaded, _WATER_DEEP, [(12, 10)])
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean
    _flood_with_water(loaded)
    _unit(loaded, 300).x = -3.5
    # Archer 201 stands on House 200's foundation, which a land unit passes.
    assert [f.unit_key for f in map_analysis.check_units_on_wrong_terrain(loaded).findings] == [(2, 301)]


def test_a_unit_on_a_bridge_over_shallow_water_is_clean_until_the_bridge_goes() -> None:
    """The bridge at (21, 20.5) covers tiles 20..21 x 19..21: Archer 300 and Villager 301 stand on it."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _set_terrain(loaded, _WATER, [(x, y) for x in range(19, 23) for y in range(18, 23)])
    _place(loaded, 102, _BRIDGE_MIDDLE, 21.0, 20.5)
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean
    _unit(loaded, 102).unit_const = _STONE_WALL
    assert [f.unit_key for f in map_analysis.check_units_on_wrong_terrain(loaded).findings] == [(2, 300), (2, 301)]


def test_a_house_on_water_is_not_a_unit_finding() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _set_terrain(loaded, _WATER_DEEP, [(10, 11)])  # under House 200, not under Archer 201 at (11, 10)
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean
    assert [f.unit_key for f in map_analysis.check_buildings_on_wrong_terrain(loaded).findings] == [(1, 200)]


@pytest.mark.usefixtures("_no_install_env")
def test_a_house_with_one_beach_footprint_tile_is_one_info() -> None:
    """House 200 at (10.5, 10.5) is drawn on tiles 10..11 x 10..11."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_buildings_on_wrong_terrain(loaded).is_clean
    _set_terrain(loaded, _BEACH, [(10, 11)])
    result = map_analysis.check_buildings_on_wrong_terrain(loaded)
    assert _rows(map_analysis.check_buildings_on_wrong_terrain, loaded) == [((1, 200), "info", (10, 10))]
    assert result.findings[0].message == "Player 1 building 70 (id 200) at (10.5, 10.5): House can't stand on Beach"
    assert map_analysis.check_units_on_wrong_terrain(loaded).is_clean


@pytest.mark.usefixtures("_no_install_env")
def test_the_fixture_archer_overlapping_its_house_is_the_one_overlap() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    result = map_analysis.check_overlapping_units(loaded)
    assert _rows(map_analysis.check_overlapping_units, loaded) == [((1, 201), "info", (11, 10))]
    assert result.findings[0].message == (
        "Player 1 unit 4 (id 201) at (11.5, 10.5) overlaps House (Player 1) at (10.5, 10.5)"
    )


def test_two_units_on_one_point_give_one_finding_each() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 301).x = 20.5
    keys = [f.unit_key for f in map_analysis.check_overlapping_units(loaded).findings]
    assert keys == [(1, 201), (2, 300), (2, 301)]


def test_a_unit_on_a_map_revealer_does_not_overlap_it() -> None:
    """Map Revealers are invisible: solid for Place Unit, not a blocker here."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _place(loaded, 102, _MAP_REVEALER, 20.5, 20.5)
    assert [f.unit_key for f in map_analysis.check_overlapping_units(loaded).findings] == [(1, 201)]


def test_a_garrisoned_unit_neither_overlaps_nor_blocks() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 203).x, _unit(loaded, 203).y = 20.5, 20.5  # still garrisoned in House 200
    assert [f.unit_key for f in map_analysis.check_overlapping_units(loaded).findings] == [(1, 201)]


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


# -- GH #230: train and research buttons -------------------------------------------

_SPANISH = 14


@pytest.fixture
def spanish_p1(monkeypatch):
    """Player 1 is Spanish; the fixture's other players stay random."""
    from descape import civ_availability

    real = civ_availability.player_civ_value
    monkeypatch.setattr(
        civ_availability, "player_civ_value", lambda loaded, player: _SPANISH if player == 1 else real(loaded, player)
    )


def _location_findings(*effects, player: int = 1, extra=None):
    """Findings for one new trigger of Change Train Location (mover, building,
    value) effects, or Change Research Location for a ("tech", id, building, value)."""
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    trigger = manager.add_trigger("gh230")
    for effect in effects:
        if effect[0] == "tech":
            _tag, tech, building, value = effect
            trigger.new_effect.change_technology_location(
                technology=tech, source_player=player, object_list_unit_id_2=building, button_location=value
            )
        else:
            mover, building, value = effect
            trigger.new_effect.change_train_location(
                object_list_unit_id=mover, source_player=player, object_list_unit_id_2=building, button_location=value
            )
    if extra is not None:
        extra(trigger)
    result = map_analysis.check_location_buttons(loaded)
    return [(f.severity, f.message) for f in result.findings]


def test_location_buttons_is_unavailable_when_triggers_do_not_parse() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    loaded.trigger_read_supported = False
    assert map_analysis.check_location_buttons(loaded).unavailable


def test_the_fixture_has_no_location_findings() -> None:
    assert map_analysis.check_location_buttons(load_map_and_units(TRIGGERS_FIXTURE)).is_clean


@pytest.mark.parametrize(
    ("effect", "severity", "text"),
    [
        ((74, 12, 0), "info", "button slot 0 gives no button"),
        ((74, 12, 17), "warning", "button slot 17 is no button"),
        ((84, 83, 1), "warning", "trained at Villager 83 moves nothing"),
        ((82, 118, 14), "warning", "the build menu's More button"),
        ((448, 12, 2), "warning", "(the lower id shows)"),
        ((38, 12, 6), "warning", "so it can't be researched"),
        (("tech", 22, 12, 1), "warning", "a unit in the slot"),
    ],
)
def test_each_location_reason_is_reported(effect, severity, text) -> None:
    findings = _location_findings(effect)
    assert len(findings) == 1
    assert findings[0][0] == severity and text in findings[0][1]
    assert findings[0][1].startswith('Trigger 4 "gh230": ')


def test_a_unit_the_civ_lacks_is_a_warning_unless_the_file_enables_it(spanish_p1) -> None:
    mule_cart = (1808, 12, 8)
    [(severity, message)] = _location_findings(mule_cart)
    assert severity == "warning" and "Spanish can't train" in message and "Enable/Disable Object" in message
    enable = lambda t: t.new_effect.enable_disable_object(object_list_unit_id=1808, source_player=1, enabled=1)  # noqa: E731
    assert _location_findings(mule_cart, extra=enable) == []


def test_a_random_civ_skips_the_civ_reasons() -> None:
    assert _location_findings((1808, 12, 8)) == []


def test_a_building_the_civ_lacks_is_a_warning_unless_one_is_placed(spanish_p1, monkeypatch) -> None:
    krepost = (74, 1251, 1)
    [(_severity, message)] = _location_findings(krepost)
    assert "Spanish can't build" in message
    monkeypatch.setattr(map_analysis, "_all_units", lambda _loaded: [(1, SimpleNamespace(unit_const=1251))])
    assert _location_findings(krepost) == []


def test_a_unit_in_the_disables_list_is_a_warning(monkeypatch) -> None:
    from descape import disables_fields

    monkeypatch.setattr(
        disables_fields, "current_ids", lambda _l, category, player: (38,) if (category, player) == ("units", 1) else ()
    )
    [(_severity, message)] = _location_findings((38, 101, 7))
    assert "in player 1's Disables list" in message


def test_moving_or_disabling_the_blocker_keeps_it_quiet() -> None:
    """Militia at Archery Range button 3 sits under the Cavalry Archer (39)."""
    militia = (74, 87, 3)
    assert "(the lower id shows)" in _location_findings(militia)[0][1]
    assert _location_findings(militia, (39, 82, 35)) == [], "the Cavalry Archer moves to an empty Castle slot"
    disable = lambda t: t.new_effect.enable_disable_object(object_list_unit_id=39, source_player=1, enabled=0)  # noqa: E731
    assert _location_findings(militia, extra=disable) == []


def test_a_blocker_the_civ_lacks_does_not_hide_the_button(spanish_p1) -> None:
    """Cavalry Archer 39 sits at Archery Range button 3; Spanish have none."""
    militia = (74, 87, 3)
    assert _location_findings(militia) == []
    assert "(the lower id shows)" in _location_findings(militia, player=2)[0][1]


def test_a_tech_the_starting_age_already_has_is_a_warning(monkeypatch) -> None:
    from descape import civ_availability

    monkeypatch.setattr(civ_availability, "player_start_age", lambda _loaded, player: 5 if player == 1 else None)
    [(severity, message)] = _location_findings(("tech", 22, 12, 8))
    assert severity == "warning"
    assert message.endswith("player 1 starts in the Imperial Age, which already has Loom, so it gets no button")
    assert _location_findings(("tech", 47, 12, 8)) == [], "Chemistry is an Imperial Age tech"
    [(_severity, covers)] = _location_findings((38, 12, 7))
    assert "covers Halberdier at" in covers, "the Pikeman upgrade there is already researched; Halberdier is not"


def test_a_clean_tech_edit_looks_up_no_name(monkeypatch) -> None:
    """tech_name() re-reads the settings, so only a finding pays for it (trigger-heavy maps)."""
    from descape import object_catalog

    calls = []
    real = object_catalog.tech_name
    monkeypatch.setattr(object_catalog, "tech_name", lambda tech: calls.append(tech) or real(tech))
    assert _location_findings(("tech", 47, 12, 8)) == []
    assert calls == []
    assert _location_findings(("tech", 22, 12, 1)) and calls == [22]


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
    _tile(loaded, 40, 40).elevation = 2  # away from the fixture House's off-grid info finding at (10, 10)
    report = map_analysis.analyze(loaded)
    mm = loaded.map_manager
    anchors = map_analysis.marker_anchors(report, mm.map_width, mm.map_height)
    errors = {tile: count for tile, severity, count in anchors if severity == "error"}
    # Each finding sits on its pair's "a" tile: (40, 40) for four of the eight pairs, a neighbour for the rest.
    assert errors[(40, 40)] == 4
    assert sum(errors.values()) == len(map_analysis.check_elevation(loaded).findings) == 8


# -- corpus ---------------------------------------------------------------------
# Real-file false-positive baselines: they stay corpus, since the synthetic test map
# (conftest.TEST_MAP_PATH) is no baseline for what real maps hold.


@pytest.mark.corpus
def test_shipping_scenarios_have_no_error_findings(scenario_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    report = map_analysis.analyze(loaded)
    errors = [f for r in report.results for f in r.findings if f.severity == "error"]
    assert errors == []


@pytest.mark.corpus
def test_shipping_scenarios_have_no_off_grid_building_warning(scenario_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    findings = map_analysis.check_off_grid_buildings(loaded).findings
    assert [f.message for f in findings if f.severity == "warning"] == []


@pytest.mark.corpus
def test_shipping_scenarios_have_no_player_owned_gaia_only_object(scenario_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    assert [f.message for f in map_analysis.check_gaia_only_ownership(loaded).findings] == []


@pytest.mark.corpus
def test_shipping_scenarios_have_no_unit_on_wrong_terrain(scenario_path: Path) -> None:
    """GH #173: the WARNING row. 0 of 21 examples/ files on 2026-10-13."""
    loaded = load_map_and_units(scenario_path)
    assert [f.message for f in map_analysis.check_units_on_wrong_terrain(loaded).findings] == []
