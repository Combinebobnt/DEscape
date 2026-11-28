"""descape/map_analysis.py: the Qt-free Map Analysis engine.

Each check runs against a real fixture mutated in memory, one condition at a
time. The corpus-marked test at the bottom is the false-positive baseline:
shipping campaign scenarios must produce no error-severity findings.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import map_analysis, map_analysis_types
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


def test_an_uncounted_check_reads_clean_unless_unavailable() -> None:
    finding = map_analysis.Finding("x", "info")
    report = map_analysis.AnalysisReport(
        (
            map_analysis.CheckResult("a", (finding,), counted=False),
            map_analysis.CheckResult("b", unavailable="nope", counted=False),
        )
    )
    assert report.finding_count == 0
    assert report.headline == "Clean on 1 of 2 checks: 0 findings, 1 unavailable"


def test_unavailable_is_not_clean() -> None:
    assert not map_analysis.CheckResult("c", unavailable="nope").is_clean


def test_findings_are_capped_with_a_summary_row() -> None:
    findings = [map_analysis.Finding(str(i), "error") for i in range(map_analysis.MAX_FINDINGS_PER_CHECK + 5)]
    result = map_analysis._capped("x", findings)
    assert len(result.findings) == map_analysis.MAX_FINDINGS_PER_CHECK + 1
    assert result.findings[-1].message == "...and 5 more not listed"


def test_the_untouched_units_fixture_has_no_errors_or_warnings_but_its_house_garrison() -> None:
    """The fixture garrisons Villager 203 in House 200, which no game-made file does."""
    report = map_analysis.analyze(load_map_and_units(UNITS_FIXTURE))
    warnings = [(r.label, f.unit_key) for r in report.results for f in r.findings if f.severity != "info"]
    assert warnings == [("Garrisons the host can't hold", (1, 203))], report


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
    monkeypatch.setattr(map_analysis_types, "MAX_FINDINGS_PER_CHECK", 1)
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


@pytest.mark.parametrize(
    ("make", "counts"),
    [
        (lambda t: t.new_effect.create_object(object_list_unit_id=1251, source_player=1), True),
        (lambda t: t.new_effect.replace_object(object_list_unit_id=82, source_player=1, object_list_unit_id_2=1251), True),
        (lambda t: t.new_effect.create_object(object_list_unit_id=82, source_player=1), False),
        (lambda t: t.new_effect.replace_object(object_list_unit_id=1251, source_player=1, object_list_unit_id_2=82), False),
    ],
    ids=["create", "replace-into", "create-other", "replace-away"],
)
def test_a_building_a_trigger_makes_counts_as_present(spanish_p1, make, counts) -> None:
    """TASK-458: Create Object and Replace Object (its second object) make the Krepost the Spanish lack."""
    findings = _location_findings((74, 1251, 1), extra=make)
    if counts:
        assert findings == []
    else:
        [(_severity, message)] = findings
        assert "Spanish can't build" in message


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


def test_missing_instructions_alone_leave_the_headline_clean() -> None:
    """TASK-096: shown in the dialog, but an info-only file reads as clean on every check."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _set_instructions(loaded, "", STRING_ID_UNSET)
    clean = map_analysis.analyze(loaded)
    _set_instructions(loaded, "Defend the castle.", STRING_ID_UNSET)
    with_text = map_analysis.analyze(loaded)
    instructions = next(r for r in clean.results if r.label == "Scenario instructions")
    assert len(instructions.findings) == 1, "still shown"
    assert clean.finding_count == with_text.finding_count
    assert clean.headline == with_text.headline


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


# -- TASK-098: unit ids, unknown ids, garrison fit, undefined players -------------------


def test_unit_ids_are_clean_on_the_fixture_and_a_shared_id_is_one_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_unit_ids(loaded).is_clean
    _unit(loaded, 301).reference_id = 300
    findings = map_analysis.check_unit_ids(loaded).findings
    assert [(f.severity, f.unit_key) for f in findings] == [("warning", (2, 300))]
    assert "held by 2 units" in findings[0].message


def test_an_id_at_the_file_counter_is_a_warning_unless_the_live_counter_covers_it() -> None:
    """The units fixture stores next_unit_id_to_place 400; the edit model's
    counter replaces it on Save once a unit was added."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 301).reference_id = 400
    findings = map_analysis.check_unit_ids(loaded).findings
    assert [(f.severity, f.unit_key) for f in findings] == [("warning", (2, 400))]
    assert "(400)" in findings[0].message
    assert map_analysis.check_unit_ids(loaded, next_unit_id=401).is_clean
    report = map_analysis.analyze(loaded, next_unit_id=401)
    assert next(r for r in report.results if r.label == "Unit ids").is_clean


def test_an_unknown_object_const_is_a_warning() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_unknown_objects(loaded).is_clean
    _unit(loaded, 201).unit_const = 59999
    findings = map_analysis.check_unknown_objects(loaded).findings
    assert [(f.severity, f.unit_key) for f in findings] == [("warning", (1, 201))]


def test_an_unknown_terrain_id_is_one_warning_per_id_with_its_count() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_unknown_terrains(loaded).is_clean
    _tile(loaded, 7, 3).terrain_id = 250
    _tile(loaded, 2, 9).terrain_id = 250
    findings = map_analysis.check_unknown_terrains(loaded).findings
    assert [(f.severity, f.tile) for f in findings] == [("warning", (7, 3))]
    assert "Terrain id 250" in findings[0].message and "2 tiles" in findings[0].message


def test_unknown_terrains_is_unavailable_without_a_terrain_table(monkeypatch) -> None:
    from descape import object_catalog

    monkeypatch.setattr(object_catalog, "dat_terrains", dict)
    result = map_analysis.check_unknown_terrains(load_map_and_units(UNITS_FIXTURE))
    assert result.unavailable and not result.is_clean


_TOWN_CENTER = 109


def test_a_villager_in_a_house_is_a_wrong_type_warning_and_in_a_town_center_is_not() -> None:
    """The units fixture garrisons Villager 203 in House 200, which holds nothing."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    findings = map_analysis.check_garrison_fit(loaded).findings
    assert [(f.severity, f.unit_key, f.tile) for f in findings] == [("warning", (1, 203), (10, 10))]
    assert "can't hold it" in findings[0].message
    _unit(loaded, 200).unit_const = _TOWN_CENTER
    assert map_analysis.check_garrison_fit(loaded).is_clean


def test_a_host_over_its_base_capacity_is_one_warning_and_a_dangling_host_is_not_this_check(monkeypatch) -> None:
    from descape import garrison

    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 200).unit_const = _WATCH_TOWER
    _unit(loaded, 201).garrisoned_in_id = 200
    assert map_analysis.check_garrison_fit(loaded).is_clean
    monkeypatch.setattr(garrison, "capacity", lambda _host: 1)
    findings = map_analysis.check_garrison_fit(loaded).findings
    assert [(f.severity, f.unit_key) for f in findings] == [("warning", (1, 200))]
    assert "holds 2 units, more than its base capacity of 1" in findings[0].message
    _unit(loaded, 201).garrisoned_in_id = 99999
    _unit(loaded, 203).garrisoned_in_id = 99999
    assert map_analysis.check_garrison_fit(loaded).is_clean


def test_units_of_an_undefined_player_are_one_warning_per_player() -> None:
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert map_analysis.check_undefined_player_units(loaded).is_clean
    units = loaded.unit_manager.units
    units[4].extend(units[2])
    findings = map_analysis.check_undefined_player_units(loaded).findings
    assert [(f.severity, f.unit_key) for f in findings] == [("warning", (4, 300))]
    assert "Player 4" in findings[0].message and "2 units" in findings[0].message


# -- TASK-098: trigger coordinates, created objects, completeness, players, starts -------

_NEW_TRIGGER_CHECKS = (
    map_analysis.check_trigger_coordinates,
    map_analysis.check_created_object_terrain,
    map_analysis.check_incomplete_triggers,
    map_analysis.check_trigger_players,
    map_analysis.check_never_started_triggers,
)


def _new_trigger(name: str = "analysis test"):
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    manager = parse_triggers(loaded)
    return loaded, manager, manager.add_trigger(name)


def _new_trigger_findings(check, loaded):
    """The findings about _new_trigger()'s trigger, list index 4."""
    return [f for f in check(loaded).findings if f.trigger_index == 4]


def test_the_new_trigger_checks_are_unavailable_when_triggers_do_not_parse() -> None:
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    loaded.trigger_read_supported = False
    for check in _NEW_TRIGGER_CHECKS:
        result = check(loaded)
        assert result.unavailable == map_analysis_types._TRIGGERS_UNAVAILABLE, check
        assert not result.is_clean


def test_the_vocabulary_checks_are_unavailable_without_a_vocabulary(monkeypatch) -> None:
    from descape import library_compat

    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    monkeypatch.setattr(library_compat, "vocabulary_is_available", lambda _version: False)
    for check in _NEW_TRIGGER_CHECKS[:4]:
        assert check(loaded).unavailable == "no trigger vocabulary for this scenario version", check
    assert not map_analysis.check_never_started_triggers(loaded).unavailable


def test_the_fixture_triggers_give_only_their_two_known_findings() -> None:
    """Trigger 3 carries a half-set area and a Patrol with no location on
    purpose (the status-colour tests use it); trigger 2 is disabled."""
    loaded = load_map_and_units(TRIGGERS_FIXTURE)
    found = {check.__name__: [f.trigger_index for f in check(loaded).findings] for check in _NEW_TRIGGER_CHECKS}
    assert found == {
        "check_trigger_coordinates": [],
        "check_created_object_terrain": [],
        "check_incomplete_triggers": [3],
        "check_trigger_players": [],
        "check_never_started_triggers": [2],
    }


def test_trigger_coordinates_sort_warnings_ahead_of_an_edge_area() -> None:
    """120x120 fixture: an area reaching past the edge is INFO, one wholly
    outside and an off-map location are WARNINGs, all three in the new trigger."""
    loaded, _manager, trigger = _new_trigger()
    trigger.new_condition.objects_in_area(area_x1=110, area_y1=100, area_x2=125, area_y2=105)
    trigger.new_condition.objects_in_area(area_x1=121, area_y1=0, area_x2=130, area_y2=5)
    trigger.new_effect.create_object(object_list_unit_id=4, source_player=1, location_x=50, location_y=140)
    trigger.new_condition.objects_in_area(area_x1=0, area_y1=0, area_x2=119, area_y2=119)
    findings = map_analysis.check_trigger_coordinates(loaded).findings
    assert [f.severity for f in findings] == ["warning", "warning", "info"]
    assert "wholly outside" in findings[0].message and findings[0].tile == (119, 0)
    assert "location (50, 140)" in findings[1].message and findings[1].tile == (50, 119)
    assert "reaches past the edge" in findings[2].message and findings[2].tile == (110, 100)
    assert {f.trigger_index for f in findings} == {4}
    assert findings[1].trigger_entry == ("effect", 0)


def test_a_created_unit_on_water_is_a_warning_and_off_map_is_left_to_coordinates() -> None:
    loaded, _manager, trigger = _new_trigger()
    _tile(loaded, 30, 30).terrain_id = _WATER
    trigger.new_effect.create_object(object_list_unit_id=4, source_player=1, location_x=30, location_y=30)
    trigger.new_effect.create_object(object_list_unit_id=4, source_player=1, location_x=31, location_y=31)
    trigger.new_effect.create_object(object_list_unit_id=4, source_player=1, location_x=130, location_y=30)
    findings = map_analysis.check_created_object_terrain(loaded).findings
    assert [(f.severity, f.tile, f.trigger_entry) for f in findings] == [("warning", (30, 30), ("effect", 0))]


def test_an_incomplete_trigger_is_one_info_and_dangling_ids_stay_with_their_checks() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.new_condition.destroy_object(unit_object=99999)
    trigger.new_condition.timer(timer=5)
    findings = _new_trigger_findings(map_analysis.check_incomplete_triggers, loaded)
    assert [f.severity for f in findings] == ["info"]
    assert findings[0].message.endswith("has no effects")
    trigger.new_effect.activate_trigger(trigger_id=99)
    assert _new_trigger_findings(map_analysis.check_incomplete_triggers, loaded) == []


def test_an_entry_type_the_vocabulary_lacks_is_an_incomplete_reason() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.new_effect.activate_trigger(trigger_id=0)
    trigger.effects[0].effect_type = 9999
    findings = _new_trigger_findings(map_analysis.check_incomplete_triggers, loaded)
    assert len(findings) == 1 and "effect 1 (type 9999): this version has no such effect type" in findings[0].message


def test_an_entry_naming_an_undefined_player_is_info_where_its_type_uses_the_field() -> None:
    """The fixture defines players 1 and 2."""
    loaded, _manager, trigger = _new_trigger()
    trigger.new_effect.change_diplomacy(source_player=1, target_player=5)
    trigger.new_effect.change_diplomacy(source_player=0, target_player=2)
    trigger.new_effect.activate_trigger(trigger_id=0)
    trigger.effects[2].source_player = 6  # activate_trigger lists no player field
    findings = map_analysis.check_trigger_players(loaded).findings
    assert [(f.severity, f.trigger_entry) for f in findings] == [("info", ("effect", 0))]
    assert "target player is player 5" in findings[0].message


def test_a_disabled_trigger_nothing_activates_never_starts() -> None:
    loaded, manager, trigger = _new_trigger()
    before = map_analysis.check_never_started_triggers(loaded).findings
    trigger.enabled = 0
    findings = map_analysis.check_never_started_triggers(loaded).findings
    assert len(findings) == len(before) + 1
    assert findings[-1].trigger_index == 4 and findings[-1].severity == "info"
    manager.triggers[0].new_effect.activate_trigger(trigger_id=trigger.trigger_id)
    assert len(map_analysis.check_never_started_triggers(loaded).findings) == len(before)


def test_a_disabled_divider_is_not_a_never_started_trigger() -> None:
    from descape import trigger_organize

    loaded, _manager, trigger = _new_trigger()
    before = len(map_analysis.check_never_started_triggers(loaded).findings)
    trigger.name = "--- Section ---"
    trigger.enabled = 0
    assert trigger_organize.is_divider(trigger.name)
    assert len(map_analysis.check_never_started_triggers(loaded).findings) == before


# -- TASK-098 review folds ----------------------------------------------------------


def test_the_player_checks_read_the_edited_player_count_when_given() -> None:
    """Raising Number of Players to 4 without saving defines player 4 for both checks."""
    loaded, _manager, trigger = _new_trigger()
    loaded.unit_manager.units[4].extend(loaded.unit_manager.units[1])
    trigger.new_effect.change_diplomacy(source_player=1, target_player=4)
    assert map_analysis.check_undefined_player_units(loaded).findings
    assert _new_trigger_findings(map_analysis.check_trigger_players, loaded)
    assert map_analysis.check_undefined_player_units(loaded, defined_players=[1, 2, 3, 4]).is_clean
    report = map_analysis.analyze(loaded, defined_players=[1, 2, 3, 4])
    by_label = {r.label: r for r in report.results}
    assert by_label["Units of undefined players"].is_clean
    assert not [f for f in by_label["Trigger entries naming an undefined player"].findings if f.trigger_index == 4]


def test_the_game_data_checks_are_unavailable_when_another_games_data_is_shown() -> None:
    report = map_analysis.analyze(load_map_and_units(UNITS_FIXTURE), foreign_data="showing other data")
    by_label = {r.label: r for r in report.results}
    for label in ("Objects the game data does not have", "Terrains the game data does not have", "Garrisons the host can't hold"):
        assert by_label[label].unavailable == "showing other data", label


def test_a_wrong_type_occupant_is_not_also_counted_against_capacity(monkeypatch) -> None:
    from descape import garrison

    loaded = load_map_and_units(UNITS_FIXTURE)
    _unit(loaded, 201).garrisoned_in_id = 200  # the house holds neither unit
    monkeypatch.setattr(garrison, "capacity", lambda _host: 1)
    findings = map_analysis.check_garrison_fit(loaded).findings
    assert [f.unit_key for f in findings] == [(1, 201), (1, 203)]
    assert all("can't hold it" in f.message for f in findings)


def test_an_unset_player_is_not_an_undefined_player() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.new_effect.change_diplomacy(source_player=-1, target_player=-1)
    assert _new_trigger_findings(map_analysis.check_trigger_players, loaded) == []


def test_a_trigger_that_only_activates_itself_never_starts() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.enabled = 0
    trigger.new_effect.activate_trigger(trigger_id=trigger.trigger_id)
    assert [f.trigger_index for f in map_analysis.check_never_started_triggers(loaded).findings][-1] == 4


def test_an_entry_label_numbers_by_display_slot_and_the_entry_keeps_its_stored_index() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.new_effect.activate_trigger(trigger_id=0)
    trigger.new_effect.change_diplomacy(source_player=1, target_player=5)
    trigger.effect_order = [1, 0]
    findings = _new_trigger_findings(map_analysis.check_trigger_players, loaded)
    assert [f.trigger_entry for f in findings] == [("effect", 1)]
    assert "effect 1 (change diplomacy)" in findings[0].message


def test_the_existing_trigger_checks_fill_trigger_index() -> None:
    loaded, _manager, trigger = _new_trigger()
    trigger.new_effect.activate_trigger(trigger_id=99)
    trigger.new_condition.destroy_object(unit_object=99999)
    trigger.new_effect.change_train_location(object_list_unit_id=4, object_list_unit_id_2=87, button_location=0)
    assert [f.trigger_index for f in map_analysis.check_trigger_references(loaded).findings] == [4]
    assert 4 in [f.trigger_index for f in map_analysis.check_unit_references(loaded).findings]
    located = [f for f in map_analysis.check_location_buttons(loaded).findings if f.trigger_index == 4]
    assert [f.trigger_entry for f in located] == [("effect", 1)]


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


@pytest.mark.corpus
@pytest.mark.parametrize(
    "check",
    [
        map_analysis.check_unit_ids,
        map_analysis.check_unknown_objects,
        map_analysis.check_unknown_terrains,
        map_analysis.check_garrison_fit,
        map_analysis.check_undefined_player_units,
        map_analysis.check_created_object_terrain,
    ],
    ids=lambda check: check.__name__,
)
def test_shipping_scenarios_fire_no_gh282_warning_check(scenario_path: Path, check) -> None:
    """GH #282: each of these fired on 0 of 21 examples/ files on 2026-11-26."""
    assert [f.message for f in check(load_map_and_units(scenario_path)).findings] == []


@pytest.mark.corpus
def test_shipping_scenarios_have_no_trigger_coordinate_warning(scenario_path: Path) -> None:
    """GH #282: no WARNING anywhere; 8tp3w9j's three areas reaching x = 259 on its 240 map are INFO."""
    findings = map_analysis.check_trigger_coordinates(load_map_and_units(scenario_path)).findings
    assert [f.message for f in findings if f.severity == "warning"] == []
    if scenario_path.name == "8tp3w9j.aoe2scenario":
        assert len(findings) == 3
