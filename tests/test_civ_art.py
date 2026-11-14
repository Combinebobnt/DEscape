"""GH #48: per-civ architecture and per-age building art.

descape/civ_art.py derives each player's (art_civ, age, tech_civ) from the scenario's
civilization, architecture and starting age; unit_sprites.resolve_entry()
turns a placed building const into the entry that owner draws; the render
paths thread it through, and a Players-mode edit re-derives it
(scenario_io.refresh_player_render_context). The render-context half is
modelled on tests/test_player_colors.py and uses a synthetic install, so it
runs without the game.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from test_invalidate_units_splice import _make_cache, _oracle

from descape import asset_source, civ_art, gate_orientation, player_fields, render_flat, render_iso, render_sprites, unit_sprites
from descape.options_model import OptionsEditModel
from descape.player_fields import parse_player_field_id, player_field_id
from descape.render_cache import FlatChunkCache
from descape.render_paint import tile_pixels_for_map
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, refresh_player_render_context
from descape.unit_model import UnitEditModel

from test_unit_sprites import CONST, FILE_NAME, build_sld

CHINESE = 6
FRANKS = 2
TEUTONS = 4
HOUSE = 70


# --- civ tokens -> .dat index -------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [
    (1, 1), (4, TEUTONS), (59, 59), (60, 60), (62, 62),
    (0, None), (63, None), (65537, None), (65539, None), (-1, None), (True, None), (None, None),
    ("HUN-CIV", 17), ("MONGOL-CIV", 12), ("CHINESE-CIV", CHINESE), ("TUPI-CIV", 59),
    ("SAXONS-CIV", 60), ("VARANGIANS-CIV", 61), ("DANES-CIV", 62),
    ("GAIA", None), ("RANDOM-CIV", None), ("FULL-RANDOM-CIV", None), ("MIRROR-RANDOM-CIV", None),
    ("NOT-A-CIV", None), ("", None),
])
def test_civ_index_maps_ints_and_str16_tokens(value, expected):
    assert civ_art.civ_index(value) == expected


def test_the_three_civs_past_the_old_enum_match_the_dats_own_names():
    """Saxons/Varangians/Danes have no CivilizationOld member, so their
    indices come from a hand table; this pins it to the .dat's civ names as
    the generator recorded them."""
    names = unit_sprites.building_art().civ_names
    for name, index in civ_art._EXTRA_CIV_INDEX.items():
        assert names[index].upper() == name
    for token, index in civ_art._EXTRA_CIV_TOKENS.items():
        assert civ_art.civ_index(token) == index
    assert max(names) == civ_art.MAX_CIV_INDEX


def test_every_concrete_token_agrees_with_its_int_form():
    from AoE2ScenarioParser.datasets.object_support import Civilization, CivilizationOld

    for member in Civilization:
        if member.name in CivilizationOld.__members__:
            assert civ_art.civ_index(member.value) == civ_art.civ_index(CivilizationOld[member.name].value)


@pytest.mark.parametrize(("value", "expected"), [
    (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (0, 2), (1, 2), (7, 2), (4294967295, 2), (-1, 2), (None, 2),
])
def test_starting_age_clamps(value, expected):
    assert civ_art.art_age(value) == expected


@pytest.mark.parametrize(("civ", "arch", "expected"), [
    (TEUTONS, FRANKS, FRANKS),             # Joan co-op P2: architecture wins
    ("HUN-CIV", "MONGOL-CIV", 12),         # old-allies P7
    ("ROMANS-CIV", "RANDOM-CIV", 43),      # random architecture falls through to the civ
    (TEUTONS, 65537, TEUTONS),
    (TEUTONS, None, TEUTONS),              # no architecture_set at all (a 1.37 file)
    ("RANDOM-CIV", "RANDOM-CIV", None),    # then to today's art
    (65537, 65539, None),
])
def test_architecture_falls_through_to_the_civ_then_to_none(civ, arch, expected):
    assert civ_art.resolve(civ, arch, 3)[:2] == (expected, 3)


@pytest.mark.parametrize(("civ", "arch", "expected"), [
    (TEUTONS, FRANKS, TEUTONS),            # architecture is art only
    ("RANDOM-CIV", "MONGOL-CIV", None),    # a random civ uses the generic upgrades
    ("ROMANS-CIV", "RANDOM-CIV", 43),
])
def test_the_tech_civ_is_always_the_civilization(civ, arch, expected):
    """TASK-205: whose tech tree and auto techs upgrade the building."""
    assert civ_art.resolve(civ, arch, 3)[2] == expected


def _blank_with_options():
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    return loaded, OptionsEditModel(loaded)


def _pending_by_field(model) -> dict[str, dict[int, object]]:
    """viewer._pending_player_values() for plain player fields, Qt-free."""
    out: dict[str, dict[int, object]] = {}
    for key, value in model.pending_values().items():
        if key.startswith("player:"):
            field_id, player_id = parse_player_field_id(key)
            out.setdefault(field_id, {})[player_id] = value
    return out


def test_gaia_is_always_today_art_and_pending_edits_override_stored_values():
    loaded, model = _blank_with_options()
    assert loaded.player_art == (civ_art.GAIA_ART,) * 9  # the template is all RANDOM-CIV
    model.set_value(player_field_id("civilization", 1), "TEUTONIC-CIV")
    model.set_value(player_field_id("architecture", 2), "CHINESE-CIV")
    model.set_value(player_field_id("starting_age", 2), 6)
    art = civ_art.player_art(loaded, _pending_by_field(model))
    assert art[0] == civ_art.GAIA_ART
    assert art[1] == (TEUTONS, 2, TEUTONS)
    assert art[2] == (CHINESE, 6, None)  # architecture only: the generic upgrades
    assert art[3:] == (civ_art.GAIA_ART,) * 6


def test_a_file_without_architecture_falls_through_to_its_civ(monkeypatch):
    loaded, model = _blank_with_options()
    real = player_fields.specs_for
    monkeypatch.setattr(
        player_fields, "specs_for", lambda lo: tuple(s for s in real(lo) if s.field_id != "architecture")
    )
    model.set_value(player_field_id("civilization", 1), "CHINESE-CIV")
    assert civ_art.player_art(loaded, _pending_by_field(model))[1] == (CHINESE, 2, CHINESE)


def test_the_portrait_civ_is_the_architecture_else_the_civilization():
    """GH #372: portraits follow art_civ, as the game does in play."""
    loaded, model = _blank_with_options()
    model.set_value(player_field_id("civilization", 1), "CHINESE-CIV")
    model.set_value(player_field_id("architecture", 1), "TEUTONIC-CIV")
    model.set_value(player_field_id("civilization", 2), "CHINESE-CIV")
    refresh_player_render_context(loaded, _pending_by_field(model))
    assert civ_art.player_portrait_civ(loaded, 1) == TEUTONS
    assert civ_art.player_portrait_civ(loaded, 2) == CHINESE, "a random architecture falls through to the civ"
    assert civ_art.player_portrait_civ(loaded, 0) is None, "GAIA"
    assert civ_art.player_portrait_civ(loaded, 3) is None, "RANDOM-CIV"
    assert civ_art.portrait_civ(None) is None
    assert civ_art.player_portrait_civ(None, 1) is None, "no open document"


def test_a_file_shown_with_the_other_games_data_has_no_portrait_civ():
    from descape import game_dataset

    loaded, model = _blank_with_options()
    model.set_value(player_field_id("civilization", 1), "CHINESE-CIV")
    refresh_player_render_context(loaded, _pending_by_field(model))
    assert civ_art.player_portrait_civ(loaded, 1) == CHINESE
    game_dataset.set_active(game_dataset.ROR)
    assert civ_art.player_portrait_civ(loaded, 1) is None


@pytest.mark.parametrize("error", [ValueError, TypeError, IndexError])
def test_surprising_player_data_never_makes_a_file_unopenable(monkeypatch, error):
    def boom(*_a, **_k):
        raise error("surprising player data")

    monkeypatch.setattr(civ_art, "player_art", boom)
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    assert loaded.player_art is None  # draws today's Gaia-table art
    assert refresh_player_render_context(loaded, {}) is False
    assert loaded.player_art is None


@pytest.mark.corpus
@pytest.mark.parametrize(("name", "player_id", "expected"), [
    ("2_Joan_coop_4_v0_13.aoe2scenario", 2, (FRANKS, 4, TEUTONS)),   # int path, civ Teutons
    ("old-allies-final-v2.aoe2scenario", 7, (12, 3, 17)),            # str16, HUN-CIV civ
    ("old-allies-final-v2.aoe2scenario", 5, (43, 5, 43)),            # RANDOM-CIV architecture
    ("0_June_Event_Scenario.aoe2scenario", 1, (TEUTONS, 4, TEUTONS)),  # 1.37: no architecture_set
    ("0_June_Event_Scenario.aoe2scenario", 0, civ_art.GAIA_ART),
])
def test_real_files_resolve_their_measured_players(request, name, player_id, expected):
    import conftest

    path = conftest._scenario_dir(request.config) / name
    if not path.is_file():
        pytest.skip(f"{name} not in examples/")
    assert load_map_and_units(str(path)).player_art[player_id] == expected


# --- resolve_entry against the committed table ---------------------------


def test_a_chinese_castle_age_house_draws_asia_age3_art():
    assert unit_sprites.resolve_entry(HOUSE, CHINESE, 4)["file_name"] == "b_asia_house_age3_x1"


def test_a_directly_placed_age_target_is_never_remapped():
    entry = unit_sprites.resolve_entry(463, CHINESE, 5)
    assert entry["file_name"] == "b_asia_house_age2_x1"
    assert unit_sprites.resolve_entry(463, None, 5) is unit_sprites.graphic_map()[463]


def test_an_unknown_civ_falls_back_to_the_gaia_table_at_its_age():
    gm = unit_sprites.graphic_map()
    assert unit_sprites.resolve_entry(HOUSE, 99, 4) is gm[464]
    assert unit_sprites.resolve_entry(HOUSE, None, 2) is gm[HOUSE]
    assert unit_sprites.resolve_entry(HOUSE, 99, 2) is gm[HOUSE]


def test_a_non_building_const_is_untouched():
    villager = 83
    assert unit_sprites.resolve_entry(villager, CHINESE, 5) is unit_sprites.graphic_map()[villager]


def test_the_age_table_reads_the_age_techs():
    """TASK-205: the generic start simulation, ages 2..6; Post-Imperial adds no age tech."""
    ages = unit_sprites.building_art().age_upgrades
    assert ages[12] == {3: 498, 4: 132, 5: 20, 6: 20}
    assert ages[HOUSE] == {3: 463, 4: 464, 5: 465, 6: 465}
    assert 463 not in ages


# --- TASK-205: civ auto techs and Post-Imperial ----------------------------

ARMENIANS, GEORGIANS, POLES, TATARS, CUMANS = 44, 45, 38, 33, 34
MONASTERY, MILL, SIEGE_WORKSHOP, SETTLEMENT = 104, 68, 49, 2556


def _drawn(const, art_civ, age, tech_civ):
    return unit_sprites.resolve_entry(const, art_civ, age, tech_civ)["file_name"]


def test_armenian_and_georgian_monasteries_stay_fortified_churches_through_every_age():
    """Age 2 is the auto tech's first fire; 3..6 need it to re-fire over the age techs' 104 -> 30 -> 31 -> 32."""
    for civ in (ARMENIANS, GEORGIANS):
        for age in range(2, 7):
            assert _drawn(MONASTERY, civ, age, civ) == "b_medi_fortified_church_x1", (civ, age)


def test_the_tech_civ_not_the_architecture_upgrades_the_monastery():
    franks_with_armenian_art = _drawn(MONASTERY, ARMENIANS, 3, FRANKS)
    assert franks_with_armenian_art.startswith("b_byza_monastery_"), franks_with_armenian_art
    assert _drawn(MONASTERY, FRANKS, 3, ARMENIANS) == "b_medi_fortified_church_x1"


def test_poles_mills_keep_mill_art_at_every_age():
    """Decision 1: the Folwark's 3x3 footprint is held out, so Poles' Mill ages as anyone's."""
    gm = unit_sprites.graphic_map()
    generic = unit_sprites.building_art().age_upgrades[MILL]
    for age in range(2, 7):
        assert unit_sprites.resolve_entry(MILL, None, age, POLES) is gm[generic.get(age, MILL)], age


def test_a_settlement_draws_its_age2_art_in_feudal_for_any_civ():
    for tech_civ in (None, FRANKS, ARMENIANS):
        assert _drawn(SETTLEMENT, None, 3, tech_civ) == "b_ande_settlement_age2_x1", tech_civ
    assert _drawn(SETTLEMENT, None, 5, None) == "b_ande_settlement_age3_x1"


def test_central_asian_siege_workshops_follow_upgrade_rams():
    """712 fires at the start for everyone but Cumans, whose 706 waits for Castle Age."""
    assert _drawn(SIEGE_WORKSHOP, TATARS, 2, TATARS) == "b_ceas_siege_workshop_age3_x1"
    assert _drawn(SIEGE_WORKSHOP, CUMANS, 3, CUMANS) == "b_ceas_siege_workshop_age2_x1"
    assert _drawn(SIEGE_WORKSHOP, CUMANS, 4, CUMANS) == "b_ceas_siege_workshop_age3_x1"


BRITONS, GOTHS, ATHENIANS = 1, 3, 47
WATCH_TOWER, FORTIFIED_GATE_NE = 79, 63


def test_towers_follow_guard_tower_at_imperial_and_keep_at_post_imperial():
    """Slice 3: a later start has its earlier ages' generic techs, unless the
    civ's tech tree disables them (French have no Keep)."""
    gm = unit_sprites.graphic_map()
    assert unit_sprites.resolve_entry(WATCH_TOWER, None, 5, BRITONS) is gm[234]
    assert unit_sprites.resolve_entry(WATCH_TOWER, None, 6, BRITONS) is gm[235]
    assert unit_sprites.resolve_entry(WATCH_TOWER, None, 4, BRITONS) is gm[WATCH_TOWER]
    assert unit_sprites.resolve_entry(WATCH_TOWER, None, 6, FRANKS) is gm[234]


def test_walls_and_gates_are_fortified_at_an_imperial_start_where_the_civ_has_it():
    gm = unit_sprites.graphic_map()
    assert unit_sprites.resolve_entry(STONE_WALL, None, 5, BRITONS) is gm[FORTIFIED_WALL]
    assert unit_sprites.resolve_entry(STONE_WALL, None, 5, GOTHS) is gm[STONE_WALL]  # no Fortified Wall
    assert _drawn(STONE_WALL, ATHENIANS, 5, ATHENIANS) == "b_greek_wall_fortified_x1"  # GH #329's own art
    assert unit_sprites.resolve_entry(STONE_GATE_NE, None, 5, BRITONS) is gm[FORTIFIED_GATE_NE]
    assert unit_sprites.resolve_entry(STONE_GATE_NE, None, 5, GOTHS) is gm[STONE_GATE_NE]


def test_a_civs_gate_siblings_name_one_flag_set():
    """TASK-433: one owner's four orientation siblings of a gate draw the same
    flag files, so Flat's turned art (a sibling's) keeps the gate's pennants."""
    art = unit_sprites.building_art()
    split = []
    for civ in (None, *sorted(art.civ_art)):
        for group, consts in sorted(gate_orientation.groups().items()):
            flag_sets = set()
            for const in consts:
                entry = unit_sprites.resolve_entry(const, civ, 2) or {}
                flag_sets.add(tuple(sorted(p["file_name"] for p in entry.get("pieces", ()) if "_flag_" in p["file_name"])))
            if len(flag_sets) > 1:
                split.append((civ, group))
    assert not split, f"gate siblings naming different flags, (civ, group): {split}"
    flags = [p for p in unit_sprites.resolve_entry(490, 46, 2)["pieces"] if "_flag_" in p["file_name"]]
    assert [(p["file_name"], p["dx"], p["dy"]) for p in flags] == [
        ("b_west_gate_stone_flag_x1", -72, -156),
        ("b_west_gate_stone_flag_x1", 72, -84),
    ]


def test_every_file_the_civ_art_table_names_is_on_this_install():
    """TASK-433's guard on install gaps: the generator swaps an unreadable gate
    piece for Gaia's, so a file this install lacks must not reach the table."""
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    art = unit_sprites.building_art()
    entries = [e for by_const in art.civ_art.values() for e in by_const.values()]
    entries += [e for by_age in art.age_art.values() for by_civ in by_age.values() for e in by_civ.values()]
    names = set()
    for entry in entries:
        names.add(entry["file_name"])
        names.update(p["file_name"] for p in entry.get("pieces", ()))
        names.update(f["file_name"] for f in entry.get("garrison_flags", ()))
    missing = sorted(name for name in names if unit_sprites.sld_frame_count(name) is None)
    assert len(names) > 100, "too few files to prove anything"
    assert not missing, f"named but not readable on this install: {missing}"


def test_a_table_without_max_age_clamps_post_imperial_to_imperial(monkeypatch):
    """Today's RoR table: no max_age key, so age 6 reads its age-5 row."""
    real = unit_sprites.building_art()
    old_shape = unit_sprites.BuildingArt(
        age_upgrades={HOUSE: {3: 463, 4: 464, 5: 465}}, civ_art={}, civ_names=real.civ_names,
    )
    assert old_shape.max_age == 5 and old_shape.civ_upgrades == {}
    monkeypatch.setattr(unit_sprites, "building_art", lambda: old_shape)
    assert unit_sprites.resolve_entry(HOUSE, None, 6) is unit_sprites.graphic_map()[465]


def test_resolve_and_player_art_carry_the_tech_civ():
    assert civ_art.GAIA_ART == (None, 2, None)
    assert civ_art.resolve("ARMENIANS-CIV", "FRANKISH-CIV", 6) == (FRANKS, 6, ARMENIANS)


# The town centres the age techs reach: Dark Age 109 and its age targets.
# 2275-2277 (satrapy town centres) draw per age through age_art instead:
# test_satrapy_town_centres_draw_their_ages_annexes.
AGE_TOWN_CENTRES = (109, 71, 141, 142)
SATRAPY_TOWN_CENTRES = (2275, 2276, 2277)


def _age_token(file_name: str) -> str:
    import re

    match = re.search(r"_age(\d)_", file_name)
    assert match, file_name
    return match.group(1)


def test_every_town_centre_entry_is_one_age():
    """GH #48 Slice 2: each piece's age matches its parent's, in the Gaia table
    and in every civ's art, so no owner draws a mixed-age town centre."""
    gm = unit_sprites.graphic_map()
    art = unit_sprites.building_art()
    checked = 0
    for const in AGE_TOWN_CENTRES:
        for entry in [gm[const], *(a[const] for a in art.civ_art.values() if const in a)]:
            parent = next(p for p in entry["pieces"] if p.get("parent"))
            ages = {_age_token(p["file_name"]) for p in entry["pieces"]}
            assert ages == {_age_token(parent["file_name"])}, (const, [p["file_name"] for p in entry["pieces"]])
            checked += 1
    assert checked > 100


def test_satrapy_town_centres_draw_their_ages_annexes():
    """TASK-205 Slice 2: at each age 3..5 every annex is that age's (a Castle
    start draws age3 annexes), for Gaia and every civ in age_art, around the
    const's own Persian age-2 back piece; Post-Imperial draws as Imperial."""
    age_art = unit_sprites.building_art().age_art
    assert sorted(age_art) == list(SATRAPY_TOWN_CENTRES)
    checked = 0
    for const in SATRAPY_TOWN_CENTRES:
        by_age = age_art[const]
        assert sorted(by_age) == [3, 4, 5, 6]
        assert by_age[6] == by_age[5]
        for age in (3, 4, 5):
            assert 0 in by_age[age], (const, age)
            for civ, entry in by_age[age].items():
                annexes = [p for p in entry["pieces"] if not p.get("parent")]
                assert {_age_token(p["file_name"]) for p in annexes} == {str(age - 1)}, (const, age, civ)
                checked += 1
    assert checked > 100
    achaemenids = 46
    dark = unit_sprites.resolve_entry(2275, achaemenids, 2, achaemenids)
    castle = unit_sprites.resolve_entry(2275, achaemenids, 4, achaemenids)
    assert castle is age_art[2275][4][achaemenids]
    assert dark is not castle
    assert unit_sprites.resolve_entry(2275, 99, 4) is age_art[2275][4][0]  # no own art: Gaia's
    assert unit_sprites.resolve_entry(2275, None, 6) is age_art[2275][5][0]


def test_a_chinese_castle_age_town_centre_draws_one_asia_age3_set():
    entry = unit_sprites.resolve_entry(109, CHINESE, 4)
    assert [p["file_name"] for p in entry["pieces"]] == [
        "b_asia_town_center_age3_main_x1",
        "b_asia_town_center_age3_back_x1",
        "b_asia_town_center_age3_center_x1",
        "b_asia_town_center_age3_front_x1",
    ]


def test_pastures_keep_their_base_entry_for_every_owner():
    gm = unit_sprites.graphic_map()
    for const in (1889, 1893, 1897):
        assert unit_sprites.resolve_entry(const, CHINESE, 5) is gm[const]


STONE_GATE_NE = 64
STONE_WALL = 117
PALISADE = 72
FORTIFIED_WALL = 155


def test_walls_and_gates_draw_their_owners_architecture():
    assert unit_sprites.resolve_entry(STONE_WALL, CHINESE, 2)["file_name"] == "b_asia_wall_stone_x1"
    gate = unit_sprites.resolve_entry(STONE_GATE_NE, CHINESE, 2)
    assert gate["file_name"] == "b_asia_gate_stone_ne_closed_x1"
    assert "b_asia_gate_stone_corner_x1" in [p["file_name"] for p in gate["pieces"]]
    # The palisade's body is the delta at the Gaia body's position in that civ's own shell.
    assert unit_sprites.resolve_entry(PALISADE, 47, 2)["file_name"] == "b_archaic_wall_palisade_x1"


def test_a_gate_upgrade_keeps_its_orientation():
    """TASK-205 Slice 3: Fortified Wall upgrades a stone gate to the fortified
    sibling at the same place in its group's cycle, in every table."""
    art = unit_sprites.building_art()
    tables = [art.age_upgrades, *art.civ_upgrades.values()]
    gate_rows = 0
    for table in tables:
        for base, by_age in table.items():
            if not gate_orientation.is_gate(base):
                continue
            for target in by_age.values():
                base_group = gate_orientation.orientation_siblings(base)
                target_group = gate_orientation.orientation_siblings(target)
                assert target_group is not None, (base, target)
                assert base_group.index(base) == target_group.index(target), (base, target)
                gate_rows += 1
    assert gate_rows, "no gate upgrade at all, so this proves nothing"
    assert unit_sprites.resolve_entry(STONE_GATE_NE, CHINESE, 4) is unit_sprites.resolve_entry(STONE_GATE_NE, CHINESE, 2)


def test_the_greek_civs_draw_their_own_fortified_wall_over_the_five_shapes():
    """Greek-set Fortified Wall art stores 11 frames, and its first five are the
    western set's five shapes in order (GH #329; the rest scaffolding and rubble
    no wall index reaches), so those civs draw it as they draw their Greek Stone
    Wall, under the same five-shape logic."""
    gm = unit_sprites.graphic_map()
    for greek in (47, 48, 54):
        entry = unit_sprites.resolve_entry(FORTIFIED_WALL, greek, 2)
        assert entry["file_name"] == "b_greek_wall_fortified_x1"
        assert unit_sprites.resolve_entry(STONE_WALL, greek, 2)["file_name"] == "b_greek_wall_stone_x1"
        assert (entry["angle_count"], entry["rotation_is_variant"]) == (5, True)
        assert entry["variant_count"] > gm[FORTIFIED_WALL]["variant_count"] == 5
    assert unit_sprites.resolve_entry(FORTIFIED_WALL, CHINESE, 2)["file_name"] == "b_asia_wall_fortified_x1"


def test_every_civs_walls_and_gates_keep_the_hard_rule_shapes():
    """AGENTS.md's wall (angle_count 5) and gate (angle_count 1, one parent)
    rules, measured on Gaia, hold for every civ's art too."""
    art = unit_sprites.building_art()
    walls = set(unit_sprites.wall_family_consts())
    gates = {c for group in gate_orientation.groups().values() for c in group}
    seen_walls = seen_gates = 0
    for consts in art.civ_art.values():
        for const, entry in consts.items():
            if const in walls:
                assert entry["angle_count"] == 5, const
                assert all(p["angle_count"] == 5 for p in entry.get("pieces", ())), const
                seen_walls += 1
            if const in gates:
                assert all(p["angle_count"] == 1 for p in entry["pieces"]), const
                assert sum(1 for p in entry["pieces"] if p.get("parent")) == 1, const
                seen_gates += 1
    assert seen_walls > 50 and seen_gates > 500


def test_every_civ_entry_keeps_its_bases_const_logic():
    """The generator's invariant, re-read off the committed file: a drawn entry
    never changes what the placed const's rotation/variant logic reads. A wall
    may store frames past its five shapes, never fewer (GH #329)."""
    gm = unit_sprites.graphic_map()
    art = unit_sprites.building_art()
    walls = set(unit_sprites.wall_family_consts())
    checked = 0
    for civ, consts in art.civ_art.items():
        for const, entry in consts.items():
            base = gm[const]
            for field in ("angle_count", "rotation_is_variant", "variant_count"):
                if field == "variant_count" and const in walls:
                    assert entry.get(field) >= base.get(field), (civ, const, field)
                    continue
                assert entry.get(field) == base.get(field), (civ, const, field)
            checked += 1
    assert checked > 1000


# --- the render context, against a synthetic install ----------------------

CIV_FILE = "t_civ_art_x1"
AGE_FILE = "t_age_art_x1"
AGE_CONST = CONST + 1


@pytest.fixture
def art_install(tmp_path, monkeypatch):
    """CONST draws FILE_NAME on the Gaia table, CIV_FILE for Chinese, and is
    upgraded to AGE_CONST (AGE_FILE) at Castle Age. Each file a different colour."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    for colour_base, name in enumerate((FILE_NAME, CIV_FILE, AGE_FILE)):
        # No PLAYERCOLOR mask: a full one tints every pixel, hiding the MAIN colour.
        (graphics / f"{name}.sld").write_bytes(build_sld(1, playercolor=False, colour_base=10 * colour_base))

    def entry(name: str) -> dict:
        return {"graphic_id": 1, "file_name": name, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1}

    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {CONST: entry(FILE_NAME), AGE_CONST: entry(AGE_FILE)})
    # lru_cache-wrapped like the real one: clear_caches() calls its cache_clear().
    monkeypatch.setattr(unit_sprites, "building_art", functools.lru_cache(maxsize=1)(
        lambda: unit_sprites.BuildingArt(
            age_upgrades={CONST: {4: AGE_CONST}},
            civ_art={CHINESE: {CONST: entry(CIV_FILE)}},
            civ_names={},
        )
    ))
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _rerender(cache, loaded, model):
    """viewer._apply_player_color_change()'s order: recompute, then invalidate."""
    changed = refresh_player_render_context(loaded, _pending_by_field(model))
    cache.invalidate_units()
    w, h = cache.canvas_dims(0)
    cache.invalidate_region((0, 0, w, h))
    return changed, cache.render_rect(0, 0, w, h, mip=0).copy()


@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize(("field", "value"), [("architecture", "CHINESE-CIV"), ("starting_age", 4)])
def test_an_art_edit_reskins_the_iso_sprite_like_a_fresh_render(art_install, style, field, value):
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    cache = _make_cache(style, loaded, sprites=True)
    w, h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, w, h, mip=0).copy()

    model.set_value(player_field_id(field, 1), value)
    changed, after = _rerender(cache, loaded, model)

    assert changed is True
    assert not np.array_equal(after, before), "the edit did not re-skin the sprite"
    assert np.array_equal(after, _oracle(style, loaded, sprites=True)[:h, :w])


def test_a_non_art_player_edit_reports_no_change():
    loaded, model = _blank_with_options()
    model.set_value(player_field_id("gold", 1), 1234)
    model.set_value(player_field_id("lock_civilization", 1), 1)
    assert refresh_player_render_context(loaded, _pending_by_field(model)) is False


def test_an_art_edit_on_a_player_without_buildings_still_reports_a_change():
    """The gate is the derived tuple, not the pixels: what the caches must drop
    is decided by the memo key, which carries the tuple."""
    loaded, model = _blank_with_options()
    model.set_value(player_field_id("architecture", 3), "MONGOL-CIV")
    assert refresh_player_render_context(loaded, _pending_by_field(model)) is True
    assert loaded.player_art[3] == (12, 2, None)


def _layer_draw(loaded):
    _, elev, proj = render_iso.render_terrain_iso_with_proj(loaded, with_sprites=True)
    layer = render_sprites.sprite_draws_by_anchor(loaded, proj, elev)
    ((draw, _x, _y, _rise),) = next(iter(layer.by_anchor.values()))
    return draw


def test_two_scenarios_player_1_with_different_architectures_never_share_a_draw(art_install):
    gaia_scn, _ = _blank_with_options()
    chinese_scn, _ = _blank_with_options()
    for scn in (gaia_scn, chinese_scn):
        UnitEditModel(scn).add(1, CONST, 4.0, 4.0)
    chinese_scn.player_art = (civ_art.GAIA_ART, (CHINESE, 2, None), *(civ_art.GAIA_ART,) * 7)

    gaia_draw = _layer_draw(gaia_scn)
    chinese_draw = _layer_draw(chinese_scn)
    assert not np.array_equal(gaia_draw.rgba, chinese_draw.rgba)
    # And the first one's draw is still the Gaia art after the second rendered.
    assert np.array_equal(_layer_draw(gaia_scn).rgba, gaia_draw.rgba)


def test_the_flat_icon_cache_does_not_serve_gaia_art_to_a_chinese_house(art_install):
    gaia = unit_sprites.icon_for(CONST, 0.0, 1, 64, 64)
    chinese = unit_sprites.icon_for(CONST, 0.0, 1, 64, 64, art=(CHINESE, 2, None))
    castle = unit_sprites.icon_for(CONST, 0.0, 1, 64, 64, art=(None, 4, None))
    assert gaia is not None and chinese is not None and castle is not None
    assert not np.array_equal(gaia.rgba, chinese.rgba)
    assert not np.array_equal(gaia.rgba, castle.rgba)


def test_a_flat_art_edit_repaints_the_icon(art_install):
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    mm = loaded.map_manager
    cache = FlatChunkCache(loaded, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    w, h = cache.canvas_dims()
    before = cache.render_rect(0, 0, w, h).copy()

    model.set_value(player_field_id("architecture", 1), "CHINESE-CIV")
    refresh_player_render_context(loaded, _pending_by_field(model))
    cache.invalidate_units()
    cache.invalidate_region((0, 0, w, h))
    assert not np.array_equal(cache.render_rect(0, 0, w, h), before)


def test_the_flat_oracle_and_the_flat_cache_agree_on_the_owners_art(art_install):
    """overlay_units() is Flat's independent second implementation; it must
    draw the owner's art too, or the two stop being comparable."""
    loaded, _ = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    gaia = render_flat.overlay_units(render_flat.render_terrain(loaded), loaded, with_sprites=True)
    loaded.player_art = (civ_art.GAIA_ART, (CHINESE, 2, None), *(civ_art.GAIA_ART,) * 7)
    chinese = render_flat.overlay_units(render_flat.render_terrain(loaded), loaded, with_sprites=True)
    assert not np.array_equal(gaia, chinese)
    mm = loaded.map_manager
    cache = FlatChunkCache(loaded, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    w, h = cache.canvas_dims()
    assert np.array_equal(cache.render_rect(0, 0, w, h), chinese[:h, :w])


@pytest.mark.gui
def test_the_viewers_players_edit_tail_rederives_the_art_and_invalidates():
    import conftest

    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    window = conftest.blank_window()
    try:
        window._ensure_option_edits()
        epoch = window._cache._mutation_epoch
        window._after_player_color_change()
        assert window._cache._mutation_epoch == epoch, "an unchanged tail evicted the canvas"
        window.option_edits.set_value(player_field_id("architecture", 1), "CHINESE-CIV")
        window._after_player_color_change()
        assert window.scenario.player_art[1] == (CHINESE, 2, None)
        assert window._cache._mutation_epoch > epoch, "the art edit evicted nothing"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_the_units_panel_portraits_follow_a_players_edit_a_data_switch_and_close(monkeypatch):
    """GH #372: the viewer pushes the owners' portrait civs on every path that changes them."""
    import conftest

    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    window = conftest.blank_window()
    try:
        panel = window.units_panel
        panel.select_owner(1)
        assert panel.placing_card._civ is None, "the template's players are all RANDOM-CIV"
        refreshed = []
        monkeypatch.setattr(window.trigger_panel, "refresh_portraits", lambda: refreshed.append(True))
        spec = {s.field_id: s for s in player_fields.specs_for(window.scenario)}["architecture"]
        window.set_player_field(spec, 1, "CHINESE-CIV")
        assert panel.placing_card._civ == panel.catalog_view._civ == CHINESE
        assert refreshed, "an open button grid or instruction preview redraws in the new civ"
        window.undo()
        assert panel.placing_card._civ is None, "the edit undone"
        assert len(refreshed) == 2
        # A context, not monkeypatch.undo(): that would also drop conftest's settings isolation.
        with monkeypatch.context() as patch:
            patch.setattr(civ_art, "player_portrait_civ", lambda _scenario, pid: TEUTONS if pid == 1 else None)
            window._refresh_game_assets()
            assert panel.placing_card._civ == TEUTONS, "a data mod or game data switch re-syncs"
        window.option_edits.set_value(player_field_id("architecture", 1), "CHINESE-CIV")
        window._after_player_color_change()
        window.close_scenario()
        assert panel.placing_card._civ is None, "no document: GAIA's portraits"
    finally:
        conftest.close_window(window)


def test_the_place_ghost_draws_the_placing_players_art(art_install):
    loaded, _ = _blank_with_options()
    loaded.player_art = (civ_art.GAIA_ART, (CHINESE, 2, None), *(civ_art.GAIA_ART,) * 7)
    unit = UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    _, elev, proj = render_iso.render_terrain_iso_with_proj(loaded, with_sprites=True)
    ((p1, _x, _y),) = render_sprites.unit_sprite_draws_at(loaded, proj, elev, None, 1, unit)
    ((p2, _x, _y),) = render_sprites.unit_sprite_draws_at(loaded, proj, elev, None, 2, unit)
    assert not np.array_equal(p1.rgba, p2.rgba)


@pytest.mark.parametrize("op", ["move", "add"])
def test_a_flat_row_splice_keeps_the_owners_art(art_install, op, monkeypatch):
    """FlatChunkCache's row splice re-resolves moved and inserted icons itself
    (render_cache's two _flat_unit_icon calls), so it must thread the art too."""
    from test_invalidate_units_splice import (
        _apply,
        _assert_flat_matches_fresh,
        _flat_cache,
        _flat_counts,
        _flat_edit,
        _flat_scenario,
    )

    scenario = _flat_scenario(CONST)
    scenario.player_art = (civ_art.GAIA_ART, (CHINESE, 2, None), (CHINESE, 4, None), *(civ_art.GAIA_ART,) * 6)
    cache = _flat_cache(scenario, sprites=True)
    counts = _flat_counts(monkeypatch)
    splices = _flat_edit(op, scenario, CONST)
    assert cache._flat_splice_eligible(splices), "fixture is not testing the splice path"
    _apply(cache, splices[0])
    assert counts == {"rows": 0, "icons": 0}, "the splice path rebuilt a whole walk"
    _assert_flat_matches_fresh(cache, scenario, sprites=True)
