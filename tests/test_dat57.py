"""tools/_dat57.py: the original game's VER 5.7 .dat reader.

The default tier packs a whole minimal Conquerors .dat with literal
`struct.pack` calls, one per record, written from the layout notes rather
than from the reader's own format strings, and checks every section reads
back under genieutils-py's field names. The corpus tier parses the real
install's empires2_x1_p1.dat.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The Conquerors install's Data folder for the corpus tier; DESCAPE_AOC_DATA overrides it.
AOC_DATA = Path(os.environ.get("DESCAPE_AOC_DATA") or Path.home() / "games" / "Age of Empires 2" / "Data")
# Read at import: conftest unsets HD_INSTALL_PATH inside every test.
HD_INSTALL = os.environ.get("HD_INSTALL_PATH")


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import _dat57
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return _dat57


dat57 = _load()


def p(fmt: str, *values) -> bytes:
    return struct.pack("<" + fmt, *values)


def unit_head(unit_type: int, unit_id: int, name: bytes, *, hot_key: int = 16000, tc: bool = True) -> bytes:
    """The fixed part of a unit record, up to its name, with distinct values in the fields the tests read."""
    out = p("bH", unit_type, len(name)) + p("hHHh", unit_id, 5000 + unit_id, 6000 + unit_id, 0)
    out += p("hh", 633, -1) + p("hhb", 630, -5, 0) + p("hf", 30, 6.0) + p("b", 0) + p("fff", 0.2, 0.3, 2.0)
    out += p("hhh", 422, 423, 3)  # train sound, damage sound, dead unit
    out += p("bbhbhbb", 1, 0, 17, 0, -1, 1, 0)  # sort, built on, icon, hide, portrait, enabled, disabled
    out += p("hh", -1, -2) + p("hh", -3, -4) + p("ff", 0.5, 0.25)
    out += p("bbhbhf", 0, 0, 7, 0, 1, 0.5)  # hill, fog, restriction, fly, capacity, decay
    out += p("bbbbbfb", 2, 4, 4, 1, 4, 1.5, 9)  # blast level .. minimap colour
    out += p("iii", 105000 + unit_id, 155000 + unit_id, hot_key)
    out += p("bbbbbbb", 0, 0, 0, 0, 0, 5, 2)  # recyclable .. obstruction class
    if tc:
        out += p("bbh", 0, 3, 7)  # trait, civilization, nothing
    out += p("bb", 1, 52) + p("fff", 0.2, 0.21, 2.0)
    out += p("hfb", 4, -1.0, 2) + p("hfb", 11, 1.0, 2) + p("hfb", 19, 1.0, 1)
    out += p("B", 1) + p("hbbb", 4563, 25, 0, 1)
    out += p("hhbb", 424, 425, 0, 0) + name + p("hh", unit_id, unit_id + 100)
    return out


def moving(*, base_armor: bytes = b"") -> bytes:
    """Speed, dead fish, bird and the type 50 block, as an archer has them."""
    out = p("f", 0.96)
    out += p("hhfbhbfb", 637, -1, 0.5, 0, -1, 0, 0.0, 0) + p("fffff", 0.0, 1.0, 2.0, 3.0, 4.0)
    out += p("hff", -1, 6.0, 1.0) + p("hh", 109, 68) + p("Bhh", 1, 422, 421) + p("B", 0)
    out += base_armor or p("h", 1000)
    out += p("H", 2) + p("hh", 27, 3) + p("hh", 3, 4) + p("H", 1) + p("hh", 4, 0)
    out += p("h", -1) + p("fff", 4.0, 0.0, 2.0) + p("hhBh", 363, 80, 0, 5) + p("fff", 0.1, 0.2, 0.3)
    out += p("Bff", 0, 0.5, 0.25) + p("hhh", 627, 0, 4) + p("ff", 4.0, 2.0)
    return out


def creatable() -> bytes:
    out = p("hhh", 1, 25, 1) + p("hhh", 3, 45, 1) + p("hhh", 4, 1, 0)
    out += p("hhb", 35, 87, 1) + p("ff", 0.0, 0.5) + p("bb", 5, 0) + p("i", -1) + p("fb", 1.0, 1)
    out += p("fff", 1.0, 2.0, 3.0) + p("ii", -1, -2) + p("bh", 0, 2)
    return out


def building(*, snow: bool = True) -> bytes:
    out = p("h", 121) + (p("h", 6406) if snow else b"")
    out += p("bhbhhhhb", 0, 0, 0, -1, 27, -1, -1, 1)
    out += p("hff", 618, 1.0, -1.0) + p("hff", 619, -0.5, 0.5) + p("hff", 620, -1.0, 1.0) + p("hff", -1, 0.0, 0.0)
    out += p("hhhh", 621, -1, -1, 17) + p("bff", 11, 0.1, 0.0) + p("h", -1) + p("6b", 1, 1, 0, 1, 1, 0)
    return out


def terrain(name: bytes = b"Grass", *, borders: int = 42) -> bytes:
    out = p("BB13s13s", 1, 0, name, b"g_grs") + p("5i", 15001, 0, 256, 100, 0)
    out += p("3B2BBB", 55, 236, 54, 0, 0, 255, 255) + p("Bhhff", 0, 0, 0, 0.0, 0.0) + p("hhfBB", 0, 0, 0.0, 0, 0)
    out += p("3h", 100, 1, 0) + b"".join(p("3h", 1, 1, 100 + k) for k in range(18))
    out += p("h2h", -1, 10, 10) + p(f"{borders}h", 5, *[0] * (borders - 1))
    out += p("30h", 411, *[0] * 29) + p("30h", 1000, *[0] * 29) + p("30B", 1, *[0] * 29) + p("hh", 1, 0)
    return out


def aoc_dat(*, civ_count: int = 19, archer_name: bytes = b"ARCHR", slots: int = 42, used: int = 41) -> bytes:
    out = b"VER 5.7\0" + p("HH", 1, used) + p("i", 7) + p("i", 8)
    out += p(f"{used}f", 1.0, *[0.5] * (used - 1)) + p("4i", -1, -2, -3, 4) * used
    out += p("H", 1) + p("9i", 0, 16, 242, 0, 0, 242, 0, 0, 0)
    out += p("H", 1) + p("hhHi", 5, 0, 1, 300000) + p("13sihhh", b"dock.wav", 5043, 100, -1, -1)
    out += p("H", 2) + p("ii", 0, 1)
    out += p("21s13s", b"ARRG2N0E", b"ARRG2N0E") + p("iBBBhB", 13, 0, 1, 5, -1, 2) + p("4h", 1, 2, 3, 4)
    out += p("HhBHH", 1, -1, 1, 1, 1) + p("fff", 1.0, 0.1, 0.0) + p("BhBB", 3, 1, 6, 0)
    out += p("hhihhhh", 1, 0, 0, 2, 3, -1, 0) + p("6h", 7, 307, -1, -1, -1, -1)
    # Terrain block: head, 19 tile sizes, padding, 42 terrains (100 in HD), 16 borders, tail.
    out += p("IIiiii", 0x6A114C, 0, 0, 0, 0, 0) + p("3h", 96, 48, 0) * 19 + p("h", 0)
    out += terrain(borders=slots) + bytes(436 + 2 * (slots - 42)) * (slots - 1) + bytes(1440) * 16
    out += p("i6f", 0, *[0.0] * 6) + p("8h", used, 0, 0, 96, 48, 24, 48, 24) + p("6h", 0, 0, 0, 0, 0, 0)
    out += p("IIBBB", 0, 0, 0, 0, 1) + bytes(21) + p("157i", 237895712, *[0] * 156)
    # Random maps: one header, then its body with one row of each list.
    out += p("Ii", 1, 0x0E290100) + p("10i", 4, 9, 9, 9, 9, 25, 10, 1, 40, 0) + p("IiIiIiIi", 1, 11, 1, 12, 1, 13, 1, 14)
    out += p("9i", 9, 9, 9, 9, 25, 10, 1, 40, 0)
    out += p("Ii", 1, 11) + p("iIiiBBhiiBBhiii", 1, 0, 10, 7, 0, 2, 0, 0, 0, 100, 2, 0, 10, 25, 8)
    out += p("Ii", 1, 12) + p("6i", 50, 14, 10, 1, 0, 20)
    out += p("Ii", 1, 13) + p("iiBBhiiiiiiii", 109, -1, 0, 0, 0, 1, 0, 1, 1, -1, 1, 0, 1)
    out += p("Ii", 1, 14) + p("6i", 1, 2, 3, 4, 5, 6)
    out += p("i", 1) + p("31sH", b"Loom", 1) + p("Bhhhf", 4, -1, 4, 0, 15.0)
    out += p("i", 2) + p("B", 0) + p("BH", 1, 1)
    out += p("hhbhhhhhhhh", 1, 0, 0, 7, -1, -1, -1, -1, -1, -1, -1) + p("fffbf", 0.0, 0.0, 0.0, 0, 0.0)
    out += p("bbhhbbb", 1, 1, 1, 0, 2, 0, 0) + p("6h", -1, -1, -1, -1, -1, -1)
    out += p("H", civ_count)
    archer = unit_head(70, 4, archer_name, hot_key=16083) + moving() + creatable()
    out += p("B20sHhh", 1, b"Gaia", 2, 348, -1) + p("2f", 200.0, 300.0) + p("B", 1)
    out += p("H", 5) + p("5i", 0, 0, 0, 0, 1) + archer
    castle = unit_head(80, 0, b"CSTL") + moving() + creatable() + building()
    arrow = unit_head(60, 1, b"ARROW") + moving() + p("bbbbbf", 0, 1, 0, 1, 0, 0.5)
    out += p("B20sHhh", 1, b"British", 0, 254, -1) + p("B", 1) + p("H", 2) + p("2i", 1, 1) + castle + arrow
    for k in range(2, civ_count):
        out += p("B20sHhh", 1, b"Civ%d" % k, 0, 254 + k, -1) + p("B", 1) + p("H", 0)
    out += p("H", 1) + p("6h", 104, -1, -1, -1, -1, -1) + p("hhB", 3, 50, 1) + p("hhB", -1, 0, 0) * 2
    out += p("hhhh", 1, -1, 0, 109) + p("HH", 7022, 8022) + p("hhhhB", 25, 22, 0, 6, 6)
    out += p("iiiH", 107022, 157022, -1, 4) + b"Loom"
    out += p("7i", 30, 7, 14, 700, 1400, 5, 10)
    common = p("21i", 1, 22, *[0] * 9, 2, *[0] * 9)
    out += p("BBBBi", 1, 1, 1, 1, 116)
    out += p("iB", 1, 2) + p("Bi", 1, 109) + p("B", 0) + p("Bi", 1, 22) + common + p("B10B10BBi", 1, *[0] * 20, 2, 0)
    out += p("iB", 109, 2) + p("B", 0) + p("Bi", 1, 83) + p("B", 0) + common + p("B5B5Bii", 1, *[0] * 10, 3, -1)
    out += p("iBi", 83, 2, 109) + common + p("i", 3) + p("B", 0) + p("iiii", 1, -1, 2, -1)
    out += p("iBi", 22, 2, 109) + p("B", 0) + p("B", 0) + p("B", 0) + common + p("iii", 1, 2, 3)
    return out


def test_a_packed_conquerors_dat_reads_every_section_back():
    dat = dat57.parse_bytes(aoc_dat())
    assert dat.version == dat57.VERSION == "VER 5.7"
    assert dat.layout == "aoc" and dat.terrains_used_1 == 41
    assert dat.float_ptr_terrain_tables == [7] and dat.terrain_pass_graphic_pointers == [8]
    restriction = dat.terrain_restrictions[0]
    assert restriction.passable_buildable_dmg_multiplier[:2] == [1.0, 0.5]
    assert restriction.terrain_pass_graphics[40] == dat57.TerrainPassGraphic(-1, -2, -3, 4)
    assert dat.player_colours[0].player_color_base == 16 and dat.player_colours[0].minimap_color == 242
    sound = dat.sounds[0]
    assert (sound.id, sound.cache_time, sound.items[0].filename, sound.items[0].resource_id) == (5, 300000, "dock.wav", 5043)


def test_graphics_keep_null_slots_and_read_deltas_and_angle_sounds():
    dat = dat57.parse_bytes(aoc_dat())
    assert dat.graphics[0] is None and dat.graphic_pointers == [0, 1]
    g = dat.graphics[1]
    assert (g.name, g.file_name, g.slp, g.layer, g.player_color, g.transparent_selection) == ("ARRG2N0E", "ARRG2N0E", 13, 5, -1, 2)
    assert (g.old_color_flag, g.coordinates, g.sound_id, g.frame_count, g.angle_count) == (1, (1, 2, 3, 4), -1, 1, 1)
    assert (g.speed_multiplier, g.sequence_type, g.id, g.mirroring_mode) == (1.0, 3, 1, 6)
    assert g.deltas == [dat57.GraphicDelta(1, 0, 0, 2, 3, -1, 0)]
    assert g.angle_sounds[0].frame_num == 7 and g.angle_sounds[0].sound_id == 307


def test_the_terrain_block_reads_terrains_borders_and_its_tail():
    block = dat57.parse_bytes(aoc_dat()).terrain_block
    assert block.virtual_function_ptr == 0x6A114C and len(block.tile_sizes) == 19
    assert len(block.terrains) == 42 and len(block.terrain_borders) == 16
    grass = block.terrains[0]
    assert (grass.name, grass.name_2, grass.slp, grass.sound_id, grass.blend_priority) == ("Grass", "g_grs", 15001, 256, 100)
    assert grass.colors == (55, 236, 54) and grass.passable_terrain == 255
    assert grass.frame_data[0] == dat57.FrameData(100, 1, 0) and grass.frame_data[18].shape_id == 117
    assert (grass.terrain_to_draw, grass.terrain_dimensions, grass.borders[:2]) == (-1, (10, 10), (5, 0))
    assert grass.terrain_unit_id[0] == 411 and grass.terrain_unit_density[0] == 1000
    assert grass.terrain_unit_centering[0] == 1 and grass.number_of_terrain_units_used == 1
    assert (block.terrains_used_2, block.tile_width, block.tile_half_width, block.elev_height) == (41, 96, 48, 24)
    assert block.fog_flag == 1 and len(block.some_bytes) == 21 and block.some_int32[0] == 237895712


def test_random_maps_effects_and_unit_headers():
    dat = dat57.parse_bytes(aoc_dat())
    header, body = dat.random_maps.map_info_1[0], dat.random_maps.map_info_2[0]
    assert dat.random_maps.random_maps_ptr == 0x0E290100
    assert header.map_id == body.map_id == 4 and header.map_lands == [] and header.map_units_ptr == 13
    assert body.border_usage == 25 and body.land_coverage == 40
    assert body.map_lands[0].land_spacing == 10 and body.map_lands[0].clumpiness == 8
    assert body.map_terrains[0].proportion == 50 and body.map_units[0].unit == 109
    assert body.map_units[0].max_distance_to_players == 1 and body.map_elevations[0].tile_spacing == 6
    effect = dat.effects[0]
    assert effect.name == "Loom" and effect.effect_commands == [dat57.EffectCommand(4, -1, 4, 0, 15.0)]
    assert dat.unit_headers[0].task_list is None
    task = dat.unit_headers[1].task_list[0]
    assert (task.task_type, task.action_type, task.enable_targeting, task.target_diplomacy) == (1, 7, 1, 2)
    assert task.resource_deposit_sound_id == -1


def test_units_read_under_genieutils_names():
    dat = dat57.parse_bytes(aoc_dat())
    assert [c.name for c in dat.civs[:2]] == ["Gaia", "British"] and len(dat.civs) == 19
    gaia = dat.civs[0]
    assert (gaia.tech_tree_id, gaia.resources, gaia.icon_set) == (348, [200.0, 300.0], 1)
    assert gaia.units[:4] == [None] * 4
    archer = gaia.units[4]
    assert (archer.type, archer.id, archer.name, archer.copy_id, archer.base_id) == (70, 4, "ARCHR", 4, 104)
    assert (archer.language_dll_name, archer.standing_graphic, archer.dying_graphic, archer.undead_graphic) == (5004, (633, -1), 630, -5)
    assert (archer.hit_points, archer.line_of_sight, archer.collision_size_z, archer.dead_unit_id) == (30, 6.0, 2.0, 3)
    assert (archer.icon_id, archer.placement_terrain, archer.clearance_size, archer.terrain_restriction) == (17, (-3, -4), (0.5, 0.25), 7)
    assert (archer.multiple_attribute_mode, archer.minimap_color, archer.hot_key) == (1.5, 9, 16083)
    assert (archer.obstruction_type, archer.obstruction_class, archer.civilization, archer.nothing) == (5, 2, 3, 7)
    assert (archer.selection_effect, archer.editor_selection_colour, archer.outline_size_y) == (1, 52, pytest.approx(0.21))
    assert archer.resource_storages[2] == dat57.ResourceStorage(19, 1.0, 1)
    assert archer.damage_graphics == [dat57.DamageGraphic(4563, 25, 0, 1)]
    assert (archer.selection_sound, archer.dying_sound) == (424, 425)
    assert archer.speed == pytest.approx(0.96) and archer.dead_fish.walking_graphic == 637
    assert archer.dead_fish.max_yaw_per_second_stationary == 4.0
    assert archer.bird.drop_sites == [109, 68] and (archer.bird.attack_sound, archer.bird.move_sound) == (422, 421)
    t50 = archer.type_50
    assert t50.base_armor == 1000 and [a.amount for a in t50.attacks] == [3, 4] and t50.armours[0].class_ == 4
    assert (t50.max_range, t50.projectile_unit_id, t50.accuracy_percent, t50.frame_delay) == (4.0, 363, 80, 5)
    assert (t50.min_range, t50.attack_graphic, t50.displayed_attack, t50.displayed_reload_time) == (0.5, 627, 4, 2.0)
    cr = archer.creatable
    assert cr.resource_costs[1] == dat57.ResourceCost(3, 45, 1)
    assert cr.train_locations == [dat57.TrainLocation(35, 87, 1, 16083)]
    assert (cr.flank_attack_modifier, cr.creatable_type, cr.max_total_projectiles) == (0.5, 5, 1)
    assert (cr.projectile_spawning_area, cr.special_graphic, cr.displayed_pierce_armour) == ((1.0, 2.0, 3.0), -2, 2)
    assert archer.projectile is None and archer.building is None


def test_buildings_and_projectiles_read_their_own_blocks():
    castle, arrow = dat57.parse_bytes(aoc_dat()).civs[1].units
    b = castle.building
    assert (b.construction_graphic_id, b.snow_graphic_id, b.foundation_terrain_id, b.can_burn) == (121, 6406, 27, 1)
    assert b.annexes[2] == dat57.BuildingAnnex(620, -1.0, 1.0) and b.head_unit == 621
    assert (b.construction_sound, b.garrison_type, b.pile_unit, b.looting_table) == (17, 11, -1, (1, 1, 0, 1, 1, 0))
    assert arrow.projectile == dat57.Projectile(0, 1, 0, 1, 0, 0.5) and arrow.creatable is None


def test_techs_globals_and_the_tech_tree():
    dat = dat57.parse_bytes(aoc_dat())
    tech = dat.techs[0]
    assert (tech.name, tech.required_techs[0], tech.effect_id, tech.icon_id) == ("Loom", 104, 22, 6)
    assert (tech.language_dll_name, tech.language_dll_help, tech.resource_costs[0].amount) == (7022, 107022, 50)
    assert tech.research_locations == [dat57.ResearchLocation(109, 25, 6, -1)]
    assert (dat.time_slice, dat.unit_hit_point_total, dat.razing_kill_total) == (30, 1400, 10)
    tree = dat.tech_tree
    assert tree.total_unit_tech_groups == 116
    age = tree.tech_tree_ages[0]
    assert (age.id, age.status, age.buildings, age.techs, age.max_age_length) == (1, 2, [109], [22], 2)
    assert age.common.slots_used == 1 and age.common.unit_research[0] == 22 and age.common.mode[0] == 2
    assert tree.building_connections[0].units == [83] and tree.building_connections[0].line_mode == 3
    unit_conn = tree.unit_connections[0]
    assert (unit_conn.upper_building, unit_conn.vertical_line, unit_conn.location_in_age, unit_conn.line_mode) == (109, 3, 1, 2)
    assert tree.research_connections[0].location_in_age == 2 and tree.research_connections[0].line_mode == 3


def test_parse_inflates_a_raw_deflate_file(tmp_path):
    packer = zlib.compressobj(9, zlib.DEFLATED, -15)
    path = tmp_path / "empires2_x1_p1.dat"
    path.write_bytes(packer.compress(aoc_dat()) + packer.flush())
    assert dat57.parse(path).civs[0].units[4].name == "ARCHR"
    path.write_bytes(aoc_dat())
    with pytest.raises(ValueError, match="raw-deflated"):
        dat57.parse(path)


def test_a_trailing_byte_fails_the_eof_check():
    with pytest.raises(ValueError, match="1 trailing bytes"):
        dat57.parse_bytes(aoc_dat() + b"\0")


def test_a_truncated_file_fails_clearly():
    with pytest.raises(ValueError, match="truncated"):
        dat57.parse_bytes(aoc_dat()[:-3])


def test_another_version_is_refused():
    with pytest.raises(ValueError, match=r"VER 5\.8"):
        dat57.parse_bytes(b"VER 5.8\0" + aoc_dat()[8:])


def test_the_self_checks_pin_the_civ_count_and_the_archer():
    with pytest.raises(ValueError, match="expected 19 civs"):
        dat57.parse_bytes(aoc_dat(civ_count=18))
    with pytest.raises(ValueError, match="'ARCHX'"):
        dat57.parse_bytes(aoc_dat(archer_name=b"ARCHX"))


def test_the_hd_edition_layout_reads_100_terrain_slots():
    assert dat57._detect_layout(42) is dat57.AOC and dat57._detect_layout(100) is dat57.HD
    dat = dat57.parse_bytes(aoc_dat(civ_count=32, slots=100, used=100))
    assert dat.layout == "hd" and dat.terrains_used_1 == 100 and len(dat.civs) == 32
    block = dat.terrain_block
    assert len(block.terrains) == 100 and block.terrains_used_2 == 100
    grass = block.terrains[0]
    assert len(grass.borders) == 100 and grass.borders[0] == 5 and grass.terrain_unit_id[0] == 411
    assert dat.civs[0].units[4].name == "ARCHR" and dat.civs[1].units[0].building.snow_graphic_id == 6406
    with pytest.raises(ValueError, match="expected 32 civs"):
        dat57.parse_bytes(aoc_dat(civ_count=19, slots=100, used=100))


def test_the_age_of_kings_layout_reads_its_narrower_records():
    assert dat57._detect_layout(32) is dat57.AOK and dat57._detect_layout(41) is dat57.AOC
    raw = unit_head(80, 0, b"ACAD", tc=False) + moving(base_armor=p("B", 232)) + creatable() + building(snow=False)
    reader = dat57._Reader(raw, dat57.AOK)
    unit = dat57.read_unit(reader)
    assert reader.pos == len(raw)
    assert unit.name == "ACAD" and unit.obstruction_class == 2 and unit.selection_effect == 1
    assert (unit.trait, unit.civilization, unit.nothing) == (None, None, None)
    assert unit.type_50.base_armor == 232 and unit.building.snow_graphic_id is None
    assert unit.building.construction_graphic_id == 121 and unit.building.head_unit == 621
    raw_terrain = terrain(borders=32)
    reader = dat57._Reader(raw_terrain, dat57.AOK)
    assert dat57.read_terrain(reader).terrain_unit_id[0] == 411 and reader.pos == len(raw_terrain)


@pytest.mark.corpus
def test_the_installed_conquerors_dat_parses_to_its_last_byte():
    path = AOC_DATA / "empires2_x1_p1.dat"
    if not path.is_file():
        pytest.skip(f"no Conquerors install at {AOC_DATA}: set DESCAPE_AOC_DATA to its Data folder")
    dat = dat57.parse(path)  # raises unless the parse ends at EOF and the self-checks hold
    assert dat.layout == "aoc" and len(dat.civs) == 19 and dat.civs[0].units[4].name == "ARCHR"
    assert [len(dat.terrain_restrictions), len(dat.player_colours), len(dat.sounds), len(dat.techs)] == [22, 15, 506, 460]
    assert sum(g is not None for g in dat.graphics) == 7014 and len(dat.unit_headers) == 866
    assert dat.civs[18].name == "Koreans" and dat.techs[22].name == "Loom"


@pytest.mark.corpus
def test_the_installed_hd_edition_dat_parses_to_its_last_byte():
    path = Path(HD_INSTALL or "") / "resources" / "_common" / "dat" / "empires2_x2_p1.dat"
    if not HD_INSTALL or not path.is_file():
        pytest.skip("no HD Edition install: set HD_INSTALL_PATH to its Age2HD folder")
    dat = dat57.parse(path)
    assert dat.layout == "hd" and len(dat.terrain_block.terrains) == 100 and dat.civs[31].name == "Vietnamese"
    assert len(dat.civs[0].units) == 1402 and dat.civs[0].units[185].name == "SLINGR"
