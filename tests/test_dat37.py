"""tools/_dat37.py: the original Age of Empires' VER 3.7 .dat reader.

The default tier packs a minimal Rise of Rome .dat with literal `struct.pack`
calls, one per record, written from the measured layout rather than from the
reader's own format strings, and checks the sections read back under
genieutils-py's field names. The corpus tier parses both installed .dat files
(`--aoe1-dir`, the install root).
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import _dat37
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return _dat37


dat37 = _load()


def p(fmt: str, *values) -> bytes:
    return struct.pack("<" + fmt, *values)


def unit_head(unit_type: int, unit_id: int, name: bytes) -> bytes:
    out = p("bH", unit_type, len(name)) + p("hHHh", unit_id, 5000 + unit_id, 6000 + unit_id, 3)
    out += p("hhhb", 400 + unit_id, 490, -1, 0) + p("hfb", 35, 7.0, 0) + p("fff", 0.2, 0.3, 2.0)
    out += p("hh", 422, 3)  # train sound, dead unit
    out += p("bbhbhb", 1, 0, 17, 0, -1, 1)  # sort, built on, icon, hide, portrait, enabled
    out += p("hh", -1, -2) + p("hh", -3, -4) + p("ff", 0.5, 0.25)
    out += p("bbhbhf", 2, 0, 4, 0, 0, 0.0) + p("bbbbbfb", 3, 2, 3, 1, 2, 1.0, 9)
    out += p("iii", 105000 + unit_id, 155000 + unit_id, 16000 + unit_id) + p("6b", 0, 1, 2, 0, 2, 52)
    out += p("fff", 1.5, 1.5, 1.5) + p("hfb", -1, 0.0, 0) + p("hfb", 30, -1.0, 2) + p("hfb", -1, 0.0, 0)
    out += p("B", 1) + p("hbbb", 131, 25, 1, 1) + p("hhbb", 0, -1, 0, 0) + name + p("h", unit_id)
    return out


def moving(*, tasks: int = 1) -> bytes:
    out = p("f", 1.2) + p("hhfbhbfb", 413, -1, 0.0, 0, -1, 0, 0.0, 0)
    out += p("hff", -1, 7.0, 1.0) + p("hh", 109, -1) + p("Bhhb", 0, 127, 128, 0) + p("H", tasks)
    for _ in range(tasks):
        out += p("hhbhhhhhhhh", 1, 0, 0, 7, -1, -1, -1, -1, -1, -1, -1) + p("fffbf", 0.0, 0.0, 0.0, 1, 3.0)
        out += p("bbhhbbb", 1, 1, 1, 0, 5, 0, 0) + p("6h", -1, -1, -1, -1, -1, -1)
    out += p("B", 1) + p("H", 1) + p("hh", 3, 4) + p("H", 1) + p("hh", 4, 2)
    out += p("h", -1) + p("fff", 5.0, 0.0, 1.4) + p("hhBh", 9, 80, 0, 5) + p("fff", 0.1, 0.2, 0.3)
    out += p("Bf", 0, 0.5) + p("hhh", 627, 1, 3) + p("ff", 5.0, 1.4)
    return out


def creatable(location: int) -> bytes:
    return p("9h", 0, 40, 1, 1, 20, 1, 4, 1, 0) + p("hhBh", 30, location, 1, 2)


def dat37_bytes(*, civ_count: int = 17, wall_name: bytes = b"Wall_Small") -> bytes:
    out = b"VER 3.7\0" + p("HH", 1, 2) + p("i", 7) + p("2f", 1.0, 0.5)
    out += p("H", 1) + p("30shhBB", b"Blue.col", 0, 0, 24, 1)
    out += p("H", 1) + p("hhHi", 5, 0, 1, 300000) + p("13sih", b"dock.wav", 5043, 100)
    out += p("H", 2) + p("ii", 0, 1)
    out += p("21s13s", b"Fidget Arch1", b"f_archr1") + p("iBBBhB", 413, 0, 0, 20, -1, 1) + p("4h", 1, 2, 3, 4)
    out += p("HhBHH", 1, -1, 1, 5, 8) + p("fff", 1.0, 0.1, 2.0) + p("BhB", 3, 1, 6)
    out += p("hhihhhh", 7, 0, 0, 2, 3, -1, 0) + p("6h", 2, 307, -1, -1, -1, -1) * 8
    # Terrain block: head, 19 tile sizes, padding, 32 terrains, 16 borders, tail.
    out += p("IIiiii", 0x577740, 0, 0, 0, 0, 0) + p("3h", 64, 32, 0) * 19 + p("h", 0)
    grass = p("BB13s13s", 1, 0, b"Grass", b"g_gr2") + p("3i", 15001, 0, 95) + p("3B2BBB", 99, 100, 101, 157, 159, 0, 16)
    grass += p("Bhhff", 0, 0, 0, 0.0, 0.0) + p("hhfBB", 0, 0, 0.0, 0, 0)
    grass += p("3h", 9, 1, 0) + b"".join(p("3h", 1, 1, 9 + k) for k in range(18)) + p("h2h", -1, 3, 3)
    grass += p("32h", 0, 3, *[0] * 30) + p("30h", 168, *[0] * 29) + p("30h", 10, *[0] * 29)
    grass += p("30B", 1, *[0] * 29) + p("hh", 1, 0)
    out += grass + bytes(len(grass)) * 31
    border = p("BB13s13s", 1, 0, b"Desert/water", b"gb_dw") + p("3i", 20000, 0, -1) + p("3BBhhffhhfBB", *[0] * 13)
    out += border + bytes(684 * 2) + p("3h", 1, 2, 3) + bytes(1440) * 15
    out += p("i14h", 0, 23, 7, 0, 64, 32, 16, 32, 16, *[0] * 6) + p("II4B", 0, 0, 0, 0, 1, 0) + p("6i", 256, *[0] * 5)
    out += p("Ii", 0, 0x025FFC88)  # no random maps
    out += p("i", 1) + p("31sH", b"A01", 1) + p("Bhhhf", 2, 13, 1, -1, 0.0)
    out += p("H", civ_count)
    academy = unit_head(80, 0, b"Academy\0") + moving(tasks=0) + creatable(118) + p("hbhbhhhhh", 82, 0, 0, 0, -1, -1, -1, 72, 0)
    wall = unit_head(80, 72, wall_name + b"\0") + moving(tasks=0) + creatable(118) + p("hbhbhhhhh", 77, 1, 0, 0, -1, -1, -1, -1, 189)
    hoplite = unit_head(70, 93, b"Soldier-Phal1\0") + moving() + creatable(0)
    arrow = unit_head(60, 9, b"Missle-Arrow\0") + moving(tasks=0) + p("bbbbbf", 0, 1, 0, 1, 0, 0.05)
    pointers = [0] * 94
    for unit_id in (0, 9, 72, 93):
        pointers[unit_id] = 1
    units = academy + arrow + wall + hoplite
    for k in range(civ_count):
        out += p("B20sHh", 1, b"Gaia" if k == 0 else b"Civ%d" % k, 2, -1) + p("2f", 200.0, 300.0) + p("B", 3)
        out += p("H", 94) + p("94i", *pointers) + units
    out += p("H", 1) + p("4h", 101, 0, -1, -1) + p("hhB", 0, 50, 1) + p("hhB", -1, 0, 0) * 2
    out += p("hhHHhhhhB", 1, 45, 7001, 8001, 30, 1, 1, 21, 4) + p("iiiH", 107001, 157001, 0, 15) + b"Light Transport"
    return out


def test_a_packed_rise_of_rome_dat_reads_every_section_back():
    dat = dat37.parse_bytes(dat37_bytes())
    assert dat.version == dat37.VERSION == "VER 3.7" and dat.layout == "aoe" and dat.terrains_used_1 == 2
    assert dat.float_ptr_terrain_tables == [7] and dat.terrain_pass_graphic_pointers == []
    assert dat.terrain_restrictions[0].passable_buildable_dmg_multiplier == [1.0, 0.5]
    assert dat.player_colours[0] == dat37.PlayerColour37("Blue.col", 0, 0, 24, 1)
    assert dat.sounds[0].items[0].filename == "dock.wav" and dat.sounds[0].items[0].civilization is None
    assert dat.graphics[0] is None
    g = dat.graphics[1]
    assert (g.name, g.file_name, g.slp, g.frame_count, g.angle_count, g.mirroring_mode, g.editor_flag) == (
        "Fidget Arch1", "f_archr1", 413, 5, 8, 6, None)
    assert g.deltas[0].graphic_id == 7 and g.angle_sounds[0].sound_id == 307
    assert dat.effects[0].name == "A01" and dat.random_maps.map_info_2 == []
    assert dat.tech_tree.tech_tree_ages == [] and dat.time_slice is None


def test_the_terrain_block_has_32_terrains_without_blend_fields():
    block = dat37.parse_bytes(dat37_bytes()).terrain_block
    assert len(block.terrains) == 32 and len(block.terrain_borders) == 16 and block.tile_sizes[0].width == 64
    grass = block.terrains[0]
    assert (grass.name, grass.slp, grass.sound_id, grass.blend_priority, grass.blend_type) == ("Grass", 15001, 95, None, None)
    assert grass.colors == (99, 100, 101) and grass.impassable_terrain == 16
    assert grass.frame_data[0].frame_count == 9 and grass.frame_data[18].shape_id == 26
    assert (grass.terrain_to_draw, grass.borders[:2], grass.terrain_unit_id[0]) == (-1, (0, 3), 168)
    assert grass.number_of_terrain_units_used == 1 and grass.string_id == -1
    assert block.terrain_borders[0].name == "Desert/water" and block.terrain_borders[0].border_style == 3
    assert (block.terrains_used_2, block.tile_width, block.map_min_x, block.fog_flag) == (23, 64, None, 1)
    assert block.some_int32[0] == 256


def test_units_read_the_narrower_1997_record():
    dat = dat37.parse_bytes(dat37_bytes())
    gaia = dat.civs[0]
    assert (gaia.name, gaia.team_bonus_id, gaia.resources, gaia.icon_set) == ("Gaia", None, [200.0, 300.0], 3)
    hoplite = gaia.units[93]
    assert (hoplite.type, hoplite.id, hoplite.name, hoplite.copy_id, hoplite.base_id) == (70, 93, "Soldier-Phal1", 93, None)
    assert (hoplite.standing_graphic, hoplite.dying_graphic, hoplite.damage_sound, hoplite.disabled) == ((493, -1), 490, None, None)
    assert (hoplite.hit_points, hoplite.dead_unit_id, hoplite.placement_terrain, hoplite.clearance_size) == (35, 3, (-3, -4), (0.5, 0.25))
    assert (hoplite.minimap_color, hoplite.hot_key, hoplite.resource_gather_group, hoplite.obstruction_type) == (9, 16093, 0, None)
    assert (hoplite.selection_effect, hoplite.editor_selection_colour) == (2, 52)
    assert hoplite.damage_graphics[0].graphic_id == 131 and hoplite.resource_storages[1].amount == -1.0
    assert hoplite.dead_fish.walking_graphic == 413 and hoplite.dead_fish.turn_radius is None
    assert hoplite.bird.drop_sites == [109, -1] and hoplite.bird.move_sound == 128
    t50 = hoplite.type_50
    assert (t50.base_armor, t50.attacks[0].amount, t50.armours[0].class_, t50.max_range) == (1, 4, 4, 5.0)
    assert (t50.projectile_unit_id, t50.min_range, t50.accuracy_dispersion, t50.displayed_attack) == (9, 0.5, None, 3)
    cr = hoplite.creatable
    assert cr.resource_costs[1].amount == 20 and cr.train_locations == [dat37.TrainLocation(30, 0, 1, 16093)]
    assert cr.displayed_pierce_armour == 2 and cr.hero_mode is None
    wall = gaia.units[72].building
    assert (wall.construction_graphic_id, wall.adjacent_mode, wall.construction_sound, wall.annexes) == (77, 1, 189, ())
    assert gaia.units[0].building.tech_id == 72
    assert gaia.units[9].projectile.projectile_arc == pytest.approx(0.05)


def test_inline_tasks_become_unit_headers_and_must_agree_across_civs():
    dat = dat37.parse_bytes(dat37_bytes())
    assert dat.unit_headers[93].exists == 1 and dat.unit_headers[93].task_list[0].action_type == 7
    assert dat.unit_headers[0].task_list == [] and dat.unit_headers[1].exists == 0
    raw = dat37_bytes()
    changed = raw[::-1].replace(p("hhbh", 1, 0, 0, 7)[::-1], p("hhbh", 1, 0, 0, 8)[::-1], 1)[::-1]
    with pytest.raises(ValueError, match="tasks differ between civs"):
        dat37.parse_bytes(changed)


def test_techs_keep_genieutils_six_required_slots():
    tech = dat37.parse_bytes(dat37_bytes()).techs[0]
    assert tech.name == "Light Transport" and tech.required_techs == (101, 0, -1, -1, -1, -1)
    assert (tech.language_dll_name, tech.language_dll_help, tech.effect_id, tech.icon_id, tech.civ) == (7001, 107001, 1, 21, None)
    assert tech.research_locations == [dat37.ResearchLocation(45, 30, 4, 0)]


def test_parse_inflates_and_the_checks_refuse_a_wrong_file(tmp_path):
    packer = zlib.compressobj(9, zlib.DEFLATED, -15)
    path = tmp_path / "empires.dat"
    path.write_bytes(packer.compress(dat37_bytes()) + packer.flush())
    assert dat37.parse(path).civs[0].units[72].name == "Wall_Small"
    with pytest.raises(ValueError, match="1 trailing bytes"):
        dat37.parse_bytes(dat37_bytes() + b"\0")
    with pytest.raises(ValueError, match="truncated"):
        dat37.parse_bytes(dat37_bytes()[:-3])
    with pytest.raises(ValueError, match=r"VER 5\.7"):
        dat37.parse_bytes(b"VER 5.7\0" + dat37_bytes()[8:])
    assert dat37.parse_bytes(dat37_bytes(civ_count=13)).civs[12].name == "Civ12"
    with pytest.raises(ValueError, match="expected 13 or 17 civs"):
        dat37.parse_bytes(dat37_bytes(civ_count=14))
    with pytest.raises(ValueError, match="'Wall_Big'"):
        dat37.parse_bytes(dat37_bytes(wall_name=b"Wall_Big"))


@pytest.mark.corpus
def test_the_installed_dats_parse_to_their_last_byte(request):
    root = request.config.getoption("--aoe1-dir")
    if not root:
        pytest.skip("no Age of Empires install: pass --aoe1-dir")
    base = dat37.parse(Path(root) / "data" / "Empires.dat")  # raises unless EOF and the self-checks hold
    ror = dat37.parse(Path(root) / "data2" / "empires.dat")
    assert (len(base.civs), len(ror.civs), len(base.techs), len(ror.techs)) == (13, 17, 118, 127)
    assert (len(base.graphics), len(ror.graphics), sum(h.exists for h in ror.unit_headers)) == (847, 941, 162)
    assert ror.civs[16].name == "Macedonian" and ror.civs[0].units[93].type_50.attacks
    assert [t.slp for t in ror.terrain_block.terrains if t.slp > 0] == [15001, 15002, 15000, 15003]
