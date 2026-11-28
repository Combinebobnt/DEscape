"""Read-only parser for the original game's `VER 5.7` empires2*.dat (The Conquerors, Age of Kings, HD Edition).

Stdlib only, so a frozen build can bundle it on its own. `parse(path)` inflates
the raw-deflate file and returns a `DatFile`; `parse_bytes(inflated)` takes the
already inflated bytes. Both raise ValueError unless the parse ends exactly at
the file's last byte and the civ count and unit 4 (the Archer) match the game.

Every record class and field uses genieutils-py's name for it, so a generator
written against genieutils-py reads the same attribute here. Fields that only
DE has are absent; fields only this format has get descriptive snake_case
names (`TerrainBlock.terrain_borders`, `TerrainBlock.map_row_offset`,
`DamageGraphic.old_apply_mode` and the like, commented per class). The int8 signedness of each field follows
genieutils-py's: its `read_int_8` reads unsigned, its struct `b` signed.
Where genieutils-py 0.1.2's 7.x path synthesizes a field from fixed ones, this
does the same (`Creatable.train_locations`, `Tech.research_locations`).

Layout provenance. The section order and field names come from genieutils-py
0.1.2's DE reader; the AoC field widths come from the genieutils C++ `GV_TC`
layout and openage's gamedata docs as recalled, since neither source could be
fetched. Every width was then checked against real data: names decode as
ASCII, ids match their slot, and the parse ends exactly at the inflated file's
last byte for both AoC files:

- empires2_x1_p1.dat (AoC 1.0c)
  sha256 e49d05b326ecf4a14e0cddd5171718c6849abe2548939bb9a93a8f3039753d9d
- empires2_x1.dat (AoC 1.0)
  sha256 153428b9c367cbeea291457a6994c0b733e1635c332713bd94e4f54a7cca64b8

The Age of Kings 2.0 `empires2.dat` (sha256
cf585b48cb6e61b92c9934b850f0c27eaf4a76adb61771699c395beb83ab5295) is also
`VER 5.7` and also parses to its last byte under the `AOK` layout: 32 terrain
slots instead of 42, no `trait`/`civilization`/`nothing` on units, a uint8
`base_armor` and no `snow_graphic_id` (those read as None).

HD Edition 5.x's `resources/_common/dat/empires2_x2_p1.dat` (with The
Forgotten, African Kingdoms and Rise of the Rajas; sha256
21591ac67251d8674635d5634f2d8e9ff80ad90f3c792f9503d60dd658d61058) is `VER 5.7`
with 100 terrain slots and every other width AoC's: the `HD` layout parses it
to its last byte, 32 civs (GAIA plus DE's civs 1..31 in DE's order).
`_detect_layout` tells the three apart from the header's terrain count.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

VERSION = "VER 5.7"

TILE_TYPE_COUNT = 19
TERRAIN_COUNT = 42
TERRAIN_BORDER_COUNT = 16
TERRAIN_UNITS_SIZE = 30
BORDER_FRAME_COUNT = 12

HD_TERRAIN_COUNT = 100

# Pinned by the self-check: GAIA plus 18 civs (13 in Age of Kings, 31 in HD Edition), unit 4 the Archer.
AOC_CIV_COUNT = 19
AOK_CIV_COUNT = 14
HD_CIV_COUNT = 32
ARCHER_NAME = "ARCHR"

UNIT_TYPE_FLAG = 20
UNIT_TYPE_DEAD_FISH = 30
UNIT_TYPE_BIRD = 40
UNIT_TYPE_COMBATANT = 50
UNIT_TYPE_PROJECTILE = 60
UNIT_TYPE_CREATABLE = 70
UNIT_TYPE_BUILDING = 80
UNIT_TYPE_AOE_TREES = 90


def _s(fmt: str) -> struct.Struct:
    return struct.Struct("<" + fmt)


_U8 = _s("B")
_U16 = _s("H")
_I16 = _s("h")
_I32 = _s("i")
_U32_I32 = _s("Ii")
_HEADER = _s("8sHH")
_PLAYER_COLOUR = _s("9i")
_PASS_GRAPHIC = _s("4i")
_SOUND = _s("hhHi")
_SOUND_ITEM = _s("13sihhh")
_GRAPHIC = _s("21s13siBBBhB4hHhBHHfffBhBB")
_DELTA = _s("hhihhhh")
_ANGLE_SOUND = _s("6h")
_TILE_SIZE = _s("3h")
_TERRAIN_BLOCK_HEAD = _s("IIiiii")


def _terrain_struct(terrain_count: int) -> struct.Struct:
    return _s(f"BB13s13s5i3B2BBBBhhffhhfBB{TILE_TYPE_COUNT * 3}h3h{terrain_count}h"
              f"{TERRAIN_UNITS_SIZE}h{TERRAIN_UNITS_SIZE}h{TERRAIN_UNITS_SIZE}Bhh")


_BORDER = _s(f"BB13s13s3i3BBhhffhhfBB{TILE_TYPE_COUNT * BORDER_FRAME_COUNT * 3}h3h")
_TERRAIN_BLOCK_TAIL = _s("i6f14hIIBBB21B157i")
_MAP_HEADER = _s("10i" + "Ii" * 4)
_MAP_BODY = _s("9i")
_MAP_LAND = _s("iIiiBBhiiBBhiii")
_MAP_TERRAIN = _s("6i")
_MAP_UNIT = _s("iiBBhiiiiiiii")
_MAP_ELEVATION = _s("6i")
_EFFECT = _s("31sH")
_EFFECT_COMMAND = _s("Bhhhf")
_TASK = _s("hhbhhhhhhhhfffbfbbhhbbbhhhhhh")
_CIV = _s("B20sHhh")
_UNIT_HEAD = _s("bHhHHh2hhhbhfbfffhhhbbhbhbb2h2h2fbbhbhfbbbbbfbiiibbbbbbbbbhbbfff")
# Age of Kings lacks trait, civilization and nothing (`bbh`).
_UNIT_HEAD_AOK = _s("bHhHHh2hhhbhfbfffhhhbbhbhbb2h2h2fbbhbhfbbbbbfbiiibbbbbbbbbfff")
_AOK_UNIT_GAP = 56
_RESOURCE_STORAGE = _s("hfb")
_DAMAGE_GRAPHIC = _s("hbbb")
_UNIT_SOUNDS = _s("hhBB")
_ID_PAIR = _s("hh")
_FLOAT = _s("f")
_DEAD_FISH = _s("hhfbhbfbfffff")
_BIRD = _s("hff2hBhhB")
_ATTACK = _s("hh")
_TYPE50_TAIL = _s("hfffhhBh3fBffhhhff")
_PROJECTILE = _s("bbbbbf")
_RESOURCE_COST = _s("hhh")
_CREATABLE = _s("hhbffbblfb3fllbh")
_BUILDING_HEAD = _s("hhbhbhhhhb")
_BUILDING_HEAD_AOK = _s("hbhbhhhhb")  # no snow_graphic_id
_ANNEX = _s("hff")
_BUILDING_TAIL = _s("hhhhbffh6b")
_TECH = _s("6h" + "hhB" * 3 + "hhhhHHhhhhBiiiH")
_GLOBALS = _s("7i")
_TECH_TREE_HEAD = _s("BBBBi")
_ID_STATUS = _s("iB")
_COMMON = _s("21i")
_AGE_TAIL = _s("B10B10BBi")
_BUILDING_CONN_TAIL = _s("B5B5Bii")
_UNIT_CONN_HEAD = _s("iBi")
_UNIT_CONN_TAIL = _s("iiii")
_RESEARCH_CONN_HEAD = _s("iBi")
_RESEARCH_CONN_TAIL = _s("iii")


@dataclass(frozen=True, slots=True)
class _Layout:
    name: str
    terrain_count: int
    terrain: struct.Struct
    unit_head: struct.Struct
    base_armor: struct.Struct
    building_head: struct.Struct
    civ_count: int
    game: str


AOC = _Layout("aoc", TERRAIN_COUNT, _terrain_struct(TERRAIN_COUNT), _UNIT_HEAD, _I16, _BUILDING_HEAD, AOC_CIV_COUNT,
              "The Conquerors")
AOK = _Layout("aok", 32, _terrain_struct(32), _UNIT_HEAD_AOK, _U8, _BUILDING_HEAD_AOK, AOK_CIV_COUNT, "Age of Kings")
HD = _Layout("hd", HD_TERRAIN_COUNT, _terrain_struct(HD_TERRAIN_COUNT), _UNIT_HEAD, _I16, _BUILDING_HEAD, HD_CIV_COUNT,
             "HD Edition")


class _Reader:
    __slots__ = ("data", "layout", "pos", "view")

    def __init__(self, data: bytes, layout: _Layout = AOC) -> None:
        self.data = data
        self.layout = layout
        self.view = memoryview(data)
        self.pos = 0

    def take(self, st: struct.Struct) -> tuple:
        try:
            values = st.unpack_from(self.view, self.pos)
        except struct.error as exc:
            raise ValueError(f"{VERSION} .dat truncated at offset {self.pos}: {exc}") from None
        self.pos += st.size
        return values

    def one(self, st: struct.Struct):
        return self.take(st)[0]

    def array(self, kind: str, count: int) -> tuple:
        return self.take(_s(f"{count}{kind}")) if count else ()

    def str16(self) -> str:
        return self.fixed(self.one(_U16))

    def fixed(self, size: int) -> str:
        end = self.pos + size
        if end > len(self.data):
            raise ValueError(f"{VERSION} .dat truncated at offset {self.pos}: a {size}-byte string runs past the end")
        raw = self.data[self.pos:end]
        self.pos = end
        return _text(raw)


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


@dataclass(slots=True)
class TerrainPassGraphic:
    exit_tile_sprite_id: int
    enter_tile_sprite_id: int
    walk_tile_sprite_id: int
    walk_sprite_rate: int  # an int32 here (openage: replication_amount), a float in later games


@dataclass(slots=True)
class TerrainRestriction:
    passable_buildable_dmg_multiplier: list[float]
    terrain_pass_graphics: list[TerrainPassGraphic]


@dataclass(slots=True)
class PlayerColour:
    id: int
    player_color_base: int
    unit_outline_color: int
    unit_selection_color_1: int
    unit_selection_color_2: int
    minimap_color: int
    minimap_color_2: int
    minimap_color_3: int
    statistics_text: int


@dataclass(slots=True)
class SoundItem:
    filename: str
    resource_id: int
    probability: int
    civilization: int
    icon_set: int


@dataclass(slots=True)
class Sound:
    id: int
    play_delay: int
    cache_time: int
    items: list[SoundItem]


@dataclass(slots=True)
class GraphicDelta:
    graphic_id: int
    padding_1: int
    sprite_ptr: int
    offset_x: int
    offset_y: int
    display_angle: int
    padding_2: int


@dataclass(slots=True)
class GraphicAngleSound:
    frame_num: int
    sound_id: int
    frame_num_2: int
    sound_id_2: int
    frame_num_3: int
    sound_id_3: int


@dataclass(slots=True)
class Graphic:
    name: str
    file_name: str
    slp: int
    is_loaded: int
    old_color_flag: int
    layer: int
    player_color: int
    transparent_selection: int
    coordinates: tuple[int, int, int, int]
    sound_id: int
    angle_sounds_used: int
    frame_count: int
    angle_count: int
    speed_multiplier: float
    frame_duration: float
    replay_delay: float
    sequence_type: int
    id: int
    mirroring_mode: int
    editor_flag: int
    deltas: list[GraphicDelta]
    angle_sounds: list[GraphicAngleSound]


@dataclass(slots=True)
class FrameData:
    frame_count: int
    angle_count: int
    shape_id: int


@dataclass(slots=True)
class Terrain:
    enabled: int
    random: int
    name: str
    name_2: str
    slp: int
    shape_ptr: int
    sound_id: int
    blend_priority: int
    blend_type: int
    colors: tuple[int, int, int]
    cliff_colors: tuple[int, int]
    passable_terrain: int
    impassable_terrain: int
    is_animated: int
    animation_frames: int
    pause_frames: int
    interval: float
    pause_between_loops: float
    frame: int
    draw_frame: int
    animate_last: float
    frame_changed: int
    drawn: int
    frame_data: list[FrameData]
    terrain_to_draw: int
    terrain_dimensions: tuple[int, int]
    borders: tuple[int, ...]  # AoC only: the border id drawn against each other terrain
    terrain_unit_id: list[int]
    terrain_unit_density: list[int]
    terrain_unit_centering: list[int]
    number_of_terrain_units_used: int
    phantom: int

    # DE fields the 1999 record lacks, at the value meaning "none": no display string,
    # no editor hiding, no layering overlay mask (the 1999 game has none).
    @property
    def string_id(self) -> int:
        return -1

    @property
    def hide_in_editor(self) -> int:
        return 0

    @property
    def overlay_mask_name(self) -> str:
        return ""


@dataclass(slots=True)
class TerrainBorder:
    """AoC only (DE dropped borders): one of the 16 terrain border overlays."""

    enabled: int
    random: int
    name: str
    name_2: str
    slp: int
    shape_ptr: int
    sound_id: int
    colors: tuple[int, int, int]
    is_animated: int
    animation_frames: int
    pause_frames: int
    interval: float
    pause_between_loops: float
    frame: int
    draw_frame: int
    animate_last: float
    frame_changed: int
    drawn: int
    frame_data: list[list[FrameData]]  # TILE_TYPE_COUNT rows of BORDER_FRAME_COUNT
    draw_tile: int
    underlay_terrain: int
    border_style: int


@dataclass(slots=True)
class TileSize:
    width: int
    height: int
    delta_y: int


@dataclass(slots=True)
class TerrainBlock:
    virtual_function_ptr: int
    map_pointer: int
    map_width: int
    map_height: int
    world_width: int
    world_height: int
    tile_sizes: list[TileSize]
    padding_ts: int
    terrains: list[Terrain]
    terrain_borders: list[TerrainBorder]  # AoC only
    map_row_offset: int  # AoC only
    map_min_x: float
    map_min_y: float
    map_max_x: float
    map_max_y: float
    map_max_x_plus_1: float
    map_max_y_plus_1: float
    terrains_used_2: int
    borders_used: int
    max_terrain: int
    tile_width: int
    tile_height: int
    tile_half_height: int
    tile_half_width: int
    elev_height: int
    cur_row: int
    cur_col: int
    block_beg_row: int
    block_end_row: int
    block_beg_col: int
    block_end_col: int
    search_map_ptr: int
    search_map_rows_ptr: int
    any_frame_change: int
    map_visible_flag: int
    fog_flag: int
    some_bytes: tuple[int, ...]  # AoC only: 21 runtime bytes saved with the block
    some_int32: tuple[int, ...]  # AoC only: 157 runtime int32s (pointers, mostly zero)


@dataclass(slots=True)
class MapUnit:
    unit: int
    host_terrain: int
    group_placing: int
    scale_flag: int
    padding_1: int
    objects_per_group: int
    fluctuation: int
    groups_per_player: int
    group_arena: int
    player_id: int
    set_place_for_all_players: int
    min_distance_to_players: int
    max_distance_to_players: int


@dataclass(slots=True)
class MapTerrain:
    proportion: int
    terrain: int
    clump_count: int
    edge_spacing: int
    placement_terrain: int
    clumpiness: int


@dataclass(slots=True)
class MapLand:
    land_id: int
    terrain: int
    land_spacing: int
    base_size: int
    zone: int
    placement_type: int
    padding_1: int
    base_x: int
    base_y: int
    land_proportion: int
    by_player_flag: int
    padding_2: int
    start_area_radius: int
    terrain_edge_fade: int
    clumpiness: int


@dataclass(slots=True)
class MapElevation:
    proportion: int
    terrain: int
    clump_count: int
    base_terrain: int
    base_elevation: int
    tile_spacing: int


@dataclass(slots=True)
class MapInfo:
    """In `map_info_1` (the stored headers) the four lists are empty; `map_info_2`
    holds them, and its `map_id` is copied from the header, as the body omits it."""

    map_id: int
    border_south_west: int
    border_north_west: int
    border_north_east: int
    border_south_east: int
    border_usage: int
    water_shape: int
    base_terrain: int
    land_coverage: int
    unused_id: int
    map_lands_size: int
    map_lands_ptr: int
    map_lands: list[MapLand]
    map_terrains_size: int
    map_terrains_ptr: int
    map_terrains: list[MapTerrain]
    map_units_size: int
    map_units_ptr: int
    map_units: list[MapUnit]
    map_elevations_size: int
    map_elevations_ptr: int
    map_elevations: list[MapElevation]


@dataclass(slots=True)
class RandomMaps:
    random_maps_ptr: int
    map_info_1: list[MapInfo]
    map_info_2: list[MapInfo]


@dataclass(slots=True)
class EffectCommand:
    type: int
    a: int
    b: int
    c: int
    d: float


@dataclass(slots=True)
class Effect:
    name: str
    effect_commands: list[EffectCommand]


@dataclass(slots=True)
class Task:
    task_type: int
    id: int
    is_default: int
    action_type: int
    class_id: int
    unit_id: int
    terrain_id: int
    resource_in: int
    resource_multiplier: int
    resource_out: int
    unused_resource: int
    work_value_1: float
    work_value_2: float
    work_range: float
    auto_search_targets: int
    search_wait_time: float
    enable_targeting: int
    combat_level_flag: int
    gather_type: int
    work_flag_2: int
    target_diplomacy: int
    carry_check: int
    pick_for_construction: int
    moving_graphic_id: int
    proceeding_graphic_id: int
    working_graphic_id: int
    carrying_graphic_id: int
    resource_gathering_sound_id: int
    resource_deposit_sound_id: int


@dataclass(slots=True)
class UnitHeaders:
    exists: int
    task_list: list[Task] | None = None


@dataclass(slots=True)
class ResourceStorage:
    type: int
    amount: float
    flag: int


@dataclass(slots=True)
class DamageGraphic:
    graphic_id: int
    damage_percent: int
    old_apply_mode: int  # AoC only
    apply_mode: int


@dataclass(slots=True)
class DeadFish:
    walking_graphic: int
    running_graphic: int
    rotation_speed: float
    old_size_class: int
    tracking_unit: int
    tracking_unit_mode: int
    tracking_unit_density: float
    old_move_algorithm: int
    turn_radius: float
    turn_radius_speed: float
    max_yaw_per_second_moving: float
    stationary_yaw_revolution_time: float
    max_yaw_per_second_stationary: float


@dataclass(slots=True)
class Bird:
    default_task_id: int
    search_radius: float
    work_rate: float
    drop_sites: list[int]
    task_swap_group: int
    attack_sound: int
    move_sound: int
    run_pattern: int


@dataclass(slots=True)
class AttackOrArmor:
    class_: int
    amount: int


@dataclass(slots=True)
class Type50:
    base_armor: int
    attacks: list[AttackOrArmor]
    armours: list[AttackOrArmor]
    defense_terrain_bonus: int
    max_range: float
    blast_width: float
    reload_time: float
    projectile_unit_id: int
    accuracy_percent: int
    break_off_combat: int
    frame_delay: int
    graphic_displacement: tuple[float, float, float]
    blast_attack_level: int
    min_range: float
    accuracy_dispersion: float
    attack_graphic: int
    displayed_melee_armour: int
    displayed_attack: int
    displayed_range: float
    displayed_reload_time: float


@dataclass(slots=True)
class Projectile:
    projectile_type: int
    smart_mode: int
    hit_mode: int
    vanish_mode: int
    area_effect_specials: int
    projectile_arc: float


@dataclass(slots=True)
class ResourceCost:
    type: int
    amount: int
    flag: int


@dataclass(slots=True)
class TrainLocation:
    train_time: int
    unit_id: int
    button_id: int
    hot_key_id: int


@dataclass(slots=True)
class Creatable:
    resource_costs: tuple[ResourceCost, ResourceCost, ResourceCost]
    train_locations: list[TrainLocation]  # one entry, built from the fixed fields as genieutils-py does
    rear_attack_modifier: float
    flank_attack_modifier: float
    creatable_type: int
    hero_mode: int
    garrison_graphic: int
    total_projectiles: float
    max_total_projectiles: int
    projectile_spawning_area: tuple[float, float, float]
    secondary_projectile_unit: int
    special_graphic: int
    special_ability: int
    displayed_pierce_armour: int

    @property
    def hero_glow_graphic(self) -> int:
        """A DE field the 1999 record lacks: no glow graphic."""
        return -1


@dataclass(slots=True)
class BuildingAnnex:
    unit_id: int
    misplacement_x: float
    misplacement_y: float


@dataclass(slots=True)
class Building:
    construction_graphic_id: int
    snow_graphic_id: int | None  # None in Age of Kings, which lacks it
    adjacent_mode: int
    graphics_angle: int
    disappears_when_built: int
    stack_unit_id: int
    foundation_terrain_id: int
    old_overlap_id: int
    tech_id: int
    can_burn: int
    annexes: tuple[BuildingAnnex, BuildingAnnex, BuildingAnnex, BuildingAnnex]
    head_unit: int
    transform_unit: int
    transform_sound: int
    construction_sound: int
    garrison_type: int
    garrison_heal_rate: float
    garrison_repair_rate: float
    pile_unit: int
    looting_table: tuple[int, int, int, int, int, int]


@dataclass(slots=True)
class Unit:
    type: int
    id: int
    language_dll_name: int
    language_dll_creation: int
    class_: int
    standing_graphic: tuple[int, int]
    dying_graphic: int
    undead_graphic: int
    undead_mode: int
    hit_points: int
    line_of_sight: float
    garrison_capacity: int
    collision_size_x: float
    collision_size_y: float
    collision_size_z: float
    train_sound: int
    damage_sound: int
    dead_unit_id: int
    sort_number: int
    can_be_built_on: int
    icon_id: int
    hide_in_editor: int
    old_portrait_pict: int
    enabled: int
    disabled: int
    placement_side_terrain: tuple[int, int]
    placement_terrain: tuple[int, int]
    clearance_size: tuple[float, float]
    hill_mode: int
    fog_visibility: int
    terrain_restriction: int
    fly_mode: int
    resource_capacity: int
    resource_decay: float
    blast_defense_level: int
    combat_level: int
    interaction_mode: int
    minimap_mode: int
    interface_kind: int
    multiple_attribute_mode: float
    minimap_color: int
    language_dll_help: int
    language_dll_hotkey_text: int
    hot_key: int
    recyclable: int
    enable_auto_gather: int
    create_doppelganger_on_death: int
    resource_gather_group: int
    occlusion_mode: int
    obstruction_type: int
    obstruction_class: int
    trait: int | None  # None in Age of Kings, which lacks it (as civilization and nothing)
    civilization: int | None
    nothing: int | None
    selection_effect: int
    editor_selection_colour: int
    outline_size_x: float
    outline_size_y: float
    outline_size_z: float
    resource_storages: tuple[ResourceStorage, ResourceStorage, ResourceStorage]
    damage_graphics: list[DamageGraphic]
    selection_sound: int
    dying_sound: int
    old_attack_reaction: int
    convert_terrain: int
    name: str
    copy_id: int
    base_id: int
    speed: float | None = None
    dead_fish: DeadFish | None = None
    bird: Bird | None = None
    type_50: Type50 | None = None
    projectile: Projectile | None = None
    creatable: Creatable | None = None
    building: Building | None = None


@dataclass(slots=True)
class Civ:
    player_type: int
    name: str
    tech_tree_id: int
    team_bonus_id: int
    resources: list[float]
    icon_set: int
    units: list[Unit | None]


@dataclass(slots=True)
class ResearchResourceCost:
    type: int
    amount: int
    flag: int


@dataclass(slots=True)
class ResearchLocation:
    location_id: int
    research_time: int
    button_id: int
    hot_key_id: int


@dataclass(slots=True)
class Tech:
    required_techs: tuple[int, int, int, int, int, int]
    resource_costs: tuple[ResearchResourceCost, ResearchResourceCost, ResearchResourceCost]
    required_tech_count: int
    civ: int
    full_tech_mode: int
    language_dll_name: int
    language_dll_description: int
    effect_id: int
    type: int
    icon_id: int
    language_dll_help: int
    language_dll_tech_tree: int
    name: str
    research_locations: list[ResearchLocation]  # one entry, built from the fixed fields as genieutils-py does


@dataclass(slots=True)
class Common:
    slots_used: int
    unit_research: tuple[int, ...]
    mode: tuple[int, ...]


@dataclass(slots=True)
class TechTreeAge:
    id: int
    status: int
    buildings: list[int]
    units: list[int]
    techs: list[int]
    common: Common
    num_building_levels: int
    buildings_per_zone: tuple[int, ...]
    group_length_per_zone: tuple[int, ...]
    max_age_length: int
    line_mode: int


@dataclass(slots=True)
class BuildingConnection:
    id: int
    status: int
    buildings: list[int]
    units: list[int]
    techs: list[int]
    common: Common
    location_in_age: int
    units_techs_total: tuple[int, ...]
    units_techs_first: tuple[int, ...]
    line_mode: int
    enabling_research: int


@dataclass(slots=True)
class UnitConnection:
    id: int
    status: int
    upper_building: int
    common: Common
    vertical_line: int
    units: list[int]
    location_in_age: int
    required_research: int
    line_mode: int
    enabling_research: int


@dataclass(slots=True)
class ResearchConnection:
    id: int
    status: int
    upper_building: int
    buildings: list[int]
    units: list[int]
    techs: list[int]
    common: Common
    vertical_line: int
    location_in_age: int
    line_mode: int


@dataclass(slots=True)
class TechTree:
    total_unit_tech_groups: int
    tech_tree_ages: list[TechTreeAge]
    building_connections: list[BuildingConnection]
    unit_connections: list[UnitConnection]
    research_connections: list[ResearchConnection]


@dataclass(slots=True)
class DatFile:
    version: str
    float_ptr_terrain_tables: list[int]
    terrain_pass_graphic_pointers: list[int]
    terrain_restrictions: list[TerrainRestriction]
    player_colours: list[PlayerColour]
    sounds: list[Sound]
    graphics: list[Graphic | None]
    terrain_block: TerrainBlock
    random_maps: RandomMaps
    effects: list[Effect]
    unit_headers: list[UnitHeaders]
    civs: list[Civ]
    techs: list[Tech]
    time_slice: int
    unit_kill_rate: int
    unit_kill_total: int
    unit_hit_point_rate: int
    unit_hit_point_total: int
    razing_kill_rate: int
    razing_kill_total: int
    tech_tree: TechTree
    terrains_used_1: int = 0  # the per-restriction terrain count stored in the header
    layout: str = field(default="aoc")  # "aoc", "aok" or "hd"; see _detect_layout

    @property
    def graphic_pointers(self) -> list[int]:
        return [0 if g is None else 1 for g in self.graphics]


def read_terrain_restriction(r: _Reader, terrain_count: int) -> TerrainRestriction:
    multipliers = list(r.array("f", terrain_count))
    graphics = [TerrainPassGraphic(*r.take(_PASS_GRAPHIC)) for _ in range(terrain_count)]
    return TerrainRestriction(multipliers, graphics)


def read_sound(r: _Reader) -> Sound:
    sound_id, play_delay, count, cache_time = r.take(_SOUND)
    items = []
    for _ in range(count):
        name, *rest = r.take(_SOUND_ITEM)
        items.append(SoundItem(_text(name), *rest))
    return Sound(sound_id, play_delay, cache_time, items)


def read_graphic(r: _Reader) -> Graphic:
    v = r.take(_GRAPHIC)
    delta_count, angle_sounds_used, angle_count = v[12], v[14], v[16]
    deltas = [GraphicDelta(*r.take(_DELTA)) for _ in range(delta_count)]
    angle_sounds = [GraphicAngleSound(*r.take(_ANGLE_SOUND)) for _ in range(angle_count)] if angle_sounds_used else []
    return Graphic(_text(v[0]), _text(v[1]), v[2], v[3], v[4], v[5], v[6], v[7], v[8:12], v[13], angle_sounds_used,
                   v[15], angle_count, v[17], v[18], v[19], v[20], v[21], v[22], v[23], deltas, angle_sounds)


def _frames(values: tuple, start: int, count: int) -> list[FrameData]:
    return [FrameData(*values[i:i + 3]) for i in range(start, start + 3 * count, 3)]


def read_terrain(r: _Reader) -> Terrain:
    count = r.layout.terrain_count
    v = r.take(r.layout.terrain)
    i = 26 + 3 * TILE_TYPE_COUNT
    frames = _frames(v, 26, TILE_TYPE_COUNT)
    borders_end = i + 3 + count
    units_end = borders_end + TERRAIN_UNITS_SIZE
    density_end = units_end + TERRAIN_UNITS_SIZE
    centering_end = density_end + TERRAIN_UNITS_SIZE
    return Terrain(
        v[0], v[1], _text(v[2]), _text(v[3]), v[4], v[5], v[6], v[7], v[8], v[9:12], v[12:14], v[14], v[15], v[16],
        v[17], v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], frames, v[i], v[i + 1:i + 3],
        v[i + 3:borders_end], list(v[borders_end:units_end]), list(v[units_end:density_end]),
        list(v[density_end:centering_end]), v[centering_end], v[centering_end + 1],
    )


def read_terrain_border(r: _Reader) -> TerrainBorder:
    v = r.take(_BORDER)
    rows = [_frames(v, 20 + row * 3 * BORDER_FRAME_COUNT, BORDER_FRAME_COUNT) for row in range(TILE_TYPE_COUNT)]
    tail = 20 + 3 * BORDER_FRAME_COUNT * TILE_TYPE_COUNT
    return TerrainBorder(v[0], v[1], _text(v[2]), _text(v[3]), v[4], v[5], v[6], v[7:10], v[10], v[11], v[12], v[13],
                         v[14], v[15], v[16], v[17], v[18], v[19], rows, v[tail], v[tail + 1], v[tail + 2])


def read_terrain_block(r: _Reader) -> TerrainBlock:
    head = r.take(_TERRAIN_BLOCK_HEAD)
    tile_sizes = [TileSize(*r.take(_TILE_SIZE)) for _ in range(TILE_TYPE_COUNT)]
    padding_ts = r.one(_I16)
    terrains = [read_terrain(r) for _ in range(r.layout.terrain_count)]
    borders = [read_terrain_border(r) for _ in range(TERRAIN_BORDER_COUNT)]
    t = r.take(_TERRAIN_BLOCK_TAIL)
    return TerrainBlock(*head, tile_sizes, padding_ts, terrains, borders, *t[:26], t[26:47], t[47:])


def _map_list(r: _Reader, count: int, st: struct.Struct, cls: type) -> list:
    return [cls(*r.take(st)) for _ in range(count)]


def read_random_maps(r: _Reader) -> RandomMaps:
    count, ptr = r.take(_U32_I32)
    headers = []
    for _ in range(count):
        h = r.take(_MAP_HEADER)
        headers.append(MapInfo(*h[:12], [], h[12], h[13], [], h[14], h[15], [], h[16], h[17], []))
    maps = []
    for header in headers:
        body = r.take(_MAP_BODY)
        lands_size, lands_ptr = r.take(_U32_I32)
        lands = _map_list(r, lands_size, _MAP_LAND, MapLand)
        terrains_size, terrains_ptr = r.take(_U32_I32)
        terrains = _map_list(r, terrains_size, _MAP_TERRAIN, MapTerrain)
        units_size, units_ptr = r.take(_U32_I32)
        units = _map_list(r, units_size, _MAP_UNIT, MapUnit)
        elevations_size, elevations_ptr = r.take(_U32_I32)
        elevations = _map_list(r, elevations_size, _MAP_ELEVATION, MapElevation)
        maps.append(MapInfo(header.map_id, *body, lands_size, lands_ptr, lands, terrains_size, terrains_ptr, terrains,
                            units_size, units_ptr, units, elevations_size, elevations_ptr, elevations))
    return RandomMaps(ptr, headers, maps)


def read_effect(r: _Reader) -> Effect:
    name, count = r.take(_EFFECT)
    return Effect(_text(name), [EffectCommand(*r.take(_EFFECT_COMMAND)) for _ in range(count)])


def read_unit_header(r: _Reader) -> UnitHeaders:
    exists = r.one(_U8)
    if not exists:
        return UnitHeaders(exists)
    return UnitHeaders(exists, [Task(*r.take(_TASK)) for _ in range(r.one(_U16))])


def read_unit(r: _Reader) -> Unit:
    v = r.take(r.layout.unit_head)
    if r.layout is AOK:
        v = (*v[:_AOK_UNIT_GAP], None, None, None, *v[_AOK_UNIT_GAP:])
    unit_type, name_length = v[0], v[1]
    storages = tuple(ResourceStorage(*r.take(_RESOURCE_STORAGE)) for _ in range(3))
    damage_graphics = [DamageGraphic(*r.take(_DAMAGE_GRAPHIC)) for _ in range(r.one(_U8))]
    selection_sound, dying_sound, old_attack_reaction, convert_terrain = r.take(_UNIT_SOUNDS)
    name = r.fixed(name_length)
    copy_id, base_id = r.take(_ID_PAIR)
    unit = Unit(
        unit_type, v[2], v[3], v[4], v[5], v[6:8], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17],
        v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26], v[27:29], v[29:31], v[31:33], *v[33:],
        storages, damage_graphics, selection_sound, dying_sound, old_attack_reaction, convert_terrain, name, copy_id,
        base_id,
    )
    if unit_type == UNIT_TYPE_AOE_TREES or unit_type < UNIT_TYPE_FLAG:
        return unit
    unit.speed = r.one(_FLOAT)
    if unit_type >= UNIT_TYPE_DEAD_FISH:
        unit.dead_fish = DeadFish(*r.take(_DEAD_FISH))
    if unit_type >= UNIT_TYPE_BIRD:
        b = r.take(_BIRD)
        unit.bird = Bird(b[0], b[1], b[2], list(b[3:5]), *b[5:])
    if unit_type >= UNIT_TYPE_COMBATANT:
        unit.type_50 = read_type_50(r)
    if unit_type == UNIT_TYPE_PROJECTILE:
        unit.projectile = Projectile(*r.take(_PROJECTILE))
    if unit_type >= UNIT_TYPE_CREATABLE:
        unit.creatable = read_creatable(r, unit.hot_key)
    if unit_type == UNIT_TYPE_BUILDING:
        unit.building = read_building(r)
    return unit


def read_type_50(r: _Reader) -> Type50:
    base_armor = r.one(r.layout.base_armor)
    attacks = [AttackOrArmor(*r.take(_ATTACK)) for _ in range(r.one(_U16))]
    armours = [AttackOrArmor(*r.take(_ATTACK)) for _ in range(r.one(_U16))]
    t = r.take(_TYPE50_TAIL)
    return Type50(base_armor, attacks, armours, *t[:8], t[8:11], *t[11:])


def read_creatable(r: _Reader, hot_key: int) -> Creatable:
    costs = tuple(ResourceCost(*r.take(_RESOURCE_COST)) for _ in range(3))
    c = r.take(_CREATABLE)
    train = [TrainLocation(c[0], c[1], c[2], hot_key)]
    return Creatable(costs, train, *c[3:10], c[10:13], *c[13:])


def read_building(r: _Reader) -> Building:
    head = r.take(r.layout.building_head)
    if r.layout is AOK:
        head = (*head[:1], None, *head[1:])
    annexes = tuple(BuildingAnnex(*r.take(_ANNEX)) for _ in range(4))
    tail = r.take(_BUILDING_TAIL)
    return Building(*head, annexes, *tail[:8], tail[8:])


def read_civ(r: _Reader) -> Civ:
    player_type, name, resource_count, tech_tree_id, team_bonus_id = r.take(_CIV)
    resources = list(r.array("f", resource_count))
    icon_set = r.one(_U8)
    count = r.one(_U16)
    pointers = r.array("i", count)
    units = [read_unit(r) if pointer else None for pointer in pointers]
    return Civ(player_type, _text(name), tech_tree_id, team_bonus_id, resources, icon_set, units)


def read_tech(r: _Reader) -> Tech:
    v = r.take(_TECH)
    costs = tuple(ResearchResourceCost(*v[i:i + 3]) for i in (6, 9, 12))
    (required_count, civ, full_tech_mode, location, dll_name, dll_description, research_time, effect_id, tech_type,
     icon_id, button_id, dll_help, dll_tech_tree, hot_key, name_length) = v[15:]
    name = r.fixed(name_length)
    return Tech(v[:6], costs, required_count, civ, full_tech_mode, dll_name, dll_description, effect_id, tech_type,
                icon_id, dll_help, dll_tech_tree, name, [ResearchLocation(location, research_time, button_id, hot_key)])


def _id_list(r: _Reader) -> list[int]:
    return list(r.array("i", r.one(_U8)))


def _common(r: _Reader) -> Common:
    c = r.take(_COMMON)
    return Common(c[0], c[1:11], c[11:21])


def read_tech_tree(r: _Reader) -> TechTree:
    age_count, building_count, unit_count, research_count, total_groups = r.take(_TECH_TREE_HEAD)
    ages = []
    for _ in range(age_count):
        age_id, status = r.take(_ID_STATUS)
        buildings, units, techs = _id_list(r), _id_list(r), _id_list(r)
        common = _common(r)
        t = r.take(_AGE_TAIL)
        ages.append(TechTreeAge(age_id, status, buildings, units, techs, common, t[0], t[1:11], t[11:21], t[21], t[22]))
    building_connections = []
    for _ in range(building_count):
        conn_id, status = r.take(_ID_STATUS)
        buildings, units, techs = _id_list(r), _id_list(r), _id_list(r)
        common = _common(r)
        t = r.take(_BUILDING_CONN_TAIL)
        building_connections.append(
            BuildingConnection(conn_id, status, buildings, units, techs, common, t[0], t[1:6], t[6:11], t[11], t[12]))
    unit_connections = []
    for _ in range(unit_count):
        conn_id, status, upper = r.take(_UNIT_CONN_HEAD)
        common = _common(r)
        vertical_line = r.one(_I32)
        units = _id_list(r)
        unit_connections.append(UnitConnection(conn_id, status, upper, common, vertical_line, units, *r.take(_UNIT_CONN_TAIL)))
    research_connections = []
    for _ in range(research_count):
        conn_id, status, upper = r.take(_RESEARCH_CONN_HEAD)
        buildings, units, techs = _id_list(r), _id_list(r), _id_list(r)
        common = _common(r)
        research_connections.append(
            ResearchConnection(conn_id, status, upper, buildings, units, techs, common, *r.take(_RESEARCH_CONN_TAIL)))
    return TechTree(total_groups, ages, building_connections, unit_connections, research_connections)


def parse_bytes(inflated: bytes) -> DatFile:
    """Parse an inflated `VER 5.7` .dat, raising ValueError on any layout or self-check failure."""
    r = _Reader(bytes(inflated))
    raw_version, restriction_count, terrains_used_1 = r.take(_HEADER)
    version = _text(raw_version)
    if version != VERSION:
        raise ValueError(f"not a {VERSION} .dat: the header says {version!r}")
    r.layout = _detect_layout(terrains_used_1)
    float_ptrs = list(r.array("i", restriction_count))
    pass_ptrs = list(r.array("i", restriction_count))
    restrictions = [read_terrain_restriction(r, terrains_used_1) for _ in range(restriction_count)]
    player_colours = [PlayerColour(*r.take(_PLAYER_COLOUR)) for _ in range(r.one(_U16))]
    sounds = [read_sound(r) for _ in range(r.one(_U16))]
    graphic_pointers = r.array("i", r.one(_U16))
    graphics = [read_graphic(r) if pointer else None for pointer in graphic_pointers]
    terrain_block = read_terrain_block(r)
    random_maps = read_random_maps(r)
    effects = [read_effect(r) for _ in range(r.one(_I32))]
    unit_headers = [read_unit_header(r) for _ in range(r.one(_I32))]
    civs = [read_civ(r) for _ in range(r.one(_U16))]
    techs = [read_tech(r) for _ in range(r.one(_U16))]
    globals_ = r.take(_GLOBALS)
    tech_tree = read_tech_tree(r)
    if r.pos != len(r.data):
        raise ValueError(f"{VERSION} .dat parse ended at offset {r.pos} of {len(r.data)}: "
                         f"{len(r.data) - r.pos} trailing bytes, so some record width is wrong")
    dat = DatFile(version, float_ptrs, pass_ptrs, restrictions, player_colours, sounds, graphics, terrain_block,
                  random_maps, effects, unit_headers, civs, techs, *globals_, tech_tree, terrains_used_1,
                  r.layout.name)
    _self_check(dat, r.layout)
    return dat


def _detect_layout(terrains_used: int) -> _Layout:
    """Age of Kings has 32 terrain slots and uses all 32; The Conquerors has 42 and uses 41; HD Edition uses 100."""
    if terrains_used > AOC.terrain_count:
        return HD
    return AOC if terrains_used > AOK.terrain_count else AOK


def _self_check(dat: DatFile, layout: _Layout) -> None:
    if len(dat.civs) != layout.civ_count:
        raise ValueError(f"expected {layout.civ_count} civs (GAIA included) in the {layout.game} .dat, "
                         f"found {len(dat.civs)}")
    archer = dat.civs[0].units[4] if len(dat.civs[0].units) > 4 else None
    if archer is None or archer.name != ARCHER_NAME:
        found = None if archer is None else archer.name
        raise ValueError(f"expected unit 4 to be {ARCHER_NAME!r} in the {layout.game} .dat, found {found!r}")


def parse(path: str | PathLike[str]) -> DatFile:
    """Inflate and parse a `VER 5.7` .dat file."""
    raw = Path(path).read_bytes()
    try:
        inflated = zlib.decompress(raw, -15)
    except zlib.error as exc:
        raise ValueError(f"{path}: not a raw-deflated .dat ({exc})") from None
    return parse_bytes(inflated)
