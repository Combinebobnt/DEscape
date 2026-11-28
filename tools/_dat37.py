"""Read-only parser for the original Age of Empires' `VER 3.7` empires.dat (base game and Rise of Rome).

Stdlib only, so a frozen build can bundle it on its own. `parse(path)` inflates
the raw-deflate file and returns a `_dat57.DatFile`; `parse_bytes(inflated)`
takes the already inflated bytes. Both raise ValueError unless the parse ends
exactly at the file's last byte and the civ count and units 72 and 93 match
the game.

Every record is one of `_dat57`'s classes, under genieutils-py's field names,
so a generator written against genieutils-py (or `_dat57`) reads the same
attribute here. A field the 1997 record does not store reads as None, as
`_dat57` does for Age of Kings, except where `_dat57` already answers with a
DE property (`Terrain.string_id` and the like). Four shapes differ:

- `Unit.standing_graphic` is `(graphic, -1)`: the record stores one.
- Tasks are stored inline in each type 40+ unit, per civ.
  `DatFile.unit_headers[i]` holds the first civ's copy that has unit i; the
  self-check refuses a file where another civ's copy differs, so that one
  list stands for every civ.
- `Tech.required_techs` is the stored four plus two -1 (none), so it has
  genieutils-py's six.
- `PlayerColour` is this module's own `PlayerColour37`: the record is a palette
  name, an id and a minimap index, nothing like the later one.

There is no tech tree (an empty `TechTree`) and no globals block (None).

Layout provenance. Measured on the two shipped files by walking them: the
section order is `_dat57`'s minus the terrain pass graphics, the unit headers,
the globals and the tech tree; each record was narrowed field by field until
every unit id matched its slot and both files parsed to their last byte. Field
names in the narrowed records follow genieutils' Age of Empires gates as
recalled (no source could be fetched); the self-checks below and the names
the generators read (`name`, `id`, `standing_graphic`, ...) are what the walk
proves, not the naming of every same-width neighbour.

- data/Empires.dat (Age of Empires 1.0a)
  sha256 8c28f83df77bb23174f85b0bd5281d2f0a2f3539a0e47342a2ed757d6252502e
- data2/empires.dat (Rise of Rome 1.0a)
  sha256 d47c59d9ecaf5b83b467c647a69d28eced6a0dc5b2fa873e930c86136df5eb9d
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import _dat57
from _dat57 import (
    UNIT_TYPE_AOE_TREES,
    UNIT_TYPE_BIRD,
    UNIT_TYPE_BUILDING,
    UNIT_TYPE_COMBATANT,
    UNIT_TYPE_CREATABLE,
    UNIT_TYPE_DEAD_FISH,
    UNIT_TYPE_FLAG,
    UNIT_TYPE_PROJECTILE,
    AttackOrArmor,
    Bird,
    Building,
    Civ,
    Creatable,
    DamageGraphic,
    DatFile,
    DeadFish,
    Effect,
    EffectCommand,
    FrameData,
    Graphic,
    GraphicAngleSound,
    GraphicDelta,
    MapElevation,
    MapInfo,
    MapLand,
    MapTerrain,
    MapUnit,
    Projectile,
    RandomMaps,
    ResearchLocation,
    ResearchResourceCost,
    ResourceCost,
    ResourceStorage,
    Sound,
    SoundItem,
    Task,
    Tech,
    TechTree,
    Terrain,
    TerrainBlock,
    TerrainBorder,
    TerrainRestriction,
    TileSize,
    TrainLocation,
    Type50,
    Unit,
    UnitHeaders,
)

VERSION = "VER 3.7"

TILE_TYPE_COUNT = _dat57.TILE_TYPE_COUNT
TERRAIN_COUNT = 32
TERRAIN_BORDER_COUNT = _dat57.TERRAIN_BORDER_COUNT
TERRAIN_UNITS_SIZE = _dat57.TERRAIN_UNITS_SIZE
BORDER_FRAME_COUNT = _dat57.BORDER_FRAME_COUNT
STORED_REQUIRED_TECHS = 4
REQUIRED_TECHS = 6

# Pinned by the self-check: GAIA plus 12 civs in the base game, plus 16 in Rise of Rome.
BASE_CIV_COUNT = 13
ROR_CIV_COUNT = 17
# Two units every AoE1 .dat has: (internal name, type, where it is trained). DE's Return of Rome
# catalog has them as the Small Wall (a building the Villager, 118, builds) and the Hoplite (trained at
# the Academy, unit 0 here).
SPOT_UNITS = {72: ("Wall_Small", 80, 118), 93: ("Soldier-Phal1", 70, 0)}


def _s(fmt: str) -> struct.Struct:
    return struct.Struct("<" + fmt)


_U8 = _s("B")
_U16 = _s("H")
_I32 = _s("i")
_U32_I32 = _s("Ii")
_HEADER = _s("8sHH")
_PLAYER_COLOUR = _s("30shhBB")
_SOUND = _s("hhHi")
_SOUND_ITEM = _s("13sih")  # no civilization or icon set
_GRAPHIC = _s("21s13siBBBhB4hHhBHHfffBhB")  # no editor flag
_DELTA = _s("hhihhhh")
_ANGLE_SOUND = _s("6h")
_TILE_SIZE = _s("3h")
_TERRAIN_BLOCK_HEAD = _s("IIiiii")
# No blend priority or blend type after the sound id.
_TERRAIN = _s(f"BB13s13s3i3B2BBBBhhffhhfBB{TILE_TYPE_COUNT * 3}h3h{TERRAIN_COUNT}h"
              f"{TERRAIN_UNITS_SIZE}h{TERRAIN_UNITS_SIZE}h{TERRAIN_UNITS_SIZE}Bhh")
_BORDER = _s(f"BB13s13s3i3BBhhffhhfBB{TILE_TYPE_COUNT * BORDER_FRAME_COUNT * 3}h3h")
# map_row_offset, 14 shorts, the two search map pointers, four bytes, six runtime int32s; no map min/max floats.
_TERRAIN_BLOCK_TAIL = _s("i14hII4B6i")
_MAP_HEADER = _s("10i" + "Ii" * 4)
_MAP_BODY = _s("9i")
_MAP_LAND = _s("iIiiBBhiiBBhiii")
_MAP_TERRAIN = _s("6i")
_MAP_UNIT = _s("iiBBhiiiiiiii")
_MAP_ELEVATION = _s("6i")
_EFFECT = _s("31sH")
_EFFECT_COMMAND = _s("Bhhhf")
_TASK = _s("hhbhhhhhhhhfffbfbbhhbbbhhhhhh")
_CIV = _s("B20sHh")  # no team bonus
# One standing graphic; no damage sound, disabled, occlusion mode, obstruction type or class.
_UNIT_HEAD = _s("bHhHHhhhhbhfbfffhhbbhbhb2h2h2fbbhbhfbbbbbfbiii6bfff")
_RESOURCE_STORAGE = _s("hfb")
_DAMAGE_GRAPHIC = _s("hbbb")
_UNIT_SOUNDS = _s("hhbb")
_COPY_ID = _s("h")  # no base id
_FLOAT = _s("f")
_DEAD_FISH = _s("hhfbhbfb")  # no turn radius or yaw fields
_BIRD = _s("hff2hBhhB")
_ATTACK = _s("hh")
_TYPE50_TAIL = _s("hfffhhBh3fBfhhhff")  # no accuracy dispersion
_PROJECTILE = _s("bbbbbf")
_CREATABLE = _s("9hhhBh")  # costs, train time, location, button, displayed pierce armour
_BUILDING = _s("hbhbhhhhh")  # no snow graphic, annexes, transform, garrison or looting fields
_TECH = _s(f"{STORED_REQUIRED_TECHS}h" + "hhB" * 3 + "hhHHhhhhBiiiH")  # no civ or full tech mode


@dataclass(slots=True)
class PlayerColour37:
    name: str
    id: int
    resource_id: int
    minimap_color: int
    type: int


class _Reader:
    __slots__ = ("data", "pos", "view")

    def __init__(self, data: bytes) -> None:
        self.data = data
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

    def fixed(self, size: int) -> str:
        end = self.pos + size
        if end > len(self.data):
            raise ValueError(f"{VERSION} .dat truncated at offset {self.pos}: a {size}-byte string runs past the end")
        raw = self.data[self.pos:end]
        self.pos = end
        return _text(raw)


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


def _frames(values: tuple, start: int, count: int) -> list[FrameData]:
    return [FrameData(*values[i:i + 3]) for i in range(start, start + 3 * count, 3)]


def read_sound(r: _Reader) -> Sound:
    sound_id, play_delay, count, cache_time = r.take(_SOUND)
    items = []
    for _ in range(count):
        name, resource_id, probability = r.take(_SOUND_ITEM)
        items.append(SoundItem(_text(name), resource_id, probability, None, None))
    return Sound(sound_id, play_delay, cache_time, items)


def read_graphic(r: _Reader) -> Graphic:
    v = r.take(_GRAPHIC)
    delta_count, angle_sounds_used, angle_count = v[12], v[14], v[16]
    deltas = [GraphicDelta(*r.take(_DELTA)) for _ in range(delta_count)]
    angle_sounds = [GraphicAngleSound(*r.take(_ANGLE_SOUND)) for _ in range(angle_count)] if angle_sounds_used else []
    return Graphic(_text(v[0]), _text(v[1]), v[2], v[3], v[4], v[5], v[6], v[7], v[8:12], v[13], angle_sounds_used,
                   v[15], angle_count, v[17], v[18], v[19], v[20], v[21], v[22], None, deltas, angle_sounds)


def read_terrain(r: _Reader) -> Terrain:
    v = r.take(_TERRAIN)
    i = 24 + 3 * TILE_TYPE_COUNT
    borders_end = i + 3 + TERRAIN_COUNT
    units_end = borders_end + TERRAIN_UNITS_SIZE
    density_end = units_end + TERRAIN_UNITS_SIZE
    centering_end = density_end + TERRAIN_UNITS_SIZE
    return Terrain(
        v[0], v[1], _text(v[2]), _text(v[3]), v[4], v[5], v[6], None, None, v[7:10], v[10:12], v[12], v[13], v[14],
        v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23], _frames(v, 24, TILE_TYPE_COUNT), v[i],
        v[i + 1:i + 3], v[i + 3:borders_end], list(v[borders_end:units_end]), list(v[units_end:density_end]),
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
    padding_ts = r.one(_s("h"))
    terrains = [read_terrain(r) for _ in range(TERRAIN_COUNT)]
    borders = [read_terrain_border(r) for _ in range(TERRAIN_BORDER_COUNT)]
    t = r.take(_TERRAIN_BLOCK_TAIL)
    no_bounds = (None,) * 6
    return TerrainBlock(*head, tile_sizes, padding_ts, terrains, borders, t[0], *no_bounds, *t[1:20], t[20:21], t[21:])


def read_random_maps(r: _Reader) -> RandomMaps:
    count, ptr = r.take(_U32_I32)
    headers = []
    for _ in range(count):
        h = r.take(_MAP_HEADER)
        headers.append(MapInfo(*h[:12], [], h[12], h[13], [], h[14], h[15], [], h[16], h[17], []))
    maps = []
    for header in headers:
        body = r.take(_MAP_BODY)
        parts = []
        for st, cls in ((_MAP_LAND, MapLand), (_MAP_TERRAIN, MapTerrain), (_MAP_UNIT, MapUnit),
                        (_MAP_ELEVATION, MapElevation)):
            size, pointer = r.take(_U32_I32)
            parts += [size, pointer, [cls(*r.take(st)) for _ in range(size)]]
        maps.append(MapInfo(header.map_id, *body, *parts))
    return RandomMaps(ptr, headers, maps)


def read_effect(r: _Reader) -> Effect:
    name, count = r.take(_EFFECT)
    return Effect(_text(name), [EffectCommand(*r.take(_EFFECT_COMMAND)) for _ in range(count)])


def read_unit(r: _Reader) -> tuple[Unit, list[Task] | None]:
    """The unit, and its inline task list (None below type 40)."""
    v = r.take(_UNIT_HEAD)
    unit_type, name_length = v[0], v[1]
    storages = tuple(ResourceStorage(*r.take(_RESOURCE_STORAGE)) for _ in range(3))
    damage_graphics = [DamageGraphic(*r.take(_DAMAGE_GRAPHIC)) for _ in range(r.one(_U8))]
    selection_sound, dying_sound, old_attack_reaction, convert_terrain = r.take(_UNIT_SOUNDS)
    name = r.fixed(name_length)
    copy_id = r.one(_COPY_ID)
    # v: type, name length, id, dll name, dll creation, class, standing, dying, undead, undead mode, hit points,
    # sight, garrison, 3 collision, train sound, dead unit, sort, built on, icon, hide, portrait, enabled,
    # 2 side terrain, 2 terrain, 2 clearance, hill .. minimap colour (12), help, hotkey text, hot key,
    # recyclable, auto gather, doppelganger, gather group, selection effect, editor colour, 3 outline.
    unit = Unit(
        unit_type, v[2], v[3], v[4], v[5], (v[6], -1), v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15],
        v[16], None, v[17], v[18], v[19], v[20], v[21], v[22], v[23], None, v[24:26], v[26:28], v[28:30],
        *v[30:42], v[42], v[43], v[44], v[45], *v[46:50], None, None, None, None, None, None, v[50], v[51],
        *v[52:55], storages, damage_graphics, selection_sound, dying_sound, old_attack_reaction, convert_terrain,
        name, copy_id, None,
    )
    if unit_type == UNIT_TYPE_AOE_TREES or unit_type < UNIT_TYPE_FLAG:
        return unit, None
    unit.speed = r.one(_FLOAT)
    tasks = None
    if unit_type >= UNIT_TYPE_DEAD_FISH:
        unit.dead_fish = DeadFish(*r.take(_DEAD_FISH), None, None, None, None, None)
    if unit_type >= UNIT_TYPE_BIRD:
        b = r.take(_BIRD)
        unit.bird = Bird(b[0], b[1], b[2], list(b[3:5]), *b[5:])
        tasks = [Task(*r.take(_TASK)) for _ in range(r.one(_U16))]
    if unit_type >= UNIT_TYPE_COMBATANT:
        unit.type_50 = read_type_50(r)
    if unit_type == UNIT_TYPE_PROJECTILE:
        unit.projectile = Projectile(*r.take(_PROJECTILE))
    if unit_type >= UNIT_TYPE_CREATABLE:
        unit.creatable = read_creatable(r, unit.hot_key)
    if unit_type == UNIT_TYPE_BUILDING:
        unit.building = read_building(r)
    return unit, tasks


def read_type_50(r: _Reader) -> Type50:
    base_armor = r.one(_U8)
    attacks = [AttackOrArmor(*r.take(_ATTACK)) for _ in range(r.one(_U16))]
    armours = [AttackOrArmor(*r.take(_ATTACK)) for _ in range(r.one(_U16))]
    t = r.take(_TYPE50_TAIL)
    return Type50(base_armor, attacks, armours, *t[:8], t[8:11], t[11], t[12], None, *t[13:])


def read_creatable(r: _Reader, hot_key: int) -> Creatable:
    c = r.take(_CREATABLE)
    costs = tuple(ResourceCost(*c[i:i + 3]) for i in (0, 3, 6))
    train = [TrainLocation(c[9], c[10], c[11], hot_key)]
    return Creatable(costs, train, None, None, None, None, None, None, None, None, None, None, None, c[12])


def read_building(r: _Reader) -> Building:
    (construction, adjacent, angle, disappears, stack, foundation, overlap, tech_id,
     construction_sound) = r.take(_BUILDING)
    return Building(construction, None, adjacent, angle, disappears, stack, foundation, overlap, tech_id, None, (),
                    None, None, None, construction_sound, None, None, None, None, None)


def read_civ(r: _Reader) -> tuple[Civ, dict[int, list[Task]]]:
    player_type, name, resource_count, tech_tree_id = r.take(_CIV)
    resources = list(r.array("f", resource_count))
    icon_set = r.one(_U8)
    pointers = r.array("i", r.one(_U16))
    units: list[Unit | None] = []
    tasks: dict[int, list[Task]] = {}
    for unit_id, pointer in enumerate(pointers):
        if not pointer:
            units.append(None)
            continue
        unit, unit_tasks = read_unit(r)
        units.append(unit)
        if unit_tasks is not None:
            tasks[unit_id] = unit_tasks
    return Civ(player_type, _text(name), tech_tree_id, None, resources, icon_set, units), tasks


def read_tech(r: _Reader) -> Tech:
    v = r.take(_TECH)
    n = STORED_REQUIRED_TECHS
    required = (*v[:n], *(-1,) * (REQUIRED_TECHS - n))
    costs = tuple(ResearchResourceCost(*v[i:i + 3]) for i in (n, n + 3, n + 6))
    (required_count, location, dll_name, dll_description, research_time, effect_id, tech_type, icon_id, button_id,
     dll_help, dll_tech_tree, hot_key, name_length) = v[n + 9:]
    name = r.fixed(name_length)
    return Tech(required, costs, required_count, None, None, dll_name, dll_description, effect_id, tech_type, icon_id,
                dll_help, dll_tech_tree, name, [ResearchLocation(location, research_time, button_id, hot_key)])


def _unit_headers(civs: list[Civ], civ_tasks: list[dict[int, list[Task]]]) -> list[UnitHeaders]:
    """genieutils-py's per-unit task lists: the first civ's copy of each unit's inline tasks."""
    count = max((len(civ.units) for civ in civs), default=0)
    headers = []
    for unit_id in range(count):
        found = next((tasks[unit_id] for tasks in civ_tasks if unit_id in tasks), None)
        headers.append(UnitHeaders(0) if found is None else UnitHeaders(1, found))
    return headers


def parse_bytes(inflated: bytes) -> DatFile:
    """Parse an inflated `VER 3.7` .dat, raising ValueError on any layout or self-check failure."""
    r = _Reader(bytes(inflated))
    raw_version, restriction_count, terrains_used_1 = r.take(_HEADER)
    version = _text(raw_version)
    if version != VERSION:
        raise ValueError(f"not a {VERSION} .dat: the header says {version!r}")
    float_ptrs = list(r.array("i", restriction_count))
    restrictions = [TerrainRestriction(list(r.array("f", terrains_used_1)), []) for _ in range(restriction_count)]
    player_colours = []
    for _ in range(r.one(_U16)):
        name, *rest = r.take(_PLAYER_COLOUR)
        player_colours.append(PlayerColour37(_text(name), *rest))
    sounds = [read_sound(r) for _ in range(r.one(_U16))]
    graphic_pointers = r.array("i", r.one(_U16))
    graphics = [read_graphic(r) if pointer else None for pointer in graphic_pointers]
    terrain_block = read_terrain_block(r)
    random_maps = read_random_maps(r)
    effects = [read_effect(r) for _ in range(r.one(_I32))]
    civs, civ_tasks = [], []
    for _ in range(r.one(_U16)):
        civ, tasks = read_civ(r)
        civs.append(civ)
        civ_tasks.append(tasks)
    techs = [read_tech(r) for _ in range(r.one(_U16))]
    if r.pos != len(r.data):
        raise ValueError(f"{VERSION} .dat parse ended at offset {r.pos} of {len(r.data)}: "
                         f"{len(r.data) - r.pos} trailing bytes, so some record width is wrong")
    dat = DatFile(version, float_ptrs, [], restrictions, player_colours, sounds, graphics, terrain_block, random_maps,
                  effects, _unit_headers(civs, civ_tasks), civs, techs, None, None, None, None, None, None, None,
                  TechTree(0, [], [], [], []), terrains_used_1, "aoe")
    _self_check(dat, civ_tasks)
    return dat


def _self_check(dat: DatFile, civ_tasks: list[dict[int, list[Task]]]) -> None:
    if len(dat.civs) not in (BASE_CIV_COUNT, ROR_CIV_COUNT):
        raise ValueError(f"expected {BASE_CIV_COUNT} or {ROR_CIV_COUNT} civs (GAIA included) in an Age of Empires "
                         f".dat, found {len(dat.civs)}")
    units = dat.civs[0].units
    for unit_id, expected in SPOT_UNITS.items():
        unit = units[unit_id] if len(units) > unit_id else None
        found = None
        if unit is not None:
            found = (unit.name, unit.type, unit.creatable.train_locations[0].unit_id if unit.creatable else None)
        if found != expected:
            raise ValueError(f"expected unit {unit_id} to be {expected!r} in an Age of Empires .dat, found {found!r}")
    for header_id, header in enumerate(dat.unit_headers):
        if any(tasks.get(header_id, header.task_list) != header.task_list for tasks in civ_tasks):
            raise ValueError(f"unit {header_id}'s tasks differ between civs, so unit_headers cannot stand for them")


def parse(path: str | PathLike[str]) -> DatFile:
    """Inflate and parse a `VER 3.7` .dat file."""
    raw = Path(path).read_bytes()
    try:
        inflated = zlib.decompress(raw, -15)
    except zlib.error as exc:
        raise ValueError(f"{path}: not a raw-deflated .dat ({exc})") from None
    return parse_bytes(inflated)
