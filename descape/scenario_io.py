"""
Loads the Map and Units sections of an .aoe2scenario file via AoE2ScenarioParser,
without going anywhere near the Triggers section.

Why this exists: AoE2ScenarioParser's own AoE2DEScenario.from_file() refuses to load
scenario files where the internal scenario version is 1.54 and the trigger-data
sub-version is 3.9 (an old, structurally different trigger format the library can't
parse) -- it raises UnsupportedVersionError before returning anything at all, even
though Map and Units parse cleanly before it ever reaches Triggers. Four of the six
files in the Joan of Arc coop mod hit exactly this case.

The fix: replicate AoE2DEScenario.from_file()'s own steps, using the library's real
version-keyed structure definitions to walk sections in order, but stop right after
the Units section instead of continuing into Triggers. Everything from that exact byte
offset onward (Triggers and anything after) is kept as one opaque bytes blob -- never
parsed, never touched, and (for a future write path) spliced back verbatim.

This reaches into several of AoE2ScenarioParser's private methods, which is acceptable
only because the install is an editable clone pinned to a known commit -- see README.md.
Pinned commit: b763e2e37006bedab50c7d349b3ce24b9c2497f6 (tag v0.8.3).

Also captures what v2's write path (descape/scenario_write.py) needs: the raw header
bytes, the full original decompressed body, and the byte offset of the terrain struct
array within that body. Writing goes through a
direct byte patch of that array rather than AoE2ScenarioParser's own commit()/
write_to_file() -- the short version is that re-serializing through the library isn't
byte-stable (confirmed against this project's own example files), while patching the
terrain block in place and recompressing is.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from AoE2ScenarioParser import settings
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
from AoE2ScenarioParser.scenarios.aoe2_scenario import (
    _decompress_bytes,
    _get_file_version,
    _get_scenario_variant,
    _initialise_version_dependencies,
)
from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
from AoE2ScenarioParser.objects.aoe2_object_manager import AoE2ObjectManager
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager

# AoE2ScenarioParser prints its own "Parsing FileHeader... Gathering FileHeader
# data... FileHeader" progress lines to the console by default, using carriage
# returns to overwrite each line without clearing it first -- readable in some
# terminals, garbled leftover-text soup in others (the "done" line is shorter
# than the "in progress" line it overwrites, so remnants of the longer one
# show through). This is a debugging aid for library development, not
# something a GUI app's end users should see. Set once, here, since every
# entry point (map_editor.py, tools/*.py) imports this module.
settings.PRINT_STATUS_UPDATES = False

# The library also warns (plus prints blank-line padding around the warning,
# unconditionally alongside it -- see AoE2ScenarioParser's scenario_store/
# store.py) whenever two simultaneously-registered scenarios have different
# versions. That's a real footgun in general -- it's about cross-referencing
# data *between* two open scenarios using the library's higher-level helpers,
# which can misbehave across versions -- but nothing in this codebase does
# that. Every LoadedScenario's managers are used independently, and this tool
# routinely loads scenarios of different versions in the same process on
# purpose (e.g. the tests/ pytest suite's corpus tier, or just opening a
# different file in the viewer), so the warning is a false positive here
# every time it fires.
settings.SHOW_SCENARIO_VERSION_WARNINGS = False

# Never write into the Workshop-synced scenario folder under a Proton prefix -- it is
# the user's only copy of these files. scenario_write.write_scenario() checks this.
FORBIDDEN_WRITE_MARKER = "compatdata"

# The donor blank map descape/scenario_new.py splices to build every File > New
# Map size -- reused rather than built from AoE2ScenarioParser's own from_default(),
# so New goes through this same byte-patch load path instead of the library's own
# (unverified, non-byte-stable -- see this module's docstring) serializer. Never a
# write destination: scenario_write.write_scenario() refuses any path inside
# TEMPLATE_DIR.
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
BLANK_TEMPLATE_TILES = 120
BLANK_TEMPLATE_PATH = TEMPLATE_DIR / f"blank_{BLANK_TEMPLATE_TILES}x{BLANK_TEMPLATE_TILES}.aoe2scenario"

# Sizes with a shipped template on disk in TEMPLATE_DIR -- just the one donor now.
# The 240x240 and 480x480 exports that used to live here moved to
# tests/fixtures/real_blank_*.aoe2scenario: descape/scenario_new.py's splice
# reproduces them byte-for-byte from this donor (see its module docstring and
# tests/test_scenario_new.py), so keeping them as live templates would mean two
# code paths producing near-but-not-quite identical files for the same File > New
# size. They're kept as byte oracles instead -- real game exports, never
# regenerable, see tests/README.md. The menu's full size list is
# scenario_new.STANDARD_MAP_SIZES, a different concept from this constant.
BLANK_TEMPLATE_SIZES = (120,)


def blank_template_path(tiles: int) -> Path:
    return TEMPLATE_DIR / f"blank_{tiles}x{tiles}.aoe2scenario"

# TerrainStruct's on-disk layout (versions/DE/*/structure.json, identical across
# every DE structure version seen so far -- 1.41, 1.54, 1.55): u8 terrain_id,
# u8 elevation, 3 bytes unused (opaque, carried through untouched), s16 (little-
# endian) layer. Confirmed against every tile of every example file in this repo.
TERRAIN_STRUCT_SIZE = 7
_LAYER_STRUCT = struct.Struct("<h")  # offset 5 within one TerrainStruct


@dataclass
class LoadedScenario:
    path: Path  # display + Save-As-default only -- write_scenario() takes an explicit
    # destination and never reads this. descape/viewer.py may replace it with a
    # display-only sentinel that does not exist on disk (a File > New document).
    scenario_version: str
    map_manager: MapManager
    unit_manager: UnitManager
    trigger_tail: bytes  # Triggers section onward, byte-exact, never parsed

    # -- v2 write-path state, all needed by descape/scenario_write.py --
    header_bytes: bytes  # FileHeader, verbatim, uncompressed, written back as-is
    decompressed_body: bytes  # the *original* full decompressed body (Map..tail)
    terrain_block_offset: int  # byte offset of the terrain struct array, within
    # decompressed_body, at load time. Patching TERRAIN_STRUCT_SIZE * i bytes
    # starting here for each tile index i is the entire write path -- see
    # scenario_write.py. -1 if terrain_write_supported is False (see below).
    terrain_write_supported: bool  # False disables Edit mode's terrain/elevation
    # tools for this file without refusing to open it read-only -- see the
    # verification in load_map_and_units() for what can make this False.

    # -- units-strip state, used by tools/strip_units.py only -- everything else
    # in this codebase reads units through unit_manager, never these bytes.
    units_block_offset: int  # byte offset of the Units section's players_units
    # array (one PlayerUnitsStruct per unit section, unit_count u32 followed by
    # that many UnitStructs) within decompressed_body. -1 if units_write_supported
    # is False.
    units_section_end: int  # byte offset where the Units section ends, i.e.
    # where trigger_tail begins -- players_units is the section's last field
    # (see versions/DE/*/structure.json), same reasoning as terrain_data above.
    number_of_unit_sections: int  # length of players_units (9: GAIA + 8 players)
    units_write_supported: bool  # False if the raw per-section unit_count u32s
    # don't match the parsed counts -- see _verify_units_block().
    map_is_square: bool  # MapManager.set_elevation (and map_size, which it goes
    # through even for a single tile) raises ValueError whenever map_width !=
    # map_height -- confirmed directly against AoE2ScenarioParser's source, not
    # just observed. Both elevation tools must stay disabled when this is False;
    # terrain painting has no such constraint.

    # AoE2ScenarioParser registers scenarios in a WeakValueDictionary keyed by uuid;
    # several manager/tile properties (e.g. TerrainTile.x/.y) look themselves back up
    # by that uuid lazily. Without a strong reference here, the scenario is garbage
    # collected the moment load_map_and_units() returns and those lookups start
    # raising "Unable to find scenario based on the given identifier".
    _scenario: AoE2DEScenario


def _verify_terrain_block(decompressed: bytes, offset: int, terrain: list) -> bool:
    """True iff the TERRAIN_STRUCT_SIZE-byte struct at `offset` for every tile
    matches that tile's already-parsed terrain_id/elevation/layer -- the
    load-time trust check for the offset math in terrain_block_offset. Catches
    a future scenario structure version changing TerrainStruct's layout
    immediately, instead of silently patching the wrong bytes on save."""
    n = len(terrain)
    end = offset + TERRAIN_STRUCT_SIZE * n
    if offset < 0 or end > len(decompressed):
        return False
    for i, tile in enumerate(terrain):
        o = offset + TERRAIN_STRUCT_SIZE * i
        terrain_id = decompressed[o]
        elevation = decompressed[o + 1]
        (layer,) = _LAYER_STRUCT.unpack_from(decompressed, o + 5)
        if terrain_id != tile.terrain_id or elevation != tile.elevation or layer != tile.layer:
            return False
    return True


def _verify_units_block(decompressed: bytes, offset: int, players_units: list) -> bool:
    """True iff the u32 unit_count at the start of each PlayerUnitsStruct in
    `players_units` (as walked from `offset`) matches that struct's already-parsed
    unit_count, and the structs' byte_lengths sum to exactly the span consumed --
    the load-time trust check for units_block_offset, mirroring
    _verify_terrain_block() above."""
    if offset < 0 or offset > len(decompressed):
        return False
    o = offset
    for section in players_units:
        if o + 4 > len(decompressed):
            return False
        (raw_count,) = struct.unpack_from("<I", decompressed, o)
        if raw_count != section.retriever_map["unit_count"].data:
            return False
        o += section.byte_length
    return o <= len(decompressed)


def load_map_and_units(path: str | Path) -> LoadedScenario:
    """Parse the Map and Units sections of an .aoe2scenario file, skipping Triggers."""
    path = Path(path)
    return _load_map_and_units(path.read_bytes(), path)


def load_map_and_units_from_bytes(raw: bytes, display_path: str | Path) -> LoadedScenario:
    """Same parse as load_map_and_units(), for bytes that were never written to disk --
    descape/scenario_new.py's generated blank maps. display_path is used only for
    LoadedScenario.path and the library's source_location bookkeeping, neither of which
    is ever stat'd or opened; it need not exist."""
    return _load_map_and_units(bytes(raw), Path(display_path))


def _load_map_and_units(raw: bytes, path: Path) -> LoadedScenario:
    # IncrementalGenerator.from_file() is exactly `open(path,'rb').read()` followed by
    # this same constructor (name=str(path), file_content=raw) -- see its source in
    # AoE2ScenarioParser.helper.incremental_generator. Building it from already-read
    # bytes here, rather than calling from_file() and requiring a real file on disk, is
    # behaviour-preserving for load_map_and_units() and is what makes
    # load_map_and_units_from_bytes() possible.
    igen = IncrementalGenerator(str(path), raw)
    scenario_version = _get_file_version(igen)
    scenario_variant = _get_scenario_variant(igen)

    scenario = AoE2DEScenario(
        "DE", scenario_version, source_location=str(path), name="", variant=scenario_variant
    )
    scenario._load_structure()
    _initialise_version_dependencies(scenario.game_version, scenario.scenario_version)
    scenario._load_header_section(igen)
    # igen wraps the whole raw byte stream up front; progress is exactly the
    # header's length at this point, so this slice is the header's own verbatim
    # bytes -- no need to re-read them separately.
    header_bytes = igen.file_content[: igen.progress]

    decompressed = _decompress_bytes(igen.get_remaining_bytes())
    data_igen = IncrementalGenerator(name="Scenario Data", file_content=decompressed)
    scenario._decompressed_file_data = decompressed

    map_section_end = None
    units_section_end = None
    for section_name in scenario.structure.keys():
        if section_name == "FileHeader":
            continue
        scenario._create_and_load_section(section_name, data_igen)
        if section_name == "Map":
            map_section_end = data_igen.progress
        if section_name == "Units":
            # Capture progress here, not after the loop: IncrementalGenerator.
            # get_remaining_bytes() (used below for trigger_tail) has an
            # off-by-one that leaves .progress at len(file_content) - 1
            # afterward, so it can't be read post-hoc.
            units_section_end = data_igen.progress
            break

    trigger_tail = data_igen.get_remaining_bytes()

    map_manager = MapManager.construct(scenario.uuid)
    unit_manager = UnitManager.construct(scenario.uuid)

    # Skipping AoE2ObjectManager.setup() (it would also build a TriggerManager and
    # hit the same Triggers-parsing crash we're avoiding), but some lazy properties
    # -- e.g. TerrainTile.xy -- go through scenario.map_manager, which reads from
    # here. Populate just the two slots we actually constructed.
    scenario._object_manager = AoE2ObjectManager(scenario.uuid)
    scenario._object_manager.managers["Map"] = map_manager
    scenario._object_manager.managers["Unit"] = unit_manager

    # The terrain struct array is the Map section's last field (terrain_data,
    # after every other Map retriever -- see versions/DE/*/structure.json), so
    # it ends exactly where the Map section itself ends.
    w, h = map_manager.map_width, map_manager.map_height
    terrain_block_offset = map_section_end - TERRAIN_STRUCT_SIZE * w * h
    terrain_write_supported = _verify_terrain_block(
        decompressed, terrain_block_offset, map_manager.terrain
    )
    if not terrain_write_supported:
        terrain_block_offset = -1

    # players_units is the Units section's last field (see
    # versions/DE/*/structure.json), same reasoning as terrain_data above -- it
    # ends exactly where the Units section itself ends.
    players_units = scenario.sections["Units"].retriever_map["players_units"].data
    number_of_unit_sections = len(players_units)
    units_block_offset = units_section_end - sum(s.byte_length for s in players_units)
    units_write_supported = _verify_units_block(decompressed, units_block_offset, players_units)
    if not units_write_supported:
        units_block_offset = -1

    return LoadedScenario(
        path=path,
        scenario_version=scenario_version,
        map_manager=map_manager,
        unit_manager=unit_manager,
        trigger_tail=trigger_tail,
        header_bytes=header_bytes,
        decompressed_body=decompressed,
        terrain_block_offset=terrain_block_offset,
        terrain_write_supported=terrain_write_supported,
        units_block_offset=units_block_offset,
        units_section_end=units_section_end,
        number_of_unit_sections=number_of_unit_sections,
        units_write_supported=units_write_supported,
        map_is_square=(w == h),
        _scenario=scenario,
    )
