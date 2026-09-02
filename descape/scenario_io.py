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

import contextlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from AoE2ScenarioParser.objects.managers.trigger_manager import TriggerManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager

# Imported for its import-time side effect as much as for its API: library_compat
# snapshots the library's poisoned classes before any scenario has been loaded, and
# this module is the only one that loads scenarios, so importing it here is what
# guarantees the snapshot is clean. Never make this import lazy -- see that
# module's docstring.
from descape import library_compat
from descape.terrain_palette import resolve_player_colors

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
    terrain_write_supported: bool  # False disables Terrain mode's terrain/elevation
    # tools for this file without refusing to open it read-only -- see the
    # verification in load_map_and_units() for what can make this False.

    # -- units byte-offset state, used by tools/strip_units.py (which never
    # touches unit_manager) and by descape/unit_model.py's UnitEditModel
    # (phase 3.5a's write path, which reads units through unit_manager *and*
    # needs these raw offsets to slice each unit's original byte blob and to
    # splice the section back in place -- see that module's own docstring).
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
    # -- trigger read state (phase 4a). Everything here is additive; nothing in
    # the Map/Units path reads it, and trigger_tail still splices verbatim on
    # save exactly as before.
    options_section_end: int  # byte offset where the Options section ends, within
    # decompressed_body. Options.number_of_triggers is that section's last
    # retriever in all 19 DE structure versions, so the counter phase 4b has to
    # patch is the 4 bytes ending here. Recorded now because the walk that knows
    # it happens at load time and nowhere else.
    trigger_version: float  # the Triggers section's own f64 version, distinct
    # from scenario_version. -1.0 if trigger_tail is too short to hold one.
    triggers_section_end: int  # byte offset where the Triggers section ends,
    # within decompressed_body. -1 until parse_triggers() has run, since finding
    # it means parsing Triggers.
    trigger_read_supported: bool | None  # None = not attempted yet (the parse is
    # lazy). False means the Triggers section refused to parse, which for the
    # 1.54/trigger-3.9 set is expected and is not a reason to fail the open.
    trigger_write_supported: bool  # False until a full section walk has proven
    # exact byte alignment -- see _trigger_alignment_ok(). Phase 4b's write gate;
    # nothing writes triggers yet.
    _trigger_manager: TriggerManager | None  # parse_triggers()'s memo

    # -- Map Options read state. Both are trusted backward-walk anchors for
    # descape/options_model.py: GlobalVictory and Diplomacy are fully parsed
    # by the walk below like Options already was, but the byte offset where
    # each *ends* is only ever knowable from data_igen.progress at the
    # moment the walk passes through it -- nowhere else recovers it after
    # the fact. Unlike options_section_end, no existing subsystem already
    # needed these; they exist solely for descape/options_model.py.
    global_victory_section_end: int  # byte offset where the GlobalVictory
    # section ends, within decompressed_body.
    diplomacy_section_end: int  # byte offset where the Diplomacy section
    # ends, within decompressed_body.
    player_data_two_section_end: int  # byte offset where the PlayerDataTwo
    # section ends, within decompressed_body -- descape/player_fields.py's
    # anchor for a backward walk across resources/separator/ai_type/ai_files,
    # the same technique as the two above. Not derived from
    # global_victory_section_end minus a retriever sum: that sum's safety
    # depends on GlobalVictory containing no str16, which is true today and
    # is not something to depend on.

    # -- Messages read/write state (scenario prose). Unlike the anchors
    # above, both ends of the section are captured -- Messages is not
    # walked backward from a trusted anchor like options_model.py's fields;
    # a plain forward walk from messages_section_start already lands exactly
    # on messages_section_end on every corpus file, so that walk is also
    # this section's own verification gate. See descape/messages_model.py.
    messages_section_start: int  # data_igen.progress right after DataHeader.
    messages_section_end: int  # data_igen.progress right after Messages.
    header_instructions_span: tuple[int, int]  # (start, end) of
    # FileHeader.scenario_instructions' *payload* within header_bytes, from a
    # forward walk of every FileHeader retriever in true on-disk order
    # (structure.json's declaration order, not the alphabetical order
    # pprint()ing the dict suggests). (-1, -1) if that walk doesn't reconcile
    # to exactly len(header_bytes).
    messages_write_supported: bool  # False disables Messages mode's write
    # path for this file without refusing to open it read-only -- set by
    # _verify_messages_block() below, the same fail-closed shape as
    # terrain_write_supported/units_write_supported.

    map_is_square: bool  # MapManager.set_elevation (and map_size, which it goes
    # through even for a single tile) raises ValueError whenever map_width !=
    # map_height -- confirmed directly against AoE2ScenarioParser's source, not
    # just observed. Both elevation tools must stay disabled when this is False;
    # terrain painting has no such constraint.

    player_colors: tuple[tuple[int, int, int], ...]  # 9 entries, indexed by
    # player_id (0 = GAIA), from resolve_player_colors() against this file's
    # own stored PlayerDataTwo color overrides -- every render call site's
    # replacement for PLAYER_COLORS[player_id % len(PLAYER_COLORS)].
    team_indices: tuple[int, ...]  # 9 entries, indexed by player_id (0 =
    # GAIA), the sibling tuple from the same resolve_player_colors() call --
    # unit_sprites.TEAM_COLORS/sprite tint index for each player's real
    # sprite, distinct from player_colors because TEAM_COLORS is GAIA-first.

    # AoE2ScenarioParser registers scenarios in a WeakValueDictionary keyed by uuid;
    # several manager/tile properties (e.g. TerrainTile.x/.y) look themselves back up
    # by that uuid lazily. Without a strong reference here, the scenario is garbage
    # collected the moment load_map_and_units() returns and those lookups start
    # raising "Unable to find scenario based on the given identifier".
    _scenario: AoE2DEScenario


def retriever_length(retriever: Any) -> int:
    """Parsed byte length of one already-loaded retriever.

    Struct retrievers report their own parse-time byte_length, which is the
    only length that can be trusted here: re-serializing a struct that holds
    a str32 can change its length. Fixed-width retrievers have no such
    problem, so their re-serialized length is exact. Shared by
    descape/trigger_model.py (walking Triggers) and descape/options_model.py
    (walking GlobalVictory/Diplomacy/Map/Options) -- both need "how many
    bytes did this retriever actually occupy in the file as parsed", never a
    hardcoded struct size.
    """
    if retriever.datatype.type == "struct":
        return sum(entry.byte_length for entry in (retriever.data or []))
    return len(retriever.get_data_as_bytes())


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


# Messages' 12 retrievers, in true on-disk order (structure.json's own
# declaration order -- confirmed identical across all 19 DE structure
# versions plus this repo's own v1.21 copy): six u32 string-table IDs, then
# six str16 payloads in the same field order.
_MESSAGE_ID_FIELDS = ("instructions", "hints", "victory", "loss", "history", "scouts")
_MESSAGE_TEXT_FIELDS = (
    "ascii_instructions",
    "ascii_hints",
    "ascii_victory",
    "ascii_loss",
    "ascii_history",
    "ascii_scouts",
)
_MESSAGE_FIELD_ORDER = _MESSAGE_ID_FIELDS + _MESSAGE_TEXT_FIELDS
_STR16_PREFIX_SIZE = 2  # str16's length prefix is a 16-bit (2-byte) int

# FileHeader's own retrievers, in true on-disk order -- same caveat as above,
# confirmed the same way.
_HEADER_WALK_ORDER = (
    "version",
    "header_length",
    "savable",
    "timestamp_of_last_save",
    "scenario_instructions",
    "player_count",
    "unknown_value",
    "unknown_value_2",
    "amount_of_unknown_numbers",
    "unknown_numbers",
    "creator_name",
    "trigger_count",
)
_STR32_PREFIX_SIZE = 4  # str32's length prefix is a 32-bit (4-byte) int
_STR32_LENGTH_PREFIX_STRUCT = struct.Struct("<I")
# The two str32 fields in _HEADER_WALK_ORDER, read straight off the raw
# bytes' own 4-byte length prefix rather than via retriever_length() --
# see _header_instructions_span()'s docstring for why.
_HEADER_STR32_FIELDS = frozenset({"scenario_instructions", "creator_name"})


def _verify_messages_block(decompressed: bytes, start: int, end: int, retriever_map: dict) -> bool:
    """True iff a forward walk of the 12 Messages retrievers from `start`
    lands exactly on `end`, every str16 payload is byte-identical to
    parsed_value.encode('utf-8'), and no field parsed to bytes instead of
    str -- the load-time trust check for messages_section_start/_end,
    mirroring _verify_terrain_block()/_verify_units_block() above."""
    pos = start
    for name in _MESSAGE_FIELD_ORDER:
        retriever = retriever_map[name]
        length = retriever_length(retriever)
        if name in _MESSAGE_TEXT_FIELDS:
            value = retriever.data
            if not isinstance(value, str):
                return False
            payload = decompressed[pos + _STR16_PREFIX_SIZE : pos + length]
            if payload != value.encode("utf-8"):
                return False
        pos += length
    return pos == end


def _header_instructions_span(header_bytes: bytes, retriever_map: dict) -> tuple[int, int]:
    """Forward walk of every FileHeader retriever, reconciling to exactly
    len(header_bytes) -- the load-time trust check for
    header_instructions_span. Returns scenario_instructions' payload span
    (excluding its own 4-byte length prefix), or (-1, -1) if the walk
    doesn't reconcile (e.g. an older structure version with an extra field
    this walk doesn't know about -- header instructions editing degrades
    gracefully rather than trusting a wrong offset; the Messages copy is
    still editable).

    The two str32 fields read their length straight off the raw bytes'
    4-byte prefix rather than via retriever_length()/get_data_as_bytes():
    the latter re-serializes a NUL terminator onto an *empty* str32 that the
    real file never wrote one for (confirmed against scenario_instructions,
    empty on 24 of 25 corpus files: true on-disk length is the bare 4-byte
    zero prefix, but get_data_as_bytes() reports 5), so trusting it would
    overcount every empty field by one byte. Non-empty str32 fields (e.g.
    creator_name, which does carry a trailing NUL on disk) happen to match
    either way -- this reads the prefix directly for both so the two cases
    don't need distinguishing here.
    """
    pos = 0
    span = (-1, -1)
    for name in _HEADER_WALK_ORDER:
        if name in _HEADER_STR32_FIELDS:
            (payload_len,) = _STR32_LENGTH_PREFIX_STRUCT.unpack_from(header_bytes, pos)
            length = _STR32_PREFIX_SIZE + payload_len
        else:
            length = retriever_length(retriever_map[name])
        if name == "scenario_instructions":
            span = (pos + _STR32_PREFIX_SIZE, pos + length)
        pos += length
    if pos != len(header_bytes) or span == (-1, -1):
        return (-1, -1)
    return span


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

    # Unit is parsed below, not lazily like Triggers, so depoisoning it after
    # the walk is too late -- see library_compat.depoison()'s docstring.
    # Unconditional rather than uuid-gated like parse_triggers()'s calls: this
    # walk always parses Units, so there is no cheap case to skip it for.
    library_compat.depoison()

    map_section_end = None
    units_section_end = None
    options_section_end = -1
    global_victory_section_end = -1
    diplomacy_section_end = -1
    player_data_two_section_end = -1
    messages_section_start = -1
    messages_section_end = -1
    for section_name in scenario.structure:
        if section_name == "FileHeader":
            continue
        scenario._create_and_load_section(section_name, data_igen)
        if section_name == "DataHeader":
            messages_section_start = data_igen.progress
        if section_name == "Messages":
            messages_section_end = data_igen.progress
        if section_name == "PlayerDataTwo":
            # PlayerDataTwo, GlobalVictory and Diplomacy all sit before
            # Options in every DE structure version, so this costs nothing
            # extra -- the walk already passes through them.
            player_data_two_section_end = data_igen.progress
        if section_name == "GlobalVictory":
            global_victory_section_end = data_igen.progress
        if section_name == "Diplomacy":
            diplomacy_section_end = data_igen.progress
        if section_name == "Options":
            options_section_end = data_igen.progress
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

    player_colors, team_indices = resolve_player_colors(_read_player_colors(scenario))

    messages_retriever_map = scenario.sections["Messages"].retriever_map
    messages_write_supported = _verify_messages_block(
        decompressed, messages_section_start, messages_section_end, messages_retriever_map
    )
    header_instructions_span = _header_instructions_span(
        header_bytes, scenario.sections["FileHeader"].retriever_map
    )

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
        options_section_end=options_section_end,
        global_victory_section_end=global_victory_section_end,
        diplomacy_section_end=diplomacy_section_end,
        player_data_two_section_end=player_data_two_section_end,
        messages_section_start=messages_section_start,
        messages_section_end=messages_section_end,
        header_instructions_span=header_instructions_span,
        messages_write_supported=messages_write_supported,
        trigger_version=_read_trigger_version(trigger_tail),
        triggers_section_end=-1,
        trigger_read_supported=None,
        trigger_write_supported=False,
        _trigger_manager=None,
        map_is_square=(w == h),
        player_colors=player_colors,
        team_indices=team_indices,
        _scenario=scenario,
    )


def _read_player_colors(scenario: AoE2DEScenario) -> list[int]:
    """The 8 raw ColorId ints (P1..P8, in that order) PlayerDataTwo stores --
    resolve_player_colors()'s input. PlayerDataTwo precedes Map/Units in every
    DE structure version and is always fully parsed by the section walk
    above, so this needs no extra parsing of its own.

    resources[0..7] are players 1-8: PlayerResources' own RetrieverObjectLink
    group builds this list player-1-first (gaia_first=False), not GAIA-first
    -- resources[8] is GAIA's own (junk, never used) slot, and resources[9..
    15] is filler. Confirmed against every file in this project's corpus.
    """
    resources = scenario.sections["PlayerDataTwo"].retriever_map["resources"].data
    return [resources[i].retriever_map["player_color"].data for i in range(8)]


def _read_trigger_version(trigger_tail: bytes) -> float:
    """library_compat.trigger_version(), degraded to a sentinel rather than
    raising: a tail too short to hold one means a file with no parseable
    Triggers section, which must still open read-only."""
    try:
        return library_compat.trigger_version(trigger_tail)
    except ValueError:
        return -1.0


def _trigger_alignment_ok(trigger_tail: bytes, triggers_igen: IncrementalGenerator) -> bool:
    """True iff walking Triggers and every section after it consumed exactly
    len(trigger_tail) bytes, with nothing left over.

    This is the alignment oracle phase 4b's write gate needs, and it is
    deliberately stricter than a round-trip: a misaligned parse can
    re-serialize to a self-consistent wrong length, but it cannot land on the
    exact end of the section it was handed. Equivalent to walking the whole
    decompressed body, because trigger_tail is exactly
    decompressed_body[units_section_end:].

    Excludes v1.36/1.37 with no special-casing: the library models no Files
    section there, so a large remainder goes unconsumed (13.2 MB on
    0_June_Event_Scenario).
    """
    return triggers_igen.progress == len(trigger_tail)


# Which scenario the library's class-level trigger state is currently set up
# for. Module-global because the poisoning it tracks is global to the trigger
# classes, not per scenario -- see parse_triggers().
_active_trigger_uuid = None


def parse_triggers(loaded: LoadedScenario) -> TriggerManager | None:
    """Parse the Triggers section, or return None if it cannot be parsed.

    Lazy and memoized: nothing calls this until the trigger UI actually asks,
    because the largest file in the corpus carries a 1.17 MB Triggers section
    and opening a file for terrain editing must not pay for it.

    Returns None rather than raising for the 1.54/trigger-3.9 set, whose
    Triggers section the library cannot serialize at all ("Unable to convert
    NoneType with non-zero repeat to bytes"). Those files open and edit
    normally for terrain and units; only trigger reading is unavailable.

    **Call this again before reading a manager you obtained earlier.** The
    library's field gating is class-level and therefore global to the process:
    parsing scenario B disables B's unsupported fields on the same Condition/
    Effect classes a previously-returned manager for scenario A reads through,
    so A's objects start raising UnsupportedAttributeError with B's version
    quoted back. Re-calling is cheap (the parse is memoized, so it costs one
    depoison() and returns the same manager object), and it is the only thing
    that makes an earlier manager safe to read again.

    After a depoison() the classes are pristine rather than gated for this
    file's version, so a field that does not exist in this scenario version
    reads as None instead of raising. That trade is deliberate: a wrong-version
    error is a bug, a missing field reading as absent is not.
    """
    global _active_trigger_uuid

    scenario = loaded._scenario
    if loaded._trigger_manager is not None:
        if _active_trigger_uuid != scenario.uuid:
            library_compat.depoison()
            _active_trigger_uuid = scenario.uuid
        return loaded._trigger_manager
    if loaded.trigger_read_supported is False:
        return None
    try:
        # The library print()s a multi-thousand-line hex dump of the failing
        # struct before raising (aoe2_file_section.py:226, bytes_parser.py:72).
        # For the 1.54/3.9 set that failure is expected and handled, so the dump
        # is pure console noise -- the same reason PRINT_STATUS_UPDATES is off at
        # the top of this module. Swallowed only around the parse attempt.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            library_compat.depoison()
            # Triggers begins at offset 0 of trigger_tail: the Units break
            # captured units_section_end, and trigger_tail is the rest of the
            # body from there.
            igen = IncrementalGenerator(name="Triggers", file_content=loaded.trigger_tail)
            started = False
            triggers_end_in_tail = -1
            for section_name in scenario.structure:
                if not started and section_name != "Triggers":
                    continue
                started = True
                scenario._create_and_load_section(section_name, igen)
                if section_name == "Triggers":
                    triggers_end_in_tail = igen.progress
            manager = TriggerManager.construct(scenario.uuid)
    except (ValueError, KeyError, IndexError, TypeError, struct.error):
        # The shapes a misparse actually throws (the 1.54/3.9 set raises
        # ValueError). Deliberately not bare `Exception`: an AttributeError from
        # a library rename, or a MemoryError on the 1.17 MB file, is a real bug
        # and must surface rather than be reported to the user as "this file has
        # no triggers". A Triggers section this tool genuinely cannot parse must
        # still degrade instead of failing the open -- routing around exactly
        # that is why scenario_io.py exists.
        loaded.trigger_read_supported = False
        return None

    _active_trigger_uuid = scenario.uuid
    loaded.trigger_read_supported = True
    loaded.trigger_write_supported = _trigger_alignment_ok(loaded.trigger_tail, igen)
    loaded.triggers_section_end = loaded.units_section_end + triggers_end_in_tail
    scenario._object_manager.managers["Trigger"] = manager
    loaded._trigger_manager = manager
    return manager
