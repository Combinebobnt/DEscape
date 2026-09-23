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
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from AoE2ScenarioParser import settings
from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
from AoE2ScenarioParser.objects.aoe2_object_manager import AoE2ObjectManager
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.trigger_manager import TriggerManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
from AoE2ScenarioParser.scenarios.aoe2_scenario import (
    _decompress_bytes,
    _get_file_version,
    _get_scenario_variant,
    _initialise_version_dependencies,
)

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


# Structure definitions this repo authors for scenario versions the library
# ships none for (today only v1.21). The library's own definition always wins.
REPO_VERSIONS_DIR = Path(__file__).resolve().parent / "versions" / "DE"


def _repo_structure_path(scenario_version: str) -> Path | None:
    """This repo's structure.json for `scenario_version`, or None if the
    library ships one itself (it wins) or neither does."""
    name = Path(f"v{scenario_version}") / "structure.json"
    if (library_compat.VERSIONS_DIR / name).is_file():
        return None
    path = REPO_VERSIONS_DIR / name
    return path if path.is_file() else None


def structure_is_available(scenario_version: str) -> bool:
    """True if load_map_and_units() has a structure for `scenario_version`:
    the library's own, or else one from REPO_VERSIONS_DIR. Says nothing about
    triggers; library_compat.vocabulary_is_available() is that probe."""
    name = Path(f"v{scenario_version}") / "structure.json"
    return (library_compat.VERSIONS_DIR / name).is_file() or (REPO_VERSIONS_DIR / name).is_file()


def _version_key(scenario_version: str) -> tuple[int, ...] | None:
    """`"1.21"` -> `(1, 21)`, for ordering. None if it isn't dotted numbers
    (viewer.py's File > New sentinel, or anything else unexpected)."""
    try:
        return tuple(int(part) for part in scenario_version.split("."))
    except ValueError:
        return None


def unsupported_version_sentence(scenario_version: str) -> str:
    """One sentence naming why AoE2ScenarioParser has no definitions for
    `scenario_version`. Shared by the load-failure modal and the trigger
    panel's repo-structure text so the two never disagree about the cause.

    The "older than" phrasing is measured against what the installed library
    actually ships, not assumed: everything unsupported today is older, but a
    scenario version newer than the pinned library would land here too."""
    key = _version_key(scenario_version)
    shipped = [k for k in (_version_key(p.name[1:]) for p in library_compat.VERSIONS_DIR.glob("v*")) if k]
    if key and shipped and key < min(shipped):
        return f"Scenario version {scenario_version} is older than any version AoE2ScenarioParser supports."
    return f"Scenario version {scenario_version} is not a version AoE2ScenarioParser supports."


def unsupported_structure_message(scenario_version: str) -> str:
    """The "Failed to load" modal's body, in place of the library's raw
    `UnknownScenarioStructureError: ... :(` text."""
    repo = sorted(p.name[1:] for p in REPO_VERSIONS_DIR.glob("v*") if (p / "structure.json").is_file())
    supplied = f" DEscape supplies its own definition for version {', '.join(repo)}, but not for this one." if repo else ""
    return (
        "DEscape can't open this file.\n\n"
        f"{unsupported_version_sentence(scenario_version)}{supplied} None of the file "
        "can be read, so there is nothing to show."
    )


class UnsupportedStructureVersion(Exception):
    """A scenario version neither the library nor this repo ships a structure
    definition for (the DE:1.32/1.35 engine test content), raised before any
    section is parsed. `str()` is the user-facing explanation, so a caller with
    no UI of its own can report it as-is."""

    def __init__(self, scenario_version: str) -> None:
        self.scenario_version = scenario_version
        super().__init__(unsupported_structure_message(scenario_version))


# The blank template's TerrainStruct stride, for descape/scenario_new.py's
# splice only: u8 terrain_id, u8 elevation, 3 bytes unused, s16 layer. Not
# universal -- v1.21's TerrainStruct is 3 bytes with no layer, so a loaded
# file's stride is LoadedScenario.terrain_struct_size, read off its own parse.
TERRAIN_STRUCT_SIZE = 7
_LAYER_STRUCT = struct.Struct("<h")  # offset 5 within a 7-byte TerrainStruct
_LAYER_OFFSET = 5


# eq=False (Batch D's D1b): identity equality/hash, so this can key a
# WeakKeyDictionary memo in render.py -- default eq=True sets __hash__ to
# None, and no code anywhere compares two LoadedScenario instances by value.
@dataclass(eq=False)
class LoadedScenario:
    path: Path  # display + Save-As-default only -- write_scenario() takes an explicit
    # destination and never reads this. descape/viewer.py may replace it with a
    # display-only sentinel that does not exist on disk (a File > New document).
    scenario_version: str
    structure_source: str  # "library", or "repo" for a version only
    # REPO_VERSIONS_DIR defines (v1.21). A repo-structure file has no trigger
    # vocabulary, so parse_triggers() never attempts it.
    map_manager: MapManager
    unit_manager: UnitManager
    trigger_tail: bytes  # Triggers section onward, byte-exact, never parsed

    # -- v2 write-path state, all needed by descape/scenario_write.py --
    header_bytes: bytes  # FileHeader, verbatim, uncompressed, written back as-is
    decompressed_body: bytes  # the *original* full decompressed body (Map..tail)
    original_compressed_body: bytes  # the exact compressed bytes the file shipped
    # with (or was constructed with, for load_map_and_units_from_bytes), reused
    # verbatim by scenario_write.py when nothing changed -- recompressing the
    # same decompressed bytes does not reliably reproduce them byte-for-byte.
    terrain_block_offset: int  # byte offset of the terrain struct array, within
    # decompressed_body, at load time. Patching terrain_struct_size * i bytes
    # starting here for each tile index i is the entire write path -- see
    # scenario_write.py. -1 if terrain_write_supported is False (see below).
    terrain_write_supported: bool  # False disables Terrain mode's terrain/elevation
    # tools for this file without refusing to open it read-only -- see the
    # verification in load_map_and_units() for what can make this False.
    terrain_struct_size: int  # bytes per TerrainStruct in this file, read off the
    # parsed terrain_data (7 on DE, 3 on v1.21). 0 if the division did not
    # come out exact, which also makes terrain_write_supported False.
    terrain_has_layer: bool  # whether this file's TerrainStruct carries `layer`.
    # False means tile.layer is TerrainTile's -1 default, never from the file,
    # so neither the verify nor the write path may touch a layer field.

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
    # where trigger_tail begins. Not where players_units ends: v1.21 declares
    # number_of_players and player_data_3 after it (see players_units_end).
    players_units_end: int  # byte offset one past the players_units array.
    # Equal to units_section_end on every DE version; the bytes between the two
    # are carried through verbatim on a units splice. -1 if
    # units_write_supported is False.
    number_of_unit_sections: int  # length of players_units (9: GAIA + 8 players)
    units_write_supported: bool  # False if the raw per-section unit_count u32s
    # don't match the parsed counts -- see _verify_units_block().
    # -- trigger read state (phase 4a). Everything here is additive; nothing in
    # the Map/Units path reads it, and trigger_tail still splices verbatim on
    # save exactly as before.
    options_section_end: int  # byte offset where the Options section ends, within
    # decompressed_body. Options.number_of_triggers is that section's last
    # retriever in all 19 DE structure versions (v1.21 has none -- see
    # has_trigger_counters), so the counter phase 4b patches is the 4 bytes
    # ending here. Recorded now because the walk that knows it happens at load
    # time and nowhere else.
    has_trigger_counters: bool  # whether this file has Options.number_of_triggers
    # and FileHeader.trigger_count at all, read off the parsed sections. False
    # on v1.21, where those 4-byte tails are ordinary content (per-player
    # starting age, unknown_numbers) that no trigger-count patch may touch.
    trigger_version: float  # the Triggers section's own f64 version, distinct
    # from scenario_version. -1.0 if trigger_tail is too short to hold one.
    triggers_section_end: int  # byte offset where the Triggers section ends,
    # within decompressed_body. -1 until parse_triggers() has run, since finding
    # it means parsing Triggers.
    trigger_read_supported: bool | None  # None = not attempted yet (the parse is
    # lazy). False means the Triggers section refused to parse, which for the
    # 1.54/trigger-3.9 set is expected and is not a reason to fail the open,
    # or that it was never attempted (structure_source "repo").
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
    header_player_count_span: tuple[int, int]  # (start, end) of
    # FileHeader.player_count within header_bytes, from the same walk, and
    # (-1, -1) on the same failure -- Number of Players' second buffer, see
    # descape/player_fields.py's player_count_targets().
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

    # Bumped by every UnitEditModel mutator (descape/unit_model.py's
    # _bump_unit_gen(), Batch D's D1a) -- never by anything else. A memo
    # keyed on this is only valid against mutations that went through
    # UnitEditModel: appending straight to unit_manager.units (as some test
    # fixtures do) does not bump it, and a memo built before such an append
    # is stale by design -- callers that mutate the list directly must bump
    # this themselves. Placed last, with a default, because every other
    # field above is positional with no default and dataclass field order
    # requires defaulted fields to come after all non-defaulted ones.
    unit_gen: int = 0


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


def _verify_terrain_block(
    decompressed: bytes, offset: int, terrain: list, stride: int, has_layer: bool
) -> bool:
    """True iff the `stride`-byte struct at `offset` for every tile matches
    that tile's already-parsed terrain_id/elevation (and layer, only if
    `has_layer`) -- the load-time trust check for terrain_block_offset.
    Catches a structure version changing TerrainStruct's layout immediately,
    instead of silently patching the wrong bytes on save."""
    if stride < 2 or (has_layer and stride < _LAYER_OFFSET + _LAYER_STRUCT.size):
        return False
    n = len(terrain)
    end = offset + stride * n
    if offset < 0 or end > len(decompressed):
        return False
    for i, tile in enumerate(terrain):
        o = offset + stride * i
        if decompressed[o] != tile.terrain_id or decompressed[o + 1] != tile.elevation:
            return False
        if has_layer and _LAYER_STRUCT.unpack_from(decompressed, o + _LAYER_OFFSET)[0] != tile.layer:
            return False
    return True


def _terrain_layout(map_section: Any, map_section_end: int, w: int, h: int) -> tuple[int, int, bool]:
    """(terrain_block_offset, terrain_struct_size, terrain_has_layer), all read
    off the parsed Map section, never a version table. terrain_data is Map's
    last retriever in every structure (v1.21 included), so the block ends at
    map_section_end. Stride is 0 if the parsed length is not an exact
    multiple of w*h: that means the parse is not what it claims."""
    terrain = map_section.retriever_map["terrain_data"]
    length = retriever_length(terrain)
    stride, remainder = divmod(length, w * h) if w * h > 0 else (0, 1)
    tiles = terrain.data or []
    has_layer = bool(tiles) and "layer" in tiles[0].retriever_map
    return map_section_end - length, (stride if remainder == 0 else 0), has_layer


def _verify_units_block(
    decompressed: bytes, offset: int, players_units: list, trailer_length: int, units_section_end: int
) -> bool:
    """True iff the u32 unit_count at the start of each PlayerUnitsStruct in
    `players_units` (as walked from `offset`) matches that struct's already-parsed
    unit_count, and the forward walk of the whole Units section (the array,
    then the `trailer_length` bytes of retrievers declared after it) lands
    exactly on the independently captured `units_section_end` -- the
    load-time trust check for units_block_offset, mirroring
    _verify_terrain_block() and _verify_messages_block() above."""
    if offset < 0 or offset > len(decompressed) or units_section_end > len(decompressed):
        return False
    o = offset
    for section in players_units:
        if o + 4 > len(decompressed):
            return False
        (raw_count,) = struct.unpack_from("<I", decompressed, o)
        if raw_count != section.retriever_map["unit_count"].data:
            return False
        o += section.byte_length
    return o + trailer_length == units_section_end


def _units_layout(units_section: Any, units_section_start: int) -> tuple[int, int, int]:
    """(units_block_offset, players_units_end, trailer_length): a forward walk
    of the Units retrievers in declaration order from the section's start.
    Not a backward walk from units_section_end, because players_units is not
    Units' last retriever everywhere (v1.21 puts number_of_players and
    player_data_3 after it). trailer_length is the byte total of the
    retrievers after players_units, 0 on every DE version."""
    pos = units_section_start
    block_offset = players_units_end = -1
    trailer = 0
    for name, retriever in units_section.retriever_map.items():
        length = retriever_length(retriever)
        if name == "players_units":
            block_offset = pos
            players_units_end = pos + length
        elif block_offset >= 0:
            trailer += length
        pos += length
    return block_offset, players_units_end, trailer


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


def _header_field_spans(header_bytes: bytes, retriever_map: dict) -> dict[str, tuple[int, int]]:
    """(start, end) within header_bytes of every FileHeader retriever, from
    one forward walk in true on-disk order -- the shared derivation behind
    both header_instructions_span and header_player_count_span. Walks only
    the _HEADER_WALK_ORDER names this file's structure actually has (v1.21
    has no creator_name or trigger_count). Empty, never raising, if the walk
    overruns or doesn't reconcile to exactly len(header_bytes) (e.g. a
    structure with a field this walk doesn't know about -- every
    header-editing feature degrades gracefully rather than trusting a wrong
    offset). Measured: the walk fails this way on one real corpus file (a
    scenario version 1.37 one) and on every v1.21 file, both because of
    individual_victories_used, so the reconcile check is load-bearing rather
    than a rubber stamp.

    A str32 field's span excludes its own 4-byte length prefix (the payload
    is what a caller splices); every other field's span is the whole field.

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
    spans: dict[str, tuple[int, int]] = {}
    for name in _HEADER_WALK_ORDER:
        if name not in retriever_map:
            continue
        if name in _HEADER_STR32_FIELDS:
            if pos + _STR32_PREFIX_SIZE > len(header_bytes):
                return {}
            (payload_len,) = _STR32_LENGTH_PREFIX_STRUCT.unpack_from(header_bytes, pos)
            length = _STR32_PREFIX_SIZE + payload_len
            spans[name] = (pos + _STR32_PREFIX_SIZE, pos + length)
        else:
            length = retriever_length(retriever_map[name])
            spans[name] = (pos, pos + length)
        pos += length
        if pos > len(header_bytes):
            return {}
    if pos != len(header_bytes):
        return {}
    return spans


def _header_instructions_span(header_bytes: bytes, retriever_map: dict) -> tuple[int, int]:
    """scenario_instructions' payload span, or (-1, -1) if the header walk
    doesn't reconcile -- see _header_field_spans()."""
    return _header_field_spans(header_bytes, retriever_map).get("scenario_instructions", (-1, -1))


def _header_player_count_span(header_bytes: bytes, retriever_map: dict) -> tuple[int, int]:
    """FileHeader.player_count's own span, or (-1, -1) if the header walk
    doesn't reconcile -- see _header_field_spans(). This is the second of
    the two buffers Number of Players writes (the other is
    DataHeader.player_data_1[i].active, in decompressed_body); the two are
    coupled (player_count == count of active flags among players 1..8,
    measured on every corpus file) and are written from one value."""
    return _header_field_spans(header_bytes, retriever_map).get("player_count", (-1, -1))


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
    if not structure_is_available(scenario_version):
        # Checked here rather than letting _load_structure() raise, so the
        # cause is named instead of surfacing the library's raw message.
        raise UnsupportedStructureVersion(scenario_version)
    repo_structure = _repo_structure_path(scenario_version)
    if repo_structure is None:
        scenario._load_structure()
        _initialise_version_dependencies(scenario.game_version, scenario.scenario_version)
    else:
        # Parsed fresh per load: the library mutates the structure dict it is
        # given. No vocabulary init, since the library ships none for this
        # version; parse_triggers() refuses these files instead.
        scenario.structure = json.loads(repo_structure.read_text(encoding="utf-8"))
    scenario._load_header_section(igen)
    # igen wraps the whole raw byte stream up front; progress is exactly the
    # header's length at this point, so this slice is the header's own verbatim
    # bytes -- no need to re-read them separately.
    header_bytes = igen.file_content[: igen.progress]

    original_compressed_body = igen.get_remaining_bytes()
    decompressed = _decompress_bytes(original_compressed_body)
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

    # After the depoison() above, so the next load's depoison() undoes it.
    library_compat.adapt_map_links(scenario.sections["Map"])
    map_manager = MapManager.construct(scenario.uuid)
    unit_manager = UnitManager.construct(scenario.uuid)

    # Skipping AoE2ObjectManager.setup() (it would also build a TriggerManager and
    # hit the same Triggers-parsing crash we're avoiding), but some lazy properties
    # -- e.g. TerrainTile.xy -- go through scenario.map_manager, which reads from
    # here. Populate just the two slots we actually constructed.
    scenario._object_manager = AoE2ObjectManager(scenario.uuid)
    scenario._object_manager.managers["Map"] = map_manager
    scenario._object_manager.managers["Unit"] = unit_manager

    w, h = map_manager.map_width, map_manager.map_height
    terrain_block_offset, terrain_struct_size, terrain_has_layer = _terrain_layout(
        scenario.sections["Map"], map_section_end, w, h
    )
    terrain_write_supported = _verify_terrain_block(
        decompressed, terrain_block_offset, map_manager.terrain, terrain_struct_size, terrain_has_layer
    )
    if not terrain_write_supported:
        terrain_block_offset = -1

    # Units starts where Map ends (they are adjacent in every structure).
    units_section = scenario.sections["Units"]
    players_units = units_section.retriever_map["players_units"].data
    number_of_unit_sections = len(players_units)
    units_block_offset, players_units_end, units_trailer = _units_layout(units_section, map_section_end)
    units_write_supported = _verify_units_block(
        decompressed, units_block_offset, players_units, units_trailer, units_section_end
    )
    if not units_write_supported:
        units_block_offset = -1
        players_units_end = -1

    player_colors, team_indices = resolve_player_colors(_read_player_colors(scenario))

    messages_retriever_map = scenario.sections["Messages"].retriever_map
    messages_write_supported = _verify_messages_block(
        decompressed, messages_section_start, messages_section_end, messages_retriever_map
    )
    header_retriever_map = scenario.sections["FileHeader"].retriever_map
    header_instructions_span = _header_instructions_span(header_bytes, header_retriever_map)
    header_player_count_span = _header_player_count_span(header_bytes, header_retriever_map)

    return LoadedScenario(
        path=path,
        scenario_version=scenario_version,
        structure_source="library" if repo_structure is None else "repo",
        map_manager=map_manager,
        unit_manager=unit_manager,
        trigger_tail=trigger_tail,
        header_bytes=header_bytes,
        decompressed_body=decompressed,
        original_compressed_body=original_compressed_body,
        terrain_block_offset=terrain_block_offset,
        terrain_write_supported=terrain_write_supported,
        terrain_struct_size=terrain_struct_size,
        terrain_has_layer=terrain_has_layer,
        units_block_offset=units_block_offset,
        units_section_end=units_section_end,
        players_units_end=players_units_end,
        number_of_unit_sections=number_of_unit_sections,
        units_write_supported=units_write_supported,
        options_section_end=options_section_end,
        has_trigger_counters=(
            "number_of_triggers" in scenario.sections["Options"].retriever_map
            and "trigger_count" in header_retriever_map
        ),
        global_victory_section_end=global_victory_section_end,
        diplomacy_section_end=diplomacy_section_end,
        player_data_two_section_end=player_data_two_section_end,
        messages_section_start=messages_section_start,
        messages_section_end=messages_section_end,
        header_instructions_span=header_instructions_span,
        header_player_count_span=header_player_count_span,
        messages_write_supported=messages_write_supported,
        trigger_version=_read_trigger_version(trigger_tail),
        triggers_section_end=-1,
        # Known up front for a repo structure: parse_triggers() never tries one.
        trigger_read_supported=None if repo_structure is None else False,
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


def refresh_player_colors(loaded: LoadedScenario, pending_colors: dict[int, int]) -> bool:
    """Re-derive player_colors/team_indices, in place, from the file's own
    stored ids overlaid with pending colour edits ({player_id: color_id}).
    True iff either tuple actually changed.

    Both tuples are computed once at load and never otherwise recomputed, but
    a Players-mode colour edit is an in-place byte patch to decompressed_body
    for the save path -- it never mutates the retriever _read_player_colors()
    reads, so nothing re-derives them and the map keeps drawing the pre-edit
    colour. This is what descape/viewer.py's _after_player_color_change()
    calls to close that gap.

    Wholesale, not an incremental patch of the cached tuple: the base ids are
    unaffected by the byte patch, so overlaying the pending edits on them
    makes undo/redo come out right for free (an undone edit simply drops out
    of pending_colors). In place rather than derived on demand because the
    chunk caches hold their own scenario handle and both player panels read
    off the same object, so one mutation reaches every reader.

    Requires a real LoadedScenario -- it reaches _scenario and its parsed
    PlayerDataTwo. The duck-typed fakes in testkit/fakes.py and
    tests/test_farm_terrain.py set both tuples as plain attributes with no
    _scenario, so no path a fake travels may reach here.
    """
    color_ids = _read_player_colors(loaded._scenario)
    for player_id, color_id in pending_colors.items():
        if 1 <= player_id <= 8:
            # color_ids is P1-first; the combo's raw value is ColorId.value,
            # 0-based over 0..7, which indexes PLAYER_COLOR_BY_ID directly.
            color_ids[player_id - 1] = int(color_id)
    player_colors, team_indices = resolve_player_colors(color_ids)
    changed = player_colors != loaded.player_colors or team_indices != loaded.team_indices
    loaded.player_colors = player_colors
    loaded.team_indices = team_indices
    return changed


def xs_attachment(loaded: LoadedScenario) -> tuple[str | None, int | None]:
    """(Map script_name, embedded Files script_file_content length in chars).

    Read-only: script_name is a variable-length string ahead of terrain_data
    in Map, so no byte-patch write path may ever change it. script_name is
    None before scenario 1.40 (no retriever). The embedded length is None
    until parse_triggers() has reached Files; this never forces that parse.
    Characters, not bytes: the value is already decoded.
    """
    sections = loaded._scenario.sections
    map_retriever = sections["Map"].retriever_map.get("script_name")
    script_name = None if map_retriever is None else (map_retriever.data or "")
    embedded = None
    files = sections.get("Files") if loaded.trigger_read_supported else None
    if files is not None:
        content = files.retriever_map.get("script_file_content")
        if content is not None:
            embedded = len(content.data or "")
    return script_name, embedded


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

    Also returns None, without attempting a parse, for a file loaded from a
    repo structure (structure_source "repo", i.e. v1.21): the library ships
    no condition/effect definitions for it and _load_map_and_units() never
    initialised any, so a parse would read another version's vocabulary.

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

    if loaded.structure_source == "repo":
        loaded.trigger_read_supported = False
        return None
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
