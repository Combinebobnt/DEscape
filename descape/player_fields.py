"""Qt-free specs for the read-only Players mode's fields -- default-tier
testable without a QApplication, and it keeps the panel itself dumb. Mirrors
descape/option_fields.py's shape, but is a separate dataclass and a separate
`_SPECS` tuple on purpose: sharing OptionFieldSpec/OptionsEditModel would be
actively harmful, since its write-support gate is all-or-nothing across every
spec that shares its tuple.

Two things this module has that option_fields.py doesn't, both from the same
root cause -- per-player data lives in *arrays*, not scalars:

- `PlayerArrayLayout` -- where GAIA sits (or doesn't) in a given array. Not
  guessable from the retriever name: AoE2ScenarioParser's own PlayerManager
  builds three different layouts from the exact same kind of array (see
  objects.managers.player_manager.PlayerManager.__init__'s gaia_first_params
  / no_gaia_params / gaia_last_params, and `_player_list()` for how a
  `gaia_first` of True/False/None becomes an insertion index of 0/8/never).
  That confirmed mapping is transcribed into the per-spec `layout` field
  below; nothing else in this codebase records it.
- `fallback` -- Pop Limit is the one field stored twice at different types
  and struct shapes (Map.per_player_population_cap, a u32, only since
  1.44; Units.player_data_4[i].population_limit, an f32, on every version).
  current_value() prefers the primary and only reads the fallback when the
  primary retriever is absent.

current_value() returns `int | str`, not `int`-only like option_fields.py's:
tribe_names is a string on every file, and civilization/architecture_set are
a string from scenario version 1.56 on (see resolve_civilization_name()).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

from AoE2ScenarioParser.datasets.object_support import Civilization, CivilizationOld, StartingAge
from AoE2ScenarioParser.datasets.players import ColorId
from AoE2ScenarioParser.helper.bytes_conversions import fixed_chars_to_bytes

from descape.option_fields import CHECKBOX, COMBO, SPINBOX
from descape.scenario_io import LoadedScenario, retriever_length

TEXT = "text"  # a plain read-only string row -- option_fields.py has no
# equivalent kind because none of its specs are ever string-valued.


class PlayerArrayLayout(Enum):
    """player_id (0 = GAIA, 1..8 = P1..P8) -> index within one of these
    arrays. Confirmed against AoE2ScenarioParser's own PlayerManager, not
    guessed: `_player_list(gaia_first)` inserts PlayerId.GAIA at index 0
    (True), index 8 (False), or not at all (None) into the P1..P8 list --
    see objects/managers/player_manager.py.
    """

    # gaia_last_params, player_manager.py:104-116 (starting_age, lock_civ,
    # lock_personality, food/wood/gold/stone/color -- i.e. PlayerDataTwo's
    # `resources` -- and active/human/civilization/architecture_set, i.e.
    # DataHeader's `player_data_1`). ai_type and ai_names are not in the
    # library's object model at all (see the maintainer plan's "not
    # confirmed" note on ai_type) but sit in the same PlayerDataTwo section
    # with the same repeat-16 shape as resources/ai_type's siblings, so this
    # layout is assumed for them by direct analogy, not independently
    # measured.
    P1_TO_P8_THEN_GAIA = "p1_to_p8_then_gaia"

    # gaia_first_params, player_manager.py:87-89 (initial_player_view_x/y,
    # i.e. Map's `initial_player_views`). The only GAIA-first array.
    GAIA_FIRST = "gaia_first"

    # no_gaia_params, player_manager.py:91-95 (population_cap, tribe_name,
    # string_table_name_id, base_priority -- and, confirmed separately at
    # player_manager.py:92, player_data_4's `population_limit`, this
    # module's Pop Limit fallback).
    NO_GAIA = "no_gaia"

    def index_for(self, player_id: int) -> int:
        """Array index for `player_id`. Raises ValueError for a player_id
        out of 0..8, or for GAIA (0) against a layout with no GAIA slot."""
        if not 0 <= player_id <= 8:
            raise ValueError(f"player_id must be 0..8, got {player_id}")
        if self is PlayerArrayLayout.GAIA_FIRST:
            return player_id
        if player_id == 0:
            if self is PlayerArrayLayout.P1_TO_P8_THEN_GAIA:
                return 8
            raise ValueError(f"GAIA has no slot in a {self.value} array")
        return player_id - 1


@dataclass(frozen=True)
class PlayerFieldSpec:
    field_id: str  # stable id, independent of label text
    label: str
    group: str  # panel group box title, e.g. "Identity"
    section: str  # scenario section name, e.g. "PlayerDataTwo"
    retriever: str  # retriever name within that section, e.g. "resources"
    kind: str  # CHECKBOX / COMBO / SPINBOX / TEXT
    layout: PlayerArrayLayout
    struct_field: str | None = None  # for struct-array retrievers, e.g.
    # "resources" + "player_color" -- None means the retriever's own repeated
    # value is the field itself (e.g. tribe_names).
    choices: tuple[tuple[int, str], ...] = ()  # (value, label) pairs, COMBO only
    minimum: int | None = None  # SPINBOX bounds
    maximum: int | None = None
    scale: float = 1.0
    tooltip: str = ""
    fallback: tuple[str, str, str | None] | None = None  # alternate
    # (section, retriever, struct_field), read only when the primary
    # retriever is absent -- see NO_GAIA's docstring for Pop Limit.


_STARTING_AGE_CHOICES = tuple((m.value, m.name.replace("_", " ").title()) for m in StartingAge)
_COLOR_CHOICES = tuple((m.value, m.name.title()) for m in ColorId if not m.name.startswith("INVALID"))
# Domain {0,1,2} on every corpus file, but nothing in AoE2ScenarioParser
# documents what it means -- see the maintainer plan's "not confirmed" note.
# Labelled honestly rather than guessed.
_PLAYER_TYPE_CHOICES = ((0, "Unknown (0)"), (1, "Unknown (1)"), (2, "Unknown (2)"))


_SPECS: tuple[PlayerFieldSpec, ...] = (
    # -- Identity ---------------------------------------------------------
    PlayerFieldSpec(
        "tribe_name", "Tribe name", "Identity",
        "DataHeader", "tribe_names", TEXT, PlayerArrayLayout.NO_GAIA,
    ),
    PlayerFieldSpec(
        "player_name_string_id", "Name string ID", "Identity",
        "DataHeader", "string_table_player_names", SPINBOX, PlayerArrayLayout.NO_GAIA,
        minimum=-2, maximum=2_147_483_647,
        tooltip="-2 means unset (no custom name string).",
    ),
    PlayerFieldSpec(
        "civilization", "Civilization", "Identity",
        "DataHeader", "player_data_1", TEXT, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="civilization",
        tooltip="An integer index below scenario version 1.56, a string "
                "(e.g. 'HUN-CIV') from 1.56 on -- see resolve_civilization_name().",
    ),
    PlayerFieldSpec(
        "architecture", "Architecture", "Identity",
        "DataHeader", "player_data_1", TEXT, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="architecture_set",
        tooltip="Same civilization/architecture dataset and the same "
                "int-then-string version switch as Civilization above -- "
                "the game expresses an architecture style as a civilization id.",
    ),
    PlayerFieldSpec(
        "lock_civilization", "Lock civilization", "Identity",
        "DataHeader", "per_player_lock_civilization", CHECKBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
    ),
    PlayerFieldSpec(
        "lock_personality", "Lock personality", "Identity",
        "DataHeader", "per_player_lock_personality", CHECKBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        tooltip="Added at scenario version 1.53.",
    ),

    # -- Start --------------------------------------------------------------
    PlayerFieldSpec(
        "starting_age", "Starting age", "Start",
        "Options", "per_player_starting_age", COMBO, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        choices=_STARTING_AGE_CHOICES,
    ),
    PlayerFieldSpec(
        "food", "Starting food", "Start",
        "PlayerDataTwo", "resources", SPINBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="food", minimum=-2_147_483_648, maximum=2_147_483_647,
    ),
    PlayerFieldSpec(
        "wood", "Starting wood", "Start",
        "PlayerDataTwo", "resources", SPINBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="wood", minimum=-2_147_483_648, maximum=2_147_483_647,
    ),
    PlayerFieldSpec(
        "stone", "Starting stone", "Start",
        "PlayerDataTwo", "resources", SPINBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="stone", minimum=-2_147_483_648, maximum=2_147_483_647,
    ),
    PlayerFieldSpec(
        "gold", "Starting gold", "Start",
        "PlayerDataTwo", "resources", SPINBOX, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="gold", minimum=-2_147_483_648, maximum=2_147_483_647,
    ),
    PlayerFieldSpec(
        "pop_limit", "Population limit", "Start",
        "Map", "per_player_population_cap", SPINBOX, PlayerArrayLayout.NO_GAIA,
        minimum=0, maximum=2_147_483_647,
        fallback=("Units", "player_data_4", "population_limit"),
        tooltip="Added at scenario version 1.44; older files read the "
                "duplicate copy in Units.player_data_4 instead. Confirmed "
                "2026-08-30 by editing a real file in the AoE2:DE Scenario "
                "Editor: this is the copy the game actually displays, "
                "per-player -- not Units.player_data_4, despite "
                "AoE2ScenarioParser's own PlayerManager reading and writing "
                "that copy on commit.",
    ),
    PlayerFieldSpec(
        "color", "Colour", "Start",
        "PlayerDataTwo", "resources", COMBO, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        struct_field="player_color", choices=_COLOR_CHOICES,
    ),

    # -- AI -------------------------------------------------------------------
    PlayerFieldSpec(
        "player_type", "Player type", "AI",
        "PlayerDataTwo", "ai_type", COMBO, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        choices=_PLAYER_TYPE_CHOICES,
        tooltip="Unconfirmed -- a hypothesis for the in-game 'Player Type' "
                "dropdown, not documented anywhere in AoE2ScenarioParser.",
    ),
    PlayerFieldSpec(
        "personality", "Personality", "AI",
        "PlayerDataTwo", "ai_names", TEXT, PlayerArrayLayout.P1_TO_P8_THEN_GAIA,
        tooltip="ai_files (the embedded .ai script source, up to several "
                "MB) is deliberately never read -- this field is name-only.",
    ),
    PlayerFieldSpec(
        "base_priority", "Base priority", "AI",
        "Options", "per_player_base_priority", SPINBOX, PlayerArrayLayout.NO_GAIA,
        minimum=0, maximum=255,
    ),

    # -- Point of View ----------------------------------------------------
    # The only GAIA_FIRST spec -- included specifically to exercise that
    # layout on real data (see the maintainer plan).
    PlayerFieldSpec(
        "initial_view_x", "Initial camera X", "Point of View",
        "Map", "initial_player_views", SPINBOX, PlayerArrayLayout.GAIA_FIRST,
        struct_field="location_x", minimum=-2_147_483_648, maximum=2_147_483_647,
        tooltip="Added at scenario version 1.40.",
    ),
    PlayerFieldSpec(
        "initial_view_y", "Initial camera Y", "Point of View",
        "Map", "initial_player_views", SPINBOX, PlayerArrayLayout.GAIA_FIRST,
        struct_field="location_y", minimum=-2_147_483_648, maximum=2_147_483_647,
        tooltip="Added at scenario version 1.40.",
    ),
)


def _retriever_present(
    loaded: LoadedScenario, section_name: str, retriever_name: str, struct_field: str | None = None
) -> bool:
    section = loaded._scenario.sections.get(section_name)
    if section is None:
        return False
    retriever = section.retriever_map.get(retriever_name)
    if retriever is None:
        return False
    if retriever_length(retriever) == 0:
        return False
    if struct_field is not None:
        # A struct-array retriever can itself be present and non-empty while
        # a *specific field within its struct* is a later addition -- e.g.
        # DataHeader.player_data_1 exists at every version this project
        # supports, but its entries only gain an `architecture_set` key from
        # a later structure.json than the one that adds `civilization`
        # (confirmed on a 1.37 corpus file: player_data_1 entries there hold
        # only active/civilization/cty_mode/human, no architecture_set at
        # all). Checking the retriever alone would list "architecture" as
        # present and then raise a KeyError in current_value() for every
        # player on such a file.
        entries = retriever.data
        if not entries or struct_field not in entries[0].retriever_map:
            return False
    return True


def specs_for(loaded: LoadedScenario) -> tuple[PlayerFieldSpec, ...]:
    """Every PlayerFieldSpec whose primary retriever -- or, failing that,
    its fallback -- is actually present and non-empty on `loaded`. Presence
    is read from the already-loaded data, never inferred from
    scenario_version, matching option_fields.specs_for()'s reasoning: a
    structure-version gate and an empty repeat-driven array both look like
    "0 bytes" and both mean "not on this file".
    """
    out = []
    for spec in _SPECS:
        if _retriever_present(loaded, spec.section, spec.retriever, spec.struct_field):
            out.append(spec)
        elif spec.fallback is not None and _retriever_present(loaded, *spec.fallback):
            out.append(spec)
    return tuple(out)


def current_value(loaded: LoadedScenario, spec: PlayerFieldSpec, player_id: int) -> int | str:
    """`spec`'s current raw value for `player_id` (0 = GAIA, 1..8 = P1..P8).
    Only valid for a spec that specs_for(loaded) actually returned, and only
    for a player_id spec.layout (or, on fallback, PlayerArrayLayout.NO_GAIA
    -- every fallback target is Units.player_data_4, itself NO_GAIA-shaped)
    has a slot for.

    Returns int or str -- never a list, never a float. Population limit's
    fallback (Units.player_data_4[i].population_limit) is stored as an f32;
    every value across this project's corpus is a whole number, so a whole
    float is coerced to int and a fractional one raises, the same
    fail-fast-on-a-surprise reasoning as option_fields.current_value()'s
    scalar-only guard.
    """
    if _retriever_present(loaded, spec.section, spec.retriever, spec.struct_field):
        section_name, retriever_name, struct_field, layout = (
            spec.section, spec.retriever, spec.struct_field, spec.layout,
        )
    elif spec.fallback is not None and _retriever_present(loaded, *spec.fallback):
        section_name, retriever_name, struct_field = spec.fallback
        layout = PlayerArrayLayout.NO_GAIA
    else:
        raise ValueError(f"{spec.field_id!r} is not present on this file and has no usable fallback")

    index = layout.index_for(player_id)
    data = loaded._scenario.sections[section_name].retriever_map[retriever_name].data
    entry = data[index]
    value = entry.retriever_map[struct_field].data if struct_field else entry

    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(
                f"{spec.field_id!r} resolved to a non-integral float {value!r} -- "
                f"current_value() only supports int/str"
            )
        value = int(value)
    if not isinstance(value, (int, str)):
        raise TypeError(
            f"{spec.field_id!r} resolved to a {type(value).__name__}, not int/str -- "
            f"current_value() only supports scalar retrievers"
        )
    return value


def resolve_civilization_name(value: int | str) -> str:
    """Human-readable name for a raw civilization/architecture value -- an
    int (CivilizationOld) below scenario version 1.56, a str (Civilization,
    e.g. 'HUN-CIV') from 1.56 on. The game expresses an architecture style
    as a civilization id, so this covers both the "civilization" and
    "architecture" specs above. Falls back to the raw value for anything
    the dataset doesn't recognize (a modded id, RANDOM/FULL_RANDOM handled
    by name already) rather than raising -- an unrecognized row should
    display as itself, not break the panel.
    """
    try:
        if isinstance(value, str):
            return Civilization(value).name.title()
        return CivilizationOld(value).name.title()
    except ValueError:
        return str(value)


# == write path (the maintainer plan's step 3a) ==============================
#
# Every offset below is derived, never hardcoded, from a byte offset
# scenario_io.py already trusts -- the same backward-walk technique
# descape/diplomacy_fields.py and descape/options_model.py use, plus one
# forward walk (DataHeader, the first section in decompressed_body, so no
# anchor is needed) and one special-cased route (Units.player_data_3.color,
# below). Route per section, all measured byte-for-byte against every
# examples/ file:
#
# - DataHeader: forward from body offset 0. A backward walk would inherit
#   `filename`'s (str16) re-serialization drift; forward never crosses it,
#   since every field this module writes sits before it in DataHeader's own
#   retriever order.
# - PlayerDataTwo: backward from player_data_two_section_end. `resources` is
#   this section's own *last* retriever, so this crosses nothing else.
# - Options: backward from options_section_end (existing anchor).
# - Map: backward from terrain_block_offset, excluding `terrain_data`
#   (existing anchor, same exclusion options_model.py applies).
# - Units: backward from units_block_offset (existing anchor, already
#   verified by scenario_io._verify_units_block()).
#
# Reimplemented locally rather than importing options_model.field_offsets():
# options_model imports this module (for the additive player-edit gate), so
# the reverse import would be a cycle -- the same reason
# diplomacy_fields.py's _section_layout() is its own local walk.


@dataclass(frozen=True)
class PlayerWriteTarget:
    offset: int  # byte offset within decompressed_body
    length: int  # byte length
    codec: str  # "s32" | "u32" | "u8" | "f32" | "c256"


@dataclass(frozen=True)
class FieldOffset:
    offset: int
    length: int


@dataclass(frozen=True)
class _Resolved:
    target: PlayerWriteTarget
    parsed_value: int | str  # the value already parsed at this exact
    # location -- verify_player_block()'s ground truth. Never re-derived
    # from decompressed_body itself, which would make the check tautological.


_CODEC_STRUCT: dict[str, struct.Struct] = {
    "u8": struct.Struct("<B"),
    "s32": struct.Struct("<i"),
    "u32": struct.Struct("<I"),
    "f32": struct.Struct("<f"),
}

# Never writable, regardless of specs_for()'s presence check: player_type's
# semantics are unconfirmed (the maintainer plan's decision 2) and Tier 2
# fields are variable-length strings no byte patch could reach.
_NEVER_WRITABLE = frozenset({"player_type", "civilization", "architecture", "personality"})


def player_field_id(field_id: str, player_id: int) -> str:
    """The synthetic OptionsEditModel id for one player's one field --
    write_targets()'s own key format, factored out so options_model.py and
    viewer.py construct and parse it the same way rather than each
    hardcoding the "player:<field>:<player>" string."""
    return f"player:{field_id}:{player_id}"


def parse_player_field_id(key: str) -> tuple[str, int]:
    """Inverse of player_field_id(): (field_id, player_id)."""
    _, field_id, player_id = key.split(":")
    return field_id, int(player_id)

# field_id -> (section, retriever, struct_field) of a second stored copy that
# must be kept in sync with the primary (the maintainer plan's decision 4:
# "editing a mirrored field writes both copies, matching what
# AoE2ScenarioParser does on commit"). Structural: present on every DE
# structure version this project supports, so a resolution failure here
# means something is genuinely wrong and the whole field is refused (see
# _resolve_all()). Every route here is NO_GAIA-shaped for players 1..8, so
# _resolve_all() always indexes it at player_id - 1 regardless of the
# *primary* spec's own layout. `color` and `pop_limit`'s own mirrors are
# each resolved separately below (_color_mirror_target(),
# _pop_limit_mirror_target()) and treated as optional, not structural --
# see _resolve_all()'s docstring for why.
_MIRRORS: dict[str, tuple[str, str, str | None]] = {
    "food": ("Units", "player_data_4", "food_duplicate"),
    "wood": ("Units", "player_data_4", "wood_duplicate"),
    "gold": ("Units", "player_data_4", "gold_duplicate"),
    "stone": ("Units", "player_data_4", "stone_duplicate"),
}


def _codec_for(retriever) -> str:
    """The codec identifying how to pack/unpack this retriever's value, taken
    from the retriever the file was parsed with -- never a hand-written
    per-spec constant, the same rule options_model.py's _PACK_STRUCT
    docstring gives."""
    return retriever.datatype.var


def _retriever_map(loaded: LoadedScenario, section_name: str) -> dict | None:
    section = loaded._scenario.sections.get(section_name)
    return None if section is None else section.retriever_map


def _struct_field_offset(entry, struct_field: str) -> tuple[int, int, int] | None:
    """(byte offset, byte length, total entry length as this walk computes
    it) for `struct_field` within one already-parsed struct entry, walking
    its own retriever_map forward in declared order -- the field's real
    position, never assumed. Always walks the *whole* entry rather than
    stopping at struct_field, so the third element (a caller's own
    span-check against entry.byte_length) is available even when
    struct_field sits near the start. None if the field is absent from this
    entry."""
    pos = 0
    field_pos = None
    field_len = None
    for name, retriever in entry.retriever_map.items():
        length = retriever_length(retriever)
        if name == struct_field:
            field_pos = pos
            field_len = length
        pos += length
    if field_pos is None:
        return None
    return field_pos, field_len, pos


def _forward_offsets(retriever_map, wanted: frozenset[str]) -> tuple[dict[str, FieldOffset], int]:
    """Byte offset+length of every name in `wanted`, walking forward from
    position 0 through retriever_map in its own declared order -- the
    DataHeader route above. Also returns the position after the *last*
    retriever walked -- the section's own consumed length as this walk
    computes it, for a caller's span-check against a trusted parse-time
    count."""
    offsets: dict[str, FieldOffset] = {}
    pos = 0
    for name, retriever in retriever_map.items():
        length = retriever_length(retriever)
        if name in wanted:
            offsets[name] = FieldOffset(pos, length)
        pos += length
    return offsets, pos


def _backward_offsets(
    retriever_map, anchor: int, wanted: frozenset[str], exclude: frozenset[str] = frozenset()
) -> dict[str, FieldOffset] | None:
    """Byte offset+length of whichever of `wanted` this file's version
    actually has, walking backward from `anchor` in reverse declared order
    and skipping `exclude` entirely -- the Options/Map/Units/PlayerDataTwo
    routes above. Partial is expected and fine (per_player_base_priority is
    a later addition than per_player_starting_age, matching
    options_model.field_offsets()'s own per-field, not all-or-nothing,
    result); a specific missing name is _array_target()'s problem, not this
    walk's. None only if `anchor` itself is untrusted."""
    if anchor < 0:
        return None
    names = [name for name in retriever_map if name not in exclude]
    offsets: dict[str, FieldOffset] = {}
    pos = anchor
    remaining = set(wanted)
    for name in reversed(names):
        length = retriever_length(retriever_map[name])
        pos -= length
        if name in remaining:
            offsets[name] = FieldOffset(pos, length)
            remaining.discard(name)
            if not remaining:
                break
    return offsets


_DATA_HEADER_WANTED = frozenset({
    "tribe_names", "string_table_player_names",
    "per_player_lock_civilization", "per_player_lock_personality",
})


def _data_header_bases(loaded: LoadedScenario) -> dict[str, FieldOffset] | None:
    """DataHeader's own Tier-1 array offsets, forward-walked from body
    offset 0. Span-checked against section.byte_length -- the section's own
    trusted parse-time total, recorded while parsing the real file
    (AoE2FileSection.set_data_from_generator), unlike retriever_length()'s
    get_data_as_bytes()-based figure, which re-serializes and so could
    drift on a str16 field (player_data_1's civilization/architecture_set)
    this walk has to cross to reach per_player_lock_civilization/
    _personality."""
    retrievers = _retriever_map(loaded, "DataHeader")
    if retrievers is None:
        return None
    offsets, total = _forward_offsets(retrievers, _DATA_HEADER_WANTED)
    if total != loaded._scenario.sections["DataHeader"].byte_length:
        return None
    # Partial is fine and expected -- per_player_lock_personality is a 1.53+
    # addition and tribe_names/string_table_player_names/
    # per_player_lock_civilization are older, so an early file legitimately
    # has some of _DATA_HEADER_WANTED and not others. _array_target()
    # already handles one missing retriever_name per-field; failing the
    # whole walk here would wipe out fields that resolved just fine.
    return offsets


def _player_data_two_bases(loaded: LoadedScenario) -> dict[str, FieldOffset] | None:
    retrievers = _retriever_map(loaded, "PlayerDataTwo")
    if retrievers is None:
        return None
    return _backward_offsets(retrievers, loaded.player_data_two_section_end, frozenset({"resources"}))


_OPTIONS_WANTED = frozenset({"per_player_starting_age", "per_player_base_priority"})


def _options_bases(loaded: LoadedScenario) -> dict[str, FieldOffset] | None:
    retrievers = _retriever_map(loaded, "Options")
    if retrievers is None:
        return None
    return _backward_offsets(retrievers, loaded.options_section_end, _OPTIONS_WANTED)


_MAP_WANTED = frozenset({"per_player_population_cap", "initial_player_views"})
_MAP_EXCLUDE = frozenset({"terrain_data"})


def _map_bases(loaded: LoadedScenario) -> dict[str, FieldOffset] | None:
    retrievers = _retriever_map(loaded, "Map")
    if retrievers is None:
        return None
    return _backward_offsets(retrievers, loaded.terrain_block_offset, _MAP_WANTED, exclude=_MAP_EXCLUDE)


# units_block_offset is the *start* of players_units, the section's own
# last retriever (see scenario_io.LoadedScenario's docstring) -- not "one
# past the section's end" the way player_data_two_section_end/
# options_section_end are. players_units must be excluded from the reversed
# walk entirely, the same reason Map excludes terrain_data above: the
# anchor already positions us right after player_data_3, one retriever
# before players_units itself.
_UNITS_EXCLUDE = frozenset({"players_units"})


def _units_bases(loaded: LoadedScenario) -> dict[str, FieldOffset] | None:
    retrievers = _retriever_map(loaded, "Units")
    if retrievers is None:
        return None
    return _backward_offsets(
        retrievers, loaded.units_block_offset, frozenset({"player_data_4"}), exclude=_UNITS_EXCLUDE
    )


_SECTION_BASES = {
    "DataHeader": _data_header_bases,
    "PlayerDataTwo": _player_data_two_bases,
    "Options": _options_bases,
    "Map": _map_bases,
    "Units": _units_bases,
}


def _array_target(
    loaded: LoadedScenario, section: str, retriever_name: str, struct_field: str | None, index: int
) -> _Resolved | None:
    """One player's byte-patchable location within (section, retriever_name)
    -- optionally inside a struct_field -- for every route above except
    player_data_3.color (see _color_mirror_target). Element count and
    stride come from the retriever's own parsed length and entry count,
    never assumed 16 or 8 (per_player_base_priority is 8-wide)."""
    bases = _SECTION_BASES[section](loaded)
    if bases is None or retriever_name not in bases:
        return None
    base = bases[retriever_name]
    retriever = loaded._scenario.sections[section].retriever_map[retriever_name]
    count = len(retriever.data)
    if count == 0 or base.length % count != 0 or not 0 <= index < count:
        return None
    stride = base.length // count
    elem_offset = base.offset + stride * index

    if struct_field is None:
        value = retriever.data[index]
        return _Resolved(PlayerWriteTarget(elem_offset, stride, _codec_for(retriever)), value)

    entry = retriever.data[index]
    located = _struct_field_offset(entry, struct_field)
    if located is None:
        return None
    field_pos, field_len, _total = located
    field_retriever = entry.retriever_map[struct_field]
    return _Resolved(
        PlayerWriteTarget(elem_offset + field_pos, field_len, _codec_for(field_retriever)),
        field_retriever.data,
    )


def _color_mirror_target(loaded: LoadedScenario, player_id: int) -> _Resolved | None:
    """Units.player_data_3[player_id-1].color -- the mirror decision 4 wants
    kept in sync with the primary (PlayerDataTwo.resources.player_color),
    per AoE2ScenarioParser's own player_data_three.py: "Duplicates, only
    here so they're replaced in the scenario file. In case it impacts
    gameplay." Confirmed in sync with the primary on all 20 examples/ files.

    Reached differently from every other target in this module:
    player_data_3 is variable-length (constant_name is a str16, and
    diplomacy_for_interaction/diplomacy_for_ai_system are themselves
    variably-repeated), so there is no trusted end-of-section anchor to
    walk backward from the way resources/player_data_4 do. Instead:
    player_data_3 sits immediately before players_units with nothing
    between them in every DE structure version, so units_block_offset
    (players_units' own trusted start) minus player_data_3's total
    retriever_length() -- itself the sum of each entry's own trusted
    byte_length, never a re-serialization -- is player_data_3's own start.
    Each entry's own start is then the sum of every *earlier* entry's
    trusted byte_length (entries can differ in length from each other).
    Only the chosen entry's own fields are walked forward (summing
    retriever_length(), which does re-serialize), and only trusted if that
    sum equals the entry's own byte_length exactly -- the span check that
    would catch a variable-length field's re-serialization drifting from
    what was actually parsed. Measured against all 20 examples/ files:
    holds byte-for-byte on every entry of every file.
    """
    if not loaded.units_write_supported:
        return None
    retrievers = _retriever_map(loaded, "Units")
    if retrievers is None:
        return None
    retriever = retrievers.get("player_data_3")
    if retriever is None or not retriever.data:
        return None
    entries = retriever.data
    index = player_id - 1
    if not 0 <= index < len(entries):
        return None

    array_start = loaded.units_block_offset - retriever_length(retriever)
    if array_start < 0:
        return None
    entry_start = array_start + sum(e.byte_length for e in entries[:index])
    entry = entries[index]

    located = _struct_field_offset(entry, "color")
    if located is None:
        return None
    field_pos, field_len, total = located
    if total != entry.byte_length:
        return None
    field_retriever = entry.retriever_map["color"]
    return _Resolved(
        PlayerWriteTarget(entry_start + field_pos, field_len, _codec_for(field_retriever)),
        field_retriever.data,
    )


def _pop_limit_mirror_target(loaded: LoadedScenario, player_id: int) -> _Resolved | None:
    """Map.per_player_population_cap[player_id-1] -- pop_limit's mirror,
    kept separate from _MIRRORS (unlike food/wood/gold/stone's) because it
    is genuinely absent pre-1.44, by design, not a resolution failure: the
    write path's own structural primary is Units.player_data_4 (present on
    every version, see _resolve_all()'s pop_limit special case below), so a
    pre-1.44 file simply has nothing to mirror into and pop_limit must stay
    writable through its primary alone.

    Deliberately the *opposite* of what current_value() reads first:
    confirmed in-game 2026-08-30 that the real Scenario Editor displays
    (and presumably the game reads) this Map copy, not player_data_4's --
    despite AoE2ScenarioParser's own PlayerManager reading/writing
    player_data_4. current_value()'s primary/fallback order was corrected
    to match; the write path keeps player_data_4 structural regardless,
    since both copies are written whenever both resolve, so which one is
    "primary" for writing doesn't change what ends up on disk -- only
    which one survives alone on a pre-1.44 file, where player_data_4 must,
    since Map doesn't exist yet to read at all."""
    return _array_target(loaded, "Map", "per_player_population_cap", None, player_id - 1)


def _resolve_all(loaded: LoadedScenario) -> dict[str, tuple[_Resolved, ...]]:
    """{f"player:{field_id}:{player_id}": (primary, *mirrors)} for every
    writable Tier-1 spec and every player 1..8 -- never GAIA (player_id 0),
    which no Tier-1 spec's write path covers.

    A spec whose primary target cannot be resolved is omitted entirely. A
    spec with a *structural* mirror (food/wood/gold/stone, from _MIRRORS)
    that cannot be resolved is also omitted entirely -- a half-written
    mirrored pair, silently leaving the two copies out of sync, is worse
    than an unwritable field. `color` and `pop_limit`'s own mirrors are
    different: each is one specific corpus fact (player_data_3 has no fixed
    shape at all; the Map array is version-gated by design, not failure),
    not a structural guarantee the way the others are, so a file where
    either fails to resolve still gets a writable row through its primary
    alone.

    pop_limit is also the one spec where the write path's own primary
    isn't `spec.section`/`spec.retriever`/`spec.struct_field`: that route
    is Map.per_player_population_cap, chosen because current_value() reads
    it first (confirmed 2026-08-30 as what the real game displays), but
    Map is version-gated (1.44+) and would make pop_limit unwritable on
    every earlier file if used as the write path's structural target too.
    Units.player_data_4 (present on every version) stays the write path's
    primary regardless of which one reads first -- see
    _pop_limit_mirror_target()'s own docstring.
    """
    out: dict[str, tuple[_Resolved, ...]] = {}
    for spec in specs_for(loaded):
        if spec.field_id in _NEVER_WRITABLE:
            continue
        for player_id in range(1, 9):
            index = spec.layout.index_for(player_id)
            if spec.field_id == "pop_limit":
                primary = _array_target(loaded, "Units", "player_data_4", "population_limit", index)
            else:
                primary = _array_target(loaded, spec.section, spec.retriever, spec.struct_field, index)
            if primary is None:
                continue
            resolved = [primary]

            mirror_route = _MIRRORS.get(spec.field_id)
            if mirror_route is not None:
                mirror = _array_target(loaded, *mirror_route, player_id - 1)
                if mirror is None:
                    continue
                resolved.append(mirror)

            if spec.field_id == "color":
                optional = _color_mirror_target(loaded, player_id)
            elif spec.field_id == "pop_limit":
                optional = _pop_limit_mirror_target(loaded, player_id)
            else:
                optional = None
            if optional is not None:
                resolved.append(optional)

            out[player_field_id(spec.field_id, player_id)] = tuple(resolved)
    return out


def write_targets(loaded: LoadedScenario) -> dict[str, tuple[PlayerWriteTarget, ...]]:
    """{f"player:{field_id}:{player_id}": (primary, *mirrors)} -- the public
    entry point _resolve_all() backs. Targets[0] is always the primary (the
    one current_value() reads); the rest are the mirrors decision 4
    requires. Only meaningful once verify_player_block() has returned True
    for this file -- see that function."""
    return {key: tuple(r.target for r in resolved) for key, resolved in _resolve_all(loaded).items()}


def verify_player_block(loaded: LoadedScenario) -> bool:
    """True iff every Tier-1 write target -- primary, every mirror, and (for
    every P1_TO_P8_THEN_GAIA-layout spec) the read-only GAIA sentinel at
    index 8 -- reproduces exactly the value already parsed there. All-or-
    nothing across the whole per-player write surface, matching
    options_write_supported()/diplomacy_write_supported(): a partial write
    surface would offer some rows and silently drop others on a file where
    the walk half-worked, with no way for a user to tell those apart.

    The GAIA sentinel check exists because per-player arrays of identical
    values (e.g. every player on Dark Age) compare equal at a shifted
    offset -- GAIA's own slot generally differs from P1..P8's, which makes
    it the cheapest real shift detector available. Never written; read-only
    even once this returns True.

    Fails closed (False) on an untrusted anchor, on units_write_supported
    being False, or on an empty target set -- never on a comparison it
    never made.
    """
    if not loaded.units_write_supported:
        return False
    resolved = _resolve_all(loaded)
    if not resolved:
        return False

    body = loaded.decompressed_body

    def _matches(item: _Resolved) -> bool:
        target = item.target
        if target.offset < 0 or target.offset + target.length > len(body):
            return False
        try:
            expected = encode_target(target, item.parsed_value)
        except ValueError:
            return False
        return body[target.offset : target.offset + target.length] == expected

    for entries in resolved.values():
        if not all(_matches(item) for item in entries):
            return False

    for spec in specs_for(loaded):
        if spec.field_id in _NEVER_WRITABLE or spec.layout is not PlayerArrayLayout.P1_TO_P8_THEN_GAIA:
            continue
        gaia = _array_target(loaded, spec.section, spec.retriever, spec.struct_field, 8)
        if gaia is None or not _matches(gaia):
            return False

    return True


def encode_target(target: PlayerWriteTarget, value: int | str) -> bytes:
    """The bytes `target` would hold for `value`.

    `c256` goes through AoE2ScenarioParser's own fixed_chars_to_bytes()/
    settings.MAIN_CHARSET, never a hand-rolled codec -- raises if the
    encoded text does not fit the slot exactly, rather than silently
    producing the wrong length (fixed_chars_to_bytes() itself only pads
    short input; it does not truncate long input). `f32` raises rather than
    rounding: an integer above 2**24 is not f32-exact, and the four
    resource fields are s32 on their primary side, so a value that lost
    precision on its f32 mirror would desync the two stored copies instead
    of erroring.
    """
    if target.codec == "c256":
        encoded = fixed_chars_to_bytes(str(value), target.length)
        if len(encoded) != target.length:
            raise ValueError(f"{value!r} does not fit this {target.length}-byte slot")
        return encoded
    if target.codec == "f32":
        packed = _CODEC_STRUCT["f32"].pack(float(value))
        (roundtripped,) = _CODEC_STRUCT["f32"].unpack(packed)
        if roundtripped != value:
            raise ValueError(
                f"{value!r} is not exactly representable as f32 -- packing it "
                f"would silently desync this field's mirrored copy"
            )
        return packed
    return _CODEC_STRUCT[target.codec].pack(value)
