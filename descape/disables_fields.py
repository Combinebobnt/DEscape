"""Qt-free read/write codec for the Options section's per-player disable
lists -- the in-game editor's "Disable Objects" control (Buildings / Units /
Techs, per player), default-tier testable without a QApplication, mirroring
descape/option_fields.py's and descape/diplomacy_fields.py's split from
their panels.

The region is the whole front of the Options section: three blocks, one per
category, each a 16-wide u32 count array followed by eight variable-length
u32 id lists (players 1..8). Its layout, measured over every corpus file
this repo carries:

- It starts exactly at scenario_io's diplomacy_section_end anchor and runs
  to the `combat_mode` retriever, the first fixed-width field after it
  (20/20 files).
- Re-encoding it from the parsed counts and id lists reproduces
  decompressed_body byte-for-byte (20/20), which is what makes a
  browse-only save byte-identical and what verify_disables_block() checks.
- Count slots 8..15 are never nonzero in the corpus, but they are copied
  through verbatim rather than zeroed: nothing here has established what
  the game does with them, and AGENTS.md's pass-it-through-verbatim rules
  apply to a field whose meaning is unconfirmed.
- A disabled id is not necessarily in the library enums. 621 ("Town
  Center") appears in a buildings list and 35 ("Battering Ram") in a units
  list, neither of which object_catalog.objects() carries. Such an id must
  round-trip untouched.

Unlike every other field descape/options_model.py writes, these lists are
variable-length: an edit is a *splice* of the whole region, not an in-place
byte patch, which is why disables_splice() below has no equivalent in
serialize_patches() and why scenario_write._patch_disables() runs after the
Units/Triggers assembly rather than with the fixed-offset patches.

v1.21 (the AoC-era fixed-length form, 16 counts then 16x30 padded id slots)
is deliberately not handled: it is not loadable at all today, so
disables_region_span() returns None for it through the ordinary
missing-retriever path rather than through a version check.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence

from descape.scenario_io import LoadedScenario, retriever_length

# Section order, which is also encode_region()'s emission order. Not
# alphabetical and not the order the dialog's tabs use -- this is the order
# the bytes are in.
CATEGORIES = ("techs", "units", "buildings")

_SINGULAR = {"techs": "tech", "units": "unit", "buildings": "building"}

# Id lists are per player 1..8; the count arrays in front of them are 16
# wide. The extra eight slots are filler -- see the module docstring.
NUM_LIST_PLAYERS = 8
NUM_COUNT_SLOTS = 16

# The first fixed-width Options retriever after the disables region. Not
# itself an OptionFieldSpec, so the cross-check in verify_disables_block()
# anchors on all_techs (two u32s further on) instead.
_REGION_END_RETRIEVER = "combat_mode"

# What sits between _REGION_END_RETRIEVER and `all_techs`: combat_mode and
# naval_mode, one u32 each. The cross-check below is only as good as this
# constant, which is why it is checked against the backward walk rather
# than used to place a byte.
_BYTES_FROM_REGION_END_TO_ALL_TECHS = 8

_CELL = struct.Struct("<I")
_MAX_ID = 0xFFFFFFFF


def count_retriever_name(category: str) -> str:
    return f"per_player_number_of_disabled_{category}"


def ids_retriever_name(category: str, player_id: int) -> str:
    return f"disabled_{_SINGULAR[category]}_ids_player_{player_id}"


# -- field ids ---------------------------------------------------------------


def disables_field_id(category: str, player_id: int) -> str:
    """Synthetic id for one (category, player) list, in the same shape
    diplomacy_fields' "stance:3:5" and player_fields' "player:food:3" use --
    these ride OptionsEditModel's flat dicts alongside those."""
    if category not in CATEGORIES:
        raise ValueError(f"not a disables category: {category!r}")
    return f"disabled:{category}:{player_id}"


def parse_disables_field_id(field_id: str) -> tuple[str, int]:
    """Inverse of disables_field_id()."""
    parts = field_id.split(":")
    if len(parts) != 3 or parts[0] != "disabled" or parts[1] not in CATEGORIES:
        raise ValueError(f"not a disables field id: {field_id!r}")
    return parts[1], int(parts[2])


def all_field_ids() -> tuple[str, ...]:
    """Every (category, player) id this module covers -- 3 x 8."""
    return tuple(
        disables_field_id(category, player_id)
        for category in CATEGORIES
        for player_id in range(1, NUM_LIST_PLAYERS + 1)
    )


# -- read path ---------------------------------------------------------------


def _options_retriever_map(loaded: LoadedScenario):
    section = loaded._scenario.sections.get("Options")
    return None if section is None else section.retriever_map


def current_ids(loaded: LoadedScenario, category: str, player_id: int) -> tuple[int, ...]:
    """The disabled ids already parsed for this (category, player), in
    stored order. Order is preserved rather than sorted: the game does not
    appear to care, and keeping it is what lets an untouched list re-encode
    byte-identically."""
    retriever_map = _options_retriever_map(loaded)
    if retriever_map is None:
        return ()
    retriever = retriever_map.get(ids_retriever_name(category, player_id))
    if retriever is None:
        return ()
    return tuple(retriever.data)


def current_counts(loaded: LoadedScenario, category: str) -> tuple[int, ...]:
    """The whole stored count array for `category`, all 16 slots -- slots
    8..15 included, since encode_region() copies them through verbatim."""
    retriever_map = _options_retriever_map(loaded)
    if retriever_map is None:
        return ()
    retriever = retriever_map.get(count_retriever_name(category))
    if retriever is None:
        return ()
    return tuple(retriever.data)


# -- region span -------------------------------------------------------------


def disables_region_span(loaded: LoadedScenario) -> tuple[int, int] | None:
    """(start, end) of the disables region within decompressed_body, or None
    if it cannot be located.

    A *forward* walk, unlike every other offset in this family: the region
    is at the front of its section, so there is nothing variable-length
    ahead of it to skip, and the retrievers it walks over are exactly the
    ones being measured. _all_techs_offset() below walks backward over the
    same section independently, and verify_disables_block() requires the two
    to agree.
    """
    start = loaded.diplomacy_section_end
    if start < 0:
        return None
    retriever_map = _options_retriever_map(loaded)
    if retriever_map is None:
        return None
    for category in CATEGORIES:
        if count_retriever_name(category) not in retriever_map:
            return None
        for player_id in range(1, NUM_LIST_PLAYERS + 1):
            if ids_retriever_name(category, player_id) not in retriever_map:
                return None

    pos = start
    for name, retriever in retriever_map.items():
        if name == _REGION_END_RETRIEVER:
            break
        pos += retriever_length(retriever)
    else:
        return None  # combat_mode absent: this is not a layout we can place
    if pos <= start or pos > loaded.options_section_end:
        return None
    return start, pos


def _all_techs_offset(loaded: LoadedScenario) -> int | None:
    """Byte offset of the Options section's `all_techs` retriever, walked
    *backward* from options_section_end -- the independent second opinion
    verify_disables_block() cross-checks the forward walk against.

    Its own walk rather than options_model.field_offsets(): that module
    imports this one (OptionsEditModel hosts the disables field set), so
    reaching back into it would be a cycle. descape/diplomacy_fields.py
    sets the same precedent for the same reason, and a walk written here is
    in any case more independent of the forward one than a shared helper
    would be.
    """
    anchor = loaded.options_section_end
    if anchor < 0:
        return None
    retriever_map = _options_retriever_map(loaded)
    if retriever_map is None or "all_techs" not in retriever_map:
        return None
    pos = anchor
    for name in reversed(list(retriever_map)):
        pos -= retriever_length(retriever_map[name])
        if name == "all_techs":
            return pos
    return None


# -- write path --------------------------------------------------------------


def encode_region(
    loaded: LoadedScenario,
    edits: Mapping[tuple[str, int], Sequence[int]] | None = None,
) -> bytes:
    """The whole disables region rebuilt from the parsed values, with every
    (category, player) in `edits` substituted.

    Rebuilds the region wholesale rather than patching one list, because a
    length change in any list shifts every byte after it. An unedited list
    re-encodes from its own parsed values, which is what keeps
    encode_region(loaded, {}) byte-identical to the file -- the property
    verify_disables_block() turns into the load-time gate, so the gate and
    the writer are the same code path and cannot drift apart.

    Counts 0..7 are re-derived from the (possibly edited) list lengths;
    slots 8..15 are copied through verbatim, never zeroed. See the module
    docstring.
    """
    edits = dict(edits or {})
    out = bytearray()
    for category in CATEGORIES:
        counts = list(current_counts(loaded, category))
        lists: list[tuple[int, ...]] = []
        for player_id in range(1, NUM_LIST_PLAYERS + 1):
            key = (category, player_id)
            ids = (
                tuple(int(v) for v in edits[key])
                if key in edits
                else current_ids(loaded, category, player_id)
            )
            lists.append(ids)
            if player_id - 1 < len(counts):
                counts[player_id - 1] = len(ids)
        out += struct.pack(f"<{len(counts)}I", *counts)
        for ids in lists:
            out += struct.pack(f"<{len(ids)}I", *ids)
    return bytes(out)


def verify_disables_block(loaded: LoadedScenario) -> bool:
    """The load-time trust check, and deliberately stronger than
    options_model.verify_options_block()'s per-field byte comparison:

    1. **Identity round-trip.** encode_region(loaded, {}) must reproduce
       decompressed_body over the located span exactly. The gate and the
       writer are then the same code, so a codec bug fails the gate rather
       than silently writing a wrong region.
    2. **Independent anchor cross-check.** The forward-walked region end
       plus the two u32s after it (combat_mode, naval_mode) must equal the
       backward-walked `all_techs` offset. Two walks from opposite ends of
       the section agreeing is what rules out a layout where the forward
       walk happens to be self-consistent but misplaced.

    Fails closed on a missing anchor, a missing retriever, an empty region,
    or a count array narrower than the eight player slots it must carry.
    """
    span = disables_region_span(loaded)
    if span is None:
        return False
    start, end = span
    body = loaded.decompressed_body
    if start < 0 or end > len(body):
        return False

    for category in CATEGORIES:
        if len(current_counts(loaded, category)) < NUM_LIST_PLAYERS:
            return False

    if encode_region(loaded, {}) != body[start:end]:
        return False

    all_techs = _all_techs_offset(loaded)
    if all_techs is None:
        return False
    return end + _BYTES_FROM_REGION_END_TO_ALL_TECHS == all_techs


def disables_splice(
    loaded: LoadedScenario,
    edits: Mapping[tuple[str, int], Sequence[int]],
) -> tuple[int, int, bytes] | None:
    """(start, end, replacement) for the whole disables region, or None when
    the span or the gate is unavailable -- the same shape
    player_fields.player_data_1_splice() hands scenario_write.py."""
    span = disables_region_span(loaded)
    if span is None or not verify_disables_block(loaded):
        return None
    start, end = span
    return start, end, encode_region(loaded, edits)


def coerce_ids(value) -> tuple[int, ...]:
    """A caller's id sequence as a deduped tuple of u32s, in first-seen
    order. Raises ValueError for anything else.

    Deduping here rather than in the dialog is what makes an edit-then-
    revert comparison in OptionsEditModel.set_value() meaningful: the stored
    original is a tuple, and `[1, 2] == (1, 2)` is False, so a list that was
    never normalized would leave a phantom pending entry behind.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{value!r} is not a sequence of disabled ids")
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{item!r} is not an integer id")
        if not 0 <= item <= _MAX_ID:
            raise ValueError(f"{item} is outside the u32 range a disabled id is stored as")
        if item not in ids:
            ids.append(item)
    return tuple(ids)
