"""Qt-free byte offsets and read path for Diplomacy mode's per-player stance
grid and allied-victory flags -- default-tier testable without a
QApplication, mirroring descape/option_fields.py's split from its panel.

Every offset here is derived, never hardcoded, from
descape/scenario_io.py's diplomacy_section_end anchor plus each retriever's
own parsed byte length (scenario_io.retriever_length()) -- the same
backward-walk technique descape/options_model.py uses for GlobalVictory/Map/
Options, but covering the whole Diplomacy section (8 retrievers) rather than
a named subset, since every cell offset below is derived from where the
grid's own struct array starts. Every claim in this module was verified
against a 159-file scenario corpus spanning every DE structure version:

- The Diplomacy section is 12616 bytes on every DE structure version
  (v1.21, v1.36-v1.58): per_player_diplomacy (1024 bytes, a 16x16 grid of
  u32 stance cells), individual_victories (11520 bytes, unused and
  all-zero in the corpus), separator (4 bytes, 0xFFFFFF9D), then
  per_player_allied_victory (64 bytes, 16 u32 flags), then four 1-byte
  team scalars.
- Only indices 0-7 of the 16x16 grid (and of the 16-entry allied-victory
  array) are real -- they are players 1-8. Indices 8-15 are filler in
  every corpus file; this module never reads or exposes them, but a write
  path built on top of it must still preserve them verbatim.
- Stance values are {ALLY: 0, NEUTRAL: 1, ENEMY: 3} -- reuse
  AoE2ScenarioParser.datasets.trigger_lists.DiplomacyState (already wired
  into descape/trigger_fields.py) rather than redefining these labels; this
  module deals only in the raw u32.
- The grid is directional (stance[i][j] need not equal stance[j][i]) and
  its diagonal is real per-file data, not a constant -- both must round-trip
  exactly, which is why stance_offsets() below covers the full 8x8 including
  the diagonal rather than just the 7 opponent rows a panel would show.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from descape.scenario_io import LoadedScenario, retriever_length

# Players 1..8 only -- see module docstring. Indices 8..15 in every stored
# array are filler/GAIA-ambiguous and out of scope for this module.
NUM_PLAYERS = 8

_CELL_STRUCT = struct.Struct("<I")  # every stance/allied-victory cell is a u32

# Total Diplomacy section length measured across the corpus (see module
# docstring). Never used to place a byte -- every offset is derived from the
# backward walk below -- but checked once in verify_diplomacy_block() as an
# extra fail-closed signal: a future structure version that changed this
# would otherwise silently derive different, still self-consistent offsets
# instead of refusing to write.
_EXPECTED_SECTION_LENGTH = 12616


@dataclass(frozen=True)
class FieldOffset:
    offset: int  # byte offset within decompressed_body
    length: int  # byte length


@dataclass(frozen=True)
class _SectionLayout:
    retrievers: dict[str, FieldOffset]  # the section's 8 top-level retrievers
    stance_row_stride: int  # bytes between consecutive rows of the grid
    allied_cell_size: int  # bytes per per_player_allied_victory entry


def _section_layout(loaded: LoadedScenario) -> _SectionLayout | None:
    """Byte offset+length of each of the Diplomacy section's 8 top-level
    retrievers, walked backward from diplomacy_section_end -- the same
    technique descape/options_model.field_offsets() uses, but not stopping
    once a wanted subset is found, since every caller here needs the grid's
    own start offset. None if the anchor could not be trusted.
    """
    anchor = loaded.diplomacy_section_end
    if anchor < 0:
        return None
    section = loaded._scenario.sections.get("Diplomacy")
    if section is None:
        return None
    retriever_map = section.retriever_map
    pos = anchor
    retrievers: dict[str, FieldOffset] = {}
    for name in reversed(list(retriever_map)):
        length = retriever_length(retriever_map[name])
        pos -= length
        retrievers[name] = FieldOffset(pos, length)

    diplomacy_fo = retrievers.get("per_player_diplomacy")
    allied_fo = retrievers.get("per_player_allied_victory")
    if diplomacy_fo is None or allied_fo is None:
        return None
    num_rows = len(retriever_map["per_player_diplomacy"].data)
    num_allied = len(retriever_map["per_player_allied_victory"].data)
    if num_rows == 0 or num_allied == 0:
        return None

    return _SectionLayout(
        retrievers=retrievers,
        stance_row_stride=diplomacy_fo.length // num_rows,
        allied_cell_size=allied_fo.length // num_allied,
    )


# -- cell ids -----------------------------------------------------------------


def stance_cell_id(row_player: int, col_player: int) -> str:
    """Cell id for `row_player`'s stance toward `col_player` (1..8 each,
    matching in-game player numbering -- converted to a 0-based array index
    internally, per module docstring's "index i means player i+1")."""
    return f"stance:{row_player}:{col_player}"


def allied_victory_cell_id(player: int) -> str:
    return f"allied_victory:{player}"


def parse_cell_id(cell_id: str) -> tuple[str, int, int] | tuple[str, int]:
    """Inverse of stance_cell_id()/allied_victory_cell_id(): returns
    ("stance", row_player, col_player) or ("allied_victory", player)."""
    kind, *rest = cell_id.split(":")
    if kind == "stance" and len(rest) == 2:
        return "stance", int(rest[0]), int(rest[1])
    if kind == "allied_victory" and len(rest) == 1:
        return "allied_victory", int(rest[0])
    raise ValueError(f"not a diplomacy cell id: {cell_id!r}")


# -- offsets --------------------------------------------------------------


def stance_offsets(loaded: LoadedScenario) -> dict[str, FieldOffset]:
    """{cell_id: FieldOffset} for every stance[row][col] with row, col in
    1..NUM_PLAYERS -- the full 8x8 grid including the diagonal, which is
    real per-file data (see module docstring) and must round-trip even
    though a panel would render it read-only. Empty if the section's anchor
    could not be trusted.
    """
    layout = _section_layout(loaded)
    if layout is None:
        return {}
    base = layout.retrievers["per_player_diplomacy"].offset
    cells: dict[str, FieldOffset] = {}
    for row in range(1, NUM_PLAYERS + 1):
        for col in range(1, NUM_PLAYERS + 1):
            offset = base + layout.stance_row_stride * (row - 1) + _CELL_STRUCT.size * (col - 1)
            cells[stance_cell_id(row, col)] = FieldOffset(offset, _CELL_STRUCT.size)
    return cells


def allied_victory_offsets(loaded: LoadedScenario) -> dict[str, FieldOffset]:
    """{cell_id: FieldOffset} for every per_player_allied_victory[player],
    player in 1..NUM_PLAYERS. Empty if the section's anchor could not be
    trusted."""
    layout = _section_layout(loaded)
    if layout is None:
        return {}
    base = layout.retrievers["per_player_allied_victory"].offset
    cells: dict[str, FieldOffset] = {}
    for player in range(1, NUM_PLAYERS + 1):
        offset = base + layout.allied_cell_size * (player - 1)
        cells[allied_victory_cell_id(player)] = FieldOffset(offset, layout.allied_cell_size)
    return cells


def stance_value(loaded: LoadedScenario, row_player: int, col_player: int) -> int:
    """The raw stance value already parsed for `row_player`'s stance toward
    `col_player` (1..NUM_PLAYERS each) -- the read-time counterpart to
    stance_offsets(), for a caller (a panel) that wants the current value
    rather than where to patch it. Same traversal verify_diplomacy_block()
    checks byte-for-byte against decompressed_body.
    """
    section = loaded._scenario.sections["Diplomacy"]
    row_struct = section.retriever_map["per_player_diplomacy"].data[row_player - 1]
    return row_struct.retriever_map["stance_with_each_player"].data[col_player - 1]


def allied_victory_value(loaded: LoadedScenario, player: int) -> int:
    """The raw per_player_allied_victory[player] value already parsed
    (1..NUM_PLAYERS), the read-time counterpart to allied_victory_offsets()."""
    section = loaded._scenario.sections["Diplomacy"]
    return section.retriever_map["per_player_allied_victory"].data[player - 1]


def verify_diplomacy_block(loaded: LoadedScenario) -> bool:
    """True iff every stance and allied-victory cell's computed offset
    reproduces exactly the value already parsed there -- the load-time trust
    check mirroring options_model.verify_options_block(). Compared
    cell-by-cell against the parsed grid values
    (retriever.data[i].retriever_map["stance_with_each_player"].data[j])
    rather than against per_player_diplomacy.get_data_as_bytes(): a struct
    retriever's re-serialization is not guaranteed byte-identical in general
    (see scenario_io.retriever_length()'s docstring), so this removes that
    dependency entirely rather than relying on it happening to hold.

    Also gates on the section's total measured length matching every corpus
    file measured (_EXPECTED_SECTION_LENGTH) -- fails closed rather than
    silently deriving offsets against a structure version this was never
    checked against. Fails closed (False) if the anchor is unavailable.
    """
    layout = _section_layout(loaded)
    if layout is None:
        return False
    total_length = sum(fo.length for fo in layout.retrievers.values())
    if total_length != _EXPECTED_SECTION_LENGTH:
        return False

    body = loaded.decompressed_body
    stance = stance_offsets(loaded)
    allied = allied_victory_offsets(loaded)

    for row in range(1, NUM_PLAYERS + 1):
        for col in range(1, NUM_PLAYERS + 1):
            fo = stance[stance_cell_id(row, col)]
            if fo.offset < 0 or fo.offset + fo.length > len(body):
                return False
            (stored,) = _CELL_STRUCT.unpack_from(body, fo.offset)
            if stored != stance_value(loaded, row, col):
                return False

        fo = allied[allied_victory_cell_id(row)]
        if fo.offset < 0 or fo.offset + fo.length > len(body):
            return False
        (stored,) = _CELL_STRUCT.unpack_from(body, fo.offset)
        if stored != allied_victory_value(loaded, row):
            return False

    return True


def defined_player_ids(loaded: LoadedScenario) -> list[int]:
    """Player numbers (1..NUM_PLAYERS) this file marks active, in ascending
    order, read from DataHeader.player_data_1[0:NUM_PLAYERS].active
    directly rather than through FileHeader.player_count, which lives in
    header_bytes -- a buffer this app writes back verbatim and cannot derive
    a count from independently. Verified working on every examples/ file, including
    pre-1.53 versions where DataHeader.gaia_player_index does not yet exist,
    since this reader does not depend on it.

    Not assumed to be a contiguous prefix of 1..8: every examples/ file
    measured happens to be one, but this reads each slot's own flag rather
    than leaning on that pattern holding for every file DEscape might open.

    Consolidate into descape/player_fields.py once TODO.md's "Player
    options write path" item lands and owns editing this count, rather than
    leaving this reader duplicated between the two feature areas.
    """
    player_data_1 = loaded._scenario.sections["DataHeader"].retriever_map["player_data_1"].data
    return [
        i + 1 for i, entry in enumerate(player_data_1[:NUM_PLAYERS]) if entry.retriever_map["active"].data
    ]


def defined_player_count(loaded: LoadedScenario) -> int:
    """Number of players this file defines (2..8) -- see
    defined_player_ids() for the read path and why DataHeader rather than
    FileHeader.player_count."""
    return len(defined_player_ids(loaded))
