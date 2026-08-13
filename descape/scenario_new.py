"""Generates a blank N x N .aoe2scenario map on demand, for File > New Map --
a sibling of descape/scenario_write.py, not a change to it. write_scenario()
patches the terrain struct array in place at its original length and never
touches map_width/map_height (see its own docstring); it structurally cannot
resize. A resize needs its own splice, the same relationship
tools/strip_units.py already has to write_scenario().

Why the splice works: a blank template's terrain array is BLANK_TERRAIN_STRUCT
repeated once per tile (terrain_id 0, elevation 0, layer -1 -- confirmed
against every tile of all three shipped templates, not sampled). In the
decompressed body, map_width and map_height are two consecutive little-endian
s32 immediately preceding the terrain array: Map's last three retrievers are
map_width, map_height, terrain_data, in that order, in all 19 DE
versions/DE/*/structure.json files from v1.36 through v1.58 -- so the -8/-4
offsets are a structural guarantee, not an accident of one file. Applying this
splice to the shipped 120x120 donor reproduces the real 240x240 and 480x480
exports byte-for-byte, except for two fields that are deliberately inherited
rather than touched:

- FileHeader's timestamp_of_last_save (u32 at header offset 12).
- DataHeader's filename (a length-prefixed str16, its terminal field). It
  can't be rewritten to a different-length name -- doing so would shift
  terrain_block_offset for every section after it.

See tests/test_scenario_new.py for the byte-for-byte comparison against the
real exports (relocated to tests/fixtures/real_blank_*.aoe2scenario as byte
oracles once this landed -- see that module's docstring for why those files
must never be regenerated from the donor).

MIN_MAP_TILES is set by a structural constraint, not the donor's own size:
Units.player_data_3's editor_camera_x/y and initial_camera_x/y are (60, 60)
in all three shipped exports regardless of map size -- the game does not
scale them -- so a generated map with N <= 60 would inherit a camera outside
its own bounds. 61 is the smallest size clear of that hazard; whether the
game actually accepts a map that small has not been confirmed in-game as of
2026-08-12 -- tracked in this project's private planning docs, not a public
path.
"""

from __future__ import annotations

import struct

from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    TERRAIN_STRUCT_SIZE,
    LoadedScenario,
    load_map_and_units,
)
from descape.scenario_write import _compress_bytes

BLANK_TERRAIN_STRUCT = b"\x00\x00\x00\xff\xff\xff\xff"

# The 7 sizes real AoE2:DE scenarios use -- confirmed empirically, not from
# memory: every one of the 20 real campaign files in examples/ (scenario
# versions 1.37-1.58) is exactly one of these. File > New Map's size submenu.
STANDARD_MAP_SIZES = (120, 144, 168, 200, 220, 240, 480)

# In-game names for each STANDARD_MAP_SIZES entry, used to label File > New
# Map's submenu. Sourced from AoE2:DE community references (Liquipedia et
# al.), not read off the real Scenario Editor's Map tab -- if the in-game
# wording ever turns out to differ, fix it here.
STANDARD_MAP_SIZE_NAMES = {
    120: "Tiny",
    144: "Small",
    168: "Medium",
    200: "Normal",
    220: "Large",
    240: "Giant",
    480: "Ludicrous",
}

# 61 is the structural floor: Units.player_data_3's editor_camera_x/y and
# initial_camera_x/y are (60, 60) in all three real game blank exports
# regardless of map size (the game doesn't scale them), so a generated map
# with N <= 60 would inherit a camera outside its own bounds. Not yet
# confirmed in-game (opens cleanly, correct dimensions, survives
# save-and-reopen) -- tracked in this project's private planning docs, not
# a public path.
MIN_MAP_TILES = 61
MAX_MAP_TILES = 480

# GUI-only: above this, File > New Map > Custom size... asks for confirmation
# before generating. Not enforced here -- validate_tiles() only knows the
# hard bounds above. The labelled 480 preset does NOT confirm: it's a
# deliberate choice on the one large size already in-game proven, not a
# custom guess, so a confirm there would just be a nag.
LARGE_MAP_CONFIRM_TILES = 240

_SIZE_STRUCT = struct.Struct("<ii")


class MapSizeError(ValueError):
    """Raised by validate_tiles()/blank_body() for a tiles value outside
    [MIN_MAP_TILES, MAX_MAP_TILES]."""


class BlankGenerationError(Exception):
    """Raised instead of generating, when the donor scenario doesn't meet
    blank_body()'s preconditions -- see its docstring."""


def validate_tiles(tiles: int) -> int:
    """Returns tiles unchanged, or raises MapSizeError. Pure -- no I/O, no Qt
    -- so batch_api/tools/ callers get the hard cap for free without going
    through the GUI's dialog."""
    if not isinstance(tiles, int) or isinstance(tiles, bool):
        raise MapSizeError(f"Map size must be an int, got {tiles!r}")
    if not (MIN_MAP_TILES <= tiles <= MAX_MAP_TILES):
        raise MapSizeError(
            f"Map size {tiles} outside supported range [{MIN_MAP_TILES}, {MAX_MAP_TILES}]"
        )
    return tiles


def blank_body(donor: LoadedScenario, tiles: int) -> bytes:
    """Returns a new decompressed body: donor.decompressed_body with
    map_width/map_height rewritten to (tiles, tiles) and the terrain struct
    array replaced by tiles*tiles copies of BLANK_TERRAIN_STRUCT. Everything
    before map_width and everything after the old terrain array -- Units, the
    never-parsed trigger tail -- is carried through byte-for-byte untouched.

    Raises:
        MapSizeError: tiles is outside [MIN_MAP_TILES, MAX_MAP_TILES].
        BlankGenerationError: the donor doesn't meet this splice's
            preconditions -- either its own terrain block failed load-time
            verification (terrain_write_supported is False, so
            terrain_block_offset is -1 and slicing from it would silently
            corrupt the body), or the raw map_width/map_height bytes at
            terrain_block_offset - 8 don't match what was parsed (the same
            check scenario_io._verify_terrain_block() makes at load time, run
            again here because a caller could hand in a donor that was
            mutated after loading).
    """
    validate_tiles(tiles)

    if not donor.terrain_write_supported:
        raise BlankGenerationError(
            f"{donor.path}: terrain block failed load-time verification -- "
            "refusing to splice bytes that may not be where they're expected."
        )

    body = donor.decompressed_body
    offset = donor.terrain_block_offset
    old_w, old_h = donor.map_manager.map_width, donor.map_manager.map_height
    old_end = offset + TERRAIN_STRUCT_SIZE * old_w * old_h
    if old_end > len(body):
        raise BlankGenerationError(
            f"{donor.path}: terrain block ({offset}..{old_end}) runs past the "
            f"decompressed body ({len(body)} bytes)"
        )

    size_offset = offset - _SIZE_STRUCT.size
    if _SIZE_STRUCT.unpack_from(body, size_offset) != (old_w, old_h):
        raise BlankGenerationError(
            f"{donor.path}: map_width/map_height bytes at {size_offset} don't "
            f"match the parsed size ({old_w}, {old_h}) -- donor may be stale"
        )

    return (
        body[:size_offset]
        + _SIZE_STRUCT.pack(tiles, tiles)
        + BLANK_TERRAIN_STRUCT * (tiles * tiles)
        + body[old_end:]
    )


def blank_scenario_bytes(tiles: int, donor: LoadedScenario | None = None) -> bytes:
    """Full .aoe2scenario file bytes for a blank tiles x tiles map: donor's
    FileHeader verbatim, followed by the recompressed spliced body. donor
    defaults to the shipped BLANK_TEMPLATE_PATH (120x120)."""
    if donor is None:
        donor = load_map_and_units(BLANK_TEMPLATE_PATH)
    return donor.header_bytes + _compress_bytes(blank_body(donor, tiles))


def load_blank_scenario(tiles: int, donor: LoadedScenario | None = None) -> LoadedScenario:
    """blank_scenario_bytes() reloaded through the real loader -- offsets are
    never carried across a length-changing splice, the same discipline
    tools/strip_units.py uses ('reload through the real loader, not just
    re-decompress in place'). Imported here rather than at module scope to
    avoid a cycle: scenario_io does not import this module."""
    from descape.scenario_io import load_map_and_units_from_bytes

    data = blank_scenario_bytes(tiles, donor)
    return load_map_and_units_from_bytes(data, f"blank_{tiles}x{tiles}.aoe2scenario")


__all__ = [
    "BLANK_TERRAIN_STRUCT",
    "STANDARD_MAP_SIZES",
    "STANDARD_MAP_SIZE_NAMES",
    "MIN_MAP_TILES",
    "MAX_MAP_TILES",
    "LARGE_MAP_CONFIRM_TILES",
    "MapSizeError",
    "BlankGenerationError",
    "validate_tiles",
    "blank_body",
    "blank_scenario_bytes",
    "load_blank_scenario",
]
