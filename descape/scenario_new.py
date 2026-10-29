"""Generates a blank N x N (or W x H) .aoe2scenario map on demand, for File > New Map… --
a sibling of descape/scenario_write.py, not a change to it. write_scenario()
patches the terrain struct array in place at its original length and never
touches map_width/map_height (see its own docstring); it structurally cannot
resize. A resize needs its own splice, the same relationship
tools/strip_units.py already has to write_scenario().

Why the splice works: a blank template's terrain array is BLANK_TERRAIN_STRUCT
repeated once per tile (terrain_id 0, elevation 0, layer -1 -- confirmed
against every tile of all three shipped templates, not sampled). That is
the default fill; File > New Map…'s dialog can pick another terrain and
elevation, which changes only bytes 0 and 1 of each struct
(blank_terrain_struct()). In the
decompressed body, map_width and map_height are two consecutive little-endian
s32 immediately preceding the terrain array: Map's last three retrievers are
map_width, map_height, terrain_data, in that order, in every DE
versions/DE/*/structure.json, library and repo (v1.36 through v1.59) -- so the -8/-4
offsets are a structural guarantee, not an accident of one file. Applying this
splice to the real 1.58 120x120 export reproduces the real 1.58 240x240 and
480x480 exports byte-for-byte, except for two fields that are deliberately
inherited rather than touched:

- FileHeader's timestamp_of_last_save (u32 at header offset 12).
- DataHeader's filename (a length-prefixed str16, its terminal field). It
  can't be rewritten to a different-length name -- doing so would shift
  terrain_block_offset for every section after it.

See tests/test_scenario_new.py for the byte-for-byte comparison against the
real exports (relocated to tests/fixtures/real_blank_*.aoe2scenario as byte
oracles once this landed -- see that module's docstring for why those files
must never be regenerated from the donor).

That 1.58 120x120 export was the shipped donor until 1.59 and is now a test
fixture, tests/fixtures/blank_120x120_v158.aoe2scenario. The shipped donor is
the same file with its three version fields set to 1.59 by
tools/gen_blank_template.py, so every new map is a 1.59 scenario.
tests/test_blank_template.py compares it against a blank map saved by the
1.59 game editor.

Trees and eye candy (blank_scenario_bytes()'s trees/doodads) are planted
into the generated file, not edited in after load: the spliced bytes are
parsed, terrain_units.plan_fill()'s units add_many()ed to GAIA under the
donor's own game's tables (game_dataset.activated()), and the body rebuilt
through scenario_write.build_patched_body(), the same unit write a save
makes. So a new map with trees opens unmodified and with nothing to undo,
like any file that already had them. Planting nothing returns the bare
bytes untouched.

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

import random
import struct
from pathlib import Path

from descape import game_dataset, iso_geometry, player_fields, terrain_units
from descape.scenario_io import (
    BLANK_ROR_TEMPLATE_PATH,
    BLANK_TEMPLATE_PATH,
    TERRAIN_STRUCT_SIZE,
    LoadedScenario,
    load_map_and_units,
    load_map_and_units_from_bytes,
)
from descape.scenario_write import WriteBlockedError, _compress_bytes, build_patched_body
from descape.unit_model import UnitEditModel, UnitEditsUnavailableError

BLANK_TERRAIN_STRUCT = b"\x00\x00\x00\xff\xff\xff\xff"

# The 7 sizes real AoE2:DE scenarios use -- confirmed empirically, not from
# memory: every one of the 20 real campaign files in examples/ (scenario
# versions 1.37-1.59) is exactly one of these. File > New Map…'s Size presets.
STANDARD_MAP_SIZES = (120, 144, 168, 200, 220, 240, 480)

# In-game names for each STANDARD_MAP_SIZES entry, used to label File > New
# Map…'s Size presets. Sourced from AoE2:DE community references (Liquipedia et
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

# GUI-only: above this, File > New Map…'s Custom size asks for confirmation
# before generating. Not enforced here -- validate_tiles() only knows the
# hard bounds above. The labelled 480 preset does NOT confirm: it's a
# deliberate choice on the one large size already in-game proven, not a
# custom guess, so a confirm there would just be a nag.
LARGE_MAP_CONFIRM_TILES = 240

# GUI-only: New Map… asks first above this many planted units (expected_unit_count()).
# A full forest: Giant (57,600) is not asked, Ludicrous (230,400, ~1 GB more memory) is.
NEW_MAP_UNIT_CONFIRM = 100_000

# Planted units belong to GAIA, as the Terrain tool's do.
_GAIA = 0

_SIZE_STRUCT = struct.Struct("<ii")

_CAMERA_STRUCTS = {
    "f32": struct.Struct("<f"),
    "s16": struct.Struct("<h"),
    "s32": struct.Struct("<i"),
}


class MapSizeError(ValueError):
    """Raised by validate_map_size()/blank_body() for a width or height
    outside [MIN_MAP_TILES, MAX_MAP_TILES]."""


class BlankFillError(ValueError):
    """Raised by blank_terrain_struct() for a terrain id outside 0..255 or
    an elevation outside iso_geometry's MIN_ELEVATION..MAX_ELEVATION."""


class BlankGenerationError(Exception):
    """Raised instead of generating, when the donor scenario doesn't meet
    blank_body()'s preconditions -- see its docstring."""


def _validate_axis(value: int, axis: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MapSizeError(f"Map {axis} must be an int, got {value!r}")
    if not (MIN_MAP_TILES <= value <= MAX_MAP_TILES):
        raise MapSizeError(
            f"Map {axis} {value} outside supported range [{MIN_MAP_TILES}, {MAX_MAP_TILES}]"
        )
    return value


def validate_map_size(width: int, height: int | None = None) -> tuple[int, int]:
    """Returns (width, height), height defaulting to width (square), or
    raises MapSizeError. Pure -- no I/O, no Qt -- so batch_api/tools/ callers
    get the hard cap for free without going through the GUI's dialog. Each
    axis is bounded on its own, so a legal rectangle is any pair in range."""
    if height is None:
        return _validate_axis(width, "size"), width
    return _validate_axis(width, "width"), _validate_axis(height, "height")


def validate_tiles(tiles: int) -> int:
    """The square form of validate_map_size(), kept for existing callers."""
    return validate_map_size(tiles)[0]


def clamp_cameras(body: bytearray, fields, shift: int, new_w: int, new_h: int) -> int:
    """Clamps every camera coordinate in `fields` (player_fields.camera_fields())
    above the new map's last tile down to it, in place, reading the *current*
    value from `body` at each field's offset plus `shift`. Values below 0 (the
    unset -1) stay as they are. Returns how many were changed. Shared by File
    > New Map… and Map > Resize Map…."""
    changed = 0
    for field in fields:
        t = field.target
        codec = _CAMERA_STRUCTS[t.codec]
        offset = t.offset + shift
        (value,) = codec.unpack_from(body, offset)
        bound = (new_w if field.axis == "x" else new_h) - 1
        if value > bound:
            codec.pack_into(body, offset, float(bound) if t.codec == "f32" else bound)
            changed += 1
    return changed


def blank_template_path_for(game_data: str) -> Path:
    """The donor for a game_dataset id: Return of Rome's own blank map for
    ROR, the AoE2 one for everything else."""
    return BLANK_ROR_TEMPLATE_PATH if game_data == game_dataset.ROR else BLANK_TEMPLATE_PATH


def blank_terrain_struct(terrain_id: int = 0, elevation: int = 0) -> bytes:
    """BLANK_TERRAIN_STRUCT with its terrain_id (byte 0) and elevation (byte
    1) replaced; the unused bytes and layer -1 stay. Raises BlankFillError
    for a value outside its range. The defaults give BLANK_TERRAIN_STRUCT."""
    for name, value, low, high in (
        ("terrain id", terrain_id, 0, 255),
        ("elevation", elevation, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise BlankFillError(f"New map {name} {value!r} outside [{low}, {high}]")
    return bytes((terrain_id, elevation)) + BLANK_TERRAIN_STRUCT[2:]


def blank_body(
    donor: LoadedScenario,
    width: int,
    height: int | None = None,
    *,
    terrain_id: int = 0,
    elevation: int = 0,
) -> bytes:
    """Returns a new decompressed body: donor.decompressed_body with
    map_width/map_height rewritten to (width, height) and the terrain struct
    array replaced by width*height copies of blank_terrain_struct(terrain_id,
    elevation) (BLANK_TERRAIN_STRUCT by default). height
    defaults to width (square). Everything
    before map_width and everything after the old terrain array -- Units, the
    never-parsed trigger tail -- is carried through byte-for-byte untouched,
    except a player camera past the new map's last tile, which is clamped
    onto it (clamp_cameras()). The Return of Rome donor has one at (29.8,
    90.2), so a RoR map of 91 tiles or fewer needs it; every AoE2 donor camera is
    at 60, so AoE2 output is unchanged.

    A non-square body is generated byte-correctly but does not load through
    the pinned library yet (its MapManager.terrain setter requires a perfect
    square); tools/gen_blank_maps.py verifies those on raw bytes instead.

    Raises:
        MapSizeError: width or height is outside [MIN_MAP_TILES, MAX_MAP_TILES].
        BlankFillError: terrain_id or elevation is out of range.
        BlankGenerationError: the donor doesn't meet this splice's
            preconditions -- either its own terrain block failed load-time
            verification (terrain_write_supported is False, so
            terrain_block_offset is -1 and slicing from it would silently
            corrupt the body), or the raw map_width/map_height bytes at
            terrain_block_offset - 8 don't match what was parsed (the same
            check scenario_io._verify_terrain_block() makes at load time, run
            again here because a caller could hand in a donor that was
            mutated after loading), or its camera fields could not be located.
    """
    width, height = validate_map_size(width, height)
    fill = blank_terrain_struct(terrain_id, elevation)

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

    fields = player_fields.camera_fields(donor)
    if fields is None:
        raise BlankGenerationError(f"{donor.path}: camera positions could not be located")
    # Before the splice, so the donor's own offsets still hold.
    body = bytearray(body)
    clamp_cameras(body, fields, 0, width, height)

    return bytes(
        body[:size_offset]
        + _SIZE_STRUCT.pack(width, height)
        + fill * (width * height)
        + body[old_end:]
    )


def blank_scenario_bytes(
    width: int,
    donor: LoadedScenario | None = None,
    height: int | None = None,
    *,
    terrain_id: int = 0,
    elevation: int = 0,
    game_data: str = game_dataset.AOE2,
    trees: bool = False,
    doodads: bool = False,
    rng: random.Random | None = None,
) -> bytes:
    """Full .aoe2scenario file bytes for a blank width x height map (square
    when height is None), every tile terrain_id at elevation: donor's
    FileHeader verbatim, followed by the recompressed spliced body. donor
    defaults to game_data's shipped 120x120 template
    (blank_template_path_for()).

    trees / doodads plant GAIA trees / eye candy on every tile the way the
    Terrain tool does (terrain_units.plan_fill(), rolled with `rng`, a fresh
    Random() when None), under the donor's own game's tables. The defaults,
    and a terrain with nothing enabled to plant, give the bare bytes exactly.

    Raises BlankGenerationError, besides blank_body()'s errors, when the
    planted units cannot be written into the spliced map."""
    if donor is None:
        donor = load_map_and_units(blank_template_path_for(game_data))
    body = blank_body(donor, width, height, terrain_id=terrain_id, elevation=elevation)
    data = donor.header_bytes + _compress_bytes(body)
    if not (trees or doodads):
        return data
    width, height = validate_map_size(width, height)
    with game_dataset.activated(game_dataset.from_variant(donor.game_variant)):
        specs = terrain_units.plan_fill(
            width, height, terrain_id, trees=trees, doodads=doodads, rng=random.Random() if rng is None else rng
        )
        if not specs:
            return data
        return _planted_bytes(data, specs, f"blank_{width}x{height}.aoe2scenario")


def _planted_bytes(data: bytes, specs: list, display_path: str) -> bytes:
    """`data` reparsed with `specs` add_many()ed to GAIA and written back
    through build_patched_body(), the way a save writes them. No undo
    history: a commit would capture every unit for a record nobody keeps."""
    loaded = load_map_and_units_from_bytes(data, display_path)
    try:
        model = UnitEditModel(loaded)
        model.add_many(_GAIA, specs)
        header, body = build_patched_body(loaded, units=model)
    except (UnitEditsUnavailableError, WriteBlockedError) as e:
        raise BlankGenerationError(f"Could not plant {len(specs)} units into the new map: {e}") from e
    return header + _compress_bytes(body)


def load_blank_scenario(
    width: int,
    donor: LoadedScenario | None = None,
    height: int | None = None,
    *,
    terrain_id: int = 0,
    elevation: int = 0,
    game_data: str = game_dataset.AOE2,
    trees: bool = False,
    doodads: bool = False,
    rng: random.Random | None = None,
) -> LoadedScenario:
    """blank_scenario_bytes() reloaded through the real loader -- offsets are
    never carried across a length-changing splice, the same discipline
    tools/strip_units.py uses ('reload through the real loader, not just
    re-decompress in place'), and none across the planted units' splice."""
    width, height = validate_map_size(width, height)
    data = blank_scenario_bytes(
        width,
        donor,
        height,
        terrain_id=terrain_id,
        elevation=elevation,
        game_data=game_data,
        trees=trees,
        doodads=doodads,
        rng=rng,
    )
    return load_map_and_units_from_bytes(data, f"blank_{width}x{height}.aoe2scenario")


__all__ = [
    "BLANK_TERRAIN_STRUCT",
    "LARGE_MAP_CONFIRM_TILES",
    "MAX_MAP_TILES",
    "MIN_MAP_TILES",
    "NEW_MAP_UNIT_CONFIRM",
    "STANDARD_MAP_SIZES",
    "STANDARD_MAP_SIZE_NAMES",
    "BlankFillError",
    "BlankGenerationError",
    "MapSizeError",
    "blank_body",
    "blank_scenario_bytes",
    "blank_template_path_for",
    "blank_terrain_struct",
    "clamp_cameras",
    "load_blank_scenario",
    "validate_map_size",
    "validate_tiles",
]
