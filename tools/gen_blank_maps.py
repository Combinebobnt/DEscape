#!/usr/bin/env python3
"""Writes real .aoe2scenario files for the in-game verification gate that
descape/scenario_new.py's splice needs before it's trusted at sizes the real
game never produced for us.

Deliberately does not go through descape.scenario_write.write_scenario():
that function only ever patches an existing terrain array in place at its
original length (see its own docstring) and has no way to write a resized
one -- scenario_new.blank_scenario_bytes() is the whole point of this tool,
not something write_scenario() could do instead.

A square size (`--sizes 168`): generate, write, then reload through the real
load_map_and_units() -- not just re-decompress the bytes in memory -- and
assert the same invariants tests/test_scenario_new.py checks, the same
reload-and-verify discipline tools/strip_units.py uses ("reload through the
real loader, not just re-decompress in place") rather than trusting
in-place state.

A non-square size (`--sizes 120x168 168x120`, width x height) is the
non-square in-game gate. A blank uniform grid cannot gate it: the bytes are
identical whether the game reads terrain row-major by width (i = y*w + x,
what DEscape assumes) or column-major by height (i = x*h + y). So two
asymmetric markers are written onto the body before compressing (see
marker_tiles()), and the file is verified on raw bytes, since the pinned
library's MapManager cannot load a non-square map yet. The tool prints what
the user should see in the real editor for each file.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    FORBIDDEN_WRITE_MARKER,
    TERRAIN_STRUCT_SIZE,
    LoadedScenario,
    is_under_compatdata,
    load_map_and_units,
)
from descape.scenario_new import (
    BLANK_TERRAIN_STRUCT,
    STANDARD_MAP_SIZES,
    blank_body,
    blank_scenario_bytes,
    validate_map_size,
)
from descape.scenario_write import WriteBlockedError, _compress_bytes

# High-contrast against the blank map's GRASS_1 (0) and against each other.
STRIPE_TERRAIN = 22  # WATER_DEEP
BLOCK_TERRAIN = 32  # SNOW

# The stripe sits this many tiles in from the origin edge; the block spans
# BLOCK_SPAN tiles per axis, BLOCK_INSET in from the short-axis origin edge
# and ending BLOCK_FAR_INSET in from the far end of the long axis.
STRIPE_INSET = 10
BLOCK_INSET = 5
BLOCK_SPAN = 11
BLOCK_FAR_INSET = 7

_SIZE_STRUCT = struct.Struct("<ii")


class GenerationVerificationError(Exception):
    """Raised instead of leaving a file on disk whose reload-through-the-
    real-loader check didn't match what was generated."""


def parse_size(text: str) -> tuple[int, int]:
    """'168' -> (168, 168); '120x168' -> (120, 168), width first."""
    parts = text.lower().split("x")
    try:
        if len(parts) == 1:
            return validate_map_size(int(parts[0]))
        if len(parts) == 2:
            return validate_map_size(int(parts[0]), int(parts[1]))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    raise argparse.ArgumentTypeError(f"expected N or WxH, got {text!r}")


def marker_tiles(width: int, height: int) -> dict[int, int]:
    """{tile index: terrain id} for the two asymmetric markers on a
    non-square map, indexed i = y*width + x (the reading under test).

    On a tall map (height > width), the plan's own markers: the whole row
    y == STRIPE_INSET, and an 11x11 block at x = 5..15, y = (height - 18) ..
    (height - 8), which is 150..160 on 120x168. The block's y range lies past
    the short axis, so a transposed reading cannot place it where expected.
    A wide map gets the exact transpose (column x == STRIPE_INSET, block at
    x = (width - 18)..(width - 8), y = 5..15), so the 168x120 file is the
    120x168 file mirrored across the map's diagonal.
    """
    if width == height:
        raise ValueError("markers are for non-square maps only")
    tall = height > width
    short, long_ = (width, height) if tall else (height, width)
    far_lo = long_ - BLOCK_FAR_INSET - BLOCK_SPAN
    if far_lo < short:
        raise ValueError(
            f"{width}x{height}: the corner block would start at {far_lo} on the long axis, "
            f"inside the short axis ({short}), so a transposed reading could not misplace it"
        )

    def index(s: int, lg: int) -> int:
        x, y = (s, lg) if tall else (lg, s)
        return y * width + x

    tiles = {index(s, STRIPE_INSET): STRIPE_TERRAIN for s in range(short)}
    for s in range(BLOCK_INSET, BLOCK_INSET + BLOCK_SPAN):
        for lg in range(far_lo, far_lo + BLOCK_SPAN):
            tiles[index(s, lg)] = BLOCK_TERRAIN
    return tiles


def apply_markers(body: bytes, terrain_offset: int, markers: dict[int, int]) -> bytes:
    """Overwrites byte 0 (terrain_id) of each marker tile's 7-byte struct."""
    patched = bytearray(body)
    for i, terrain_id in markers.items():
        patched[terrain_offset + TERRAIN_STRUCT_SIZE * i] = terrain_id
    return bytes(patched)


def _verify(path: Path, tiles: int) -> None:
    reloaded = load_map_and_units(path)
    mm = reloaded.map_manager
    if not (mm.map_width == mm.map_height == tiles):
        raise GenerationVerificationError(
            f"{path}: reloaded as {mm.map_width}x{mm.map_height}, expected {tiles}x{tiles}"
        )
    if not reloaded.map_is_square:
        raise GenerationVerificationError(f"{path}: not square after reload")
    if not reloaded.terrain_write_supported:
        raise GenerationVerificationError(f"{path}: terrain block failed reload verification")
    if not all(t.terrain_id == 0 and t.elevation == 0 and t.layer == -1 for t in mm.terrain):
        raise GenerationVerificationError(f"{path}: not uniformly blank after reload")
    if sum(len(units) for units in reloaded.unit_manager.units) != 0:
        raise GenerationVerificationError(f"{path}: has units after reload")


def _verify_raw(path: Path, donor: LoadedScenario, width: int, height: int, markers: dict[int, int]) -> None:
    """The non-square branch: no loader can read the file yet, so check its
    bytes. The header is the donor's verbatim, the size pair sits at
    terrain_block_offset - 8, the block is 7*w*h long with the donor's tail
    after it, and against the unmarked blank body only the marker tiles
    differ, each only in byte 0."""
    fail = GenerationVerificationError
    data = path.read_bytes()
    header = donor.header_bytes
    if not data.startswith(header):
        raise fail(f"{path}: FileHeader is not the donor's")
    body = zlib.decompress(data[len(header) :], -zlib.MAX_WBITS)

    donor_body = donor.decompressed_body
    off = donor.terrain_block_offset
    donor_end = off + TERRAIN_STRUCT_SIZE * donor.map_manager.map_width * donor.map_manager.map_height
    if _SIZE_STRUCT.unpack_from(body, off - _SIZE_STRUCT.size) != (width, height):
        raise fail(f"{path}: size pair at {off - 8} is {_SIZE_STRUCT.unpack_from(body, off - 8)}")
    end = off + TERRAIN_STRUCT_SIZE * width * height
    if len(body) - len(donor_body) != end - donor_end:
        raise fail(f"{path}: terrain block is not {TERRAIN_STRUCT_SIZE}*{width}*{height} bytes")
    if body[: off - 8] != donor_body[: off - 8] or body[end:] != donor_body[donor_end:]:
        raise fail(f"{path}: bytes outside the size pair and terrain block changed")

    unmarked = blank_body(donor, width, height)
    if unmarked[off:end] != BLANK_TERRAIN_STRUCT * (width * height):
        raise fail(f"{path}: the pre-marker body is not uniformly blank")
    differing = {
        i for i in range(width * height)
        if body[off + 7 * i : off + 7 * i + 7] != unmarked[off + 7 * i : off + 7 * i + 7]
    }
    if differing != set(markers):
        raise fail(f"{path}: {len(differing ^ set(markers))} tiles differ from the marker set")
    for i, terrain_id in markers.items():
        o = off + TERRAIN_STRUCT_SIZE * i
        if body[o] != terrain_id or body[o + 1 : o + 7] != BLANK_TERRAIN_STRUCT[1:]:
            raise fail(f"{path}: tile {i} is not terrain {terrain_id} with blank bytes 1..6")


def expectation(width: int, height: int) -> str:
    """What the user should see in the real editor. Tile (0, 0) is the west
    tip; x grows along the upper-left (NW) edge, y along the lower-left (SW)
    edge (docs/INGAME_EDITOR_REFERENCE.md, 'Coordinate axes')."""
    tall = height > width
    short, long_ = (width, height) if tall else (height, width)
    far_lo = long_ - BLOCK_FAR_INSET - BLOCK_SPAN
    far = f"{far_lo}..{far_lo + BLOCK_SPAN - 1}"
    near = f"{BLOCK_INSET}..{BLOCK_INSET + BLOCK_SPAN - 1}"
    lines = [
        (
            f"  Size: {width} x {height} tiles. The NW and SE edges are {width} tiles long, "
            f"the SW and NE edges {height}, so the map and minimap are a rectangle, not a square."
        ),
    ]
    if tall:
        lines += [
            (
                f"  Deep water stripe: the row y={STRIPE_INSET}, every x 0..{width - 1}. It runs parallel to "
                f"the NW edge, {STRIPE_INSET} tiles in from it, the full length of that edge (west tip to north tip)."
            ),
            (
                f"  Snow block (11x11): x={near}, y={far}. Near the SOUTH tip, {BLOCK_INSET} tiles in from the "
                f"SW edge and {BLOCK_FAR_INSET} tiles in from the SE edge."
            ),
        ]
    else:
        lines += [
            (
                f"  Deep water stripe: the column x={STRIPE_INSET}, every y 0..{height - 1}. It runs parallel to "
                f"the SW edge, {STRIPE_INSET} tiles in from it, the full length of that edge (west tip to south tip)."
            ),
            (
                f"  Snow block (11x11): x={far}, y={near}. Near the NORTH tip, {BLOCK_INSET} tiles in from the "
                f"NW edge and {BLOCK_FAR_INSET} tiles in from the NE edge."
            ),
        ]
    lines.append(
        "  Wrong if: the stripe is shorter than that edge or runs parallel to the other edge, or the "
        f"block is missing or sits anywhere else (its long-axis range {far} lies past the short axis "
        f"{short}). Either means the game reads the terrain array transposed from DEscape's assumption."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into")
    parser.add_argument(
        "--sizes",
        type=parse_size,
        nargs="+",
        default=[(n, n) for n in STANDARD_MAP_SIZES],
        help=f"Sizes to generate, N or WxH (default: {STANDARD_MAP_SIZES})",
    )
    args = parser.parse_args()

    if is_under_compatdata(args.out):
        raise WriteBlockedError(f"Refusing to write under a Proton {FORBIDDEN_WRITE_MARKER}/ folder: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    donor = None
    for width, height in args.sizes:
        path = args.out / f"blank_{width}x{height}.aoe2scenario"
        if is_under_compatdata(path):  # a per-file symlink inside --out
            raise WriteBlockedError(f"Refusing to write under a Proton {FORBIDDEN_WRITE_MARKER}/ folder: {path}")
        if width == height:
            data = blank_scenario_bytes(width)
            path.write_bytes(data)
            _verify(path, width)
            print(f"{path}: {len(data)} bytes, verified {width}x{height} blank")
            continue
        if donor is None:
            donor = load_map_and_units(BLANK_TEMPLATE_PATH)
        markers = marker_tiles(width, height)
        body = apply_markers(blank_body(donor, width, height), donor.terrain_block_offset, markers)
        data = donor.header_bytes + _compress_bytes(body)
        path.write_bytes(data)
        _verify_raw(path, donor, width, height, markers)
        print(f"{path}: {len(data)} bytes, verified {width}x{height} on raw bytes, {len(markers)} marker tiles")
        print(expectation(width, height))


if __name__ == "__main__":
    main()
