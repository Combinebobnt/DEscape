#!/usr/bin/env python3
"""Writes real .aoe2scenario files for the in-game verification gate that
descape/scenario_new.py's splice needs before it's trusted at sizes the real
game never produced for us.

Deliberately does not go through descape.scenario_write.write_scenario():
that function only ever patches an existing terrain array in place at its
original length (see its own docstring) and has no way to write a resized
one -- scenario_new.blank_scenario_bytes() is the whole point of this tool,
not something write_scenario() could do instead.

For each size: generate, write, then reload through the real
load_map_and_units() -- not just re-decompress the bytes in memory -- and
assert the same invariants tests/test_scenario_new.py checks, the same
reload-and-verify discipline tools/strip_units.py uses ("reload through the
real loader, not just re-decompress in place") rather than trusting
in-place state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape.scenario_io import load_map_and_units
from descape.scenario_new import STANDARD_MAP_SIZES, blank_scenario_bytes


class GenerationVerificationError(Exception):
    """Raised instead of leaving a file on disk whose reload-through-the-
    real-loader check didn't match what was generated."""


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory to write into")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(STANDARD_MAP_SIZES),
        help=f"Tile sizes to generate (default: {STANDARD_MAP_SIZES})",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for tiles in args.sizes:
        path = args.out / f"blank_{tiles}x{tiles}.aoe2scenario"
        data = blank_scenario_bytes(tiles)
        path.write_bytes(data)
        _verify(path, tiles)
        print(f"{path}: {len(data)} bytes, verified {tiles}x{tiles} blank")


if __name__ == "__main__":
    main()
