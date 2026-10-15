#!/usr/bin/env python3
"""Generates descape/templates/blank_120x120.aoe2scenario, the donor every
File > New Map size is spliced from (descape/scenario_new.py).

Provenance. The source is tests/fixtures/blank_120x120_v158.aoe2scenario: a
real AoE2:DE 1.58 export of the "Blank" map style with its scatter doodads
removed by tools/strip_units.py, and the template itself until 1.59. This
tool sets the three fields that carry a scenario's version and nothing else:

- FileHeader.version (ASCII, header bytes 0..4).
- DataHeader.version (f32, decompressed body bytes 4..8).
- The Triggers section's trigger_version (f64, the section's first field, at
  the end of the Units section). It moves with the scenario version in every
  real file measured: 1.58 files carry 4.9 and 1.59 files carry 5.0. A file
  patched on the first two alone reads as 1.59 everywhere in DEscape and is
  still a pair the game never writes.

That is the whole difference for a file with no units and no conditions: the
library's 1.58 and 1.59 structures differ only in UnitStruct.capture_flag and
ConditionStruct.allow_in_fog, and this file has no bytes in either.

Measured against tests/fixtures/real_blank_120x120_v159.aoe2scenario, a blank
120x120 map saved by the 1.59 game editor: the output equals it byte for byte
except three fields a template inherits anyway. Those are
FileHeader.timestamp_of_last_save (header bytes 12..16), DataHeader.filename
(a length-prefixed str16) and Player 1's editor_camera_x/y in
Units.player_data_3 (the save's camera was moved off the centre).
tests/test_blank_template.py repeats that comparison on every default-tier
run, and pins the template's version to the newest one DEscape has a trigger
vocabulary for.

The next format bump, when that test fails after a library pin moves:

1. Open a File > New Map 120x120 output in the new game's editor and Save As
   without touching it.
2. Run the oracle comparison in tests/test_blank_template.py against that
   save.
3. If only the version carriers and the three inherited fields differ, set
   SCENARIO_VERSION and TRIGGER_VERSION below to the save's, replace the
   oracle fixture with the save, and rerun this tool.
4. Anything else that differs is a format finding. Understand it before this
   file changes.

Deterministic: two runs produce identical bytes. Does not go through
descape.scenario_write.write_scenario(), which refuses the template
directory by design.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixture_bytes import V158_DONOR_PATH, GenerationVerificationError

from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    BLANK_TEMPLATE_TILES,
    is_workshop_subscribed,
    load_map_and_units,
    load_map_and_units_from_bytes,
    parse_triggers,
)
from descape.scenario_write import WriteBlockedError, _compress_bytes

SOURCE_VERSION = "1.58"
SOURCE_TRIGGER_VERSION = 4.9
SCENARIO_VERSION = "1.59"
TRIGGER_VERSION = 5.0

_FILE_HEADER_VERSION = slice(0, 4)  # ASCII, e.g. b"1.58"
_DATA_HEADER_VERSION = struct.Struct("<f")  # at decompressed body offset 4
_DATA_HEADER_VERSION_OFFSET = 4
_TRIGGER_VERSION = struct.Struct("<d")  # at LoadedScenario.units_section_end


def _unit_count(loaded) -> int:
    return sum(len(units) for units in loaded.unit_manager.units)


def build_template_bytes(source_path: Path = V158_DONOR_PATH) -> bytes:
    """The source with its three version carriers set. Refuses unless each
    reads the source value first and the file has no units and no triggers,
    so a changed source fails here, not silently."""
    loaded = load_map_and_units(source_path)
    header = bytearray(loaded.header_bytes)
    body = bytearray(loaded.decompressed_body)
    triggers_at = loaded.units_section_end

    if bytes(header[_FILE_HEADER_VERSION]) != SOURCE_VERSION.encode("ascii"):
        raise GenerationVerificationError(f"{source_path}: FileHeader.version is {bytes(header[_FILE_HEADER_VERSION])!r}")
    data_version = bytes(body[_DATA_HEADER_VERSION_OFFSET : _DATA_HEADER_VERSION_OFFSET + _DATA_HEADER_VERSION.size])
    if data_version != _DATA_HEADER_VERSION.pack(float(SOURCE_VERSION)):
        raise GenerationVerificationError(
            f"{source_path}: DataHeader.version is {_DATA_HEADER_VERSION.unpack(data_version)[0]}, expected {SOURCE_VERSION}"
        )
    if bytes(body[triggers_at : triggers_at + _TRIGGER_VERSION.size]) != _TRIGGER_VERSION.pack(SOURCE_TRIGGER_VERSION):
        raise GenerationVerificationError(
            f"{source_path}: trigger version is {loaded.trigger_version}, expected {SOURCE_TRIGGER_VERSION}"
        )
    if _unit_count(loaded):
        raise GenerationVerificationError(f"{source_path}: source already has {_unit_count(loaded)} units")
    manager = parse_triggers(loaded)
    if manager is None or manager.triggers:
        raise GenerationVerificationError(f"{source_path}: source's Triggers section is not empty and parseable")

    header[_FILE_HEADER_VERSION] = SCENARIO_VERSION.encode("ascii")
    _DATA_HEADER_VERSION.pack_into(body, _DATA_HEADER_VERSION_OFFSET, float(SCENARIO_VERSION))
    _TRIGGER_VERSION.pack_into(body, triggers_at, TRIGGER_VERSION)
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return bytes(header) + deflate.compress(bytes(body)) + deflate.flush()


def verify_template(data: bytes) -> None:
    """Reloads the bytes through the real loader on both unit walks, before
    anything is written."""
    name = BLANK_TEMPLATE_PATH.name
    for fast_units in (True, False):
        walk = "fast" if fast_units else "library"
        loaded = load_map_and_units_from_bytes(data, name, fast_units=fast_units)
        if loaded.scenario_version != SCENARIO_VERSION or loaded.trigger_version != TRIGGER_VERSION:
            raise GenerationVerificationError(
                f"{name} ({walk} walk): reloaded as {loaded.scenario_version} with trigger version {loaded.trigger_version}"
            )
        size = (loaded.map_manager.map_width, loaded.map_manager.map_height)
        if size != (BLANK_TEMPLATE_TILES, BLANK_TEMPLATE_TILES):
            raise GenerationVerificationError(f"{name} ({walk} walk): map size is {size}")
        if not (loaded.terrain_write_supported and loaded.units_write_supported and loaded.messages_write_supported):
            raise GenerationVerificationError(f"{name} ({walk} walk): a load-time write gate failed after reload")
        manager = parse_triggers(loaded)
        if manager is None or manager.triggers or not loaded.trigger_write_supported:
            raise GenerationVerificationError(f"{name} ({walk} walk): Triggers did not reload empty, readable and writable")
        if _unit_count(loaded):
            raise GenerationVerificationError(f"{name} ({walk} walk): {_unit_count(loaded)} units after reload")
        if loaded.header_bytes + _compress_bytes(loaded.decompressed_body) != data:
            raise GenerationVerificationError(f"{name} ({walk} walk): a zero-edit re-serialize is not byte-identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BLANK_TEMPLATE_PATH, help="Destination file")
    args = parser.parse_args()

    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    data = build_template_bytes()
    verify_template(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The build bundles the template directory, so the temp file sits one level up from it.
    tmp = args.out.parent.parent / f"{args.out.name}.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, args.out)
    print(f"{args.out}: {len(data)} bytes, scenario version {SCENARIO_VERSION}, trigger version {TRIGGER_VERSION}")


if __name__ == "__main__":
    main()
