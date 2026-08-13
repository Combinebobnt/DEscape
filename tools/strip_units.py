#!/usr/bin/env python3
"""Strips every unit from an .aoe2scenario file's Units section via a direct
byte splice, mirroring descape/scenario_write.py's terrain patch rather than
going through AoE2ScenarioParser's own writer -- see that module's docstring
for why: re-serializing through the library isn't byte-stable, confirmed
again here (a zero-edit round trip of tests/fixtures/golden_blank_120x120.
aoe2scenario through write_to_file() shrinks it from 746 to 736 bytes).

Written to prepare descape/templates/blank_{120,240,480}x*.aoe2scenario for
File > New: those exports carry GRASS_GREEN ground-scatter doodads (~6% tile
density) rather than being genuinely blank, which breaks the "unit-free"
invariant tests/test_new_map.py and tests/test_strip_units.py assert on the
shipped templates.

The splice: each of the Units section's `number_of_unit_sections`
PlayerUnitsStructs is `unit_count u32` followed by that many UnitStructs.
Replacing the whole players_units array (scenario_io.load_map_and_units's
units_block_offset .. units_section_end) with `number_of_unit_sections` zero
u32s removes every unit and leaves every other section -- including the never
-parsed trigger_tail -- byte-for-byte untouched. Also zeroes DataHeader's
first field, next_unit_id_to_place (a u32 at decompressed_body[0:4] that
tracks the highest assigned unit id in every real file measured), so a
stripped file reports the same fresh-scenario value as a genuinely unit-free
one.

Deliberately does not go through scenario_write.write_scenario(): that
function's WriteBlockedError refuses any path inside TEMPLATE_DIR by design
(descape/scenario_write.py:99-102), and this tool's whole job is preparing
shipped templates -- weakening that guard to let this through would defeat
its purpose for every other caller.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape.scenario_io import LoadedScenario, load_map_and_units
from descape.scenario_write import _compress_bytes


class StripVerificationError(Exception):
    """Raised instead of writing, when a file's units block failed load-time
    verification or the strip result doesn't match its own preconditions."""


def strip_units(scenario: LoadedScenario) -> bytes:
    """Returns a new decompressed body with every unit removed and
    next_unit_id_to_place zeroed. Raises StripVerificationError instead of
    silently patching the wrong bytes if the file's units block wasn't
    verified at load time (see scenario_io._verify_units_block)."""
    if not scenario.units_write_supported:
        raise StripVerificationError(
            f"{scenario.path}: units block failed load-time verification -- "
            "refusing to patch bytes that may not be where they're expected."
        )

    body = bytearray(scenario.decompressed_body)
    struct.pack_into("<I", body, 0, 0)  # next_unit_id_to_place
    zero_counts = b"\0\0\0\0" * scenario.number_of_unit_sections
    new_body = bytes(body[: scenario.units_block_offset]) + zero_counts + bytes(body[scenario.units_section_end :])
    return new_body


def _self_test(golden_path: Path) -> None:
    """Stripping an already-unit-free file must be a byte-exact no-op --
    catches a *length* error in the splice, though not an offset error (see
    tests/test_strip_units.py for the check that also exercises a real
    strip)."""
    golden = load_map_and_units(golden_path)
    result = strip_units(golden)
    if result != golden.decompressed_body:
        raise StripVerificationError(
            f"Self-test failed: stripping the already unit-free golden file "
            f"{golden_path} changed its body ({len(golden.decompressed_body)} "
            f"-> {len(result)} bytes). Refusing to strip anything until this "
            f"is fixed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="+", type=Path, help="Files to strip, in place")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests/fixtures/golden_blank_120x120.aoe2scenario",
        help="Known-good unit-free file the self-test strips as a no-op check before touching anything",
    )
    args = parser.parse_args()

    _self_test(args.golden)
    print(f"Self-test passed against {args.golden}")

    for path in args.scenarios:
        scenario = load_map_and_units(path)
        before = sum(len(units) for units in scenario.unit_manager.units)
        new_body = strip_units(scenario)
        compressed = _compress_bytes(new_body)
        path.write_bytes(scenario.header_bytes + compressed)

        # Reload through the real loader, not just re-decompress in place --
        # this exercises the exact same section walk a future load will use.
        reloaded = load_map_and_units(path)
        after = sum(len(units) for units in reloaded.unit_manager.units)
        print(f"{path}: {before} -> {after} units, {len(new_body)} bytes decompressed")
        if after != 0:
            raise StripVerificationError(f"{path}: still has {after} units after stripping")


if __name__ == "__main__":
    main()
