#!/usr/bin/env python3
"""Generates tests/fixtures/units_120x120.aoe2scenario, the default tier's
only fixture with a non-trivial Units section.

Why it has to be generated rather than copied: every file this repo ships is
either unit-free or (descape/templates/blank_120x120.aoe2scenario itself) has
zero units, and a zero-unit Units section trivially passes every write-path
byte-identity check. Deriving one from that same donor, which is already
shipped, inherits whatever licensing answer shipping the donor already made --
the same reasoning tools/gen_trigger_fixture.py records for the trigger
fixture.

Deliberately serializes the Units section through AoE2ScenarioParser's own
manager/retrievers, never through descape.unit_model. The tests this fixture
feeds compare a *parse* of it against something, so who produced it does not
matter to them, but generating it with the code under test would make the one
test that regenerates a section circular.

The trap this generator exists to avoid: a fixture built by adding units
*through the library* carries the library's length-1 caption trail on every
unit, whether or not that unit's own caption is empty. That is the drifted
form already, so every write-path test would pass -- and disabling the
normalizer would pass too -- without measuring anything. See
tools/_fixture_bytes.py's _game_style_bytes() for the fix, shared with
tools/gen_trigger_fixture.py.

Unlike tests/fixtures/real_blank_*.aoe2scenario (byte oracles for a generator,
never to be regenerated), this file is a test *input*. Regenerating it is fine
and expected. It is deterministic: two runs produce identical bytes, which
test_units_fixture.py checks.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixture_bytes import GenerationVerificationError, _game_style_bytes

from descape.scenario_io import BLANK_TEMPLATE_PATH, LoadedScenario, load_map_and_units

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "units_120x120.aoe2scenario"

# unit_const values, from descape/tree_unit_ids.json (FOAK, FPIN) and the
# library's own datasets (House, Archer, Villager). 117 is one of
# descape/unit_sprites.py's hand-verified _ROTATION_VARIANT_CONSTS (a Wall):
# its rotation is a shape-variant selector, not an angle, and real files
# encode that selector two ways -- see _WALL_ROTATION_RADIANS below.
_UNIT_CONST_TREE_OAK = 349
_UNIT_CONST_TREE_PINE = 350
_UNIT_CONST_WALL = 117
_UNIT_CONST_HOUSE = 70
_UNIT_CONST_ARCHER = 4
_UNIT_CONST_VILLAGER = 83

_PLAYER_GAIA = 0
_PLAYER_ONE = 1
_PLAYER_TWO = 2

# GAIA rotation is a doodad-variant index for most GAIA objects (AGENTS.md's
# hard rule): plain integers well outside [0, 2*pi), never angles.
_TREE_OAK_ROTATION = 7.0
_TREE_PINE_ROTATION = 41.0
# The second encoding AGENTS.md names for the Wall family: the same variant
# index expressed as k*2pi/5 radians instead of a literal 0..4. k=2.
_WALL_ROTATION_RADIANS = 2 * (2 * math.pi / 5)

# reference_ids are assigned explicitly, not through the library's own
# generator (plan fact 11: UnitManager.next_unit_id is a side-effecting
# generator that a full commit() does not reliably synchronize with
# DataHeader.next_unit_id_to_place). A gap at 202 and NEXT_UNIT_ID above the
# highest assigned id here both mirror what every real corpus file looks
# like (plan finding 8).
_REF_TREE_OAK = 100
_REF_TREE_PINE = 101
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
# 202 deliberately unused -- the gap.
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300
_REF_VILLAGER_P2 = 301
NEXT_UNIT_ID = 400

NON_EMPTY_CAPTION = "Fixture caption"

# Per-player unit counts, keyed by player index (0 = GAIA). Used by
# verify_fixture() and by test_units_fixture.py; kept as one table so the two
# can't drift out of sync with what _build_units() actually places.
UNIT_COUNTS = {_PLAYER_GAIA: 3, _PLAYER_ONE: 3, _PLAYER_TWO: 2}
UNIT_COUNT = sum(UNIT_COUNTS.values())


def _build_units(loaded: LoadedScenario) -> None:
    """Populates the donor's (empty) UnitManager in place.

    Table of what's here and why:

    - 2 GAIA trees with rotation as a variant index -- pins verbatim
      rotation pass-through; a normalizing write path fails here.
    - 1 GAIA Wall with rotation as k*2pi/5 radians -- the second encoding
      AGENTS.md names; both must survive.
    - Player 1: 1 building + 2 units at non-integer (.5) coords -- a
      reassign source, and a real on-map building for batch_api.is_building()
      to exercise against a default-tier fixture, not only the corpus tier.
    - Player 2: 2 units -- a reassign destination that is not GAIA.
    - The Player 1 villager is garrisoned in the Player 1 house -- exercises
      the remove-path garrisoned_in_id reference check.
    - The Player 2 archer carries a non-empty caption -- the only coverage
      anywhere for the caption normalizer's unmeasured non-empty branch.
    """
    manager = loaded.unit_manager
    for player in range(9):
        if manager.units[player]:
            raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor already has units on player {player}")

    manager.add_unit(
        player=_PLAYER_GAIA, unit_const=_UNIT_CONST_TREE_OAK, x=5.5, y=5.5, rotation=_TREE_OAK_ROTATION,
        reference_id=_REF_TREE_OAK,
    )
    manager.add_unit(
        player=_PLAYER_GAIA, unit_const=_UNIT_CONST_TREE_PINE, x=6.5, y=5.5, rotation=_TREE_PINE_ROTATION,
        reference_id=_REF_TREE_PINE,
    )
    manager.add_unit(
        player=_PLAYER_GAIA, unit_const=_UNIT_CONST_WALL, x=7.5, y=5.5, rotation=_WALL_ROTATION_RADIANS,
        reference_id=_REF_WALL,
    )

    manager.add_unit(
        player=_PLAYER_ONE, unit_const=_UNIT_CONST_HOUSE, x=10.5, y=10.5, reference_id=_REF_HOUSE,
    )
    manager.add_unit(
        player=_PLAYER_ONE, unit_const=_UNIT_CONST_ARCHER, x=11.5, y=10.5, reference_id=_REF_ARCHER_P1,
    )
    manager.add_unit(
        player=_PLAYER_ONE, unit_const=_UNIT_CONST_VILLAGER, x=12.5, y=10.5, reference_id=_REF_VILLAGER_P1,
        garrisoned_in_id=_REF_HOUSE,
    )

    manager.add_unit(
        player=_PLAYER_TWO, unit_const=_UNIT_CONST_ARCHER, x=20.5, y=20.5, reference_id=_REF_ARCHER_P2,
        caption_string=NON_EMPTY_CAPTION,
    )
    manager.add_unit(
        player=_PLAYER_TWO, unit_const=_UNIT_CONST_VILLAGER, x=21.5, y=20.5, reference_id=_REF_VILLAGER_P2,
    )

    manager.commit()


def build_fixture_bytes(donor_path: Path = BLANK_TEMPLATE_PATH) -> bytes:
    """The whole fixture as a full .aoe2scenario byte string.

    Splices freshly-serialized players_units bytes into the donor's otherwise
    verbatim decompressed body, then patches DataHeader.next_unit_id_to_place
    (the leading u32 at decompressed_body[0:4] -- plan finding 8), which a
    library commit() does not reliably set to a value above manually-assigned
    reference_ids (plan fact 11). The same units_block_offset/units_section_end
    boundaries descape/scenario_write.py's units branch will use, hand-rolled
    here so the generator stays independent of the code its output tests.
    """
    loaded = load_map_and_units(donor_path)
    _build_units(loaded)

    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    section_bytes = b"".join(_game_style_bytes(player_units) for player_units in players_units)

    body = bytearray(
        loaded.decompressed_body[: loaded.units_block_offset]
        + section_bytes
        + loaded.decompressed_body[loaded.units_section_end :]
    )
    struct.pack_into("<I", body, 0, NEXT_UNIT_ID)

    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return bytes(loaded.header_bytes) + deflate.compress(bytes(body)) + deflate.flush()


def verify_fixture(path: Path) -> None:
    """Reloads through the real loader, not just re-decompressing in place --
    the same discipline tools/gen_trigger_fixture.py and tools/strip_units.py
    use."""
    reloaded = load_map_and_units(path)
    if not reloaded.units_write_supported:
        raise GenerationVerificationError(f"{path}: failed the units alignment gate after reload")
    if not reloaded.terrain_write_supported:
        raise GenerationVerificationError(f"{path}: terrain block failed reload verification")
    if reloaded.number_of_unit_sections != 9:
        raise GenerationVerificationError(
            f"{path}: number_of_unit_sections is {reloaded.number_of_unit_sections}, expected 9"
        )

    manager = reloaded.unit_manager
    for player, expected in UNIT_COUNTS.items():
        actual = len(manager.units[player])
        if actual != expected:
            raise GenerationVerificationError(f"{path}: player {player} has {actual} units, expected {expected}")

    reference_ids = [unit.reference_id for units in manager.units for unit in units]
    if len(reference_ids) != len(set(reference_ids)):
        raise GenerationVerificationError(f"{path}: duplicate reference_id among {reference_ids}")
    if 202 in reference_ids:
        raise GenerationVerificationError(f"{path}: the deliberate reference_id gap at 202 was filled")

    next_unit_id = struct.unpack_from("<I", reloaded.decompressed_body, 0)[0]
    if next_unit_id <= max(reference_ids):
        raise GenerationVerificationError(
            f"{path}: next_unit_id_to_place is {next_unit_id}, not above the highest reference_id {max(reference_ids)}"
        )

    villager = next(u for u in manager.units[_PLAYER_ONE] if u.reference_id == _REF_VILLAGER_P1)
    if villager.garrisoned_in_id != _REF_HOUSE:
        raise GenerationVerificationError(f"{path}: villager's garrisoned_in_id did not survive the reload")

    captioned = next(u for u in manager.units[_PLAYER_TWO] if u.reference_id == _REF_ARCHER_P2)
    if captioned.caption_string != NON_EMPTY_CAPTION:
        raise GenerationVerificationError(
            f"{path}: non-empty caption read back as {captioned.caption_string!r}, expected {NON_EMPTY_CAPTION!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="Destination file")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = build_fixture_bytes()
    args.out.write_bytes(data)
    verify_fixture(args.out)
    print(f"{args.out}: {len(data)} bytes, verified {UNIT_COUNT} units")


if __name__ == "__main__":
    main()
