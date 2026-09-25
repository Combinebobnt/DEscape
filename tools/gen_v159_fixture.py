#!/usr/bin/env python3
"""Generates tests/fixtures/v159_units_triggers.aoe2scenario, the default
tier's only scenario version 1.59 file.

Why it has to be generated: CI has no 1.59 file to read. It is derived from
the shipped descape/templates/blank_120x120.aoe2scenario donor (a 1.58 file,
so this inherits whatever licensing answer shipping the donor already made)
the same way tools/gen_units_fixture.py and tools/gen_trigger_fixture.py are:

1. Patch the donor's two version carriers to 1.59: FileHeader.version (ASCII,
   header bytes 0..4) and DataHeader.version (f32, body bytes 4..8). Measured
   by diffing every FileHeader/DataHeader value of a real game-saved 1.59
   file against a 1.58 one; nothing else in those sections carries it. The
   donor has no units or conditions, so its body is already valid 1.59.
2. Load that through the real loader, which reads it with the vendored
   descape/versions/DE/v1.59/structure.json.
3. Add units and triggers through AoE2ScenarioParser's own managers, commit,
   then set the two 1.59-only fields (`capture_flag`, `allow_in_fog`)
   directly on the committed slots: 0.8.3's Unit/Condition can't take them.
4. Serialize through the library's own retrievers with empty strings written
   the game's way (tools/_fixture_bytes.py), never through descape code.

What it holds, and why:

- Units on GAIA, Player 1 and Player 2 with capture flags -1, 0, 1, 2 and 3,
  ordered so every unit's list neighbours carry a different flag. That is
  what makes a slot-shift regression (delete-then-move, reassign-then-move,
  delete-then-add) visible: a flag left in its slot lands on a neighbour.
- Trigger 0: a condition 27 (Object Visible Multiplayer) with
  allow_in_fog=1 between two timer conditions, so removing the first timer
  shifts it down a slot.
- Trigger 1: a condition 27 with allow_in_fog left at -1.

Deterministic: two runs produce identical bytes (tests/test_v159_fixture.py).

Maintainer-side cross-check, not part of any test: the fixture parses with
AoE2ScenarioParser's own upstream 1.59 support (branch feat/v1-59-support,
commit faadf3fd, in a throwaway `git worktree add --detach`, then
PYTHONPATH=<worktree>), reading back every capture_flag and allow_in_fog
below, and its FileHeader/DataHeader field layout matches a real game-saved
1.59 file's.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fixture_bytes import GenerationVerificationError, _game_style_bytes

from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    LoadedScenario,
    load_map_and_units,
    load_map_and_units_from_bytes,
    parse_triggers,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "v159_units_triggers.aoe2scenario"

SCENARIO_VERSION = "1.59"
_FILE_HEADER_VERSION = slice(0, 4)  # ASCII, e.g. b"1.58"
_DATA_HEADER_VERSION = struct.Struct("<f")  # at decompressed body offset 4
_DATA_HEADER_VERSION_OFFSET = 4

_ARCHER = 4
_VILLAGER = 83
_SHEEP = 594
_TREE_OAK = 349

# (player, reference_id, unit_const, x, y, capture_flag), in list order.
# Neighbours within each player's list always differ in flag.
UNITS = (
    (0, 100, _SHEEP, 5.5, 5.5, 1),
    (0, 101, _TREE_OAK, 6.5, 5.5, -1),
    (0, 102, _SHEEP, 7.5, 5.5, 3),
    (0, 103, _SHEEP, 8.5, 5.5, 0),
    (1, 200, _ARCHER, 20.5, 10.5, 2),
    (1, 201, _VILLAGER, 21.5, 10.5, -1),
    (1, 202, _ARCHER, 22.5, 10.5, 1),
    (1, 203, _VILLAGER, 23.5, 10.5, 0),
    (2, 300, _ARCHER, 40.5, 30.5, -1),
    (2, 301, _VILLAGER, 41.5, 30.5, 3),
    (2, 302, _ARCHER, 42.5, 30.5, 2),
)
CAPTURE_FLAGS = {ref: flag for _p, ref, _c, _x, _y, flag in UNITS}
UNIT_COUNTS = {player: sum(1 for p, *_ in UNITS if p == player) for player in {p for p, *_ in UNITS}}
NEXT_UNIT_ID = 400

TRIGGER_COUNT = 2
FOG_TRIGGER = 0
FOG_CONDITION = 1  # between two timers
VISIBLE_UNIT = 200
# (trigger, condition) -> allow_in_fog, for every condition in the file.
ALLOW_IN_FOG = {(0, 0): -1, (0, 1): 1, (0, 2): -1, (1, 0): -1}
CONDITION_TYPES = {(0, 0): 10, (0, 1): 27, (0, 2): 10, (1, 0): 27}


def _patched_donor(donor_path: Path) -> bytes:
    """The donor with both version carriers set to 1.59. Refuses if either
    doesn't read 1.58 first, so a changed donor fails here, not silently."""
    loaded = load_map_and_units(donor_path)
    header = bytearray(loaded.header_bytes)
    if bytes(header[_FILE_HEADER_VERSION]) != b"1.58":
        raise GenerationVerificationError(f"{donor_path}: FileHeader.version is {bytes(header[:4])!r}")
    (version,) = _DATA_HEADER_VERSION.unpack_from(loaded.decompressed_body, _DATA_HEADER_VERSION_OFFSET)
    if abs(version - 1.58) > 1e-6:
        raise GenerationVerificationError(f"{donor_path}: DataHeader.version is {version}")
    header[_FILE_HEADER_VERSION] = SCENARIO_VERSION.encode("ascii")
    body = bytearray(loaded.decompressed_body)
    _DATA_HEADER_VERSION.pack_into(body, _DATA_HEADER_VERSION_OFFSET, float(SCENARIO_VERSION))
    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return bytes(header) + deflate.compress(bytes(body)) + deflate.flush()


def _build_units(loaded: LoadedScenario) -> None:
    manager = loaded.unit_manager
    for player in range(9):
        if manager.units[player]:
            raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor already has units on player {player}")
    for player, ref, const, x, y, _flag in UNITS:
        manager.add_unit(player=player, unit_const=const, x=x, y=y, reference_id=ref)
    manager.commit()
    for player_units in loaded._scenario.sections["Units"].retriever_map["players_units"].data:
        for entry in player_units.retriever_map["units"].data:
            ref = entry.retriever_map["reference_id"].data
            entry.retriever_map["capture_flag"].set_data(CAPTURE_FLAGS[ref], affect_dirty=False)


def _build_triggers(loaded: LoadedScenario) -> None:
    manager = parse_triggers(loaded)
    if manager is None or manager.triggers:
        raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor's Triggers section is not empty and parseable")
    fog = manager.add_trigger("Fixture: visible in fog")
    fog.new_condition.timer(timer=1)
    fog.new_condition.object_visible_multiplayer(unit_object=VISIBLE_UNIT, source_player=2)
    fog.new_condition.timer(timer=2)
    fog.new_effect.display_instructions(source_player=1, message="Seen.", display_time=5, instruction_panel_position=0)
    plain = manager.add_trigger("Fixture: visible")
    plain.new_condition.object_visible_multiplayer(unit_object=VISIBLE_UNIT, source_player=2)
    manager.commit()
    for t, entry in enumerate(loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data):
        for c, condition in enumerate(entry.retriever_map["condition_data"].data):
            condition.retriever_map["allow_in_fog"].set_data(ALLOW_IN_FOG[(t, c)], affect_dirty=False)


def _patch_counter(buffer: bytearray, end_offset: int, value: int) -> None:
    """The u32 that ends at end_offset (both trigger counters end their
    section, as in tools/gen_trigger_fixture.py)."""
    struct.pack_into("<I", buffer, end_offset - 4, value)


def build_fixture_bytes(donor_path: Path = BLANK_TEMPLATE_PATH) -> bytes:
    loaded = load_map_and_units_from_bytes(_patched_donor(donor_path), FIXTURE_PATH.name)
    if loaded.scenario_version != SCENARIO_VERSION or loaded.structure_source != "repo":
        raise GenerationVerificationError(
            f"patched donor loaded as {loaded.scenario_version}/{loaded.structure_source}, expected 1.59/repo"
        )
    _build_units(loaded)
    _build_triggers(loaded)

    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    units_bytes = b"".join(_game_style_bytes(player_units) for player_units in players_units)
    section_bytes = _game_style_bytes(loaded._scenario.sections["Triggers"])
    donor = loaded.decompressed_body
    body = bytearray(
        donor[: loaded.units_block_offset]
        + units_bytes
        + donor[loaded.players_units_end : loaded.units_section_end]
        + section_bytes
        + donor[loaded.triggers_section_end :]
    )
    struct.pack_into("<I", body, 0, NEXT_UNIT_ID)
    _patch_counter(body, loaded.options_section_end, TRIGGER_COUNT)
    header = bytearray(loaded.header_bytes)
    _patch_counter(header, len(header), TRIGGER_COUNT)

    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return bytes(header) + deflate.compress(bytes(body)) + deflate.flush()


def read_capture_flags(loaded: LoadedScenario) -> dict[int, int]:
    """reference_id -> capture_flag, straight off the parsed slots."""
    return {
        entry.retriever_map["reference_id"].data: entry.retriever_map["capture_flag"].data
        for player_units in loaded._scenario.sections["Units"].retriever_map["players_units"].data
        for entry in player_units.retriever_map["units"].data
    }


def read_allow_in_fog(loaded: LoadedScenario) -> dict[tuple[int, int], int]:
    """(trigger, condition) -> allow_in_fog, straight off the parsed slots.
    Requires parse_triggers() to have run."""
    return {
        (t, c): condition.retriever_map["allow_in_fog"].data
        for t, entry in enumerate(loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data)
        for c, condition in enumerate(entry.retriever_map["condition_data"].data)
    }


def verify_fixture(path: Path) -> None:
    """Reloads through the real loader, like the other generators."""
    reloaded = load_map_and_units(path)
    if reloaded.scenario_version != SCENARIO_VERSION or reloaded.structure_source != "repo":
        raise GenerationVerificationError(f"{path}: reloaded as {reloaded.scenario_version}/{reloaded.structure_source}")
    if not (reloaded.terrain_write_supported and reloaded.units_write_supported and reloaded.messages_write_supported):
        raise GenerationVerificationError(f"{path}: a load-time write gate failed after reload")
    for player, expected in UNIT_COUNTS.items():
        if len(reloaded.unit_manager.units[player]) != expected:
            raise GenerationVerificationError(f"{path}: player {player} unit count is wrong after reload")
    if read_capture_flags(reloaded) != CAPTURE_FLAGS:
        raise GenerationVerificationError(f"{path}: capture flags read back as {read_capture_flags(reloaded)}")
    manager = parse_triggers(reloaded)
    if manager is None or len(manager.triggers) != TRIGGER_COUNT or not reloaded.trigger_write_supported:
        raise GenerationVerificationError(f"{path}: Triggers did not reload readable and writable")
    types = {(t, c): cond.condition_type for t, trig in enumerate(manager.triggers) for c, cond in enumerate(trig.conditions)}
    if types != CONDITION_TYPES or read_allow_in_fog(reloaded) != ALLOW_IN_FOG:
        raise GenerationVerificationError(f"{path}: conditions read back as {types}, {read_allow_in_fog(reloaded)}")
    if struct.unpack_from("<I", reloaded.decompressed_body, 0)[0] != NEXT_UNIT_ID:
        raise GenerationVerificationError(f"{path}: next_unit_id_to_place did not survive the reload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="Destination file")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = build_fixture_bytes()
    args.out.write_bytes(data)
    verify_fixture(args.out)
    print(f"{args.out}: {len(data)} bytes, verified {len(UNITS)} units and {TRIGGER_COUNT} triggers")


if __name__ == "__main__":
    main()
