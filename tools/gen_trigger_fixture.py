#!/usr/bin/env python3
"""Generates tests/fixtures/triggers_120x120.aoe2scenario, the default tier's
only trigger-bearing scenario.

Why it has to be generated rather than copied: every file this repo ships has
zero triggers, and zero-trigger files are exactly the ones that trivially pass
a byte-identity check, so none of them can exercise the phase 4b write path.
Copying a real examples/ scenario in would raise a licensing question this
repo has not answered; deriving one from descape/templates/blank_120x120.
aoe2scenario, which is already shipped, inherits whatever answer shipping the
donor already made.

Deliberately serializes the Triggers section through AoE2ScenarioParser's own
retrievers, never through descape.trigger_model. The tests this fixture feeds
compare a *parse* of it against something, so who produced it does not matter
to them, but generating it with the code under test would make the one test
that regenerates a section circular. Keeping the generator on the library's own
serializer shuts that door by construction.

One deliberate departure from the library's output, without which the fixture
is useless: empty strings are written the way the game writes them (length 0)
rather than the way the library writes them (length 1 holding a NUL). See
tools/_fixture_bytes.py's _game_style_bytes() for the measurement and for what
silently stops being tested without it -- shared with tools/gen_units_fixture.py,
which needs the identical fix for unit captions.

Unlike tests/fixtures/real_blank_*.aoe2scenario (byte oracles for a generator,
never to be regenerated), this file is a test *input*. Regenerating it is fine
and expected. It is deterministic: two runs produce identical bytes, which
test_trigger_fixture.py checks.
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
    parse_triggers,
)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "triggers_120x120.aoe2scenario"

# What the fixture deliberately contains, and why each entry is here rather
# than being one more trigger for volume's sake:
#
# 0 "Fixture: setup"           str16 fields with real content, so the NUL-
#                              padding drift the plan's finding 2 describes
#                              has somewhere to show up.
# 1 "Fixture: armour split"    a MODIFY_ATTRIBUTE effect on the Armor
#                              attribute, whose quantity is bit-split across
#                              _quantity_int/_quantity_float/variable. Finding
#                              8's 2-byte int-to-f32 retype lives in exactly
#                              that slot; without this trigger the assertion
#                              that the retype disappears for untouched
#                              triggers could only run in the corpus tier.
# 2 "Fixture: references"      (de)activate-trigger effects pointing at other
#                              triggers by id. trigger_id is positional, so
#                              reorder/remove renumber it and remap these --
#                              the case that decides which blobs go stale.
# 3 "Fixture: variable"        a CHANGE_VARIABLE effect plus a named variable,
#                              so variable_data is not empty and the block
#                              after unknown_bytes actually has content.
TRIGGER_COUNT = 4
VARIABLE_NAME = "fixture_var"
VARIABLE_ID = 0

# AoE2 attribute ids, from the in-game editor's Object Attribute list.
_ATTRIBUTE_ARMOR = 8
_ATTRIBUTE_HIT_POINTS = 0
# Operation ids: 1 = Set, 2 = Add.
_OPERATION_SET = 1
_OPERATION_ADD = 2
# Unit id 4 is the Archer, an ordinary trainable unit -- any valid id works,
# this one is just stable across every DE version.
_UNIT_ARCHER = 4
_PLAYER_ONE = 1


def _build_triggers(loaded: LoadedScenario) -> None:
    """Populates the donor's (empty) TriggerManager in place."""
    manager = parse_triggers(loaded)
    if manager is None:
        raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor's Triggers section did not parse")
    if len(manager.triggers) != 0:
        raise GenerationVerificationError(
            f"{BLANK_TEMPLATE_PATH}: donor already has {len(manager.triggers)} triggers, expected 0"
        )

    setup = manager.add_trigger(
        "Fixture: setup",
        description="Fires once, five seconds in.",
        short_description="setup",
        display_on_screen=True,
        execute_on_load=True,
    )
    setup.new_condition.timer(timer=5)
    setup.new_effect.display_instructions(
        source_player=_PLAYER_ONE,
        message="Fixture scenario loaded.",
        display_time=10,
        instruction_panel_position=0,
    )

    armour = manager.add_trigger("Fixture: armour split", looping=True)
    armour.new_condition.timer(timer=1)
    # armour_attack_quantity/armour_attack_class, not quantity: on the Armor
    # attribute the library packs both into one slot, and writing quantity
    # directly bypasses that packing (it warns about exactly this).
    armour.new_effect.modify_attribute(
        object_list_unit_id=_UNIT_ARCHER,
        source_player=_PLAYER_ONE,
        operation=_OPERATION_ADD,
        object_attributes=_ATTRIBUTE_ARMOR,
        armour_attack_quantity=2,
        armour_attack_class=3,
    )
    armour.new_effect.modify_attribute(
        object_list_unit_id=_UNIT_ARCHER,
        source_player=_PLAYER_ONE,
        operation=_OPERATION_SET,
        object_attributes=_ATTRIBUTE_HIT_POINTS,
        quantity=45,
    )

    references = manager.add_trigger("Fixture: references", enabled=False)
    references.new_condition.timer(timer=2)
    references.new_effect.activate_trigger(trigger_id=setup.trigger_id)
    references.new_effect.deactivate_trigger(trigger_id=armour.trigger_id)

    manager.add_variable(VARIABLE_NAME, VARIABLE_ID)
    variable = manager.add_trigger("Fixture: variable")
    variable.new_condition.timer(timer=3)
    variable.new_effect.change_variable(
        quantity=1, operation=_OPERATION_SET, variable=VARIABLE_ID, message=VARIABLE_NAME
    )

    manager.commit()


def _patch_counter(buffer: bytearray, end_offset: int, value: int) -> None:
    """Overwrites the u32 that *ends* at end_offset. Both trigger counters are
    the last retriever of their section (verified across all 19 DE structure
    versions), so 'ends at the section end' is how each is addressed."""
    struct.pack_into("<I", buffer, end_offset - 4, value)


def build_fixture_bytes(donor_path: Path = BLANK_TEMPLATE_PATH) -> bytes:
    """The whole fixture as a full .aoe2scenario byte string.

    Splices a freshly-serialized Triggers section into the donor's otherwise
    verbatim decompressed body, then patches the two trigger counters that
    live outside that section (Options.number_of_triggers and
    FileHeader.trigger_count). The same three-region layout descape/
    scenario_write.py's triggers branch uses, hand-rolled here so the
    generator stays independent of the code its output tests.
    """
    loaded = load_map_and_units(donor_path)
    _build_triggers(loaded)

    section_bytes = _game_style_bytes(loaded._scenario.sections["Triggers"])
    body = bytearray(
        loaded.decompressed_body[: loaded.units_section_end]
        + section_bytes
        + loaded.decompressed_body[loaded.triggers_section_end :]
    )
    # Options sits before Map, so its offset is unaffected by the splice.
    _patch_counter(body, loaded.options_section_end, TRIGGER_COUNT)

    header = bytearray(loaded.header_bytes)
    _patch_counter(header, len(header), TRIGGER_COUNT)

    deflate = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return bytes(header) + deflate.compress(bytes(body)) + deflate.flush()


def verify_fixture(path: Path) -> None:
    """Reloads through the real loader, not just re-decompressing in place --
    the same discipline tools/gen_blank_maps.py and tools/strip_units.py use."""
    reloaded = load_map_and_units(path)
    manager = parse_triggers(reloaded)
    if manager is None:
        raise GenerationVerificationError(f"{path}: Triggers section did not parse after reload")
    if len(manager.triggers) != TRIGGER_COUNT:
        raise GenerationVerificationError(
            f"{path}: reloaded with {len(manager.triggers)} triggers, expected {TRIGGER_COUNT}"
        )
    if not reloaded.trigger_write_supported:
        raise GenerationVerificationError(f"{path}: failed the trigger alignment gate after reload")
    if not reloaded.terrain_write_supported:
        raise GenerationVerificationError(f"{path}: terrain block failed reload verification")
    if len(manager.variables) != 1 or manager.variables[0].name != VARIABLE_NAME:
        raise GenerationVerificationError(f"{path}: variable block did not survive the reload")
    counter = struct.unpack_from("<I", reloaded.decompressed_body, reloaded.options_section_end - 4)[0]
    if counter != TRIGGER_COUNT:
        raise GenerationVerificationError(
            f"{path}: Options.number_of_triggers is {counter}, expected {TRIGGER_COUNT}"
        )
    counter = struct.unpack_from("<I", reloaded.header_bytes, len(reloaded.header_bytes) - 4)[0]
    if counter != TRIGGER_COUNT:
        raise GenerationVerificationError(
            f"{path}: FileHeader.trigger_count is {counter}, expected {TRIGGER_COUNT}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH, help="Destination file")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = build_fixture_bytes()
    args.out.write_bytes(data)
    verify_fixture(args.out)
    print(f"{args.out}: {len(data)} bytes, verified {TRIGGER_COUNT} triggers")


if __name__ == "__main__":
    main()
