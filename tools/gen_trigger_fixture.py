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
    FORBIDDEN_WRITE_MARKER,
    LoadedScenario,
    is_under_compatdata,
    load_map_and_units,
    parse_triggers,
)
from descape.scenario_write import WriteBlockedError

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
#                              after unknown_bytes actually has content. Then
#                              the map coordinates descape.trigger_geometry
#                              reads: a condition area, a patrol (area +
#                              location, the run-line case), a location-only
#                              effect, a half-set area and a whole-map area.
#                              Appended to this trigger rather than a fifth
#                              one, which would reshape every display-order
#                              and trigger-count assertion in the suite.
#                              Last, the placed-unit references
#                              descape.unit_references resolves: a Destroy
#                              Object on the house, a Task Object carrying the
#                              corpus's 60/61 shape (two selected archers, the
#                              house as location_object_reference and its own
#                              tile as location), a Patrol with selected ids
#                              and no location, and a Destroy Object naming
#                              an id nothing carries.
#
# Three Player 1 units back those references. The Units section is spliced
# from the library's own serializer the same way tools/gen_units_fixture.py
# does it, never through descape.unit_model.
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
_UNSET = -1

# Trigger 3's coordinates, inclusive corners. Asymmetric on purpose, so
# a swapped x/y reads back wrong rather than coincidentally right.
GEOMETRY_CONDITION_AREA = (10, 20, 19, 29)
GEOMETRY_PATROL_AREA = (30, 40, 34, 46)
GEOMETRY_PATROL_LOCATION = (60, 50)
GEOMETRY_CREATE_LOCATION = (70, 80)
# x1 unset, x2 set: the one half-set direction validate_coords() leaves alone
# (x1 set with x2 unset gets x2 := x1 at parse time and reads back as 1 wide).
GEOMETRY_HALF_SET_AREA = (-1, 5, 12, 9)
GEOMETRY_WHOLE_MAP_AREA = (0, 0, 119, 119)

# Placed units for the reference entries. Asymmetric x/y, and the house is
# 2x2 anchored at tile + span/2 like every corpus building.
_UNIT_HOUSE = 70
REF_ARCHER_A = 500
REF_ARCHER_B = 501
REF_HOUSE = 502
ARCHER_A_POS = (50.5, 55.5)
ARCHER_B_POS = (52.5, 57.5)
HOUSE_POS = (41.0, 63.0)
HOUSE_TILE = (41, 63)
PLACED_REFERENCE_IDS = (REF_ARCHER_A, REF_ARCHER_B, REF_HOUSE)
# Above the highest id placed, as in every corpus file.
NEXT_UNIT_ID = 600
DANGLING_REFERENCE_ID = 999

# Where the reference entries land in trigger 3, after the geometry ones.
REFERENCE_TRIGGER = 3
DESTROY_CONDITION = 2
DANGLING_CONDITION = 3
TASK_OBJECT_EFFECT = 5
PATROL_IDS_EFFECT = 6
TASK_SELECTED_IDS = (REF_ARCHER_A, REF_ARCHER_B)
TASK_TARGET_ID = REF_HOUSE
PATROL_SELECTED_IDS = (REF_ARCHER_B,)


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

    # source_player unset (the library would default it to 1): player_stats
    # counts trigger 3 as naming no player, and several tests pin that.
    x1, y1, x2, y2 = GEOMETRY_CONDITION_AREA
    variable.new_condition.objects_in_area(quantity=1, source_player=_UNSET, area_x1=x1, area_y1=y1, area_x2=x2, area_y2=y2)
    x1, y1, x2, y2 = GEOMETRY_PATROL_AREA
    variable.new_effect.patrol(
        source_player=_UNSET,
        area_x1=x1, area_y1=y1, area_x2=x2, area_y2=y2,
        location_x=GEOMETRY_PATROL_LOCATION[0], location_y=GEOMETRY_PATROL_LOCATION[1],
    )
    variable.new_effect.create_object(
        object_list_unit_id=_UNIT_ARCHER,
        source_player=_UNSET,
        location_x=GEOMETRY_CREATE_LOCATION[0],
        location_y=GEOMETRY_CREATE_LOCATION[1],
    )
    x1, y1, x2, y2 = GEOMETRY_HALF_SET_AREA
    variable.new_effect.kill_object(source_player=_UNSET, area_x1=x1, area_y1=y1, area_x2=x2, area_y2=y2)
    x1, y1, x2, y2 = GEOMETRY_WHOLE_MAP_AREA
    variable.new_effect.remove_object(source_player=_UNSET, area_x1=x1, area_y1=y1, area_x2=x2, area_y2=y2)

    variable.new_condition.destroy_object(unit_object=TASK_TARGET_ID)
    variable.new_condition.destroy_object(unit_object=DANGLING_REFERENCE_ID)
    variable.new_effect.task_object(
        source_player=_UNSET,
        selected_object_ids=list(TASK_SELECTED_IDS),
        location_object_reference=TASK_TARGET_ID,
        location_x=HOUSE_TILE[0],
        location_y=HOUSE_TILE[1],
    )
    variable.new_effect.patrol(source_player=_UNSET, selected_object_ids=list(PATROL_SELECTED_IDS))

    manager.commit()


def _build_units(loaded: LoadedScenario) -> None:
    """Places the three Player 1 units the reference entries name."""
    manager = loaded.unit_manager
    for player in range(9):
        if manager.units[player]:
            raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor already has units on player {player}")
    for const, (x, y), ref in (
        (_UNIT_ARCHER, ARCHER_A_POS, REF_ARCHER_A),
        (_UNIT_ARCHER, ARCHER_B_POS, REF_ARCHER_B),
        (_UNIT_HOUSE, HOUSE_POS, REF_HOUSE),
    ):
        manager.add_unit(player=_PLAYER_ONE, unit_const=const, x=x, y=y, reference_id=ref)
    manager.commit()


def _patch_counter(buffer: bytearray, end_offset: int, value: int) -> None:
    """Overwrites the u32 that *ends* at end_offset. Both trigger counters are
    the last retriever of their section (verified across every DE structure
    version, library and repo), so 'ends at the section end' is how each is
    addressed."""
    struct.pack_into("<I", buffer, end_offset - 4, value)


def build_fixture_bytes(donor_path: Path = BLANK_TEMPLATE_PATH) -> bytes:
    """The whole fixture as a full .aoe2scenario byte string.

    Splices freshly-serialized players_units and Triggers bytes into the
    donor's otherwise verbatim decompressed body, then patches the two trigger
    counters that live outside that section (Options.number_of_triggers and
    FileHeader.trigger_count) and DataHeader.next_unit_id_to_place. The same
    region layout descape/scenario_write.py's units and triggers branches
    use, hand-rolled here so the generator stays independent of the code its
    output tests. Both splices are cut from the donor's own offsets in one
    expression, since the first one shifts everything after it.
    """
    loaded = load_map_and_units(donor_path, fast_units=False)
    _build_units(loaded)
    _build_triggers(loaded)
    return _spliced_bytes(loaded, TRIGGER_COUNT, NEXT_UNIT_ID)


def _spliced_bytes(loaded: LoadedScenario, trigger_count: int, next_unit_id: int) -> bytes:
    """build_fixture_bytes()'s splice and counter patches, shared with the Find fixture."""
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
    struct.pack_into("<I", body, 0, next_unit_id)
    # Options sits before Map, so its offset is unaffected by the splice.
    _patch_counter(body, loaded.options_section_end, trigger_count)

    header = bytearray(loaded.header_bytes)
    _patch_counter(header, len(header), trigger_count)

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
    if not reloaded.units_write_supported:
        raise GenerationVerificationError(f"{path}: failed the units alignment gate after reload")
    placed = sorted(u.reference_id for units in reloaded.unit_manager.units for u in units)
    if placed != sorted(PLACED_REFERENCE_IDS) or len(reloaded.unit_manager.units[_PLAYER_ONE]) != 3:
        raise GenerationVerificationError(f"{path}: placed units read back as {placed}")
    task = manager.triggers[REFERENCE_TRIGGER].effects[TASK_OBJECT_EFFECT]
    if tuple(task.selected_object_ids) != TASK_SELECTED_IDS or task.location_object_reference != TASK_TARGET_ID:
        raise GenerationVerificationError(f"{path}: the task_object references did not survive the reload")
    if len(manager.variables) != 1 or manager.variables[0].name != VARIABLE_NAME:
        raise GenerationVerificationError(f"{path}: variable block did not survive the reload")
    half_set = manager.triggers[3].effects[3]
    read_back = (half_set.area_x1, half_set.area_y1, half_set.area_x2, half_set.area_y2)
    if read_back != GEOMETRY_HALF_SET_AREA:
        raise GenerationVerificationError(
            f"{path}: half-set area reads back as {read_back}, expected {GEOMETRY_HALF_SET_AREA}"
        )
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


# -- tests/fixtures/find_triggers_120x120.aoe2scenario (GH #144 Part B) ----------
#
# A separate output, never folded into triggers_120x120 (about 30 test files
# read that one). It carries the shapes Edit > Find and Replace's trigger
# search needs and the corpus barely has (zero create_decision effects, one
# script_call, no display_instructions with a newline):
#
# 0 "[Find] Messages"   a tagged name; a \r\n description; display_instructions
#                       with a \r\n message and a sound_name; create_decision
#                       (its option messages are str32 the vocabulary calls
#                       INT); change_variable naming a variable (an
#                       identifier); script_call with a CR-separated XS body.
# 1 "--- Section ---"   a divider.
# 2 "Retarget"          object_attacked naming Archer next to a selected unit
#                       (tier d), then effects sorted for a Replace of archer
#                       FIND_ARCHER_A: task_object pairing it with an Archer
#                       type filter (tier a), create_object of an Archer (tier
#                       b), patrol over both archers (tier c), task_object
#                       targeting it via location_object_reference (tier d),
#                       attack_move with an Archer class filter (tier e),
#                       replace_object into Archer (object_list_unit_id_2) and
#                       build_object naming a House (building_list).
# 3 "Find: disabled"    disabled, a timer only.
FIND_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "find_triggers_120x120.aoe2scenario"
FIND_TRIGGER_COUNT = 4
FIND_VARIABLE_NAME = "find_var"
FIND_ARCHER_A = 500
FIND_ARCHER_B = 501
FIND_HOUSE = 502
FIND_VILLAGER = 503
_UNIT_VILLAGER = 83
_CLASS_ARCHER = 0
FIND_NEXT_UNIT_ID = 600
FIND_DESCRIPTION = "First line of the description.\r\nSecond line."
FIND_INSTRUCTIONS = "Hold the bridge.\r\nArchers to the walls."
FIND_SOUND = "find_sound"
FIND_DECISION = ("Spare the village?", "Spare it", "Burn it")
FIND_XS = "void findMain() {\r   int findCounter = 0;\r}"
FIND_TAGGED_NAME = "[Find] Messages"
FIND_DIVIDER_NAME = "--- Section ---"
FIND_RETARGET_TRIGGER = 2


def _build_find_units(loaded: LoadedScenario) -> None:
    manager = loaded.unit_manager
    for const, (x, y), ref in (
        (_UNIT_ARCHER, (50.5, 55.5), FIND_ARCHER_A),
        (_UNIT_ARCHER, (52.5, 57.5), FIND_ARCHER_B),
        (_UNIT_HOUSE, (41.0, 63.0), FIND_HOUSE),
        (_UNIT_VILLAGER, (44.5, 60.5), FIND_VILLAGER),
    ):
        manager.add_unit(player=_PLAYER_ONE, unit_const=const, x=x, y=y, reference_id=ref)
    manager.commit()


def _build_find_triggers(loaded: LoadedScenario) -> None:
    manager = parse_triggers(loaded)
    if manager is None or manager.triggers:
        raise GenerationVerificationError(f"{BLANK_TEMPLATE_PATH}: donor's Triggers section is not an empty parse")
    manager.add_variable(FIND_VARIABLE_NAME, VARIABLE_ID)

    messages = manager.add_trigger(FIND_TAGGED_NAME, description=FIND_DESCRIPTION)
    messages.new_condition.timer(timer=1)
    messages.new_effect.display_instructions(
        source_player=_PLAYER_ONE, message=FIND_INSTRUCTIONS, sound_name=FIND_SOUND, display_time=10,
        instruction_panel_position=0,
    )
    prompt, option1, option2 = FIND_DECISION
    messages.new_effect.create_decision(decision_id=0, message=prompt, message_option1=option1, message_option2=option2)
    messages.new_effect.change_variable(
        quantity=1, operation=_OPERATION_SET, variable=VARIABLE_ID, message=FIND_VARIABLE_NAME
    )
    messages.new_effect.script_call(message=FIND_XS)

    manager.add_trigger(FIND_DIVIDER_NAME)

    retarget = manager.add_trigger("Retarget")
    retarget.new_condition.object_attacked(
        source_player=_PLAYER_ONE, object_list=_UNIT_ARCHER, unit_object=FIND_ARCHER_A
    )
    retarget.new_effect.task_object(
        source_player=_PLAYER_ONE, object_list_unit_id=_UNIT_ARCHER, selected_object_ids=[FIND_ARCHER_A],
        location_x=60, location_y=60,
    )
    retarget.new_effect.create_object(
        object_list_unit_id=_UNIT_ARCHER, source_player=_PLAYER_ONE, location_x=70, location_y=70
    )
    retarget.new_effect.patrol(
        source_player=_PLAYER_ONE, object_list_unit_id=_UNIT_ARCHER,
        selected_object_ids=[FIND_ARCHER_A, FIND_ARCHER_B], location_x=65, location_y=65,
    )
    retarget.new_effect.task_object(
        source_player=_PLAYER_ONE, object_list_unit_id=_UNIT_ARCHER, selected_object_ids=[FIND_VILLAGER],
        location_object_reference=FIND_ARCHER_A,
    )
    retarget.new_effect.attack_move(
        source_player=_PLAYER_ONE, object_list_unit_id=_UNIT_ARCHER, object_group=_CLASS_ARCHER,
        selected_object_ids=[FIND_ARCHER_A], location_x=66, location_y=66,
    )
    retarget.new_effect.replace_object(
        source_player=_PLAYER_ONE, target_player=_PLAYER_ONE, object_list_unit_id=_UNIT_VILLAGER,
        object_list_unit_id_2=_UNIT_ARCHER,
    )
    retarget.new_effect.build_object(
        source_player=_PLAYER_ONE, building_list=_UNIT_HOUSE, selected_object_ids=[FIND_VILLAGER],
        location_x=30, location_y=30,
    )

    disabled = manager.add_trigger("Find: disabled", enabled=False)
    disabled.new_condition.timer(timer=9)
    manager.commit()


def build_find_fixture_bytes(donor_path: Path = BLANK_TEMPLATE_PATH) -> bytes:
    loaded = load_map_and_units(donor_path, fast_units=False)
    _build_find_units(loaded)
    _build_find_triggers(loaded)
    return _spliced_bytes(loaded, FIND_TRIGGER_COUNT, FIND_NEXT_UNIT_ID)


def verify_find_fixture(path: Path) -> None:
    reloaded = load_map_and_units(path)
    manager = parse_triggers(reloaded)
    if manager is None or len(manager.triggers) != FIND_TRIGGER_COUNT:
        raise GenerationVerificationError(f"{path}: did not reload with {FIND_TRIGGER_COUNT} triggers")
    if not (reloaded.trigger_write_supported and reloaded.units_write_supported and reloaded.terrain_write_supported):
        raise GenerationVerificationError(f"{path}: failed a write-path alignment gate after reload")
    messages = manager.triggers[0]
    if messages.description != FIND_DESCRIPTION or messages.effects[0].message != FIND_INSTRUCTIONS:
        raise GenerationVerificationError(f"{path}: the \\r\\n prose did not survive the reload")
    decision = messages.effects[1]
    if (decision.message, decision.message_option1, decision.message_option2) != FIND_DECISION:
        raise GenerationVerificationError(f"{path}: create_decision read back wrong")
    if messages.effects[3].message != FIND_XS:
        raise GenerationVerificationError(f"{path}: the XS body read back wrong")
    retarget = manager.triggers[FIND_RETARGET_TRIGGER]
    if len(retarget.effects) != 7 or tuple(retarget.effects[2].selected_object_ids) != (FIND_ARCHER_A, FIND_ARCHER_B):
        raise GenerationVerificationError(f"{path}: the Retarget effects read back wrong")
    if manager.triggers[3].enabled:
        raise GenerationVerificationError(f"{path}: the disabled trigger reads back enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Destination file")
    parser.add_argument(
        "--find", action="store_true", help=f"Write the Find and Replace fixture ({FIND_FIXTURE_PATH.name}) instead"
    )
    args = parser.parse_args()
    out = args.out or (FIND_FIXTURE_PATH if args.find else FIXTURE_PATH)

    if is_under_compatdata(out):
        raise WriteBlockedError(f"Refusing to write under a Proton {FORBIDDEN_WRITE_MARKER}/ folder: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_find_fixture_bytes() if args.find else build_fixture_bytes()
    out.write_bytes(data)
    if args.find:
        verify_find_fixture(out)
    else:
        verify_fixture(out)
    count = FIND_TRIGGER_COUNT if args.find else TRIGGER_COUNT
    print(f"{out}: {len(data)} bytes, verified {count} triggers")


if __name__ == "__main__":
    main()
