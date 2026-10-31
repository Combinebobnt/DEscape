#!/usr/bin/env python3
"""Writes one .aoe2scenario for checking the Display Instructions box (TASK-550)
against AoE2:DE, from examples/blank_map through the real write path
(TriggerEditModel.structural_edit() + scenario_write.write_scenario(), the same
code the GUI uses), never a hand-crafted byte patch.

Four screens, 20 s each, every message's display time ending before the next:

1. BOTTOM alone.
2. MIDDLE alone.
3. TOP <BLUE>, MIDDLE <GREEN> and BOTTOM untagged, together, each with an icon.
4. A long message, one with no icon, and one with position -1.

Screens 1 and 2 come first because they isolate the position field: with several
messages up, arrival-order stacking would look the same as the field picking the
slot. Each message names its screen and position, so a screenshot reads alone.

Writes into build/task550_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel

BLANK = ROOT / "examples" / "blank_map.aoe2scenario"
OUT = ROOT / "build" / "task550_pair"
SCENARIO = OUT / "instruction_slots.aoe2scenario"

TOP, MIDDLE, BOTTOM, UNSET = 0, 1, 2, -1
SCOUT, CASTLE, VILLAGER = 448, 82, 83
NO_ICON = -1
FIRST_SCREEN_S = 5
SCREEN_S = 20
DISPLAY_S = 15
LONG = "Screen 4, long message at TOP: " + "a long line of narration that keeps going. " * 8


class Message(NamedTuple):
    position: int
    text: str
    icon: int


SCREENS: tuple[tuple[Message, ...], ...] = (
    (Message(BOTTOM, "Screen 1: BOTTOM alone", SCOUT),),
    (Message(MIDDLE, "Screen 2: MIDDLE alone", SCOUT),),
    (
        Message(TOP, "<BLUE>Screen 3: TOP, blue tag", SCOUT),
        Message(MIDDLE, "<GREEN>Screen 3: MIDDLE, green tag", CASTLE),
        Message(BOTTOM, "Screen 3: BOTTOM, untagged", VILLAGER),
    ),
    (
        Message(TOP, LONG, SCOUT),
        Message(MIDDLE, "Screen 4: MIDDLE, no icon", NO_ICON),
        Message(UNSET, "Screen 4: position -1", VILLAGER),
    ),
)


def _build() -> None:
    loaded = load_map_and_units(BLANK)
    model = TriggerEditModel(loaded)
    assert not model.manager().triggers, "blank_map assumption: no triggers"

    def mutate(manager) -> None:
        for number, messages in enumerate(SCREENS):
            trigger = manager.add_trigger(f"Screen {number + 1}")
            trigger.new_condition.timer(timer=FIRST_SCREEN_S + number * SCREEN_S)
            for message in messages:
                trigger.new_effect.display_instructions(
                    source_player=1,
                    object_list_unit_id=message.icon,
                    display_time=DISPLAY_S,
                    instruction_panel_position=message.position,
                    play_sound=0,
                    message=message.text,
                )

    model.structural_edit(mutate)
    write_scenario(loaded, SCENARIO, triggers=model)


def _check() -> None:
    manager = parse_triggers(load_map_and_units(SCENARIO))
    assert len(manager.triggers) == len(SCREENS), "a screen is missing"
    for trigger, messages in zip(manager.triggers, SCREENS, strict=True):
        got = [(e.instruction_panel_position, e.message, e.object_list_unit_id) for e in trigger.effects]
        assert got == [(m.position, m.text, m.icon) for m in messages], f"{trigger.name} read back differently"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _build()
    _check()
    print(f"{SCENARIO.relative_to(ROOT)}: read back, {len(SCREENS)} screens")
    for number, messages in enumerate(SCREENS):
        at = FIRST_SCREEN_S + number * SCREEN_S
        print(f"  {at} s, up for {DISPLAY_S} s: " + "; ".join(m.text[:40] for m in messages))


if __name__ == "__main__":
    main()
