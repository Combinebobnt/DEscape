#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming Section Down (GH
#133) in the real AoE2:DE editor. Both sides go through the real write path
(TriggerEditModel.structural_edit() + scenario_write.write_scenario(), the
same code the GUI uses), never a hand-crafted byte patch.

The edit is one section move on F7_3_York (base-game civs only, so a tester
without DLC can open it): "--- Raiding Mechanism ---" (4 members) moves
below "--- Settling Down ---" (5 members). Unequal lengths, so it is a
rotation of 11 display slots rather than a swap of two, and the after file
is read back to confirm only the display-order region changed.

Writes into build/section_move_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

from descape import trigger_organize
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel, moved_section_display_order

SOURCE = ROOT / "examples" / "F7_3_York (865).aoe2scenario"
OUT = ROOT / "build" / "section_move_pair"
MOVED = "--- Raiding Mechanism ---"
DELTA = 1


def _body(path: Path) -> bytes:
    loaded = load_map_and_units(path)
    return _decompress_bytes(path.read_bytes()[len(loaded.header_bytes) :])


def _summary(names: list[str], order: list[int], around: int) -> list[str]:
    parts = trigger_organize.sections(names, order)
    slot = {index: s for s, index in enumerate(order)}
    lines = []
    for position in range(max(0, around - 1), min(len(parts), around + 3)):
        section = parts[position]
        first = slot[section.header_index]
        last = first + len(section.member_indices)
        lines.append(f"    display # {first}-{last}: {section.title!r} ({len(section.member_indices)} members)")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load_map_and_units(SOURCE)
    before = OUT / "york_section_move_before.aoe2scenario"
    after = OUT / "york_section_move_after.aoe2scenario"
    write_scenario(loaded, before)

    model = TriggerEditModel(loaded)
    names = [t.name or "" for t in model.manager().triggers]
    order = list(model.manager().trigger_display_order)
    parts = trigger_organize.sections(names, order)
    position = next(p for p, s in enumerate(parts) if s.title == MOVED)
    header = parts[position].header_index
    moved = moved_section_display_order(order, names, header, DELTA)
    model.structural_edit(lambda m: setattr(m, "trigger_display_order", list(moved)))
    write_scenario(loaded, after, triggers=model)

    # Read the written file back rather than trusting the in-memory objects.
    reread = parse_triggers(load_map_and_units(after))
    assert [t.name or "" for t in reread.triggers] == names, "a trigger name changed"
    assert list(reread.trigger_display_order) == moved, "the written order is not the moved one"
    expected = list(parts)
    expected[position], expected[position + DELTA] = expected[position + DELTA], expected[position]
    assert trigger_organize.sections(names, moved) == expected, "a section changed membership"

    a, b = _body(before), _body(after)
    assert len(a) == len(b)
    differing = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    start = loaded.units_section_end + model.regions.triggers_end
    end = loaded.units_section_end + model.regions.display_order_end
    assert differing and all(start <= i < end for i in differing), "bytes changed outside display order"
    slots = sum(1 for x, y in zip(order, moved, strict=True) if x != y)

    print(f"{SOURCE.name}: Section Down on {MOVED!r} (trigger ID {header})")
    print("  before:")
    print("\n".join(_summary(names, order, position)))
    print("  after:")
    print("\n".join(_summary(names, moved, position)))
    print(f"  {slots} display slots changed, {len(differing)} bytes, all inside the display-order array")
    print(f"  {before.relative_to(ROOT)} / {after.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
