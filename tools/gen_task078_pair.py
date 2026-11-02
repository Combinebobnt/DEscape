#!/usr/bin/env python3
"""Writes before/after .aoe2scenario pairs for confirming Renumber IDs to
Display Order (GH #2) in AoE2:DE. Every side goes through the real write path
(TriggerEditModel.structural_edit() + scenario_write.write_scenario(), the
same code the GUI uses), never a hand-crafted byte patch.

Run-order pair, from examples/blank_map: triggers "A".."E" each Send Chat
"run A".."run E"; "F linked" starts disabled, waits on a 5 s Timer and says
"F linked", and E activates it; "G waits on F" says "G saw F" once F is
active (Trigger Active). Legacy execution order, display order
G, C, F, A, E, B, D, so both links change ID under the renumber. Playing
each file prints a different chat transcript, listed below when this runs.

Real-file pair: atilla_1_scn_resaved untouched and renumbered, with the rows
the library rewrites on every re-serialized trigger printed so they can be
looked up in the editor.

Writes into build/task078_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.objects.managers.trigger_manager import get_trigger_referencing_ce

from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import (
    TriggerEditModel,
    exec_order_value,
    exec_order_write_supported,
    renumber_refusal,
    renumber_to_display_order,
)

BLANK = ROOT / "examples" / "blank_map.aoe2scenario"
ATILLA = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
OUT = ROOT / "build" / "task078_pair"
RUNS = ["A", "B", "C", "D", "E"]
F_NAME, G_NAME = "F linked", "G waits on F"
# List indices: A..E are 0..4, F is 5, G is 6.
DISPLAY = [6, 2, 5, 0, 4, 1, 3]
ROWS = 3


def _links(manager) -> dict[str, list[str]]:
    """Each trigger's links, as the names of the triggers they point at."""
    names = [t.name or "" for t in manager.triggers]
    return {
        t.name: [names[ce.trigger_id] if 0 <= ce.trigger_id < len(names) else str(ce.trigger_id) for ce in refs]
        for t in manager.triggers
        if (refs := get_trigger_referencing_ce(t))
    }


def _renumbered(source: Path, out: Path) -> TriggerEditModel:
    loaded = load_map_and_units(source)
    model = TriggerEditModel(loaded)
    reason = renumber_refusal(model.manager())
    assert not reason, f"{source.name}: {reason}"
    model.structural_edit(renumber_to_display_order)
    write_scenario(loaded, out, triggers=model)
    return model


def _build_run_order(before: Path) -> None:
    loaded = load_map_and_units(BLANK)
    model = TriggerEditModel(loaded)
    assert not model.manager().triggers, "blank_map assumption: no triggers"

    def mutate(m) -> None:
        for name in RUNS:
            m.add_trigger(name).new_effect.send_chat(source_player=1, message=f"run {name}")
        f = m.add_trigger(F_NAME, enabled=False)
        f.new_condition.timer(timer=5)
        f.new_effect.send_chat(source_player=1, message=F_NAME)
        m.triggers[RUNS.index("E")].new_effect.activate_trigger(trigger_id=f.trigger_id)
        g = m.add_trigger(G_NAME)
        g.new_condition.trigger_active(trigger_id=f.trigger_id)
        g.new_effect.send_chat(source_player=1, message="G saw F")
        m.trigger_display_order = list(DISPLAY)

    model.structural_edit(mutate)
    model.set_exec_order(1)
    write_scenario(loaded, before, triggers=model)


def _run_order_pair() -> tuple[Path, Path]:
    blank = load_map_and_units(BLANK)
    if parse_triggers(blank) is None or not exec_order_write_supported(blank):
        sys.exit(f"{BLANK.name} cannot store the execution-order flag; refusing to write a pair that tests nothing")
    before = OUT / "run_order_before.aoe2scenario"
    after = OUT / "run_order_after.aoe2scenario"
    _build_run_order(before)
    _renumbered(before, after)

    expected_ids = {before: [*RUNS, F_NAME, G_NAME], after: [G_NAME, "C", F_NAME, "A", "E", "B", "D"]}
    for path, ids in expected_ids.items():
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
        assert [t.name for t in manager.triggers] == ids, f"{path.name}: ID order"
        assert [ids[i] for i in manager.trigger_display_order] == [G_NAME, "C", F_NAME, "A", "E", "B", "D"]
        assert _links(manager) == {"E": [F_NAME], G_NAME: [F_NAME]}, f"{path.name}: a link lost F"
        assert exec_order_value(loaded) == 1, f"{path.name}: not Legacy"
    return before, after


def _raw(loaded) -> list:
    return loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data or []


def _effect_values(struct) -> list[dict]:
    return [{n: r.data for n, r in e.retriever_map.items()} for e in struct.retriever_map["effect_data"].data or []]


def _rewrites(before: Path, after: Path, order: list[int], dirty: set[int]) -> dict[str, list[str]]:
    """Up to ROWS rows of each library rewrite, by trigger name and effect index."""
    old_loaded = load_map_and_units(before)
    parse_triggers(old_loaded)
    old = [_effect_values(s) for s in _raw(old_loaded)]
    names = [s.retriever_map["trigger_name"].data for s in _raw(old_loaded)]
    new_loaded = load_map_and_units(after)
    parse_triggers(new_loaded)
    new = [_effect_values(s) for s in _raw(new_loaded)]
    kinds: dict[str, list[str]] = {"item_id filled": [], "legacy location moved": []}
    for position, index in enumerate(order):
        if position not in dirty:
            continue
        for k, (o, n) in enumerate(zip(old[index], new[position], strict=True)):
            where = f"{names[index]!r} effect {k} (type {o['effect_type']})"
            if o.get("item_id") != n.get("item_id"):
                kinds["item_id filled"].append(f"{where}: item_id {o['item_id']} -> {n['item_id']}")
            if o.get("legacy_location_object_reference") != n.get("legacy_location_object_reference"):
                kinds["legacy location moved"].append(
                    f"{where}: location object {o['location_object_reference']} -> {n['location_object_reference']}"
                )
    return kinds


def _atilla_pair() -> None:
    before = OUT / "atilla_before.aoe2scenario"
    after = OUT / "atilla_after.aoe2scenario"
    loaded = load_map_and_units(ATILLA)
    manager = parse_triggers(loaded)
    order = list(manager.trigger_display_order)
    names = [t.name or "" for t in manager.triggers]
    write_scenario(loaded, before)
    model = _renumbered(ATILLA, after)
    dirty = set(model.dirty_indices())

    reread = parse_triggers(load_map_and_units(after))
    assert [t.name or "" for t in reread.triggers] == [names[i] for i in order], "atilla: ID order"
    print(f"{ATILLA.name}: {len(order)} triggers, {len(dirty)} re-serialized")
    print("  first three in display order (both files):", ", ".join(repr(names[i]) for i in order[:3]))
    for kind, rows in _rewrites(before, after, order, dirty).items():
        print(f"  {kind}: {len(rows)}")
        # One row per trigger, so each names a different place to look.
        firsts = {row.split(" effect ")[0]: row for row in reversed(rows)}
        for row in list(reversed(firsts.values()))[:ROWS]:
            print(f"    {row}")
    print(f"  {before.relative_to(ROOT)} / {after.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before, after = _run_order_pair()
    print(f"{BLANK.name}: Legacy, list shows {G_NAME}, C, {F_NAME}, A, E, B, D")
    print("  before (IDs A..E, F, G), chat: run A, run B, run C, run D, run E, G saw F, about 5 s later F linked")
    print("  after (IDs follow the list), chat: run C, run A, run E, run B, run D, G saw F, about 5 s later F linked")
    print(f"  {before.relative_to(ROOT)} / {after.relative_to(ROOT)}")
    _atilla_pair()


if __name__ == "__main__":
    main()
