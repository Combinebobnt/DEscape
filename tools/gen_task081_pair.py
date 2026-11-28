#!/usr/bin/env python3
"""Writes before/after .aoe2scenario pairs for confirming cross-version
trigger paste (GH #317) in AoE2:DE. Every after file goes through the real
path the GUI uses: trigger_clipboard.copy_block(), paste_into() inside
TriggerEditModel.structural_edit() with the display order kept as the
viewer keeps it (the block lands last), then scenario_write.write_scenario().

- newmap_159_before / newmap_159_after: a File > New Map 120x120 1.59 map,
  untouched, and with every trigger of R4_LeLoi_4 (1.56) pasted, bar the one
  holding effect 91 (no vocabulary defines it, so Paste refuses it).
- atilla_158_before / atilla_158_after: atilla_1_scn_resaved (1.58) through a
  zero-edit write, and with every trigger of 8tp3w9j (1.59) pasted.

Writes into build/task081_pair/, which is gitignored.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import trigger_clipboard
from descape.scenario_io import load_map_and_units, load_map_and_units_from_bytes, parse_triggers
from descape.scenario_new import blank_scenario_bytes
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel, display_order_with_block_inserted

EXAMPLES = ROOT / "examples"
OUT = ROOT / "build" / "task081_pair"
UNKNOWN_EFFECTS = range(91, 98)
_KEEP_ALIVE: list = []  # the library's scenario store is weak


def _load(path: Path):
    loaded = load_map_and_units(path)
    _KEEP_ALIVE.append(loaded)
    return loaded


def _paste(source, destination, out: Path) -> None:
    manager = parse_triggers(source)
    indices = [i for i, t in enumerate(manager.triggers) if not any(e.effect_type in UNKNOWN_EFFECTS for e in t.effects)]
    left = len(manager.triggers) - len(indices)
    block = trigger_clipboard.copy_block(manager, indices, doc_id="source", scenario_version=source.scenario_version)
    model = TriggerEditModel(destination)
    before = len(model.manager().triggers)
    results = []

    def paste(m) -> None:
        # As the viewer does: the import flattens display order, so the old one is pushed back with the block at the end.
        before_order = list(m.trigger_display_order)
        result = trigger_clipboard.paste_into(m, block, doc_id="dest", scenario_version=destination.scenario_version)
        results.append(result)
        new_indices = list(range(before, before + len(result.triggers)))
        m.trigger_display_order = display_order_with_block_inserted(before_order, new_indices, None)

    model.structural_edit(paste)
    write_scenario(destination, out, backup=False, triggers=model)
    result = results[0]
    reloaded = _load(out)
    pasted = parse_triggers(reloaded).triggers[before:]
    first_display = next(
        ((t.name, e.message) for t in pasted for e in t.effects if e.effect_type == 20 and e.message), ("(none)", "")
    )
    print(
        f"{out.name}: {source.scenario_version} into {destination.scenario_version}, "
        f"{len(pasted)} triggers pasted after the {before} already there ({left} left behind), "
        f"{sum(len(t.conditions) for t in pasted)} conditions, {sum(len(t.effects) for t in pasted)} effects; "
        f"{result.cleared_unit_refs} unit refs and {result.cleared_trigger_links} outside links cleared; "
        f"reload trigger-writable={reloaded.trigger_write_supported}"
    )
    print(f"  first pasted Display Instructions: trigger {first_display[0]!r}, text {first_display[1][:60]!r}")


def main() -> int:
    warnings.simplefilter("ignore")
    OUT.mkdir(parents=True, exist_ok=True)

    blank = OUT / "newmap_159_before.aoe2scenario"
    blank.write_bytes(blank_scenario_bytes(120))
    new_map = load_map_and_units_from_bytes(blank.read_bytes(), blank)
    _KEEP_ALIVE.append(new_map)
    _paste(_load(EXAMPLES / "R4_LeLoi_4.aoe2scenario"), new_map, OUT / "newmap_159_after.aoe2scenario")

    atilla_path = EXAMPLES / "atilla_1_scn_resaved.aoe2scenario"
    write_scenario(_load(atilla_path), OUT / "atilla_158_before.aoe2scenario", backup=False)
    _paste(_load(EXAMPLES / "8tp3w9j.aoe2scenario"), _load(atilla_path), OUT / "atilla_158_after.aoe2scenario")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
