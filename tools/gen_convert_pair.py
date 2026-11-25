#!/usr/bin/env python3
"""Writes the scenario kit for confirming File > Convert to Version 1.59
(GH #307) in the real AoE2:DE editor, the half nothing in-session can drive.

Three files, all through the real write path:

- before_158: a zero-edit write_scenario() of the 1.58 source;
- after_159: scenario_convert.convert_scenario() then write_scenario(), the
  same code the File action runs before Save;
- after_159_edited: the converted document, then through the real edit
  models one landmark set to Capture Flag "Convert Multiple Times" and one
  new trigger "TASK-346 check" with an Object Visible Multiplayer condition,
  so the editor shows both 1.59 fields on a converted file.

README.txt beside them lists what to check. Its facts come from the source
load, before converting, and the edits run on the converted document with no
load in between: the library's version state is process-global, so a load
between the conversion and an edit would give the edit the wrong defaults.

The source loads through batch_api.load(), which activates its own game's
data (GH #67). The default source's civs are all base-game, so a tester
without DLC can open the kit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api, object_catalog
from descape.render_units import NON_BUILDING_SPAN
from descape.scenario_convert import convert_scenario
from descape.scenario_io import is_workshop_subscribed, load_map_and_units, parse_triggers
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
CHECK_TRIGGER = "TASK-346 check"
CAPTURE_FLAG = 2  # Convert Multiple Times
OBJECT_VISIBLE_MULTIPLAYER = 27


def _landmarks(loaded) -> list:
    """The largest-footprint unit per player, as checkable spots."""
    best = {}
    for player, units in enumerate(loaded.unit_manager.units):
        spans = [(sx * sy, u) for u in units for sx, sy in [tile_span(u.unit_const, NON_BUILDING_SPAN)] if sx * sy >= 4]
        if spans:
            best[player] = max(spans, key=lambda pair: pair[0])[1]
    return list(best.items())


def _trigger_facts(manager, limit: int = 3) -> list[tuple[int, str, int, int]]:
    """(index, name, conditions, effects) of the first triggers with both."""
    picked = [(i, t.name, len(t.conditions), len(t.effects)) for i, t in enumerate(manager.triggers) if t.conditions and t.effects]
    return picked[:limit]


def _edit(result) -> tuple:
    """Capture Flag on the first landmark and the check trigger, through the real models. Returns the landmark."""
    loaded = result.loaded
    units = UnitEditModel(loaded)
    landmark = next(u for player, u in _landmarks(loaded) if player > 0)  # a player's, not GAIA's
    units.set_capture_flag(landmark, CAPTURE_FLAG)
    triggers = TriggerEditModel(loaded)
    triggers.structural_edit(lambda m: m.add_trigger(CHECK_TRIGGER))
    index = triggers.trigger_count - 1
    trigger = triggers.manager().triggers[index]
    trigger._add_condition(OBJECT_VISIBLE_MULTIPLAYER)
    trigger.conditions[0].unit_object = landmark.reference_id
    trigger.conditions[0].source_player = 2
    triggers.mark_dirty(index)
    return landmark, units, triggers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="A 1.58 scenario to convert")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "task346", help="Output directory")
    args = parser.parse_args()
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    if not args.source.is_file():
        print(f"missing source: {args.source}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    source = batch_api.load(args.source)
    before_path = args.out / "before_158.aoe2scenario"
    write_scenario(source, before_path, backup=False)
    manager = parse_triggers(source)
    trigger_count = len(manager.triggers) if manager is not None else 0
    triggers = _trigger_facts(manager) if manager is not None else []
    landmarks = [(p, object_catalog.combined_object_name(u.unit_const), u.reference_id, u.x, u.y) for p, u in _landmarks(source)]

    result = convert_scenario(source)
    after_path = args.out / "after_159.aoe2scenario"
    write_scenario(result.loaded, after_path, backup=False)
    landmark, unit_edits, trigger_edits = _edit(result)
    edited_path = args.out / "after_159_edited.aoe2scenario"
    write_scenario(result.loaded, edited_path, backup=False, units=unit_edits, triggers=trigger_edits)
    for path in (after_path, edited_path):
        back = load_map_and_units(path)
        if back.scenario_version != result.target_version or parse_triggers(back) is None:
            raise WriteBlockedError(f"{path} did not reload as {result.target_version} with readable triggers")

    lines = [
        f"GH #307 kit: {args.source.name} ({result.source_version}) converted to {result.target_version}.",
        "",
        f"before_158.aoe2scenario        the source, a zero-edit write ({result.source_version})",
        f"after_159.aoe2scenario         converted, nothing else changed ({result.target_version})",
        "after_159_edited.aoe2scenario  converted, plus one Capture Flag and one new trigger",
        "",
        f"Converted {result.units} objects and {result.conditions} conditions.",
        "",
        "after_159, in the 1.59 editor:",
        "  1. It opens with no error. These buildings stand on these tiles (x, y), as in before_158:",
    ]
    lines += [f"       player {p}: {name} #{ref} at ({x:g}, {y:g})" for p, name, ref, x, y in landmarks]
    lines.append(f"  2. The Triggers tab lists {trigger_count} triggers. Three of them:")
    lines += [f"       #{i} '{name}': conditions {c}, effects {e}" for i, name, c, e in triggers]
    lines += [
        "  3. Test the scenario from the editor: it starts.",
        "",
        "after_159_edited:",
        (
            f"  4. {object_catalog.combined_object_name(landmark.unit_const)} #{landmark.reference_id} at "
            f"({landmark.x:g}, {landmark.y:g}) shows Capture Flag 'Convert Multiple Times'."
        ),
        f"  5. The last trigger, '{CHECK_TRIGGER}', has one Object Visible Multiplayer condition naming that object.",
    ]
    readme = args.out / "README.txt"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (before_path, after_path, edited_path, readme):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
