#!/usr/bin/env python3
"""Writes build/task222/before.aoe2scenario and after.aoe2scenario, a pair
asking the real AoE2:DE editor whether each particle waterfall faces as
DEscape draws it once Cycle Variant has turned it (GH #330).

Through the real write path, from examples/R4_LeLoi_4.aoe2scenario (the only
corpus map placing waterfalls), loaded through batch_api.load():

- "before" is a zero-edit write_scenario() of the source.
- "after" sets the lone WFALL 896 at (219, 37) to variant 2 and the WFALL 1635
  ending the y 129 row (x 186) to variant 0 with UnitEditModel.set_variant(),
  adds one GAIA SWFALL 2598 beside it with UnitEditModel.add() and sets it to
  variant 1 the same way, then write_scenario().

Every variant goes in as the literal integer the game stored on the four
corpus placements. The after file is read back and the three units' rotations
checked.

    tools/gen_gh330_pair.py [--source <file>] [--out <dir>] [--swfall-at X Y]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import batch_api
from descape.edit_history import EditHistory
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "R4_LeLoi_4.aoe2scenario"
OUT = ROOT / "build" / "task222"
BEFORE_NAME, AFTER_NAME = "before.aoe2scenario", "after.aoe2scenario"
WFALL, WFALL_BACKGROUND, SWFALL = 896, 1635, 2598
GAIA = 0
# (const, x, y) of each stored waterfall to turn, and the variant it gets.
TURNS = (((WFALL, 219.0, 37.0), 2.0), ((WFALL_BACKGROUND, 186.0, 129.0), 0.0))
# Four tiles west of 1635 along its row, so the two arts do not overlap.
SWFALL_AT = (182.0, 129.0)
SWFALL_VARIANT = 1.0


def _find(loaded, const: int, x: float, y: float):
    hits = [u for u in loaded.unit_manager.units[GAIA] if u.unit_const == const and (u.x, u.y) == (x, y)]
    if len(hits) != 1:
        raise SystemExit(f"expected one GAIA {const} at ({x:g}, {y:g}), found {len(hits)}")
    return hits[0]


def build(source: Path, swfall_at: tuple[float, float]):
    """(loaded, unit model, [(label, unit, variant)]) with every edit applied in memory."""
    loaded = batch_api.load(source)
    units = UnitEditModel(loaded)
    targets = [(f"{const} at ({x:g}, {y:g})", _find(loaded, const, x, y), variant) for (const, x, y), variant in TURNS]
    units.begin_unit_edit([GAIA])
    added = units.add(GAIA, SWFALL, *swfall_at)
    units.commit_unit_edit("Place the SWFALL", EditHistory())
    targets.append((f"{SWFALL} at ({swfall_at[0]:g}, {swfall_at[1]:g}), added", added, SWFALL_VARIANT))
    units.begin_unit_edit([GAIA], fields_only=True)
    for _label, unit, variant in targets:
        units.set_variant(unit, variant)
    units.commit_unit_edit("Turn the waterfalls", EditHistory())
    return loaded, units, targets


def _check(path: Path, targets) -> list[str]:
    reloaded = batch_api.load(path)
    problems = []
    for label, unit, variant in targets:
        hits = [
            u for u in reloaded.unit_manager.units[GAIA]
            if u.unit_const == unit.unit_const and (float(u.x), float(u.y)) == (float(unit.x), float(unit.y))
        ]
        if [float(u.rotation) for u in hits] != [variant]:
            problems.append(f"{label}: read back {[float(u.rotation) for u in hits]}, wrote {variant:g}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="Scenario holding R4_LeLoi_4's waterfalls")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--swfall-at", type=float, nargs=2, metavar=("X", "Y"), default=SWFALL_AT)
    args = parser.parse_args()
    if is_workshop_subscribed(args.out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    before = args.out / BEFORE_NAME
    write_scenario(batch_api.load(args.source), before, backup=False)
    print(f"wrote {before} (no edits)")
    loaded, units, targets = build(args.source, tuple(args.swfall_at))
    after = args.out / AFTER_NAME
    write_scenario(loaded, after, units=units, backup=False)
    print(f"wrote {after}")
    for label, unit, variant in targets:
        print(f"  GAIA {label}: rotation {variant:g}, initial_animation_frame {unit.initial_animation_frame}")
    problems = _check(after, targets)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
