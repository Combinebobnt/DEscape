#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair for confirming a map resize
(TASK-107, GH #45) in the real AoE2:DE editor, the half nothing in-session
can drive.

Deliberately asymmetric: a 144x144 file shrunk to 120x120 anchored
BOTTOM_RIGHT, so the old map's far corner stays put and 24 tiles are cut
off the x = 0 and y = 0 edges (the two edges that meet at the west tip).
A silently mirrored offset would cut the other two edges instead, which is
visible rather than merely plausible.

Both files go through the real write path: the "before" one is a zero-edit
write_scenario(), the "after" one scenario_resize.resize_scenario() then
write_scenario() with its unit and trigger models, the same code File >
Resize Map… runs. The source's civs are all base-game (GOTCHAS), so a tester
without DLC can open the pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import library_compat, object_catalog, trigger_geometry
from descape.render import NON_BUILDING_SPAN
from descape.scenario_io import FORBIDDEN_WRITE_MARKER, is_under_compatdata, load_map_and_units, parse_triggers
from descape.scenario_resize import Anchor, resize_scenario
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span

SOURCE = ROOT / "examples" / "atilla_1_scn_resaved.aoe2scenario"
NEW_SIZE = 120
ANCHOR = Anchor.BOTTOM_RIGHT


def _landmarks(loaded, limit: int = 6) -> list:
    """The largest-footprint units, one per player first, as checkable spots."""
    best: dict[int, object] = {}
    for player, units in enumerate(loaded.unit_manager.units):
        spans = [(tile_span(u.unit_const, NON_BUILDING_SPAN), u) for u in units]
        spans = [(sx * sy, u) for (sx, sy), u in spans if sx * sy >= 4]
        if spans:
            best[player] = max(spans, key=lambda pair: pair[0])[1]
    return list(best.values())[:limit]


def _trigger_areas(loaded, limit: int = 4) -> list[tuple[str, tuple]]:
    manager = parse_triggers(loaded)
    if manager is None:
        return []
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    out = []
    for index, trigger in enumerate(manager.triggers):
        for shape in trigger_geometry.shapes_for_trigger(trigger, vocabulary, trigger_index=index):
            if shape.shape == trigger_geometry.SHAPE_AREA and shape.coords != (0, 0, 0, 0):
                out.append((f"trigger {index} '{trigger.name}', {shape.entry_kind} {shape.entry_index}", shape.coords))
                break
        if len(out) >= limit:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE, help="A square scenario to shrink")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "resize_pair", help="Output directory")
    args = parser.parse_args()
    if is_under_compatdata(args.out):
        raise WriteBlockedError(f"Refusing to write under a Proton {FORBIDDEN_WRITE_MARKER}/ folder: {args.out}")
    if not args.source.is_file():
        print(f"missing source: {args.source}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    source = load_map_and_units(args.source)
    old_size = source.map_manager.map_width
    before_path = args.out / f"before_{old_size}x{old_size}.aoe2scenario"
    write_scenario(source, before_path, backup=False)
    landmarks = [(object_catalog.combined_object_name(u.unit_const), u.reference_id, u.x, u.y) for u in _landmarks(source)]
    areas = _trigger_areas(source)

    source = load_map_and_units(args.source)
    result = resize_scenario(source, NEW_SIZE, NEW_SIZE, ANCHOR)
    after_path = args.out / f"after_{NEW_SIZE}x{NEW_SIZE}_anchored_bottom_right.aoe2scenario"
    write_scenario(result.loaded, after_path, backup=False, units=result.unit_edits, triggers=result.trigger_edits)
    after = load_map_and_units(after_path)
    plan = result.plan
    by_ref = {u.reference_id: u for units in after.unit_manager.units for u in units}
    after_areas = dict(_trigger_areas(after, limit=len(areas) or 1))

    print(f"wrote {before_path}")
    print(f"wrote {after_path}")
    print()
    print(f"{plan.summary()}")
    print(f"(dx, dy) = ({plan.dx}, {plan.dy}): old tile (x, y) is new tile (x{plan.dx:+d}, y{plan.dy:+d}).")
    print()
    print("What to check in the AoE2:DE editor:")
    print(f"  1. Open the 'after' file. The Map tab reports {NEW_SIZE}x{NEW_SIZE}; the 'before' file {old_size}x{old_size}.")
    print("  2. The cut edges: compare the two minimaps. The 'after' map is the 'before' one with a")
    print(f"     {-plan.dx}-tile strip removed along BOTH edges that meet at the WEST tip (x = 0 and y = 0).")
    print("     The east tip's corner is unchanged. Strips gone from the north-east/south-east edges")
    print("     instead mean the offset is mirrored.")
    print("  3. Landmarks: each should sit on the same ground as in the 'before' file.")
    for name, ref, x, y in landmarks:
        unit = by_ref.get(ref)
        where = f"now ({unit.x}, {unit.y})" if unit is not None else "deleted (was in the cut strip)"
        print(f"       {name} #{ref}: was ({x}, {y}), {where}")
    print("  4. Elevation: hills near the kept edges look the same, and the editor raises no error on open.")
    print("  5. Triggers: these areas should still cover the same ground (Triggers tab, select the")
    print("     effect/condition, the area highlights on the map):")
    for label, coords in areas:
        print(f"       {label}: was {coords}, now {after_areas.get(label, 'see file')}")
    print("  6. Save the 'after' file from the in-game editor under a new name, reopen it, and repeat 1-5.")
    print(f"     Then test-play it briefly: {plan.units_moved} objects should be present. A player whose")
    print("     town was in the cut strip may lose at once, and triggers naming a deleted object may")
    print("     misfire; both are expected of a shrink this deep, not a resize fault.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
