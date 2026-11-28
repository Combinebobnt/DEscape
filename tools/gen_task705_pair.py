#!/usr/bin/env python3
"""Writes a before/after .aoe2scenario pair asking the real AoE2:DE editor
whether DE bridge pieces (1840-2041) read right before and after a turn
(TASK-705): each run one continuous bridge, its end ramps dropping to land.

Both files go through the real write path. "before" is the source with three
straight runs along y added over painted water, each crossing it (batch_api.load(), UnitEditModel.add() and raw
terrain, then write_scenario()), each listed from low y to high y:

- ab: 1840 frame 1, 1844 frame 2, 1840 frame 0
- cd: 1993 frame 1, 1996 frame 0, 1999 frame 2, 1993 frame 0
- wood: 2039 frame 0, 2039 frame 1

"after" is that document with two regions turned in place, as the editor's
Rotate Group on a Select region does (region_transform.plan_in_place(), then
the setters gen_task256_pair's _write() calls, set_variant() for a frame):
the ab and cd runs by Rotate Group anticlockwise (r3), the wood run clockwise
(r), the elements under which every end's frame index changes. Every piece
then runs along x. The water turns with its region.

Run from the repo root; output goes to build/task705/ (gitignored).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gen_task256_pair import _check_on_map, _first_units, _write, find_area

from descape import batch_api, bridge_piece_frames, region_clipboard, region_transform, render_units
from descape.edit_history import EditHistory
from descape.mirror_tools import walls_are_radian
from descape.scenario_io import is_workshop_subscribed
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.unit_model import UnitEditModel

SOURCE = ROOT / "examples" / "ring75_v0_scx_resaved.aoe2scenario"
OUT = ROOT / "build" / "task705"
PLAYER = 0
WATER = 1
# (const, frame) per run, low y first; every piece is (1, 2).
RUNS = {
    "ab": [(1840, 1), (1844, 2), (1840, 0)],
    "cd": [(1993, 1), (1996, 0), (1999, 2), (1993, 0)],
    "wood": [(2039, 0), (2039, 1)],
}
# Region 1 (ab at local column 2, cd at 6) is 10 x 10, region 2 (wood at column 2) 6 x 6, side by side.
REGION_1, REGION_2, GAP = 10, 6, 4


def _lay_run(mm, model, pieces, x: int, y0: int) -> list:
    """Adds the run's pieces along column x from y0, and paints water under the
    tiles between its two ends, three tiles wide."""
    units = []
    for k, (const, frame) in enumerate(pieces):
        units.append(model.add(PLAYER, const, *render_units.span_anchor(x, y0 + 2 * k, 1, 2), rotation=float(frame)))
    # Under the middles, or under the joint of a two-piece run, so every run crosses water.
    for y in range(y0 + 2, y0 + 2 * len(pieces) - 2) or range(y0 + 1, y0 + 3):
        for tx in (x - 1, x, x + 1):
            mm.terrain[y * mm.map_width + tx].terrain_id = WATER
    return units


def _turn(mm, loaded, units, rect, element, file_radian):
    tiles = dataclasses.replace(region_clipboard.copy_region(mm, loaded.unit_manager, *rect), units=())
    plan = region_transform.plan_in_place(_first_units(units), element, rect, file_radian, tiles=tiles)
    _check_on_map(plan, mm)
    return plan


def _paste_tiles(mm, loaded, plan) -> None:
    region_clipboard.paste_terrain(mm, plan.block, plan.rect[0], plan.rect[1], carry_layer=loaded.terrain_has_layer)


def _check_run(name: str, units) -> list[str]:
    """Each end's inward vector pointing at the run's middle."""
    problems = []
    centre_x = sum(u.x for u in units) / len(units)
    centre_y = sum(u.y for u in units) / len(units)
    for u in units:
        kind, *rest = bridge_piece_frames.frames(u.unit_const)[int(u.rotation)]
        if kind == bridge_piece_frames.END:
            (vx, vy) = rest[0]
            if vx * (centre_x - u.x) + vy * (centre_y - u.y) <= 0:
                problems.append(f"{name}: end {u.unit_const} frame {int(u.rotation)} at ({u.x}, {u.y}) faces away")
    return problems


def write_pair(source: Path, out: Path) -> int:
    from descape.terrain_palette import tile_span

    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    before_path, after_path = out / "before.aoe2scenario", out / "after.aoe2scenario"

    loaded = batch_api.load(source)
    mm = loaded.map_manager
    ax, ay = find_area(loaded)
    r1 = (ax, ay, ax + REGION_1, ay + REGION_1)
    r2 = (ax + REGION_1 + GAP, ay, ax + REGION_1 + GAP + REGION_2, ay + REGION_2)
    model = UnitEditModel(loaded)
    history = EditHistory()
    model.begin_unit_edit([PLAYER])
    runs = {
        "ab": _lay_run(mm, model, RUNS["ab"], ax + 2, ay + 1),
        "cd": _lay_run(mm, model, RUNS["cd"], ax + 6, ay + 1),
        "wood": _lay_run(mm, model, RUNS["wood"], r2[0] + 2, ay + 1),
    }
    model.commit_unit_edit("TASK-705 pair layout", history)
    write_scenario(loaded, before_path, backup=False, units=model)
    refs = {name: [u.reference_id for u in units] for name, units in runs.items()}
    before = {name: [(u.unit_const, u.x, u.y, u.rotation) for u in units] for name, units in runs.items()}

    file_radian = walls_are_radian(u for units in loaded.unit_manager.units for u in units)
    turn_1 = _turn(mm, loaded, runs["ab"] + runs["cd"], r1, region_transform.ROTATE_CCW, file_radian)
    turn_2 = _turn(mm, loaded, runs["wood"], r2, region_transform.ROTATE_CW, file_radian)
    model.begin_unit_edit([PLAYER], fields_only=True)
    _write(model, runs["ab"] + runs["cd"], turn_1)
    _write(model, runs["wood"], turn_2)
    model.commit_unit_edit("TASK-705 turns", history)
    _paste_tiles(mm, loaded, turn_1)
    _paste_tiles(mm, loaded, turn_2)
    write_scenario(loaded, after_path, backup=False, units=model)

    reread = batch_api.load(after_path)
    by_ref = {u.reference_id: u for units in reread.unit_manager.units for u in units}
    problems = []
    for name, ids in refs.items():
        units = [by_ref.get(i) for i in ids]
        if None in units:
            problems.append(f"{name}: a reference_id is missing after the turn")
            continue
        spans = [tile_span(u.unit_const, render_units.NON_BUILDING_SPAN) for u in units]
        problems += [f"{name}: {u.unit_const} is not (2, 1)" for u, span in zip(units, spans, strict=True) if span != (2, 1)]
        problems += _check_run(name, units)

    print(f"wrote {before_path}\nwrote {after_path}\n")
    print(f"Source {source.name}, all GAIA. Region 1 tiles x {r1[0]}..{r1[2] - 1}, y {r1[1]}..{r1[3] - 1}")
    print(f"(ab and cd runs, turned anticlockwise); region 2 x {r2[0]}..{r2[2] - 1}, y {r2[1]}..{r2[3] - 1} (wood, clockwise).")
    for name, units in runs.items():
        print(f"{name}:")
        for unit, (const, x, y, rotation) in zip(units, before[name], strict=True):
            print(
                f"  ref {unit.reference_id:6d}  {const:4d}/{int(rotation)} at ({x:6.2f}, {y:6.2f})"
                f" -> {unit.unit_const:4d}/{int(unit.rotation)} at ({unit.x:6.2f}, {unit.y:6.2f})"
            )
    if problems:
        print("\nREAD-BACK FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("\nread-back: every piece a (2, 1) const with its reference_id, every end facing its run's middle")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
