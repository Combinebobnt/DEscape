#!/usr/bin/env python3
"""Writes probe scenarios asking the real AoE2:DE game what it does with an
object whose box crosses the map edge when it loads a scenario (GH #152).

"before" is a zero-edit write_scenario() of the source; each after_* file adds
its probes with UnitEditModel.add() then write_scenario(), the code Place
Unit runs. Every coordinate derives from the loaded map's size.

- after_units: span-1 objects with a clearance box (Villager, Knight, Relic,
  Battering Ram, Lion, Flag A, Oak, Gold Mine, Watch Tower) at offsets 0.0,
  0.1, clearance - 0.05 and clearance from the x = 0 or y = 0 edge, Villager
  and Lion on both and also at 0.5; no-box objects (Hawk, Stump) at 0.0 and
  0.05; the same kinds at W - 0.001, W - 0.01 and W - clearance on the far
  edges; a Villager at (0, 0) and at (W - 0.001, H - 0.001).
- after_buildings: Barracks and Farm 0.5, 1.0 and 1.5 from x = 0, a Wonder
  0.5, 1.5 and 2.5 from y = 0, a Barracks at W - 0.5 and W - 1.5, a Mill at
  1.0 (the control), a Mountain at 2.5 and 3.0, and the corner control, a
  Villager at (0.5, 0.5), kept 2 tiles clear of the (0, 0) one by its file.
- after_offmap: a Villager at x = -0.5 and one at x = W + 0.5.

probes.json records each probe's file, label, owner, const, intended (x, y)
and reference_id. `--readback FILE...` compares a file's objects against its
probes (matched by reference_id, else the nearest unused object of the same
owner and const within 3 tiles) and prints STAY, MOVED or MISSING per probe,
the objects no probe matched, and a per-kind summary. It only reads, so FILE
may be anywhere, a mods/subscribed/ path included; its probe set is picked by
"units", "buildings" or "offmap" in its name.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import object_catalog
from descape.edit_history import EditHistory
from descape.scenario_io import BLANK_TEMPLATE_PATH, is_workshop_subscribed, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import tile_span
from descape.unit_model import UnitEditModel

OUT = ROOT / "build" / "gh152_pair"
GAIA, PLAYER = 0, 1
VILLAGER, KNIGHT, RELIC, RAM, LION, FLAG = 83, 38, 285, 1258, 1029, 600
OAK, GOLD_MINE, WATCH_TOWER, HAWK, STUMP = 349, 66, 79, 96, 809
BARRACKS, FARM, WONDER, MILL, MOUNTAIN = 12, 50, 276, 68, 744
GAIA_CONSTS = {RELIC, LION, OAK, GOLD_MINE, HAWK, STUMP, MOUNTAIN}
# Villager and Lion sit on both edges so the two axes compare directly.
BOXED = {"x": (VILLAGER, KNIGHT, RELIC, RAM, LION), "y": (VILLAGER, LION, FLAG, OAK, GOLD_MINE, WATCH_TOWER)}
NO_BOX = {"x": (HAWK,), "y": (STUMP,)}
CONTROLS_AT_HALF = (VILLAGER, LION)
CORNER_CLEAR = 8  # tiles kept empty at each corner
SPAN1_STEP = 4  # tiles between consecutive span-1 probes along an edge
MULTI_GAP = 3  # tiles between consecutive multi-tile footprints
MIN_BOX_GAP = 2.0  # asserted between any two probe boxes in one file
STAY_TOL = 1e-4  # float32 rounding
MATCH_RADIUS = 3.0
SETS = ("units", "buildings", "offmap")


@dataclass(frozen=True)
class Probe:
    set: str
    label: str
    player: int
    unit_const: int
    x: float
    y: float
    axis: str | None = None  # the coordinate an edge probe tests, "x" or "y"
    side: str | None = None  # "low" (the 0 edge) or "high" (the W or H edge)
    offset: float | None = None  # the anchor's distance from that edge


def _clearance(const: int) -> tuple[float, float]:
    box = object_catalog.dat_objects().get(const, {}).get("clearance")
    return (float(box[0]), float(box[1])) if box else (0.0, 0.0)


def _half_box(const: int) -> tuple[float, float]:
    """The larger of the clearance box and the footprint, for the spacing check."""
    span = tile_span(const, (1, 1))
    cx, cy = _clearance(const)
    if span == (1, 1):
        return cx, cy
    return max(cx, span[0] / 2), max(cy, span[1] / 2)


def _owner(const: int) -> int:
    return GAIA if const in GAIA_CONSTS else PLAYER


def _name(const: int) -> str:
    return object_catalog.combined_object_name(const)


def _edge_row(set_name, rows, axis, side, size_cross, size_along) -> list[Probe]:
    """`rows` [(const, offset)] along one edge, from CORNER_CLEAR in: span-1
    probes SPAN1_STEP apart on tile centres, multi-tile ones MULTI_GAP tiles
    apart with each along-edge anchor on the game's grid."""
    probes = []
    along = prev_half = None
    for const, offset in rows:
        span = tile_span(const, (1, 1))
        half = (span[1] if axis == "x" else span[0]) / 2
        if along is None:
            along = CORNER_CLEAR + max(half, 0.5)
        else:
            along += SPAN1_STEP if span == (1, 1) and prev_half == 0.5 else prev_half + half + MULTI_GAP
        if along + half > size_along - CORNER_CLEAR:
            raise ValueError(f"the {axis} {side} edge row does not fit a {size_along}-tile map")
        prev_half = max(half, 0.5)
        coord = offset if side == "low" else size_cross - offset
        x, y = (coord, along) if axis == "x" else (along, coord)
        edge = f"{axis}={'0' if side == 'low' else ('W' if axis == 'x' else 'H')}"
        label = f"{_name(const)} {edge} offset {offset:g}"
        probes.append(Probe(set_name, label, _owner(const), const, x, y, axis, side, offset))
    return probes


def _near_offsets(const: int, axis: str) -> list[float]:
    c = _clearance(const)[0 if axis == "x" else 1]
    if c == 0:
        return [0.0, 0.05]
    offsets = {0.0, 0.1, round(c - 0.05, 6), c}
    if const in CONTROLS_AT_HALF:
        offsets.add(0.5)
    return sorted(o for o in offsets if o >= 0)


def _far_offsets(const: int, axis: str) -> list[float]:
    c = _clearance(const)[0 if axis == "x" else 1]
    # A no-box kind's W - clearance is W itself, an off-map anchor: after_offmap's job.
    return sorted({0.001, 0.01, c} - {0.0})


def layout(width: int, height: int) -> list[Probe]:
    probes: list[Probe] = []
    for axis in ("x", "y"):
        cross, along = (width, height) if axis == "x" else (height, width)
        kinds = BOXED[axis] + NO_BOX[axis]
        near = [(k, o) for k in kinds for o in _near_offsets(k, axis)]
        far = [(k, o) for k in kinds for o in _far_offsets(k, axis)]
        probes += _edge_row("units", near, axis, "low", cross, along)
        probes += _edge_row("units", far, axis, "high", cross, along)
    probes += [
        Probe("units", "Villager corner (0, 0)", PLAYER, VILLAGER, 0.0, 0.0),
        Probe("units", "Villager corner (W - 0.001, H - 0.001)", PLAYER, VILLAGER, width - 0.001, height - 0.001),
    ]

    x_low = [(BARRACKS, o) for o in (0.5, 1.0, 1.5)] + [(FARM, o) for o in (0.5, 1.0, 1.5)] + [(MILL, 1.0)]
    y_low = [(WONDER, o) for o in (0.5, 1.5, 2.5)] + [(MOUNTAIN, 2.5), (MOUNTAIN, 3.0)]
    probes += _edge_row("buildings", x_low, "x", "low", width, height)
    probes += _edge_row("buildings", y_low, "y", "low", height, width)
    probes += _edge_row("buildings", [(BARRACKS, 0.5), (BARRACKS, 1.5)], "x", "high", width, height)
    probes.append(Probe("buildings", "Villager corner (0.5, 0.5) control", PLAYER, VILLAGER, 0.5, 0.5))

    mid = height // 2 + 0.5
    probes += [
        Probe("offmap", "Villager x = -0.5", PLAYER, VILLAGER, -0.5, mid, "x", "low", -0.5),
        Probe("offmap", "Villager x = W + 0.5", PLAYER, VILLAGER, width + 0.5, mid, "x", "high", -0.5),
    ]
    _check_spacing(probes)
    return probes


def _check_spacing(probes: list[Probe]) -> None:
    for i, a in enumerate(probes):
        ax, ay = _half_box(a.unit_const)
        for b in probes[i + 1 :]:
            if a.set != b.set:
                continue
            bx, by = _half_box(b.unit_const)
            gap = max(abs(a.x - b.x) - ax - bx, abs(a.y - b.y) - ay - by)
            if gap < MIN_BOX_GAP:
                raise AssertionError(f"{a.label!r} and {b.label!r} boxes are {gap:.2f} tiles apart, under {MIN_BOX_GAP}")
    labels = [(p.set, p.label) for p in probes]
    if len(set(labels)) != len(labels):
        raise AssertionError("probe labels repeat within a file")


EXPECTED = """Expected reading if the game moves an object whose box crosses the edge:
  after_units: a boxed probe at an offset under its clearance MOVED inward until
      its box touches the edge (offset after = clearance, or clearance + 0.01);
      at its clearance or at 0.5 it STAYs. Hawk and Stump STAY at 0.0 and 0.05.
      The (0, 0) Villager lands near (0.2, 0.2); far-edge probes mirror the near ones.
  after_buildings: Barracks and Farm at 0.5 and 1.0 MOVED to 1.5, at 1.5 STAY;
      the Wonder at 0.5 and 1.5 MOVED to 2.5, at 2.5 STAYs; the far Barracks at
      W - 0.5 MOVED to W - 1.5; the Mill at 1.0 STAYs. The Mountain at 2.5 STAYs
      if the game reads its 4x4 box, MOVED if its 6x6 footprint; at 3.0 it STAYs.
  after_offmap: each Villager is MISSING (dropped) or pulled onto the map.
A no-box probe moving, or one at clearance or 0.5, contradicts the hypothesis."""


def write_pair(source: Path, out: Path) -> int:
    if is_workshop_subscribed(out):
        raise WriteBlockedError(f"Refusing to write into a subscribed-mods (mods/subscribed/) folder: {out}")
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 1
    loaded = load_map_and_units(source)
    mm = loaded.map_manager
    probes = layout(mm.map_width, mm.map_height)
    out.mkdir(parents=True, exist_ok=True)
    before = out / "before.aoe2scenario"
    write_scenario(loaded, before, backup=False)
    print(f"wrote {before}")

    records = []
    for set_name in SETS:
        loaded = load_map_and_units(source)
        model = UnitEditModel(loaded)
        model.begin_unit_edit([GAIA, PLAYER])
        mine = [p for p in probes if p.set == set_name]
        added = [model.add(p.player, p.unit_const, p.x, p.y) for p in mine]
        model.commit_unit_edit(f"Place GH #152 {set_name} probes", EditHistory())
        path = out / f"after_{set_name}.aoe2scenario"
        write_scenario(loaded, path, backup=False, units=model)
        print(f"wrote {path} ({len(mine)} probes)")
        records += [{**asdict(p), "file": path.name, "reference_id": u.reference_id} for p, u in zip(mine, added, strict=True)]
    manifest = out / "probes.json"
    manifest.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {manifest}\n")
    print(f"Source {source.name}, {mm.map_width}x{mm.map_height}.\n")
    print(EXPECTED)
    return 0


def _set_for(path: Path) -> str:
    stem = path.stem.lower()
    for set_name in ("offmap", "buildings", "units"):
        if set_name in stem:
            return set_name
    raise SystemExit(f"{path}: name holds none of {', '.join(SETS)}, so its probe set is unknown")


def _references(manifest: Path) -> dict[tuple[str, str], int]:
    if not manifest.is_file():
        print(f"(no {manifest}: matching by position only)")
        return {}
    return {(r["set"], r["label"]): r["reference_id"] for r in json.loads(manifest.read_text(encoding="utf-8"))}


def _offset_after(p: Probe, x: float, y: float, width: int, height: int) -> float | None:
    if p.axis is None:
        return None
    coord, size = (x, width) if p.axis == "x" else (y, height)
    return coord if p.side == "low" else size - coord


def readback(path: Path, manifest: Path) -> int:
    set_name = _set_for(path)
    loaded = load_map_and_units(path)
    width, height = loaded.map_manager.map_width, loaded.map_manager.map_height
    probes = [p for p in layout(width, height) if p.set == set_name]
    refs = _references(manifest)
    units = [(player, u) for player, us in enumerate(loaded.unit_manager.units) for u in us]
    used: set[int] = set()
    rows = []
    for p in probes:
        ref = refs.get((p.set, p.label))
        same = [(pl, u) for pl, u in units if pl == p.player and u.unit_const == p.unit_const and id(u) not in used]
        hit = next(
            (u for _pl, u in same if ref is not None and u.reference_id == ref and math.hypot(u.x - p.x, u.y - p.y) <= MATCH_RADIUS),
            None,
        )
        if hit is None:
            near = [u for _pl, u in same if math.hypot(u.x - p.x, u.y - p.y) <= MATCH_RADIUS]
            hit = min(near, key=lambda u: math.hypot(u.x - p.x, u.y - p.y), default=None)
        if hit is not None:
            used.add(id(hit))
        rows.append((p, hit))

    print(f"Read back {path} ({set_name}, {width}x{height}):")
    print(f"  {'probe':44s} {'own':>3s} {'const':>5s} {'intended':>20s} {'stored':>20s} {'delta':>18s} verdict")
    counts = {"STAY": 0, "MOVED": 0, "MISSING": 0}
    verdicts = []
    for p, u in rows:
        intended = f"({p.x:.3f}, {p.y:.3f})"
        if u is None:
            verdict, stored, delta = "MISSING", "", ""
        else:
            dx, dy = u.x - p.x, u.y - p.y
            verdict = "STAY" if abs(dx) <= STAY_TOL and abs(dy) <= STAY_TOL else "MOVED"
            stored, delta = f"({u.x:.4f}, {u.y:.4f})", f"({dx:+.4f}, {dy:+.4f})"
        counts[verdict] += 1
        verdicts.append(verdict)
        print(f"  {p.label:44s} {p.player:3d} {p.unit_const:5d} {intended:>20s} {stored:>20s} {delta:>18s} {verdict}")

    unmatched = [(pl, u) for pl, u in units if id(u) not in used]
    print(f"\n{counts['STAY']} STAY, {counts['MOVED']} MOVED, {counts['MISSING']} MISSING, {len(unmatched)} unmatched.")
    for pl, u in unmatched:
        print(f"  unmatched: owner {pl} const {u.unit_const} {_name(u.unit_const)} id {u.reference_id} at ({u.x:.4f}, {u.y:.4f})")

    print("\nPer kind (edge probes; offsets from the tested edge):")
    for const in dict.fromkeys(p.unit_const for p in probes):
        mine = [(p, u, v) for (p, u), v in zip(rows, verdicts, strict=True) if p.unit_const == const and p.offset is not None]
        if not mine:
            continue
        moved = [p.offset for p, _u, v in mine if v != "STAY"]
        stayed = [p.offset for p, _u, v in mine if v == "STAY"]
        landed = [f"{p.offset:g}->{_offset_after(p, u.x, u.y, width, height):.4f}" for p, u, v in mine if v == "MOVED"]
        print(
            f"  {_name(const):22s} clearance {_clearance(const)[0]:g}, span {tile_span(const, (1, 1))}: "
            f"largest moved {max(moved, default=None)}, smallest stayed {min(stayed, default=None)}"
            + (f", landed {' '.join(landed)}" if landed else "")
        )
    print("\n" + EXPECTED + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=BLANK_TEMPLATE_PATH, help="The scenario to add the probes to")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory")
    parser.add_argument("--readback", type=Path, nargs="+", help="Only compare each FILE against its probes")
    parser.add_argument("--probes", type=Path, default=OUT / "probes.json", help="The reference_id manifest --readback reads")
    args = parser.parse_args()
    if args.readback:
        for path in args.readback:
            readback(path, args.probes)
        return 0
    return write_pair(args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
