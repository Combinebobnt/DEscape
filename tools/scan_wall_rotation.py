#!/usr/bin/env python3
"""Measures the wall/gate connectivity override across examples/, install-free
(no genieutils, no AoE2:DE install -- scenario_io.load_map_and_units only).

A prior attempt at this same measurement was never committed, so its
"~1410 of 2014" figures are not re-derivable -- this script exists so the
numbers below always are. Run it any time render.wall_variant_rotation_overrides()
or unit_sprites.wall_connector_consts() changes.

Per file, reports:
  - the wall-const rotation int/radian split (int counts 0.0 too -- it's
    index 0 under both conventions and so isn't evidence for either),
  - the neighbour-mask -> today's-resolved-index correlation (sanity check
    against the 98.9/99.1/96-100% figures measured on the integer corpus),
  - the changed-tile count if the connectivity override were applied,
  - how many 0.0-rotation walls sit on a non-zero mask whose derived index
    isn't 0 -- the question that decides whether a literal-looking 0.0 can
    be trusted verbatim inside a file that also has real radian-encoded
    walls (see render.wall_variant_rotation_overrides()'s docstring).

The connector tile set (for computing neighbour masks) is built over ALL
units, every player, ignoring any filter -- a wall's real shape does not
depend on which players a viewer happens to be showing.

**--connector-set lets a candidate set be measured without editing
descape/unit_sprites.py** (2026-09-12 generated-replacement plan). It works
by swapping `unit_sprites.wall_connector_consts()` for the duration of the
run, so both this tool's own connector-tile walk and
render.wall_variant_rotation_overrides() (the shipped function under test)
see the same candidate. `hand` is a frozen literal of the pre-2026-09-12
15-const set, kept independent of the module so the before/after comparison
stays reproducible even after the shipped set moves again.

**--agreement** computes the metric every historical figure quotes, exactly:
connector tiles over all units/all players; a file is scored only when every
rotation_variant_eligible wall in it stores a literal variant index (an
integer-only file); for each such wall with a non-zero neighbour mask, it
agrees iff wall_variant_from_neighbours(mask) == variant_index(rotation, 5);
mask-zero (isolated) pieces are excluded and counted separately. Measured
2026-09-12 against this repo's examples/: `shipped` (the generated
_ROTATION_VARIANT_CONSTS + gate_orientation.groups() union) scores 99.00%
(n=5219), against 98.25% (n=5204) for `hand`, the old 15-const set --
`walls-only` alone scores 93.46% (n=5180). The 98.7%/96.9% figures once
quoted for this decision predate this script and are not re-derivable; the
96.9% in particular was a *towers* measurement, never a `class == 39` one.

**--diff-sets A B** prints the override rows that differ between two named
sets. `shipped` against `hand` is **0 rows** by measurement -- the expected
outcome, not a broken run, because render.py skips integer-only files
outright and none of examples/'s radian-encoded files places one of the 90
newly-added consts. `walls-only` against `hand` gives **76** rows across 7
files, which is the self-check that this mode actually works.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import gate_orientation, unit_sprites
from descape.render import (
    stored_rotation,
    unit_occupied_tiles,
    unit_tile_bounds,
    wall_variant_rotation_overrides,
)
from descape.scenario_io import load_map_and_units

_MASK_NAME = {
    0: "0000 (isolated)",
    unit_sprites.WEST | unit_sprites.EAST: "1100 (+-x)",
    unit_sprites.NORTH | unit_sprites.SOUTH: "0011 (+-y)",
}

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "descape" / "object_catalog.json"

# Frozen literal of the pre-2026-09-12 hand-kept set (8 walls, 7 gates picked
# by the 2026-09-02 counterexamples plan's corpus scan) -- kept independent of
# unit_sprites so this candidate stays measurable after the shipped set moves.
_HAND_CONNECTOR_CONSTS = frozenset({
    72, 117, 119, 155, 370, 788, 1062, 2678,  # walls
    64, 88, 95, 659, 667, 793, 797,           # the 7 hand-picked gates
})


def _allgates_connector_consts() -> frozenset[int]:
    """walls + gate_orientation.groups() flattened -- the same rule
    wall_connector_consts() ships today, recomputed independently here so it
    stays a fixed candidate even if the shipped rule changes again later."""
    return unit_sprites._ROTATION_VARIANT_CONSTS | frozenset(
        const for group in gate_orientation.groups().values() for const in group
    )


def _class39_connector_consts() -> frozenset[int]:
    """walls + every raw `class == 39` const (unfiltered by gate_orientation's
    code-regex derivation), so const 1192 -- which groups() drops for having
    no unit_graphic_map.json entry -- is included here instead."""
    objects = json.loads(_CATALOG_PATH.read_text())["objects"]
    graphics = unit_sprites.graphic_map()
    class39 = {
        int(const) for const, entry in objects.items()
        if entry.get("class") == 39 and int(const) in graphics
    }
    return unit_sprites._ROTATION_VARIANT_CONSTS | frozenset(class39)


_CONNECTOR_SET_FACTORIES = {
    "shipped": unit_sprites.wall_connector_consts,
    "walls-only": lambda: unit_sprites._ROTATION_VARIANT_CONSTS,
    "hand": lambda: _HAND_CONNECTOR_CONSTS,
    "allgates": _allgates_connector_consts,
    "class39": _class39_connector_consts,
}


@contextlib.contextmanager
def _connector_set(name: str):
    """Swaps unit_sprites.wall_connector_consts() for `name`'s candidate for
    the duration of the block. Every caller reaches it through the module
    attribute (this tool's own _connector_tiles(), and
    render.wall_variant_rotation_overrides()), so the swap is picked up by
    both without either needing a parameter added."""
    consts = _CONNECTOR_SET_FACTORIES[name]()
    original = unit_sprites.wall_connector_consts
    unit_sprites.wall_connector_consts = lambda: consts
    try:
        yield consts
    finally:
        unit_sprites.wall_connector_consts = original


def _connector_tiles(scenario) -> set[tuple[int, int]]:
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    connector_consts = unit_sprites.wall_connector_consts()
    tiles: set[tuple[int, int]] = set()
    for units in scenario.unit_manager.units:
        for unit in units:
            if unit.unit_const not in connector_consts:
                continue
            occupied = unit_occupied_tiles(unit, tile_w, tile_h)
            if occupied is not None:
                tiles.update(occupied)
    return tiles


def _neighbour_mask(tx: int, ty: int, connector_tiles: set[tuple[int, int]]) -> int:
    mask = 0
    if (tx - 1, ty) in connector_tiles:
        mask |= unit_sprites.WEST
    if (tx + 1, ty) in connector_tiles:
        mask |= unit_sprites.EAST
    if (tx, ty - 1) in connector_tiles:
        mask |= unit_sprites.NORTH
    if (tx, ty + 1) in connector_tiles:
        mask |= unit_sprites.SOUTH
    return mask


def scan_file(path: Path) -> dict:
    """Measures the SHIPPED render.wall_variant_rotation_overrides() against
    this file, so a regression in that function shows up here rather than in
    a second, hand-duplicated copy of its logic. The mask/today-index
    diagnostics below (for --correlation and the 0.0-on-nonzero-mask count)
    are informational only and use their own local mask computation, since
    they need the mask even for units the real function does NOT override."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    graphic_map = unit_sprites.graphic_map()
    connector_tiles = _connector_tiles(scenario)
    overrides = wall_variant_rotation_overrides(scenario)

    candidates = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                continue
            angle_count = int(graphic_map[unit.unit_const]["angle_count"])
            # The shipped seam, not a local copy of it -- same reason the
            # docstring gives for calling wall_variant_rotation_overrides()
            # rather than reimplementing it.
            rotation = stored_rotation(player_id, unit)
            candidates.append((player_id, i, unit, rotation, angle_count))

    int_count = sum(1 for *_, rotation, angle_count in candidates if unit_sprites.is_literal_variant_index(rotation, angle_count))
    radian_count = len(candidates) - int_count

    changed = 0
    mask_to_today_index: dict[int, Counter] = defaultdict(Counter)
    zero_rotation_on_nonzero_mask = 0

    for player_id, i, unit, rotation, angle_count in candidates:
        bounds = unit_tile_bounds(unit, tile_w, tile_h)
        if bounds is None:
            continue
        tx, ty = bounds[0], bounds[2]
        mask = _neighbour_mask(tx, ty, connector_tiles)
        today_index = unit_sprites.variant_index(rotation, angle_count)
        mask_to_today_index[mask][today_index] += 1

        if rotation == 0.0 and mask != 0:
            derived = unit_sprites.wall_variant_from_neighbours(mask)
            if derived is not None and derived != 0:
                zero_rotation_on_nonzero_mask += 1

        override = overrides.get((player_id, i))
        if override is not None and int(override) != today_index:
            changed += 1

    total_walls = len(candidates)
    return {
        "path": path.name,
        "total_walls": total_walls,
        "int_count": int_count,
        "radian_count": radian_count,
        "changed": changed,
        "mask_to_today_index": mask_to_today_index,
        "zero_rotation_on_nonzero_mask": zero_rotation_on_nonzero_mask,
    }


def _agreement_summary(files: list[Path]) -> dict:
    """The metric every historical figure quotes (see module docstring),
    computed under whichever connector set is currently active in
    unit_sprites.wall_connector_consts() -- callers scope this inside
    _connector_set()."""
    ok = 0
    n = 0
    isolated = 0
    mask_to_index: dict[int, Counter] = defaultdict(Counter)

    for path in files:
        try:
            scenario = load_map_and_units(path)
        except Exception:  # noqa: BLE001 -- a scan tool, not production code
            continue
        mm = scenario.map_manager
        tile_w, tile_h = mm.map_width, mm.map_height
        connector_tiles = _connector_tiles(scenario)

        candidates = []
        for player_id, units in enumerate(scenario.unit_manager.units):
            for unit in units:
                if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                    continue
                candidates.append((unit, stored_rotation(player_id, unit)))
        if not candidates:
            continue
        # Only integer-only files are scored: a literal-looking value inside a
        # radian-encoded file carries no more shape information than any other
        # radian value (render.wall_variant_rotation_overrides()'s own
        # docstring), so it isn't ground truth there.
        if not all(unit_sprites.is_literal_variant_index(rotation, 5) for _unit, rotation in candidates):
            continue

        for unit, rotation in candidates:
            bounds = unit_tile_bounds(unit, tile_w, tile_h)
            if bounds is None:
                continue
            tx, ty = bounds[0], bounds[2]
            mask = _neighbour_mask(tx, ty, connector_tiles)
            if mask == 0:
                isolated += 1
                continue
            stored_index = unit_sprites.variant_index(rotation, 5)
            mask_to_index[mask][stored_index] += 1
            n += 1
            if unit_sprites.wall_variant_from_neighbours(mask) == stored_index:
                ok += 1

    return {"ok": ok, "n": n, "isolated": isolated, "mask_to_index": mask_to_index}


def _diff_sets(files: list[Path], set_a: str, set_b: str) -> list[tuple[str, int, int, int, float | None, float | None]]:
    """The override rows that differ between two named connector sets, per
    (file, player_id, i, unit_const, old, new). `old`/`new` are None where one
    set overrides a unit and the other doesn't at all."""
    rows: list[tuple[str, int, int, int, float | None, float | None]] = []
    for path in files:
        try:
            scenario = load_map_and_units(path)
        except Exception:  # noqa: BLE001 -- a scan tool, not production code
            continue
        with _connector_set(set_a):
            overrides_a = wall_variant_rotation_overrides(scenario)
        with _connector_set(set_b):
            overrides_b = wall_variant_rotation_overrides(scenario)
        for player_id, i in sorted(set(overrides_a) | set(overrides_b)):
            old = overrides_a.get((player_id, i))
            new = overrides_b.get((player_id, i))
            if old != new:
                unit_const = scenario.unit_manager.units[player_id][i].unit_const
                rows.append((path.name, player_id, i, unit_const, old, new))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples")
    parser.add_argument("--correlation", action="store_true", help="Print the mask->today-index table across all files")
    parser.add_argument(
        "--connector-set", choices=sorted(_CONNECTOR_SET_FACTORIES), default="shipped",
        help="Which candidate connector set to measure (default: shipped, i.e. today's "
             "unit_sprites.wall_connector_consts()). Applies to the per-file report, "
             "--correlation and --agreement.",
    )
    parser.add_argument(
        "--agreement", action="store_true",
        help="Print the integer-only agreement summary (ok/n/percent, isolated count, "
             "mask->index distribution) for --connector-set, the metric every historical "
             "figure in this decision quotes.",
    )
    parser.add_argument(
        "--diff-sets", nargs=2, metavar=("SET_A", "SET_B"), choices=sorted(_CONNECTOR_SET_FACTORIES),
        help="Print the override rows that differ between two named connector sets. "
             "'shipped hand' is 0 rows by measurement (expected, not a broken run); "
             "'walls-only hand' is 76 rows across 7 files, the self-check that this mode works.",
    )
    args = parser.parse_args()

    files = sorted(
        p for p in args.scenario_dir.iterdir()
        if p.suffix in (".aoe2scenario", ".scx2") or p.name == "play_Test"
    )
    if not files:
        print(f"No scenario files found under {args.scenario_dir}", file=sys.stderr)
        sys.exit(1)

    if args.diff_sets:
        set_a, set_b = args.diff_sets
        rows = _diff_sets(files, set_a, set_b)
        print(f"{set_a} vs {set_b}: {len(rows)} differing override row(s)")
        for file_name, player_id, i, unit_const, old, new in rows:
            print(f"  {file_name} player={player_id} i={i} const={unit_const} {old} -> {new}")
        return

    with _connector_set(args.connector_set):
        if args.agreement:
            summary = _agreement_summary(files)
            ok, n = summary["ok"], summary["n"]
            percent = 100.0 * ok / n if n else 0.0
            print(f"connector-set={args.connector_set}: {ok}/{n} ({percent:.2f}%), isolated={summary['isolated']}")
            print("mask -> stored variant_index() distribution (integer-only files, combined):")
            for mask, counter in sorted(summary["mask_to_index"].items()):
                label = _MASK_NAME.get(mask, format(mask, "04b"))
                total = sum(counter.values())
                dist = ", ".join(f"{idx}:{n}" for idx, n in sorted(counter.items()))
                print(f"  {label:14s} n={total:5d}  {dist}")
            return

        print(f"{'file':45s} {'walls':>6s} {'int':>6s} {'radian':>6s} {'changed':>8s}")
        total_changed_radian = 0
        total_changed_int = 0
        total_zero_on_nonzero = 0
        combined_mask_index: dict[int, Counter] = defaultdict(Counter)
        for path in files:
            try:
                result = scan_file(path)
            except Exception as exc:  # noqa: BLE001 -- a scan tool, not production code
                print(f"{path.name:45s} FAILED: {exc}")
                continue
            print(
                f"{result['path']:45s} {result['total_walls']:6d} {result['int_count']:6d} "
                f"{result['radian_count']:6d} {result['changed']:8d}"
            )
            if result["radian_count"] > 0:
                total_changed_radian += result["changed"]
            else:
                total_changed_int += result["changed"]
            total_zero_on_nonzero += result["zero_rotation_on_nonzero_mask"]
            for mask, counter in result["mask_to_today_index"].items():
                combined_mask_index[mask].update(counter)

        print()
        print(f"Changed tiles across radian-encoded files: {total_changed_radian}")
        print(f"Changed tiles across integer-only files (must be 0):  {total_changed_int}")
        print(f"0.0-rotation walls on a non-zero mask whose derived index != 0: {total_zero_on_nonzero}")

        if args.correlation:
            print()
            print("mask -> today's variant_index() distribution (all files combined):")
            for mask, counter in sorted(combined_mask_index.items()):
                label = _MASK_NAME.get(mask, format(mask, "04b"))
                total = sum(counter.values())
                dist = ", ".join(f"{idx}:{n}" for idx, n in sorted(counter.items()))
                print(f"  {label:14s} n={total:5d}  {dist}")


if __name__ == "__main__":
    main()
