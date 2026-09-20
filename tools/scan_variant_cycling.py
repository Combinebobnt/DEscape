#!/usr/bin/env python3
"""Measures the Cycle Variant population across examples/, so the numbers in
the GAIA variant cycling plan (2026-09-11) stay re-derivable.

Install-free: the corpus walk uses scenario_io.load_map_and_units only, and the
cycle modulus is the committed variant_count (unit_variant.variant_count_for()).
tests/test_unit_variant.py's install-gated test is what pins that committed
count to the real .sld.

Reports, over every placement on a unit_variant.is_cyclable() const: placements
per const, the literal/radian split, literals past the variant count (which
variant_index() wraps), and catalog-wide totals.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descape import unit_rotation, unit_sprites, unit_variant
from descape.scenario_io import load_map_and_units


def scan_file(path: Path) -> dict[int, Counter]:
    scenario = load_map_and_units(path)
    per_const: dict[int, Counter] = {}
    for unit in scenario.unit_manager.get_all_units():
        const = unit.unit_const
        if not unit_variant.is_cyclable(const):
            continue
        rotation = float(unit.rotation)
        literal = abs(rotation - round(rotation)) < 1e-6
        counter = per_const.setdefault(const, Counter())
        counter["placements"] += 1
        counter["literal" if literal else "radian"] += 1
        counter["wraps"] += int(literal and round(rotation) >= unit_variant.variant_count_for(const))
    return per_const


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples")
    args = parser.parse_args()

    files = sorted(
        p for p in args.scenario_dir.iterdir()
        if p.suffix in (".aoe2scenario", ".scx2") or p.name == "play_Test"
    )
    if not files:
        print(f"No scenario files found under {args.scenario_dir}", file=sys.stderr)
        sys.exit(1)

    combined: dict[int, Counter] = {}
    for path in files:
        try:
            per_const = scan_file(path)
        except Exception as exc:
            print(f"{path.name}: FAILED: {exc}", file=sys.stderr)
            continue
        for const, counter in per_const.items():
            combined.setdefault(const, Counter()).update(counter)

    graphic_map = unit_sprites.graphic_map()
    print(f"{'const':>6s} {'file_name':32s} {'vars':>5s} {'n':>7s} {'lit':>7s} {'rad':>5s} {'wrap':>5s}")
    totals: Counter = Counter()
    for const in sorted(combined):
        c = combined[const]
        name = str(graphic_map.get(const, {}).get("file_name", "?"))
        print(
            f"{const:6d} {name:32s} {unit_variant.variant_count_for(const):5d} {c['placements']:7d} "
            f"{c['literal']:7d} {c['radian']:5d} {c['wraps']:5d}"
        )
        totals.update(c)

    catalog = [const for const in graphic_map if unit_variant.is_cyclable(const)]
    divergent = [
        const for const in graphic_map
        if unit_rotation.semantics_for(const) == unit_rotation.VARIANT
        and const not in unit_variant.excluded_consts()
        and unit_variant.variant_count_for(const) != unit_rotation.angle_count_for(const)
    ]
    uncounted = [const for const in catalog if const not in unit_variant._variant_counts()]
    print()
    print(f"Files scanned: {len(files)}")
    print(f"Cyclable consts placed: {len(combined)}")
    print(f"Placements on them: {totals['placements']}")
    print(f"  literal integer: {totals['literal']} (of which past the variant count, wrapping: {totals['wraps']})")
    print(f"  radian-encoded: {totals['radian']}")
    print(f"Cyclable consts in the catalog: {len(catalog)}")
    print(f"  variant_count != angle_count (VARIANT, not excluded): {divergent}")
    print(f"  no committed variant_count (.sld missing at generation): {uncounted}")


if __name__ == "__main__":
    main()
