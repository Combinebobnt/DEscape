#!/usr/bin/env python3
"""
Regenerates descape/terrain_blend_map.json: a terrain_id -> blend priority,
blend type and layer overlay mask table, read by descape/terrain_blend.py.

Pure factual data extracted from the game's own terrain table (via
genieutils-py, which parses empires2_x2_p1.dat), not game asset content
itself. Three fields per enabled terrain:

    blend_priority     which of two neighbours bleeds into the other (higher wins)
    blend_type         0..7: land, farm, beach, water, shallows, road, ice, snow;
                       the row/column of terrain_blend.MODE_TABLE
    overlay_mask_name  a file under resources/_common/terrain/masks/, or ''

This script FAILS rather than writing a file if two enabled terrains share a
priority or a blend_type leaves 0..7: the blend order and the mode lookup
both assume those hold, so a game update that breaks either must be
re-measured, not papered over.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install, only for regenerating this file if the game updates its terrain table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLEND_TYPES = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    args = parser.parse_args()

    from genieutils.datfile import DatFile

    dat_path = args.aoe2de_root / "resources/_common/dat/empires2_x2_p1.dat"
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = DatFile.parse(str(dat_path))
    terrains = data.terrain_block.terrains

    table: dict[str, dict] = {}
    violations: list[str] = []
    owner_of_priority: dict[int, int] = {}
    for i, t in enumerate(terrains):
        if not t.enabled:
            continue
        priority, blend_type = int(t.blend_priority), int(t.blend_type)
        if not 0 <= blend_type < BLEND_TYPES:
            violations.append(f"  id={i} name={t.name!r}: blend_type {blend_type} outside 0..{BLEND_TYPES - 1}")
        if priority in owner_of_priority:
            violations.append(f"  id={i} name={t.name!r}: blend_priority {priority} already used by id={owner_of_priority[priority]}")
        owner_of_priority[priority] = i
        table[str(i)] = {
            "priority": priority,
            "blend_type": blend_type,
            "overlay_mask": t.overlay_mask_name or "",
        }

    if violations:
        print("A terrain blend invariant no longer holds:", file=sys.stderr)
        for line in violations:
            print(line, file=sys.stderr)
        raise SystemExit("Refusing to write: re-measure before touching descape/terrain_blend.py.")

    out_path = Path(__file__).resolve().parent.parent / "descape" / "terrain_blend_map.json"
    out = {
        "_comment": (
            "terrain_id -> {priority, blend_type, overlay_mask} for every enabled "
            "terrain. Pure factual data extracted from the game's own terrain table "
            "via genieutils-py, not game asset content itself. Regenerate with "
            "tools/gen_terrain_blend_map.py."
        ),
        "terrains": table,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    masks = sorted({e["overlay_mask"] for e in table.values()})
    print(f"Wrote {len(table)} entries to {out_path}")
    print(f"  overlay masks named: {masks}")


if __name__ == "__main__":
    main()
