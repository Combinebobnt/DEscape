#!/usr/bin/env python3
"""
Regenerates descape/unit_render_data.json: per-unit_const rendering hints used
by descape/render.py, extracted from the game's own unit table (via
genieutils-py, which parses empires2_x2_p1.dat) -- not game asset content
itself, same reasoning as terrain_texture_map.json / tree_unit_ids.json.

Two tables, both keyed by unit_const:

- "buildings": [radius_x, radius_y] in tiles, for every unit whose `building`
  field is populated (i.e. it's an actual placeable structure, not a
  resource/decoration/mobile unit). Each axis is
  round(clearance_size_<axis>) independently -- clearance_size is the game's
  own per-unit half-footprint in tiles. No minimum floor: a rounded radius of
  0 is correct and renders as a real 1-tile-wide dot, not "invisible" (a
  single dot is already visible -- there's no reason to inflate it).
  Confirmed against real footprints: Town Center/Castle clearance (2.0, 2.0)
  are the game's actual 4x4 buildings; Mill/House clearance (1.0, 1.0) are
  the actual 2x2 buildings; stone/palisade Wall clearance (0.5, 0.5) rounds
  to (0, 0) -- a real 1x1 wall segment, matching the game (a wall line is
  many 1-tile segments, not one thick object); axis-aligned Gate segments
  clearance (2.0, 0.5)/(0.5, 2.0) round to a long, 1-tile-wide strip along
  their axis, matching the real 1x4 gate shape; diagonal Gate end caps
  clearance (0.5, 0.5) round to (0, 0), a real 1x1 tile, same as the main
  Wall; the diagonal Gate's own center piece is a separate, larger
  unit_const. Earlier version of this script collapsed both axes to a single
  square radius via max(x, y) and floored it at 1 -- that turned every 1-tile
  wall segment into an oversized 3x3 square and squashed long gate strips
  into fat squares; per-axis radius with no floor fixes both.
- "resource_colors": real RGB per non-building unit whose `minimap_color`
  field is nonzero, resolved against the game's own palette
  (resources/_common/palettes/original.pal, JASC-PAL format). The field is a
  signed byte in the .dat (wraps negative for palette indices >= 128); this
  script unwraps it before the palette lookup. Confirmed this reproduces
  exactly what a player would guess from the real minimap: food sources
  (deer/boar/fish/forage bush, all minimap_color -87) resolve to a desaturated
  light green, gold (GOLDM/PGOLD) to a golden yellow, relics to white. Trees
  are deliberately excluded from this table even though the game also colors
  them dark green on the minimap -- their minimap_color is 0/unset in the
  data (unlike every other case here), meaning that coloring is a hardcoded
  engine special-case rather than data-driven; see tree_unit_ids.json /
  TREE_COLOR in descape/terrain_palette.py for that separate, hand-matched
  path. Buildings are excluded here even on the rare occasion one carries a
  nonzero minimap_color (some gate/farm-adjacent auxiliary entries do) --
  in-game, buildings always render in owner player color on the minimap, not
  a resource color, so render.py should never consult this table for a unit
  that's also in "buildings".

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its unit table or palette.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_palette(pal_path: Path) -> list[tuple[int, int, int]]:
    lines = pal_path.read_text().splitlines()
    if lines[0] != "JASC-PAL":
        raise SystemExit(f"Unexpected palette format in {pal_path}")
    count = int(lines[2])
    entries = []
    for i in range(count):
        r, g, b = (int(v) for v in lines[3 + i].split())
        entries.append((r, g, b))
    return entries


def _unwrap_signed_byte(value: int) -> int:
    return value & 0xFF


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
    pal_path = args.aoe2de_root / "resources/_common/palettes/original.pal"
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")
    if not pal_path.is_file():
        raise SystemExit(f"Not found: {pal_path}")

    data = DatFile.parse(str(dat_path))
    palette = _load_palette(pal_path)
    units = data.civs[0].units

    buildings: dict[str, list[int]] = {}
    resource_colors: dict[str, list[int]] = {}

    for unit_const, unit in enumerate(units):
        if unit is None:
            continue
        if unit.building is not None:
            cx, cy = unit.clearance_size
            buildings[str(unit_const)] = [round(cx), round(cy)]
            continue  # buildings never get a resource_colors entry
        if unit.minimap_color:
            idx = _unwrap_signed_byte(unit.minimap_color)
            if idx < len(palette):
                resource_colors[str(unit_const)] = list(palette[idx])

    out_path = Path(__file__).resolve().parent.parent / "descape" / "unit_render_data.json"
    out_path.write_text(
        json.dumps(
            {
                "_comment": "unit_const -> rendering hints -- see tools/gen_unit_render_data.py",
                "buildings": dict(sorted(buildings.items(), key=lambda kv: int(kv[0]))),
                "resource_colors": dict(
                    sorted(resource_colors.items(), key=lambda kv: int(kv[0]))
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Wrote {len(buildings)} building footprints and "
        f"{len(resource_colors)} resource colors to {out_path}"
    )


if __name__ == "__main__":
    main()
