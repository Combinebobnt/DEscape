#!/usr/bin/env python3
"""
Regenerates descape/unit_render_data.json: per-unit_const rendering hints used
by descape/render.py, extracted from the game's own unit table (via
genieutils-py, which parses empires2_x2_p1.dat) -- not game asset content
itself, same reasoning as terrain_texture_map.json / tree_unit_ids.json.

Five tables, all keyed by unit_const:

- "buildings": [span_x, span_y] -- the footprint's real width and height in
  TILES, for every unit whose `building` field is populated (i.e. it's an
  actual placeable structure, not a resource/decoration/mobile unit). Each
  axis is max(1, round(2 * clearance_size_<axis>)) independently.
  clearance_size is the game's own per-unit HALF-footprint, so doubling it is
  what makes even spans expressible at all: Town Center/Castle clearance
  (2.0, 2.0) are the game's actual 4x4 buildings, Mill/House clearance
  (1.0, 1.0) the actual 2x2 ones, stone/palisade Wall clearance (0.5, 0.5) a
  real 1x1 segment (a wall line is many 1-tile segments, not one thick
  object), and axis-aligned Gate segments clearance (2.0, 0.5)/(0.5, 2.0) the
  real 1x4 strip along their axis. Diagonal Gate end caps are (0.5, 0.5),
  1x1, same as the main Wall; the diagonal Gate's own center piece is a
  separate, larger unit_const.

  2 * clearance is an exact integer for 918 of the 940 axis values across all
  470 building consts. The 22 exceptions are small decoratives at clearance
  0.2/0.25/0.3, and 66 further axis values are exactly 0 -- all want a 1x1
  dot, which is what the max(1, ...) floor gives them.

  Two superseded versions of this rule, both kept because each failure is
  easy to reintroduce: the first collapsed both axes to one square radius via
  max(x, y) floored at 1, which turned every 1-tile wall segment into a 3x3
  square and squashed long gate strips into fat squares. The second emitted
  round(clearance) as a per-axis RADIUS, which fixed those two but could only
  ever produce odd spans downstream (2r + 1 tiles) -- so every even-span
  building rendered a tile too large per axis, and round()'s round-half-even
  made Farm (1.5) doubly wrong. Per-axis span, doubled before rounding, fixes
  all of it.
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
- "foundation_terrain": int terrain_id -- every building unit's own
  `building.foundation_terrain_id` field, when it's a valid table index
  (>= 0; -1 means "no foundation terrain," used by several invisible
  internal helper consts). This is a raw mirror of the .dat field, not a
  filtered "which of these should render as terrain" set -- render.py's
  own policy (see _terrain_overlay_for) decides that by also checking
  unit_graphic_map.json, since a real .sld should win over a terrain
  override wherever one exists (e.g. Wonder and several other consts here
  also have a foundation terrain, purely for the in-game construction
  outline, and must keep drawing as their sprite). Farm and the rest of
  its family are the only consts that end up with a foundation terrain
  AND no .sld today.
- "building_tiles": unit_const -> [[ox, oy], ...] local tile offsets inside
  the "buildings" bbox, present ONLY for the 36 class_ == 39 (gate) consts
  whose real occupied footprint is smaller than their bbox -- a diagonal
  (NE/SE/E/N) gate's true footprint is a 2x2 solid centre
  (`collision_size`, doubled the same way clearance_size is) plus two 1x1
  corner pillars (`building.annexes`' real misplacement slots), 6 tiles
  inside the 4x4 bbox "buildings" gives it, not all 16. Axis-aligned gate
  segments (bbox (4,1)/(1,4)) also carry a narrower collision_size than
  clearance_size, but their real footprint genuinely IS the whole strip
  (that mismatch means something else for them, not sparsity) -- gated to
  span == (4, 4) so they never enter this table.

  Offsets are in the same local frame `_span_start`-derived bbox tile 0 is
  the origin of: the solid centre always occupies local (1,1)-(2,2)
  inclusive (own tile at (2,2), the invariant `render.py`'s
  `unit_tile_bounds` docstring already relies on), and each pillar is
  `floor(span/2 + annex.misplacement)` per axis. Measured against the real
  .dat: exactly 36 consts qualify, in exactly two 6-tile shapes (an 18/18
  split), every annex misplacement is exactly (+-1.5, +-1.5), and the own
  tile is inside the occupied set for all 36 -- see this script's
  docstring commit and `tests/test_unit_footprints.py`.

- "object_spans": [span_x, span_y] -- the same [span_x, span_y] shape as
  "buildings", for the 96 `class_ == 34` (cliff) consts, which are NOT
  buildings (`unit.building is None`, `type == 10`) and so get no
  "buildings" entry at all. Without one they fell back to render.py's
  NON_BUILDING_SPAN of (1, 1), and `_span_start`'s span-1 branch re-snapped
  their already-exact coordinate to `int(x) + 0.5` -- half a tile off for
  the 32% of corpus cliffs whose piece has an even span on some axis.

  **Deliberately a separate key, not more "buildings" rows.** render.py's
  `_unit_color` does `unit.unit_const in BUILDING_TILE_SPANS` as its
  is-a-building test, so folding cliffs into that dict would recolour every
  one of them as a building. terrain_palette merges the two into
  UNIT_TILE_SPANS for the span lookups alone.

  Sourced from `collision_size`, NOT `clearance_size`: every cliff const
  reports a flat clearance of (1.5, 1.5), which is useless, while
  collision_size varies per piece and matches the sub-tile coordinate parity
  of all 18,232 corpus cliff placements exactly (a 3-wide axis sits on .5, a
  2-wide axis on .0). Doubled and floored with the same
  `max(1, round(2 * v))` rule "buildings" uses.

  Scoped to class 34 rather than "every non-building whose collision_size
  disagrees with its clearance_size": free-placed class-14 decorations
  (grass, flowers, shrubs) carry arbitrary non-.5 coordinates and are
  re-snapped by the same branch, but they are genuinely free-placed
  eye-candy rather than a grid family, and the displacement is invisible on
  a 1-tile sprite. Widening this gate would move them for no gain.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its unit table or palette.
"""

from __future__ import annotations

import argparse
import json
import math
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


_GATE_CLASS = 39
# Cliffs. The whole class, all 96 consts -- ten full families of nine pieces
# plus the six-const CLF01..CLF08 partial family (1339-1346) that shares the
# Desert graphic. Taken from the .dat's own class_ rather than a hand-kept
# const list precisely so that partial family can't be missed.
_CLIFF_CLASS = 34


def _gate_tile_offsets(unit, span_x: int, span_y: int) -> list[list[int]] | None:
    """The 6 local tile offsets a diagonal gate (`unit_const` in class 39,
    bbox 4x4) actually occupies, or None for every other class-39 building
    (axis-aligned gate segments and standalone corner-pillar consts, whose
    collision_size also differs from clearance_size but whose real footprint
    is genuinely the full bbox -- see the module docstring's
    "building_tiles" section)."""
    if (span_x, span_y) != (4, 4):
        return None
    core_x = max(1, round(unit.collision_size_x * 2))
    core_y = max(1, round(unit.collision_size_y * 2))
    if (core_x, core_y) == (span_x, span_y):
        return None
    core_x0 = span_x // 2 - core_x // 2
    core_y0 = span_y // 2 - core_y // 2
    offsets = {(core_x0 + i, core_y0 + j) for i in range(core_x) for j in range(core_y)}
    for annex in unit.building.annexes:
        if annex.unit_id is None or annex.unit_id < 0:
            continue
        ox = math.floor(span_x / 2 + annex.misplacement_x)
        oy = math.floor(span_y / 2 + annex.misplacement_y)
        offsets.add((ox, oy))
    own_tile = (span_x // 2, span_y // 2)
    assert own_tile in offsets, (
        f"own tile {own_tile} not in generated occupied set {sorted(offsets)} -- "
        f"the unit_tile_bounds() own-tile invariant would break"
    )
    return [list(o) for o in sorted(offsets)]


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
    building_tiles: dict[str, list[list[int]]] = {}
    object_spans: dict[str, list[int]] = {}
    resource_colors: dict[str, list[int]] = {}
    foundation_terrain: dict[str, int] = {}

    for unit_const, unit in enumerate(units):
        if unit is None:
            continue
        if unit.class_ == _CLIFF_CLASS:
            object_spans[str(unit_const)] = [
                max(1, round(unit.collision_size_x * 2)),
                max(1, round(unit.collision_size_y * 2)),
            ]
        if unit.building is not None:
            cx, cy = unit.clearance_size
            span_x, span_y = max(1, round(cx * 2)), max(1, round(cy * 2))
            buildings[str(unit_const)] = [span_x, span_y]
            terrain_id = unit.building.foundation_terrain_id
            if terrain_id is not None and terrain_id >= 0:
                foundation_terrain[str(unit_const)] = terrain_id
            if unit.class_ == _GATE_CLASS:
                offsets = _gate_tile_offsets(unit, span_x, span_y)
                if offsets is not None:
                    building_tiles[str(unit_const)] = offsets
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
                "building_tiles": dict(
                    sorted(building_tiles.items(), key=lambda kv: int(kv[0]))
                ),
                "object_spans": dict(
                    sorted(object_spans.items(), key=lambda kv: int(kv[0]))
                ),
                "resource_colors": dict(
                    sorted(resource_colors.items(), key=lambda kv: int(kv[0]))
                ),
                "foundation_terrain": dict(
                    sorted(foundation_terrain.items(), key=lambda kv: int(kv[0]))
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Wrote {len(buildings)} building footprints, "
        f"{len(building_tiles)} sparse building_tiles, "
        f"{len(object_spans)} object_spans, "
        f"{len(resource_colors)} resource colors and "
        f"{len(foundation_terrain)} foundation terrains to {out_path}"
    )


if __name__ == "__main__":
    main()
