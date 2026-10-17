#!/usr/bin/env python3
"""
Regenerates descape/unit_render_data.json: per-unit_const rendering hints used
by descape/render.py, extracted from the game's own unit table (via
genieutils-py, which parses empires2_x2_p1.dat) -- not game asset content
itself, same reasoning as terrain_texture_map.json / tree_unit_ids.json.

Nine tables, eight keyed by unit_const and one ("hero_glow") a sorted list of
consts:

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
  disagrees with its clearance_size". The worry was free-placed class-14
  decorations (grass, flowers, shrubs), which carry arbitrary non-.5
  coordinates and would be re-snapped. Re-measured on 2026-10-13 (TASK-134),
  it does not apply to the class-14 objects that are really big: every
  decoration it protects has collision_size 0 (132 consts, 9,618 corpus
  placements, big only in outline_size), so a collision gate leaves them all
  1x1. Class 14 gets its own key for that reason, "doodad_spans" below, and
  its four blockers keep "blocker_spans".

- "blocker_spans": [span_x, span_y] -- the same shape again, for exactly the
  four consts in `_BLOCKER_SPAN_CONSTS` (Blocker 1x3/3x1 2423/2424 and
  Buildable Blocker 1x3/3x1 2429/2430, GH #121). Class 14 like the
  decorations above, so neither table has them and they read as 1x1: the
  marker, picking, outlines and occupied tiles all covered the centre tile of
  a three-tile barrier. Sourced from `collision_size` like "object_spans"
  ((0.5, 1.5) and (1.5, 0.5), matching the names' 1x3/3x1).

  An explicit allowlist, not a class rule, so no decoration moves. Safe
  because the span cannot move a placed blocker: all 37 corpus placements
  (16 of 2423, 21 of 2424) sit at x.5/y.5, and an odd span centres on that
  same tile. Its own key rather than more "object_spans" rows, which
  `tests/test_unit_footprints.py` pins to exactly the 96 cliffs.

- "fish_spans": [span_x, span_y] -- the same shape, for exactly the eight
  deep-water fish in `_FISH_SPAN_CONSTS` (Perch 53, Dolphin 61, Marlin 450,
  Dorado 455, Salmon 456, Tuna 457, Snapper 458 and Whale 2625, GH #178).
  Not buildings or cliffs, so they read as 1x1, where the in-game selection
  box is about 2x2 (3x3 for the Whale). Sourced from `outline_size`, the
  field the in-game selection box draws, doubled and floored by the same
  `max(1, round(2 * v))` rule: 1.0 gives 2x2 and the Whale's 1.5 gives 3x3.
  The generator asserts `clearance_size` agrees for every listed const, so a
  .dat change fails loudly instead of silently picking one field.

  An explicit allowlist, not a class rule: Shore Fish 69, Box Turtles 1141
  and Oysters 2170 are 0.5, which is 1x1 already, and the other non-buildings
  the .dat calls bigger than 1x1 are a separate task. A real span also
  anchors newly placed fish where the corpus puts them: 1028 of 1049 corpus
  deep-fish placements sit on whole coordinates (53 417/428, 450 13/13, 455
  29/33, 456 480/482, 457 37/37, 458 52/56), which is where an even span
  centres. The Whale (3x3, centred on .5) has no corpus placement. Its own
  key for the same reason "blocker_spans" has one.

- "doodad_spans": [span_x, span_y] -- the same shape, for every class-14
  (`_DOODAD_CLASS`) non-building outside `_BLOCKER_SPAN_CONSTS` whose
  collision span, `max(1, round(2 * collision_size))` like cliffs and
  blockers, is above 1x1 (TASK-134, which folded in TASK-209): mountains,
  ruins, rock formations, the Theatre, Stonehenge. A class rule, not an
  allowlist, measured on 2026-10-13 against the current .dat and the 21
  loadable examples/ files:

  - 46 consts: 11 mountains 6x6 (310, 311, 744, 745, 1041-1047), Stonehenge
    1979 and Theatre 2530 5x5, MRKT 1646 and CastleRuins 1837 4x4, 16 at
    3x3, 15 at 2x2. `outline_size` gives the same span for all 46, and the
    generator refuses when it stops doing so. `clearance_size` agrees for 35
    and says 4x4 for the mountains.
  - The decorations the old class-34 scoping protected have collision 0: 132
    other class-14 consts are big in `outline_size` only (grass, plants,
    every Rubble), with 9,618 corpus placements. All stay 1x1.
  - Corpus parity matches the collision span with no counterexample: 205
    placements of 17 of the consts, 181 on the half-tile grid, all 181 on
    their span's parity (even span on .0, odd on .5). The other 24 are
    free-placed (11 mountains, 13 of 42 ROCKB1 1148) and quantize like an
    off-grid building.
  - Mountains are 6x6, not clearance's 4x4: at half_w 48 their ink is 4.8
    to 6.0 tiles wide and reaches 3.2 to 4.1 tile-heights below the hotspot,
    where a 6x6 diamond reaches 3.0 and a 4x4 2.0.

  Only the footprint is new: render._resolve_unit_sprite keeps a doodad's
  sprite at its stored point and sorts it at its own tile, as before, and
  placement_rules keeps the clearance box for the object rule. Its own key,
  as for "blocker_spans" and "fish_spans". Accepted with no code of their
  own: with sprites off, or in Flat, a mountain's mark is a 6x6 slab and
  Flat's icon contain-fits the 6x6 rect; snapped Place Unit and a group
  move's edge clamp put an even-span doodad on whole coordinates and keep
  its footprint on the map; Replace re-anchors on the low corner when the
  span changes (root AGENTS.md), so a tree replaced by a mountain lands the
  mountain's low corner on the tree's tile; Scatter and Mirror treat doodads
  as multi-tile; a free-placed doodad's footprint quantizes like an off-grid
  building's, up to 0.74 tile from its sprite; and a 1x1 unit under a
  later-painted doodad joins a stack group (+55 corpus-wide).

- "hero_glow": [unit_const, ...] -- the heroes the game draws its golden glow
  around (GH #39). The rule is `(creatable.hero_mode & 0xff) & (1 | 64)` AND
  `creatable.hero_glow_graphic in 12218..12224`, which gives 260 consts.

  Neither half alone is enough. 529 ordinary units (ARCHR, KNGHT...) also
  carry a HeroGlow graphic, and so does KINGX 434 (mode 34, no glow
  in-game), so the graphic cannot mark a hero by itself. Hero-flagged
  buildings (HTAT, SEAFORF..., mode 1) have glow graphic -1 and drop out,
  matching the game.

  12218..12224 are the seven gold `HeroGlow <Infantry|Cavalry|Elephant|Ship|
  Siege|Animal|Admiral>` graphics. Every one of them, and their
  Small/Medium/Big/Huge deltas 12271..12274, has `file_name None` and
  `frame_count 0`: the game draws the glow procedurally, so DEscape builds a
  gold outline from each sprite's alpha (unit_sprites._with_glow) instead of
  loading art. The three non-gold custom glows are out of scope: 12716
  (black, WOLF2), 13006 (GreenDragon) and 13008 (WindingBlades).

  Against AoE2ScenarioParser's `HeroInfo` (244 consts), 17 are only in the
  .dat and 1 (2171, Oystering Ship, mode 0) is only in `HeroInfo`. Modes -111
  (0x91, 22 Three Kingdoms heroes) and -63 (0xC1, HWOLF 700) set bit 128,
  which AGE documents as "invert flags". The `& 65` reading keeps all 23;
  whether HWOLF really glows is settled in-game, not guessed here. Gaia
  (`civs[0]`) is read, as for every other table: the one civ override of
  hero_mode (civ 31, CHIEFSYURT) is a building.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its unit table or palette.

Normally run through tools/regen_game_data.py, which also restamps
tools/game_data_stamp.json.
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
# Class-14 blockers with a real multi-tile collision; see the "blocker_spans" docs.
_BLOCKER_SPAN_CONSTS = (2423, 2424, 2429, 2430)
# Deep-water fish sized like the in-game selection box; see the "fish_spans" docs.
_FISH_SPAN_CONSTS = (53, 61, 450, 455, 456, 457, 458, 2625)
# Scenery, ruins, mountains; the class rule behind "doodad_spans" (see its docs).
_DOODAD_CLASS = 14


def _fish_span(unit_const: int, unit) -> list[int]:
    outline = (unit.outline_size_x, unit.outline_size_y)
    if tuple(unit.clearance_size) != outline:
        raise SystemExit(
            f"fish {unit_const} outline_size {outline} != clearance_size "
            f"{tuple(unit.clearance_size)} -- re-verify _FISH_SPAN_CONSTS against the .dat"
        )
    return [max(1, round(v * 2)) for v in outline]


def _doodad_span(unit_const: int, unit) -> list[int] | None:
    """A class-14 non-building's collision span when it is above 1x1, else
    None. Refuses when `outline_size` gives a different span."""
    span = [max(1, round(unit.collision_size_x * 2)), max(1, round(unit.collision_size_y * 2))]
    if span == [1, 1]:
        return None
    outline = [max(1, round(unit.outline_size_x * 2)), max(1, round(unit.outline_size_y * 2))]
    if outline != span:
        raise SystemExit(
            f"doodad {unit_const} collision span {span} != outline span {outline} "
            "-- re-measure the doodad_spans rule against the .dat"
        )
    return span


# The seven gold HeroGlow graphics; see the "hero_glow" docs above.
_HERO_GLOW_GRAPHICS = range(12218, 12225)
_HERO_MODE_GLOW_BITS = 1 | 64


def _is_glowing_hero(unit) -> bool:
    creatable = getattr(unit, "creatable", None)
    if creatable is None:
        return False
    return bool((creatable.hero_mode & 0xFF) & _HERO_MODE_GLOW_BITS) and (
        creatable.hero_glow_graphic in _HERO_GLOW_GRAPHICS
    )


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "descape",
        help="Where to write unit_render_data.json (default: descape/)",
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
    blocker_spans: dict[str, list[int]] = {}
    fish_spans: dict[str, list[int]] = {}
    doodad_spans: dict[str, list[int]] = {}
    resource_colors: dict[str, list[int]] = {}
    foundation_terrain: dict[str, int] = {}
    hero_glow: list[int] = []

    for unit_const, unit in enumerate(units):
        if unit is None:
            continue
        if _is_glowing_hero(unit):
            hero_glow.append(unit_const)
        if unit.class_ == _CLIFF_CLASS:
            object_spans[str(unit_const)] = [
                max(1, round(unit.collision_size_x * 2)),
                max(1, round(unit.collision_size_y * 2)),
            ]
        if unit_const in _BLOCKER_SPAN_CONSTS:
            blocker_spans[str(unit_const)] = [
                max(1, round(unit.collision_size_x * 2)),
                max(1, round(unit.collision_size_y * 2)),
            ]
        if unit_const in _FISH_SPAN_CONSTS:
            fish_spans[str(unit_const)] = _fish_span(unit_const, unit)
        if unit.class_ == _DOODAD_CLASS and unit.building is None and unit_const not in _BLOCKER_SPAN_CONSTS:
            span = _doodad_span(unit_const, unit)
            if span is not None:
                doodad_spans[str(unit_const)] = span
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "unit_render_data.json"
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
                "blocker_spans": dict(
                    sorted(blocker_spans.items(), key=lambda kv: int(kv[0]))
                ),
                "fish_spans": dict(
                    sorted(fish_spans.items(), key=lambda kv: int(kv[0]))
                ),
                "doodad_spans": dict(
                    sorted(doodad_spans.items(), key=lambda kv: int(kv[0]))
                ),
                "resource_colors": dict(
                    sorted(resource_colors.items(), key=lambda kv: int(kv[0]))
                ),
                "foundation_terrain": dict(
                    sorted(foundation_terrain.items(), key=lambda kv: int(kv[0]))
                ),
                "hero_glow": sorted(hero_glow),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Wrote {len(buildings)} building footprints, "
        f"{len(building_tiles)} sparse building_tiles, "
        f"{len(object_spans)} object_spans, "
        f"{len(blocker_spans)} blocker_spans, "
        f"{len(fish_spans)} fish_spans, "
        f"{len(doodad_spans)} doodad_spans, "
        f"{len(resource_colors)} resource colors, "
        f"{len(foundation_terrain)} foundation terrains and "
        f"{len(hero_glow)} glowing heroes to {out_path}"
    )


if __name__ == "__main__":
    main()
