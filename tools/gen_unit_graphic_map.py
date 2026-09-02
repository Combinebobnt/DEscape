#!/usr/bin/env python3
"""
Regenerates descape/unit_graphic_map.json: the unit_const -> .sld graphic
lookup the sprite renderer needs, extracted from the game's own unit and
graphic tables (via genieutils-py, which parses empires2_x2_p1.dat) -- not game
asset content itself, same reasoning as unit_render_data.json /
terrain_texture_map.json / tree_unit_ids.json.

One table, keyed by unit_const, one entry per unit whose standing graphic
resolves to a named graphic:

    "<unit_const>": {
      "graphic_id":    int,  -- index into the .dat's own graphics table
      "file_name":     str,  -- on-disk stem, no extension and no path
      "angle_count":   int,  -- angles the graphic is authored for
      "mirroring_mode": int, -- how many of those angles are actually stored
      "frame_count":   int,  -- frames per angle
      "pieces": [            -- OPTIONAL. Present only for a multi-graphic
                              -- composite building (town centres, pastures);
                              -- absent means "one graphic", same as today.
        {"unit_id": int,       -- the piece's OWN resolving unit_id (this
                                -- const itself for the parent's own piece,
                                -- an annex's unit_id otherwise) -- the only
                                -- correct input to a per-graphic rotation
                                -- dispatch, since that is a property of the
                                -- piece's own graphic, not of whichever
                                -- unit is standing on the map.
         "file_name": str, "angle_count": int, "frame_count": int,
         "dx": int, "dy": int},   -- native-pixel offset from the parent's
                                  -- own anchor; see "Composite buildings"
        ...                       -- below. Depth-sorted: draw in list order.
      ]
    }

The renderer turns file_name into
`resources/_common/drs/graphics/{file_name}.sld` against the user's own
configured install; nothing here is a path, and no install is consulted at
runtime. A unit_const missing from this table, or one whose .sld is absent or
unreadable on a particular install, falls back to today's colored mark -- the
sprite path is strictly additive.

Field notes, each measured against the real table rather than assumed:

- standing_graphic is a PAIR. The first element is the graphic this table
  records; the second is a secondary/alternate-state graphic (e.g. unit 705
  "Cow Black and White" carries (8221, 845)) and is deliberately ignored --
  a scenario editor draws one static pose per unit, not an animation or a
  state machine. Run this script to see the surveyed split; the summary it
  prints is the record, not a committed field.
- angle_count and mirroring_mode together decide which stored frame_index a
  unit's `rotation` maps to. Both are recorded raw here because that mapping
  is the renderer's decision, not this script's.
- frame_count is frames per angle, so a graphic's stored frame count is
  frame_count * (however many angles mirroring_mode leaves stored). It is
  emitted because it is the only cheap cross-check that an angle->frame
  mapping lands inside the file at all; without it the renderer can only
  discover an out-of-range frame_index by failing to decode one.

Not filtered against the install's actual files on disk. Doing so would bake
one install's version into a committed data table, and the renderer already
has to handle a missing or unreadable .sld anyway. The script does REPORT that
coverage when --scan-sld is given, since the rate is worth knowing and the
scan is free while an install path is already in hand.

Needs genieutils-py (`pip install -r requirements-dev.txt`) and a real AoE2DE
install -- neither of which this repo depends on for normal use, only for
regenerating this file if the game updates its unit or graphic tables.

**Legacy shell graphics.** A unit's standing_graphic sometimes names a pre-DE graphic slot
DE's asset pipeline never renamed -- the .dat's own file_name is still an
old SLP-era string like FARM0NNG, or the literal string "None". Two
different things turned out to be true of that bucket, checked against the
real .dat and a real install rather than assumed:

- Some of these are genuine *shell* graphics: file_name is unusable but the
  graphic's own `deltas` list points at the real, modern (`_x1`-suffixed)
  sprite -- e.g. DOCK's shell (file_name "None") deltas to
  `b_dark_dock_age1_x1`, which exists on disk. `_resolve_modern_graphic`
  walks that chain and, when it finds one, uses THAT graphic's file_name
  *and* its angle_count/mirroring_mode/frame_count -- verified necessary:
  sibling deltas of the same shell can carry a different frame_count than
  the shell itself (e.g. a 30-frame sail animation beside a 1-frame one), so
  taking the metadata from anywhere but the resolved graphic mis-describes
  the file being pointed at.
- Farm and its 14 related consts (RFARM, FARMDROP, PASTURE, ...) are not a
  shell case -- every civ's copy resolves to the same legacy graphic, whose
  own deltas terminate in further legacy names, and no graphic anywhere in
  the table has a "farm"/"field"/"crop" file_name. The .dat's own terrain
  table (`terrain_block.terrains`) names entries "Farm1", "Farm2", "Farm
  Cnst1-3", "RFarm1-2": in real AoE2:DE a farm's visible crop is a terrain
  tile blend, not a unit sprite, and standing_graphic is a vestigial
  pointer left over from a pre-terrain-farm engine version. No unit .sld
  can supply this; consts that resolve this way are omitted rather than
  entered under a filename that will never exist on any install.

**Composite buildings.** A town centre (and a pasture) is drawn from several
co-located graphics, not one -- e.g. RTWC's own standing_graphic is
`b_dark_town_center_age1_back_x1`, note `_back_`, one quarter of the real
building. The other pieces are never placed as their own scenario units; the
edge is `unit.building.annexes`, a fixed 4-slot array of (unit_id,
misplacement_x, misplacement_y), unused slots carrying unit_id -1. Each real
annex's OWN standing_graphic is resolved through the same
`_resolve_modern_graphic` walk used for legacy shells above, and its screen
offset is `iso_screen(misplacement) + graphic.deltas[0].offset` (native
pixels, half_w=48/half_h=24 at the iso projection's native scale) -- the
parent's own art carries a null delta whose offset exactly cancels its own
misplacement, which is what keeps a town centre's four pre-aligned pieces at
`(0, 0)` net while letting a pasture's four repeated corner-post copies land
at their real, uncancelled offsets.

`_COMPOSITE_SCOPE` is a hand-verified allowlist, not "every building whose
annexes resolve" -- gate consts also resolve their corner-pillar annexes
through this exact mechanism (confirmed 2026-08-29), which is a separate,
not-yet-built follow-on, and three pasture-named consts (1893, 1897, 2078)
never reach the pieces walk at all because their OWN standing_graphic is an
unresolvable Farm-family legacy shell -- they get no base entry, same as
before this feature.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Every .sld begins with this, and the u16 at offset 10 is a layout-variant tag
# taking exactly two values across the install, both readable: 16, and the 14
# variant whose only difference is a two-byte-shorter header. Kept as a literal
# rather than imported so this stays a standalone genieutils script; the
# authority is descape.sld_decoder.SUPPORTED_LAYOUT_TAGS. Only used by
# --scan-sld's reporting, never by the emitted table.
SLD_MAGIC = b"SLDX"
SLD_LAYOUT_READABLE = (14, 16)


def _sld_status(path: Path) -> str:
    """Classify one .sld the way the renderer's fallback will: 'ok' or why not."""
    try:
        head = path.read_bytes()[:12]
    except OSError:
        return "missing"
    if len(head) < 12:
        return "empty" if not head else "truncated"
    if head[:4] != SLD_MAGIC:
        return "bad_magic"
    layout = int.from_bytes(head[10:12], "little")
    return "ok" if layout in SLD_LAYOUT_READABLE else f"layout_{layout}"


def _looks_modern(file_name: str | None) -> bool:
    """DE's own on-disk convention for a real, renderable .sld stem."""
    return bool(file_name) and file_name != "None" and file_name.endswith("_x1")


def _resolve_modern_graphic(graphics: list, graphic_id: int, seen: set[int] | None = None):
    """Find the graphic actually worth rendering for graphic_id: itself if its
    file_name already looks modern, else the first delta descendant that
    does. See the module docstring's "Legacy shell graphics" section."""
    if seen is None:
        seen = set()
    if graphic_id is None or graphic_id < 0 or graphic_id >= len(graphics) or graphic_id in seen:
        return None
    seen.add(graphic_id)
    graphic = graphics[graphic_id]
    if graphic is None:
        return None
    if _looks_modern(graphic.file_name):
        return graphic
    for delta in graphic.deltas:
        resolved = _resolve_modern_graphic(graphics, delta.graphic_id, seen)
        if resolved is not None:
            return resolved
    return None


# HAND-VERIFIED against the real .dat (2026-08-29), per the module docstring's
# "Composite buildings" section -- NOT "every unit_const whose annexes
# resolve", which also catches gate consts (their corner-pillar annexes
# resolve through this same _resolve_modern_graphic walk too, confirmed while
# building this table; that is deliberately a separate follow-on, not built
# here). 1893/1897/2078 are farm-family legacy shells with no modern
# replacement for their OWN standing_graphic, so they never reach the pieces
# walk regardless of scope membership -- listed in
# tests/test_unit_graphic_map.py's FARM_FAMILY_CONSTS, not here.
_COMPOSITE_SCOPE: frozenset[int] = frozenset({
    71, 109, 141, 142, 2275, 2276, 2277,  # town centres (all ages/civs)
    1889, 1890, 2079, 2080,               # pastures
})

# Native-pixel iso half-dimensions, mirroring descape.unit_sprites.
# NATIVE_TILE_W (96) // 2 and // 2 again -- kept as literals rather than an
# import so this stays a standalone genieutils script.
_NATIVE_HALF_W = 48
_NATIVE_HALF_H = 24


def _piece_screen_offset(mx: float, my: float, piece_graphic) -> tuple[int, int]:
    """Native-pixel (dx, dy) an annex piece paints at, relative to the
    parent's own anchor: `iso_screen(misplacement) + graphic.deltas[0].offset`
    -- see the module docstring's "Composite buildings" section. The offset
    survives untouched when the piece graphic carries no delta (pastures); it
    cancels to (0, 0) when it does (town centres), which is the whole reason
    the art carries a null delta there at all."""
    iso_dx = (mx + my) * _NATIVE_HALF_W
    iso_dy = (my - mx) * _NATIVE_HALF_H
    if piece_graphic.deltas:
        iso_dx += piece_graphic.deltas[0].offset_x
        iso_dy += piece_graphic.deltas[0].offset_y
    return round(iso_dx), round(iso_dy)


def _resolve_pieces(
    units: list, graphics: list, unit_const: int, parent_graphic
) -> list[dict[str, object]] | None:
    """The composite `pieces` list for unit_const, depth-sorted (main -> back
    -> center -> front for a town centre), or None if unit_const is out of
    scope or has no resolvable annex. `parent_graphic` is the already-resolved
    graphic the caller is about to emit as the entry's own five fields; it
    becomes pieces[0]'s data at (dx, dy) = (0, 0), since a const's own art has
    zero misplacement by definition -- the parent piece is drawn like any
    other, just at its own depth slot rather than appended as an "extra"."""
    if unit_const not in _COMPOSITE_SCOPE:
        return None
    building = units[unit_const].building
    pieces = [
        (0.0, {
            "unit_id": unit_const,
            "file_name": parent_graphic.file_name,
            "angle_count": parent_graphic.angle_count,
            "frame_count": parent_graphic.frame_count,
            "dx": 0,
            "dy": 0,
        })
    ]
    for annex in building.annexes:
        if not (0 <= annex.unit_id < len(units)):
            continue
        annex_unit = units[annex.unit_id]
        if annex_unit is None:
            continue
        standing = annex_unit.standing_graphic
        graphic_id = standing[0] if standing else -1
        piece_graphic = _resolve_modern_graphic(graphics, graphic_id)
        if piece_graphic is None:
            continue
        dx, dy = _piece_screen_offset(annex.misplacement_x, annex.misplacement_y, piece_graphic)
        depth = annex.misplacement_y - annex.misplacement_x
        pieces.append((depth, {
            "unit_id": annex.unit_id,
            "file_name": piece_graphic.file_name,
            "angle_count": piece_graphic.angle_count,
            "frame_count": piece_graphic.frame_count,
            "dx": dx,
            "dy": dy,
        }))

    if len(pieces) < 2:
        raise SystemExit(
            f"_COMPOSITE_SCOPE includes {unit_const} but its annexes no longer "
            f"resolve to any modern piece -- re-verify against the .dat rather "
            f"than silently dropping the composite"
        )
    pieces.sort(key=lambda dp: dp[0])
    return [p for _, p in pieces]


# unit_const -> the delta graphic_id that holds the real BODY, for consts whose
# standing_graphic is a modern-named DECORATION rather than the structure
# itself. _resolve_modern_graphic below only guards against LEGACY shells: it
# returns the standing graphic untouched as soon as its file_name looks modern,
# which is wrong when that modern name is an overlay.
#
# All three entries here are the same case, read off the .dat 2026-08-27:
# standing_graphic is b_dark_wall_palisade_flag_x1, an animated flag
# (sequence_type 3, frame_count 90, frame_duration 0.044), whose .sld
# carries MAIN layers on only frames 180-269 -- i.e. art for ONE of the
# five wall shapes, the tall tower the flag actually sits on. The wall
# body is a delta hanging off it, static
# (sequence_type 2, frame_count 1) with all five shape frames present.
# Rendering the shell therefore drew a floating flag at one rotation and
# nothing at the other four.
#
# 788's shell has TWO static deltas. 6595 b_scen_wall_sea_x1 is the pick
# because it carries a PLAYERCOLOR layer and the same 5-variant shape
# signature as every other confirmed wall body (a 56x104 narrow column at
# variant 4); 6593 b_scen_wall_sea_underwater_x1 has no PLAYERCOLOR and is the
# submerged base, a second composite piece rather than the body.
#
# HAND-VERIFIED, and deliberately not a heuristic. Both candidate automatic
# gates were measured and rejected: `sequence_type == 3` plus a static
# same-angle_count delta also catches 8 p_bolt_fire_x1 projectile consts, where
# the flaming graphic is plausibly the right one; and thresholding on .sld MAIN
# coverage would put sprite decoding inside this generator, which is
# standalone-genieutils on purpose. A direct .sld-coverage scan bounds the real
# defect at these 3 consts out of 2,307, so an override table is both
# sufficient and auditable.
#
# The flag is genuinely part of a palisade wall and is LOST by this retarget.
# Recovering it needs the multi-piece delta compositing planned in
# descape-gate-composite.md; filed as a follow-up there, not solved here.
_BODY_GRAPHIC_OVERRIDES: dict[int, int] = {
    72: 587,    # Palisade Wall        -> b_dark_wall_palisade_x1
    119: 605,   # Fortified Palisade   -> b_scen_wall_palisade_fortified_x1
    788: 6595,  # Sea Wall             -> b_scen_wall_sea_x1
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "aoe2de_root",
        type=Path,
        help="Path to the AoE2DE install root (contains resources/_common/...)",
    )
    parser.add_argument(
        "--scan-sld",
        action="store_true",
        help="Also report how many entries have a readable .sld in this install",
    )
    args = parser.parse_args()

    from genieutils.datfile import DatFile

    dat_path = args.aoe2de_root / "resources/_common/dat/empires2_x2_p1.dat"
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    data = DatFile.parse(str(dat_path))
    graphics = data.graphics
    units = data.civs[0].units

    entries: dict[str, dict[str, object]] = {}
    overridden: dict[int, tuple[str, str]] = {}
    secondary = Counter()
    skipped = Counter()
    composited: set[int] = set()

    for unit_const, unit in enumerate(units):
        if unit is None:
            skipped["no_unit"] += 1
            continue
        standing = unit.standing_graphic
        graphic_id = standing[0] if standing else -1
        secondary["set" if len(standing) > 1 and standing[1] >= 0 else "unset"] += 1
        if graphic_id is None or graphic_id < 0 or graphic_id >= len(graphics):
            skipped["no_standing_graphic"] += 1
            continue
        graphic = graphics[graphic_id]
        if graphic is None or not graphic.file_name:
            skipped["no_file_name"] += 1
            continue
        override_id = _BODY_GRAPHIC_OVERRIDES.get(unit_const)
        if override_id is not None:
            body = graphics[override_id] if 0 <= override_id < len(graphics) else None
            if body is None or not body.file_name:
                raise SystemExit(
                    f"_BODY_GRAPHIC_OVERRIDES[{unit_const}] = {override_id} does not "
                    f"resolve to a named graphic in this .dat -- re-verify it rather "
                    f"than dropping it silently"
                )
            # Loud on purpose: a shell that stops being a decoration is a real
            # change in the .dat, not something to absorb quietly.
            if override_id not in {d.graphic_id for d in graphic.deltas}:
                raise SystemExit(
                    f"_BODY_GRAPHIC_OVERRIDES[{unit_const}] = {override_id} is no longer "
                    f"a delta of standing graphic {graphic.id} ({graphic.file_name}) -- "
                    f"re-verify against the .dat"
                )
            overridden[unit_const] = (graphic.file_name, body.file_name)
            graphic = body
        elif not _looks_modern(graphic.file_name):
            resolved = _resolve_modern_graphic(graphics, graphic_id)
            if resolved is None:
                skipped["legacy_no_modern_replacement"] += 1
                continue
            graphic = resolved
        entry: dict[str, object] = {
            "graphic_id": graphic.id,
            "file_name": graphic.file_name,
            "angle_count": graphic.angle_count,
            "mirroring_mode": graphic.mirroring_mode,
            "frame_count": graphic.frame_count,
        }
        pieces = _resolve_pieces(units, graphics, unit_const, graphic)
        if pieces is not None:
            entry["pieces"] = pieces
            composited.add(unit_const)
        entries[str(unit_const)] = entry

    out_path = Path(__file__).resolve().parent.parent / "descape" / "unit_graphic_map.json"
    out_path.write_text(
        json.dumps(
            {
                "_comment": (
                    "unit_const -> standing .sld graphic -- see "
                    "tools/gen_unit_graphic_map.py"
                ),
                "graphics": dict(sorted(entries.items(), key=lambda kv: int(kv[0]))),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(entries)} graphic entries to {out_path}")
    print(f"  units skipped: {dict(sorted(skipped.items()))}")
    print(f"  body-graphic overrides applied: {len(overridden)}/{len(_BODY_GRAPHIC_OVERRIDES)}")
    for const, (shell, body) in sorted(overridden.items()):
        print(f"    {const}: {shell} -> {body}")
    print(f"  standing_graphic[1]: {dict(sorted(secondary.items()))}")
    print(f"  composite (pieces) entries: {len(composited)}/{len(_COMPOSITE_SCOPE)} scoped consts")
    not_composited = sorted(_COMPOSITE_SCOPE - composited)
    if not_composited:
        print(f"    in scope but no entry (expected -- legacy shell): {not_composited}")

    if args.scan_sld:
        graphics_dir = args.aoe2de_root / "resources/_common/drs/graphics"
        status = Counter()
        distinct: dict[str, str] = {}
        for entry in entries.values():
            name = str(entry["file_name"])
            if name not in distinct:
                distinct[name] = _sld_status(graphics_dir / f"{name}.sld")
            status[distinct[name]] += 1
        by_file = Counter(distinct.values())
        total = sum(status.values())
        ok = status["ok"]
        print(f"  .sld scan of {graphics_dir}:")
        print(f"    consts with a readable .sld: {ok}/{total} ({100 * ok / total:.1f}%)")
        print(f"    by const: {dict(sorted(status.items()))}")
        print(f"    by distinct file: {dict(sorted(by_file.items()))}")


if __name__ == "__main__":
    main()
