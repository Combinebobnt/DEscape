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
      "rotation_is_variant": true,  -- OPTIONAL, omitted when false. The .dat's
                              -- unit.type != 70 (non-creatable) and the
                              -- resolved graphic's angle_count > 1, with the
                              -- trebuchet consts forced false (see
                              -- _EXTRA_ANGLE_CONSTS) -- descape.unit_sprites.
                              -- rotation_is_variant() reads this instead of
                              -- its own hand-kept frozensets (2026-09-06 Tier
                              -- B plan).
      "variant_count": int,  -- OPTIONAL, only beside rotation_is_variant:
                              -- the .sld's real frame count // frame_count,
                              -- omitted when the .sld is unreadable here.
                              -- The edit path's install-free cycle modulus
                              -- (descape.unit_variant); render still reads
                              -- the file itself.
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
         "dx": int, "dy": int,    -- native-pixel offset from the parent's
                                  -- own anchor; see "Composite buildings"
                                  -- below.
         "mx": float, "my": float,  -- _COMPOSITE_SCOPE only: the annex's raw
                                  -- misplacement in tiles, which dx/dy cannot
                                  -- be inverted back to (a town centre's
                                  -- cancels to (0, 0)); the depth-slot input.
         "slot": [int, int],     -- _COMPOSITE_SCOPE only: the footprint tile,
                                  -- offset from its low corner, whose depth
                                  -- moment paints this piece (_PIECE_SLOTS).
                                  -- Absent: paint at the unit's own anchor.
         "seeded": true,        -- OPTIONAL, _ANNEX_TREE_SCOPE only: the frame is a
                                  -- shape variant picked per placed unit.
         "parent": true},       -- OPTIONAL, on exactly ONE piece per entry:
                                  -- the piece whose failure to resolve drops
                                  -- the whole composite to the coloured mark.
                                  -- Explicit rather than inferred from
                                  -- unit_id == unit_const, which every piece
                                  -- of a direct-delta gate or a wall shares.
        ...                       -- Depth-sorted: draw in list order.
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
  entered under a filename that will never exist on any install. The one
  exception is the placed Pasture (1893/1897): its own graphic is this shell,
  but its annex TREE reaches modern art, so it gets an entry via
  `_resolve_annex_tree` (see "Pasture annex tree" below). 1894/1898 stay out.

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
through this exact mechanism (confirmed 2026-08-29), but are built by a
separate function (`_resolve_gate_pieces`, see "Gates" below), not folded
into this allowlist. 2078 (Pasture Annex Fences) never reaches the pieces
walk because its OWN standing_graphic is an unresolvable Farm-family legacy
shell; 1893/1897 have the same shell but are handled by the annex-tree walk.

**Pasture annex tree.** A placed Pasture (`_ANNEX_TREE_SCOPE`, 1893/1897) is a
two-level annex tree, not one level of annexes: root -> 1890 (hut, plus 4x
1888 posts at (+-2, +-2)) and root -> 2078 -> 2x 2079 at (0, +-2) and 2x 2080
at (+-2, 0), each with 4 fence annexes (1885/1886) at +-0.65 and +-1.3 along
its edge. `_resolve_annex_tree` walks it depth-first, summing misplacement,
and emits every node whose own graphic resolves: 25 pieces. The hut is the
parent and supplies the top-level fields, unlike a town centre whose parent
is its own art. Posts and fences are marked `seeded`: their frames are shape
variants, and the renderer picks one per placed unit rather than from the
root's rotation.

**Gates.** A gate (`class_ == 39`) is a 2-5 piece composite: a middle span
plus, on a real gate rather than a bare corner pillar, two corner towers and
two flags. Unlike a town centre, a gate's pieces are not one modern delta
per annex -- a corner pillar's OWN standing_graphic is itself a legacy shell
whose deltas point at BOTH its corner-tower graphic and its flag graphic,
so the first-match walk `_resolve_modern_graphic` uses everywhere else
silently drops the flag. `_resolve_all_modern_deltas` is the class-39-only
counterpart: it emits every modern delta descendant of a graphic, not just
the first, whether that graphic is a gate's own root (an X-state composite
gate encodes all 5 pieces as direct deltas of one legacy shell, no annexes
involved) or an annex's (a bare directional gate's own root is already
modern -- just the middle -- and its two corner-pillar annexes each
resolve to 2 pieces this way). Measured over all 97 class-39 consts, and
matching the generator's own "gate composite entries" report exactly: 96
resolve (24 at 2 pieces -- bare corner pillars and the four sea-gate consts,
both resolved via their own root; 72 at 5 pieces -- directional/composite
gates), the same single holdout as any other scoping (const 1192, a legacy
duplicate with no filename, already excluded above). A gate's pieces carry
no per-piece depth slot, unlike a town centre's, so they all paint at the
unit's single anchor tile.

**Per-civ and per-age building art (GH #48).** Every civ has its own copy of
the unit table, and building consts (type 80) carry different standing
graphics per architecture. The same `_entry_for()` runs over every civ's
table and writes a second file, descape/building_art_map.json:

    "civ_names":    {"<civ_index>": "<.dat civ name>"},
    "age_upgrades": {"<base_const>": {"3"|"4"|"5": const drawn at that
                     StartingAge}}, from the age techs' "upgrade unit"
                     commands; only ages where the const changes.
    "entries":      [entry, ...], unit_graphic_map.json-shaped, each
                     distinct one once.
    "civ_art":      {"<civ_index>": {"<const>": index into entries}}, only
                     where that civ's entry differs from Gaia's.

The run fails unless every per-civ entry and every age target agrees with
its reference on angle_count, rotation_is_variant, variant_count and the
unit's clearance_size, apart from the measured `_ART_INVARIANT_HOLDOUTS`.
A decoration-shell body is the delta at the same position in that civ's own
shell as the hand-verified Gaia body.

**Decoration-shell walls.** The three `_BODY_GRAPHIC_OVERRIDES` consts keep
the body as their top-level graphic but also carry `pieces`: the shell's
modern deltas in .dat list order at their own offsets, with the flag shell
itself at its `graphic_id == -1` delta's slot and offset, and the body marked
parent. See `_resolve_wall_pieces`.
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


def _sld_frame_count(path: Path) -> int | None:
    """The .sld header's own frame count, or None when the file isn't readable.

    Header only (magic, version, frame_count at offset 6), same literal-read
    reasoning as SLD_MAGIC above."""
    try:
        with path.open("rb") as handle:
            head = handle.read(12)
    except OSError:
        return None
    if len(head) < 12 or head[:4] != SLD_MAGIC:
        return None
    if int.from_bytes(head[10:12], "little") not in SLD_LAYOUT_READABLE:
        return None
    return int.from_bytes(head[6:8], "little")


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


def _resolve_all_modern_deltas(graphics: list, graphic_id: int) -> list[tuple[object, int, int]]:
    """Class-39-only counterpart to `_resolve_modern_graphic`: every modern
    delta descendant of graphic_id, each paired with its own delta offset --
    `[(graphic, 0, 0)]` alone if graphic_id already looks modern. See the
    module docstring's "Gates" section for why first-match is wrong here (it
    drops a corner pillar's flag). A single level of deltas is enough --
    measured zero deltas anywhere in class 39 point at a non-modern target,
    so this deliberately does not recurse the way `_resolve_modern_graphic`
    does."""
    if graphic_id is None or graphic_id < 0 or graphic_id >= len(graphics):
        return []
    graphic = graphics[graphic_id]
    if graphic is None:
        return []
    if _looks_modern(graphic.file_name):
        return [(graphic, 0, 0)]
    result = []
    for delta in graphic.deltas:
        dg_id = delta.graphic_id
        if dg_id is None or not (0 <= dg_id < len(graphics)):
            continue
        dg = graphics[dg_id]
        if dg is not None and _looks_modern(dg.file_name):
            result.append((dg, delta.offset_x, delta.offset_y))
    return result


def _resolve_gate_pieces(
    units: list, graphics: list, unit_const: int, unit
) -> list[tuple[int, object, int, int]] | None:
    """The composite pieces for a class-39 (gate-family) unit_const, as
    `(unit_id, graphic, dx, dy)` tuples in draw order, or None if nothing
    resolves. `dx, dy` are native-pixel offsets from the parent's own anchor,
    same convention as `_piece_screen_offset` below.

    Two sources, concatenated: the root's own `_resolve_all_modern_deltas`
    (a bare directional gate's already-modern middle graphic resolves to
    itself alone here; an X-state composite gate's legacy shell resolves to
    its full direct-delta piece list -- towers, middle, flags -- with no
    annexes involved at all), then each real building annex's own
    `_resolve_all_modern_deltas`, offset by `iso_screen(misplacement) + that
    delta's own offset` -- the same positioning rule
    `_piece_screen_offset` uses, generalized from "first delta" to "every
    delta". A corner-pillar const (e.g. 81) is not itself a gate but is
    class 39 and reached this same way when it is the unit being resolved
    directly, not just as an annex target.

    Deduped to at most one piece per distinct (dx, dy) offset, first-in-list
    wins -- the only class this affects is the four sea-gate consts, whose
    root deltas to "sides" and "...underwater" land on the same offset;
    "sides" is listed first and wins, "underwater" (which has no
    PLAYERCOLOR layer and is the submerged base, not a second real piece)
    drops."""
    standing = unit.standing_graphic
    root_graphic_id = standing[0] if standing else -1
    raw: list[tuple[int, object, int, int]] = [
        (unit_const, g, dx, dy)
        for g, dx, dy in _resolve_all_modern_deltas(graphics, root_graphic_id)
    ]

    building = unit.building
    if building is not None:
        for annex in building.annexes:
            if not (0 <= annex.unit_id < len(units)):
                continue
            annex_unit = units[annex.unit_id]
            if annex_unit is None:
                continue
            a_standing = annex_unit.standing_graphic
            a_graphic_id = a_standing[0] if a_standing else -1
            iso_dx = (annex.misplacement_x + annex.misplacement_y) * _NATIVE_HALF_W
            iso_dy = (annex.misplacement_y - annex.misplacement_x) * _NATIVE_HALF_H
            for g, ddx, ddy in _resolve_all_modern_deltas(graphics, a_graphic_id):
                raw.append((annex.unit_id, g, round(iso_dx + ddx), round(iso_dy + ddy)))

    seen_offsets: set[tuple[int, int]] = set()
    deduped: list[tuple[int, object, int, int]] = []
    for uid, g, dx, dy in raw:
        offset = (dx, dy)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        deduped.append((uid, g, dx, dy))
    return deduped or None


# The .dat's own unit.type value for a creatable (trainable/buildable) unit --
# re-literalised here per this module's own "standalone genieutils script"
# convention rather than imported; descape/unit_rotation.py is the emitted
# field's authority and consumer.
_CREATABLE_TYPE = 70

# Trebuchet, packed and unpacked: mobile units the .dat types as buildings
# (type 80, not creatable) whose rotation is nonetheless a facing, not a
# shape variant. Mirrors unit_rotation._EXTRA_ANGLE_CONSTS exactly; kept as a
# literal copy for the same standalone-script reason as _CREATABLE_TYPE above.
_EXTRA_ANGLE_CONSTS: frozenset[int] = frozenset({42, 331, 1690, 1691})

_GATE_CLASS = 39


# HAND-VERIFIED against the real .dat (2026-08-29), per the module docstring's
# "Composite buildings" section -- NOT "every unit_const whose annexes
# resolve", which also catches gate consts (their corner-pillar annexes
# resolve through this same _resolve_modern_graphic walk too, confirmed while
# building this table; that is deliberately a separate follow-on, not built
# here). 2078 is a farm-family legacy shell with no modern replacement for its
# OWN standing_graphic, so it never reaches the pieces walk; 1893/1897 have the
# same shell but reach modern art through _ANNEX_TREE_SCOPE below.
_COMPOSITE_SCOPE: frozenset[int] = frozenset({
    71, 109, 141, 142, 2275, 2276, 2277,  # town centres (all ages/civs)
    1889, 1890, 2079, 2080,               # pastures
})

# unit_const -> each piece's depth slot [sx, sy], in emitted piece order: the
# footprint tile, as an offset from the footprint's low corner, whose moment in
# the depth walk paints that piece. Output of tools/measure_piece_slots.py
# (needs the real .sld art, so it is committed rather than derived here);
# re-verified by tests/test_unit_graphic_map.py's corpus slot test. 1890/2079/
# 2080 have a [1, 1] span, so every piece collapses onto the one tile.
#
# Town centres are not here: their slots differ by art set, so they are in
# _PIECE_SLOTS_BY_ART below (GH #48 Slice 2).
_PIECE_SLOTS: dict[int, list[list[int]]] = {
    1889: [[3, 0], [0, 3], [0, 2], [0, 1], [0, 3]],
    1890: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    2079: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    2080: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    # 1893/1897: _ANNEX_TREE_SCOPE, identical trees on a [4, 4] footprint.
    1893: [[3, 0], [3, 0], [3, 0], [2, 0], [3, 1], [1, 0], [3, 2], [1, 0], [3, 2], [0, 0], [3, 3],
           [1, 2], [0, 2], [0, 1], [0, 2], [0, 1], [0, 3], [0, 2], [0, 3], [0, 3], [0, 3], [0, 3],
           [0, 3], [0, 3], [0, 3]],
    1897: [[3, 0], [3, 0], [3, 0], [2, 0], [3, 1], [1, 0], [3, 2], [1, 0], [3, 2], [0, 0], [3, 3],
           [1, 2], [0, 2], [0, 1], [0, 2], [0, 1], [0, 3], [0, 2], [0, 3], [0, 3], [0, 3], [0, 3],
           [0, 3], [0, 3], [0, 3]],
}

# The town-centre consts of _COMPOSITE_SCOPE, whose slots are keyed by art set.
_TOWN_CENTRE_SCOPE: frozenset[int] = frozenset({71, 109, 141, 142, 2275, 2276, 2277})

# Town-centre art set (its pieces' file names, in emitted order) -> each
# piece's depth slot, as _PIECE_SLOTS. Keyed by the whole set, not the parent
# file: the Persian age-2 back piece heads sets whose annexes differ, and they
# measure differently. Output of tools/measure_piece_slots.py.
_PIECE_SLOTS_BY_ART: dict[tuple[str, ...], list[list[int]]] = {
    (
        "b_afri_town_center_age2_main_x1",
        "b_afri_town_center_age2_back_x1",
        "b_afri_town_center_age2_center_x1",
        "b_afri_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_afri_town_center_age3_main_x1",
        "b_afri_town_center_age3_back_x1",
        "b_afri_town_center_age3_center_x1",
        "b_afri_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_afri_town_center_age4_main_x1",
        "b_afri_town_center_age4_back_x1",
        "b_afri_town_center_age4_center_x1",
        "b_afri_town_center_age4_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_ande_town_center_age2_main_x1",
        "b_ande_town_center_age2_back_x1",
        "b_ande_town_center_age2_center_x1",
        "b_ande_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_ande_town_center_age3_main_x1",
        "b_ande_town_center_age3_back_x1",
        "b_ande_town_center_age3_center_x1",
        "b_ande_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_ande_town_center_age4_main_x1",
        "b_ande_town_center_age4_back_x1",
        "b_ande_town_center_age4_center_x1",
        "b_ande_town_center_age4_front_x1",
    ): [[2, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_archaic_town_center_age1_main_x1",
        "b_archaic_town_center_age1_back_x1",
        "b_archaic_town_center_age1_center_x1",
        "b_archaic_town_center_age1_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_archaic_town_center_age1_main_x1",
        "b_persian_town_center_age2_back_x1",
        "b_archaic_town_center_age1_center_x1",
        "b_archaic_town_center_age1_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_asia_town_center_age2_main_x1",
        "b_asia_town_center_age2_back_x1",
        "b_asia_town_center_age2_center_x1",
        "b_asia_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_asia_town_center_age3_main_x1",
        "b_asia_town_center_age3_back_x1",
        "b_asia_town_center_age3_center_x1",
        "b_asia_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_asia_town_center_age4_main_x1",
        "b_asia_town_center_age4_back_x1",
        "b_asia_town_center_age4_center_x1",
        "b_asia_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_ceas_town_center_age2_main_x1",
        "b_ceas_town_center_age2_back_x1",
        "b_ceas_town_center_age2_center_x1",
        "b_ceas_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_ceas_town_center_age3_main_x1",
        "b_ceas_town_center_age3_back_x1",
        "b_ceas_town_center_age3_center_x1",
        "b_ceas_town_center_age3_front_x1",
    ): [[0, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_ceas_town_center_age4_main_x1",
        "b_ceas_town_center_age4_back_x1",
        "b_ceas_town_center_age4_center_x1",
        "b_ceas_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_dark_town_center_age1_main_x1",
        "b_dark_town_center_age1_back_x1",
        "b_dark_town_center_age1_center_x1",
        "b_dark_town_center_age1_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_dark_town_center_age1_main_x1",
        "b_persian_town_center_age2_back_x1",
        "b_dark_town_center_age1_center_x1",
        "b_dark_town_center_age1_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_east_town_center_age2_main_x1",
        "b_east_town_center_age2_back_x1",
        "b_east_town_center_age2_center_x1",
        "b_east_town_center_age2_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_east_town_center_age3_main_x1",
        "b_east_town_center_age3_back_x1",
        "b_east_town_center_age3_center_x1",
        "b_east_town_center_age3_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_east_town_center_age4_main_x1",
        "b_east_town_center_age4_back_x1",
        "b_east_town_center_age4_center_x1",
        "b_east_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_greek_town_center_age2_main_x1",
        "b_greek_town_center_age2_back_x1",
        "b_greek_town_center_age2_center_x1",
        "b_greek_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_greek_town_center_age3_main_x1",
        "b_greek_town_center_age3_back_x1",
        "b_greek_town_center_age3_center_x1",
        "b_greek_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_greek_town_center_age4_main_x1",
        "b_greek_town_center_age4_back_x1",
        "b_greek_town_center_age4_center_x1",
        "b_greek_town_center_age4_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_indi_town_center_age2_main_x1",
        "b_indi_town_center_age2_back_x1",
        "b_indi_town_center_age2_center_x1",
        "b_indi_town_center_age2_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_indi_town_center_age3_main_x1",
        "b_indi_town_center_age3_back_x1",
        "b_indi_town_center_age3_center_x1",
        "b_indi_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_indi_town_center_age4_main_x1",
        "b_indi_town_center_age4_back_x1",
        "b_indi_town_center_age4_center_x1",
        "b_indi_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_medi_town_center_age2_main_x1",
        "b_medi_town_center_age2_back_x1",
        "b_medi_town_center_age2_center_x1",
        "b_medi_town_center_age2_front_x1",
    ): [[1, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_medi_town_center_age3_main_x1",
        "b_medi_town_center_age3_back_x1",
        "b_medi_town_center_age3_center_x1",
        "b_medi_town_center_age3_front_x1",
    ): [[1, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_medi_town_center_age4_main_x1",
        "b_medi_town_center_age4_back_x1",
        "b_medi_town_center_age4_center_x1",
        "b_medi_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_meso_town_center_age2_main_x1",
        "b_meso_town_center_age2_back_x1",
        "b_meso_town_center_age2_center_x1",
        "b_meso_town_center_age2_front_x1",
    ): [[1, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_meso_town_center_age3_main_x1",
        "b_meso_town_center_age3_back_x1",
        "b_meso_town_center_age3_center_x1",
        "b_meso_town_center_age3_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_meso_town_center_age4_main_x1",
        "b_meso_town_center_age4_back_x1",
        "b_meso_town_center_age4_center_x1",
        "b_meso_town_center_age4_front_x1",
    ): [[1, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_nors_town_center_age2_main_x1",
        "b_nors_town_center_age2_back_x1",
        "b_nors_town_center_age2_center_x1",
        "b_nors_town_center_age2_front_x1",
    ): [[2, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_nors_town_center_age3_main_x1",
        "b_nors_town_center_age3_back_x1",
        "b_nors_town_center_age3_center_x1",
        "b_nors_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_nors_town_center_age4_main_x1",
        "b_nors_town_center_age4_back_x1",
        "b_nors_town_center_age4_center_x1",
        "b_nors_town_center_age4_front_x1",
    ): [[1, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_orie_town_center_age2_main_x1",
        "b_orie_town_center_age2_back_x1",
        "b_orie_town_center_age2_center_x1",
        "b_orie_town_center_age2_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_orie_town_center_age3_main_x1",
        "b_orie_town_center_age3_back_x1",
        "b_orie_town_center_age3_center_x1",
        "b_orie_town_center_age3_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_orie_town_center_age4_main_x1",
        "b_orie_town_center_age4_back_x1",
        "b_orie_town_center_age4_center_x1",
        "b_orie_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_persian_town_center_age2_main_x1",
        "b_persian_town_center_age2_back_x1",
        "b_persian_town_center_age2_center_x1",
        "b_persian_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_persian_town_center_age3_main_x1",
        "b_persian_town_center_age3_back_x1",
        "b_persian_town_center_age3_center_x1",
        "b_persian_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_persian_town_center_age4_main_x1",
        "b_persian_town_center_age4_back_x1",
        "b_persian_town_center_age4_center_x1",
        "b_persian_town_center_age4_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_puru_town_center_age2_main_x1",
        "b_puru_town_center_age2_back_x1",
        "b_puru_town_center_age2_center_x1",
        "b_puru_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_puru_town_center_age3_Main_x1",
        "b_puru_town_center_age3_Back_x1",
        "b_puru_town_center_age3_Center_x1",
        "b_puru_town_center_age3_Front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_puru_town_center_age4_main_x1",
        "b_puru_town_center_age4_back_x1",
        "b_puru_town_center_age4_Center_x1",
        "b_puru_town_center_age4_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_seas_town_center_age2_main_x1",
        "b_seas_town_center_age2_back_x1",
        "b_seas_town_center_age2_center_x1",
        "b_seas_town_center_age2_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_seas_town_center_age3_main_x1",
        "b_seas_town_center_age3_back_x1",
        "b_seas_town_center_age3_center_x1",
        "b_seas_town_center_age3_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_seas_town_center_age4_main_x1",
        "b_seas_town_center_age4_back_x1",
        "b_seas_town_center_age4_center_x1",
        "b_seas_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_slav_town_center_age2_main_x1",
        "b_slav_town_center_age2_back_x1",
        "b_slav_town_center_age2_center_x1",
        "b_slav_town_center_age2_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_slav_town_center_age3_main_x1",
        "b_slav_town_center_age3_back_x1",
        "b_slav_town_center_age3_center_x1",
        "b_slav_town_center_age3_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_slav_town_center_age4_main_x1",
        "b_slav_town_center_age4_back_x1",
        "b_slav_town_center_age4_center_x1",
        "b_slav_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
    (
        "b_thracian_town_center_age2_main_x1",
        "b_thracian_town_center_age2_back_x1",
        "b_thracian_town_center_age2_center_x1",
        "b_thracian_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_thracian_town_center_age3_main_x1",
        "b_thracian_town_center_age3_back_x1",
        "b_thracian_town_center_age3_center_x1",
        "b_thracian_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_thracian_town_center_age4_main_x1",
        "b_thracian_town_center_age4_back_x1",
        "b_thracian_town_center_age4_center_x1",
        "b_thracian_town_center_age4_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_west_town_center_age2_main_x1",
        "b_west_town_center_age2_back_x1",
        "b_west_town_center_age2_center_x1",
        "b_west_town_center_age2_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_west_town_center_age3_main_x1",
        "b_west_town_center_age3_back_x1",
        "b_west_town_center_age3_center_x1",
        "b_west_town_center_age3_front_x1",
    ): [[1, 2], [0, 3], [0, 3], [0, 3]],
    (
        "b_west_town_center_age4_main_x1",
        "b_west_town_center_age4_back_x1",
        "b_west_town_center_age4_center_x1",
        "b_west_town_center_age4_front_x1",
    ): [[0, 3], [0, 3], [0, 3], [0, 3]],
}

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
    units: list, graphics: list, unit_const: int, parent_graphic,
    annex_upgrades: dict[int, int] | None = None,
) -> list[dict[str, object]] | None:
    """The composite `pieces` list for unit_const, depth-sorted (main -> back
    -> center -> front for a town centre), or None if unit_const is out of
    scope or has no resolvable annex. `parent_graphic` is the already-resolved
    graphic the caller is about to emit as the entry's own five fields; it
    becomes pieces[0]'s data at (dx, dy) = (0, 0), since a const's own art has
    zero misplacement by definition -- the parent piece is drawn like any
    other, just at its own depth slot rather than appended as an "extra".

    `annex_upgrades` maps an annex unit_id to the one drawn instead (GH #48
    Slice 2): a Feudal-age town centre (71) still lists the Dark Age annexes
    618-620 in the .dat, and the age techs upgrade those to 614-616 at the
    same age, so its pieces are resolved through the same age table."""
    if unit_const not in _COMPOSITE_SCOPE:
        return None
    building = units[unit_const].building
    annex_upgrades = annex_upgrades or {}
    pieces = [
        (0.0, {
            "unit_id": unit_const,
            "file_name": parent_graphic.file_name,
            "angle_count": parent_graphic.angle_count,
            "frame_count": parent_graphic.frame_count,
            "dx": 0,
            "dy": 0,
            "mx": 0.0,
            "my": 0.0,
            "parent": True,
        })
    ]
    for annex in building.annexes:
        annex_id = annex_upgrades.get(annex.unit_id, annex.unit_id)
        if not (0 <= annex_id < len(units)):
            continue
        annex_unit = units[annex_id]
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
            "unit_id": annex_id,
            "file_name": piece_graphic.file_name,
            "angle_count": piece_graphic.angle_count,
            "frame_count": piece_graphic.frame_count,
            "dx": dx,
            "dy": dy,
            "mx": round(float(annex.misplacement_x), 6),
            "my": round(float(annex.misplacement_y), 6),
        }))

    if len(pieces) < 2:
        raise SystemExit(
            f"_COMPOSITE_SCOPE includes {unit_const} but its annexes no longer "
            f"resolve to any modern piece -- re-verify against the .dat rather "
            f"than silently dropping the composite"
        )
    pieces.sort(key=lambda dp: dp[0])
    result = [p for _, p in pieces]
    # Keyed on scope membership, never on "has pieces": gates and walls carry
    # pieces too, and deliberately no slot.
    if unit_const in _TOWN_CENTRE_SCOPE:
        key: object = tuple(str(p["file_name"]) for p in result)
        slots = _PIECE_SLOTS_BY_ART.get(key)
        table = "_PIECE_SLOTS_BY_ART"
    else:
        key = unit_const
        slots = _PIECE_SLOTS.get(unit_const)
        table = "_PIECE_SLOTS"
    if slots is None or len(slots) != len(result):
        raise SystemExit(
            f"{table} has {'no' if slots is None else len(slots)} slot(s) for "
            f"{key!r} (_COMPOSITE_SCOPE const {unit_const}), which emits {len(result)} "
            f"pieces -- re-run tools/measure_piece_slots.py and commit its table"
        )
    for piece, slot in zip(result, slots, strict=True):
        piece["slot"] = list(slot)
    return result


# HAND-VERIFIED against the real .dat (2026-09-21, GH #66): the placeable
# Pasture consts, whose own standing_graphic is the Farm-family FARM0NNG shell
# but whose annex TREE carries the hut, posts and fences. See _resolve_annex_tree.
_ANNEX_TREE_SCOPE: frozenset[int] = frozenset({1893, 1897})

# The hut (1890's own graphic) supplies the entry's top-level fields.
_ANNEX_TREE_PARENT_UNIT = 1890

# Pieces an in-scope root must yield: hut + 4 posts + 4 edges x 5 fences.
_ANNEX_TREE_PIECE_COUNT = 25


def _resolve_annex_tree(units: list, graphics: list, unit_const: int):
    """(parent_graphic, pieces) for an _ANNEX_TREE_SCOPE const, pieces
    depth-sorted like _resolve_pieces().

    A DFS over building.annexes, summing misplacement down the tree: every
    node whose OWN standing_graphic resolves modern emits a piece at the summed
    misplacement; a legacy node (the root, 2078) emits nothing but its annexes
    still walk. Unlike a town centre, the parent is an annex's art (1890's
    hut), not the const's own. Every piece but the hut is `seeded`: its
    variant is picked per placed unit, not by the root's rotation."""
    found: list[tuple[int, float, float, object]] = []

    def walk(uid: int, mx: float, my: float, path: frozenset[int]) -> None:
        unit = units[uid]
        if unit is None:
            return
        if uid != unit_const:
            standing = unit.standing_graphic
            graphic = _resolve_modern_graphic(graphics, standing[0] if standing else -1)
            if graphic is not None:
                found.append((uid, round(mx, 6), round(my, 6), graphic))
        if unit.building is None:
            return
        for annex in unit.building.annexes:
            child = annex.unit_id
            # Per-path guard: identical sibling annexes (four 1888 posts) must all walk.
            if not (0 <= child < len(units)) or child in path:
                continue
            walk(child, mx + annex.misplacement_x, my + annex.misplacement_y, path | {child})

    walk(unit_const, 0.0, 0.0, frozenset({unit_const}))
    if len(found) != _ANNEX_TREE_PIECE_COUNT:
        raise SystemExit(
            f"_ANNEX_TREE_SCOPE const {unit_const} yields {len(found)} pieces, not "
            f"{_ANNEX_TREE_PIECE_COUNT} -- re-verify its annex tree against the .dat"
        )
    parents = [f for f in found if f[0] == _ANNEX_TREE_PARENT_UNIT]
    if len(parents) != 1:
        raise SystemExit(
            f"_ANNEX_TREE_SCOPE const {unit_const} has {len(parents)} "
            f"{_ANNEX_TREE_PARENT_UNIT} hut pieces, expected exactly 1"
        )
    pieces = []
    for uid, mx, my, graphic in found:
        dx, dy = _piece_screen_offset(mx, my, graphic)
        is_parent = uid == _ANNEX_TREE_PARENT_UNIT
        pieces.append((my - mx, {
            "unit_id": uid,
            "file_name": graphic.file_name,
            "angle_count": graphic.angle_count,
            "frame_count": graphic.frame_count,
            "dx": dx,
            "dy": dy,
            "mx": float(mx),
            "my": float(my),
            **({"parent": True} if is_parent else {"seeded": True}),
        }))
    pieces.sort(key=lambda dp: dp[0])
    result = [p for _, p in pieces]
    slots = _PIECE_SLOTS.get(unit_const)
    if slots is None or len(slots) != len(result):
        raise SystemExit(
            f"_PIECE_SLOTS has {'no' if slots is None else len(slots)} slot(s) for "
            f"_ANNEX_TREE_SCOPE const {unit_const}, which emits {len(result)} pieces -- "
            f"re-run tools/measure_piece_slots.py and commit its table"
        )
    for piece, slot in zip(result, slots, strict=True):
        piece["slot"] = list(slot)
    return parents[0][3], result


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
# The retarget only picks the entry's top-level graphic. The flag, and 788's
# submerged base, come back as composite pieces via _resolve_wall_pieces below
# (2026-09-02 wall-flag plan): the body stays the parent, so a consumer that
# ignores `pieces` still draws the wall.
_BODY_GRAPHIC_OVERRIDES: dict[int, int] = {
    72: 587,    # Palisade Wall        -> b_dark_wall_palisade_x1
    119: 605,   # Fortified Palisade   -> b_scen_wall_palisade_fortified_x1
    788: 6595,  # Sea Wall             -> b_scen_wall_sea_x1
}


def _resolve_wall_pieces(
    graphics: list, unit_const: int, shell, body_id: int
) -> list[dict[str, object]]:
    """The composite pieces for a _BODY_GRAPHIC_OVERRIDES const: every modern
    delta of the decoration shell in .dat list order at its own offset, with
    the shell itself (the flag) placed by its `graphic_id == -1` delta, the
    engine's slot for the parent graphic. 72 gives body -> flag at (0, -40),
    119 the same at (0, -60), 788 underwater -> body -> flag at (0, -60).
    At (0, 0) the flag's pole base sits on the ground anchor, mid-wall (GH #51).

    Not the gate helpers: their one-piece-per-offset dedupe would collapse
    the body pieces, which all sit at (0, 0). unit_id is unit_const on every
    piece, so the parent is marked on the body explicitly."""
    shell_piece = {
        "unit_id": unit_const,
        "file_name": shell.file_name,
        "angle_count": shell.angle_count,
        "frame_count": shell.frame_count,
    }
    slots = [d for d in shell.deltas if d.graphic_id == -1]
    if len(slots) != 1:
        raise SystemExit(
            f"decoration shell {shell.file_name!r} (unit_const {unit_const}) has "
            f"{len(slots)} graphic_id -1 deltas, expected exactly 1 to place it by"
        )
    pieces: list[dict[str, object]] = []
    for delta in shell.deltas:
        dg_id = delta.graphic_id
        if dg_id == -1:
            pieces.append({**shell_piece, "dx": delta.offset_x, "dy": delta.offset_y})
            continue
        if dg_id is None or not (0 <= dg_id < len(graphics)):
            continue
        dg = graphics[dg_id]
        if dg is None or not _looks_modern(dg.file_name):
            continue
        pieces.append({
            "unit_id": unit_const,
            "file_name": dg.file_name,
            "angle_count": dg.angle_count,
            "frame_count": dg.frame_count,
            "dx": delta.offset_x,
            "dy": delta.offset_y,
            **({"parent": True} if dg_id == body_id else {}),
        })
    for piece in pieces:
        # Same guard as the gate branch's angle_count == 1, inverted: a wall
        # piece that stops being a 5-shape variant graphic is a real .dat change.
        if piece["angle_count"] != 5:
            raise SystemExit(
                f"wall piece {piece['file_name']!r} (unit_const {unit_const}) has "
                f"angle_count {piece['angle_count']}, not 5 -- re-verify "
                f"_BODY_GRAPHIC_OVERRIDES against the .dat"
            )
    return pieces


def _body_delta_positions(units: list, graphics: list) -> dict[int, int]:
    """unit_const -> the index, in its decoration shell's `deltas`, of the
    body `_BODY_GRAPHIC_OVERRIDES` names on the Gaia table. Every other civ's
    body is the delta at that same position in ITS shell (GH #48), so the
    hand-verified table stays a Gaia graphic id and the per-civ body is derived
    by relationship rather than listed."""
    positions: dict[int, int] = {}
    for unit_const, override_id in _BODY_GRAPHIC_OVERRIDES.items():
        unit = units[unit_const]
        standing = unit.standing_graphic if unit is not None else (-1,)
        shell = graphics[standing[0]] if 0 <= standing[0] < len(graphics) else None
        body = graphics[override_id] if 0 <= override_id < len(graphics) else None
        if body is None or not body.file_name:
            raise SystemExit(
                f"_BODY_GRAPHIC_OVERRIDES[{unit_const}] = {override_id} does not "
                f"resolve to a named graphic in this .dat -- re-verify it rather "
                f"than dropping it silently"
            )
        # Loud on purpose: a shell that stops being a decoration is a real
        # change in the .dat, not something to absorb quietly.
        ids = [] if shell is None else [d.graphic_id for d in shell.deltas]
        if ids.count(override_id) != 1:
            raise SystemExit(
                f"_BODY_GRAPHIC_OVERRIDES[{unit_const}] = {override_id} is no longer "
                f"exactly one delta of standing graphic {standing[0]} -- "
                f"re-verify against the .dat"
            )
        positions[unit_const] = ids.index(override_id)
    return positions


def _entry_for(
    units: list, graphics: list, graphics_dir: Path, unit_const: int,
    body_positions: dict[int, int],
    annex_upgrades: dict[int, dict[int, int]] | None = None,
) -> tuple[dict[str, object] | None, str]:
    """(entry, kind) for one const of one civ's unit table: the entry this
    script emits for it, or None with `kind` naming the skip reason. `kind` is
    otherwise "gate", "tree", "override", "composite" or "plain". Run for
    every civ (GH #48), so nothing in here may read a Gaia-only fact.
    `annex_upgrades` is `_annex_upgrades()`'s table, keyed by composite const."""
    unit = units[unit_const]
    if unit is None:
        return None, "no_unit"
    standing = unit.standing_graphic
    graphic_id = standing[0] if standing else -1

    if unit.class_ == _GATE_CLASS:
        gate_pieces = _resolve_gate_pieces(units, graphics, unit_const, unit)
        if gate_pieces is None:
            return None, "gate_no_modern_replacement"
        for _, piece_graphic, _, _ in gate_pieces:
            if piece_graphic.angle_count != 1:
                raise SystemExit(
                    f"class-39 piece {piece_graphic.file_name!r} (unit_const "
                    f"{unit_const}) has angle_count {piece_graphic.angle_count}, "
                    f"not 1 -- gate pieces are assumed non-rotating; re-verify "
                    f"tools/gen_unit_graphic_map.py's gate section against the "
                    f".dat before shipping this"
                )
        # The parent is pieces[0] because it supplies the entry's top-level
        # fields. For an X-state gate (e.g. 487) that is a corner pillar, not
        # the middle span; retargeting the top-level fields changes which
        # piece's failure drops the whole gate.
        _, primary, _, _ = gate_pieces[0]
        return {
            "graphic_id": primary.id,
            "file_name": primary.file_name,
            "angle_count": primary.angle_count,
            "mirroring_mode": primary.mirroring_mode,
            "frame_count": primary.frame_count,
            "pieces": [
                {
                    "unit_id": uid,
                    "file_name": g.file_name,
                    "angle_count": g.angle_count,
                    "frame_count": g.frame_count,
                    "dx": dx,
                    "dy": dy,
                    **({"parent": True} if i == 0 else {}),
                }
                for i, (uid, g, dx, dy) in enumerate(gate_pieces)
            ],
        }, "gate"

    if unit_const in _ANNEX_TREE_SCOPE:
        hut, tree_pieces = _resolve_annex_tree(units, graphics, unit_const)
        return {
            "graphic_id": hut.id,
            "file_name": hut.file_name,
            "angle_count": hut.angle_count,
            "mirroring_mode": hut.mirroring_mode,
            "frame_count": hut.frame_count,
            "pieces": tree_pieces,
        }, "tree"

    if graphic_id is None or graphic_id < 0 or graphic_id >= len(graphics):
        return None, "no_standing_graphic"
    graphic = graphics[graphic_id]
    if graphic is None or not graphic.file_name:
        return None, "no_file_name"
    position = body_positions.get(unit_const)
    wall_pieces = None
    if position is not None:
        body_id = graphic.deltas[position].graphic_id if position < len(graphic.deltas) else -1
        body = graphics[body_id] if 0 <= body_id < len(graphics) else None
        if body is None or not _looks_modern(body.file_name):
            raise SystemExit(
                f"unit_const {unit_const}'s shell {graphic.file_name!r} has no modern "
                f"body at delta position {position} -- re-verify _BODY_GRAPHIC_OVERRIDES"
            )
        wall_pieces = _resolve_wall_pieces(graphics, unit_const, graphic, body_id)
        graphic = body
    elif not _looks_modern(graphic.file_name):
        resolved = _resolve_modern_graphic(graphics, graphic_id)
        if resolved is None:
            return None, "legacy_no_modern_replacement"
        graphic = resolved
    entry: dict[str, object] = {
        "graphic_id": graphic.id,
        "file_name": graphic.file_name,
        "angle_count": graphic.angle_count,
        "mirroring_mode": graphic.mirroring_mode,
        "frame_count": graphic.frame_count,
    }
    if unit_const not in _EXTRA_ANGLE_CONSTS and (
        unit.type != _CREATABLE_TYPE and graphic.angle_count > 1
    ):
        entry["rotation_is_variant"] = True
        frames = _sld_frame_count(graphics_dir / f"{graphic.file_name}.sld")
        if frames is not None and graphic.frame_count > 0:
            entry["variant_count"] = frames // graphic.frame_count
    if wall_pieces is not None:
        entry["pieces"] = wall_pieces
        return entry, "override"
    pieces = _resolve_pieces(
        units, graphics, unit_const, graphic, (annex_upgrades or {}).get(unit_const)
    )
    if pieces is not None:
        entry["pieces"] = pieces
        return entry, "composite"
    return entry, "plain"


# .dat type of a building const. GH #48's per-civ art is scoped to these.
_BUILDING_TYPE = 80

# The age-up techs, in order, and the StartingAge value each one reaches. The
# .dat's own tech names are shifted by one: 101 is named "Middle Age".
_AGE_TECHS: tuple[tuple[int, int], ...] = ((101, 3), (102, 4), (103, 5))

# Effect command type "upgrade unit": a -> b.
_UPGRADE_UNIT_COMMAND = 3


def _age_upgrades(data, building_consts: set[int]) -> dict[int, dict[int, int]]:
    """base const -> {age: const drawn at that age}, ages 3..5, only where it
    differs from the base. Each "upgrade unit" command of the age techs, in
    order, replaces the player's slot for exactly its source const `a`, so a
    slot is retargeted by every later command naming it and by nothing else:
    base_id does not chain (498's base_id is 498), and a directly placed 463
    is never a source, so it draws its own art."""
    slots: dict[int, int] = {}
    result: dict[int, dict[int, int]] = {}
    for tech_id, age in _AGE_TECHS:
        effect = data.effects[data.techs[tech_id].effect_id]
        for command in effect.effect_commands:
            if command.type != _UPGRADE_UNIT_COMMAND:
                continue
            if command.a in building_consts and command.b in building_consts:
                slots[command.a] = command.b
        for base, target in slots.items():
            if target != base:
                result.setdefault(base, {})[age] = target
    return result


# The fields an entry's const-keyed logic reads (rotation dispatch, variant
# cycling, footprints): they must not change with the art.
_ART_INVARIANT_FIELDS = ("angle_count", "rotation_is_variant", "variant_count")


# Building consts whose per-civ art breaks _ART_INVARIANT_FIELDS, measured
# 2026-10-07. The civs whose art breaks it keep Gaia's art for that const; its
# other civs keep their own. The generator fails unless the measured
# violators are exactly these, so a new one is still loud.
_ART_INVARIANT_HOLDOUTS: dict[int, str] = {
    155: "Fortified Wall: Greek art (civs 47, 48, 54) stores 11 variants, not 5",
    446: "PORT, hidden in the editor: a projectile graphic, angle_count 32 vs 1",
}


def _art_invariant_violations(
    what: str, entry: dict, unit, ref_entry: dict, ref_unit,
) -> list[str]:
    """Every way `entry` (for `unit`) disagrees with the reference on a field
    the placed const's own logic reads. The AGENTS.md hard rules (walls
    angle_count 5, gates 1) were measured on Gaia; this is what extends them
    to every civ and every age target (GH #48)."""
    found = [
        f"{what}: {field} {entry.get(field)!r} != reference {ref_entry.get(field)!r}"
        for field in _ART_INVARIANT_FIELDS
        if entry.get(field) != ref_entry.get(field)
    ]
    if tuple(unit.clearance_size) != tuple(ref_unit.clearance_size):
        found.append(
            f"{what}: clearance_size {tuple(unit.clearance_size)} != reference "
            f"{tuple(ref_unit.clearance_size)}"
        )
    return found


# .dat class of a wall (gates are _GATE_CLASS).
_WALL_CLASS = 27


def _structure_violations(what: str, entry: dict | None, unit, gaia_entry: dict | None, gaia_unit) -> list[str]:
    """GH #48 Slice 3: only a wall's or gate's art may vary by civ. The gate
    sibling groups (gate_orientation, from each const's code and class) and the
    wall-connector set are keyed on the const, so the class, the code and
    whether the const resolves at all must match Gaia's in every civ, and a
    composite must still mark exactly one parent piece."""
    found = []
    if gaia_unit is not None and gaia_unit.class_ in (_GATE_CLASS, _WALL_CLASS):
        if unit is None or unit.class_ != gaia_unit.class_:
            found.append(f"{what}: class {getattr(unit, 'class_', None)} != Gaia's {gaia_unit.class_}")
        elif gaia_unit.class_ == _GATE_CLASS and unit.name != gaia_unit.name:
            found.append(f"{what}: gate code {unit.name!r} != Gaia's {gaia_unit.name!r}")
        if (entry is None) != (gaia_entry is None):
            found.append(f"{what}: resolves {'nothing' if entry is None else 'a graphic'}, unlike Gaia")
    if entry is not None and "pieces" in entry:
        parents = sum(1 for piece in entry["pieces"] if piece.get("parent"))
        if parents != 1:
            found.append(f"{what}: {parents} parent pieces, not exactly 1")
    return found


def _annex_upgrades(ages: dict[int, dict[int, int]]) -> dict[int, dict[int, int]]:
    """composite const -> {annex unit_id: the one drawn instead}, for every
    const the age table reaches at some age: its annexes are upgraded by the
    same table at that same age (Slice 2; see _resolve_pieces())."""
    age_of: dict[int, int] = {}
    for by_age in ages.values():
        for age, target in by_age.items():
            if age_of.setdefault(target, age) != age:
                raise SystemExit(f"const {target} is an age target at two ages -- re-verify")
    return {
        target: {base: by_age[age] for base, by_age in ages.items() if age in by_age}
        for target, age in age_of.items()
        if target in _COMPOSITE_SCOPE
    }


def _building_art(
    data, graphics: list, graphics_dir: Path, body_positions: dict[int, int],
    gaia_entries: dict[str, dict[str, object]], building_consts: set[int],
    ages: dict[int, dict[int, int]], annex_upgrades: dict[int, dict[int, int]],
) -> dict[str, object]:
    """descape/building_art_map.json's content (GH #48): the age table, every
    civ's building entries that differ from Gaia's, and the civ names."""
    gaia_units = data.civs[0].units

    civ_art: dict[str, dict[str, dict[str, object]]] = {}
    no_gaia_entry: set[int] = set()
    violations: list[str] = []
    held_out: set[int] = set()
    # Gates carry no age art: their orientation siblings are keyed on the const.
    violations += [
        f"age table: gate const {c} is upgraded by an age tech"
        for base, by_age in ages.items()
        for c in (base, *by_age.values())
        if gaia_units[c].class_ == _GATE_CLASS
    ]
    for civ_index, civ in enumerate(data.civs):
        units = civ.units
        art: dict[str, dict[str, object]] = {}
        for const in sorted(building_consts):
            entry, _kind = _entry_for(
                units, graphics, graphics_dir, const, body_positions, annex_upgrades
            )
            gaia_entry = gaia_entries.get(str(const))
            violations += _structure_violations(
                f"civ {civ_index} const {const}", entry, units[const], gaia_entry, gaia_units[const]
            )
            if entry is None:
                continue
            if gaia_entry is None:
                # Const 183: no Gaia graphic to fall back from or check against.
                no_gaia_entry.add(const)
                continue
            found = _art_invariant_violations(
                f"civ {civ_index} const {const}", entry, units[const],
                gaia_entry, gaia_units[const],
            )
            if found and const in _ART_INVARIANT_HOLDOUTS:
                # Only this civ's art is held back; the rest of the const's civs keep theirs.
                held_out.add(const)
                continue
            violations += found
            if civ_index and entry != gaia_entry:
                art[str(const)] = entry
        for base, by_age in ages.items():
            base_entry = art.get(str(base)) or gaia_entries.get(str(base))
            for age, target in by_age.items():
                target_entry = art.get(str(target)) or gaia_entries.get(str(target))
                if base_entry is None or target_entry is None:
                    raise SystemExit(
                        f"civ {civ_index}: age {age} upgrade {base} -> {target} has no "
                        f"entry on one side -- re-verify the age techs against the .dat"
                    )
                violations += _art_invariant_violations(
                    f"civ {civ_index} age {age} {base} -> {target}", target_entry,
                    units[target], base_entry, units[base],
                )
        if art:
            civ_art[str(civ_index)] = art
    if violations or held_out != set(_ART_INVARIANT_HOLDOUTS):
        raise SystemExit(
            "per-civ/per-age building art breaks the placed const's own logic -- "
            "re-verify against the .dat:\n  " + "\n  ".join(violations[:40])
            + f"\n  holdouts measured {sorted(held_out)}, listed {sorted(_ART_INVARIANT_HOLDOUTS)}"
        )
    print(f"  building consts: {len(building_consts)}; age-upgraded bases: {len(ages)}")
    differing = {c for art in civ_art.values() for c in art}
    print(f"  building consts whose art differs by civ: {len(differing)}")
    print(f"    held out on Gaia art (invariant break): {sorted(held_out)}")
    if no_gaia_entry:
        print(f"    skipped, no Gaia entry to fall back from: {sorted(no_gaia_entry)}")
    # Civs sharing an art set share entries, so each distinct one is stored once.
    shared: list[dict[str, object]] = []
    index_of: dict[str, int] = {}
    civ_art_indexed: dict[str, dict[str, int]] = {}
    for civ_key, art in civ_art.items():
        for const_key, entry in art.items():
            key = json.dumps(entry, sort_keys=True)
            if key not in index_of:
                index_of[key] = len(shared)
                shared.append(entry)
            civ_art_indexed.setdefault(civ_key, {})[const_key] = index_of[key]
    return {
        "_comment": (
            "GH #48 per-civ and per-age building art -- see "
            "tools/gen_unit_graphic_map.py"
        ),
        "civ_names": {str(i): civ.name for i, civ in enumerate(data.civs)},
        "age_upgrades": {
            str(base): {str(age): target for age, target in sorted(by_age.items())}
            for base, by_age in sorted(ages.items())
        },
        "entries": shared,
        "civ_art": civ_art_indexed,
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "descape",
        help="Where to write both JSON tables (default: descape/)",
    )
    args = parser.parse_args()

    from genieutils.datfile import DatFile

    dat_path = args.aoe2de_root / "resources/_common/dat/empires2_x2_p1.dat"
    if not dat_path.is_file():
        raise SystemExit(f"Not found: {dat_path}")

    graphics_dir = args.aoe2de_root / "resources/_common/drs/graphics"
    data = DatFile.parse(str(dat_path))
    graphics = data.graphics
    units = data.civs[0].units
    body_positions = _body_delta_positions(units, graphics)
    building_consts = {
        c for c, u in enumerate(units) if u is not None and u.type == _BUILDING_TYPE
    }
    ages = _age_upgrades(data, building_consts)
    annex_upgrades = _annex_upgrades(ages)

    entries: dict[str, dict[str, object]] = {}
    overridden: dict[int, tuple[str, str]] = {}
    secondary = Counter()
    skipped = Counter()
    composited: set[int] = set()
    gate_composited: set[int] = set()
    tree_composited: set[int] = set()

    for unit_const, unit in enumerate(units):
        if unit is not None:
            standing = unit.standing_graphic
            secondary["set" if len(standing) > 1 and standing[1] >= 0 else "unset"] += 1
        entry, kind = _entry_for(
            units, graphics, graphics_dir, unit_const, body_positions, annex_upgrades
        )
        if entry is None:
            skipped[kind] += 1
            continue
        entries[str(unit_const)] = entry
        if kind == "gate":
            gate_composited.add(unit_const)
        elif kind == "tree":
            tree_composited.add(unit_const)
        elif kind == "composite":
            composited.add(unit_const)
        elif kind == "override":
            shell = graphics[unit.standing_graphic[0]]
            overridden[unit_const] = (shell.file_name, str(entry["file_name"]))

    for unit_const, entry in entries.items():
        parents = sum(1 for piece in entry.get("pieces", ()) if piece.get("parent"))
        if "pieces" in entry and parents != 1:
            raise SystemExit(
                f"unit_const {unit_const} emitted {parents} parent pieces, not exactly 1 -- "
                f"sprite_pieces_for() bails on the parent, so this must be unambiguous"
            )

    building_art = _building_art(
        data, graphics, graphics_dir, body_positions, entries, building_consts, ages, annex_upgrades
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    art_path = args.out_dir / "building_art_map.json"
    art_path.write_text(json.dumps(building_art, indent=1) + "\n")
    print(f"Wrote {art_path} ({art_path.stat().st_size} bytes)")

    out_path = args.out_dir / "unit_graphic_map.json"
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
    print(f"  annex-tree entries: {sorted(tree_composited)}")
    gate_piece_counts = Counter(len(entries[str(c)]["pieces"]) for c in gate_composited)
    print(f"  gate (class {_GATE_CLASS}) composite entries: {len(gate_composited)}")
    print(f"    piece-count histogram: {dict(sorted(gate_piece_counts.items()))}")

    if args.scan_sld:
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
