"""Verifies descape/unit_graphic_map.json -- the unit_const -> .sld graphic table.

**Why this file exists.** The table is generated (tools/gen_unit_graphic_map.py)
and, until the sprite renderer consumes it, nothing else in the suite would
notice if a regeneration emitted a differently-shaped or empty file. A
generated data table with no consumer is exactly the kind of thing that rots
silently, so the shape it promises is pinned here.

Two kinds of assertion, deliberately separated:

- **Ground truth from outside the code.** A handful of unit_const -> file_name
  pairs read off the game's own unit/graphic tables and recognizable by name
  (a House is a house, a Villager is a villager). These catch a generator that
  reads the wrong .dat field or slips an index, which a self-consistency check
  cannot -- the exact failure mode that made a green suite miss the building
  footprint bug (see tests/test_unit_footprints.py).
- **Invariants swept over every entry.** file_name must be a bare on-disk stem
  because the renderer joins it to a graphics directory and appends ".sld"
  itself; a value carrying a path separator or an extension would silently
  resolve outside that directory or not at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GRAPHIC_MAP_PATH = Path(__file__).resolve().parent.parent / "descape" / "unit_graphic_map.json"

# unit_const -> (label, file_name), read from empires2_x2_p1.dat via
# tools/gen_unit_graphic_map.py and each cross-checked against the unit's own
# name in AoE2ScenarioParser's UnitInfo. Hardcoded on purpose: deriving these
# from the table would just restate the data under test.
KNOWN_GRAPHICS = {
    70: ("House", "b_dark_house_age1_x1"),
    109: ("Town Center", "b_dark_town_center_age1_back_x1"),
    83: ("Villager (male)", "u_vil_male_villager_idleA_x1"),
    93: ("Spearman", "u_inf_spearman_idleA_x1"),
    74: ("Militia", "u_inf_militia_idleA_x1"),
    40: ("Cataphract", "u_cav_cataphract_idleA_x1"),
    705: ("Cow (black and white)", "a_herd_cow_blackandwhite_idleA_x1"),
    # Legacy shell graphic (.dat file_name "None") resolved through its
    # delta chain rather than read directly -- see gen_unit_graphic_map.py's
    # "Legacy shell graphics" docstring section.
    45: ("Dock", "b_dark_dock_age1_x1"),
    # Modern-named DECORATION shells resolved through _BODY_GRAPHIC_OVERRIDES.
    # Their standing_graphic is b_dark_wall_palisade_flag_x1, an animated flag,
    # not the wall -- see gen_unit_graphic_map.py's override table. Named here
    # rather than only in the generator so a regeneration that dropped the
    # overrides is caught.
    72: ("Palisade Wall", "b_dark_wall_palisade_x1"),
    119: ("Fortified Palisade Wall", "b_scen_wall_palisade_fortified_x1"),
    788: ("Sea Wall", "b_scen_wall_sea_x1"),
}

# The shell those three used to resolve to. It is a real graphic and other
# consts may legitimately use it one day, but never these: it is the flag.
PALISADE_FLAG_SHELL = "b_dark_wall_palisade_flag_x1"
DECORATION_SHELL_CONSTS = (72, 119, 788)

# Farm and its family (RFARM, FARMDROP, PASTURE, ...) share one legacy
# graphic whose deltas terminate in further legacy names, and no graphic in
# the .dat has a farm/field/crop file_name at all -- a real farm's crop is
# terrain, not a unit sprite (see gen_unit_graphic_map.py). These must stay
# absent rather than carry a filename that will never exist on any install.
#
# 1889 (PASTURE_BUILD) is deliberately NOT in this tuple: it resolves to a
# real modern .sld and correctly HAS a graphic_map entry, unlike its sibling
# consts here. 1193/1194/1195 (FARMDROP/FARMSTACK/RFARMDROP) are invisible
# internal dropsite helpers with foundation_terrain_id == -1 -- absent for
# the same reason as the rest, but not renderable as terrain either (see
# render.py's FOUNDATION_TERRAIN / _terrain_overlay_for).
FARM_FAMILY_CONSTS = (
    50, 357, 1187, 1188, 1193, 1194, 1195, 1893, 1894, 1897, 1898,
)

FIELDS = {
    "graphic_id": int,
    "file_name": str,
    "angle_count": int,
    "mirroring_mode": int,
    "frame_count": int,
}

# Optional key, present only on a multi-graphic composite building (town
# centres, pastures) -- see gen_unit_graphic_map.py's "Composite buildings"
# section. Each piece carries its own file_name/angle_count/frame_count (a
# sibling graphic can have a different frame_count than its parent) plus a
# native-pixel (dx, dy) offset from the parent's own anchor.
PIECE_FIELDS = {
    "unit_id": int,
    "file_name": str,
    "angle_count": int,
    "frame_count": int,
    "dx": int,
    "dy": int,
}

# unit_const -> depth-sorted piece file_names, read off the real .dat
# (2026-08-29) via tools/gen_unit_graphic_map.py's annex walk. 109 is the
# reported bug (RTWC drew only its back quarter); 71 is the regression guard
# against reintroducing per-civ/age retargeting -- Incas' own back piece is
# Andean-style, but the game composites it with the SAME Dark Age
# main/center/front graphics as every other town centre const, not with
# Andean-suffixed siblings (see descape-annex-composite.md finding 6).
KNOWN_PIECES = {
    109: [
        "b_dark_town_center_age1_main_x1",
        "b_dark_town_center_age1_back_x1",
        "b_dark_town_center_age1_center_x1",
        "b_dark_town_center_age1_front_x1",
    ],
    71: [
        "b_dark_town_center_age1_main_x1",
        "b_west_town_center_age2_back_x1",
        "b_dark_town_center_age1_center_x1",
        "b_dark_town_center_age1_front_x1",
    ],
}


@pytest.fixture(scope="module")
def graphics() -> dict[str, dict]:
    return json.loads(GRAPHIC_MAP_PATH.read_text())["graphics"]


def test_known_units_resolve_to_their_own_named_graphic(graphics):
    for unit_const, (label, file_name) in KNOWN_GRAPHICS.items():
        entry = graphics.get(str(unit_const))
        assert entry is not None, f"{label} ({unit_const}) missing from the table"
        assert entry["file_name"] == file_name, f"{label} ({unit_const})"


def test_decoration_shells_resolve_to_the_body_not_the_overlay(graphics):
    """The palisade family's standing_graphic is an animated flag whose .sld
    carries art for only ONE of the five wall shapes -- the tower the flag sits
    on. Resolving to it drew a floating flag at one rotation and nothing at the
    other four, which is what a live window showed 2026-08-27.

    Stated as "not the shell" as well as "is the body" on purpose: the failure
    this guards is the generator's _looks_modern check accepting an overlay,
    and that check would reappear as exactly this file_name.
    """
    for unit_const in DECORATION_SHELL_CONSTS:
        entry = graphics[str(unit_const)]
        assert entry["file_name"] != PALISADE_FLAG_SHELL, (
            f"unit_const {unit_const} resolved to the flag overlay again -- "
            "_BODY_GRAPHIC_OVERRIDES was dropped or _looks_modern regressed"
        )
        # The body graphics are static one-frame shape sets; the flag is a
        # 90-frame loop. Pinning this catches a retarget to some OTHER
        # animated overlay, which the file_name check alone would let past.
        assert entry["frame_count"] == 1, (
            f"unit_const {unit_const} points at a {entry['frame_count']}-frame "
            "animation; a wall body is a single static frame per shape"
        )
        assert entry["angle_count"] == 5, unit_const


def test_farm_family_has_no_sld_entry(graphics):
    for unit_const in FARM_FAMILY_CONSTS:
        assert str(unit_const) not in graphics, (
            f"unit_const {unit_const} (Farm family) should be omitted, not "
            "carry a filename that never resolves on any install"
        )


def test_static_buildings_have_one_angle_and_mobile_units_have_many(graphics):
    """A building is drawn from a single stored angle; a unit is not.

    This is the cheapest available check that angle_count is the field the
    generator thinks it is -- a slipped index would not reproduce that split.
    """
    assert graphics["70"]["angle_count"] == 1  # House
    assert graphics["109"]["angle_count"] == 1  # Town Center
    assert graphics["83"]["angle_count"] > 1  # Villager
    assert graphics["93"]["angle_count"] > 1  # Spearman


def test_every_entry_has_the_promised_fields_and_types(graphics):
    assert graphics, "table is empty"
    for key, entry in graphics.items():
        assert key.isdigit(), f"key {key!r} is not a unit_const"
        fields = set(entry) - {"pieces"}
        assert fields == set(FIELDS), f"unit_const {key} has fields {sorted(entry)}"
        for field, kind in FIELDS.items():
            assert isinstance(entry[field], kind), f"unit_const {key}.{field}"
        if "pieces" in entry:
            assert isinstance(entry["pieces"], list) and len(entry["pieces"]) >= 2, (
                f"unit_const {key}: pieces must be the whole composite, parent included"
            )
            for i, piece in enumerate(entry["pieces"]):
                assert set(piece) == set(PIECE_FIELDS), f"unit_const {key} piece {i}: {sorted(piece)}"
                for field, kind in PIECE_FIELDS.items():
                    assert isinstance(piece[field], kind), f"unit_const {key} piece {i}.{field}"


def test_file_names_are_bare_stems(graphics):
    """The renderer builds `<graphics dir>/<file_name>.sld` itself."""
    for key, entry in graphics.items():
        names = [entry["file_name"], *(p["file_name"] for p in entry.get("pieces", []))]
        for name in names:
            assert name, f"unit_const {key} has an empty file_name"
            assert "/" not in name and "\\" not in name, f"unit_const {key}: {name!r}"
            assert "." not in name, f"unit_const {key}: {name!r}"


def test_composite_buildings_resolve_to_the_full_depth_sorted_piece_set(graphics):
    for unit_const, file_names in KNOWN_PIECES.items():
        entry = graphics[str(unit_const)]
        assert "pieces" in entry, f"unit_const {unit_const} should carry a composite pieces list"
        got = [p["file_name"] for p in entry["pieces"]]
        assert got == file_names, f"unit_const {unit_const}: order or piece set changed"


def test_a_composite_buildings_own_piece_is_at_zero_offset(graphics):
    """The parent's own art has zero misplacement by definition, so its own
    piece entry -- wherever it lands in depth order -- must be at (0, 0)."""
    for unit_const in KNOWN_PIECES:
        entry = graphics[str(unit_const)]
        own = next(p for p in entry["pieces"] if p["file_name"] == entry["file_name"])
        assert (own["dx"], own["dy"]) == (0, 0), unit_const


def test_town_centre_pieces_cancel_to_zero_but_pasture_pieces_do_not(graphics):
    """Finding 5's discriminator, checked by construction on the generator's
    own output rather than re-deriving it from the .dat: every TC-family
    piece slot has a null delta that exactly cancels its own misplacement, so
    every TC piece -- not just the parent -- lands at net (0, 0). Pastures
    have no such delta, so their corner posts must NOT collapse onto the
    building's own anchor -- reversing this check (asserting pastures cancel
    too) must fail, which is what makes it an assertion and not a tautology.
    """
    for unit_const in (109, 71, 141, 142, 2275, 2276, 2277):
        entry = graphics[str(unit_const)]
        for piece in entry["pieces"]:
            assert (piece["dx"], piece["dy"]) == (0, 0), (unit_const, piece["file_name"])

    for unit_const in (1889, 1890, 2079, 2080):
        entry = graphics[str(unit_const)]
        nonzero = [p for p in entry["pieces"] if (p["dx"], p["dy"]) != (0, 0)]
        assert len(nonzero) >= 2, f"pasture {unit_const} should have real corner-post offsets"


def test_counts_are_positive(graphics):
    for key, entry in graphics.items():
        assert entry["graphic_id"] >= 0, f"unit_const {key}"
        assert entry["angle_count"] >= 1, f"unit_const {key}"
        assert entry["frame_count"] >= 1, f"unit_const {key}"
