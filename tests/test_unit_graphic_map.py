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

import importlib
import json
from collections import Counter
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
# consts here. Nor are 1893/1897 (placed Pastures): their annex tree reaches
# modern art even though their own graphic doesn't (GH #66, ANNEX_TREE_CONSTS). 1193/1194/1195 (FARMDROP/FARMSTACK/RFARMDROP) are invisible
# internal dropsite helpers with foundation_terrain_id == -1 -- absent for
# the same reason as the rest, but not renderable as terrain either (see
# render.py's FOUNDATION_TERRAIN / _terrain_overlay_for).
FARM_FAMILY_CONSTS = (
    50, 357, 1187, 1188, 1193, 1194, 1195, 1894, 1898,
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
# native-pixel (dx, dy) offset from the parent's own anchor. Exactly one piece
# per entry additionally carries `"parent": true`, checked per entry, not here.
PIECE_FIELDS = {
    "unit_id": int,
    "file_name": str,
    "angle_count": int,
    "frame_count": int,
    "dx": int,
    "dy": int,
}
# A _COMPOSITE_SCOPE piece also carries its raw annex misplacement, the depth-slot
# input. Two exact shapes rather than a subset check, so a stray field still fails.
COMPOSITE_PIECE_FIELDS = {**PIECE_FIELDS, "mx": float, "my": float, "slot": list}

# Mirrors tools/gen_unit_graphic_map.py's _COMPOSITE_SCOPE (town centres, pastures).
# Mirrors tools/gen_unit_graphic_map.py's _ANNEX_TREE_SCOPE (placed Pastures).
ANNEX_TREE_CONSTS = frozenset({1893, 1897})
# Every const whose pieces carry mx/my/slot.
SLOTTED_CONSTS = frozenset(
    {71, 109, 141, 142, 2275, 2276, 2277, 1889, 1890, 2079, 2080}
) | ANNEX_TREE_CONSTS
COMPOSITE_SCOPE_CONSTS = SLOTTED_CONSTS - ANNEX_TREE_CONSTS

# unit_const -> depth-sorted piece file_names, read off the real .dat
# (2026-08-29) via tools/gen_unit_graphic_map.py's annex walk. 109 is the
# reported bug (RTWC drew only its back quarter). 71 is the Feudal Age town
# centre: the .dat lists the Dark Age annexes 618-620 on it, but the age techs
# upgrade those to 614-616 at the same age, so since GH #48 Slice 2 its pieces
# are one age (they used to pair a Feudal back with Dark Age main/center/front).
KNOWN_PIECES = {
    109: [
        "b_dark_town_center_age1_main_x1",
        "b_dark_town_center_age1_back_x1",
        "b_dark_town_center_age1_center_x1",
        "b_dark_town_center_age1_front_x1",
    ],
    71: [
        "b_west_town_center_age2_main_x1",
        "b_west_town_center_age2_back_x1",
        "b_west_town_center_age2_center_x1",
        "b_west_town_center_age2_front_x1",
    ],
}

# unit_const -> (file_name, dx, dy) depth-ordered piece list, read off the
# real .dat (2026-09-02) via tools/gen_unit_graphic_map.py's class-39
# "every modern delta" rule -- see that module's "Gates" docstring section.
# 64/78 are bare directional (NE) gates, whose 4 non-middle pieces come from
# their two corner-pillar (const 81) annexes; 487 is the X-state composite
# gate, which carries the same 5 pieces as direct deltas of its own legacy
# shell, no annexes involved -- cross-checked against 64/78 below since the
# plan's whole correction rests on these two independent paths agreeing.
KNOWN_GATE_PIECES = {
    64: [
        ("b_west_gate_stone_ne_closed_x1", 0, 0),
        ("b_west_gate_stone_corner_x1", -72, 36),
        ("b_west_gate_stone_flag_x1", -72, -84),
        ("b_west_gate_stone_corner_x1", 72, -36),
        ("b_west_gate_stone_flag_x1", 72, -156),
    ],
    78: [
        ("b_west_gate_stone_ne_open_x1", 0, 0),
        ("b_west_gate_stone_corner_x1", -72, 36),
        ("b_west_gate_stone_flag_x1", -72, -84),
        ("b_west_gate_stone_corner_x1", 72, -36),
        ("b_west_gate_stone_flag_x1", 72, -156),
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
        # The flag is not lost, it moved: a non-parent piece after the body.
        # Without this the test would stay green whether or not it ever came back.
        names = [piece["file_name"] for piece in entry["pieces"]]
        assert names[-1] == PALISADE_FLAG_SHELL and not entry["pieces"][-1].get("parent"), unit_const


# unit_const -> (file_name, dx, dy, is_parent) per piece, read off the real .dat
# 2026-09-02: deltas in .dat list order, the flag shell at its -1 delta's slot.
# Its non-zero dy lifts the pole onto the tower top; at (0, 0) it hung mid-wall (GH #51).
KNOWN_WALL_PIECES = {
    72: [("b_dark_wall_palisade_x1", 0, 0, True), (PALISADE_FLAG_SHELL, 0, -40, False)],
    119: [("b_scen_wall_palisade_fortified_x1", 0, 0, True), (PALISADE_FLAG_SHELL, 0, -60, False)],
    788: [
        ("b_scen_wall_sea_underwater_x1", 0, 0, False),
        ("b_scen_wall_sea_x1", 0, 0, True),
        (PALISADE_FLAG_SHELL, 0, -60, False),
    ],
}


def test_decoration_shell_walls_carry_their_flag_and_base_as_pieces(graphics):
    for unit_const, expected in KNOWN_WALL_PIECES.items():
        pieces = graphics[str(unit_const)]["pieces"]
        got = [(p["file_name"], p["dx"], p["dy"], bool(p.get("parent"))) for p in pieces]
        assert got == expected, f"unit_const {unit_const}: {got}"
        for piece in pieces:
            # Mirrors the generator's angle_count == 5 guard, without a real .dat.
            assert piece["angle_count"] == 5, (unit_const, piece["file_name"])
            assert piece["unit_id"] == unit_const, (unit_const, piece["file_name"])
            # 90-frame flag vs 1-frame bodies: stops a retarget onto some other overlay passing.
            want_frames = 90 if piece["file_name"] == PALISADE_FLAG_SHELL else 1
            assert piece["frame_count"] == want_frames, (unit_const, piece["file_name"])


# unit_const -> (graphic_id, _underwater body), read off the real .dat 2026-10-09.
# The standing graphic is the surface splash, empty or a sliver at frame 0 (GH #178).
FISH_BODIES = {
    53: (1713, "a_fish_perch_underwater_x1"),
    61: (2184, "a_fish_dolphin_underwater_x1"),
    69: (3187, "a_fish_shore_underwater_x1"),
    450: (2183, "a_fish_marlin1_underwater_x1"),
    451: (2183, "a_fish_marlin1_underwater_x1"),
    455: (2185, "a_fish_dorado_underwater_x1"),
    456: (2190, "a_fish_salmon_underwater_x1"),
    457: (2191, "a_fish_tuna_underwater_x1"),
    458: (2192, "a_fish_snapper_underwater_x1"),
    1141: (3192, "a_fish_boxturtles_underwater_x1"),
    2625: (11398, "a_fish_whale_underwater_x1"),
}


def test_fish_resolve_to_their_underwater_body_not_the_splash(graphics):
    for unit_const, (graphic_id, file_name) in FISH_BODIES.items():
        entry = graphics[str(unit_const)]
        assert (entry["graphic_id"], entry["file_name"]) == (graphic_id, file_name), unit_const
        assert "pieces" not in entry, unit_const
    # Dorado keeps its splash's two variants, so Cycle Variant's scope is unchanged.
    dorado = graphics["455"]
    assert (dorado["angle_count"], dorado.get("rotation_is_variant"), dorado.get("variant_count")) == (2, True, 2)


def test_only_the_marlin_body_carries_a_frame_offset(graphics):
    """Its frames 0-4 hold no ink and 5-6 a few px; every other body has a fish at frame 0."""
    offsets = {int(k): e["frame_offset"] for k, e in graphics.items() if "frame_offset" in e}
    assert offsets == {450: 7, 451: 7}


def test_only_the_perch_body_carries_a_top_level_offset(graphics):
    """Perch's `_underwater` delta sits at .dat offset (0, -30); every other
    fish body's is (0, 0), so only 53 carries dx/dy (amendment 2026-10-09)."""
    offsets = {int(k): (e["dx"], e["dy"]) for k, e in graphics.items() if "dx" in e or "dy" in e}
    assert offsets == {53: (0, -30)}


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
        fields = set(entry) - {
            "pieces", "rotation_is_variant", "variant_count", "frame_offset", "dx", "dy",
            "particle", "art_scale", "garrison_flags",
        }
        assert fields == set(FIELDS), f"unit_const {key} has fields {sorted(entry)}"
        for field, kind in FIELDS.items():
            assert isinstance(entry[field], kind), f"unit_const {key}.{field}"
        if "particle" in entry or "art_scale" in entry:
            # Both or neither, on a single graphic drawn from frame 0 only.
            assert entry.get("particle") is True, key
            assert isinstance(entry.get("art_scale"), float) and entry["art_scale"] > 0, key
            assert (entry["angle_count"], entry["frame_count"]) == (1, 1), key
            assert not {"pieces", "dx", "rotation_is_variant", "frame_offset"} & set(entry), key
        if "frame_offset" in entry:
            # Added to a dispatched frame run, so it must stay inside one run.
            assert isinstance(entry["frame_offset"], int), key
            assert 0 < entry["frame_offset"] < entry["frame_count"], key
        if "dx" in entry or "dy" in entry:
            # Both or neither, ints, never (0, 0), and never beside pieces, whose
            # draw path would drop a top-level offset.
            assert isinstance(entry.get("dx"), int) and isinstance(entry.get("dy"), int), key
            assert (entry["dx"], entry["dy"]) != (0, 0), key
            assert "pieces" not in entry, key
        if "rotation_is_variant" in entry:
            assert entry["rotation_is_variant"] is True, (
                f"unit_const {key}: rotation_is_variant is omitted when false, "
                f"never written as false"
            )
        if "variant_count" in entry:
            # Only beside rotation_is_variant, and only when the .sld was
            # readable at generation time -- the edit path's cycle modulus.
            assert entry.get("rotation_is_variant") is True, (
                f"unit_const {key}: variant_count belongs only on a variant-indexed graphic"
            )
            assert isinstance(entry["variant_count"], int) and entry["variant_count"] >= 1, (
                f"unit_const {key}.variant_count"
            )
        if "pieces" in entry:
            assert isinstance(entry["pieces"], list) and len(entry["pieces"]) >= 2, (
                f"unit_const {key}: pieces must be the whole composite, parent included"
            )
            parents = [p for p in entry["pieces"] if "parent" in p]
            assert len(parents) == 1 and parents[0]["parent"] is True, (
                f"unit_const {key}: exactly one piece must be marked parent, got {len(parents)}"
            )
            shape = COMPOSITE_PIECE_FIELDS if int(key) in SLOTTED_CONSTS else PIECE_FIELDS
            for i, piece in enumerate(entry["pieces"]):
                assert set(piece) - {"parent", "seeded"} == set(shape), f"unit_const {key} piece {i}: {sorted(piece)}"
                if "seeded" in piece:
                    assert piece["seeded"] is True and int(key) in ANNEX_TREE_CONSTS, (key, i)
                    assert "parent" not in piece, (key, i)
                for field, kind in shape.items():
                    assert isinstance(piece[field], kind), f"unit_const {key} piece {i}.{field}"
        if "garrison_flags" in entry:
            _assert_garrison_flags(key, entry["garrison_flags"])


def _assert_garrison_flags(key: str, flags) -> None:
    """GH #203: omitted when empty, one or two pieces, each a static one-angle
    graphic dispatched on the host's own const."""
    assert isinstance(flags, list) and 1 <= len(flags) <= 2, f"unit_const {key}: {flags!r}"
    for i, flag in enumerate(flags):
        assert set(flag) == set(PIECE_FIELDS), f"unit_const {key} flag {i}: {sorted(flag)}"
        for field, kind in PIECE_FIELDS.items():
            assert isinstance(flag[field], kind), f"unit_const {key} flag {i}.{field}"
        assert flag["unit_id"] == int(key) and flag["angle_count"] == 1, (key, flag)
        assert flag["file_name"].endswith("_garrison_flag_x1"), (key, flag)


def test_file_names_are_bare_stems(graphics):
    """The renderer builds `<graphics dir>/<file_name>.sld` itself."""
    for key, entry in graphics.items():
        names = [
            entry["file_name"],
            *(p["file_name"] for p in (*entry.get("pieces", ()), *entry.get("garrison_flags", ()))),
        ]
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


def test_the_parent_piece_is_the_entrys_own_graphic_wherever_it_sorts(graphics):
    """sprite_pieces_for() drops the whole composite when the marked parent
    fails, so the marker must sit on the piece a non-composite consumer draws:
    the entry's own top-level graphic. Pinned positions: a town centre's own
    back piece sorts to index 1, and an X-state gate's parent is its corner
    pillar at index 0, not the middle span (see the generator's gate branch)."""
    for key, entry in graphics.items():
        for piece in entry.get("pieces", ()):
            if piece.get("parent"):
                assert piece["file_name"] == entry["file_name"], key
    assert [i for i, p in enumerate(graphics["109"]["pieces"]) if p.get("parent")] == [1]
    assert graphics["487"]["pieces"][0].get("parent") is True
    assert graphics["487"]["pieces"][0]["file_name"] == "b_west_gate_stone_corner_x1"


def test_composite_pieces_carry_their_raw_misplacement(graphics):
    """Read off the real .dat 2026-09-02 (depth-slots plan, finding 1). A town
    centre's dx/dy all cancel to (0, 0), so this is the only record of where
    each piece really sits."""
    tc = [(p["file_name"], p["mx"], p["my"]) for p in graphics["109"]["pieces"]]
    assert tc == [
        ("b_dark_town_center_age1_main_x1", 1.0, -1.0),
        ("b_dark_town_center_age1_back_x1", 0.0, 0.0),
        ("b_dark_town_center_age1_center_x1", -0.5, 0.5),
        ("b_dark_town_center_age1_front_x1", -1.0, 1.0),
    ]
    pasture = [(p["mx"], p["my"]) for p in graphics["1889"]["pieces"]]
    assert pasture == [(2.0, -2.0), (0.0, 0.0), (2.0, 2.0), (-2.0, -2.0), (-2.0, 2.0)]


# unit_const -> each piece's [sx, sy] depth slot, in piece order, measured by
# tools/measure_piece_slots.py against the real art (see the corpus test below,
# which re-runs it). A town centre's `main` slots three tiles earlier than the
# rest; a pasture's four corner posts each land on their own tile.
KNOWN_SLOTS = {
    109: [[1, 2], [0, 3], [0, 3], [0, 3]],
    1889: [[3, 0], [0, 3], [0, 2], [0, 1], [0, 3]],
}


def test_composite_pieces_carry_their_measured_depth_slot(graphics):
    for unit_const, slots in KNOWN_SLOTS.items():
        assert [p["slot"] for p in graphics[str(unit_const)]["pieces"]] == slots, unit_const


def test_every_slot_lies_inside_its_parents_own_footprint(graphics):
    """Invariant 1: a piece anchored outside the building would let terrain
    beyond it paint over the piece's feet."""
    from descape.terrain_palette import tile_span

    for unit_const in SLOTTED_CONSTS:
        entry = graphics.get(str(unit_const))
        if entry is None:
            continue
        span_x, span_y = tile_span(unit_const, (1, 1))
        for piece in entry["pieces"]:
            sx, sy = piece["slot"]
            assert 0 <= sx < span_x and 0 <= sy < span_y, (unit_const, piece["file_name"])


def test_only_composite_scope_entries_carry_a_slot(graphics):
    """Gates and decoration-shell walls deliberately have none: their pieces
    paint at the unit's single anchor tile, which render.py must keep
    supporting (see tests/test_sprite_chunks.py's no-slot fallback pin)."""
    slotted = {int(k) for k, e in graphics.items()
               if any("slot" in p for p in e.get("pieces", ()))}
    assert slotted == {c for c in SLOTTED_CONSTS if str(c) in graphics}


@pytest.mark.corpus
def test_the_committed_slots_still_match_the_real_art(request):
    """`_PIECE_SLOTS` is measured, not derived: the generator cannot compute it
    without decoding .sld art. This is the only check that it still describes
    the installed game, modelled on
    tests/test_sprite_edit_bbox.py's reach test -- same env-var-only install
    route, same assert-it-read-something-first non-vacuity guard."""
    from descape import asset_source

    import conftest

    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The "
            "install configured in config.yaml does NOT count here: "
            "conftest._isolated_settings redirects CONFIG_PATH for every test"
        )
    measure = conftest.load_verify_module("measure_piece_slots")
    generator = measure._generator_module()

    measured = {}
    for unit_const in sorted(
        (generator._COMPOSITE_SCOPE | generator._ANNEX_TREE_SCOPE) - generator._TOWN_CENTRE_SCOPE
    ):
        result = measure.measure(unit_const)
        if result is not None:
            measured[unit_const] = [list(row["slot"]) for row in result["rows"]]
    assert measured, "no composite piece art was readable, so this proves nothing"
    for unit_const, slots in measured.items():
        assert generator._PIECE_SLOTS[unit_const] == slots, (
            f"unit_const {unit_const}: the real art now wants {slots}. Re-run "
            f"tools/measure_piece_slots.py and commit its table"
        )
    # GH #48: town centres per art set, over both committed tables.
    art_sets = measure.town_centre_art_sets(generator)
    assert len(art_sets) > 40, "the per-civ town-centre sets are missing"
    for key, (unit_const, entry) in art_sets.items():
        result = measure.measure(unit_const, entry)
        assert result is not None, f"{key}: art unreadable on this install"
        slots = [list(row["slot"]) for row in result["rows"]]
        assert generator._PIECE_SLOTS_BY_ART.get(key) == slots, (
            f"{key}: the real art now wants {slots}. Re-run tools/measure_piece_slots.py "
            f"and commit its _PIECE_SLOTS_BY_ART"
        )
        assert [p["slot"] for p in entry["pieces"]] == slots, f"{key}: the committed entry carries stale slots"


@pytest.mark.corpus
@pytest.mark.slow
def test_the_generator_reproduces_both_committed_tables(tmp_path):
    """GH #48: the per-civ refactor, its invariant asserts and the
    relationship-derived wall bodies must hold against the installed .dat,
    and must leave unit_graphic_map.json byte-identical. Same env-var-only
    install route as the slot test above."""
    import subprocess
    import sys

    from descape import asset_source

    install = asset_source.get_install_path()
    if install is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")
    pytest.importorskip("genieutils")
    tool = Path(__file__).resolve().parent.parent / "tools" / "gen_unit_graphic_map.py"
    run = subprocess.run(
        [sys.executable, str(tool), str(install), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    descape_dir = GRAPHIC_MAP_PATH.parent
    for name in ("unit_graphic_map.json", "building_art_map.json"):
        assert (tmp_path / name).read_bytes() == (descape_dir / name).read_bytes(), (
            f"{name} no longer regenerates byte-identical -- re-run tools/gen_unit_graphic_map.py"
        )


# GH #66: 1897's annex tree, (unit_id, mx, my) in depth order, read off the
# real .dat 2026-09-21. Hut 1890 + 4 posts 1888 + 5 fences per edge (2079/1885
# along y = +-2, 2080/1886 along x = +-2), misplacement summed down the tree.
KNOWN_PASTURE_TREE = [
    (1888, 2.0, -2.0), (1885, 1.3, -2.0), (1886, 2.0, -1.3), (1885, 0.65, -2.0),
    (1886, 2.0, -0.65), (2079, 0.0, -2.0), (2080, 2.0, 0.0), (1885, -0.65, -2.0),
    (1886, 2.0, 0.65), (1885, -1.3, -2.0), (1886, 2.0, 1.3), (1890, 0.0, 0.0),
    (1888, 2.0, 2.0), (1888, -2.0, -2.0), (1885, 1.3, 2.0), (1886, -2.0, -1.3),
    (1885, 0.65, 2.0), (1886, -2.0, -0.65), (2079, 0.0, 2.0), (2080, -2.0, 0.0),
    (1885, -0.65, 2.0), (1886, -2.0, 0.65), (1885, -1.3, 2.0), (1886, -2.0, 1.3),
    (1888, -2.0, 2.0),
]


def test_placed_pasture_draws_its_whole_annex_tree(graphics):
    for unit_const in ANNEX_TREE_CONSTS:
        entry = graphics[str(unit_const)]
        pieces = entry["pieces"]
        assert [(p["unit_id"], p["mx"], p["my"]) for p in pieces] == KNOWN_PASTURE_TREE, unit_const
        # The hut is the parent and supplies the top-level fields, unlike a
        # town centre whose parent is the const's own art.
        assert entry["file_name"] == "b_dark_pasture_x1"
        assert [p["unit_id"] for p in pieces if p.get("parent")] == [1890]
        assert sum(1 for p in pieces if p.get("seeded")) == 24
        names = Counter(p["file_name"] for p in pieces)
        assert names == {
            "b_dark_pasture_x1": 1,
            "b_dark_pasture_corner_posts_x1": 4,
            "b_dark_pasture_broken_perimeter_fencesA_x1": 10,
            "b_dark_pasture_broken_perimeter_fencesB_x1": 10,
        }
        # FencesA sit on the y = +-2 edges, fencesB on x = +-2.
        for p in pieces:
            if p["file_name"].endswith("fencesA_x1"):
                assert abs(p["my"]) == 2.0, p
            if p["file_name"].endswith("fencesB_x1"):
                assert abs(p["mx"]) == 2.0, p


def test_counts_are_positive(graphics):
    for key, entry in graphics.items():
        assert entry["graphic_id"] >= 0, f"unit_const {key}"
        assert entry["angle_count"] >= 1, f"unit_const {key}"
        assert entry["frame_count"] >= 1, f"unit_const {key}"


def test_bare_directional_gates_agree_with_the_x_state_composites_own_deltas(graphics):
    """64/78's annex-derived pieces (two corner-pillar annexes, each resolved
    through its own multi-delta walk) reproduce const 487's direct-delta
    piece list exactly -- the plan's central correction, that a bare gate's
    annexes and an X-state gate's direct deltas are two paths to the SAME
    real pieces, not two different mechanisms. See
    gen_unit_graphic_map.py's "Gates" docstring section."""
    x_state_non_middle = {
        (p["file_name"], p["dx"], p["dy"])
        for p in graphics["487"]["pieces"]
        if p["file_name"] != "b_west_gate_stone_ne_closed_x1"
    }
    for unit_const, expected in KNOWN_GATE_PIECES.items():
        entry = graphics[str(unit_const)]
        got = [(p["file_name"], p["dx"], p["dy"]) for p in entry["pieces"]]
        assert got == expected, f"unit_const {unit_const}: {got}"
        non_middle = {t for t in got if t[0] != entry["file_name"]}
        assert non_middle == x_state_non_middle, (
            f"unit_const {unit_const} disagrees with const 487's own direct deltas"
        )


def test_bare_corner_pillar_carries_its_own_flag(graphics):
    """A standalone corner-pillar const (95, 81, ...) is not itself a gate
    but is class 39, and its own legacy shell has the same first-match-drops-
    the-flag problem a bare gate's annex does -- see the "Gates" docstring
    section's `_resolve_all_modern_deltas` rationale."""
    entry = graphics["95"]
    got = [(p["file_name"], p["dx"], p["dy"]) for p in entry["pieces"]]
    assert got == [
        ("b_west_gate_stone_corner_x1", 0, 0),
        ("b_west_gate_stone_flag_x1", 0, -120),
    ]


def test_sea_gate_drops_the_underwater_duplicate_offset(graphics):
    """1381/1385/1389/1393's root deltas to "sides" and "...underwater" share
    one offset; the dedupe rule keeps "sides" (first in the delta list, and
    the one with a real PLAYERCOLOR layer) and drops "underwater"."""
    for unit_const in (1381, 1385, 1389, 1393):
        entry = graphics[str(unit_const)]
        got = [(p["file_name"], p["dx"], p["dy"]) for p in entry["pieces"]]
        assert got == [
            ("b_scen_gate_sea_sides_x1", 0, 0),
            ("b_scen_gate_sea_flag_x1", -3, -85),
        ], unit_const


def test_gate_pieces_are_all_non_rotating(graphics):
    """Mirrors the generator's own SystemExit guard (every emitted class-39
    piece must have angle_count == 1) so a regression is caught here too,
    without needing a real .dat to re-run the generator."""
    for unit_const in (64, 78, 487, 95, 81, 1381, 1385, 1389, 1393):
        for piece in graphics[str(unit_const)]["pieces"]:
            assert piece["angle_count"] == 1, (unit_const, piece["file_name"])


def test_non_gate_composite_scope_is_unaffected_by_the_gate_rule(graphics):
    """Regression guard on the Dock shell path (and every other ordinary
    legacy-shell resolution): the class-39 "every modern delta" rule must
    stay scoped to class 39 and never leak a "pieces" key onto an unrelated
    single-graphic entry. `pieces` outside class 39 is exactly the town
    centre/pasture scope plus the three decoration-shell walls, and since
    GH #20 the 20 sailing ships (SHIP_SAIL_CONSTS)."""
    assert "pieces" not in graphics["45"]  # Dock, resolved via its own shell delta
    non_gate = {
        int(k) for k, e in graphics.items() if "pieces" in e and "_gate_" not in e["file_name"]
    }
    assert non_gate == SLOTTED_CONSTS | set(KNOWN_WALL_PIECES) | SHIP_SAIL_CONSTS


# GH #20: every const whose standing graphic is a legacy shell with two or more
# direct modern deltas, read off the real .dat 2026-10-12. All are ships.
SHIP_SAIL_CONSTS = frozenset({
    17, 21, 114, 420, 442, 527, 528, 529, 532, 535, 536, 539, 545, 691, 1103, 1104,
    2626, 2627, 2628, 2633,
})

# unit_const -> (file_name, dx, dy) per piece in .dat delta order, hull first.
# Galleon's five sails at (0, -25) are why this rule has no offset dedupe.
KNOWN_SHIP_PIECES = {
    442: [
        ("u_shp_galleon_x1", 0, 0),
        ("u_shp_sail_west_ship_2_x1", 0, -40),
        ("u_shp_sail_west_ship_3_x1", 0, -25),
        ("u_shp_sail_west_ship_4_x1", 0, -25),
        ("u_shp_sail_west_ship_5_x1", 0, -25),
        ("u_shp_sail_west_ship_6_x1", 0, -25),
        ("u_shp_sail_west_ship_7_x1", 0, -25),
        ("u_shp_sail_west_ship_8_x1", 0, -40),
    ],
    539: [("u_shp_galley_x1", 0, 0), ("u_shp_sail_west_boat_4_x1", 0, -13)],
}

# Single-graphic ships: Fishing Ship, Longship, Turtle Ship, Dragon Ship, Dromon, Lou Chuan.
PLAIN_SHIP_CONSTS = (13, 250, 831, 938, 1795, 1948)


def test_exactly_the_sailing_ships_carry_shell_delta_pieces(graphics):
    """The rule is "legacy shell with several direct modern deltas", not a const
    list, so the set it hits is pinned here: a .dat change that widens it to a
    non-ship would draw that const from pieces nobody looked at."""
    found = {int(k) for k, e in graphics.items() if e["file_name"].startswith("u_shp_") and "pieces" in e}
    assert found == SHIP_SAIL_CONSTS
    for unit_const in PLAIN_SHIP_CONSTS:
        assert "pieces" not in graphics[str(unit_const)], unit_const


def test_ship_pieces_are_the_hull_then_every_sail_in_dat_order(graphics):
    for unit_const, expected in KNOWN_SHIP_PIECES.items():
        got = [(p["file_name"], p["dx"], p["dy"]) for p in graphics[str(unit_const)]["pieces"]]
        assert got == expected, f"unit_const {unit_const}: {got}"
    for unit_const in SHIP_SAIL_CONSTS:
        entry = graphics[str(unit_const)]
        hull, *sails = entry["pieces"]
        # The hull stays the entry's own graphic and the piece whose failure drops the ship.
        assert hull.get("parent") is True and hull["file_name"] == entry["file_name"], unit_const
        assert (hull["dx"], hull["dy"], hull["frame_count"]) == (0, 0, 1), unit_const
        assert "rotation_is_variant" not in entry, unit_const
        for piece in entry["pieces"]:
            # The ship's own id on every piece: one facing dispatch for hull and sails.
            assert piece["unit_id"] == unit_const, (unit_const, piece["file_name"])
            assert piece["angle_count"] == 16, (unit_const, piece["file_name"])
        for sail in sails:
            assert "_sail_" in sail["file_name"] and sail["frame_count"] == 30, (unit_const, sail)
            assert sail["dx"] == 0 and sail["dy"] < 0, (unit_const, sail)


# GH #20: unit_const -> (graphic_id, particle effect stem, the effect's Scale), read
# off the real .dat 2026-10-12 (Seagulls 2026-10-17, GH #67). Their standing graphic
# has file_name "None".
PARTICLE_ENTRIES = {
    303: (4411, "dock_seagulls", 0.525),
    1608: (12300, "butterflies_1_low", 0.5),
    1609: (12301, "butterflies_2_low", 0.5),
    1610: (12302, "butterflies_3_low", 0.5),
}


BUILDING_ART_PATH = GRAPHIC_MAP_PATH.parent / "building_art_map.json"

# GH #203: unit_const -> its Gaia garrison flags as (file_name, dx, dy), read off
# the real .dat 2026-10-12 (each const's creatable.garrison_graphic shell).
KNOWN_GARRISON_FLAGS = {
    109: [("b_misc_garrison_flag_x1", -31, -158)],  # Town Center
    79: [("b_west_garrison_flag_x1", -1, -179)],  # Watch Tower
    12: [("b_misc_garrison_flag_x1", 84, -59), ("b_misc_garrison_flag_x1", -7, -101)],  # Barracks
    70: [("b_seas_garrison_flag_x1", -17, -61)],  # House: a flag with 0 places
    545: [("b_west_garrison_flag_x1", 2, -24)],  # Transport Ship, a non-building host
}
# Mosque 655 holds units in game but has no garrison graphic, so it draws no flag.
NO_FLAG_HOSTS = (655,)
JAPANESE = 5
# Japanese Castle: its own asia art and two asia flags, not Gaia's single west one.
JAPANESE_CASTLE_FLAGS = [("b_asia_garrison_flag_x1", -54, -276), ("b_asia_garrison_flag_x1", 23, -314)]


def _flag_tuples(entry) -> list[tuple[str, int, int]]:
    return [(f["file_name"], f["dx"], f["dy"]) for f in entry.get("garrison_flags", ())]


def test_garrisoned_hosts_carry_their_measured_flags(graphics):
    for unit_const, expected in KNOWN_GARRISON_FLAGS.items():
        assert _flag_tuples(graphics[str(unit_const)]) == expected, unit_const
    for unit_const in NO_FLAG_HOSTS:
        assert "garrison_flags" not in graphics[str(unit_const)], unit_const
    flagged = {k for k, e in graphics.items() if "garrison_flags" in e}
    # The 99 consts with a garrison graphic, measured over every civ table.
    assert len(flagged) == 99, len(flagged)


def test_a_civs_building_art_carries_its_own_flags():
    """The flag is attached where per-civ art is built, so a civ's castle
    draws its own set's flags at its own offsets."""
    art = json.loads(BUILDING_ART_PATH.read_text())
    entry = art["entries"][art["civ_art"][str(JAPANESE)]["82"]]
    assert entry["file_name"] == "b_asia_castle_age3_x1"
    assert _flag_tuples(entry) == JAPANESE_CASTLE_FLAGS
    flagged = [item["garrison_flags"] for item in art["entries"] if "garrison_flags" in item]
    assert flagged
    for flags in flagged:
        _assert_garrison_flags(str(flags[0]["unit_id"]), flags)


def _shell_case(copies: int = 1, **change):
    """(unit, graphics) for one garrison shell in the measured shape, Town
    Center 109's, with `copies` deltas and `change` applied to the shell, its
    delta or its flag."""
    from types import SimpleNamespace as NS

    flag = NS(file_name="b_misc_garrison_flag_x1", deltas=[], angle_count=1, frame_count=90)
    delta = NS(graphic_id=1, display_angle=-1, offset_x=-31, offset_y=-158)
    shell = NS(file_name="None", deltas=[delta] * copies, angle_count=1)
    parts = {"shell": shell, "delta": delta, "flag": flag}
    for name, value in change.items():
        part, field = name.split("__")
        setattr(parts[part], field, value)
    return NS(creatable=NS(garrison_graphic=0)), [shell, flag]


@pytest.mark.parametrize(
    "change",
    [
        {"shell__file_name": "b_misc_garrison_shell_x1"},
        {"copies": 0},
        {"copies": 3},
        {"delta__display_angle": 2},
        {"delta__graphic_id": 7},
        {"flag__file_name": "GARRFLAG"},
        {"flag__angle_count": 8},
        {"flag__deltas": ["a sub-delta"]},
    ],
    ids=[
        "modern_shell", "no_delta", "three_deltas", "one_angle_only", "dangling", "legacy_flag", "many_angles",
        "flag_deltas",
    ],
)
def test_the_generator_refuses_a_garrison_shell_outside_the_measured_shape(monkeypatch, change):
    """The renderer draws each flag as one static frame-0 piece at its delta's
    offset; a shell that stops matching the 2026-10-12 measurement must stop
    the generator rather than emit something drawn wrong. Install-free: the
    check runs on stand-ins shaped like genieutils' objects."""
    monkeypatch.syspath_prepend(str(GRAPHIC_MAP_PATH.parent.parent / "tools"))
    gen = importlib.import_module("gen_unit_graphic_map")
    flag = {"unit_id": 109, "file_name": "b_misc_garrison_flag_x1", "angle_count": 1, "frame_count": 90,
            "dx": -31, "dy": -158}
    assert gen._garrison_flags(*_shell_case(), 109) == [flag]
    assert gen._garrison_flags(*_shell_case(copies=2), 109) == [flag, flag]
    unit, graphics = _shell_case(**change)
    with pytest.raises(SystemExit, match="re-verify the garrison flag rule"):
        gen._garrison_flags(unit, graphics, 109)


def test_a_data_mod_dat_narrows_the_hand_measured_lists_to_the_consts_it_has(monkeypatch):
    """GH #58: a VER 7.7 mod's .dat ends at 1902 units and holds nameless
    placeholders at 1893/1897, so the Pasture, Whale and later town-centre
    entries must drop out of scope rather than fail their guards."""
    import importlib.util
    from types import SimpleNamespace as NS

    monkeypatch.syspath_prepend(str(GRAPHIC_MAP_PATH.parent.parent / "tools"))
    path = GRAPHIC_MAP_PATH.parent.parent / "tools" / "gen_unit_graphic_map.py"
    spec =importlib.util.spec_from_file_location("gen_unit_graphic_map_narrow", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)  # a private copy: the narrowing rewrites module globals
    units = [NS(name="unit", standing_graphic=(0, -1)) for _ in range(1902)]
    units[1893] = units[1897] = NS(name="", standing_graphic=(-1, -1))
    units[61] = None
    gen._narrow_to_dat(units, [NS(file_name="b_dark_house_age1_x1"), None])
    assert gen._DATA_MOD_DAT
    assert not gen._ANNEX_TREE_SCOPE
    assert not {1893, 1897} & set(gen._PIECE_SLOTS)
    assert sorted(gen._TOWN_CENTRE_SCOPE) == [71, 109, 141, 142]
    assert 2625 not in gen._UNDERWATER_BODY_CONSTS and 61 not in gen._UNDERWATER_BODY_CONSTS
    assert 53 in gen._UNDERWATER_BODY_CONSTS
    assert not gen._BODY_FRAME_OFFSETS
    assert sorted(gen._ART_INVARIANT_HOLDOUTS) == [155, 446]


# GH #262: a data mod's .dat breaks invariants measured on the stock one.
# Lenient (dat_derive's run), each guard degrades only what it guards and
# records a warning; strict (regen_game_data.py) it still fails the run.


@pytest.fixture()
def gen(monkeypatch):
    monkeypatch.syspath_prepend(str(GRAPHIC_MAP_PATH.parent.parent / "tools"))
    return importlib.import_module("gen_unit_graphic_map")


@pytest.fixture()
def lenient(gen):
    """gen's _game_root lenient for the test, its WARNINGS returned; strict and empty again after."""
    root = gen._game_root
    root.WARNINGS.clear()
    root.set_lenient(True)
    yield root.WARNINGS
    root.set_lenient(False)
    root.WARNINGS.clear()


def _graphic(gid: int, file_name: str, deltas=(), angle_count: int = 1, layer: int = 20):
    from types import SimpleNamespace as NS

    return NS(id=gid, file_name=file_name, deltas=list(deltas), angle_count=angle_count,
              frame_count=10, layer=layer, mirroring_mode=0)


def _delta(graphic_id: int, offset_y: int = 0, display_angle: int = -1):
    from types import SimpleNamespace as NS

    return NS(graphic_id=graphic_id, display_angle=display_angle, offset_x=0, offset_y=offset_y)


def _units(unit_const: int, *, type_: int = 80, class_: int = 3, building=None) -> list:
    """A unit table holding one unit at unit_const, standing graphic 0, no garrison graphic."""
    from types import SimpleNamespace as NS

    units: list = [None] * (unit_const + 1)
    units[unit_const] = NS(standing_graphic=(0, -1), type=type_, class_=class_, building=building, creatable=None)
    return units


def _dock_shell() -> list:
    """GH #262's shape: a legacy shell with a modern hull then a sail, on a building."""
    return [
        _graphic(0, "-C1", deltas=[_delta(1), _delta(2, offset_y=-25)]),
        _graphic(1, "BSHIPY1_S-C1_x1"),
        _graphic(2, "BSHIPY1_S-C1_sail_x1"),
    ]


def test_a_buildings_multi_delta_shell_is_refused_when_strict(gen):
    graphics = _dock_shell()
    with pytest.raises(SystemExit, match=r"unit_const 45 \(type 80\) has a multi-delta legacy shell '-C1' but is not type 70"):
        gen._resolve_shell_delta_pieces(graphics, 45, 80, graphics[0], graphics[1])


def test_a_buildings_multi_delta_shell_is_drawn_as_its_hull_when_lenient(gen, lenient):
    graphics = _dock_shell()
    assert gen._resolve_shell_delta_pieces(graphics, 45, 80, graphics[0], graphics[1]) is None
    entry, kind = gen._entry_for(_units(45), graphics, Path("no-graphics"), 45, {})
    assert (entry, kind) == ({
        "graphic_id": 1, "file_name": "BSHIPY1_S-C1_x1", "angle_count": 1, "mirroring_mode": 0, "frame_count": 10,
    }, "plain")
    assert lenient == [
        (
            "unit_const 45 (type 80) has a multi-delta legacy shell '-C1' but is not type 70, so its pieces would "
            "not dispatch on a facing. Re-verify the ship-sail rule against the .dat (drawn as 'BSHIPY1_S-C1_x1' alone)"
        )
    ]
    ship, ship_kind = gen._entry_for(_units(45, type_=70), graphics, Path("no-graphics"), 45, {})
    assert ship_kind == "shell_deltas" and len(ship["pieces"]) == 2, "a creatable keeps its sails"


@pytest.mark.parametrize("change", [{"copies": 3}, {"delta__display_angle": 2}], ids=["three_deltas", "one_angle_only"])
def test_a_garrison_shell_outside_the_measured_shape_gives_no_flags_when_lenient(gen, lenient, change):
    unit, graphics = _shell_case(**change)
    assert gen._garrison_flags(unit, graphics, 109) == []
    assert len(lenient) == 1 and lenient[0].endswith("(drawn without garrison flags)")


@pytest.mark.parametrize(
    "shell_angles, deltas, match",
    [(5, [-1, -1, 1], "graphic_id -1 deltas"), (1, [1, -1], "not 5")],
    ids=["two_flag_slots", "flag_not_five_shapes"],
)
def test_a_wall_shell_outside_the_measured_shape_draws_its_body_alone(gen, lenient, shell_angles, deltas, match):
    shell = _graphic(0, "b_wall_flag_x1", deltas=[_delta(d) for d in deltas], angle_count=shell_angles)
    graphics = [shell, _graphic(1, "b_wall_x1", angle_count=5)]
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match=match):
        gen._resolve_wall_pieces(graphics, 72, shell, 1)
    gen._game_root.set_lenient(True)
    assert gen._resolve_wall_pieces(graphics, 72, shell, 1) is None
    assert len(lenient) == 1 and lenient[0].endswith("(drawn as its body alone)")


def test_a_rotating_gate_piece_skips_the_gate_when_lenient(gen, lenient):
    units, graphics = _units(64, class_=39), [_graphic(0, "b_gate_x1", angle_count=8)]
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="gate pieces are assumed non-rotating"):
        gen._entry_for(units, graphics, Path("no-graphics"), 64, {})
    gen._game_root.set_lenient(True)
    assert gen._entry_for(units, graphics, Path("no-graphics"), 64, {}) == (None, "gate_no_modern_replacement")
    assert len(lenient) == 1


def test_a_pasture_whose_annex_tree_changed_is_a_plain_entry_when_lenient(gen, lenient):
    """The tree helper returns None instead of a tuple; the entry falls through to the generic path."""
    from types import SimpleNamespace as NS

    units, graphics = _units(1893, building=NS(annexes=[])), [_graphic(0, "b_pasture_x1")]
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="yields 0 pieces"):
        gen._entry_for(units, graphics, Path("no-graphics"), 1893, {})
    gen._game_root.set_lenient(True)
    entry, kind = gen._entry_for(units, graphics, Path("no-graphics"), 1893, {})
    assert kind == "plain" and "pieces" not in entry and entry["file_name"] == "b_pasture_x1"
    assert len(lenient) == 1


def test_a_fish_without_its_underwater_body_is_a_plain_entry_when_lenient(gen, lenient):
    """Never tagged underwater: no frame_offset, though the splash has the Marlin body's name."""
    units, graphics = _units(450), [_graphic(0, "a_fish_marlin1_underwater_x1")]
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="expected exactly 1"):
        gen._entry_for(units, graphics, Path("no-graphics"), 450, {})
    gen._game_root.set_lenient(True)
    entry, kind = gen._entry_for(units, graphics, Path("no-graphics"), 450, {})
    assert kind == "plain" and "frame_offset" not in entry and "dx" not in entry
    assert len(lenient) == 1 and lenient[0].endswith("(drawn as its standing graphic)")

def test_a_sail_off_the_shell_layer_draws_the_ship_as_its_hull_when_lenient(gen, lenient):
    graphics = _dock_shell()
    graphics[2].layer = 19
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="is layer 19, display_angle -1, not layer 20"):
        gen._resolve_shell_delta_pieces(graphics, 45, 70, graphics[0], graphics[1])
    gen._game_root.set_lenient(True)
    entry, kind = gen._entry_for(_units(45, type_=70), graphics, Path("no-graphics"), 45, {})
    assert kind == "plain" and "pieces" not in entry and entry["file_name"] == "BSHIPY1_S-C1_x1"
    assert len(lenient) == 1 and lenient[0].endswith("(drawn as 'BSHIPY1_S-C1_x1' alone)")


def test_a_town_centre_whose_art_has_no_slots_is_drawn_without_annexes_when_lenient(gen, lenient):
    """GBG's 109: its annex art is no _PIECE_SLOTS_BY_ART key."""
    from types import SimpleNamespace as NS

    units = _units(109, building=NS(annexes=[NS(unit_id=0, misplacement_x=1.0, misplacement_y=0.0)]))
    units[0] = NS(standing_graphic=(1, -1))
    graphics = [_graphic(0, "b_tc_x1"), _graphic(1, "b_tc_annex_x1")]
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="_PIECE_SLOTS_BY_ART has no slot"):
        gen._resolve_pieces(units, graphics, 109, graphics[0])
    gen._game_root.set_lenient(True)
    assert gen._resolve_pieces(units, graphics, 109, graphics[0]) is None
    assert len(lenient) == 1 and lenient[0].endswith("(drawn without its annexes)")


def test_a_decoration_shell_without_its_body_is_not_drawn_when_lenient(gen, lenient, monkeypatch):
    """The standing graphic of a _BODY_GRAPHIC_OVERRIDES const is only the pennant."""
    monkeypatch.setattr(gen, "_BODY_GRAPHIC_OVERRIDES", {72: 1})
    units = _units(72)
    graphics = [_graphic(0, "b_palisade_flag_x1", deltas=[_delta(2)]), _graphic(1, "b_palisade_x1"),
                _graphic(2, "legacy_body")]
    assert gen._body_delta_positions(units, graphics) == {}
    assert lenient[-1].endswith("(not drawn: its standing graphic is only a decoration over that body)")
    assert gen._entry_for(units, graphics, Path("no-graphics"), 72, {}) == (None, "body_override_missing")
    gen._game_root.set_lenient(False)
    with pytest.raises(SystemExit, match="has no modern body at delta position 0"):
        gen._entry_for(units, graphics, Path("no-graphics"), 72, {72: 0})
    gen._game_root.set_lenient(True)
    assert gen._entry_for(units, graphics, Path("no-graphics"), 72, {72: 0}) == (None, "body_override_missing")
    assert len(lenient) == 2 and lenient[-1].endswith("(not drawn: its standing graphic is only a decoration over that body)")


def _building_art_case(gen, monkeypatch):
    """Gaia and one civ over consts 10-14. Civ 1's art for 10 rotates (angle_count
    8, an invariant break), its 11 is a valid civ art; the age pair 11 -> 12 at
    age 3 breaks angle_count, 11 -> 13 at age 4 is fine."""
    from types import SimpleNamespace as NS

    def table():
        units: list = [None] * 15
        for const in range(10, 15):
            units[const] = NS(class_=3, name="b", clearance_size=(1.0, 1.0))
        return units

    gaia_units, civ_units = table(), table()
    plain = {"graphic_id": 1, "file_name": "b_x1", "angle_count": 1}
    gaia = {const: dict(plain) for const in range(10, 15)}
    gaia[12] = {**plain, "angle_count": 8}
    civ = {**gaia, 10: {**plain, "file_name": "civ10_x1", "angle_count": 8}, 11: {**plain, "file_name": "civ11_x1"}}
    by_civ = {id(gaia_units): gaia, id(civ_units): civ}
    monkeypatch.setattr(gen, "_entry_for", lambda units, *args: (dict(by_civ[id(units)][args[2]]), "plain"))
    monkeypatch.setattr(gen, "_ART_INVARIANT_HOLDOUTS", {})
    data = NS(civs=[NS(name="Gaia", units=gaia_units), NS(name="Civ", units=civ_units)])
    gaia_entries = {str(const): entry for const, entry in gaia.items()}
    return data, gaia_entries, {11: {3: 12, 4: 13}}


def test_building_art_violations_fail_the_run_when_strict(gen, monkeypatch):
    data, gaia_entries, ages = _building_art_case(gen, monkeypatch)
    with pytest.raises(SystemExit) as failed:
        gen._building_art(data, [], Path("no-graphics"), {}, gaia_entries, set(range(10, 15)), ages, {})
    assert failed.value.code == (
        "per-civ/per-age building art breaks the placed const's own logic -- re-verify against the .dat:\n"
        "  civ 0 age 3 11 -> 12: angle_count 8 != reference 1\n"
        "  civ 1 const 10: angle_count 8 != reference 1\n"
        "  civ 1 age 3 11 -> 12: angle_count 8 != reference 1\n"
        "  holdouts measured [], listed []"
    ), "the strict message and its violation order are today's"


def test_building_art_holds_out_a_civ_entry_and_drops_an_age_pair_when_lenient(gen, lenient, monkeypatch):
    data, gaia_entries, ages = _building_art_case(gen, monkeypatch)
    art = gen._building_art(data, [], Path("no-graphics"), {}, gaia_entries, set(range(10, 15)), ages, {})
    assert art["civ_art"] == {"1": {"11": 0}} and art["entries"][0]["file_name"] == "civ11_x1"
    assert art["age_upgrades"] == {"11": {"4": 13}}, "the bad pair is dropped from the one shared table"
    assert lenient == [
        "civ 0 age 3 11 -> 12: angle_count 8 != reference 1 (dropped from the age table for every civ)",
        "civ 1 const 10: angle_count 8 != reference 1 (drawn with Gaia's art)",
    ], "a lenient hold-out is not a measured one, so the holdout list check stays quiet"


def test_the_generator_gives_a_unit_without_a_garrison_graphic_no_flags(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.syspath_prepend(str(GRAPHIC_MAP_PATH.parent.parent / "tools"))
    gen = importlib.import_module("gen_unit_graphic_map")
    assert gen._garrison_flags(NS(creatable=None), [], 1) == []
    assert gen._garrison_flags(NS(creatable=NS(garrison_graphic=-1)), [], 655) == []


def test_only_the_butterflies_and_seagulls_are_particle_entries(graphics):
    found = {int(k): (e["graphic_id"], e["file_name"], e["art_scale"]) for k, e in graphics.items() if e.get("particle")}
    assert found == PARTICLE_ENTRIES
    for unit_const, (_graphic_id, _stem, scale) in PARTICLE_ENTRIES.items():
        entry = graphics[str(unit_const)]
        assert entry == {
            "graphic_id": entry["graphic_id"], "file_name": entry["file_name"],
            "angle_count": 1, "mirroring_mode": 0, "frame_count": 1,
            "particle": True, "art_scale": scale,
        }, unit_const
