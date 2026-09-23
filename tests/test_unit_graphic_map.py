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
# reported bug (RTWC drew only its back quarter); 71 is the regression guard
# against reintroducing per-civ/age retargeting -- Incas' own back piece is
# Andean-style, but the game composites it with the SAME Dark Age
# main/center/front graphics as every other town centre const, not with
# Andean-suffixed siblings.
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
        fields = set(entry) - {"pieces", "rotation_is_variant", "variant_count"}
        assert fields == set(FIELDS), f"unit_const {key} has fields {sorted(entry)}"
        for field, kind in FIELDS.items():
            assert isinstance(entry[field], kind), f"unit_const {key}.{field}"
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
    for unit_const in sorted(generator._COMPOSITE_SCOPE | generator._ANNEX_TREE_SCOPE):
        result = measure.measure(unit_const)
        if result is not None:
            measured[unit_const] = [list(row["slot"]) for row in result["rows"]]
    assert measured, "no composite piece art was readable, so this proves nothing"
    for unit_const, slots in measured.items():
        assert generator._PIECE_SLOTS[unit_const] == slots, (
            f"unit_const {unit_const}: the real art now wants {slots}. Re-run "
            f"tools/measure_piece_slots.py and commit its table"
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
    centre/pasture scope plus the three decoration-shell walls."""
    assert "pieces" not in graphics["45"]  # Dock, resolved via its own shell delta
    non_gate = {
        int(k) for k, e in graphics.items() if "pieces" in e and "_gate_" not in e["file_name"]
    }
    assert non_gate == SLOTTED_CONSTS | set(KNOWN_WALL_PIECES)
