"""Cliff rendering: the rotation seam, the frame dispatch, and the range check.

The reported symptom was "cliffs render wrong" -- every cliff on every map drew
frame 0 of its shape set, so a cliff fort read as a scatter of identical rock
chunks instead of connected runs and corners. Three separate defects produced
it, and each is pinned separately below because fixing any two of them still
leaves cliffs broken:

1. `render.stored_rotation` zeroed every GAIA unit's rotation before lookup, so
   the shape index never reached the sprite path at all.
2. Un-zeroing alone is worse than the bug: without cliffs in the variant set
   the index goes through `angle_index()`, which scrambles it and can return an
   index past the file's last frame -- a coloured mark instead of a sprite.
3. The integer branch of `variant_index()` was bounded by the .dat's
   `angle_count`, which disagrees with the real .sld frame count in BOTH
   directions across the cliff families. Being over-tight is the dangerous one:
   it doesn't raise, it falls through to the radian branch and returns a
   plausible wrong frame.

The half-tile anchoring half of the same fix lives in test_unit_footprints.py,
next to the other ground-truth footprint tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import asset_source, render, unit_sprites
from descape.scenario_io import load_map_and_units

# Two cliff consts whose graphics bracket the angle_count-vs-real-frames
# disagreement, with the numbers measured off the real files.
DEFAULT_CLIFF = 264  # n_cliff_default_x1: angle_count 25, 24 real frames
SHORT_CLIFF = 2199  # n_short_all_cliff_x1: angle_count 23, 24 real frames
MARBLE_CLIFF = 2178  # n_cliff_marble_x1: angle_count 25, 23 real frames

TREE_CONST = 349  # oak: 42 variants, was drawing variant 0 before Tier B (see below)
CREATABLE_GAIA_CONST = 705  # Cow Black and White: type 70, angle_count > 1 -- stays ANGLE


@dataclass
class Unit:
    unit_const: int
    rotation: float


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


# -- 1. the rotation seam ------------------------------------------------


def test_a_gaia_cliff_keeps_its_stored_rotation():
    """The whole reported symptom in one assertion: this returned 0.0 for every
    GAIA unit, and every cliff is GAIA."""
    assert render.stored_rotation(0, Unit(DEFAULT_CLIFF, 7.0)) == 7.0


def test_a_gaia_tree_now_keeps_its_stored_rotation():
    """Tier B (2026-09-06 plan): the generated rotation_is_variant field widens
    the narrowing from "hand-verified walls and cliffs only" to every
    non-creatable graphic, which includes trees. This is the fix for the
    reported bug -- every tree in every map drew variant 0 -- and
    test_cliffs.py's own module docstring names it as the deliberately
    deferred item Track A2 left open."""
    assert unit_sprites.rotation_is_variant(TREE_CONST)
    assert render.stored_rotation(0, Unit(TREE_CONST, 19.0)) == 19.0


def test_a_gaia_creatable_unit_still_zeroes():
    """The narrowing is to non-creatable graphics, not to GAIA generally: a
    GAIA-placed but creatable unit (e.g. wildlife) stays on the angle_index()
    path, so its rotation keeps zeroing under the "we don't know a facing
    from an angle offset for this GAIA unit" rule. Asserted so a later
    widening past creatable units is a decision rather than an accident."""
    assert not unit_sprites.rotation_is_variant(CREATABLE_GAIA_CONST)
    assert render.stored_rotation(0, Unit(CREATABLE_GAIA_CONST, 3.0)) == 0.0


def test_a_non_gaia_units_rotation_was_never_touched():
    assert render.stored_rotation(3, Unit(TREE_CONST, 19.0)) == 19.0


# -- 2. the frame dispatch -----------------------------------------------


def test_a_cliff_resolves_its_stored_index_not_an_angle():
    """angle_index() would permute 3.0 into 12 at angle_count 25. The two
    answers are asserted against each other, not just against 3, so this stays
    meaningful if either function's internals change."""
    entry = unit_sprites.graphic_map()[DEFAULT_CLIFF]
    assert unit_sprites._frame_for(DEFAULT_CLIFF, entry, 3.0) == 3
    assert unit_sprites.angle_index(3.0, int(entry["angle_count"])) == 12


@pytest.mark.parametrize("unit_const", [DEFAULT_CLIFF, SHORT_CLIFF, MARBLE_CLIFF, 1339])
def test_every_cliff_family_is_variant_indexed(unit_const):
    assert unit_sprites.rotation_is_variant(unit_const)


def test_cliffs_are_not_wall_connectors():
    """_CLIFF_VARIANT_CONSTS is deliberately a second set rather than more
    entries in _ROTATION_VARIANT_CONSTS: WALL_CONNECTOR_CONSTS is built from
    that one, so folding cliffs in would make every cliff count as a wall
    neighbour and reshape real walls through
    render.wall_variant_rotation_overrides()."""
    assert not (unit_sprites._CLIFF_VARIANT_CONSTS & unit_sprites.WALL_CONNECTOR_CONSTS)


# -- 3. the range check --------------------------------------------------


def test_an_out_of_range_literal_index_wraps_rather_than_falls_through():
    """Tier B (2026-09-06 plan): an exact integer at or past variant_count is
    definitionally not a radian encoding of anything -- 23.0 is not
    `k*2*pi/23` for any k -- so it wraps by modulo instead of falling through
    to the radian branch, which used to return a plausible-looking wrong
    frame (15, previously asserted here) with no error and no None. Affects
    119 real corpus placements: mangrove, forage bush, skeleton and mole all
    store an integer past their own frame count."""
    assert unit_sprites.variant_index(23.0, 23) == 0


def test_the_real_frame_count_is_what_bounds_a_literal_index():
    assert unit_sprites.variant_index(23.0, 23, 24) == 23


def test_a_tighter_real_frame_count_still_rejects_an_out_of_file_index():
    """Marble fails the other way: 25 declared angles over a 23-frame file, so
    23 and 24 are literal-looking values with no frame behind them. The bound
    has to move in both directions, not just up."""
    assert unit_sprites.variant_index(23.0, 25) == 23
    assert unit_sprites.variant_index(23.0, 25, 23) != 23


def test_variant_count_defaults_to_angle_count():
    """Every pre-existing caller passes two arguments; none of them may move."""
    for rotation in (0.0, 2.0, 4.0, 2.513274):
        assert unit_sprites.variant_index(rotation, 5) == unit_sprites.variant_index(rotation, 5, 5)


# -- real assets ---------------------------------------------------------


@pytest.mark.parametrize(
    ("file_name", "expected_frames"),
    [
        ("n_cliff_default_x1", 24),
        ("n_cliff_sand_x1", 24),
        ("n_cliff_snow_x1", 24),
        ("n_cliff_limestone_x1", 24),
        ("n_cliff_marble_x1", 23),
        ("n_short_marble_cliff_x1", 23),
        ("n_short_all_cliff_x1", 24),
        ("n_short_sand_cliff_x1", 24),
        ("n_short_snow_cliff_x1", 24),
        ("n_cliff_terrace_x1", 24),
    ],
)
def test_the_real_frame_counts_are_what_the_range_check_assumes(file_name, expected_frames):
    """Ground truth from outside the code: the numbers every assertion above is
    written against, read back off the installed files. A game patch that
    changes one turns this red rather than silently shifting which frame a
    stored index resolves to."""
    _require_install()
    assert unit_sprites.sld_frame_count(file_name) == expected_frames


@pytest.mark.parametrize("rotation", [0.0, 1.0, 12.0, 23.0])
def test_a_cliff_actually_paints_at_the_frames_the_dispatch_picks(rotation):
    """`sprite_for()` returning None is a supported outcome that degrades to a
    coloured mark, so nothing here may treat it as acceptable -- the same
    reasoning test_wall_variants_real_assets.py's docstring spells out. Frame
    23 is the one the old angle_count bound could not reach on a Short cliff.
    """
    _require_install()
    for unit_const in (DEFAULT_CLIFF, SHORT_CLIFF):
        assert unit_sprites.sprite_for(unit_const, rotation, 0, 32) is not None, (
            f"const {unit_const} at rotation {rotation} paints nothing"
        )


# -- 4. the placement tool (Track B Stage 1) -----------------------------


def test_span_anchor_is_the_inverse_of_span_start():
    """span_anchor(tile, span) must make _span_start(anchor, span) == tile
    for every span shape a real cliff piece carries (A4's own table) -- the
    Cliff tool's own snap formula, used forward (tile -> coordinate) instead
    of _span_start's backward (coordinate -> tile)."""
    for span_x, span_y in [(3, 3), (1, 3), (3, 1), (2, 3), (3, 2), (2, 2)]:
        x, y = render.span_anchor(10, 20, span_x, span_y)
        assert render._span_start(x, span_x) == 10
        assert render._span_start(y, span_y) == 20


def test_cliff_catalog_covers_every_cliff_const():
    """cliff_catalog.families() -- the Cliff tool's own Family/Piece source
    -- must cover exactly the 96 consts _CLIFF_VARIANT_CONSTS does, or the
    picker silently omits a real placeable piece."""
    from descape import cliff_catalog

    pieces = {p.unit_const for fam in cliff_catalog.families().values() for p in fam}
    assert pieces == unit_sprites.cliff_consts()
    assert len(pieces) == 96


def test_distinct_stored_indices_give_distinct_cliff_shapes():
    """The frames are SHAPES, so different indices must produce different
    pixels. Before the fix every cliff resolved index 0, which is exactly the
    state this rules out -- and a byte-identity check is what catches a future
    change that re-collapses them without failing anything else."""
    _require_install()
    seen = {}
    for rotation in range(8):
        draw = unit_sprites.sprite_for(DEFAULT_CLIFF, float(rotation), 0, 32)
        assert draw is not None, rotation
        seen[rotation] = draw.rgba.tobytes()
    assert len(set(seen.values())) == len(seen), "some cliff shapes are byte-identical"


# -- 5. Tier B: the generated rotation_is_variant widening ---------------

_OLD_VARIANT_CONSTS = unit_sprites._ROTATION_VARIANT_CONSTS | unit_sprites._CLIFF_VARIANT_CONSTS

# a_fish_dorado_x1 (455, type 30): investigated directly (tools/
# scan_tree_rotation.py's own development) and confirmed pre-existing --
# frame 0, the OLD dispatch's own resolved frame for every one of this
# const's 33 corpus placements, already has no MAIN layer (a sparse
# decay/idle animation gap, not a Tier B regression). Excluded here so a
# known, already-there gap doesn't mask a genuinely new one.
_PRE_EXISTING_NO_SPRITE_CONSTS = frozenset({455})


@pytest.mark.corpus
def test_tier_b_introduces_no_new_none_sprite(corpus_files):
    """The failure the plan calls out as otherwise silent: a const that
    becomes variant-indexed here must not, for any real corpus placement,
    newly resolve to a frame `sprite_for()` can't decode where the pre-Tier-B
    dispatch (angle_index(), since none of these consts were in the old
    hand-kept variant sets) resolved to a real sprite."""
    _require_install()
    graphic_map = unit_sprites.graphic_map()
    regressions = []
    for path in corpus_files:
        scenario = load_map_and_units(path)
        for player_id, units in enumerate(scenario.unit_manager.units):
            for unit in units:
                const = unit.unit_const
                if const in _OLD_VARIANT_CONSTS or const in _PRE_EXISTING_NO_SPRITE_CONSTS:
                    continue
                if not unit_sprites.rotation_is_variant(const):
                    continue
                entry = graphic_map.get(const)
                if entry is None:
                    continue
                angle_count = max(1, int(entry["angle_count"]))
                frame_count = max(1, int(entry["frame_count"]))
                old_rotation = 0.0 if player_id == 0 else float(unit.rotation)
                old_index = unit_sprites.angle_index(old_rotation, angle_count) * frame_count
                old_ok = unit_sprites._native_frame(str(entry["file_name"]), old_index) is not None
                if not old_ok:
                    continue
                new_rotation = render.stored_rotation(player_id, unit)
                if unit_sprites.sprite_for(const, new_rotation, player_id, 32) is None:
                    regressions.append((path.name, const, unit.rotation))
    assert not regressions, regressions


@pytest.mark.corpus
def test_a_forest_heavy_file_now_draws_many_distinct_tree_frames(corpus_files):
    """A measurable before/after, not an eyeball: before Tier B every oak placement
    resolved frame 0 (GAIA zeroed, then angle_index() collapsed anything that
    slipped through); after, a forest-heavy file's real placements should
    span most of the 42 stored variants."""
    _require_install()
    forest_file = next((p for p in corpus_files if "Dos Pilas" in p.name), None)
    if forest_file is None:
        pytest.skip("the forest-heavy corpus file isn't in this run's corpus_files")
    scenario = load_map_and_units(forest_file)
    entry = unit_sprites.graphic_map()[TREE_CONST]
    angle_count = int(entry["angle_count"])

    indices = set()
    placements = 0
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if unit.unit_const != TREE_CONST:
                continue
            placements += 1
            rotation = render.stored_rotation(player_id, unit)
            indices.add(unit_sprites._frame_for(TREE_CONST, entry, rotation))

    assert placements > 1000, "forest-heavy fixture no longer has enough oak placements to be meaningful"
    assert len(indices) >= angle_count - 2, (
        f"only {len(indices)} distinct frames drawn across {placements} oak placements, "
        f"expected close to angle_count ({angle_count})"
    )

    # A distinct index number means nothing if it's the same art -- confirm
    # two of them are actually different pixels, the same byte-identity bar
    # test_distinct_stored_indices_give_distinct_cliff_shapes holds cliffs to.
    zero = unit_sprites.sprite_for(TREE_CONST, 0.0, 0, 32)
    other_rotation = next(
        r for r in range(1, angle_count)
        if unit_sprites._frame_for(TREE_CONST, entry, float(r))
        != unit_sprites._frame_for(TREE_CONST, entry, 0.0)
    )
    other = unit_sprites.sprite_for(TREE_CONST, float(other_rotation), 0, 32)
    assert zero is not None and other is not None
    assert zero.rgba.tobytes() != other.rgba.tobytes()
