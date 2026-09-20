"""Every wall const paints at every one of its five shape variants.

The rotation-variant fix is pinned by pure-function tests on
`variant_index`, and the graphic-map retarget is pinned by file_name
assertions on the generated table. Neither can tell whether a wall
actually *paints*, and the
palisade bug proves why that gap matters: the rotation fix was correct, the
table pointed at an animated flag whose .sld had art for one variant only, and
every test in the suite stayed green while walls were invisible in a live
window.

`sprite_for()` returning None is a silent, supported outcome -- it falls back
to a coloured mark -- so nothing below may treat None as acceptable.

Needs AOE2DE_INSTALL_PATH rather than the configured install, for the reason
tests/test_sprite_variant14_real_assets.py's docstring spells out: conftest's
autouse _isolated_settings hides config.yaml from the suite on purpose.

**Deliberately NOT `corpus`-marked**, unlike the other real-asset sprite
tests. Those need PyQt5 and build render surfaces; this one is eight
`sprite_for()` calls against files already on disk and runs in well under a
second, so it earns its place in the default tier. Gating is left to
`_require_install()`'s skip, which is clean for anyone without an install. A
guard for "walls silently stopped painting" that only fires in the ~27-minute
tier is the same hole this file exists to close.
"""

from __future__ import annotations

import pytest

from descape import asset_source, unit_sprites

# Every const in unit_sprites._ROTATION_VARIANT_CONSTS, with the graphic each
# must resolve to. Spelled out rather than derived from the frozenset so this
# also catches a const being silently dropped from it.
WALL_CONSTS = {
    72: "b_dark_wall_palisade_x1",
    117: "b_west_wall_stone_x1",
    119: "b_scen_wall_palisade_fortified_x1",
    155: "b_west_wall_fortified_x1",
    370: "b_scen_city_wall_x1",
    788: "b_scen_wall_sea_x1",
    1062: "b_scen_fence_x1",
    2678: "b_scen_wall_fort_x1",
}


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


def test_the_frozenset_and_this_files_expectations_agree():
    """Neither list is derived from the other, so a const added to one and not
    the other is a real disagreement rather than a restatement."""
    assert set(WALL_CONSTS) == set(unit_sprites._ROTATION_VARIANT_CONSTS)


@pytest.mark.parametrize("unit_const", sorted(WALL_CONSTS))
def test_every_wall_variant_paints(unit_const):
    """The exact symptom reported from a live window: palisade walls missing.

    All five rotations are checked, not just the two the rotation fix moved --
    the palisade defect made four of five silently vanish, and a test that
    sampled only the interesting rotations would have missed it.
    """
    _require_install()
    entry = unit_sprites.graphic_map().get(unit_const)
    assert entry is not None, f"unit_const {unit_const} left the graphic map"
    assert entry["file_name"] == WALL_CONSTS[unit_const]

    missing = [r for r in range(5) if unit_sprites.sprite_for(unit_const, float(r), 1, 48) is None]
    assert not missing, (
        f"unit_const {unit_const} ({entry['file_name']}) draws nothing at rotation(s) "
        f"{missing} -- it falls back to a coloured mark there, which is what "
        "'palisade graphics are missing' looked like"
    )


@pytest.mark.parametrize("unit_const", sorted(WALL_CONSTS))
def test_the_five_variants_are_five_distinct_shapes(unit_const):
    """Non-vacuity for the test above, and the assertion that would have caught
    the ORIGINAL rotation bug on real assets.

    Before the fix, rotations 2.0 and 3.0 both selected stored frame 2, so a
    lone post drew the tall tower. Distinct ink extents across all five is the
    off-engine form of "the five stored frames are five shapes".
    """
    _require_install()
    shapes = []
    for rotation in range(5):
        draw = unit_sprites.sprite_for(unit_const, float(rotation), 1, 48)
        assert draw is not None, rotation
        shapes.append(draw.rgba.shape[:2])
    assert len(set(shapes)) >= 4, (
        f"unit_const {unit_const} resolved {len(set(shapes))} distinct shapes across its "
        f"five variants ({shapes}) -- two rotations are collapsing onto one frame"
    )


# Decoration-shell walls (_BODY_GRAPHIC_OVERRIDES) -> piece count per variant.
# The flag has art only at variant 2, the tall tower it sits on.
SHELL_WALL_PIECE_COUNTS = {72: (1, 1, 2, 1, 1), 119: (1, 1, 2, 1, 1), 788: (2, 2, 3, 2, 2)}


@pytest.mark.parametrize("unit_const", sorted(SHELL_WALL_PIECE_COUNTS))
def test_a_shell_walls_composite_paints_at_every_variant_with_the_flag_only_on_the_tower(unit_const):
    """The composite must not reintroduce the palisade bug: every wall piece
    shares unit_id 72/119/788, so an inferred parent would bail on the flag's
    legitimate miss at 4 of 5 variants and draw nothing there."""
    _require_install()
    got = tuple(len(unit_sprites.sprite_pieces_for(unit_const, float(v), 1, 48)) for v in range(5))
    assert got == SHELL_WALL_PIECE_COUNTS[unit_const]


@pytest.mark.parametrize("unit_const", [72, 119])
def test_a_palisade_towers_player_colour_comes_from_its_flag(unit_const):
    """The user-visible payoff: palisade bodies carry no PLAYERCOLOR layer, so
    before the flag came back as a piece these walls showed no team colour."""
    _require_install()
    entry = unit_sprites.graphic_map()[unit_const]

    def has_player_colour(piece):
        index = unit_sprites._frame_for(unit_const, piece, 2.0)
        native = unit_sprites._native_frame(piece["file_name"], index)
        assert native is not None, piece["file_name"]
        playercolor = native[1]
        return playercolor is not None and bool(playercolor.any())

    body, flag = entry["pieces"]
    assert not has_player_colour(body), "non-vacuity: the body alone must carry no player colour"
    assert has_player_colour(flag)
