"""unit_rotation's semantics classifier and step helper.

The pure cases run in the default tier -- unit_rotation reads only the two
committed JSON tables, with no library import and no configured install. The
corpus-marked test at the bottom is what pins the type-70 whitelist to real
data rather than to the judgement of the session that wrote it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from descape import unit_rotation

# Hand-picked representatives, each named for why it is in this list.
ARCHER = 4  # type 70, angle_count 16 -- the plain whitelist case
WALL = 117  # type 80, angle_count 5 -- shape variants, AGENTS.md's hard rule
STONE_GATE = 64  # type 80, angle_count 1 -- orientation lives in unit_const
OAK_TREE = 349  # GAIA flora, a doodad-variant index
TREBUCHET_PACKED = 331  # type 80 but measured angle-like -- the extension
MULE_CART = 1808  # type 80, checked and deliberately NOT whitelisted


def test_a_creatable_unit_rotation_is_an_angle():
    assert unit_rotation.semantics_for(ARCHER) == unit_rotation.ANGLE
    assert unit_rotation.rotation_is_angle(ARCHER)


def test_a_wall_is_a_variant_not_an_angle():
    assert unit_rotation.semantics_for(WALL) == unit_rotation.VARIANT
    assert not unit_rotation.rotation_is_angle(WALL)


def test_a_gate_is_inert_because_it_stores_one_frame():
    """Gates carry no rotation at all: angle_count == 1 across all 24 visible
    gate consts, so there is no second frame for a rotation to select."""
    assert unit_rotation.angle_count_for(STONE_GATE) == 1
    assert unit_rotation.semantics_for(STONE_GATE) == unit_rotation.INERT
    assert not unit_rotation.rotation_is_angle(STONE_GATE)


def test_a_tree_is_not_rotatable():
    assert not unit_rotation.rotation_is_angle(OAK_TREE)


def test_the_trebuchet_extension_is_an_angle_despite_being_typed_a_building():
    """42/331 are mobile units the .dat types as buildings (80). 30 of their
    33 corpus placements are angle-like -- the type-70 signature."""
    assert unit_rotation.semantics_for(TREBUCHET_PACKED) == unit_rotation.ANGLE
    assert unit_rotation.semantics_for(42) == unit_rotation.ANGLE


def test_mule_cart_was_checked_and_stays_excluded():
    """Both its corpus placements are integer-in-range at angle_count 16 --
    the variant signature. "It's a mobile unit" was the wrong prior, and this
    test is what stops that prior being re-applied later."""
    assert not unit_rotation.rotation_is_angle(MULE_CART)


def test_an_unknown_const_is_inert_rather_than_an_error():
    unknown = next(c for c in range(1, 100_000) if unit_rotation.angle_count_for(c) == 1)
    assert unit_rotation.semantics_for(unknown) == unit_rotation.INERT


@pytest.mark.parametrize("angle_count,expected", [(16, 4), (32, 8), (8, 2), (6, 2), (4, 1), (2, 1)])
def test_a_quarter_turn_is_always_a_whole_number_of_frames(angle_count, expected):
    """Rounds to whole frames rather than writing an exact 90 degrees: at
    angle_count 6 a quarter turn is 1.5 frames, and writing the raw value
    would put the stored rotation off the graphic's own grid."""
    assert unit_rotation.quarter_turn_steps(angle_count) == expected


def test_one_step_advances_by_exactly_one_frame():
    assert unit_rotation.rotate_step(0.0, 16, 1) == pytest.approx(2 * math.pi / 16)
    assert unit_rotation.rotate_step(0.0, 32, 1) == pytest.approx(2 * math.pi / 32)


def test_a_full_turn_of_steps_returns_to_the_start():
    for angle_count in (2, 6, 16, 32):
        assert unit_rotation.rotate_step(0.0, angle_count, angle_count) == pytest.approx(0.0)


def test_stepping_wraps_into_range_rather_than_growing():
    value = unit_rotation.rotate_step(2 * math.pi - 0.01, 16, 1)
    assert 0.0 <= value < 2 * math.pi


def test_stepping_backwards_wraps_the_other_way():
    value = unit_rotation.rotate_step(0.0, 16, -1)
    assert value == pytest.approx(2 * math.pi * 15 / 16)


def test_zero_steps_normalizes_a_junk_stored_value():
    """7.0 appears 574 times in the corpus and is outside [0, 2*pi). A typed
    inspector value goes through the same helper, so it is clamped the same
    way without moving off its own frame."""
    assert unit_rotation.rotate_step(7.0, 16, 0) == pytest.approx(7.0 - 2 * math.pi)


def test_rotating_an_inert_graphic_raises():
    with pytest.raises(ValueError):
        unit_rotation.rotate_step(0.0, 1, 1)


@pytest.mark.corpus
def test_the_whitelist_has_zero_counterexamples_on_real_files(corpus_files):
    """The discriminator itself, re-measured rather than trusted.

    Every type-70 unit whose graphic stores more than one frame must carry a
    rotation that is angle-like, or zero -- which is index 0 under either
    convention and so is evidence for neither. A counterexample would mean a
    whitelisted const genuinely stores a variant index, i.e. that the Rotate
    action can corrupt it.

    "Zero" means within the same 1e-6 the integer test itself uses, not
    `== 0.0`. The field is stored as a float32 and real files carry values
    like 2.3e-08 and 4.8e-07 for it (5 units across the quick corpus subset).
    Those are 0 at float32 resolution and cannot be a variant index, since
    index 0 IS zero -- an exact-equality check reports them as failures for a
    reason that has nothing to do with the whitelist.
    """
    from descape import scenario_io

    counterexamples = []
    for path in corpus_files:
        loaded = scenario_io.load_map_and_units(path)
        for units in loaded.unit_manager.units:
            for unit in units:
                const = unit.unit_const
                if not unit_rotation.rotation_is_angle(const):
                    continue
                angle_count = unit_rotation.angle_count_for(const)
                rotation = unit.rotation
                nearest = round(rotation)
                integer_in_range = abs(rotation - nearest) < 1e-6 and 0 <= nearest < angle_count
                if integer_in_range and abs(rotation) > 1e-6:
                    counterexamples.append((path.name, const, rotation, angle_count))
    assert not counterexamples, f"{len(counterexamples)} counterexamples: {counterexamples[:10]}"


# -- the in-app leg: a rotate must be VISIBLE, not merely stored ---------------
#
# Needs a real AoE2:DE install (AOE2DE_INSTALL_PATH, not the configured
# config.yaml -- conftest._isolated_settings deliberately hides that one, see
# tests/test_sprite_edit_real_assets.py's own docstring on why).

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_REF_ARCHER_P2 = 300  # unoccluded; the P1 archer at ref 201 draws under a house
_REF_VILLAGER_P2 = 301  # its untouched neighbour, the asymmetry control


@pytest.mark.corpus
def test_rotating_changes_the_rendered_sprite_and_leaves_its_neighbour_alone():
    """"The value changed" is not the claim. This asserts that DEscape's own
    render draws a DIFFERENT sprite afterwards, and -- the half a
    same-everywhere check would miss -- that the adjacent untouched unit of a
    different const is byte-identical, so a global repaint or a mirrored write
    cannot pass.
    """
    from descape import asset_source, render, unit_rotation
    from descape.scenario_io import load_map_and_units
    from descape.unit_model import UnitEditModel

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")

    loaded = load_map_and_units(_FIXTURE)
    model = UnitEditModel(loaded)
    archer = next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == _REF_ARCHER_P2)
    villager = next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == _REF_VILLAGER_P2)
    assert unit_rotation.rotation_is_angle(archer.unit_const)

    def sprite_shapes():
        _img, elevations, proj = render.render_terrain_iso_with_proj(loaded, with_sprites=True)
        layer = render.sprite_draws_by_anchor(loaded, proj, elevations)
        shapes = {}
        for anchor, draws in layer.by_anchor.items():
            for item in draws:
                draw = item[0] if isinstance(item, tuple) else item
                shapes[anchor] = (draw.rgba.shape, int((draw.rgba[..., 3] > 0).sum()))
        return shapes

    archer_anchor = (int(archer.x), int(archer.y))
    villager_anchor = (int(villager.x), int(villager.y))
    before = sprite_shapes()
    assert archer_anchor in before, "the archer resolved no sprite, so this proves nothing"

    quarter = unit_rotation.quarter_turn_steps(unit_rotation.angle_count_for(archer.unit_const))
    model.set_rotation(
        archer,
        unit_rotation.rotate_step(archer.rotation, unit_rotation.angle_count_for(archer.unit_const), quarter),
    )
    after = sprite_shapes()

    assert after[archer_anchor] != before[archer_anchor], "the rotated unit draws the same frame"
    assert after[villager_anchor] == before[villager_anchor], "an untouched neighbour changed too"
