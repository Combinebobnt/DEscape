"""descape/place_rotation.py (GH #172): the rotation Place Unit's ghost and
click share, per const class. Qt-free, install-free."""

from __future__ import annotations

import math

import pytest

from descape import gate_orientation, unit_rotation, unit_sprites, unit_variant
from descape.place_rotation import (
    CYCLED,
    GATE,
    NOTHING,
    TURNED,
    PlaceRotation,
    rotation_for,
    step,
    steps_for_coarse,
)

ARCHER = 4  # ANGLE, 16 frames
EIGHT_FRAMES = 1956  # ANGLE, 8 frames
SIX_FRAMES = 2431  # ANGLE, 6 frames
OAK = 349  # cyclable, 42 variants
PINE = 350  # cyclable, 27 variants
STONE_GATE = 64
HOUSE = 33  # INERT
STONE_WALL = 117
P1 = 1
GAIA = 0


def _cliff() -> int:
    return min(unit_sprites._CLIFF_VARIANT_CONSTS)


def test_the_fixture_consts_are_what_the_names_say() -> None:
    assert unit_rotation.rotation_is_angle(ARCHER) and unit_rotation.angle_count_for(ARCHER) == 16
    assert unit_rotation.rotation_is_angle(EIGHT_FRAMES) and unit_rotation.angle_count_for(EIGHT_FRAMES) == 8
    assert unit_rotation.rotation_is_angle(SIX_FRAMES) and unit_rotation.angle_count_for(SIX_FRAMES) == 6
    assert unit_variant.is_cyclable(OAK) and unit_variant.variant_count_for(OAK) == 42
    assert unit_variant.is_cyclable(PINE)
    assert gate_orientation.is_gate(STONE_GATE)
    assert unit_rotation.semantics_for(HOUSE) == unit_rotation.INERT
    assert unit_rotation.semantics_for(STONE_WALL) == unit_rotation.VARIANT
    assert not unit_variant.is_cyclable(STONE_WALL)


# --- ANGLE --------------------------------------------------------------------


def test_an_angle_step_is_one_frame_on_that_consts_grid() -> None:
    state, const, outcome = step(PlaceRotation(), ARCHER, P1, 1)
    assert (const, outcome) == (ARCHER, TURNED)
    assert rotation_for(state, ARCHER, P1) == pytest.approx(2 * math.pi / 16)
    state, _, _ = step(state, ARCHER, P1, -2)
    assert rotation_for(state, ARCHER, P1) == pytest.approx(2 * math.pi * 15 / 16)


def test_the_coarse_angle_step_is_a_quarter_turn() -> None:
    steps = steps_for_coarse(ARCHER, 1)
    assert steps == 4
    state, _, _ = step(PlaceRotation(), ARCHER, P1, steps)
    assert rotation_for(state, ARCHER, P1) == pytest.approx(math.pi / 2)
    assert steps_for_coarse(EIGHT_FRAMES, -1) == -2


def test_the_facing_carries_across_consts_snapped_to_each_grid() -> None:
    state, _, _ = step(PlaceRotation(), ARCHER, P1, 6)  # 135 degrees
    assert rotation_for(state, ARCHER, P1) == pytest.approx(math.radians(135))
    assert rotation_for(state, EIGHT_FRAMES, P1) == pytest.approx(math.radians(135))
    # 135 degrees is 2.25 frames of 60: the nearest stored frame is 120.
    assert rotation_for(state, SIX_FRAMES, P1) == pytest.approx(math.radians(120))
    # A step on the coarser grid starts from its own snapped frame.
    state, _, _ = step(state, SIX_FRAMES, P1, 1)
    assert rotation_for(state, SIX_FRAMES, P1) == pytest.approx(math.radians(180))
    assert rotation_for(state, ARCHER, P1) == pytest.approx(math.radians(180))


def test_every_angle_value_is_exactly_on_the_consts_grid() -> None:
    state = PlaceRotation()
    for _ in range(40):
        state, _, _ = step(state, ARCHER, P1, 3)
        value = rotation_for(state, SIX_FRAMES, P1)
        facing = unit_rotation.rotation_to_facing(value, 6)
        assert value == unit_rotation.facing_to_rotation(facing, 6)
        assert 0.0 <= value < 2 * math.pi


def test_gaia_angle_turns_and_places_its_facing_like_a_player() -> None:
    """GH #217: GAIA creatables have a facing the game draws, so the owner no longer matters."""
    state, const, outcome = step(PlaceRotation(), ARCHER, GAIA, 3)
    assert (const, outcome) == (ARCHER, TURNED)
    assert rotation_for(state, ARCHER, GAIA) == pytest.approx(2 * math.pi * 3 / 16)
    assert rotation_for(state, ARCHER, GAIA) == rotation_for(state, ARCHER, P1)


# --- Cyclable -----------------------------------------------------------------


def test_a_variant_wraps_at_its_count() -> None:
    state, const, outcome = step(PlaceRotation(), OAK, P1, -1)
    assert (const, outcome) == (OAK, CYCLED)
    assert rotation_for(state, OAK, P1) == 41.0
    state, _, _ = step(state, OAK, P1, 1)
    assert rotation_for(state, OAK, P1) == 0.0


def test_a_variant_resets_on_a_different_const() -> None:
    state, _, _ = step(PlaceRotation(), OAK, P1, 5)
    assert rotation_for(state, OAK, P1) == 5.0
    assert rotation_for(state, PINE, P1) == 0.0
    state, _, _ = step(state, PINE, P1, 2)
    assert rotation_for(state, PINE, P1) == 2.0
    assert rotation_for(state, OAK, P1) == 0.0, "the variant belongs to the const it was chosen on"


def test_a_variant_is_always_a_literal_integer_and_gaia_still_cycles() -> None:
    state = PlaceRotation()
    for steps in (1, 7, -3, steps_for_coarse(OAK, 1), 40):
        state, _, outcome = step(state, OAK, GAIA, steps)
        assert outcome == CYCLED
        value = rotation_for(state, OAK, GAIA)
        assert isinstance(value, float) and value == int(value)
        assert 0 <= value < 42
    assert steps_for_coarse(OAK, 1) == unit_rotation.quarter_turn_steps(42)


def test_gh217_changes_only_angle_consts_for_gaia() -> None:
    """GH #217 widened GAIA to ANGLE only: a GAIA variant index (well outside
    [0, 2pi)) and an INERT const place exactly what a player's would."""
    state, _, outcome = step(PlaceRotation(), OAK, GAIA, 41)
    assert outcome == CYCLED
    assert rotation_for(state, OAK, GAIA) == rotation_for(state, OAK, P1) == 41.0
    start = PlaceRotation(facing=2.0, variant=4, variant_const=HOUSE)
    assert step(start, HOUSE, GAIA, 1) == step(start, HOUSE, P1, 1) == (start, HOUSE, NOTHING)
    assert rotation_for(start, HOUSE, GAIA) == rotation_for(start, HOUSE, P1) == 0.0


def test_a_variant_does_not_leak_into_the_facing_and_back() -> None:
    state, _, _ = step(PlaceRotation(), OAK, P1, 9)
    assert rotation_for(state, ARCHER, P1) == 0.0
    state, _, _ = step(state, ARCHER, P1, 2)
    assert rotation_for(state, OAK, P1) == 9.0


# --- Gate -----------------------------------------------------------------------


def test_a_gate_step_returns_its_cycle_sibling_and_keeps_the_state() -> None:
    start = PlaceRotation(facing=1.0, variant=3, variant_const=OAK)
    state, const, outcome = step(start, STONE_GATE, P1, 1)
    assert (state, outcome) == (start, GATE)
    assert const == gate_orientation.cycle_const(STONE_GATE, 1)
    _, const, _ = step(start, STONE_GATE, P1, steps_for_coarse(STONE_GATE, 1))
    assert const == gate_orientation.cycle_const(STONE_GATE, 2)
    assert rotation_for(start, STONE_GATE, P1) == 0.0


# --- Nothing to rotate ------------------------------------------------------------


@pytest.mark.parametrize("const", [STONE_WALL, HOUSE, "cliff"])
def test_walls_cliffs_and_inert_are_a_no_op_at_zero(const) -> None:
    const = _cliff() if const == "cliff" else const
    start = PlaceRotation(facing=2.0, variant=4, variant_const=const)
    state, same, outcome = step(start, const, P1, 1)
    assert (state, same, outcome) == (start, const, NOTHING)
    assert rotation_for(start, const, P1) == 0.0
    assert rotation_for(start, const, GAIA) == 0.0
    assert steps_for_coarse(const, -1) == -1
