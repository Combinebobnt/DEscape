"""The pending turn of a Move drag (GH #280): each class steps as Rotate does,
and a turn that nets a whole cycle hands the stored value back by identity.
The drag that holds one is tests/test_unit_drag_preview.py's."""

from __future__ import annotations

import math

import pytest

from descape import gate_orientation, move_turn, unit_rotation, unit_variant
from descape.move_turn import MoveTurn

_ARCHER = 4  # ANGLE, 16 frames
_EIGHT_FRAME = 1956  # ANGLE, 8 frames
_OAK = 349  # cyclable, 42 variants
_PINE = 350  # cyclable, 27 variants
_GATE = 64
_WALL = 117
_CLIFF = 264
_HOUSE = 33  # INERT
_TURN = 2 * math.pi


def test_the_sample_consts_are_what_this_module_says():
    assert unit_rotation.angle_count_for(_ARCHER) == 16
    assert unit_rotation.angle_count_for(_EIGHT_FRAME) == 8
    assert unit_rotation.rotation_is_angle(_EIGHT_FRAME)
    assert unit_variant.variant_count_for(_OAK) == 42
    assert unit_variant.variant_count_for(_PINE) == 27


def test_a_turn_is_falsy_only_when_all_four_counts_are_zero():
    assert not MoveTurn()
    for field in ("facing", "gate", "variant", "variant_quarters"):
        assert MoveTurn(**{field: 1})
    assert len({MoveTurn(), MoveTurn(facing=0), MoveTurn(facing=1)}) == 2, "hashable, by value"


def test_fine_steps_each_class_as_rotate_does():
    turn = move_turn.fine(move_turn.fine(MoveTurn(), 1), 2)
    assert turn == MoveTurn(facing=3, gate=3, variant=3)
    assert move_turn.turned(turn, _ARCHER, 0.5) == (_ARCHER, unit_rotation.rotate_step(0.5, 16, 3))
    assert move_turn.turned(turn, _OAK, 5.0) == (_OAK, 8.0)
    assert move_turn.turned(turn, _GATE, 7.0) == (gate_orientation.cycle_const(_GATE, 3), 7.0)


@pytest.mark.parametrize(("const", "frames"), [(_ARCHER, 4), (_EIGHT_FRAME, 2)])
def test_coarse_is_a_quarter_turn_on_the_frame_count_it_is_given(const, frames):
    assert unit_rotation.quarter_turn_steps(unit_rotation.angle_count_for(const)) == frames
    turn = move_turn.coarse(MoveTurn(), 1, frames)
    assert turn == MoveTurn(facing=frames, gate=gate_orientation.QUARTER_TURN_STEPS, variant_quarters=1)
    assert move_turn.turned(turn, const, 0.0)[1] == pytest.approx(math.pi / 2)


def test_coarse_takes_a_variant_quarter_of_each_consts_own_count():
    turn = move_turn.coarse(MoveTurn(), 1, 4)
    assert move_turn.turned(turn, _OAK, 0.0) == (_OAK, 11.0)
    assert move_turn.turned(turn, _PINE, 0.0) == (_PINE, 7.0)
    assert move_turn.turned(move_turn.coarse(MoveTurn(), -1, 4), _OAK, 0.0) == (_OAK, 31.0)


def test_coarse_and_fine_add_up_on_a_gate():
    turn = move_turn.fine(move_turn.coarse(MoveTurn(), 1, 4), -1)
    assert move_turn.turned(turn, _GATE, 0.0)[0] == gate_orientation.cycle_const(_GATE, 1)


@pytest.mark.parametrize(
    ("const", "rotation"),
    [
        (_ARCHER, 7.0),  # the junk sentinel, off every frame
        (_ARCHER, 1.2345678),
        (_ARCHER, -0.5),
        (_OAK, 2 * _TURN / 5),  # radian-encoded
        (_OAK, 53.0),  # past its own count
        (_GATE, 7.0),
        (_WALL, _TURN / 5),
        (_HOUSE, 7.0),
    ],
)
def test_turning_and_turning_back_returns_the_stored_value_itself(const, rotation):
    """By identity: an `==` could not tell a value re-encoded by a round trip."""
    there = move_turn.fine(MoveTurn(), 1)
    back = move_turn.fine(there, -1)
    assert not back
    got_const, got_rotation = move_turn.turned(back, const, rotation)
    assert got_const == const and got_rotation is rotation


@pytest.mark.parametrize(
    ("const", "steps"),
    [(_ARCHER, 16), (_ARCHER, -32), (_EIGHT_FRAME, 8), (_OAK, 42), (_PINE, -27), (_GATE, 4), (_GATE, -8)],
)
def test_a_whole_cycle_returns_the_stored_value_itself(const, steps):
    """Truthy, unlike the case above, so only turned()'s own cycle test holds the value."""
    rotation = 7.0000001
    turn = move_turn.fine(MoveTurn(), steps)
    assert turn
    got_const, got_rotation = move_turn.turned(turn, const, rotation)
    assert got_const == const and got_rotation is rotation


def test_a_whole_cycle_of_quarters_returns_the_stored_value_itself():
    rotation = 3.0
    turn = MoveTurn(variant_quarters=42)  # 42 quarters of 11 steps each: 462 = 11 whole cycles
    assert move_turn.turned(turn, _OAK, rotation)[1] is rotation
    # A mixed turn is one sum: three quarters of 11 and a fine 9 are 42.
    assert move_turn.turned(MoveTurn(variant=9, variant_quarters=3), _OAK, rotation)[1] is rotation


def test_the_whole_cycle_is_each_consts_own():
    """One group, one turn: 16 frames is a whole cycle for an Archer and not for a gate's four or an oak."""
    turn = move_turn.fine(MoveTurn(), 18)
    assert move_turn.turned(turn, _ARCHER, 1.0) == (_ARCHER, unit_rotation.rotate_step(1.0, 16, 18))
    assert move_turn.turned(move_turn.fine(MoveTurn(), 16), _GATE, 0.0) == (_GATE, 0.0)
    assert move_turn.turned(move_turn.fine(MoveTurn(), 16), _OAK, 0.0) == (_OAK, 16.0)


@pytest.mark.parametrize("const", [_WALL, _CLIFF, _HOUSE])
@pytest.mark.parametrize("turn", [MoveTurn(facing=3, gate=3, variant=3), MoveTurn(facing=4, gate=2, variant_quarters=1)])
def test_walls_cliffs_and_inert_consts_pass_through(const, turn):
    rotation = 2.5
    got_const, got_rotation = move_turn.turned(turn, const, rotation)
    assert got_const == const and got_rotation is rotation


def test_a_variant_is_written_as_a_literal_from_a_radian_encoded_store():
    """cycle_step()'s rule, reached by a real turn, not only the whole-cycle pass-through."""
    _const, rotation = move_turn.turned(MoveTurn(variant=1), _OAK, 5.0)
    assert rotation == 6.0 and float(rotation).is_integer()
