"""Covers descape/unit_fields.py: the Qt-free field-spec module for the
Units-mode inspector.

Static-tuple module, so these tests pin its declared shape rather than any
computed presence-gate logic -- there is none here, unlike option_fields.py.
"""

from __future__ import annotations

import math

from descape import unit_fields


def test_every_field_has_a_unique_id():
    ids = [spec.field_id for spec in unit_fields.FIELDS]
    assert len(ids) == len(set(ids))


def test_fields_by_id_mirrors_fields():
    assert unit_fields.FIELDS_BY_ID == {spec.field_id: spec for spec in unit_fields.FIELDS}


def test_only_xyz_owner_and_rotation_are_editable():
    """D5: X, Y, Z and Owner get real editors, plus Rotation as of b3 (which
    is additionally per-const gated). Everything else (Name, Unit ID,
    Reference ID, Garrisoned in) stays a read-only label."""
    editable = {spec.field_id for spec in unit_fields.FIELDS if spec.editable}
    assert editable == {"x", "y", "z", "player", "rotation"}


def test_rotation_is_the_only_conditional_field():
    """AGENTS.md's hard rule survives as a conditional, not a blanket ban:
    rotation is a shape-variant index for most GAIA objects and every wall, so
    its editor is gated per-const rather than always live."""
    conditional = {spec.field_id for spec in unit_fields.FIELDS if spec.conditional}
    assert conditional == {"rotation"}
    assert unit_fields.FIELDS_BY_ID["rotation"].conditional == "rotation_is_angle"


def test_rotation_is_bounded_to_one_turn_in_raw_radians():
    """Raw radians, not degrees -- the value actually stored. The bound is
    what makes a typed value normalizable by the same helper the tool path
    uses."""
    spec = unit_fields.FIELDS_BY_ID["rotation"]
    assert spec.kind == unit_fields.FLOAT
    assert (spec.minimum, spec.maximum) == (0.0, 2 * math.pi)
    assert spec.decimals > 2, "two decimals would round a stored frame off its own grid"


def test_float_fields_declare_bounds():
    for spec in unit_fields.FIELDS:
        if spec.kind == unit_fields.FLOAT:
            assert spec.minimum is not None
            assert spec.maximum is not None
            assert spec.minimum < spec.maximum


def test_owner_field_is_the_player_kind():
    spec = unit_fields.FIELDS_BY_ID["player"]
    assert spec.kind == unit_fields.PLAYER
