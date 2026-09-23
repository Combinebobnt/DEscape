"""Covers descape/unit_fields.py: the Qt-free field-spec module for the
Units-mode inspector.

Static-tuple module, so these tests pin its declared shape rather than any
computed presence-gate logic -- there is none here, unlike option_fields.py.
"""

from __future__ import annotations

from descape import unit_fields


def test_every_field_has_a_unique_id():
    ids = [spec.field_id for spec in unit_fields.FIELDS]
    assert len(ids) == len(set(ids))


def test_fields_by_id_mirrors_fields():
    assert {spec.field_id: spec for spec in unit_fields.FIELDS} == unit_fields.FIELDS_BY_ID


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


def test_rotation_is_edited_as_a_facing_with_a_per_const_range():
    """GH #61: a whole facing, not raw radians. The range is the const's own
    angle_count, so the spec carries no bounds of its own."""
    spec = unit_fields.FIELDS_BY_ID["rotation"]
    assert spec.kind == unit_fields.FACING
    assert (spec.minimum, spec.maximum) == (None, None)


def test_float_fields_declare_bounds():
    for spec in unit_fields.FIELDS:
        if spec.kind == unit_fields.FLOAT:
            assert spec.minimum is not None
            assert spec.maximum is not None
            assert spec.minimum < spec.maximum


def test_owner_field_is_the_player_kind():
    spec = unit_fields.FIELDS_BY_ID["player"]
    assert spec.kind == unit_fields.PLAYER
