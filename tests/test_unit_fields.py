"""Covers descape/unit_fields.py: the Qt-free field-spec module for the
Units-mode inspector (descape-units-edit-ui.md's 1.2/D5).

Static-tuple module, so these tests pin its declared shape rather than any
computed presence-gate logic -- there is none here, unlike option_fields.py.
"""

from __future__ import annotations

from descape import unit_fields


def test_every_field_has_a_unique_id():
    ids = [spec.field_id for spec in unit_fields.FIELDS]
    assert len(ids) == len(set(ids))


def test_fields_by_id_mirrors_fields():
    assert unit_fields.FIELDS_BY_ID == {spec.field_id: spec for spec in unit_fields.FIELDS}


def test_only_xyz_and_owner_are_editable():
    """D5: X, Y, Z and Owner get real editors; everything else (Name, Unit
    ID, Rotation, Reference ID, Garrisoned in) stays a read-only label."""
    editable = {spec.field_id for spec in unit_fields.FIELDS if spec.editable}
    assert editable == {"x", "y", "z", "player"}


def test_rotation_is_never_editable_and_carries_no_scale():
    """AGENTS.md's hard rule: rotation is a raw shape-variant index for most
    GAIA objects, never an angle -- so it must never get a FLOAT (degree-
    formattable) editor."""
    spec = unit_fields.FIELDS_BY_ID["rotation"]
    assert spec.editable is False
    assert spec.kind == unit_fields.TEXT


def test_float_fields_declare_bounds():
    for spec in unit_fields.FIELDS:
        if spec.kind == unit_fields.FLOAT:
            assert spec.minimum is not None
            assert spec.maximum is not None
            assert spec.minimum < spec.maximum


def test_owner_field_is_the_player_kind():
    spec = unit_fields.FIELDS_BY_ID["player"]
    assert spec.kind == unit_fields.PLAYER
