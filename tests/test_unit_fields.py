"""Covers descape/unit_fields.py: the Qt-free field-spec module for the
Units-mode inspector.

Static-tuple module, so these tests pin its declared shape rather than any
computed presence-gate logic -- there is none here, unlike option_fields.py.
"""

from __future__ import annotations

import pytest

from descape import unit_fields


def test_every_field_has_a_unique_id():
    ids = [spec.field_id for spec in unit_fields.FIELDS]
    assert len(ids) == len(set(ids))


def test_fields_by_id_mirrors_fields():
    assert {spec.field_id: spec for spec in unit_fields.FIELDS} == unit_fields.FIELDS_BY_ID


def test_only_xyz_owner_rotation_and_the_two_choices_are_editable():
    """D5: X, Y, Z and Owner get real editors, plus Rotation as of b3 (which
    is additionally per-const gated) and Unit State and Capture Flag as of
    GH #186. Everything else (Name, Unit ID, Reference ID, Garrisoned in)
    stays a read-only label."""
    editable = {spec.field_id for spec in unit_fields.FIELDS if spec.editable}
    assert editable == {"x", "y", "z", "player", "rotation", "status", "capture_flag"}


def test_capture_flag_and_unit_state_sit_between_rotation_and_reference_id():
    """GH #186: the in-game editor's own order, Capture Flag first."""
    ids = [spec.field_id for spec in unit_fields.FIELDS]
    start = ids.index("rotation")
    assert ids[start : start + 4] == ["rotation", "capture_flag", "status", "reference_id"]
    for field_id, label in (("capture_flag", "Capture Flag"), ("status", "Unit State")):
        spec = unit_fields.FIELDS_BY_ID[field_id]
        assert (spec.kind, spec.label, spec.editable) == (unit_fields.CHOICE, label, True)


def test_the_choice_values_are_the_parser_enums():
    """The raw values are the pinned library's own enums, so a pin bump that
    renumbers, adds or drops a member fails here rather than mislabelling."""
    from AoE2ScenarioParser.datasets.trigger_lists import CaptureFlag, ObjectState

    assert {value for value, _ in unit_fields.CAPTURE_FLAG_CHOICES} == {int(member) for member in CaptureFlag}
    assert {value for value, _ in unit_fields.UNIT_STATE_CHOICES} == {int(member) for member in ObjectState}
    assert unit_fields.FIELDS_BY_ID["capture_flag"].choices == unit_fields.CAPTURE_FLAG_CHOICES
    assert unit_fields.FIELDS_BY_ID["status"].choices == unit_fields.UNIT_STATE_CHOICES


def test_the_choice_labels_are_the_games_own_and_distinct():
    assert dict(unit_fields.CAPTURE_FLAG_CHOICES) == {
        -1: "Default",
        0: "Convert Never",
        1: "Convert Once",
        2: "Convert Multiple Times",
        3: "Convert Never (Gaia Aggressive)",
    }
    assert dict(unit_fields.UNIT_STATE_CHOICES) == {
        0: "Foundation",
        1: "Almost Alive",
        2: "Alive",
        3: "Dead",
        4: "Almost Dead",
        5: "Really Dead",
        6: "Undead",
        7: "Remove",
        8: "Going",
    }


def test_only_capture_flag_is_gated_per_file():
    """Unit State's byte is in every version; Capture Flag is stored from
    1.59. The gate is its own attribute, not a `conditional` (per const)."""
    gated = {spec.field_id: spec.requires_codec_field for spec in unit_fields.FIELDS if spec.requires_codec_field}
    assert gated == {"capture_flag": "capture_flag"}
    assert not [spec.field_id for spec in unit_fields.FIELDS if spec.choices and spec.kind != unit_fields.CHOICE]


def test_rotation_is_the_only_conditional_field():
    """AGENTS.md's hard rule survives as a conditional, not a blanket ban:
    rotation is a shape-variant index for most GAIA objects and every wall, so
    its editor is gated per-const rather than always live. GH #123 widened the
    rule from ANGLE-only to ANGLE or cyclable."""
    conditional = {spec.field_id for spec in unit_fields.FIELDS if spec.conditional}
    assert conditional == {"rotation"}
    assert unit_fields.FIELDS_BY_ID["rotation"].conditional == "rotation_is_editable"


def test_rotation_is_edited_as_a_facing_with_a_per_const_range():
    """GH #61: a whole facing, not raw radians. The range is the const's own
    angle_count, so the spec carries no bounds of its own."""
    spec = unit_fields.FIELDS_BY_ID["rotation"]
    assert spec.kind == unit_fields.FACING
    assert (spec.minimum, spec.maximum) == (None, None)


def test_float_fields_declare_bounds():
    """The static bounds are the no-scenario fallback only (GH #118): with a
    map loaded the panel applies coordinate_bounds() instead."""
    for spec in unit_fields.FIELDS:
        if spec.kind == unit_fields.FLOAT:
            assert spec.minimum is not None
            assert spec.maximum is not None
            assert spec.minimum < spec.maximum


# --- GH #118: map-derived coordinate bounds ------------------------------------


def test_coordinate_bounds_use_width_for_x_and_height_for_y():
    assert unit_fields.coordinate_bounds("x", 120, 168) == pytest.approx((0.0, 119.99))
    assert unit_fields.coordinate_bounds("y", 120, 168) == pytest.approx((0.0, 167.99))


def test_coordinate_upper_bound_stays_below_the_map_size():
    """render.unit_tile_bounds is half-open, so x == size is off the map."""
    for field_id in ("x", "y"):
        _lo, hi = unit_fields.coordinate_bounds(field_id, 240, 240)
        assert hi < 240
        assert round(hi, 2) == 239.99


def test_z_bounds_are_the_elevation_cap_on_any_map():
    for size in ((120, 120), (240, 144), (480, 480)):
        assert unit_fields.coordinate_bounds("z", *size) == (0.0, 15.0)


def test_widen_extends_just_far_enough_to_include_the_values():
    assert unit_fields.widen((0.0, 119.99), [999.0]) == (0.0, 999.0)
    assert unit_fields.widen((0.0, 119.99), [-5.0, 40.0]) == (-5.0, 119.99)
    assert unit_fields.widen((0.0, 119.99), [4.5, 8.5]) == (0.0, 119.99)


def test_widen_with_no_values_is_the_bounds_themselves():
    assert unit_fields.widen((0.0, 15.0), []) == (0.0, 15.0)


def test_owner_field_is_the_player_kind():
    spec = unit_fields.FIELDS_BY_ID["player"]
    assert spec.kind == unit_fields.PLAYER
