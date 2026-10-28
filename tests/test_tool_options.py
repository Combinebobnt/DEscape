"""TASK-272: descape/tool_options.py, the Tool Options row's schema. Qt-free.

normalize() never coerces, so a hand-edited config.yaml can't smuggle a
string, a YAML `true` or a float into a control; the mirror tests pin the
schema's literal member lists and the wall default to the modules they copy.
"""

from __future__ import annotations

import math

import pytest

from descape import brush, eraser, iso_geometry, tool_options, unit_filter, unit_sprites
from descape.tool_options import BOOL, CHOICE, ID, INT, SET


def test_names_are_unique_and_every_group_is_known() -> None:
    names = [spec.name for spec in tool_options.OPTIONS]
    assert len(names) == len(set(names))
    group_ids = [group_id for group_id, _title in tool_options.GROUPS]
    assert len(group_ids) == len(set(group_ids))
    assert {spec.group for spec in tool_options.OPTIONS} == set(group_ids), "a group with no rows, or a row with no group"
    assert {spec.kind for spec in tool_options.OPTIONS} <= set(tool_options.KINDS)


def test_every_default_survives_its_own_normalize() -> None:
    bad = []
    for spec in tool_options.OPTIONS:
        try:
            if tool_options.normalize(spec.name, spec.default) != spec.default:
                bad.append(spec.name)
        except ValueError as e:
            bad.append(f"{spec.name}: {e}")
    assert bad == []


def test_the_cliff_trio_restores_in_chain_order() -> None:
    names = [spec.name for spec in tool_options.OPTIONS]
    assert names.index("cliff_family") < names.index("cliff_piece") < names.index("cliff_frame")


# Inputs no kind accepts: the traps a YAML file can hold.
_NEVER = [None, "1", "true", "false", 1.0, math.nan, {}, {"a": 1}, [], b"x"]


@pytest.mark.parametrize("raw", [*_NEVER, 1, 0])
def test_a_bool_takes_only_a_real_bool(raw) -> None:
    with pytest.raises(ValueError):
        tool_options.normalize("paint_trees", raw)
    assert tool_options.normalize("paint_trees", False) is False


@pytest.mark.parametrize("raw", [*_NEVER, True, False, 0, 10, -1])
def test_an_int_takes_only_a_real_int_in_range(raw) -> None:
    with pytest.raises(ValueError):
        tool_options.normalize("brush_size", raw)
    assert tool_options.normalize("brush_size", 9) == 9


def test_an_unbounded_int_has_only_its_minimum() -> None:
    assert tool_options.normalize("cliff_frame", 10_000) == 10_000
    with pytest.raises(ValueError):
        tool_options.normalize("cliff_frame", -1)


@pytest.mark.parametrize("raw", [*_NEVER, "Square", 0, True])
def test_a_choice_takes_only_a_member(raw) -> None:
    with pytest.raises(ValueError):
        tool_options.normalize("brush_shape", raw)
    assert tool_options.normalize("brush_shape", "plus") == "plus"


def test_an_id_takes_its_type_and_none_only_where_nullable() -> None:
    assert tool_options.normalize("beach_terrain", None) is None
    assert tool_options.normalize("beach_terrain", 2) == 2
    assert tool_options.normalize("cliff_family", "a_cliff.sld") == "a_cliff.sld"
    for name, raw in [("last_wall", None), ("last_wall", True), ("last_wall", "117"), ("last_wall", 117.0),
                      ("cliff_family", 5), ("cliff_piece", "264"), ("beach_terrain", False)]:
        with pytest.raises(ValueError):
            tool_options.normalize(name, raw)


def test_a_set_is_a_list_in_member_order_and_never_takes_a_bool_or_a_set() -> None:
    assert tool_options.normalize("eraser_targets", ("trees", "units", "trees")) == ["units", "trees"]
    assert tool_options.normalize("convert_sources", []) == []
    for raw in [{"units"}, "units", None, 3, ["Units"]]:
        with pytest.raises(ValueError):
            tool_options.normalize("eraser_targets", raw)
    # True == 1 and True in range(9) are both true in Python, so [True] is the trap.
    for raw in [[True], [1.0], ["1"]]:
        with pytest.raises(ValueError):
            tool_options.normalize("convert_sources", raw)


def test_lenient_drops_a_sets_illegal_members_and_strict_raises() -> None:
    raw = ["units", "bogus", True, "cliffs"]
    with pytest.raises(ValueError):
        tool_options.normalize("eraser_targets", raw)
    assert tool_options.normalize("eraser_targets", raw, lenient=True) == ["units", "cliffs"]
    with pytest.raises(ValueError):  # lenient still needs a list
        tool_options.normalize("eraser_targets", "units", lenient=True)


def test_an_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        tool_options.normalize("no_such_option", 1)
    with pytest.raises(ValueError):
        tool_options.default("no_such_option")


def test_prune_drops_defaults_and_unknown_names_and_keeps_the_rest() -> None:
    assert tool_options.prune_to_overrides(
        {
            "brush_size": 1,
            "brush_shape": "circle",
            "no_such_option": 4,
            "eraser_targets": ["buildings", "units"],
            "convert_sources": [0, 1],
        }
    ) == {"brush_shape": "circle", "convert_sources": [0, 1]}


def test_a_set_default_is_a_fresh_list_every_time() -> None:
    first = tool_options.default("eraser_targets")
    first.append("trees")
    assert tool_options.default("eraser_targets") == ["units", "buildings"]


def test_kind_specific_fields_are_set_only_on_their_kind() -> None:
    for spec in tool_options.OPTIONS:
        assert (spec.minimum is not None) == (spec.kind == INT), spec.name
        assert bool(spec.choices) == (spec.kind == CHOICE), spec.name
        assert bool(spec.members) == (spec.kind == SET), spec.name
        assert (spec.id_type is not None) == (spec.kind == ID), spec.name
        assert bool(spec.note) == (spec.kind == ID), spec.name
        if spec.kind == BOOL:
            assert isinstance(spec.default, bool), spec.name


# -- mirrors: the literals here against the modules they copy -----------------


def test_eraser_members_and_defaults_mirror_eraser() -> None:
    assert tuple(name for name, _label in eraser.TARGET_LABELS) == tool_options.ERASER_TARGETS
    defaults = eraser.EraseTargets()
    assert tool_options.default("eraser_targets") == [n for n in tool_options.ERASER_TARGETS if getattr(defaults, n)]


def test_convert_members_are_the_player_id_range() -> None:
    assert tuple(range(unit_filter.GAIA_PLAYER_ID, unit_filter.MAX_PLAYER_ID + 1)) == tool_options.CONVERT_SOURCES


def test_the_wall_default_is_a_wall() -> None:
    assert tool_options.default("last_wall") in unit_sprites.wall_family_consts()


def test_ranges_mirror_their_sources() -> None:
    assert tool_options.option("elevation_level").maximum == iso_geometry.MAX_ELEVATION
    size = tool_options.option("brush_size")
    assert (size.minimum, size.maximum) == (brush.BRUSH_SIZE_MIN, brush.BRUSH_SIZE_MAX)
    thickness = tool_options.option("brush_thickness")
    assert (thickness.minimum, thickness.maximum) == (brush.BRUSH_THICKNESS_MIN, brush.BRUSH_THICKNESS_MAX)
    assert tool_options.option("brush_shape").choices == brush.BRUSH_SHAPES
