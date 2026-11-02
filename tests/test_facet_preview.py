"""GH #167: the facet-to-direction mapping behind the trigger panel's preview.

The mapping functions use no Qt, but they share a module with the widget.
"""

from __future__ import annotations

import math

import pytest

import conftest

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


@pytest.mark.parametrize("facet", [-1, -3])
def test_a_negative_facet_has_no_direction(facet: int) -> None:
    from descape.facet_preview import facet_rotation

    assert facet_rotation(facet) is None


@pytest.mark.parametrize("facet", range(16))
def test_facet_k_is_k_sixteenths_of_a_turn(facet: int) -> None:
    from descape.facet_preview import facet_rotation

    assert facet_rotation(facet) == pytest.approx(facet * math.pi / 8)


@pytest.mark.parametrize(("facet", "same_as"), [(16, 0), (20, 4), (31, 15)])
def test_a_facet_past_sixteen_wraps(facet: int, same_as: int) -> None:
    from descape.facet_preview import facet_rotation

    assert facet_rotation(facet) == pytest.approx(facet_rotation(same_as))


def test_rotation_zero_points_up_right_and_a_quarter_turn_down_right() -> None:
    """World +x is screen up-right and +y down-right (tile_screen_origin())."""
    from descape import iso_geometry
    from descape.facet_preview import TILE_PX, arrow_vector

    half_w, half_h = iso_geometry.half_dims(TILE_PX)
    dx, dy = arrow_vector(0.0)
    assert dx > 0 and dy < 0
    assert (dx, dy) == pytest.approx((half_w, -half_h))
    dx, dy = arrow_vector(math.pi / 2)
    assert dx > 0 and dy > 0
    assert (dx, dy) == pytest.approx((half_w, half_h))


@pytest.mark.parametrize("facet", range(16))
def test_the_arrow_agrees_with_the_sprite_frame_it_selects(facet: int) -> None:
    """The arrow and the frame the sprite path draws for the same rotation
    point the same way, up to the iso squash. Frame i of n faces i*360/n
    clockwise from screen-right (unit_sprites' knight measurement), and the
    tolerance is the squash residual at the diagonals, derived from the
    tile's own aspect, not typed in."""
    from descape import iso_geometry, unit_sprites
    from descape.facet_preview import FACET_STEPS, TILE_PX, arrow_vector, facet_rotation, screen_angle_deg

    rotation = facet_rotation(facet)
    frame = unit_sprites.angle_index(rotation, FACET_STEPS, unit_sprites.ANGLE_ZERO_OFFSET_DEG)
    frame_deg = frame * 360.0 / FACET_STEPS
    arrow_deg = screen_angle_deg(*arrow_vector(rotation))
    half_w, half_h = iso_geometry.half_dims(TILE_PX)
    residual = 45.0 - math.degrees(math.atan2(half_h, half_w))
    assert residual == pytest.approx(18.43, abs=0.01)
    gap = abs((arrow_deg - frame_deg + 180.0) % 360.0 - 180.0)
    assert gap <= residual + 1e-9, f"facet {facet}: arrow {arrow_deg:.2f}, frame {frame} at {frame_deg:.2f}"


# -- GH #167: the one facing rule the panel and the map's ghosts share ----------------

_ARCHER = 4
_CASTLE = 82
_UNKNOWN = 4242


def test_unit_facet_rotation_has_none_for_an_unset_facet_or_a_const_with_no_facing() -> None:
    from descape.facet_preview import facet_rotation, unit_facet_rotation

    assert unit_facet_rotation(_ARCHER, -1) is None
    assert unit_facet_rotation(_CASTLE, 4) is None, "a known const with no facing never takes a facet"
    assert unit_facet_rotation(_ARCHER, 4) == pytest.approx(facet_rotation(4))
    assert unit_facet_rotation(_UNKNOWN, 4) == pytest.approx(facet_rotation(4)), "as the panel's arrow does today"
    assert unit_facet_rotation(-1, 6) == pytest.approx(facet_rotation(6)), "an unset or differing unit keeps the arrow"


def test_the_ghost_draws_the_default_facing_until_the_mapping_is_measured(monkeypatch) -> None:
    from descape import facet_preview

    assert facet_preview.GHOST_FACING_MEASURED is False, "TASK-194's Phase D turns this on"
    for const, facet in ((_ARCHER, 4), (_ARCHER, -1), (_CASTLE, 2), (_UNKNOWN, 9)):
        assert facet_preview.ghost_rotation(const, facet) == 0.0, (const, facet)

    monkeypatch.setattr(facet_preview, "GHOST_FACING_MEASURED", True)
    assert facet_preview.ghost_rotation(_ARCHER, 4) == pytest.approx(facet_preview.facet_rotation(4))
    assert facet_preview.ghost_rotation(_UNKNOWN, 9) == pytest.approx(facet_preview.facet_rotation(9))
    assert facet_preview.ghost_rotation(_ARCHER, -1) == 0.0
    assert facet_preview.ghost_rotation(_CASTLE, 2) == 0.0


def test_the_arrow_and_the_sprite_take_unit_facet_rotations_value(monkeypatch) -> None:
    """A stand-in value no facet maps to: a preview still reading
    facet_rotation() directly would show 4/16 of a turn instead."""
    from descape import constant_picker, facet_preview

    conftest.ensure_qapp()
    asked = []
    monkeypatch.setattr(facet_preview, "unit_facet_rotation", lambda unit, facet: 1.25)
    monkeypatch.setattr(constant_picker, "preview_pixmap", lambda unit, rotation: asked.append((unit, rotation)))
    entry = {"facet": 4, "object_list_unit_id": _ARCHER}
    preview = facet_preview.FacetPreview()
    preview.bind(
        {"facet": None, "object_list_unit_id": None}, "facet", "object_list_unit_id", lambda e, name: e[name], entry=entry
    )
    assert preview.rotation == 1.25
    assert preview.tile.rotation == 1.25 and preview.arrow_shown
    assert asked == [(_ARCHER, 1.25)]
