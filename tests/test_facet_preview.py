"""GH #167: the facet-to-direction mapping behind the trigger panel's preview.

The mapping functions use no Qt, but they share a module with the widget.
"""

from __future__ import annotations

import math

import pytest

import conftest

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


# Read in AoE2:DE on 2026-11-02 from the GH #167 tester's screenshots of the
# probe scenario: (facet, the created unit's angle_count, the stored index it faced).
# Archers 16, Trebuchet 32, Dummy Target 8. Negatives faced at random, so None.
_MEASURED = (
    [(k, 16, k) for k in range(16)]
    + [(16, 16, 0), (17, 16, 0), (20, 16, 0), (35, 16, 0)]
    + [(16, 32, 16), (24, 32, 24), (31, 32, 31)]
    + [(8, 8, 0)]
    + [(-1, 16, None), (-2, 16, None), (-3, 16, None)]
)


def test_a_facet_is_the_measured_stored_angle_index() -> None:
    from descape.facet_preview import facet_angle_index

    bad = [(f, n, want, got) for f, n, want in _MEASURED if (got := facet_angle_index(f, n)) != want]
    assert not bad, bad


@pytest.mark.parametrize(("facet", "angle_count", "index"), [row for row in _MEASURED if row[2] is not None])
def test_facet_rotation_draws_the_measured_index(facet: int, angle_count: int, index: int) -> None:
    from descape import unit_sprites
    from descape.facet_preview import facet_rotation

    assert unit_sprites.angle_index(facet_rotation(facet, angle_count), angle_count) == index


def test_the_testers_cavalier_at_facet_two_faces_down_right_along_the_tile_edge() -> None:
    """GH #167's report: Cavalier (16 angles), facet 2. The game showed stored
    index 2; the first preview drew index 0, pointing screen-right."""
    from descape import iso_geometry, unit_sprites
    from descape.facet_preview import TILE_PX, arrow_vector, facet_rotation, screen_angle_deg, unit_facet_rotation

    rotation = unit_facet_rotation(_CAVALIER, 2)
    assert rotation == pytest.approx(facet_rotation(2, 16))
    assert unit_sprites.angle_index(rotation, 16) == 2
    half_w, half_h = iso_geometry.half_dims(TILE_PX)
    edge = math.degrees(math.atan2(half_h, half_w))
    assert edge == pytest.approx(26.57, abs=0.01)
    assert screen_angle_deg(*arrow_vector(rotation)) == pytest.approx(edge)


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


@pytest.mark.parametrize("angle_count", sorted({n for _f, n, _i in _MEASURED}))
def test_the_arrow_agrees_with_the_sprite_frame_it_selects(angle_count: int) -> None:
    """The arrow and the frame the sprite path draws for the same rotation
    point the same way, up to the iso squash. Frame i of n faces i*360/n
    clockwise from screen-right (unit_sprites' knight measurement), and the
    tolerance is the squash's largest residual, derived from the tile's own
    aspect, not typed in: a vertical squash s turns a direction t off the
    major axis to atan(s tan t), furthest out at tan t = 1/sqrt(s). That is
    19.47 degrees at 2:1, past the 18.43 of the diagonals, which 32 angles
    sample close to. Every facet of each measured count."""
    from descape import iso_geometry, unit_sprites
    from descape.facet_preview import TILE_PX, arrow_vector, facet_rotation, screen_angle_deg

    half_w, half_h = iso_geometry.half_dims(TILE_PX)
    squash = half_h / half_w
    residual = math.degrees(math.atan(1 / math.sqrt(squash)) - math.atan(math.sqrt(squash)))
    assert residual == pytest.approx(19.47, abs=0.01)
    bad = []
    for facet in range(angle_count):
        rotation = facet_rotation(facet, angle_count)
        frame = unit_sprites.angle_index(rotation, angle_count, unit_sprites.ANGLE_ZERO_OFFSET_DEG)
        frame_deg = frame * 360.0 / angle_count
        arrow_deg = screen_angle_deg(*arrow_vector(rotation))
        gap = abs((arrow_deg - frame_deg + 180.0) % 360.0 - 180.0)
        if gap > residual + 1e-9:
            bad.append((facet, round(arrow_deg, 2), frame, frame_deg))
    assert not bad, bad


# -- GH #167: the one facing rule the panel and the map's ghosts share ----------------

_ARCHER = 4
_CAVALIER = 283
_TREBUCHET = 42
_DUMMY_TARGET = 1956
_CASTLE = 82
_UNKNOWN = 4242


def test_unit_facet_rotation_has_none_for_an_unset_facet_or_a_const_with_no_facing() -> None:
    from descape.facet_preview import facet_rotation, unit_facet_rotation

    assert unit_facet_rotation(_ARCHER, -1) is None
    assert unit_facet_rotation(_ARCHER, -2) is None, "every negative is Unset"
    assert unit_facet_rotation(_CASTLE, 4) is None, "a known const with no facing never takes a facet"
    assert unit_facet_rotation(_ARCHER, 4) == pytest.approx(facet_rotation(4))
    assert unit_facet_rotation(_UNKNOWN, 4) == pytest.approx(facet_rotation(4)), "an unknown const counts in 16"
    assert unit_facet_rotation(-1, 6) == pytest.approx(facet_rotation(6)), "an unset or differing unit keeps the arrow"


@pytest.mark.parametrize(("unit", "facet", "index"), [(_TREBUCHET, 24, 24), (_TREBUCHET, 31, 31), (_DUMMY_TARGET, 8, 0), (_ARCHER, 20, 0)])
def test_a_known_unit_counts_facets_in_its_own_angles(unit: int, facet: int, index: int) -> None:
    """Q2 of the measurement: a Trebuchet's 24 is 24 of 32, not 8 of 16."""
    from descape import unit_rotation, unit_sprites
    from descape.facet_preview import unit_facet_rotation

    count = unit_rotation.angle_count_for(unit)
    assert unit_sprites.angle_index(unit_facet_rotation(unit, facet), count) == index


def test_the_ghost_takes_the_facet_facing_and_the_constant_still_switches_it_off(monkeypatch) -> None:
    from descape import facet_preview

    assert facet_preview.GHOST_FACING_MEASURED is True, "measured 2026-11-02 (TASK-194)"
    monkeypatch.setattr(facet_preview, "GHOST_FACING_MEASURED", False)
    for const, facet in ((_ARCHER, 4), (_ARCHER, -1), (_CASTLE, 2), (_UNKNOWN, 9)):
        assert facet_preview.ghost_rotation(const, facet) == 0.0, (const, facet)

    monkeypatch.setattr(facet_preview, "GHOST_FACING_MEASURED", True)
    assert facet_preview.ghost_rotation(_ARCHER, 4) == pytest.approx(facet_preview.facet_rotation(4))
    assert facet_preview.ghost_rotation(_TREBUCHET, 24) == pytest.approx(facet_preview.facet_rotation(24, 32))
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
