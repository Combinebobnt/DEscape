"""TASK-023: the facing dial's geometry, and the viewer keeping its projection.

The geometry functions use no Qt, but they share a module with the widget.
"""

from __future__ import annotations

import math

import pytest

import conftest

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")

# iso_geometry's 2:1 diamond: world +x reads atan(1/2) above screen-right, +y the same below.
_ISO_SLOPE_DEG = math.degrees(math.atan2(1, 2))
_COUNTS = [2, 4, 6, 8, 16, 32, 72]


def _angle(vector) -> float:
    from descape.facet_preview import screen_angle_deg

    return screen_angle_deg(*vector)


def _close_deg(a: float, b: float, period: float = 360.0) -> bool:
    diff = (a - b) % period
    return min(diff, period - diff) < 1e-6


def test_iso_rotation_zero_points_up_right_and_a_quarter_turn_down_right() -> None:
    from descape.facing_dial import screen_vector

    assert _close_deg(_angle(screen_vector(0.0, False)), 360 - _ISO_SLOPE_DEG)
    assert _close_deg(_angle(screen_vector(math.pi / 2, False)), _ISO_SLOPE_DEG)
    assert round(360 - _ISO_SLOPE_DEG, 2) == 333.43


def test_flat_follows_the_measured_flat_zero_point() -> None:
    from descape import unit_sprites
    from descape.facing_dial import screen_vector

    offset = unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG
    assert _close_deg(_angle(screen_vector(0.0, True)), offset)
    assert _close_deg(_angle(screen_vector(math.pi / 2, True)), 90.0 + offset)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("rotation", [0.0, 0.3, 1.0, 2.5, 4.0, 5.9])
def test_screen_vector_is_unit_length_and_rotation_at_inverts_it(flat: bool, rotation: float) -> None:
    from descape.facing_dial import rotation_at, screen_vector

    dx, dy = screen_vector(rotation, flat)
    assert math.hypot(dx, dy) == pytest.approx(1.0)
    assert _close_deg(math.degrees(rotation_at(dx, dy, flat)), math.degrees(rotation))
    # Scale-free: a click further out on the same spoke reads the same.
    assert _close_deg(math.degrees(rotation_at(40 * dx, 40 * dy, flat)), math.degrees(rotation))


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("count", _COUNTS)
def test_facing_at_round_trips_every_facing(flat: bool, count: int) -> None:
    from descape import unit_rotation
    from descape.facing_dial import facing_at, ring_point, screen_vector

    for facing in range(count):
        rotation = unit_rotation.facing_to_rotation(facing, count)
        assert facing_at(*screen_vector(rotation, flat), count, flat) == facing
        assert facing_at(*ring_point(rotation, flat), count, flat) == facing


@pytest.mark.parametrize("flat", [False, True])
def test_the_next_facing_is_clockwise_on_screen(flat: bool) -> None:
    """Wheel up is +1, and AoE2's rotation grows clockwise: the dial must agree."""
    from descape import unit_rotation
    from descape.facing_dial import screen_vector

    for count in _COUNTS[2:]:
        for facing in range(count):
            here = _angle(screen_vector(unit_rotation.facing_to_rotation(facing, count), flat))
            ahead = _angle(screen_vector(unit_rotation.facing_to_rotation(facing + 1, count), flat))
            assert 0 < (ahead - here) % 360 < 180


def test_the_iso_ring_has_the_2_to_1_squash_and_flat_is_a_circle() -> None:
    from descape.facing_dial import ring_point

    # World 45 degrees is screen-right in iso, world -45 straight up.
    assert ring_point(math.pi / 4, False) == pytest.approx((1.0, 0.0))
    assert ring_point(-math.pi / 4, False) == pytest.approx((0.0, -0.5))
    for rotation in (0.0, 0.7, 2.0, 4.4):
        assert math.hypot(*ring_point(rotation, True)) == pytest.approx(1.0)


# The screen axis each gate name runs along in iso, mod 180, clockwise from screen-right.
_GATE_NAME_AXES = {
    "SW to NE": 180 - _ISO_SLOPE_DEG,
    "W to E": 0.0,
    "NW to SE": _ISO_SLOPE_DEG,
    "N to S": 90.0,
}


def test_gate_axes_land_on_their_names_screen_directions_in_iso() -> None:
    from descape.facing_dial import gate_rotation, ring_point
    from descape.units_panel import _GATE_ORIENTATION_NAMES

    for index, name in enumerate(_GATE_ORIENTATION_NAMES):
        assert _close_deg(_angle(ring_point(gate_rotation(index), False)), _GATE_NAME_AXES[name], 180.0), name


@pytest.mark.parametrize("flat", [False, True])
def test_either_end_of_a_gate_axis_picks_the_same_orientation(flat: bool) -> None:
    from descape.facing_dial import GATE_AXES, gate_at, gate_rotation, screen_vector

    for index in range(GATE_AXES):
        dx, dy = screen_vector(gate_rotation(index), flat)
        assert gate_at(dx, dy, flat) == index
        assert gate_at(-dx, -dy, flat) == index


def test_the_gate_step_is_stated_once_in_gate_orientation() -> None:
    from descape import gate_orientation
    from descape.facing_dial import gate_rotation

    assert gate_rotation(1) == pytest.approx(math.radians(gate_orientation.RUN_ANGLE_STEP_DEG))


@pytest.mark.parametrize(("count", "ticks"), [(2, 2), (8, 8), (16, 16), (24, 12), (32, 16), (72, 8)])
def test_ticks_thin_out_with_the_direction_count(count: int, ticks: int) -> None:
    from descape.facing_dial import tick_rotations

    assert len(tick_rotations(count)) == ticks


# --- the viewer keeps the dial on the map's projection ---------------------


@pytest.mark.gui
def test_the_dial_follows_elevation_view_and_isometric_view() -> None:
    window = conftest.blank_window()
    try:
        dial = window.units_panel.unit_rotation_dial
        window.terrain_style_combo.setCurrentText("Stepped")
        assert window._render_style == "stepped"
        assert dial.flat is False
        window.terrain_style_combo.setCurrentText("Flat")
        window.iso_action.setChecked(False)
        assert window._render_style == "flat"
        assert dial.flat is True
        window.iso_action.setChecked(True)
        assert window._render_style == "stepped"
        assert dial.flat is False
        window.iso_action.setChecked(False)
        assert dial.flat is True
        window.terrain_style_combo.setCurrentText("Sloped")
        assert dial.flat is False
    finally:
        conftest.close_window(window)
