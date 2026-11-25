"""The facing constants, pinned against AoE2:DE's own reading and derivation.

**What was measured.** A GH #167 tester played a scenario holding control
units of three graphics at known rotations and posted in-game screenshots on
2026-11-02. Each control's stored `rotation` is j * 2*pi / angle_count, taken
from the scenario file as written and played, and cross-checked against the
layout that generated it (every one a whole grid step, the two sources in
agreement to float32 precision). It was never recomputed from today's
angle_index(), which would only describe the code under test. Each screenshot
was then read by masked template matching against the raw .sld frames, one
template per stored frame. So each row below is the game's own answer to
"rotation r draws stored frame k": had it drawn frame m, the match would have
picked template m. Archer (const 4, 16 angles): 16 of 16 controls, weakest
margin 9.24, shot gate accepted. Dummy Target (const 1956, 8 angles): 8 of 8,
weakest 4.98, gate accepted. Trebuchet (const 42, 32 angles): its gate
rejected the shot only because one control (index 16) lay outside the frame,
so the 4 controls inside it are used on their own margins, 29.76 or more. A wrong ANGLE_DIRECTION would read archer k as (-k - 4) mod 16,
a +45 offset as k + 4 and a 180 degree error as k + 8, at three angle counts.

**Not covered:** an angle_count that is not a multiple of 8. The one such shot
(Gunpowder Keg, 6 angles) failed its gate on near-symmetric art, so the
whole-step guard in angle_index() keeps no game reading.

**Flat's offset is derived, not measured on its own.** Flat draws the iso
art, so its zero point is the iso one plus the screen turn from iso to Flat
(45 degrees, from iso_geometry's projection) times the sense in which the
stored index walks on screen. tools/verify_flat_facing.py's index walk sense
arm measures that sense on real art; INDEX_WALK_SENSE below is its result.

**Mutation arm, run by hand and confirmed 2026-11-25, not asserted:** setting
unit_sprites.ANGLE_DIRECTION to -1, or ANGLE_ZERO_OFFSET_DEG to +45.0 or
135.0, turns test_angle_index_matches_the_in_game_reading red, and setting
FLAT_ANGLE_ZERO_OFFSET_DEG to 180.0 or -90.0 turns
test_flat_offset_is_the_iso_offset_turned_into_flat red.
"""

from __future__ import annotations

import math

import pytest

from descape import iso_geometry, unit_sprites

# (angle_count, j, the stored frame the game drew), rotation = j * 2*pi / angle_count.
_ARCHERS = [
    (16, 2, 0), (16, 3, 1), (16, 4, 2), (16, 5, 3), (16, 6, 4), (16, 7, 5), (16, 8, 6), (16, 9, 7),
    (16, 10, 8), (16, 11, 9), (16, 12, 10), (16, 13, 11), (16, 14, 12), (16, 15, 13), (16, 0, 14), (16, 1, 15),
]
_DUMMIES = [(8, 1, 0), (8, 2, 1), (8, 3, 2), (8, 4, 3), (8, 5, 4), (8, 6, 5), (8, 7, 6), (8, 0, 7)]
_TREBUCHETS = [(32, 4, 0), (32, 8, 4), (32, 12, 8), (32, 28, 24)]
_READINGS = _ARCHERS + _DUMMIES + _TREBUCHETS

# The galley's stored index walks clockwise on screen (y down), measured by
# tools/verify_flat_facing.py's index walk sense arm (smallest step 12.41
# degrees, all 16 the same sign, 2026-11-25).
INDEX_WALK_SENSE = 1


def test_the_table_is_the_28_readings():
    assert len(_READINGS) == 28
    assert len(set(_READINGS)) == 28


@pytest.mark.parametrize(("angle_count", "j", "observed"), _READINGS)
def test_angle_index_matches_the_in_game_reading(angle_count, j, observed):
    rotation = j * 2 * math.pi / angle_count
    assert unit_sprites.angle_index(rotation, angle_count) == observed


def _unsquashed_iso_angle_deg(dx: int, dy: int) -> float:
    """World step (dx, dy)'s signed screen angle under the iso projection,
    clockwise from screen-right with y down, the 2:1 squash undone."""
    proj = iso_geometry.canvas_size_and_origin(
        8, 8, 64, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    )
    x0, y0 = iso_geometry.tile_screen_origin(2, 2, 0, proj)
    x1, y1 = iso_geometry.tile_screen_origin(2 + dx, 2 + dy, 0, proj)
    sx, sy = x1 - x0, (y1 - y0) * proj.half_w / proj.half_h
    return math.degrees(math.atan2(sy, sx))


def iso_to_flat_turn_deg() -> float:
    """Flat's screen angle of world +x (0) minus its unsquashed iso angle."""
    return 0.0 - _unsquashed_iso_angle_deg(1, 0)


def test_the_iso_to_flat_turn_is_a_rotation_of_45_degrees():
    assert _unsquashed_iso_angle_deg(1, 0) == pytest.approx(-45.0)
    assert _unsquashed_iso_angle_deg(0, 1) == pytest.approx(45.0)
    # World +y sits at 90 under Flat's grid, so it turns by the same amount as +x.
    assert 90.0 - _unsquashed_iso_angle_deg(0, 1) == pytest.approx(iso_to_flat_turn_deg())
    assert iso_to_flat_turn_deg() == pytest.approx(45.0)


def test_flat_offset_is_the_iso_offset_turned_into_flat():
    derived = unit_sprites.ANGLE_ZERO_OFFSET_DEG + INDEX_WALK_SENSE * iso_to_flat_turn_deg()
    gap = (unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG - derived) % 360.0
    assert min(gap, 360.0 - gap) == pytest.approx(0.0, abs=1e-9)
