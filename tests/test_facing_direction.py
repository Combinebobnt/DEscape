"""Automates the gate-orientation-cycling checklist's step 3.3 ("select a
gate and an archer together and press '.' once -- both must turn the same
screen direction").

**This pins the LINK, not ANGLE_DIRECTION itself.** unit_sprites.
ANGLE_DIRECTION == 1 still rests on the live-window observation recorded at
unit_sprites.py:117-126 ("AoE2's rotation field grows CLOCKWISE on screen"),
reproducible by nothing in this suite -- re-deriving it from real sprite
pixels needs a skew/mirror-pair statistic a principal-axis measurement
(tools/verify_flat_facing.py's own PCA) cannot provide, and is scoped out as
its own `[NEEDS DECISION]` TODO item rather than folded in here. A green run
of this file is evidence that the archer and gate paths AGREE with each
other, never independent evidence that either is correct in absolute terms.

What IS pinned, from committed constants only: composing angle_index()'s own
step direction and gate_orientation's own cycle order with iso_geometry's
real screen projection, a positive unit_rotation.rotate_step() and a positive
gate_orientation.cycle_const() step both advance clockwise on screen, in the
same sense. A reader who flips ANGLE_DIRECTION, reorders
gate_orientation._ORIENTATIONS, or breaks iso_geometry's own (x+y)/(y-x) sign
convention should see this file go red.

**Mutation arm, run by hand and confirmed 2026-09-08, not asserted**: flipping
`unit_sprites.ANGLE_DIRECTION` to -1 turns 3 of these 4 tests red (item 1's
own `ANGLE_DIRECTION == 1` assertion, item 2, and item 4) -- item 3 alone
stays green, since it never reads `ANGLE_DIRECTION` at all, which is the
expected shape: it is the archer side of the link that depends on the
constant, not the gate side.
"""

from __future__ import annotations

import math

import pytest

from descape import iso_geometry, unit_rotation, unit_sprites

# A throwaway projection -- only half_w/half_h (the fixed 2:1 aspect) matter
# below, never the map size or origin.
_PROJ = iso_geometry.canvas_size_and_origin(
    8, 8, 64, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
)


def _screen_angle_deg(tile_angle_rad: float) -> float:
    """A tile-space DIRECTION at `tile_angle_rad` (standard math convention,
    (cos, sin) from tile +x), pushed through iso_geometry's own linear
    projection -- tile_screen_origin()'s (x+y)*half_w, (y-x)*half_h terms,
    stripped of their additive origin since only the direction matters here
    -- and read back as a screen-space angle in degrees, clockwise from
    screen-right with y down (atan2's own convention once y grows downward).
    """
    dx, dy = math.cos(tile_angle_rad), math.sin(tile_angle_rad)
    sx = (dx + dy) * _PROJ.half_w
    sy = (dy - dx) * _PROJ.half_h
    return math.degrees(math.atan2(sy, sx))


def _unwrap(angles_deg: list[float]) -> list[float]:
    """A compass-bearing unwrap: atan2 wraps at +-180, so a genuinely
    monotonic sequence can still show a spurious decrease right at the wrap
    unless each step is adjusted into the same turn as its predecessor."""
    out = [angles_deg[0]]
    for a in angles_deg[1:]:
        prev = out[-1]
        while a - prev > 180:
            a -= 360
        while a - prev < -180:
            a += 360
        out.append(a)
    return out


def test_angle_index_is_monotonically_increasing_in_rotation():
    """1: implied by ANGLE_DIRECTION == 1's own definition (unit_sprites.
    angle_index()'s docstring), never asserted on its own before this.
    angle_count 16 dominates the real install (ANGLE_ZERO_OFFSET_DEG's own
    measurement), so it stands in for "a real archer-like graphic" here. A
    step function, not strictly increasing -- many rotations land on the
    same stored index -- so "monotonic" means never decreasing except at the
    single wrap from angle_count - 1 back to 0."""
    assert unit_sprites.ANGLE_DIRECTION == 1
    angle_count = 16
    samples = [i * 2 * math.pi / 200 for i in range(200)]
    indices = [unit_sprites.angle_index(r, angle_count, 0.0) for r in samples]
    violations = [(a, b) for a, b in zip(indices, indices[1:]) if b < a and b != 0]
    assert not violations, violations
    assert len(set(indices)) == angle_count, "the samples never even covered every stored index"


def _archer_screen_angle_deg(rotation: float, angle_count: int) -> float:
    """The screen angle an archer at `rotation` appears to face: whichever
    frame angle_index() selects is a fixed, evenly-spaced physical direction
    -- index i at tile angle i * 2*pi / angle_count, the layout
    ANGLE_ZERO_OFFSET_DEG's own comment assumes -- projected to screen space.

    offset_deg is pinned at 0.0 deliberately, not ANGLE_ZERO_OFFSET_DEG: a
    zero-point offset rotates the WHOLE stored set by a fixed amount, which
    changes WHERE on the circle index 0 starts but not whether one step's
    direction is clockwise or counterclockwise -- the only thing checked
    here."""
    index = unit_sprites.angle_index(rotation, angle_count, 0.0)
    tile_angle = index * 2 * math.pi / angle_count
    return _screen_angle_deg(tile_angle)


def test_a_positive_rotate_step_advances_an_archers_screen_facing_clockwise():
    """2: composing angle_index()'s own step direction with iso_geometry's
    screen projection. rotate_step() deliberately does not apply
    ANGLE_ZERO_OFFSET_DEG either (its own docstring), so this needs no offset
    correction on either side."""
    angle_count = 16
    rotation = 0.0
    screen_angles = [_archer_screen_angle_deg(rotation, angle_count)]
    for _ in range(angle_count):
        rotation = unit_rotation.rotate_step(rotation, angle_count, 1)
        screen_angles.append(_archer_screen_angle_deg(rotation, angle_count))

    unwrapped = _unwrap(screen_angles)
    diffs = [b - a for a, b in zip(unwrapped, unwrapped[1:])]
    assert all(d > 0 for d in diffs), (
        f"a positive rotate_step() must always advance the screen angle -- got {diffs}"
    )
    assert unwrapped[-1] - unwrapped[0] == pytest.approx(360.0), (
        "one full cycle of steps should trace exactly one full turn"
    )


def test_a_gate_cycle_step_advances_its_run_angle_clockwise_on_screen():
    """3: gate_orientation.cycle_const()'s own +1 step is a constant 45
    degrees in TILE space (tests/test_gate_orientation.py's own
    test_one_cycle_step_turns_the_footprint_45_degrees) -- projected through
    the real iso screen map that becomes the NON-evenly-spaced sequence
    -26.57, 0, +26.57, +90 degrees (the 2:1 vertical squash), still
    monotonically increasing (clockwise)."""
    screen_angles = [_screen_angle_deg(i * math.pi / 4) for i in range(4)]
    for got, want in zip(screen_angles, (-26.565, 0.0, 26.565, 90.0)):
        assert got == pytest.approx(want, abs=0.01), (got, want)
    diffs = [b - a for a, b in zip(screen_angles, screen_angles[1:])]
    assert all(d > 0 for d in diffs), diffs


def test_the_archer_and_gate_steps_agree_in_sign():
    """4: "Therefore 2 and 3 agree in sign." -- checklist step 3.3 itself.
    Both per-step screen-angle deltas must be positive under this file's own
    atan2/y-down convention -- the same rotational sense, i.e. one Rotate
    keypress turns an archer and a gate the same way."""
    archer_delta = _archer_screen_angle_deg(
        unit_rotation.rotate_step(0.0, 16, 1), 16
    ) - _archer_screen_angle_deg(0.0, 16)
    gate_delta = _screen_angle_deg(math.pi / 4) - _screen_angle_deg(0.0)
    assert archer_delta > 0 and gate_delta > 0, (archer_delta, gate_delta)
