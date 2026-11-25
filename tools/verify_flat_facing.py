#!/usr/bin/env python3
"""Verifies unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG against a real install.

Install-gated, which is why it lives here rather than in tests/: the suite's
standing posture is synthetic bytes and no game assets, and a synthetic frame
has no real facing to measure. What the suite can check is that the two zero
points are threaded and applied consistently; what only this script can check
is whether the flat one is the RIGHT number. Those are different failures, and
the second is the one that costs a round trip to a human with the game open.

The measurement, reproducing the table in maintainer's own plan for this
change: u_shp_war_galley_x1 stores angle_count 16 and frame_count 1, so a
frame index IS an angle index. Each frame's alpha mask gets a principal-axis
angle (the eigenvector of its covariance matrix, largest eigenvalue),
expressed clockwise from screen-right with y down. A war galley is long and
thin, so its principal axis is its hull, i.e. the direction it faces.

**Arms 1-2 compare mod 180, because a principal axis has no direction.**
Frames 4 and 12 are both a vertical boat and both measure ~90, so those two
arms would pass just as happily at a Flat offset of 180. The mod-360 sign
comes instead from a derivation: Flat draws the iso art, so its offset is
the iso offset (game-measured, see tests/test_ingame_angle_mapping.py) plus
the iso-to-flat screen turn of 45 degrees times the sense in which the
stored index walks on screen. Arm 3 measures that sense, and
tests/test_ingame_angle_mapping.py pins the sum.

Four arms, since the point is a comparison and not an absolute:

1. **Flat.** At FLAT_ANGLE_ZERO_OFFSET_DEG, rotation 0 must read along flat
   +x (screen-horizontal) and pi/2 along flat +y (screen-vertical), with pi
   and 3*pi/2 the same axes again.
2. **Isometric, as a control.** At ANGLE_ZERO_OFFSET_DEG the same rotations
   must read along the ISO screen directions of world +x and +y, which
   iso_geometry's own projection gives as 153.43 and 26.57 degrees. If arm 1
   passed and this one failed, the flat number would be right by accident on
   a graphic whose frames are not where they are believed to be.
3. **Index walk sense.** index_walk_sense() takes every consecutive axis
   difference of the galley's 16 frames, each wrapped into (-90, 90], and
   requires them all to share one sign: +1, clockwise on screen with y
   down. A sense is not a heading, so a direction-blind axis does measure
   it. The projection puts consecutive frames 11.7 to 39.6 screen degrees
   apart, far inside the 90 degree limit of a wrapped difference, and the
   arm fails below SENSE_MIN_STEP_DEG. Its control reruns the statistic on
   the frames with each adjacent pair swapped, which must read mixed.
4. **Knight decode, a consistency check.** horizontal_skew(), a
   third-moment statistic, is direction-sensitive where a covariance-based
   axis is not, and a genuine mirror pair (u_cav_knight_idleC_x1's angles
   0 and 8, the horse facing screen-RIGHT and LEFT) must have skews of
   OPPOSITE sign. That confirms the two frames are still a real facing pair
   on this install. The mod-360 sign rests on arm 3 and the derivation
   above, not on this arm.

Usage: python3 tools/verify_flat_facing.py [--install PATH]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from descape import asset_source, unit_sprites

GRAPHIC = "u_shp_war_galley_x1"
GRAPHIC_CONST = 21
ANGLE_COUNT = 16

# The galley's hull is long and thin, so its principal axis is well
# conditioned, but a frame index is still a 22.5-degree bucket and the ink is
# antialiased. 8 degrees is comfortably inside half a bucket, so a whole-frame
# error cannot hide under it, and comfortably outside the 1.5-degree spread
# the real frames show against their own nominal angles.
TOLERANCE_DEG = 8.0


def _alpha_covariance(alpha: np.ndarray) -> np.ndarray:
    """The mask's 2x2 (x, y) covariance, weighted by alpha."""
    ys, xs = np.nonzero(alpha)
    w = alpha[ys, xs].astype(np.float64)
    x = xs.astype(np.float64) - np.average(xs, weights=w)
    y = ys.astype(np.float64) - np.average(ys, weights=w)
    return np.array([
        [np.average(x * x, weights=w), np.average(x * y, weights=w)],
        [np.average(x * y, weights=w), np.average(y * y, weights=w)],
    ])


def principal_axis_deg(alpha: np.ndarray) -> float:
    """The mask's principal axis, in degrees clockwise from screen-right with
    y down, mod 180. Weighted by alpha rather than thresholded, so an
    antialiased hull edge contributes proportionally."""
    values, vectors = np.linalg.eigh(_alpha_covariance(alpha))
    vx, vy = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def axis_elongation(alpha: np.ndarray) -> float:
    """Largest over smallest covariance eigenvalue: how far a principal axis
    can be trusted. Near 1 the mask is round and its axis is noise."""
    values = np.linalg.eigvalsh(_alpha_covariance(alpha))
    return float(values[-1] / values[0]) if values[0] > 0 else math.inf


KNIGHT_GRAPHIC = "u_cav_knight_idleC_x1"
KNIGHT_GRAPHIC_CONST = 38
KNIGHT_ANGLE_RIGHT = 0
KNIGHT_ANGLE_LEFT = 8


def horizontal_skew(alpha: np.ndarray) -> float:
    """Pearson's third-moment skewness of the mask's ink along x, weighted by
    alpha. Unlike a principal axis (direction-blind by construction), skew's
    SIGN flips between mirror images -- a shape whose ink reaches further to
    one side than the other reads as skewed toward that side, and a horse
    facing right vs the same horse facing left is exactly that kind of
    mirror pair."""
    ys, xs = np.nonzero(alpha)
    w = alpha[ys, xs].astype(np.float64)
    mean = np.average(xs, weights=w)
    variance = np.average((xs - mean) ** 2, weights=w)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return float(np.average(((xs - mean) / std) ** 3, weights=w))


def axis_gap_deg(a: float, b: float) -> float:
    """Separation of two undirected axes: never more than 90 degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def iso_screen_dirs() -> tuple[float, float]:
    """The iso screen directions of world +x and +y, mod 180, derived from
    iso_geometry's own projection rather than hardcoded.

    tile_screen_origin() is sx = origin_x + (x+y)*half_w and
    sy = origin_y + (y-x)*half_h, so a unit step in world +x moves the screen
    point by (half_w, -half_h) and one in +y by (half_w, +half_h)."""
    from descape import iso_geometry

    proj = iso_geometry.canvas_size_and_origin(
        8, 8, 64, iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION
    )
    plus_x = math.degrees(math.atan2(-proj.half_h, proj.half_w)) % 180.0
    plus_y = math.degrees(math.atan2(proj.half_h, proj.half_w)) % 180.0
    return plus_x, plus_y


def _assert_graphic_shape(
    graphic: str = GRAPHIC, graphic_const: int = GRAPHIC_CONST, angle_count: int = ANGLE_COUNT
) -> None:
    """A game patch could reshape the galley's angle set, and the whole
    measurement rests on frame index == angle index. Fail loudly here rather
    than reporting confident nonsense from a graphic that no longer has
    frame_count 1."""
    entry = unit_sprites.graphic_map().get(graphic_const, {})
    actual = (entry.get("file_name"), entry.get("angle_count"), entry.get("frame_count"))
    wanted = (graphic, angle_count, 1)
    if actual != wanted:
        raise SystemExit(
            f"const {graphic_const} now reads {actual}, not {wanted}. Re-measure before "
            f"trusting anything below: frame index == angle index only at frame_count 1."
        )


def frame_masks(graphic: str = GRAPHIC, angle_count: int = ANGLE_COUNT) -> list[np.ndarray]:
    """Every stored angle's alpha mask, index-ordered (frame_count 1 only)."""
    out = []
    for index in range(angle_count):
        native = unit_sprites._native_frame(graphic, index)
        if native is None:
            raise SystemExit(
                f"{graphic} frame {index} did not decode. This install does not hold the "
                f"graphic the offset was measured on, so nothing below would mean anything."
            )
        out.append(native[0][..., 3])
    return out


def frame_axes(graphic: str = GRAPHIC, angle_count: int = ANGLE_COUNT) -> list[float]:
    """Every stored angle's measured axis, index-ordered."""
    return [principal_axis_deg(mask) for mask in frame_masks(graphic, angle_count)]


def _run_arm(
    name: str,
    offset_deg: float,
    expected: dict[float, float],
    axes: list[float],
    angle_count: int = ANGLE_COUNT,
) -> bool:
    """One projection's arm: every cardinal rotation's resolved frame must
    read along that projection's own screen direction for the same world
    axis.

    Underscore-prefixed, and NOT named check_*, deliberately. That prefix is a
    convention across tools/verify_*.py: a check_* takes the fixture arguments
    tests/migration_manifest.py declares for it and returns (ok, detail), which
    is what tests/test_legacy_adapter.py wraps. This takes four arguments and
    returns a bool, so it is not one."""
    print(f"\n=== {name} (offset {offset_deg} degrees) ===")
    ok = True
    for rotation, want in expected.items():
        index = unit_sprites.angle_index(rotation, angle_count, offset_deg)
        got = axes[index]
        gap = axis_gap_deg(got, want)
        verdict = "PASS" if gap <= TOLERANCE_DEG else "FAIL"
        ok = ok and gap <= TOLERANCE_DEG
        print(
            f"  {verdict}  rotation {rotation:.4f} -> frame {index:>2}: "
            f"axis {got:7.2f} vs wanted {want:6.2f} (off by {gap:.2f})"
        )
    return ok


def _run_knight_arm() -> bool:
    """Arm 4: the knight mirror-pair consistency check. Skips (does not fail) if this install
    doesn't hold the knight graphic, matching every other failure mode this
    script treats as "nothing to measure" rather than a hard error --
    tests/README.md's own convention for install-gated tools/verify_*.py."""
    print(f"\n=== Knight decode ({KNIGHT_GRAPHIC}, mirror-pair consistency) ===")
    entry = unit_sprites.graphic_map().get(KNIGHT_GRAPHIC_CONST, {})
    if (entry.get("file_name"), entry.get("angle_count")) != (KNIGHT_GRAPHIC, ANGLE_COUNT):
        print(f"  SKIP  const {KNIGHT_GRAPHIC_CONST} is not {KNIGHT_GRAPHIC} at {ANGLE_COUNT} angles here")
        return True
    # Frames are angle-major, so angle N starts at frame N * frame_count.
    frame_count = entry["frame_count"]
    right = unit_sprites._native_frame(KNIGHT_GRAPHIC, KNIGHT_ANGLE_RIGHT * frame_count)
    left = unit_sprites._native_frame(KNIGHT_GRAPHIC, KNIGHT_ANGLE_LEFT * frame_count)
    if right is None or left is None:
        print(f"  SKIP  {KNIGHT_GRAPHIC} did not decode on this install")
        return True
    skew_right = horizontal_skew(right[0][..., 3])
    skew_left = horizontal_skew(left[0][..., 3])
    ok = skew_right != 0 and skew_left != 0 and (skew_right > 0) != (skew_left > 0)
    verdict = "PASS" if ok else "FAIL"
    print(f"  {verdict}  angle {KNIGHT_ANGLE_RIGHT:>2} (screen-RIGHT) skew {skew_right:+.4f}")
    print(f"  {verdict}  angle {KNIGHT_ANGLE_LEFT:>2} (screen-LEFT)  skew {skew_left:+.4f}")
    if not ok:
        print("  the two frames' skew did not disagree in sign -- they may no longer be a mirror pair")
    return ok


def wrapped_steps(axes: list[float]) -> list[float]:
    """Each consecutive axis difference, the last frame back to the first
    included, wrapped into (-90, 90]. Unambiguous while every true step is
    under 90 degrees, which the galley's 11.7 to 39.6 screen steps are."""
    out = []
    for a, b in zip(axes, axes[1:] + axes[:1], strict=True):
        d = (b - a) % 180.0
        out.append(d - 180.0 if d > 90.0 else d)
    return out


def index_walk_sense(axes: list[float]) -> tuple[int, float] | None:
    """The screen sense the stored index walks in (+1 clockwise, y down, -1
    counter-clockwise) plus the smallest absolute step, or None when the steps
    disagree in sign. A sense, not a heading, so a direction-blind axis can
    measure it."""
    steps = wrapped_steps(axes)
    if all(s > 0 for s in steps):
        sense = 1
    elif all(s < 0 for s in steps):
        sense = -1
    else:
        return None
    return sense, min(abs(s) for s in steps)


SENSE_MIN_STEP_DEG = 6.0


def pair_swapped(axes: list[float]) -> list[float]:
    """The frame list with each adjacent pair swapped (0,1,2,3 -> 1,0,3,2): out of order, so mixed."""
    return [axes[index ^ 1] for index in range(len(axes) - len(axes) % 2)]


def _run_sense_arm(axes: list[float]) -> bool:
    """Arm 3: every consecutive frame of the galley must turn clockwise on
    screen, by at least SENSE_MIN_STEP_DEG. Its control reruns the statistic
    on the frames with each adjacent pair swapped, which must read mixed (a
    reversed list cannot serve: wrapped_steps() is antisymmetric)."""
    print(f"\n=== Index walk sense ({GRAPHIC}, all {len(axes)} steps) ===")
    for index, step in enumerate(wrapped_steps(axes)):
        print(f"  frame {index:>2} -> {(index + 1) % len(axes):>2}: {step:+7.2f}")
    measured = index_walk_sense(axes)
    swapped_sense = index_walk_sense(pair_swapped(axes))
    ok = (
        measured is not None and measured[0] == 1 and measured[1] >= SENSE_MIN_STEP_DEG
        and swapped_sense is None
    )
    verdict = "PASS" if ok else "FAIL"
    sense = "mixed" if measured is None else f"{measured[0]:+d}"
    control = "mixed" if swapped_sense is None else f"{swapped_sense[0]:+d}"
    print(f"  {verdict}  sense {sense} (want +1), pair-swapped control {control} (want mixed)")
    if measured is not None:
        margin = measured[1] - SENSE_MIN_STEP_DEG
        print(f"  smallest step {measured[1]:.2f}, margin {margin:.2f} over the {SENSE_MIN_STEP_DEG} floor")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--install", type=Path, help="AoE2:DE install root override")
    args = parser.parse_args()

    if args.install:
        asset_source.set_install_path_override(args.install)
    if asset_source.get_install_path() is None:
        raise SystemExit(
            "No AoE2:DE install configured. This script reads real sprite pixels and "
            "has nothing to measure without one. Pass --install."
        )
    print(f"Flat facing check, install {asset_source.get_install_path()}")

    _assert_graphic_shape()
    axes = frame_axes()
    print(f"\n=== {GRAPHIC}: measured principal axis per stored angle ===")
    for index, axis in enumerate(axes):
        print(f"  idx {index:>2}  {axis:7.2f}")

    iso_x, iso_y = iso_screen_dirs()
    arms = [
        ("Flat (non-isometric)", unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG, {
            0.0: 0.0, math.pi / 2: 90.0, math.pi: 180.0, 3 * math.pi / 2: 270.0,
        }),
        ("Isometric (control)", unit_sprites.ANGLE_ZERO_OFFSET_DEG, {
            0.0: iso_x, math.pi / 2: iso_y, math.pi: iso_x, 3 * math.pi / 2: iso_y,
        }),
    ]
    failures = sum(
        0 if _run_arm(name, offset, expected, axes) else 1 for name, offset, expected in arms
    )
    failures += 0 if _run_sense_arm(axes) else 1
    failures += 0 if _run_knight_arm() else 1
    total = len(arms) + 2

    print(f"\n{total - failures}/{total} arms passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
