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

**Everything below is compared mod 180, because a principal axis has no
direction.** Frames 4 and 12 are both a vertical boat and both measure ~90.
That is why the flat zero point's SIGN, as opposed to its magnitude, was
settled separately by decoding u_cav_knight_idleC_x1 (angle 0's horse faces
screen-RIGHT, angle 8 faces LEFT) rather than here. This script would pass
just as happily at an offset of 180, so it is one half of the evidence, not
all of it. See FLAT_ANGLE_ZERO_OFFSET_DEG's own comment for the other half.

Three arms, since the point is a comparison and not an absolute:

1. **Flat.** At FLAT_ANGLE_ZERO_OFFSET_DEG, rotation 0 must read along flat
   +x (screen-horizontal) and pi/2 along flat +y (screen-vertical), with pi
   and 3*pi/2 the same axes again.
2. **Isometric, as a control.** At ANGLE_ZERO_OFFSET_DEG the same rotations
   must read along the ISO screen directions of world +x and +y, which
   iso_geometry's own projection gives as 153.43 and 26.57 degrees. If arm 1
   passed and this one failed, the flat number would be right by accident on
   a graphic whose frames are not where they are believed to be.
3. **Knight decode (mod-360 sign).** Arms 1-2 both compare mod 180 (a
   principal axis is direction-blind), so neither can tell "faces right"
   from "faces left" -- that half of the evidence rested on a one-off prose
   decode of u_cav_knight_idleC_x1 (angle 0's horse faces screen-RIGHT,
   angle 8 faces LEFT), re-run by no code before this. A principal axis
   still can't resolve it, so this arm uses a THIRD-MOMENT (skew) statistic
   instead: horizontal_skew() is direction-sensitive where a covariance-based
   axis is not, and a genuine mirror pair (which angle 0 and angle 8 are, 8
   steps apart on a 16-angle graphic) must have skews of OPPOSITE sign. That
   confirms the two frames are still a real facing pair on this install --
   it does not independently re-derive which sign means "right" in the
   abstract. A from-scratch re-derivation of that sign is a separate, harder
   research task (it would need a labelled corpus of known-facing sprites,
   not just one hand-decoded pair), deliberately not attempted here.

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


def principal_axis_deg(alpha: np.ndarray) -> float:
    """The mask's principal axis, in degrees clockwise from screen-right with
    y down, mod 180. Weighted by alpha rather than thresholded, so an
    antialiased hull edge contributes proportionally."""
    ys, xs = np.nonzero(alpha)
    w = alpha[ys, xs].astype(np.float64)
    x = xs.astype(np.float64) - np.average(xs, weights=w)
    y = ys.astype(np.float64) - np.average(ys, weights=w)
    cov = np.array([
        [np.average(x * x, weights=w), np.average(x * y, weights=w)],
        [np.average(x * y, weights=w), np.average(y * y, weights=w)],
    ])
    values, vectors = np.linalg.eigh(cov)
    vx, vy = vectors[:, int(np.argmax(values))]
    return math.degrees(math.atan2(vy, vx)) % 180.0


KNIGHT_GRAPHIC = "u_cav_knight_idleC_x1"
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


def _assert_graphic_shape() -> None:
    """A game patch could reshape the galley's angle set, and the whole
    measurement rests on frame index == angle index. Fail loudly here rather
    than reporting confident nonsense from a graphic that no longer has
    frame_count 1."""
    entry = unit_sprites.graphic_map().get(GRAPHIC_CONST, {})
    actual = (entry.get("file_name"), entry.get("angle_count"), entry.get("frame_count"))
    wanted = (GRAPHIC, ANGLE_COUNT, 1)
    if actual != wanted:
        raise SystemExit(
            f"const {GRAPHIC_CONST} now reads {actual}, not {wanted}. Re-measure before "
            f"trusting anything below: frame index == angle index only at frame_count 1."
        )


def frame_axes() -> list[float]:
    """Every stored angle's measured axis, index-ordered."""
    out = []
    for index in range(ANGLE_COUNT):
        native = unit_sprites._native_frame(GRAPHIC, index)
        if native is None:
            raise SystemExit(
                f"{GRAPHIC} frame {index} did not decode. This install does not hold the "
                f"graphic the offset was measured on, so nothing below would mean anything."
            )
        out.append(principal_axis_deg(native[0][..., 3]))
    return out


def _run_arm(name: str, offset_deg: float, expected: dict[float, float], axes: list[float]) -> bool:
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
        index = unit_sprites.angle_index(rotation, ANGLE_COUNT, offset_deg)
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
    """Arm 3: the mod-360 sign check. Skips (does not fail) if this install
    doesn't hold the knight graphic, matching every other failure mode this
    script treats as "nothing to measure" rather than a hard error --
    tests/README.md's own convention for install-gated tools/verify_*.py."""
    print(f"\n=== Knight decode ({KNIGHT_GRAPHIC}, mod-360 sign) ===")
    right = unit_sprites._native_frame(KNIGHT_GRAPHIC, KNIGHT_ANGLE_RIGHT)
    left = unit_sprites._native_frame(KNIGHT_GRAPHIC, KNIGHT_ANGLE_LEFT)
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
    failures += 0 if _run_knight_arm() else 1
    total = len(arms) + 1

    print(f"\n{total - failures}/{total} arms passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
