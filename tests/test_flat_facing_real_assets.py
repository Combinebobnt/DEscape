"""GH #83: tools/verify_flat_facing.py's real-art arms, run as tests.

The suite's synthetic frames have no real facing, so whether
FLAT_ANGLE_ZERO_OFFSET_DEG is the RIGHT number can only be measured on game
art. The tool holds the measurement; this file runs it against the galley
(both projections), a battering ram (Flat only, see its test) and the knight's
mirror pair.

Needs AOE2DE_INSTALL_PATH rather than the configured install, for the reason
tests/test_sprite_variant14_real_assets.py's docstring spells out: conftest's
autouse _isolated_settings hides config.yaml from the suite on purpose.

**Deliberately NOT `corpus`-marked**, per tests/test_wall_variants_real_assets.py:
every frame decodes from files already on disk, well under a second in total,
so it earns the default tier. Gating is left to _require_install()'s skip.
"""

from __future__ import annotations

import math

import pytest

from descape import asset_source, unit_sprites

import conftest

# The battering ram: the only long, thin siege shape. Still far rounder than
# the galley (elongation about 1.6 side-on, 1.1 end-on, against the galley's 3.6+).
RAM_GRAPHIC = "u_sie_battering_ram_idleA_x1"
RAM_GRAPHIC_CONST = 35

# Below this a principal axis is too close to round to hold the 8 degree
# tolerance; the galley's least elongated frame reads 3.6.
MIN_ELONGATION = 1.5


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


@pytest.fixture
def tool():
    _require_install()
    return conftest.load_verify_module("verify_flat_facing")


def _masks(tool, graphic: str, graphic_const: int) -> list:
    """The graphic's per-angle masks, after the tool's own shape check. A
    reshaped table entry fails; art this install lacks skips."""
    entry = unit_sprites.graphic_map().get(graphic_const, {})
    actual = (entry.get("file_name"), entry.get("angle_count"), entry.get("frame_count"))
    assert actual == (graphic, tool.ANGLE_COUNT, 1), (
        f"const {graphic_const} now reads {actual}: frame index == angle index only at frame_count 1"
    )
    if unit_sprites._native_frame(graphic, 0) is None:
        pytest.skip(f"{graphic} does not decode on this install")
    return tool.frame_masks(graphic, tool.ANGLE_COUNT)


def _arm_failures(tool, masks, offset_deg: float, expected: dict[float, float]) -> list[str]:
    """Each rotation's resolved frame must be elongated enough to trust, then
    read along the expected screen axis within the tool's tolerance."""
    failures = []
    for rotation, want in expected.items():
        index = unit_sprites.angle_index(rotation, tool.ANGLE_COUNT, offset_deg)
        elongation = tool.axis_elongation(masks[index])
        if elongation < MIN_ELONGATION:
            failures.append(f"rotation {rotation:.4f} -> frame {index}: elongation {elongation:.2f}, too round to read")
            continue
        got = tool.principal_axis_deg(masks[index])
        gap = tool.axis_gap_deg(got, want)
        if gap > tool.TOLERANCE_DEG:
            failures.append(f"rotation {rotation:.4f} -> frame {index}: axis {got:.2f} vs {want:.2f} (off {gap:.2f})")
    return failures


def _flat_expected() -> dict[float, float]:
    return {0.0: 0.0, math.pi / 2: 90.0, math.pi: 180.0, 3 * math.pi / 2: 270.0}


def test_the_galley_faces_along_the_flat_axes(tool) -> None:
    masks = _masks(tool, tool.GRAPHIC, tool.GRAPHIC_CONST)
    assert not _arm_failures(tool, masks, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG, _flat_expected())


def test_the_galley_faces_along_the_iso_axes_as_a_control(tool) -> None:
    """If the Flat arm passed and this failed, the flat number would be right
    by accident on frames that are not where they are believed to be."""
    masks = _masks(tool, tool.GRAPHIC, tool.GRAPHIC_CONST)
    iso_x, iso_y = tool.iso_screen_dirs()
    expected = {0.0: iso_x, math.pi / 2: iso_y, math.pi: iso_x, 3 * math.pi / 2: iso_y}
    assert not _arm_failures(tool, masks, unit_sprites.ANGLE_ZERO_OFFSET_DEG, expected)


def test_the_ram_faces_along_the_flat_x_axis(tool) -> None:
    """Rotations 0 and pi only: the ram's end-on frames are nearly round, and
    its iso frames sit 6-7 degrees off even when right, so neither arm can
    tell one frame from the next. A 90 degree offset error still lands 0 on
    an end-on frame, which the elongation guard fails."""
    masks = _masks(tool, RAM_GRAPHIC, RAM_GRAPHIC_CONST)
    expected = {0.0: 0.0, math.pi: 180.0}
    assert not _arm_failures(tool, masks, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG, expected)


def test_the_knight_mirror_pair_skews_in_opposite_directions(tool) -> None:
    """The mod-360 half the axis arms cannot see: angle 0 faces right, angle
    8 left, so their horizontal skews must disagree in sign."""
    entry = unit_sprites.graphic_map().get(tool.KNIGHT_GRAPHIC_CONST, {})
    assert (entry.get("file_name"), entry.get("angle_count")) == (tool.KNIGHT_GRAPHIC, tool.ANGLE_COUNT)
    frame_count = entry["frame_count"]
    right = unit_sprites._native_frame(tool.KNIGHT_GRAPHIC, tool.KNIGHT_ANGLE_RIGHT * frame_count)
    left = unit_sprites._native_frame(tool.KNIGHT_GRAPHIC, tool.KNIGHT_ANGLE_LEFT * frame_count)
    if right is None or left is None:
        pytest.skip(f"{tool.KNIGHT_GRAPHIC} does not decode on this install")
    skew_right = tool.horizontal_skew(right[0][..., 3])
    skew_left = tool.horizontal_skew(left[0][..., 3])
    assert skew_right != 0 and skew_left != 0
    assert (skew_right > 0) != (skew_left > 0), (skew_right, skew_left)
