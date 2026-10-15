"""unit_sprites.rotation_facing_viewer() (GH #68, TASK-075 S2): the rotation
the Units catalog's sprite thumbnails use, down-left toward the viewer, for
ANGLE consts only. Install-free except the one real-asset check."""

from __future__ import annotations

import math

import pytest

from descape import asset_source, unit_rotation, unit_sprites

_TREE = 349
_WALL = 117
_GATE = 64
_PIECE_OF_THE_TRUE_CROSS = 688  # hidden, ANGLE, 16 frames, no portrait


def _angle_consts() -> list[int]:
    return [const for const in unit_sprites.graphic_map() if unit_rotation.rotation_is_angle(const)]


def test_every_angle_const_faces_the_viewer() -> None:
    consts = _angle_consts()
    assert len(consts) > 800, "the premise: the table covers the creatables"
    counts = set()
    for const in consts:
        count = unit_rotation.angle_count_for(const)
        counts.add(count)
        rotation = unit_sprites.rotation_facing_viewer(const)
        # Facts (measured 2026-10-08): pi is down-left for every count in use.
        assert rotation == pytest.approx(math.pi), const
        if count % 8 == 0:
            assert unit_sprites.angle_index(rotation, count) == 3 * count // 8, const
    assert {16, 32, 8} <= counts, "the multiple-of-8 counts must all be exercised"
    assert counts - {8, 16, 32}, "and at least one count with no zero-point offset"


def test_a_count_with_no_offset_takes_the_frame_pi_selects() -> None:
    six = next(const for const in _angle_consts() if unit_rotation.angle_count_for(const) == 6)
    assert unit_sprites.angle_index(unit_sprites.rotation_facing_viewer(six), 6) == 3


@pytest.mark.parametrize("const", [_TREE, _WALL, _GATE, "cliff"])
def test_nothing_that_is_not_a_facing_turns(const) -> None:
    if const == "cliff":
        const = min(unit_sprites.cliff_consts())
    assert unit_sprites.rotation_facing_viewer(const) is None


def test_an_unknown_const_does_not_turn() -> None:
    assert unit_sprites.rotation_facing_viewer(999999) is None


def test_the_real_true_cross_turns_to_a_different_frame() -> None:
    """Needs AOE2DE_INSTALL_PATH (conftest hides config.yaml)."""
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
    const = _PIECE_OF_THE_TRUE_CROSS
    facing = unit_sprites.rotation_facing_viewer(const)
    turned = unit_sprites.sprite_pieces_for(const, facing, 0, 32)
    default = unit_sprites.sprite_pieces_for(const, 0.0, 0, 32)
    assert turned and default, "the real sprite must resolve, or this proves nothing"
    assert turned[0].draw.rgba.tobytes() != default[0].draw.rgba.tobytes()
