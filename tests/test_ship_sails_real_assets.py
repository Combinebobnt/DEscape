"""A sailing ship draws its sails and its owner's colour (GH #20).

A ship's standing graphic is a legacy shell whose deltas are the hull and then
one graphic per sail. The graphic map used to keep the hull alone, and on War
Galley, Galleon, Fire Ship and Transport Ship the hull has no player-colour
pixels, so those drew bare and unowned. tests/test_unit_graphic_map.py pins
which files each ship's `pieces` name; only a real `.sld` decode can say
whether those pieces paint, where, and in whose colour.

Needs AOE2DE_INSTALL_PATH, like tests/test_wall_variants_real_assets.py:
conftest._isolated_settings hides config.yaml. Not `corpus`-marked for the
same reason as that file: a few hundred frame decodes from files on disk.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from descape import asset_source, unit_sprites

GALLEON, WAR_GALLEY, FISHING_SHIP = 442, 21, 13

# Mirrors tests/test_unit_graphic_map.py's SHIP_SAIL_CONSTS.
SHIP_SAIL_CONSTS = (
    17, 21, 114, 420, 442, 527, 528, 529, 532, 535, 536, 539, 545, 691, 1103, 1104,
    2626, 2627, 2628, 2633,
)

# Ships whose hull carries no PLAYERCOLOR pixels at any facing: the sails are
# the only place their owner shows. Measured 2026-10-12 over all 16 angles.
BARE_HULL_CONSTS = (WAR_GALLEY, GALLEON, 529, 545)

# sprite_scale() == 1.0, so every offset below is in native px.
NATIVE_HALF_W = unit_sprites.NATIVE_TILE_W // 2

# The four screen diagonals plus an off-grid facing.
ROTATIONS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.7)
ALL_FACINGS = tuple(i * 2 * math.pi / 16 for i in range(16))

# Galleon's sails lift its ink top 40..73 px over the hull's across ROTATIONS.
MIN_GALLEON_SAIL_RISE = 30

RED = 2


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


def _ink_top(pieces) -> int:
    """The highest opaque row over every piece, relative to the unit's anchor."""
    return min(
        int(np.nonzero((p.draw.rgba[..., 3] > 0).any(axis=1))[0][0]) - p.draw.hotspot_y + p.dy
        for p in pieces
    )


def _tinted_px(unit_const: int, rotation: float, pieces_for) -> int:
    """Pixels the owner's colour changes: RED's draw against GAIA's untinted one."""
    red, plain = (pieces_for(unit_const, rotation, team) for team in (RED, 0))
    assert len(red) == len(plain)
    return sum(
        int((a.draw.rgba != b.draw.rgba).any(axis=2).sum()) for a, b in zip(red, plain, strict=True)
    )


def _hull_only(unit_const: int, rotation: float, team: int):
    draw = unit_sprites.sprite_for(unit_const, rotation, team, NATIVE_HALF_W)
    assert draw is not None, (unit_const, rotation)
    return [unit_sprites.SpritePiece(draw=draw, dx=0, dy=0)]


def _composite(unit_const: int, rotation: float, team: int):
    return unit_sprites.sprite_pieces_for(unit_const, rotation, team, NATIVE_HALF_W)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_galleon_sails_rise_above_its_hull(rotation):
    _require_install()
    hull_top = _ink_top(_hull_only(GALLEON, rotation, RED))
    composite = _composite(GALLEON, rotation, RED)
    assert len(composite) >= 2, f"Galleon drew {len(composite)} piece(s) at {rotation}: no sails"
    rise = hull_top - _ink_top(composite)
    assert rise >= MIN_GALLEON_SAIL_RISE, (
        f"Galleon's ink top is only {rise} px above its bare hull at rotation {rotation}"
    )


@pytest.mark.parametrize("unit_const", BARE_HULL_CONSTS)
def test_a_bare_hulled_ship_shows_its_owner_on_its_sails(unit_const):
    """The "player colour lives in the sails" claim, measured: the hull alone
    takes no tint at all, the composite does."""
    _require_install()
    for rotation in ROTATIONS:
        assert _tinted_px(unit_const, rotation, _hull_only) == 0, (unit_const, rotation)
        tinted = _tinted_px(unit_const, rotation, _composite)
        # The least across these ships and facings is Fire Ship's 232.
        assert tinted >= 150, (unit_const, rotation, tinted)


@pytest.mark.parametrize("unit_const", BARE_HULL_CONSTS)
def test_a_bare_hulled_ships_flat_icon_shows_its_owner_too(unit_const):
    """Flat assembles the same pieces into one icon, so the sails' colour reaches it."""
    _require_install()
    for rotation in ROTATIONS[:4]:
        red, plain = (unit_sprites.icon_for(unit_const, rotation, team, 64, 64) for team in (RED, 0))
        assert red is not None and red.rgba.shape == plain.rgba.shape, (unit_const, rotation)
        # The least across these ships and facings is Fire Ship's 24.
        assert int((red.rgba != plain.rgba).any(axis=2).sum()) >= 10, (unit_const, rotation)


@pytest.mark.parametrize("unit_const", SHIP_SAIL_CONSTS)
def test_every_sailing_ship_draws_a_sail_at_every_facing(unit_const):
    """Each sail graphic holds ink for half the facings only, so a ship's sail
    set changes with its angle. No facing may leave the hull bare."""
    _require_install()
    entry = unit_sprites.graphic_map()[unit_const]
    for rotation in ALL_FACINGS:
        pieces = _composite(unit_const, rotation, RED)
        assert 2 <= len(pieces) <= len(entry["pieces"]), (unit_const, rotation, len(pieces))
        # The hull paints first, unmoved, and every sail sits above the anchor.
        (hull,) = _hull_only(unit_const, rotation, RED)
        assert np.array_equal(pieces[0].draw.rgba, hull.draw.rgba) and (pieces[0].dx, pieces[0].dy) == (0, 0)
        assert all(sail.dx == 0 and sail.dy < 0 for sail in pieces[1:]), (unit_const, rotation)


def test_every_sail_file_holds_one_frame_run_per_angle():
    """Each sail dispatches angle_index * frame_count on its OWN frame_count, so
    the file must really hold angle_count runs of that length."""
    _require_install()
    sails = {
        p["file_name"]: p
        for const in SHIP_SAIL_CONSTS
        for p in unit_sprites.graphic_map()[const]["pieces"]
        if not p.get("parent")
    }
    assert len(sails) == 15, sorted(sails)
    for name, piece in sails.items():
        frames = unit_sprites.sld_frame_count(name)
        assert frames == piece["angle_count"] * piece["frame_count"] == 480, (name, frames)


def test_fishing_ship_is_still_one_graphic():
    """The control: a ship with no sail deltas draws exactly what it drew before."""
    _require_install()
    for rotation in ROTATIONS:
        (piece,) = _composite(FISHING_SHIP, rotation, RED)
        single = unit_sprites.sprite_for(FISHING_SHIP, rotation, RED, NATIVE_HALF_W)
        assert (piece.dx, piece.dy) == (0, 0)
        assert np.array_equal(piece.draw.rgba, single.rgba)
