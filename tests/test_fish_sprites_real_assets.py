"""Every fish paints a real body at its stored rotation (GH #178).

A fish's standing graphic is the surface splash, whose frame 0 is empty for
Dorado, Snapper, Perch, Dolphin, Marlin and Whale (no sprite, the coloured
mark instead) and a 17-229 px sliver for Salmon, Tuna, Shore Fish and Box
Turtles. The graphic-map tests pin which file each const names; only a real
`.sld` decode can say whether that file paints, so this is the frame-selection
guard maintainer notes ask for in the default tier.

Needs AOE2DE_INSTALL_PATH, like tests/test_wall_variants_real_assets.py:
conftest._isolated_settings hides config.yaml. Not `corpus`-marked for the
same reason as that file: a handful of `sprite_for()` calls on disk files.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, unit_sprites

# The fewest opaque px any fish body has at native scale is Shore Fish's 102;
# every splash this replaced but Salmon's (229) is under 50.
MIN_BODY_INK = 90

# unit_const -> rotations to check: Dorado's two variants, one pose elsewhere.
FISH = {53: (0.0,), 61: (0.0,), 69: (0.0,), 450: (0.0,), 451: (0.0,), 455: (0.0, 1.0),
        456: (0.0,), 457: (0.0,), 458: (0.0,), 1141: (0.0,), 2625: (0.0,)}


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


@pytest.mark.parametrize("unit_const", sorted(FISH))
def test_every_fish_draws_its_body(unit_const):
    _require_install()
    entry = unit_sprites.graphic_map()[unit_const]
    assert entry["file_name"].endswith("_underwater_x1"), entry["file_name"]
    for rotation in FISH[unit_const]:
        draw = unit_sprites.sprite_for(unit_const, rotation, 0, 48)
        assert draw is not None, (
            f"{unit_const} ({entry['file_name']}) draws nothing at rotation {rotation}: "
            "it falls back to the coloured mark"
        )
        ink = int((draw.rgba[..., 3] > 0).sum())
        assert ink >= MIN_BODY_INK, (unit_const, rotation, ink)


PERCH = 53
# sprite_scale() == 1.0, so rows are native px. A 2x2 footprint's near corner
# sits 2 * (half_w / 2) = half_w px below the anchor, its far corner as far above.
NATIVE_HALF_W = unit_sprites.NATIVE_TILE_W // 2


def _ink_rows_below_anchor(rgba, hotspot_y: int, dy: int):
    return np.nonzero(rgba[..., 3] > 0)[0] - hotspot_y + dy


def _assert_perch_straddles_its_anchor(rows):
    # Measured 2026-10-09: with the .dat delta's (0, -30) the ink spans rows
    # -23..41 (mean 2.6); dropped, it is 7..71 (mean 32.6), past the near corner.
    assert abs(rows.mean()) <= 10, rows.mean()
    assert rows.max() < NATIVE_HALF_W, rows.max()
    assert rows.min() < 0, rows.min()


def test_perch_body_draws_at_its_dat_delta_offset():
    """Perch's body hangs at .dat delta offset (0, -30) off its splash. Iso
    sprites, sprite_for() and Flat's native walk all fold it into the hotspot."""
    _require_install()
    entry = unit_sprites.graphic_map()[PERCH]
    (piece,) = unit_sprites.sprite_pieces_for(PERCH, 0.0, 0, NATIVE_HALF_W)
    _assert_perch_straddles_its_anchor(
        _ink_rows_below_anchor(piece.draw.rgba, piece.draw.hotspot_y, piece.dy)
    )
    single = unit_sprites.sprite_for(PERCH, 0.0, 0, NATIVE_HALF_W)
    assert (single.hotspot_x, single.hotspot_y) == (piece.draw.hotspot_x, piece.draw.hotspot_y)
    (native,) = unit_sprites._native_pieces_for(PERCH, entry, 0.0)
    _assert_perch_straddles_its_anchor(
        _ink_rows_below_anchor(native.main, native.hotspot_y, native.dy)
    )
