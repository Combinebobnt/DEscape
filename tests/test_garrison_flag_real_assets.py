"""A garrisoned host's flag, drawn from the real game art (GH #203).

tests/test_unit_graphic_map.py pins which flag files and offsets the tables
name; only a real `.sld` decode says whether each file has ink at frame 0,
takes the owner's colour, and stands on the building at the .dat's offset.
That last claim is the plan's anchor assumption: a delta offset counts from
the anchor the body draws at, measured here for the two composite-heavy
cases, Town Center 109 (four pieces) and Barracks 12 (two flags).

Measured on the install 2026-10-22 at native scale (half_w 48): all 14 flag
files hold 90 frames and draw 463-907 ink px at frame 0, 201-689 of them
tinted. Every flag's hotspot is its pole's foot. Town Center 109's body ink
spans x -189..188, y -176..50 from the anchor, its flag x -31..4, y
-213..-157, the foot on body ink (41 of the 7x7 px around it). Barracks 12
spans x -148..141, y -133..71, its flags x 84..119 and -7..28, both feet on
body ink (49 of 49).

Needs AOE2DE_INSTALL_PATH, like tests/test_ship_sails_real_assets.py:
conftest._isolated_settings hides config.yaml. Not `corpus`-marked: a few
dozen frame decodes from files on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, unit_sprites

TOWN_CENTER, BARRACKS = 109, 12
NATIVE_HALF_W = unit_sprites.NATIVE_TILE_W // 2
GAIA, RED = 0, 2
FLAG_FILE_COUNT = 14
# Opaque body px a flag's foot must stand on, of the 7x7 px around it.
MIN_FOOT_INK = 30


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


def _flag_items() -> dict[str, dict]:
    """One garrison_flags item per distinct flag file, over both tables."""
    art = unit_sprites.building_art()
    entries = [*unit_sprites.graphic_map().values(), *(e for civ in art.civ_art.values() for e in civ.values())]
    items: dict[str, dict] = {}
    for entry in entries:
        for flag in entry.get("garrison_flags", ()):
            items.setdefault(flag["file_name"], flag)
    return items


def _ink_box(pieces) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of every piece's opaque px, from the unit's anchor."""
    boxes = []
    for p in pieces:
        ys, xs = np.nonzero(p.draw.rgba[..., 3] > 0)
        ox, oy = p.dx - p.draw.hotspot_x, p.dy - p.draw.hotspot_y
        boxes.append((xs.min() + ox, ys.min() + oy, xs.max() + 1 + ox, ys.max() + 1 + oy))
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _body_ink(pieces) -> tuple[np.ndarray, int, int]:
    """The body pieces' opaque px on one canvas, and the canvas origin from the anchor."""
    x0, y0, x1, y1 = _ink_box(pieces)
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for p in pieces:
        ys, xs = np.nonzero(p.draw.rgba[..., 3] > 0)
        canvas[ys + p.dy - p.draw.hotspot_y - y0, xs + p.dx - p.draw.hotspot_x - x0] = True
    return canvas, x0, y0


def test_every_flag_file_draws_ink_at_frame_zero_in_its_owners_colour():
    _require_install()
    items = _flag_items()
    assert len(items) == FLAG_FILE_COUNT, sorted(items)
    for name, flag in items.items():
        red = unit_sprites._draw_for_entry(flag["unit_id"], flag, 0.0, RED, NATIVE_HALF_W)
        gaia = unit_sprites._draw_for_entry(flag["unit_id"], flag, 0.0, GAIA, NATIVE_HALF_W)
        assert red is not None and gaia is not None, name
        # The least across the 14 files is b_east's 463 ink px and 201 tinted.
        assert int((red.rgba[..., 3] > 0).sum()) >= 400, name
        assert int((red.rgba != gaia.rgba).any(axis=2).sum()) >= 150, name


@pytest.mark.parametrize(("const", "flags"), [(TOWN_CENTER, 1), (BARRACKS, 2)], ids=["town_center", "barracks"])
def test_a_flag_stands_on_its_buildings_ink_at_the_dat_offset(const, flags):
    """The foot of each flag pole lands on the body's own ink, and the flag
    stays inside the body's horizontal extent."""
    _require_install()
    body = unit_sprites.sprite_pieces_for(const, 0.0, RED, NATIVE_HALF_W)
    pieces = unit_sprites.garrison_flag_pieces_for(const, RED, NATIVE_HALF_W)
    assert len(pieces) == flags
    canvas, x0, y0 = _body_ink(body)
    bx0, _by0, bx1, _by1 = _ink_box(body)
    for piece in pieces:
        fx0, _fy0, fx1, fy1 = _ink_box([piece])
        assert bx0 <= fx0 and fx1 <= bx1, (const, (fx0, fx1), (bx0, bx1))
        # The pole's foot is the flag's lowest ink row, at the .dat offset.
        assert fy1 - 1 <= piece.dy <= fy1, (const, piece.dy, fy1)
        foot_x, foot_y = piece.dx - x0, piece.dy - y0
        around = canvas[foot_y - 3 : foot_y + 4, foot_x - 3 : foot_x + 4]
        assert int(around.sum()) >= MIN_FOOT_INK, (const, (piece.dx, piece.dy), int(around.sum()))
