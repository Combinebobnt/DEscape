#!/usr/bin/env python3
"""Measures each composite piece's depth slot: which footprint tile's moment in
the depth walk should paint it, so a unit standing inside a town centre can
paint between the building's back and front pieces.

Informational, like scan_sprite_reach.py: exits 0 with a message when no
AoE2:DE install is configured. Its output is what `_PIECE_SLOTS` in
tools/gen_unit_graphic_map.py is committed from, and tests/test_unit_graphic_map.py's
corpus-marked slot test re-runs `measure()` to keep that table honest.

## The slot rule, per piece, in the entry's authored (depth-sorted) order

1. **Derived slot.** The piece unit's own footprint, centred on the parent
   footprint's centre plus the piece's raw misplacement `(mx, my)` and sized by
   the piece unit's own span, gives an anchor tile via
   `unit_sprites.sprite_anchor_tile()`. It is clamped into the parent footprint:
   a piece anchored outside it would let terrain beyond the building paint
   over its feet.
2. **Push forward** to the piece's minimum safe slot. Every tile whose terrain
   diamond touches the piece's opaque ink, inside the footprint or not, must
   paint before the piece, so each raises the slot to the earliest footprint
   tile at or after it in depth order. Needs the real art, which is why this
   is measured offline and committed, never computed at render time.
3. **Order pass.** For every pair i < j whose ink boxes overlap on screen,
   slot[i] <= slot[j], raising the later piece as little as possible. Pairwise,
   not a sweep over the whole list: a pasture's five pieces never overlap, and
   a global sweep would collapse its four distinct corner slots into two.

Ink is the union over every stored angle's first frame, since the frame drawn
depends on the placed unit's rotation. Measured on a flat footprint at native
scale (half_w=48, half_h=24) with the parent's origin at tile (0, 0), so a
resolved slot is directly the `[sx, sy]` offset `_PIECE_SLOTS` stores.

Tiles scanned for step 2 are every tile touching ink, not just the footprint:
a pasture corner post sits on a footprint corner, and a footprint-only scan
slots two posts early enough that neighbouring grass erases 11-18 px of their
feet. An out-of-footprint tile that still paints after the latest footprint
tile is terrain no slot can fix; those are counted and printed, and today's
single-anchor placement has exactly the same ones.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import asset_source, iso_geometry, render, unit_sprites

HALF_W = unit_sprites.NATIVE_TILE_W // 2
HALF_H = HALF_W // 2


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "gen_unit_graphic_map", ROOT / "tools" / "gen_unit_graphic_map.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _depth_key(tile: tuple[int, int]) -> tuple[int, int]:
    """The same (d, x) key sprite_anchor_tile() and depth_order() sort on."""
    return tile[1] - tile[0], tile[0]


def _later(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a if _depth_key(a) >= _depth_key(b) else b


def _ink(piece: dict):
    """(mask, x0, y0) of the piece's union ink over all stored angles, in
    hotspot-relative native px, or None if no frame decodes."""
    stored = unit_sprites.sld_frame_count(piece["file_name"])
    if stored is None:
        return None
    frames = []
    for index in range(0, stored, max(1, int(piece["frame_count"]))):
        native = unit_sprites._native_frame(piece["file_name"], index)
        if native is not None:
            main, _pc, hx, hy = native
            frames.append((main[..., 3] > 0, -hx, -hy))
    if not frames:
        return None
    x0 = min(fx for _m, fx, _fy in frames)
    y0 = min(fy for _m, _fx, fy in frames)
    x1 = max(fx + m.shape[1] for m, fx, _fy in frames)
    y1 = max(fy + m.shape[0] for m, _fx, fy in frames)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for m, fx, fy in frames:
        mask[fy - y0 : fy - y0 + m.shape[0], fx - x0 : fx - x0 + m.shape[1]] |= m
    return mask, x0, y0


def _touching_tiles(mask: np.ndarray, sx0: int, sy0: int) -> set[tuple[int, int]]:
    """Every tile whose flat diamond covers at least one opaque ink pixel; the
    ink's top-left sits at screen (sx0, sy0) with tile (0, 0)'s bbox at (0, 0)."""
    ys, xs = np.nonzero(mask)
    px = xs + sx0
    py = ys + sy0
    # Candidate tiles from each pixel's two nearest diamond rows, then an exact test.
    u = np.floor_divide(px, HALF_W)
    v = np.floor_divide(py, HALF_H)
    touched: set[tuple[int, int]] = set()
    candidates = set()
    for a, b in set(zip(u.tolist(), v.tolist(), strict=True)):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                s, t = a + da, b + db  # s = x + y, t = y - x
                if (s + t) % 2 == 0:
                    candidates.add(((s - t) // 2, (s + t) // 2))
                else:
                    candidates.add(((s - t - 1) // 2, (s + t - 1) // 2))
                    candidates.add(((s - t + 1) // 2, (s + t + 1) // 2))
    for tx, ty in candidates:
        ox = (tx + ty) * HALF_W
        oy = (ty - tx) * HALF_H
        local_x = px - ox
        local_y = py - oy
        inside = (local_x >= 0) & (local_x < 2 * HALF_W) & (local_y >= 0) & (local_y < 2 * HALF_H)
        if inside.any() and iso_geometry.diamond_membership(
            local_x[inside], local_y[inside], HALF_W, HALF_H
        ).any():
            touched.add((tx, ty))
    return touched


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def measure(unit_const: int) -> dict | None:
    """The per-piece measurement for one composite const, or None if it has no
    pieces or any piece's art is unreadable on this install."""
    entry = unit_sprites.graphic_map().get(unit_const)
    if entry is None or "pieces" not in entry:
        return None
    span_x, span_y = render.tile_span(unit_const, render.NON_BUILDING_SPAN)
    footprint = {(x, y) for x in range(span_x) for y in range(span_y)}
    order = sorted(footprint, key=_depth_key)
    cx, cy = span_x / 2, span_y / 2
    hot_x = round((cx + cy) * HALF_W)
    hot_y = round((cy - cx) * HALF_H) + HALF_H

    rows = []
    for piece in entry["pieces"]:
        ink = _ink(piece)
        if ink is None:
            return None
        mask, ix0, iy0 = ink
        sx0 = hot_x + int(piece["dx"]) + ix0
        sy0 = hot_y + int(piece["dy"]) + iy0
        box = (sx0, sy0, sx0 + mask.shape[1], sy0 + mask.shape[0])

        p_span_x, p_span_y = render.tile_span(int(piece["unit_id"]), render.NON_BUILDING_SPAN)
        px0 = render._span_start(cx + float(piece["mx"]), p_span_x)
        py0 = render._span_start(cy + float(piece["my"]), p_span_y)
        own = [(x, y) for x in range(px0, px0 + p_span_x) for y in range(py0, py0 + p_span_y)]
        dx_, dy_ = unit_sprites.sprite_anchor_tile(own)
        derived = (min(max(dx_, 0), span_x - 1), min(max(dy_, 0), span_y - 1))

        touched = _touching_tiles(mask, sx0, sy0)
        minimum = None
        for tile in touched:
            covering = [t for t in order if _depth_key(t) >= _depth_key(tile)]
            need = covering[0] if covering else order[-1]
            minimum = need if minimum is None else _later(minimum, need)
        pushed = derived if minimum is None else _later(derived, minimum)
        rows.append({
            "file_name": piece["file_name"],
            "derived": derived,
            "minimum": minimum,
            "pushed": pushed,
            "box": box,
            "touched": touched,
        })

    for j, row in enumerate(rows):
        slot = row["pushed"]
        for i in range(j):
            if _boxes_overlap(rows[i]["box"], row["box"]):
                slot = _later(slot, rows[i]["slot"])
        row["slot"] = slot
        row["outside"] = sum(
            1 for t in row["touched"] - footprint if _depth_key(t) > _depth_key(slot)
        )

    keys = [_depth_key(r["slot"]) for r in rows]
    window = [t for t in order if min(keys) < _depth_key(t) < max(keys)]
    return {"span": (span_x, span_y), "order": order, "rows": rows, "window": window}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--install", type=Path, help="AoE2:DE install root override")
    args = parser.parse_args()

    if args.install:
        asset_source.set_install_path_override(args.install)
    install = asset_source.get_install_path()
    if install is None:
        print(
            "No AoE2:DE install configured, so there is no piece art to "
            "measure. Configure one (see README.md) or pass --install. "
            "Informational tool: this is not a failure."
        )
        return

    generator = _generator_module()
    committed = getattr(generator, "_PIECE_SLOTS", {})
    measured: dict[int, list[list[int]]] = {}
    print(f"Piece slot measurement -- install {install}")
    for unit_const in sorted(generator._COMPOSITE_SCOPE):
        result = measure(unit_const)
        if result is None:
            print(f"\n{unit_const}: no pieces or unreadable art, skipped")
            continue
        order = result["order"]
        print(f"\n{unit_const} span {result['span']}  (slot = depth index in the footprint)")
        for row in result["rows"]:
            minimum = "none" if row["minimum"] is None else order.index(row["minimum"])
            print(
                f"  {row['file_name']:<42} derived {order.index(row['derived']):>2}  "
                f"min {minimum!s:>4}  pushed {order.index(row['pushed']):>2}  "
                f"slot {order.index(row['slot']):>2} {row['slot']}  "
                f"out-of-footprint later tiles touching ink: {row['outside']}"
            )
        window = result["window"]
        print(f"  sandwich window: {len(window)} tile(s) {window}")
        measured[unit_const] = [list(row["slot"]) for row in result["rows"]]

    print("\n_PIECE_SLOTS = {")
    for unit_const, slots in measured.items():
        print(f"    {unit_const}: {slots},")
    print("}")
    mismatched = sorted(c for c in measured if committed.get(c) != measured[c])
    if mismatched:
        print(f"\nDIFFERS FROM THE COMMITTED _PIECE_SLOTS for: {mismatched}")
    else:
        print("\nEvery measured slot matches the committed _PIECE_SLOTS.")


if __name__ == "__main__":
    main()
