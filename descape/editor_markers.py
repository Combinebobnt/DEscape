"""Editor-only markers for the objects the game never draws (GH #53 Part B).

Invisible Object A-E, Map Revealers, Blockers and the rest of
unit_kind.invisible_consts() have no standing graphic, so without this they
fall back to the coloured box every unmapped unit gets. unit_sprites hands
them one of these instead, through the same sprite seam real art uses, so
depth order, chunk seams, splices, the drag ghost and Flat's icons come for
free.

numpy + Pillow + asset_source only, never unit_sprites (which imports this),
so there is no cycle. This module builds the untinted layers; the team tint
and the SpriteDraw wrapper live in unit_sprites.marker_for(), next to the
_tinted() multiply real sprites go through.

**Where the art comes from.** The game ships exactly one usable image:
VISIBILITY_ICON_SUBPATH, the lobby's visibility-setting eye, used for
revealers when an install is configured. Nothing else fits (searched
2026-09-21): the scenario-editor atlas holds only panels and buttons, "Erase
Invisible Objects" is a plain checkbox, none of these consts has an icon, and
drs/graphics has no invisible, revealer or blocker art. Every other category,
and the revealer without an install, gets a glyph generated here. Game art is
read from the user's install at runtime and never bundled.

**Layout.** One tile's diamond (2*half_w wide, half_w tall, hotspot at its
centre), filled mid-grey and semi-opaque with an opaque ring, both marked in
the coverage layer so the owner's colour multiplies onto them the way it does
onto a sprite's player-colour mask. The symbol sits upright on top, white
with a dark outline and outside the coverage, so it reads the same for every
owner. Generated glyphs are drawn at _SUPERSAMPLE x and LANCZOS-downsampled,
which is deterministic: the same inputs give the same bytes.

Below MIN_SYMBOL_PX of symbol height a glyph is just noise, so the marker is
the plain badge alone there.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from descape import asset_source

# Install-relative; the lobby's visibility-setting icon (112x112 RGBA, an eye inside a diamond).
VISIBILITY_ICON_SUBPATH = "resources/_common/wpfg/resources/Icons/visibility.png"

CATEGORIES = ("invisible", "revealer", "blocker", "other")

MIN_SYMBOL_PX = 8

_SUPERSAMPLE = 4
_BADGE_GREY = 220
_BADGE_ALPHA = 160
_RING_GREY = 150
_OUTLINE = (24, 24, 24, 255)
_SYMBOL = (255, 255, 255, 255)

# Symbol box as a fraction of the diamond's height (half_w): it must sit inside one tile.
_SYMBOL_H = 0.72
_SYMBOL_W = 1.2


def clear_caches() -> None:
    """Called from unit_sprites.clear_caches(), which an install change reaches."""
    marker_layers.cache_clear()
    _visibility_icon.cache_clear()


def _diamond(w: int, h: int, inset: float) -> list[tuple[float, float]]:
    cx, cy = w / 2, h / 2
    return [(cx, inset), (w - 2 * inset, cy), (cx, h - inset), (2 * inset, cy)]


def _stroke(h: int) -> int:
    return max(_SUPERSAMPLE, round(h * 0.13))


def _eye(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> None:
    """A filled white almond with the pupil cut out, so the outline shows through it dark."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    eh = (y1 - y0) * 0.72
    draw.ellipse((x0, cy - eh / 2, x1, cy + eh / 2), fill=255)
    r = eh * 0.3
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=0)


def _symbol_mask(category: str, w: int, h: int) -> Image.Image:
    """The symbol's strokes as an L mask of size (w, h), supersampled."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    lw = _stroke(h)
    pad = lw
    box = (pad, pad, w - pad, h - pad)
    if category in ("invisible", "revealer"):
        _eye(draw, box)
        if category == "invisible":
            # The slash, with a gap cut either side so it reads as crossing the eye; off-centre so the pupil shows.
            gap = Image.new("L", (w, h), 0)
            cx, cy = w / 2 + h * 0.5, h / 2
            dx, dy = h * 0.55, h / 2 - pad
            slash = (cx + dx, cy - dy, cx - dx, cy + dy)
            ImageDraw.Draw(gap).line(slash, fill=255, width=round(lw * 2.6))
            mask.paste(0, (0, 0), gap)
            draw.line(slash, fill=255, width=lw)
    elif category == "blocker":
        side = h - 2 * pad
        sx0 = (w - side) / 2
        sq = (sx0, pad, sx0 + side, pad + side)
        hatch = Image.new("L", (w, h), 0)
        hd = ImageDraw.Draw(hatch)
        step = max(lw * 2, side / 3.2)
        k = -side
        while k < side * 2:
            hd.line((sx0 + k, pad + side, sx0 + k + side, pad), fill=255, width=max(1, round(lw * 0.7)))
            k += step
        clip = Image.new("L", (w, h), 0)
        ImageDraw.Draw(clip).rectangle(sq, fill=255)
        mask.paste(hatch, (0, 0), clip)
        draw.rectangle(sq, outline=255, width=lw)
        draw.line((sq[0], sq[1], sq[2], sq[3]), fill=255, width=lw)
        draw.line((sq[0], sq[3], sq[2], sq[1]), fill=255, width=lw)
    else:
        # A dashed diamond with a question mark drawn from shapes (no font dependency).
        pts = _diamond(w, h, pad)
        dash = max(1, round(lw * 0.7))
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1], strict=True):
            for t0 in (0.08, 0.58):
                t1 = t0 + 0.34
                draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1), fill=255, width=dash)
        cx = w / 2
        r = h * 0.17
        top = h * 0.24
        draw.arc((cx - r, top, cx + r, top + 2 * r), start=180, end=90, fill=255, width=lw)
        draw.line((cx, top + 2 * r, cx, h * 0.64), fill=255, width=lw)
        d = lw * 0.7
        draw.ellipse((cx - d, h * 0.74 - d, cx + d, h * 0.74 + d), fill=255)
    return mask


def _outlined(mask: Image.Image, lw: int) -> Image.Image:
    """White strokes over a dark outline one stroke-width wider, as RGBA."""
    grow = lw | 1
    halo = mask.filter(ImageFilter.MaxFilter(grow if grow >= 3 else 3))
    out = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    out.paste(Image.new("RGBA", mask.size, _OUTLINE), (0, 0), halo)
    out.paste(Image.new("RGBA", mask.size, _SYMBOL), (0, 0), mask)
    return out


@lru_cache(maxsize=4)
def _visibility_icon(install: str | None) -> Image.Image | None:
    """visibility.png from this install as RGBA, or None if it can't be read."""
    if install is None:
        return None
    try:
        with Image.open(f"{install}/{VISIBILITY_ICON_SUBPATH}") as img:
            return img.convert("RGBA")
    except (OSError, ValueError):
        return None


def _contain(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


@lru_cache(maxsize=256)
def marker_layers(category: str, half_w: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(main RGBA, coverage RGBA, hotspot_x, hotspot_y) for one category at this
    half_w, untinted. `coverage` has unit_sprites' playercolor layout: alpha > 0
    where the owner's colour applies, channel 0 its strength."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown marker category {category!r}")
    w, h = 2 * max(1, half_w), max(1, half_w)
    ss = _SUPERSAMPLE
    W, H = w * ss, h * ss

    badge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cover = Image.new("L", (W, H), 0)
    ring = max(ss, 2 * ss if h >= 12 else ss)
    poly = _diamond(W, H, ss * 0.5)
    ImageDraw.Draw(badge).polygon(poly, fill=(_BADGE_GREY,) * 3 + (_BADGE_ALPHA,))
    ImageDraw.Draw(badge).line(poly + poly[:1], fill=(_RING_GREY,) * 3 + (255,), width=ring, joint="curve")
    ImageDraw.Draw(cover).polygon(poly, fill=255)
    ImageDraw.Draw(cover).line(poly + poly[:1], fill=255, width=ring, joint="curve")

    sym_h = round(h * _SYMBOL_H)
    if sym_h >= MIN_SYMBOL_PX:
        sym_w = round(h * _SYMBOL_W)
        ox, oy = (W - sym_w * ss) // 2, (H - sym_h * ss) // 2
        art = _visibility_icon(_install_str()) if category == "revealer" else None
        if art is not None:
            fitted = _contain(art, sym_w * ss, round(h * 0.9) * ss)
            px, py = (W - fitted.width) // 2, (H - fitted.height) // 2
            badge.alpha_composite(fitted, (px, py))
            cover.paste(0, (px, py), fitted.getchannel("A"))
        else:
            mask = _symbol_mask(category, sym_w * ss, sym_h * ss)
            symbol = _outlined(mask, _stroke(sym_h * ss))
            badge.alpha_composite(symbol, (ox, oy))
            cover.paste(0, (ox, oy), symbol.getchannel("A"))

    main = np.asarray(badge.resize((w, h), Image.LANCZOS), dtype=np.uint8).copy()
    # LANCZOS leaves near-zero-alpha fringe pixels with arbitrary RGB; drop them.
    main[main[..., 3] < 8] = 0
    cov = np.asarray(cover.resize((w, h), Image.LANCZOS), dtype=np.uint8)
    cov = np.where(main[..., 3] > 0, cov, 0).astype(np.uint8)
    coverage = np.zeros((h, w, 4), dtype=np.uint8)
    coverage[..., 0] = cov
    coverage[..., 3] = np.where(cov > 0, 255, 0)
    main.setflags(write=False)
    coverage.setflags(write=False)
    return main, coverage, w // 2, h // 2


def _install_str() -> str | None:
    path = asset_source.get_install_path()
    return None if path is None else str(path)
