"""TASK-291: a unit ghost draws the mip level the canvas paints.

descape/ghost_sprite.py's per-level memo and entries, the leveled caches'
level_sprite_inputs(), and UnitGhostItem's level path: which level it picks,
what it draws there, its bounding rect and its memo. The ghost-against-the-
placed-sprite pixel test is tests/test_place_preview.py's
test_the_ghost_sits_on_the_placed_sprite_at_every_level.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_place_preview import _hover_tile, _place_at, level_install, level_window, sprite_install  # noqa: F401
from test_sloped_mip import _cache as sloped_cache
from test_sloped_mip import _scenario

from descape import ghost_sprite, render_iso, render_sprites
from descape.render_cache import IsoChunkCache
from testkit.render_family import set_render_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def counting_resolver(monkeypatch, delegate=None) -> list:
    """Replaces unit_sprite_draws_at() with a call-counting fake (the real one, or a fresh one-piece list)."""
    real = render_sprites.unit_sprite_draws_at if delegate is None else delegate
    calls: list = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    set_render_global(monkeypatch, "unit_sprite_draws_at", fake)
    return calls


def _one_piece(*_args, **_kwargs) -> list:
    return [(SimpleNamespace(rgba=np.zeros((2, 3, 4), np.uint8), hotspot_x=1, hotspot_y=1), 10, 20)]


# --- level_sprite_inputs() -------------------------------------------------------


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_every_levels_sprite_inputs_are_its_own_proj_and_rise(style) -> None:
    scn = _scenario(units=False)
    if style == "Sloped":
        cache = sloped_cache(scn)
    else:
        elevations, proj = render_iso.elevations_and_proj(scn)
        cache = IsoChunkCache(scn, elevations, proj, proj.tile_px)
    assert len(cache.mip_levels()) > 1
    for mip in cache.mip_levels():
        proj, rise = cache.level_sprite_inputs(mip)
        assert proj is cache._levels[mip].proj
        assert proj.tile_px == cache.mip_tile_px(mip)
        if style == "Sloped":
            assert rise is cache._level_rise(cache._levels[mip]).corner_rise
            assert (rise is cache.corner_rise) == (mip == 0)
        else:
            assert rise is None
    # Geometry only: no level's unit layers were built.
    assert all(lvl.sprites is None and lvl.unit_pack is None for lvl in cache._levels.values())


# --- GhostSprite and level_entries() ---------------------------------------------


def test_a_level_resolves_once_per_proj_and_rise_with_the_captured_inputs(monkeypatch) -> None:
    calls = counting_resolver(monkeypatch, _one_piece)
    member = ghost_sprite.GhostSprite(
        "scn", "elev", 3, "unit", rotation_override=1.5, tree_scale=0.5, hero_glow=True, wall_alpha=0.4, fold="fold"
    )
    assert member.level0() == []
    proj, other = object(), object()
    first = member.draws(1, proj, None)
    assert member.draws(1, proj, None) is first
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("scn", proj, "elev", None, 3, "unit")
    assert kwargs == {"rotation_override": 1.5, "tree_scale": 0.5, "hero_glow": True, "wall_alpha": 0.4, "fold": "fold"}
    member.draws(1, other, None)
    member.draws(1, other, np.zeros(1))
    assert len(calls) == 3, "another proj or corner_rise object is another resolve"
    base = member.draws(0, proj, None)
    assert member.level0() is base


class _Cache:
    """The leveled cache's mip surface: levels -2..1 at 16..96 px over a 64 px reference."""

    def __init__(self) -> None:
        self.TILE_PX = {-2: 16, -1: 32, 0: 64, 1: 96}
        self.inputs = {mip: (object(), None) for mip in self.TILE_PX}

    def mip_levels(self) -> list[int]:
        return sorted(self.TILE_PX)

    def mip_scale(self, mip: int = 0) -> float:
        return 64 / self.TILE_PX[mip]

    def mip_for_scale(self, scale: float) -> int:
        return next((m for m in self.mip_levels() if self.TILE_PX[m] >= scale * 64), self.mip_levels()[-1])

    def level_sprite_inputs(self, mip: int):
        return self.inputs[mip]


class _Member:
    """GhostSprite's surface over fixed per-level draws, counting each level asked."""

    def __init__(self, by_level: dict[int, list]) -> None:
        self.by_level = by_level
        self.asked: list[int] = []

    def draws(self, mip: int, proj, corner_rise) -> list:
        self.asked.append(mip)
        return self.by_level.get(mip, [])

    def level0(self) -> list:
        return self.by_level[0]


def _piece(w: int, h: int, hotspot=(0, 0), rgb=(255, 255, 255)):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = rgb, 255
    return SimpleNamespace(rgba=rgba, hotspot_x=hotspot[0], hotspot_y=hotspot[1])


def test_level_entries_take_each_levels_corner_and_scale_and_fall_back_to_level_0() -> None:
    cache = _Cache()
    a = _piece(4, 2, hotspot=(1, 1))
    resolved = _Member({0: [(_piece(6, 3), 30, 40)], 1: [(a, 45, 60)]})
    pixelless = _Member({0: [(_piece(2, 2), 7, 8)], 1: [(_piece(0, 0), 1, 1)]})
    c = _piece(2, 2)
    blank = _Member({0: [(c, 5, 5)]})
    got = ghost_sprite.level_entries([(resolved, None), (pixelless, "refused"), (blank, "garrison")], cache, 1)
    # A level piece with no pixels draws nothing; a level that resolves to nothing keeps level 0, at scale 1.
    assert got == [(a, 44.0, 59.0, 64 / 96, None), (c, 5.0, 5.0, 1.0, "garrison")]


def test_the_pad_holds_the_measured_overhang_at_the_coarsest_level() -> None:
    # Step 0 measured up to 14 scene px at mip_scale 4 (the hero glow plus rounding) over every graphic_map() const.
    from descape import unit_sprites

    assert unit_sprites.HERO_GLOW_RADIUS == 2
    assert ghost_sprite.level_pad(_Cache()) == 17.0 > 14 + 1


# --- UnitGhostItem's level path ---------------------------------------------------


def paint_at(item, scale: float):
    """item.paint() into an ARGB image of its bounding rect at device `scale`."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPainter

    rect = item.boundingRect()
    image = QImage(int(rect.width() * scale) + 2, int(rect.height() * scale) + 2, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.scale(scale, scale)
    painter.translate(-rect.left(), -rect.top())
    item.paint(painter, None)
    painter.end()
    return image


def _item():
    conftest.ensure_qapp()
    from descape.viewer_canvas import UnitGhostItem

    return UnitGhostItem()


def test_the_item_draws_the_canvas_level_and_pads_its_rect() -> None:
    cache = _Cache()
    piece = _piece(6, 3)
    member = _Member({0: [(piece, 30, 40)], -1: [(_piece(3, 2), 16, 21)]})
    item = _item()
    assert item.set_content([(piece, 30, 40)], (), sprites=[(member, None)], cache=cache)
    pad = ghost_sprite.level_pad(cache)
    assert item.boundingRect().getRect() == (30 - pad, 40 - pad, 6 + 2 * pad, 3 + 2 * pad)

    paint_at(item, 0.8)
    assert member.asked == [] and item._level_images == {}, "a level-0 paint resolves no other level"
    paint_at(item, 0.5)
    paint_at(item, 0.45)
    assert member.asked == [-1], "one resolve per level and content"
    ((image, _source, _hold, x, y, s),) = item._level_images[-1]
    assert (x * s, y * s, image.width() * s, image.height() * s) == (32.0, 42.0, 6.0, 4.0)
    assert item.boundingRect().contains(32.0, 42.0) and item.boundingRect().contains(37.9, 45.9)

    item.set_content([(piece, 30, 40)], (), sprites=[(member, None)], cache=cache)
    assert item._level_images == {}, "new content drops the level memo"


def test_without_a_cache_the_item_draws_level_0_at_every_zoom() -> None:
    """A trigger-created ghost (TASK-313) sets content with no members and no cache."""
    item = _item()
    item.set_content([(_piece(4, 4, rgb=(0, 255, 0)), 10, 10)], ())
    assert item._cache is None
    image = paint_at(item, 1.5)
    assert item._level_images == {}
    assert image.pixelColor(3, 3).getRgb() == (0, 255, 0, 255)
    assert image.pixelColor(7, 7).alpha() == 0, "drawn at level 0's own 4 px, scaled by 1.5"


def test_a_refused_members_level_art_takes_the_tint_once_across_moves() -> None:
    from PyQt5.QtGui import QColor

    cache = _Cache()
    level = _piece(4, 4)
    member = _Member({0: [(_piece(6, 6), 0, 0)], 1: [(level, 0, 0)]})
    item = _item()
    item.set_refusal_color(QColor(255, 0, 0))
    item.set_content((), (), refused_draws=[(member.level0()[0])], sprites=[(member, item.REFUSED)], cache=cache)
    image = paint_at(item, 1.25)
    ((tinted, *_rest),) = item._level_images[1]
    assert (id(level.rgba), QColor(255, 0, 0).rgb()) in item._tint_cache
    # Scene (1, 1) sits inside the level art (0..2.67 scene px); paint_at() starts at the padded rect's corner.
    at = int((1 - item.boundingRect().left()) * 1.25)
    px = image.pixelColor(at, at)
    assert px.alpha() == 255 and px.red() > px.green(), "the level art is tinted toward the refusal colour"
    # A group drag re-sends the same sources every move: the level tint is reused, not re-blended.
    item.set_content((), (), refused_draws=[(member.level0()[0])], sprites=[(member, item.REFUSED)], cache=cache)
    paint_at(item, 1.25)
    assert item._level_images[1][0][0] is tinted


@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_every_levels_images_are_the_placed_units_draws_at_that_level(level_install, style) -> None:  # noqa: F811
    window = level_window(style)
    try:
        view = window.map_view
        cache = view._canvas_item._cache
        scene = _hover_tile(window, 40, 40)
        item = view._unit_ghost_item
        assert item is not None and item._cache is cache
        got = {mip: [(x, y, img.width(), img.height()) for img, _s, _h, x, y, _sc in item._images_at(mip)]
               for mip in cache.mip_levels() if mip != 0}
        rect = item.boundingRect()
        entry = _place_at(window, scene)
        for mip, entries in got.items():
            proj, rise = cache.level_sprite_inputs(mip)
            draws = render_sprites.unit_sprite_draws_at(
                window.scenario, proj, view._iso_elevations, rise, entry.player_id, entry.unit,
                tree_scale=window._layers.tree_scale, hero_glow=window._layers.hero_glow,
                fold=cache.fold if style == "Sloped" else None,
            )
            want = [(float(ax - d.hotspot_x), float(ay - d.hotspot_y), d.rgba.shape[1], d.rgba.shape[0])
                    for d, ax, ay in draws]
            assert entries == want and want, (style, mip)
            s = cache.mip_scale(mip)
            for x, y, w, h in entries:
                assert rect.contains(x * s, y * s) and rect.contains((x + w) * s, (y + h) * s), (style, mip)
    finally:
        conftest.close_window(window)


def test_a_level_0_paint_resolves_no_other_level_and_a_new_hover_drops_the_memo(level_install, monkeypatch) -> None:  # noqa: F811
    window = level_window("Stepped")
    try:
        view = window.map_view
        _hover_tile(window, 40, 40)
        calls = counting_resolver(monkeypatch)
        item = view._unit_ghost_item
        paint_at(item, 1.0)
        assert calls == []
        paint_at(item, 0.5)
        paint_at(item, 0.5)
        assert len(calls) == 1 and list(item._level_images) == [-1]
        _hover_tile(window, 44, 44)
        assert len(calls) == 2, "the new content resolves its level 0 only"
        assert view._unit_ghost_item._level_images == {}
    finally:
        conftest.close_window(window)
