"""GH #39: the gold glow round hero units.

Three layers: the generated HERO_GLOW_CONSTS set (the .dat's hero_mode bits
and hero_glow_graphic), unit_sprites._with_glow() (a ring dilated off the
scaled sprite's alpha), and the View > Layers > Hero Glow row threaded through
render.py and the chunk caches the way Small Trees is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, render, render_cache, unit_sprites
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.terrain_palette import HERO_GLOW_CONSTS
from descape.unit_sprites import HERO_GLOW_GOLD, HERO_GLOW_RADIUS, SpriteDraw
from descape.view_layers import LayerState
from testkit.fakes import FakeScenario, SyntheticTile

from test_unit_sprites import build_sld

HLEIF, HKHAN, HLUBU, HWOLF = 106, 1275, 2032, 700
ARCHR, KINGX, WOLF2, HTAT = 4, 434, 202, 684
NON_HERO_CONST = 999003
SPRITE_FILE = "t_hero_synthetic_x1"
# Big enough that a sprite one tile in front overlaps the hero's (occlusion test).
SPRITE_CANVAS = 96
MAP_W = MAP_H = 16


# --- the generated set ------------------------------------------------


def test_the_hero_set_is_the_dat_rule_not_a_hand_list():
    assert len(HERO_GLOW_CONSTS) == 260
    # Mode 1, mode 64, and a Three Kingdoms hero whose mode sets bit 128 (-111).
    assert {HLEIF, HKHAN, HLUBU} <= HERO_GLOW_CONSTS
    # A glow graphic without the hero bits, mode 34 with a glow graphic, the
    # black custom glow, and a hero-flagged building with no glow graphic.
    assert not {ARCHR, KINGX, WOLF2, HTAT} & HERO_GLOW_CONSTS


def test_hwolf_follows_the_mask_reading_until_the_in_game_check():
    """Mode -63 sets bit 128 ("invert flags" in AGE). The `& 65` reading keeps
    it; the in-game pass decides, and this pins which reading shipped."""
    assert HWOLF in HERO_GLOW_CONSTS


# --- the ring ---------------------------------------------------------


def _plus_draw() -> SpriteDraw:
    """A non-rectangular silhouette with soft-alpha edge pixels, so the ring
    has corners and concavities to follow, and 'alpha > 0' is exercised."""
    rgba = np.zeros((9, 11, 4), dtype=np.uint8)
    rgba[3:6, 1:10] = (40, 90, 160, 255)
    rgba[1:8, 4:7] = (200, 30, 30, 255)
    rgba[4, 0] = (10, 10, 10, 60)
    return SpriteDraw(rgba=rgba, hotspot_x=5, hotspot_y=7)


def _expected_ring_alpha(mask: np.ndarray, r: int) -> np.ndarray:
    """Brute-force oracle: each padded px's Euclidean distance to the nearest
    silhouette px, independent of _with_glow's roll-and-OR dilation."""
    h, w = mask.shape
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=bool)
    padded[r:r + h, r:r + w] = mask
    ys, xs = np.nonzero(padded)
    out = np.zeros(padded.shape, dtype=np.uint8)
    for y in range(padded.shape[0]):
        for x in range(padded.shape[1]):
            if padded[y, x]:
                continue
            d2 = int(((ys - y) ** 2 + (xs - x) ** 2).min())
            out[y, x] = 255 if d2 <= 1 else 128 if d2 <= 4 else 0
    return out


def test_the_ring_hugs_the_silhouette_at_its_own_distance():
    draw = _plus_draw()
    glowed = unit_sprites._with_glow(draw)
    r = HERO_GLOW_RADIUS
    mask = draw.rgba[..., 3] > 0
    ring_alpha = _expected_ring_alpha(mask, r)

    inside = np.zeros(glowed.rgba.shape[:2], dtype=bool)
    inside[r:r + mask.shape[0], r:r + mask.shape[1]] = mask
    outside = ~inside
    assert np.array_equal(glowed.rgba[..., 3][outside], ring_alpha[outside])
    ring = outside & (ring_alpha > 0)
    assert ring.any()
    assert (glowed.rgba[..., :3][ring] == HERO_GLOW_GOLD).all()
    assert (glowed.rgba[outside & ~ring] == 0).all()


def test_the_sprite_itself_is_byte_identical_and_the_hotspot_follows_the_pad():
    draw = _plus_draw()
    glowed = unit_sprites._with_glow(draw)
    r = HERO_GLOW_RADIUS
    h, w = draw.rgba.shape[:2]
    assert glowed.rgba.shape == (h + 2 * r, w + 2 * r, 4)
    mask = draw.rgba[..., 3] > 0
    assert np.array_equal(glowed.rgba[r:r + h, r:r + w][mask], draw.rgba[mask])
    assert (glowed.hotspot_x, glowed.hotspot_y) == (draw.hotspot_x + r, draw.hotspot_y + r)


# --- through sprite_pieces_for --------------------------------------------


@pytest.fixture
def install(tmp_path, monkeypatch):
    """One synthetic .sld registered for a real hero const and a non-hero one,
    so only the const test in render._resolve_unit_sprite() can tell them apart."""
    assert NON_HERO_CONST not in HERO_GLOW_CONSTS
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{SPRITE_FILE}.sld").write_bytes(build_sld(1, canvas=SPRITE_CANVAS))
    entry = {"graphic_id": 1, "file_name": SPRITE_FILE, "angle_count": 1,
             "mirroring_mode": 6, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {HLEIF: entry, NON_HERO_CONST: entry})
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _one_draw(const: int, **kwargs) -> SpriteDraw:
    pieces = unit_sprites.sprite_pieces_for(const, 0.0, 1, 32, **kwargs)
    assert len(pieces) == 1, "fixture resolved to no sprite, the test would be vacuous"
    return pieces[0].draw


def test_sprite_pieces_for_rings_the_scaled_sprite(install):
    plain = _one_draw(HLEIF)
    glowed = _one_draw(HLEIF, hero_glow=True)
    r = HERO_GLOW_RADIUS
    h, w = plain.rgba.shape[:2]
    assert glowed.rgba.shape[:2] == (h + 2 * r, w + 2 * r)
    assert np.array_equal(glowed.rgba[r:r + h, r:r + w], plain.rgba)
    assert (glowed.hotspot_x, glowed.hotspot_y) == (plain.hotspot_x + r, plain.hotspot_y + r)
    assert (glowed.rgba[0:r, :, 3] > 0).any(), "no ring painted in the pad"


def test_the_glow_gets_its_own_scaled_cache_entry(install):
    """_scaled_cache is process-global, so a key without the flag would hand a
    ringed sprite to a plain render, as a cache hit."""
    glowed = _one_draw(HLEIF, hero_glow=True)
    plain = _one_draw(HLEIF)
    again = _one_draw(HLEIF, hero_glow=True)
    assert plain.rgba.shape != glowed.rgba.shape
    assert np.array_equal(again.rgba, glowed.rgba)


# --- the render and the caches --------------------------------------------


@dataclass
class PlacedUnit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


def _scenario(*, overlap: bool = False) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=0, terrain_id=1) for y in range(MAP_H) for x in range(MAP_W)]
    units = [[] for _ in range(9)]
    # Well apart unless `overlap`, where the non-hero stands one tile in front.
    front = (6.5, 6.5) if overlap else (12.5, 12.5)
    units[1] = [
        PlacedUnit(x=5.5, y=5.5, unit_const=HLEIF, reference_id=1),
        PlacedUnit(x=front[0], y=front[1], unit_const=NON_HERO_CONST, reference_id=2),
    ]
    return FakeScenario(MAP_W, MAP_H, tiles, units)


def _shapes(scn, **kwargs) -> dict[tuple[int, int], tuple[int, int]]:
    elevations, proj = render.elevations_and_proj(scn)
    layer = render.sprite_draws_by_anchor(scn, proj, elevations, **kwargs)
    return {tile: draws[0][0].rgba.shape[:2] for tile, draws in layer.by_anchor.items()}


def test_only_hero_consts_are_ringed(install):
    scn = _scenario()
    plain = _shapes(scn)
    glowed = _shapes(scn, hero_glow=True)
    hero_tile, other_tile = (5, 5), (12, 12)
    assert set(plain) == set(glowed) == {hero_tile, other_tile}, plain
    pad = 2 * HERO_GLOW_RADIUS
    assert glowed[hero_tile] == (plain[hero_tile][0] + pad, plain[hero_tile][1] + pad)
    assert glowed[other_tile] == plain[other_tile]


def test_the_render_default_is_byte_identical_to_passing_false(install):
    scn = _scenario()
    elevations, proj = render.elevations_and_proj(scn)
    omitted = render.sprite_draws_by_anchor(scn, proj, elevations)
    passed = render.sprite_draws_by_anchor(scn, proj, elevations, hero_glow=False)
    assert omitted.bboxes == passed.bboxes
    for tile, draws in omitted.by_anchor.items():
        for (draw, px, py), (pdraw, ppx, ppy) in zip(draws, passed.by_anchor[tile], strict=True):
            assert (px, py) == (ppx, ppy)
            assert np.array_equal(draw.rgba, pdraw.rgba)


def test_the_bbox_widens_with_the_ring(install):
    """The dirty-rect math reads these bboxes, so the ring has to be inside them."""
    scn = _scenario()
    elevations, proj = render.elevations_and_proj(scn)
    plain = render.sprite_draws_by_anchor(scn, proj, elevations).bboxes[(5, 5)]
    glowed = render.sprite_draws_by_anchor(scn, proj, elevations, hero_glow=True).bboxes[(5, 5)]
    r = HERO_GLOW_RADIUS
    assert glowed == (plain[0] - r, plain[1] - r, plain[2] + r, plain[3] + r)


def _iso_cache(scn, **kwargs) -> IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _sloped_cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(
        scn, elevations, corner_rise, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs
    )


def _whole_canvas(cache) -> np.ndarray:
    w, h = cache.canvas_dims(0)
    return cache.render_rect(0, 0, w, h)


def _gold_px(canvas: np.ndarray) -> int:
    return int(np.all(canvas == HERO_GLOW_GOLD, axis=2).sum())


@pytest.mark.parametrize("builder", [_iso_cache, _sloped_cache])
def test_the_layer_defaults_on_and_round_trips_byte_identically(builder, install):
    """Build-time, like Small Trees: set_layers() has to rebuild the sprite
    layer, not just evict, or a level keeps serving the old ring state."""
    cache = builder(_scenario(), sprites=True)
    assert cache.layers.hero_glow is True
    on = _whole_canvas(cache).copy()
    assert _gold_px(on) > 0, "the default layer state drew no ring"

    cache.set_layers(LayerState(hero_glow=False))
    off = _whole_canvas(cache).copy()
    assert _gold_px(off) == 0
    assert not np.array_equal(on, off)

    cache.set_layers(LayerState())
    assert np.array_equal(on, _whole_canvas(cache))


def test_the_ring_survives_a_unit_splice(install):
    """The splice re-resolves the moved unit on its own; a site left on the
    default False would drop the ring the first time a hero is moved."""
    scn = _scenario()
    cache = _iso_cache(scn, sprites=True)
    _whole_canvas(cache)
    hero = scn.unit_manager.units[1][0]
    hero.x, hero.y = 8.5, 8.5
    splice = render_cache.UnitSplice(
        player_id=1, index=0, unit=hero,
        old_own_tile=(5, 5), new_own_tile=(8, 8),
        old_tiles=frozenset({(5, 5)}), new_tiles=frozenset({(8, 8)}),
    )
    cache.invalidate_units([splice])
    _whole_canvas(cache)
    by_anchor = cache._level(0).sprites.by_anchor
    assert (5, 5) not in by_anchor
    plain = _shapes(scn)[(8, 8)]
    pad = 2 * HERO_GLOW_RADIUS
    assert by_anchor[(8, 8)][0][0].rgba.shape[:2] == (plain[0] + pad, plain[1] + pad)


def test_a_sprite_in_front_hides_the_ring(install):
    """Baked into the sprite, so depth order occludes the ring like any art:
    wherever the front unit's own sprite covers, the canvas is the same with
    the layer on or off."""
    scn = _scenario(overlap=True)
    cache = _iso_cache(scn, sprites=True)
    on = _whole_canvas(cache).copy()
    cache.set_layers(LayerState(hero_glow=False))
    off = _whole_canvas(cache).copy()

    layer = cache._level(0).sprites
    (draw, px, py), = layer.by_anchor[(6, 6)]
    x0, y0 = px - draw.hotspot_x, py - draw.hotspot_y
    h, w = draw.rgba.shape[:2]
    cover = draw.rgba[..., 3] == 255
    on_crop, off_crop = on[y0:y0 + h, x0:x0 + w], off[y0:y0 + h, x0:x0 + w]
    assert np.array_equal(on_crop[cover], off_crop[cover])

    # Not vacuous: some of the hero's ring px do fall under that cover.
    front = np.zeros(on.shape[:2], dtype=bool)
    front[y0:y0 + h, x0:x0 + w] = cover
    cache.set_layers(LayerState())
    (hero, hpx, hpy), = cache._level(0).sprites.by_anchor[(5, 5)]
    hh, hw = hero.rgba.shape[:2]
    ring = np.all(hero.rgba[..., :3] == HERO_GLOW_GOLD, axis=2) & (hero.rgba[..., 3] > 0)
    hx0, hy0 = hpx - hero.hotspot_x, hpy - hero.hotspot_y
    assert (front[hy0:hy0 + hh, hx0:hx0 + hw] & ring).any(), "the front sprite hides none of the ring"


@pytest.mark.parametrize("full_render", [render.render_terrain_iso_with_proj, render.render_terrain_sloped_with_proj])
def test_the_full_render_rings_heroes_like_the_default_caches(full_render, install):
    """The full renders are the stitched-vs-full oracle for caches built on the
    default layer state, so they have to draw the ring too."""
    img = full_render(_scenario(), with_sprites=True)[0]
    assert _gold_px(img) > 0


def test_the_drag_ghost_path_takes_the_layer(install):
    scn = _scenario()
    elevations, proj = render.elevations_and_proj(scn)
    hero = scn.unit_manager.units[1][0]
    plain = render.unit_sprite_draws_at(scn, proj, elevations, None, 1, hero)
    glowed = render.unit_sprite_draws_at(scn, proj, elevations, None, 1, hero, hero_glow=True)
    assert plain and glowed
    pad = 2 * HERO_GLOW_RADIUS
    assert glowed[0][0].rgba.shape[0] == plain[0][0].rgba.shape[0] + pad


def test_flat_ignores_the_layer(install):
    """Flat's icons come from icon_for(), which the ring does not reach, which
    is why the registry greys the row there."""
    cache = FlatChunkCache(_scenario(), tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, sprites=True)
    before = _whole_canvas(cache).copy()
    cache.set_layers(LayerState(hero_glow=False))
    assert np.array_equal(before, _whole_canvas(cache))
