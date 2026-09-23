"""GH #53 Part B: editor-only markers for the objects the game never draws.

Three layers: unit_kind.invisible_category() sorts invisible_consts() into
marker categories; descape/editor_markers.py builds each category's untinted
glyph; unit_sprites.marker_for() tints it and hands it to every compositor
through sprite_pieces_for() / icon_for(), the seam real art uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, editor_markers, iso_geometry, render, render_cache, unit_kind, unit_sprites
from descape.terrain_palette import PLAYER_COLORS

INVISIBLE_OBJECT_A = 1291
MAP_REVEALER = 837
BLOCKER = 1776
BLOCKER_3X1 = 2424
EMPTY_TC_ANNEX = 890
FARM = 50  # no map entry, but real art: not invisible, no marker

MAP_W = MAP_H = 12
CHUNK = 128


@pytest.fixture(autouse=True)
def _no_install():
    """Every test here starts with no install and cold caches."""
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# --- categories -------------------------------------------------------


@pytest.mark.parametrize(
    ("const", "category"),
    [
        (INVISIBLE_OBJECT_A, "invisible"),
        (2563, "invisible"),
        (MAP_REVEALER, "revealer"),
        (1775, "revealer"),
        (BLOCKER, "blocker"),
        (2423, "blocker"),
        (EMPTY_TC_ANNEX, "other"),
        (0, "other"),
        (FARM, None),
        (4, None),
    ],
)
def test_invisible_category_sorts_by_dat_class(const, category):
    assert unit_kind.invisible_category(const) == category


def test_every_invisible_const_gets_a_category():
    for const in unit_kind.invisible_consts():
        assert unit_kind.invisible_category(const) in editor_markers.CATEGORIES


# --- marker_for -------------------------------------------------------


def test_marker_for_is_deterministic():
    a = unit_sprites.marker_for("invisible", 1, 32).rgba.copy()
    unit_sprites.clear_caches()
    b = unit_sprites.marker_for("invisible", 1, 32).rgba
    assert np.array_equal(a, b)


@pytest.mark.parametrize("category", editor_markers.CATEGORIES)
@pytest.mark.parametrize("half_w", [4, 8, 16, 32, 64])
def test_a_marker_is_one_tile_centred_on_its_hotspot(category, half_w):
    draw = unit_sprites.marker_for(category, 1, half_w)
    assert draw.rgba.shape == (half_w, 2 * half_w, 4)
    assert (draw.hotspot_x, draw.hotspot_y) == (half_w, half_w // 2)
    scale = unit_sprites.sprite_scale(half_w)
    assert half_w <= unit_sprites.MAX_SPRITE_REACH_LEFT * scale
    assert half_w <= unit_sprites.MAX_SPRITE_REACH_RIGHT * scale
    assert half_w // 2 <= unit_sprites.MAX_SPRITE_REACH_UP * scale
    assert half_w // 2 <= unit_sprites.MAX_SPRITE_REACH_DOWN * scale


def test_a_markers_badge_follows_its_owner_and_gaia_stays_grey():
    blue = unit_sprites.marker_for("blocker", 1, 32).rgba
    red = unit_sprites.marker_for("blocker", 2, 32).rgba
    gaia = unit_sprites.marker_for("blocker", 0, 32).rgba
    assert not np.array_equal(blue, red)
    main, _cov, _hx, _hy = editor_markers.marker_layers("blocker", 32)
    assert np.array_equal(gaia, main), "GAIA is white, so its badge must come back untinted"
    # The centre of the badge takes the owner's hue; the symbol's white does not.
    badge_px = blue[2, 32]
    assert badge_px[2] > badge_px[0] and badge_px[2] > badge_px[1]


def test_each_category_draws_a_different_symbol():
    arrays = [unit_sprites.marker_for(c, 1, 32).rgba for c in editor_markers.CATEGORIES]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            assert not np.array_equal(arrays[i], arrays[j])


def test_tiny_zoom_falls_back_to_the_plain_badge():
    """Below MIN_SYMBOL_PX a symbol is noise: every category is the same badge."""
    tiny = [unit_sprites.marker_for(c, 1, 8).rgba for c in editor_markers.CATEGORIES]
    assert all(np.array_equal(tiny[0], t) for t in tiny[1:])
    white = np.all(tiny[0][..., :3] > 230, axis=2) & (tiny[0][..., 3] > 0)
    assert not white.any(), "the plain badge must carry no white symbol pixels"


def test_the_revealer_falls_back_to_its_glyph_without_the_game_art(tmp_path, monkeypatch):
    asset_source.set_install_path_override(tmp_path)  # an "install" with no visibility.png
    fallback = unit_sprites.marker_for("revealer", 1, 32).rgba.copy()
    monkeypatch.setattr(editor_markers, "VISIBILITY_ICON_SUBPATH", "no/such/icon.png")
    unit_sprites.clear_caches()
    assert np.array_equal(unit_sprites.marker_for("revealer", 1, 32).rgba, fallback)
    assert not np.array_equal(fallback, unit_sprites.marker_for("invisible", 1, 32).rgba)


def test_the_revealer_uses_the_games_visibility_icon_when_installed(tmp_path):
    import os
    from pathlib import Path

    root = os.environ.get("AOE2DE_INSTALL_PATH")
    if not root or not (Path(root) / editor_markers.VISIBILITY_ICON_SUBPATH).is_file():
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one")
    asset_source.set_install_path_override(tmp_path)
    glyph = unit_sprites.marker_for("revealer", 1, 32).rgba.copy()
    asset_source.set_install_path_override(Path(root))
    art = unit_sprites.marker_for("revealer", 1, 32).rgba
    assert not np.array_equal(art, glyph)
    assert art.shape == glyph.shape


# --- the sprite seam --------------------------------------------------


@pytest.mark.parametrize("const", [INVISIBLE_OBJECT_A, MAP_REVEALER, BLOCKER])
def test_sprite_pieces_for_returns_the_marker(const):
    pieces = unit_sprites.sprite_pieces_for(const, 0.0, 1, 32)
    assert len(pieces) == 1
    piece = pieces[0]
    assert (piece.dx, piece.dy, piece.slot) == (0, 0, None)
    expected = unit_sprites.marker_for(unit_kind.invisible_category(const), 1, 32)
    assert np.array_equal(piece.draw.rgba, expected.rgba)


@pytest.mark.parametrize("const", [INVISIBLE_OBJECT_A, MAP_REVEALER, BLOCKER])
def test_icon_for_contain_fits_the_marker_into_its_footprint(const):
    for fw, fh in ((64, 64), (32, 32), (64, 192)):
        icon = unit_sprites.icon_for(const, 0.0, 1, fw, fh)
        assert icon is not None
        ih, iw = icon.rgba.shape[:2]
        assert iw <= fw and ih <= fh
        assert iw >= fw - 1 or ih == fh, "contain-fit should touch the footprint on one axis"
        assert (icon.hotspot_x, icon.hotspot_y) == (0, 0)


def test_a_const_with_no_category_and_no_map_entry_still_resolves_to_nothing():
    assert unit_sprites.sprite_pieces_for(FARM, 0.0, 1, 32) == []
    assert unit_sprites.icon_for(FARM, 0.0, 1, 64, 64) is None


# --- rendering: each compositor --------------------------------------


@dataclass
class Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(*, elevation=0):
    tiles = [Tile(x, y, elevation) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, [
        [Unit(3.5, 3.5, MAP_REVEALER)],
        [Unit(5.5, 5.5, INVISIBLE_OBJECT_A), Unit(8.5, 7.5, BLOCKER), Unit(3.5, 9.5, BLOCKER_3X1)],
    ])


def _chunked_iso(scn, full, elevations, proj, sprites):
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    units_by_tile = render._units_by_tile(scn)
    bboxes = render._building_bboxes_iso(scn, mm.map_width, mm.map_height, proj, elevations)
    merged = render.merge_sprite_bboxes(bboxes, sprites)
    out = np.zeros_like(full)
    h, w = full.shape[:2]
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = render.composite_rect_iso(
                scn, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, merged, sprites=sprites,
            )
    return out


def test_stepped_draws_markers_instead_of_marks_and_stitches():
    scn = _scenario()
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    every = {id(u) for units in scn.unit_manager.units for u in units}
    assert sprites.skip_ids == every, "every invisible object should resolve to a marker, hiding its mark"
    assert np.array_equal(_chunked_iso(scn, full, elevations, proj, sprites), full)


def test_a_stepped_marker_lands_on_its_own_tile():
    scn = _Scenario([Tile(x, y, 0) for y in range(MAP_H) for x in range(MAP_W)],
                    [[], [Unit(5.5, 5.5, INVISIBLE_OBJECT_A)]])
    bare, _e, _p = render.render_terrain_iso_with_proj(_Scenario(scn.map_manager.terrain, [[], []]), with_sprites=True)
    full, _elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    changed = np.argwhere(np.any(bare != full, axis=2))
    assert len(changed), "the marker painted nothing"
    sx, sy = iso_geometry.tile_screen_origin(5, 5, 0, proj)
    ys, xs = changed[:, 0], changed[:, 1]
    assert xs.min() >= sx and xs.max() < sx + 2 * proj.half_w
    assert ys.min() >= sy and ys.max() < sy + 2 * proj.half_h


def test_sloped_matches_stepped_on_a_flat_map_with_markers():
    scn = _scenario(elevation=2)
    sloped, _e, _c, _sp = render.render_terrain_sloped_with_proj(scn, with_sprites=True)
    stepped, _e2, _p = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    assert sloped.shape == stepped.shape
    assert np.array_equal(sloped, stepped)


def test_the_sloped_cache_stitches_markers_on_a_ramp():
    scn = _scenario()
    for tile in scn.map_manager.terrain:
        tile.elevation = 1 if tile.x >= MAP_W // 2 else 0
    mm = scn.map_manager
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    cache = render_cache.SlopedChunkCache(
        scn, elevations, corner_rise, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True
    )
    w, h = cache.canvas_dims()
    full, _e, _c, _p = render.render_terrain_sloped_with_proj(scn, with_sprites=True)
    assert np.array_equal(cache.render_rect(0, 0, w, h), full[:h, :w])


def _stitched_flat(cache, mip=0):
    w, h = cache.canvas_dims(mip)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = cache._composite_rect(mip, x0, y0, x1, y1)
    return out


def test_flat_icons_stay_in_their_footprints_and_match_the_full_render():
    scn = _scenario()
    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    cache = render_cache.FlatChunkCache(scn, tile_px, sprites=True)
    assert len(cache._level_icons(0)) == 4, "every invisible object should resolve to a Flat icon"
    full = render.overlay_units(render.render_terrain(scn), scn, with_sprites=True)
    assert np.array_equal(_stitched_flat(cache), full)

    terrain_only = _stitched_flat(render_cache.FlatChunkCache(scn, tile_px, with_units=False))
    changed = np.any(terrain_only != full, axis=2)
    inside = np.zeros_like(changed)
    for units in scn.unit_manager.units:
        for unit in units:
            tx0, tx1, ty0, ty1 = render.unit_tile_bounds(unit, MAP_W, MAP_H)
            dx, dy = render.unit_paint_offset(unit)
            ox, oy = round(dx * tile_px), round(dy * tile_px)
            inside[ty0 * tile_px + oy:ty1 * tile_px + oy, tx0 * tile_px + ox:tx1 * tile_px + ox] = True
    assert changed.any()
    assert not (changed & ~inside).any()


def _render(style, scn, with_sprites):
    if style == "flat":
        return render.overlay_units(render.render_terrain(scn), scn, with_sprites=with_sprites)
    if style == "iso":
        return render.render_terrain_iso_with_proj(scn, with_sprites=with_sprites)[0]
    return render.render_terrain_sloped_with_proj(scn, with_sprites=with_sprites)[0]


@pytest.mark.parametrize("style", ["flat", "iso", "sloped"])
def test_sprites_off_is_byte_identical_to_the_pre_marker_render(style, monkeypatch):
    """With sprites off these objects keep their coloured box, byte-identical to
    a build with no markers at all (invisible_category() forced to None). With
    no markers, sprites on is the box too, which is the pre-Part-B render."""
    scn = _scenario()
    on = _render(style, scn, True)
    off = _render(style, scn, False)
    assert not np.array_equal(on, off), "no marker painted, so this proves nothing"

    monkeypatch.setattr(unit_kind, "invisible_category", lambda const: None)
    unit_sprites.clear_caches()
    assert np.array_equal(_render(style, scn, False), off)
    assert np.array_equal(_render(style, scn, True), off)
