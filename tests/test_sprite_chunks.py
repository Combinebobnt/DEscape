"""Stitched chunks must be byte-identical to a full sprite render (P3-g3).

**This is the load-bearing check of the whole sprite step**, and it is here
because the thing it guards is invisible in any single-chunk render.

A sprite is far larger than the tile diamond it stands on -- a villager's is a
200x200 native canvas against a 64x32 diamond -- so its pixels routinely cross
chunk boundaries. composite_rect_iso only visits tiles it believes affect the
rect, and _building_bboxes_iso deliberately skips single-tile units on the
stated grounds that they "can't make a neighbour a bystander". That premise is
true for coloured marks and false for sprites. If merge_sprite_bboxes does not
lift it, every sprite clips at its owning chunk's edge -- a visible artifact,
not a latent one, and one that a full-canvas render never shows.

The second property pinned here is that a sprite paints ONCE. The diamond path
buckets a multi-tile building into every footprint tile and paints one diamond
per bucket; doing that with a sprite would draw it N times, and because a
sprite is mostly transparent the repeats would show as a smeared stack rather
than as a clean overdraw.

Synthetic bytes and a tmp install throughout -- no game assets, matching the
suite's standing posture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, render_cache, unit_sprites
from descape.terrain_palette import PLAYER_COLORS
from test_unit_sprites import CONST, FILE_NAME, build_sld

MAP_W = MAP_H = 12
CHUNK = 128


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
    """Same duck-typing the other default-tier render tests use for synthetic
    maps; this module keeps its own copy rather than importing one, matching
    tests/test_unit_filter.py's own reasoning."""

    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        # Identity default -- no synthetic scenario here stores a color
        # override.
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player, *, elevation=0):
    tiles = [Tile(x, y, elevation) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one sprite that is much bigger than its tile.

    The canvas is 4 * NATIVE_TILE_W, so the sprite reaches two whole tiles to
    either side of its anchor. That size is the point, not incidental: at
    1 * NATIVE_TILE_W it reaches only half a diamond, and
    iso_geometry.tiles_in_screen_rect already pulls the immediate neighbours in
    for unrelated reasons -- so the widening's own mutation check passes
    vacuously and proves nothing. Real sprites are 200 to 1300px canvases
    against a 64x32 diamond, so this is the realistic case, not a stressed one.
    """
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(
        build_sld(4, canvas=4 * unit_sprites.NATIVE_TILE_W)
    )
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _chunked(scn, full, elevations, proj, sprites):
    """Every CHUNK-sized rect composited in isolation, stitched back together
    -- the same shape tests/test_flat_chunks.py's own stitching check uses."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    units_by_tile = render._units_by_tile(scn)
    bboxes = render._building_bboxes_iso(units_by_tile, mm.map_width, mm.map_height, proj, elevations)
    merged = render.merge_sprite_bboxes(bboxes, sprites)
    out = np.zeros_like(full)
    h, w = full.shape[:2]
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = render.composite_rect_iso(
                scn, x0, y0, x1, y1, elevations, proj, tile_px,
                units_by_tile, merged, sprites=sprites,
            )
    return out


def test_stitched_chunks_match_the_full_sprite_render(sprite_install):
    scn = _scenario([[], [Unit(4.0, 4.0, CONST), Unit(8.0, 7.0, CONST)]])
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert sprites.skip_ids, "the fixture resolved no sprites, so this proves nothing"

    got = _chunked(scn, full, elevations, proj, sprites)
    assert np.array_equal(got, full)


def test_without_the_widening_a_sprite_clips_at_a_chunk_edge(sprite_install):
    """The mutation check: drop merge_sprite_bboxes and the stitch must FAIL.

    Without this, the test above passes just as happily if composite_rect_iso
    happened to visit the anchor tile for some unrelated reason, and the
    widening would be untested while looking covered.
    """
    scn = _scenario([[], [Unit(4.0, 4.0, CONST)]])
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)

    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    units_by_tile = render._units_by_tile(scn)
    unwidened = render._building_bboxes_iso(
        units_by_tile, mm.map_width, mm.map_height, proj, elevations
    )
    out = np.zeros_like(full)
    h, w = full.shape[:2]
    for y0 in range(0, h, CHUNK):
        for x0 in range(0, w, CHUNK):
            x1, y1 = min(x0 + CHUNK, w), min(y0 + CHUNK, h)
            out[y0:y1, x0:x1] = render.composite_rect_iso(
                scn, x0, y0, x1, y1, elevations, proj, tile_px,
                units_by_tile, unwidened, sprites=sprites,
            )
    assert not np.array_equal(out, full)


def test_a_sprite_paints_once_not_once_per_footprint_tile(sprite_install, monkeypatch):
    """A multi-tile building reaches the diamond path once per footprint tile.
    Its sprite must reach the compositor exactly once."""
    span = (3, 3)
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, span)
    scn = _scenario([[], [Unit(5.5, 5.5, CONST)]])
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)

    total = sum(len(v) for v in sprites.by_anchor.values())
    assert total == 1, f"painted {total} times for a {span[0]}x{span[1]} footprint"
    # And at the footprint tile that comes LAST in depth_order, so every
    # footprint tile's terrain is already down when the sprite lands.
    bounds = render.unit_tile_bounds(scn.unit_manager.units[1][0], MAP_W, MAP_H)
    assert next(iter(sprites.by_anchor)) == unit_sprites.sprite_anchor_tile(*bounds)


def test_a_sprite_bearing_units_mark_is_not_drawn_underneath(sprite_install):
    """A sprite has transparent pixels, so an un-skipped coloured diamond shows
    as a fringe around the unit rather than being hidden by it."""
    scn = _scenario([[], [Unit(4.0, 4.0, CONST)]])
    unit = scn.unit_manager.units[1][0]
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert id(unit) in sprites.skip_ids


@pytest.fixture
def composite_sprite_install(tmp_path, monkeypatch):
    """A tmp install holding a two-piece composite building (the annex-
    composite shape: a town centre, a pasture) -- both pieces much bigger
    than their tile, same reasoning as sprite_install's own docstring, and
    the second piece offset well outside the first's own canvas so a bbox
    union bug (only unioning the parent's own piece) is visible rather than
    accidentally covered by overlap."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    canvas = 4 * unit_sprites.NATIVE_TILE_W
    piece_b_name = "t_piece_b_x1"
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(1, canvas=canvas))
    (graphics / f"{piece_b_name}.sld").write_bytes(build_sld(1, canvas=canvas))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {
            "graphic_id": 1, "file_name": FILE_NAME, "angle_count": 1,
            "mirroring_mode": 0, "frame_count": 1,
            "pieces": [
                {"unit_id": CONST, "file_name": FILE_NAME, "angle_count": 1,
                 "frame_count": 1, "dx": 0, "dy": 0},
                {"unit_id": CONST + 1, "file_name": piece_b_name, "angle_count": 1,
                 "frame_count": 1, "dx": 2 * canvas, "dy": -canvas},
            ],
        }},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_a_composite_units_stitched_chunks_match_the_full_render(composite_sprite_install):
    """Every piece of a multi-graphic composite (a town centre, a pasture)
    must reach the chunked compositor exactly like a plain sprite does --
    the P3-g3 stitching guarantee (this module's own opening docstring)
    extends to each piece's own offset anchor, not just the parent's."""
    scn = _scenario([[], [Unit(4.0, 4.0, CONST)]])
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert sprites.skip_ids, "the fixture resolved no sprites, so this proves nothing"

    anchor = next(iter(sprites.by_anchor))
    assert len(sprites.by_anchor[anchor]) == 2, "both composite pieces must reach the same anchor slot"

    got = _chunked(scn, full, elevations, proj, sprites)
    assert np.array_equal(got, full)


def test_an_unresolvable_unit_keeps_its_coloured_mark(sprite_install):
    """The fallback is what makes the sprite path strictly additive."""
    scn = _scenario([[], [Unit(4.0, 4.0, CONST + 1)]])
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert not sprites.skip_ids and not sprites.by_anchor

    with_sprites, _, _ = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    without, _, _ = render.render_terrain_iso_with_proj(scn, with_sprites=False)
    assert np.array_equal(with_sprites, without)


@pytest.mark.parametrize("span", [(1, 1), (3, 3), (5, 5)])
def test_an_odd_span_sprite_anchors_on_its_tiles_ground_centre(sprite_install, monkeypatch, span):
    """The check that was missing while every sprite rendered floating half a
    tile above the ground (found in a live window 2026-08-24, fixed same day).

    Nothing caught it because everything that touched the anchor shared its
    error: `sprite_draws_by_anchor` put ay at the tile's bounding-box TOP, and
    `_dirty_screen_bbox`'s slack band was written as `tile_origin_y -+ half_h`
    to match. Two wrongs agreeing is a green suite. Even the g3 plan's visual
    confirmation passed, because half a tile of float reads as plausible
    contact shadow -- so this asserts against `tile_screen_origin` instead,
    which is what actually paints the terrain.

    ODD spans only. An even span's footprint centre is a tile CORNER, not a
    tile centre (`_span_start`'s half-tile branch), so it legitimately sits
    half a tile off; parametrising over odd spans keeps the expected value a
    single unambiguous point. `test_the_bbox_covers_the_sprite_at_every_span_
    parity` is where the even cases are pinned.
    """
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, span)
    tx = ty = MAP_W // 2
    scn = _scenario([[], [Unit(tx + 0.5, ty + 0.5, CONST)]], elevation=3)
    _full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)

    layer = render.sprite_draws_by_anchor(scn, proj, elevations)
    assert layer.by_anchor, "no sprite resolved -- this test would assert nothing"
    _draw, ax, ay = next(iter(layer.by_anchor.values()))[0]

    sx, sy = iso_geometry.tile_screen_origin(tx, ty, 3, proj)
    centre = (sx + proj.half_w, sy + proj.half_h)
    assert (ax, ay) == centre, (
        f"span {span} sprite anchors at {(ax, ay)} but tile ({tx}, {ty})'s ground centre is "
        f"{centre} -- an offset of {(ax - centre[0], ay - centre[1])}. A dy of -half_h "
        f"({-proj.half_h}) is the float bug: tile_screen_origin returns the diamond's "
        f"BOUNDING-BOX top-left, so the ground point is half_h below it, not at it"
    )


def test_sprites_off_is_byte_identical_to_the_pre_sprite_render(sprite_install):
    """P3-g3 ships with SPRITES_ENABLED False, and the whole existing suite is
    that assertion. Pinned explicitly so flipping the default in P3-g4 is a
    deliberate act rather than something a refactor can do silently.

    Both assertions are wanted, and they say different things as of P3-g's
    toggle: SPRITES_ENABLED is now only IsoChunkCache's CONSTRUCTION default,
    while cache.sprites_enabled is the live per-instance flag everything
    actually reads. Pin the default AND the fact that it reaches an
    unparameterised cache."""
    scn = _scenario([[], [Unit(4.0, 4.0, CONST)]])
    assert render_cache.SPRITES_ENABLED is False
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=False)
    cache = render_cache.IsoChunkCache(
        scn, elevations, proj, render.tile_pixels_for_map(MAP_W, MAP_H), chunk_px=CHUNK
    )
    assert cache.sprites_enabled is False
    assert cache._level(0).sprites is None

    w, h = cache.canvas_dims(0)
    assert np.array_equal(cache.render_rect(0, 0, min(w, 256), min(h, 256)), full[: min(h, 256), : min(w, 256)])


def test_the_chunk_cache_renders_sprites_when_enabled(sprite_install):
    """The viewer composites through IsoChunkCache, never through the full
    render, so the switch has to reach that path too or P3-g4 measures nothing.

    sprites=True at construction rather than a monkeypatch of SPRITES_ENABLED:
    as of P3-g's toggle that constant is only a default argument, evaluated
    once at import, so patching it would no longer reach this cache at all.
    """
    scn = _scenario([[], [Unit(4.0, 4.0, CONST), Unit(8.0, 7.0, CONST)]])
    full, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    cache = render_cache.IsoChunkCache(
        scn, elevations, proj, render.tile_pixels_for_map(MAP_W, MAP_H), chunk_px=CHUNK, sprites=True
    )
    assert cache._level(0).sprites is not None

    w, h = cache.canvas_dims(0)
    assert np.array_equal(cache.render_rect(0, 0, w, h), full[:h, :w])
