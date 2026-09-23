"""View > Grid baked into the chunk composite: each tile draws its own grid
edges between its own terrain and its own units, so no Z value is needed to
put the grid under a sprite.

The gates, in order of how much they protect:
  - grid off (and a centred blend) is byte-identical to omitting the spec,
    through every compositor;
  - stitched chunks equal the independent full render with the grid ON, in
    all three styles, and after an incremental patch();
  - a unit sprite's visible pixels are untouched by the grid, the reported
    bug as a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, grid_overlay, iso_geometry, render, unit_sprites
from descape.grid_overlay import DEFAULT_GRID, GridBake
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from testkit.fakes import FakeScenario, SyntheticTile

from test_unit_sprites import build_sld

MAP_W = MAP_H = 16
TERRAIN = 1
GRID = grid_overlay.grid_bake(True, -100, 1)
THICK_GRID = grid_overlay.grid_bake(True, -100, 3)
BUILDING_CONST = 999101
BUILDING_SPAN = (3, 3)
SPRITE_FILE = "t_grid_bake_building_x1"
SPRITE_CANVAS = 128


@dataclass
class PlacedUnit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 1


def _scenario(elevation=lambda x, y: 0, units=()) -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=elevation(x, y), terrain_id=TERRAIN)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    units_by_player = [[] for _ in range(9)]
    units_by_player[1] = list(units)
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


def _bumpy(x: int, y: int) -> int:
    return (x // 3 + y // 4) % 3


def _iso_cache(scn, **kwargs) -> IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _sloped_cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(scn, elevations, corner_rise, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _flat_cache(scn, **kwargs) -> FlatChunkCache:
    return FlatChunkCache(scn, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, **kwargs)


def _whole(cache, mip: int = 0) -> np.ndarray:
    w, h = cache.canvas_dims(mip)
    return cache.render_rect(0, 0, w, h, mip=mip)


def _full(style: str, scn, grid: GridBake, *, sprites: bool = False) -> np.ndarray:
    if style == "stepped":
        return render.render_terrain_iso_with_proj(scn, with_sprites=sprites, grid=grid)[0]
    if style == "sloped":
        return render.render_terrain_sloped_with_proj(scn, with_sprites=sprites, grid=grid)[0]
    return render.render_scenario(scn, isometric=False, grid=grid)


_BUILDERS = {"stepped": _iso_cache, "sloped": _sloped_cache, "flat": _flat_cache}


@pytest.fixture
def building_sprite(tmp_path, monkeypatch):
    """A synthetic 3x3 building drawn as one solid, fully opaque square."""
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, BUILDING_CONST, BUILDING_SPAN)
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{SPRITE_FILE}.sld").write_bytes(build_sld(1, canvas=SPRITE_CANVAS, playercolor=False))
    entry = {"graphic_id": 1, "file_name": SPRITE_FILE, "angle_count": 1, "mirroring_mode": 6, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {BUILDING_CONST: entry})
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# --- the spec -------------------------------------------------------------------


def test_the_spec_is_frozen_and_compares_by_value() -> None:
    assert grid_overlay.grid_bake(True, -100, 1) == GRID
    assert grid_overlay.grid_bake(False, -100, 1) != GRID
    assert not DEFAULT_GRID.paints
    assert not grid_overlay.grid_bake(True, 0, 1).paints
    assert GRID.paints
    with pytest.raises(AttributeError):
        GRID.enabled = False  # type: ignore[misc]


def test_each_tile_owns_its_low_edges_and_only_the_border_or_a_cliff_high_edge() -> None:
    assert [e for e, _ in grid_overlay.owned_edges(3, 5, MAP_W, MAP_H)] == ["x_low", "y_low"]
    assert [e for e, _ in grid_overlay.owned_edges(MAP_W - 1, MAP_H - 1, MAP_W, MAP_H)] == [
        "x_low", "y_low", "x_high", "y_high",
    ]
    elevations = np.zeros((MAP_H, MAP_W), dtype=np.int64)
    elevations[5, 4] = 1  # east neighbour of (3, 5)
    assert [e for e, _ in grid_overlay.owned_edges(3, 5, MAP_W, MAP_H, elevations)] == ["x_low", "y_low", "x_high"]
    majors = dict(grid_overlay.owned_edges(4, 5, MAP_W, MAP_H))
    assert majors == {"x_low": grid_overlay.is_major(4), "y_low": grid_overlay.is_major(5)}


# --- byte identity with the grid off ----------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
@pytest.mark.parametrize("spec", [GridBake(), grid_overlay.grid_bake(False, -100, 4), grid_overlay.grid_bake(True, 0, 2)])
def test_a_grid_that_paints_nothing_is_byte_identical_to_omitting_it(style, spec) -> None:
    scn = _scenario(_bumpy)
    omitted = _whole(_BUILDERS[style](scn)).copy()
    cache = _BUILDERS[style](scn)
    cache.set_grid(spec)
    assert np.array_equal(omitted, _whole(cache))
    assert np.array_equal(_full(style, scn, DEFAULT_GRID), _full(style, scn, spec))


# --- chunk path == full render, grid on ---------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
@pytest.mark.parametrize("spec", [GRID, THICK_GRID, grid_overlay.grid_bake(True, 60, 4)])
def test_stitched_chunks_match_the_full_render_with_the_grid_on(style, spec) -> None:
    scn = _scenario(_bumpy)
    cache = _BUILDERS[style](scn)
    cache.set_grid(spec)
    stitched = _whole(cache)
    full = _full(style, scn, spec)
    h, w = stitched.shape[:2]
    assert np.array_equal(stitched, full[:h, :w])
    assert not np.array_equal(stitched, _whole(_BUILDERS[style](scn))), "the grid must actually paint"


@pytest.mark.parametrize("spec", [GRID, THICK_GRID])
def test_sloped_equals_stepped_on_a_flat_map_with_the_grid_on(spec) -> None:
    scn = _scenario()
    stepped = _full("stepped", scn, spec)
    sloped = _full("sloped", scn, spec)
    assert np.array_equal(stepped, sloped)


# --- where the pixels land ------------------------------------------------------------


def _changed(style: str, scn, spec: GridBake) -> np.ndarray:
    off = _full(style, scn, DEFAULT_GRID)
    on = _full(style, scn, spec)
    return np.any(off != on, axis=2)


def test_stepped_grid_pixels_land_on_tile_edges_and_nowhere_else() -> None:
    scn = _scenario()
    _, proj = render.elevations_and_proj(scn)
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    changed = _changed("stepped", scn, GRID)
    allowed = np.zeros_like(changed)
    for y in range(MAP_H):
        for x in range(MAP_W):
            bx, by = iso_geometry.tile_screen_origin(x, y, 0, proj)
            for side in ("left", "right", "up_left", "up_right"):
                dy, dx = iso_geometry.tile_edge_indices(tile_px, side)
                allowed[by + dy, bx + dx] = True
    assert changed.any()
    assert not (changed & ~allowed).any()


def test_flat_grid_pixels_land_on_block_edges_and_nowhere_else() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    changed = _changed("flat", scn, GRID)
    rows, cols = np.nonzero(changed)
    on_edge = (rows % tile_px == 0) | (cols % tile_px == 0) | (rows == MAP_H * tile_px - 1) | (cols == MAP_W * tile_px - 1)
    assert rows.size
    assert on_edge.all()
    # Every interior tile boundary is drawn: a full-height column at each x line.
    for i in range(MAP_W):
        assert changed[:, i * tile_px].all()


@pytest.mark.parametrize("style", ["stepped", "flat"])
def test_a_shared_interior_edge_is_composited_once(style) -> None:
    """Exactly one lerp at a mid-edge pixel of a minor line: a second
    composite would darken it to the squared value."""
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    img = _full(style, scn, GRID)
    off = _full(style, scn, DEFAULT_GRID)
    tx, ty = 5, 6  # neither index major, well inside the map
    assert not grid_overlay.is_major(tx) and not grid_overlay.is_major(ty)
    if style == "flat":
        py, px = ty * tile_px + tile_px // 2, tx * tile_px
    else:
        _, proj = render.elevations_and_proj(scn)
        bx, by = iso_geometry.tile_screen_origin(tx, ty, 0, proj)
        dy, dx = iso_geometry.tile_edge_indices(tile_px, "left")
        mid = dy.size // 4
        py, px = by + dy[mid], bx + dx[mid]
    alpha = GRID.minor[3]
    expected = [(int(c) * (255 - alpha)) // 255 for c in off[py, px]]
    assert img[py, px].tolist() == expected


def test_a_stepped_cliff_lip_carries_its_own_line() -> None:
    """A raised tile's up-screen x-high edge sits above its lower east
    neighbour's; the drape's two-copies-at-two-heights shape."""
    raised = (6, 6)
    scn_flat = _scenario()
    scn = _scenario(lambda x, y: 2 if (x, y) == raised else 0)
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    _, proj = render.elevations_and_proj(scn)
    bx, by = iso_geometry.tile_screen_origin(*raised, 2, proj)
    dy, dx = iso_geometry.tile_edge_indices(tile_px, "up_right")
    changed = _changed("stepped", scn, GRID)
    assert changed[by + dy, bx + dx].all()
    fbx, fby = iso_geometry.tile_screen_origin(*raised, 0, proj)
    assert not _changed("stepped", scn_flat, GRID)[fby + dy[2:-2], fbx + dx[2:-2]].any(), (
        "on flat ground this edge belongs to the east neighbour, not this tile"
    )


# --- the reported bug ---------------------------------------------------------------


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
def test_a_building_sprite_is_never_crossed_by_a_grid_line(style, building_sprite) -> None:
    building = PlacedUnit(x=8.5, y=8.5, unit_const=BUILDING_CONST)
    with_unit = _scenario(units=[building])
    without = _scenario()

    def whole(scn, spec):
        cache = _BUILDERS[style](scn, sprites=True)
        cache.set_grid(spec)
        return _whole(cache).copy()

    sprite = np.any(whole(with_unit, DEFAULT_GRID) != whole(without, DEFAULT_GRID), axis=2)
    assert sprite.sum() > 1000, "the sprite never painted -- test would be vacuous"
    grid_under = np.any(whole(without, GRID) != whole(without, DEFAULT_GRID), axis=2)
    assert (grid_under & sprite).any(), "no grid line runs under the sprite -- test would be vacuous"

    diff = np.any(whole(with_unit, GRID) != whole(with_unit, DEFAULT_GRID), axis=2)
    assert not (diff & sprite).any(), f"{(diff & sprite).sum()} sprite pixels crossed by the grid"


# --- the cache ---------------------------------------------------------------------


def test_set_grid_evicts_and_an_equal_spec_is_a_no_op() -> None:
    cache = _flat_cache(_scenario())
    _whole(cache)
    cached = len(cache._cache)
    assert cached > 0
    cache.set_grid(DEFAULT_GRID)
    assert len(cache._cache) == cached
    cache.set_grid(GRID)
    assert cache.grid == GRID
    assert len(cache._cache) == 0
    _whole(cache)
    cache.set_grid(grid_overlay.grid_bake(True, -100, 1))
    assert len(cache._cache) == cached


@pytest.mark.parametrize("builder", [_iso_cache, _flat_cache])
def test_the_grid_reaches_every_mip_level(builder) -> None:
    cache = builder(_scenario(_bumpy))
    levels = cache.mip_levels()
    assert len(levels) > 1
    before = {mip: _whole(cache, mip).copy() for mip in levels}
    cache.set_grid(GRID)
    for mip in levels:
        assert not np.array_equal(before[mip], _whole(cache, mip)), f"level {mip} kept grid-free pixels"


def test_a_stepped_patch_after_an_elevation_edit_matches_a_fresh_composite() -> None:
    """The incremental-path guard: a neighbour's cliff-lip edge appears or
    vanishes with the edit, so the real dirty bbox has to reach it."""
    scn = _scenario(_bumpy)
    cache = _iso_cache(scn)
    cache.set_grid(THICK_GRID)
    _whole(cache)
    mm = scn.map_manager
    index = next(i for i, t in enumerate(mm.terrain) if (t.x, t.y) == (7, 7))
    mm.terrain[index].elevation += 2
    changed: set = set()
    bbox = render.dirty_screen_bbox_iso(scn, [index], cache.elevations, cache._levels[0].proj, elevation_changed=changed)
    assert bbox is not None
    cache.patch(bbox, elevation_changed=changed)
    stitched = _whole(cache)
    fresh = _iso_cache(scn)
    fresh.set_grid(THICK_GRID)
    assert np.array_equal(stitched, _whole(fresh))
    h, w = stitched.shape[:2]
    assert np.array_equal(stitched, _full("stepped", scn, THICK_GRID)[:h, :w])


def test_a_sloped_patch_after_an_elevation_edit_matches_a_fresh_composite() -> None:
    scn = _scenario(_bumpy)
    cache = _sloped_cache(scn)
    cache.set_grid(GRID)
    _whole(cache)
    mm = scn.map_manager
    index = next(i for i, t in enumerate(mm.terrain) if (t.x, t.y) == (7, 7))
    mm.terrain[index].elevation += 1
    changed: set = set()
    bbox = render.dirty_screen_bbox_sloped(scn, [index], cache.elevations, cache.proj, elevation_changed=changed)
    assert bbox is not None
    cache.patch(bbox, elevation_changed=changed)
    fresh = _sloped_cache(scn)
    fresh.set_grid(GRID)
    assert np.array_equal(_whole(cache), _whole(fresh))


def test_a_patch_after_a_unit_move_keeps_the_grid(building_sprite) -> None:
    """patch() recomposites through _composite_rect, which reads self.grid,
    so a moved unit's repaint cannot punch a grid-free hole."""
    building = PlacedUnit(x=8.5, y=8.5, unit_const=BUILDING_CONST)
    scn = _scenario(units=[building])
    cache = _iso_cache(scn, sprites=True)
    cache.set_grid(GRID)
    _whole(cache)
    building.x, building.y = 5.5, 10.5
    cache.invalidate_units()
    w, h = cache.canvas_dims(0)
    cache.patch((0, 0, w, h))
    fresh = _iso_cache(scn, sprites=True)
    fresh.set_grid(GRID)
    assert np.array_equal(_whole(cache), _whole(fresh))
    assert np.array_equal(_whole(cache), _full("stepped", scn, GRID, sprites=True)[:h, :w])
