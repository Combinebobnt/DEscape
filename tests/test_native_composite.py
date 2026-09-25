"""The native composite kernel against the numpy path it replaces, byte for
byte. Batch F's oracle: numpy is never re-baselined, so any difference here
is a kernel bug.

Both backends run in one process via composite_backend.use_backend(), over
the same cache's _composite_rect(), which is the exact call every chunk and
patch goes through. Skips when the extension isn't built, unless
DESCAPE_REQUIRE_NATIVE=1 (CI), where a missing or stale build fails instead.

Terrain textures are synthetic 512x512 arrays with distinct per-pixel
content per terrain id: CI has no AoE2 install, so without them only the
flat-colour branch would run, and a solid colour can't expose a wrong uv crop.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from test_farm_terrain import FAKE_FARM_CONST, fake_farm  # noqa: F401 -- fake_farm is a fixture
from test_sprite_edit_bbox import CONST, EDIT_X, EDIT_Y, Unit, _fixture_scenario, sprite_install  # noqa: F401

from descape import asset_source, composite_backend, iso_geometry, render, settings
from descape.grid_overlay import grid_bake
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units
from descape.view_layers import LayerState

TEXTURE_SIZE = 512
# A spread of real ids, so the textures-off branch gets distinct flat colours too.
TERRAIN_IDS = (0, 1, 2, 3, 6, 9, 10, 13)
FARM_X, FARM_Y = 40, 80
PLAIN_UNIT_CONST = 4  # no sprite in sprite_install's graphic map, so it draws a mark
RISES = range(1, iso_geometry.MAX_ELEVATION + 1)
TILE_PX_LADDER = (16, 32, 64, 128)


@pytest.fixture
def native_kernel():
    if composite_backend.available():
        return
    if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
    pytest.skip(composite_backend.unavailable_reason)


def _texture(terrain_id: int) -> np.ndarray:
    y, x = np.mgrid[0:TEXTURE_SIZE, 0:TEXTURE_SIZE]
    t = terrain_id + 1
    return np.stack(
        [(x * 7 + y * 13 + t * 31) % 256, (x * y + t * 17) % 256, (x ^ (y * t)) % 256], axis=-1
    ).astype(np.uint8)


@pytest.fixture
def synthetic_textures(monkeypatch):
    cache: dict[int, np.ndarray] = {}

    def fake(terrain_id: int) -> np.ndarray:
        if terrain_id not in cache:
            cache[terrain_id] = _texture(terrain_id)
        return cache[terrain_id]

    # set_install_path_override() calls this on the real lru_cache.
    fake.cache_clear = lambda: None
    monkeypatch.setattr(asset_source, "get_terrain_texture_array", fake)


def _assert_backends_agree(composite, rects) -> None:
    for rect in rects:
        with composite_backend.use_backend("numpy"):
            expected = composite(*rect)
        with composite_backend.use_backend("native"):
            got = composite(*rect)
        assert expected.any(), f"rect {rect} composited to all-black -- proves nothing"
        assert got.shape == expected.shape, rect
        if not np.array_equal(got, expected):
            ys, xs = np.nonzero(np.any(got != expected, axis=-1))
            pytest.fail(
                f"rect {rect}: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]}): "
                f"native {got[ys[0], xs[0]]} vs numpy {expected[ys[0], xs[0]]}"
            )


def _rects(cache, mip: int, anchors) -> list[tuple[int, int, int, int]]:
    """The chunk holding the canvas centre, one straddling chunk edges, and
    one around each anchor (a canvas-absolute tile origin at this mip). Pass
    the map's corner tiles as anchors to get rects hanging off canvas edges."""
    w, h = cache.canvas_dims(mip)
    c = cache.chunk_px
    cx, cy = (w // 2) // c * c, (h // 2) // c * c
    rects = [
        (cx, cy, cx + c, cy + c),
        (w // 2 - c // 2 - 7, h // 2 - c // 3, w // 2 + c // 2 + 5, h // 2 + c // 3 + 9),
    ]
    for ax, ay in anchors:
        rects.append((ax - c // 2 - 5, ay - c // 2 + 3, ax + c // 2 + 11, ay + c // 3))
    return rects


def _corner_tiles(mm) -> list[tuple[int, int]]:
    w, h = mm.map_width, mm.map_height
    return [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]


def _cliff_scenario():
    """sprite_install's template map with every tile at a random height, a
    terrain patchwork, the sprite unit, a plain unit mark and a farm. Random
    heights on 120x120 hit every rise 1..15 on every neighbour side; the
    coverage is asserted, not assumed."""
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    rng = np.random.default_rng(20260924)
    heights = rng.integers(0, iso_geometry.MAX_ELEVATION + 1, size=(mm.map_height, mm.map_width))
    for tile in mm.terrain:
        tile.elevation = int(heights[tile.y, tile.x])
        tile.terrain_id = TERRAIN_IDS[(tile.x // 3 + tile.y // 5) % len(TERRAIN_IDS)]
    scenario.unit_manager.units[1].append(Unit(float(EDIT_X - 8), float(EDIT_Y + 5), PLAIN_UNIT_CONST))
    scenario.unit_manager.units[2].append(Unit(FARM_X + 0.5, FARM_Y + 0.5, FAKE_FARM_CONST))
    for dy, dx in ((0, 1), (1, 0), (-1, 0), (0, -1), (-1, 1)):
        diffs = heights - np.roll(heights, (dy, dx), axis=(0, 1))
        assert set(RISES) <= set(np.unique(diffs).tolist()), f"fixture misses a rise on side {(dy, dx)}"
    return scenario


def _anchor_origins(proj, elevations, tiles):
    return [iso_geometry.tile_screen_origin(x, y, int(elevations[y, x]), proj) for x, y in tiles]


@pytest.mark.parametrize(
    ("sprites", "grid", "textures"),
    [(True, False, True), (False, True, True), (True, True, False), (False, False, False)],
    ids=["sprites", "grid", "sprites-grid-flat", "flat"],
)
def test_stepped_every_mip(native_kernel, sprite_install, fake_farm, synthetic_textures, sprites, grid, textures):  # noqa: F811
    scenario = _cliff_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(
        scenario, elevations, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height),
        sprites=sprites, layers=LayerState(terrain_textures=textures),
    )
    if grid:
        cache.set_grid(grid_bake(True, 60, 2))
    assert {cache.mip_tile_px(m) for m in cache.mip_levels()} == set(TILE_PX_LADDER)
    for mip in cache.mip_levels():
        lvl_proj = cache._level(mip).proj
        anchors = _anchor_origins(lvl_proj, elevations, [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y), *_corner_tiles(mm)])
        _assert_backends_agree(lambda *r, m=mip: cache._composite_rect(m, *r), _rects(cache, mip, anchors))


@pytest.mark.parametrize("pct", [settings.ELEV_STEP_PCT_MIN, settings.ELEV_STEP_PCT_MAX])
def test_stepped_elevation_step_extremes(native_kernel, sprite_install, fake_farm, synthetic_textures, monkeypatch, pct):  # noqa: F811
    """pct 200 empties every contact-shadow band; 25 makes the shallowest cliffs."""
    monkeypatch.setattr(settings, "get_elev_step_pct", lambda: pct)
    scenario = _cliff_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height))
    anchors = _anchor_origins(proj, elevations, [(EDIT_X, EDIT_Y), *_corner_tiles(mm)])
    _assert_backends_agree(lambda *r: cache._composite_rect(0, *r), _rects(cache, 0, anchors))


class _CountingKernel:
    def __init__(self, module):
        self._module = module
        self.calls = 0

    def paint_diamond(self, *args):
        self.calls += 1
        return self._module.paint_diamond(*args)


def test_native_path_is_actually_taken(native_kernel, synthetic_textures):
    """Guards the oracle itself: equal output means nothing if render.py
    never reaches the kernel."""
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height))
    with composite_backend.use_backend("native"):
        counter = _CountingKernel(composite_backend.native)
        composite_backend.native = counter  # use_backend restores it on exit
        w, h = cache.canvas_dims()
        cache._composite_rect(0, w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256)
    assert counter.calls > 0


def _sloped_scenario():
    """A +-1 hill on the template (Sloped's legal shape), with the same
    terrain patchwork and units as the Stepped fixture."""
    scenario = _fixture_scenario()
    mm = scenario.map_manager
    for tile in mm.terrain:
        d = max(abs(tile.x - EDIT_X), abs(tile.y - EDIT_Y))
        tile.elevation = max(0, iso_geometry.MAX_ELEVATION - d // 2)
        tile.terrain_id = TERRAIN_IDS[(tile.x // 3 + tile.y // 5) % len(TERRAIN_IDS)]
    scenario.unit_manager.units[1].append(Unit(float(EDIT_X - 8), float(EDIT_Y + 5), PLAIN_UNIT_CONST))
    scenario.unit_manager.units[2].append(Unit(FARM_X + 0.5, FARM_Y + 0.5, FAKE_FARM_CONST))
    return scenario


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
@pytest.mark.parametrize(("sprites", "grid", "textures"), [(True, True, True), (False, False, False)], ids=["all-on", "flat"])
def test_sloped_every_tile_px(native_kernel, sprite_install, fake_farm, synthetic_textures, monkeypatch, tile_px, sprites, grid, textures):  # noqa: F811
    """Sloped has one level per cache, so each tile_px on the Stepped ladder
    gets its own cache."""
    monkeypatch.setattr(render, "tile_pixels_for_map", lambda w, h: tile_px)
    scenario = _sloped_scenario()
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    cache = SlopedChunkCache(
        scenario, elevations, corner_rise, proj, tile_px, sprites=sprites, layers=LayerState(terrain_textures=textures),
    )
    if grid:
        cache.set_grid(grid_bake(True, 60, 2))
    anchors = _anchor_origins(proj, elevations, [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y), *_corner_tiles(scenario.map_manager)])
    _assert_backends_agree(lambda *r: cache._composite_rect(0, *r), _rects(cache, 0, anchors))


# --- index-producer destination uniqueness --------------------------------
#
# A numpy fancy-index read-modify-write reads each destination once; a C loop
# writing in place would darken or lerp a repeated destination twice. So no
# single producer call may repeat a destination pixel.


def _elev_steps(tile_px: int) -> set[int]:
    _half_w, half_h = iso_geometry.half_dims(tile_px)
    return {max(1, round(half_h * pct / 100)) for pct in settings.ELEV_STEP_PCT_STOPS}


def _assert_unique(dst_y, dst_x, what: str) -> None:
    if dst_y.size == 0:
        return
    key = (dst_y.astype(np.int64) - int(dst_y.min())) * (int(dst_x.max()) - int(dst_x.min()) + 1) + (
        dst_x.astype(np.int64) - int(dst_x.min())
    )
    assert np.unique(key).size == key.size, f"{what} repeats a destination pixel"


def _stepped_producer_calls(tile_px: int):
    yield "diamond_indices", iso_geometry.diamond_indices(tile_px)
    yield "seam_apex_indices", iso_geometry.seam_apex_indices(tile_px)
    for side in ("up_left", "up_right"):
        yield f"seam_edge_indices {side}", iso_geometry.seam_edge_indices(tile_px, side)
    for side in ("left", "right", "up_left", "up_right"):
        yield f"tile_edge_indices {side}", iso_geometry.tile_edge_indices(tile_px, side)
    for elev_step in sorted(_elev_steps(tile_px)):
        for rise in RISES:
            px = rise * elev_step
            for side in ("left", "right"):
                yield f"skirt_quad_indices {px} {side}", iso_geometry.skirt_quad_indices(tile_px, px, side)
            for side in ("up_left", "up_right"):
                yield f"shadow_quad_indices {px} {side}", iso_geometry.shadow_quad_indices(tile_px, px, side)
                yield f"shadow_tip_indices {px} {side}", iso_geometry.shadow_tip_indices(tile_px, px, side)
            for sides in ("both", "up_left", "up_right"):
                yield f"shadow_apex_indices {px} {sides}", iso_geometry.shadow_apex_indices(tile_px, px, sides)


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_stepped_index_producers_never_repeat_a_destination(tile_px):
    for what, arrays in _stepped_producer_calls(tile_px):
        _assert_unique(arrays[0], arrays[1], f"{what} at tile_px {tile_px}")


@pytest.mark.parametrize("tile_px", TILE_PX_LADDER)
def test_sloped_index_producers_never_repeat_a_destination(tile_px):
    """sloped_quad_indices' key space is open (corner rises are averages), so
    this sweeps a bounded sample: each corner at 0, one or two elevation
    steps, at the smallest and largest elev_step."""
    steps = sorted(_elev_steps(tile_px))
    for elev_step in (steps[0], steps[-1]):
        levels = (0, elev_step, 2 * elev_step)
        for d_nw in levels:
            for d_ne in levels:
                for d_sw in levels:
                    for d_se in levels:
                        corners = (d_nw, d_ne, d_sw, d_se)
                        arrays = iso_geometry.sloped_quad_indices(tile_px, *corners)
                        _assert_unique(arrays[0], arrays[1], f"sloped_quad_indices {corners}")
                        for side in ("left", "right", "up_left", "up_right"):
                            ey, ex = iso_geometry.sloped_tile_edge_indices(tile_px, side, *corners)
                            _assert_unique(ey, ex, f"sloped_tile_edge_indices {side} {corners}")


# --- corpus -----------------------------------------------------------------


@pytest.mark.corpus
def test_corpus_scenario_backends_agree(native_kernel, scenario_path):
    """Real maps, real textures when an install is configured. Stepped at every
    mip plus Sloped, sprites on, a handful of rects each."""
    scenario = load_map_and_units(str(scenario_path))
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    centre = [(mm.map_width // 2, mm.map_height // 2), (mm.map_width // 4, mm.map_height * 3 // 4)]

    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    for mip in cache.mip_levels():
        anchors = _anchor_origins(cache._level(mip).proj, elevations, centre + _corner_tiles(mm))
        _assert_backends_agree(lambda *r, m=mip: cache._composite_rect(m, *r), _rects(cache, mip, anchors))

    s_elevations, corner_rise, s_proj = render.sloped_elevations_and_proj(scenario)
    s_cache = SlopedChunkCache(scenario, s_elevations, corner_rise, s_proj, tile_px, sprites=True)
    anchors = _anchor_origins(s_proj, s_elevations, centre + _corner_tiles(mm))
    _assert_backends_agree(lambda *r: s_cache._composite_rect(0, *r), _rects(s_cache, 0, anchors))


# --- the backend switch itself (no native build needed) ---------------------


def test_use_backend_restores_the_previous_backend():
    before = composite_backend.native
    with composite_backend.use_backend("numpy"):
        assert composite_backend.active_backend() == "numpy"
    assert composite_backend.native is before


def test_use_backend_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown composite backend"), composite_backend.use_backend("cuda"):
        pass


def test_a_stale_kernel_abi_falls_back_to_numpy(monkeypatch):
    stale = type(composite_backend)("descape._composite_native")
    stale.KERNEL_ABI = composite_backend.EXPECTED_KERNEL_ABI - 1
    monkeypatch.setitem(sys.modules, "descape._composite_native", stale)
    import descape

    monkeypatch.setattr(descape, "_composite_native", stale, raising=False)
    module, reason = composite_backend._load()
    assert module is None
    assert "stale" in reason
