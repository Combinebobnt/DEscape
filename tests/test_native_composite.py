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

from descape import asset_source, composite_backend, iso_geometry, native_composite, render, settings
from descape.grid_overlay import grid_bake
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice
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


@pytest.mark.parametrize("sprites", [True, False], ids=["sprites", "marks"])
@pytest.mark.parametrize(("grid", "textures"), [(True, True), (False, False)], ids=["grid", "flat"])
def test_flat_every_mip(native_kernel, sprite_install, fake_farm, synthetic_textures, sprites, grid, textures):  # noqa: F811
    """Flat shares the rgba blit (icons) and the grid lerp with the iso paths."""
    scenario = _cliff_scenario()
    mm = scenario.map_manager
    cache = FlatChunkCache(
        scenario, render.tile_pixels_for_map(mm.map_width, mm.map_height),
        sprites=sprites, layers=LayerState(terrain_textures=textures),
    )
    if grid:
        cache.set_grid(grid_bake(True, 60, 2))
    for mip in cache.mip_levels():
        px = cache.mip_tile_px(mip)
        anchors = [(x * px, y * px) for x, y in [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y), *_corner_tiles(mm)]]
        _assert_backends_agree(lambda *r, m=mip: cache._composite_rect(m, *r), _rects(cache, mip, anchors))


class _CountingKernel:
    """Counts calls per kernel entry point, passing each through."""

    def __init__(self, module):
        self._module = module
        self.calls: dict[str, int] = {}

    def __getattr__(self, name):
        fn = getattr(self._module, name)

        def counted(*args):
            self.calls[name] = self.calls.get(name, 0) + 1
            return fn(*args)

        return counted


def _count_native_calls(composite, dims) -> dict[str, int]:
    w, h = dims
    with composite_backend.use_backend("native"):
        counter = _CountingKernel(composite_backend.native)
        composite_backend.native = counter  # use_backend restores it on exit
        for x, y in ((w // 4, h // 4), (w // 2, h // 2), (3 * w // 4, 3 * h // 4)):
            composite(x - 512, y - 512, x + 512, y + 512)
    return counter.calls


def _per_tile_iso(cache, mip: int = 0):
    """composite_rect_iso over the cache's own level state but with no unit
    pack: the per-tile native path every non-cache caller takes, and the N2
    fallback."""
    lvl = cache._level(mip)
    return lambda *r: render.composite_rect_iso(
        cache.scenario, *r, cache.elevations, lvl.proj, lvl.tile_px, cache.units_by_tile, lvl.building_bboxes,
        cache.with_units, sprites=lvl.sprites, bystander_grid=lvl.bystander_grid, layers=cache.layers, grid=cache.grid,
    )


def _per_tile_sloped(cache):
    return lambda *r: render.composite_rect_sloped(
        cache.scenario, *r, cache.corner_rise, cache.proj, cache.tile_px, cache.units_by_tile, cache.building_bboxes,
        cache.with_units, sprites=cache.sprites, bystander_grid=cache.bystander_grid, layers=cache.layers, grid=cache.grid,
    )


def _all_on_caches(tile_px: int):
    scenario = _cliff_scenario()
    elevations, proj = render.elevations_and_proj(scenario)
    stepped = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
    stepped.set_grid(grid_bake(True, 60, 2))
    s_scenario = _sloped_scenario()
    s_elevations, corner_rise, s_proj = render.sloped_elevations_and_proj(s_scenario)
    sloped = SlopedChunkCache(s_scenario, s_elevations, corner_rise, s_proj, tile_px, sprites=True)
    sloped.set_grid(grid_bake(True, 60, 2))
    return scenario, stepped, sloped


def test_native_path_is_actually_taken(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """Guards the oracle itself: equal output means nothing if render.py
    never reaches the kernel. A cache composite is exactly one whole-rect
    call, never the per-tile fallback; a pack-less composite takes every
    per-tile entry point."""
    mm = _cliff_scenario().map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    scenario, stepped, sloped = _all_on_caches(tile_px)

    calls = _count_native_calls(lambda *r: stepped._composite_rect(0, *r), stepped.canvas_dims())
    assert calls.keys() == {"composite_iso"}, calls
    calls = _count_native_calls(_per_tile_iso(stepped), stepped.canvas_dims())
    assert {"render_tile_iso", "paint_solid", "lerp", "blit_rgba"} <= calls.keys(), calls

    calls = _count_native_calls(lambda *r: sloped._composite_rect(0, *r), sloped.canvas_dims())
    assert calls.keys() == {"composite_sloped"}, calls
    calls = _count_native_calls(_per_tile_sloped(sloped), sloped.canvas_dims())
    assert {"paint_sloped", "paint_solid", "lerp", "blit_rgba"} <= calls.keys(), calls

    flat = FlatChunkCache(scenario, tile_px, sprites=True)
    flat.set_grid(grid_bake(True, 60, 2))
    calls = _count_native_calls(lambda *r: flat._composite_rect(0, *r), flat.canvas_dims())
    assert {"lerp", "blit_rgba"} <= calls.keys(), calls


def test_per_tile_path_backends_agree(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """N1's per-tile kernel stays covered: it is every pack-less caller's path
    and the whole-rect path's fallback."""
    mm = _cliff_scenario().map_manager
    _scenario, stepped, sloped = _all_on_caches(render.tile_pixels_for_map(mm.map_width, mm.map_height))
    for mip in stepped.mip_levels():
        anchors = _anchor_origins(stepped._level(mip).proj, stepped.elevations, [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y)])
        _assert_backends_agree(_per_tile_iso(stepped, mip), _rects(stepped, mip, anchors))
    anchors = _anchor_origins(sloped.proj, sloped.elevations, [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y)])
    _assert_backends_agree(_per_tile_sloped(sloped), _rects(sloped, 0, anchors))


def test_a_rise_past_the_tables_falls_back_per_tile(native_kernel, sprite_install, fake_farm, synthetic_textures, monkeypatch):  # noqa: F811
    """The kernel refuses a rise its tables don't hold, painting nothing, and
    the rect is composited per tile instead: same pixels, both paths called."""
    monkeypatch.setattr(native_composite, "MAX_RISE", 1)
    native_composite.stepped_tables.cache_clear()
    try:
        mm = _cliff_scenario().map_manager
        _scenario, stepped, _sloped = _all_on_caches(render.tile_pixels_for_map(mm.map_width, mm.map_height))
        composite = lambda *r: stepped._composite_rect(0, *r)  # noqa: E731
        calls = _count_native_calls(composite, stepped.canvas_dims())
        assert {"composite_iso", "render_tile_iso"} <= calls.keys(), calls
        anchors = _anchor_origins(stepped.proj, stepped.elevations, [(EDIT_X, EDIT_Y), (FARM_X, FARM_Y)])
        _assert_backends_agree(composite, _rects(stepped, 0, anchors))
    finally:
        native_composite.stepped_tables.cache_clear()


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


# --- N2 invalidation: edits through the real funnels ------------------------

NEW_TERRAIN_ID = 17  # not in TERRAIN_IDS, so a paint adds a texture the map never had


def _edit_cache(style: str, scenario, sprites: bool):
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites)
    else:
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=sprites)
    cache.set_grid(grid_bake(True, 60, 2))
    return cache


def _patch_tiles(cache, style: str, scenario, tiles, sprites: bool) -> None:
    """The viewer's edit funnel: dirty bbox (which syncs cache.elevations and
    reports the tiles whose elevation moved), then patch()."""
    w = scenario.map_manager.map_width
    dirty = [y * w + x for x, y in tiles]
    changed: set = set()
    bbox_fn = render.dirty_screen_bbox_iso if style == "stepped" else render.dirty_screen_bbox_sloped
    bbox = bbox_fn(
        scenario, dirty, cache.elevations, cache.proj, with_units=True, with_sprites=sprites,
        elevation_changed=changed,
    )
    assert bbox is not None
    cache.patch(bbox, elevation_changed=changed)


def _assert_matches_fresh(cache, style: str, scenario, sprites: bool, what: str) -> None:
    mips = [m for m in cache.mip_levels() if m in (0, -1)]
    with composite_backend.use_backend("numpy"):
        fresh = _edit_cache(style, scenario, sprites)
        want = {m: fresh.render_rect(0, 0, *fresh.canvas_dims(m), mip=m) for m in mips}
    with composite_backend.use_backend("native"):
        for m in mips:
            got = cache.render_rect(0, 0, *cache.canvas_dims(m), mip=m)
            if not np.array_equal(got, want[m]):
                ys, xs = np.nonzero(np.any(got != want[m], axis=-1))
                pytest.fail(f"{style} mip {m} after {what}: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]})")


def _pack_of(cache, style: str):
    return cache._levels[0].unit_pack if style == "stepped" else cache._unit_pack


@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
def test_edits_through_the_real_funnels_match_a_fresh_render(
    native_kernel, sprite_install, fake_farm, synthetic_textures, style, sprites,  # noqa: F811
):
    """One live native cache, warm at every compared level, taken through a
    terrain paint of an id the map never had, elevation edits beside a grid
    edge and under a mark, a sprite and a farm, and unit moves. After each,
    the whole canvas must equal a fresh numpy render of the edited scenario.
    The unit pack must survive the elevation and move edits as a refresh
    (same object), so the splice-tile path is what these compare."""
    scenario = _sloped_scenario()
    mm = scenario.map_manager
    mark = scenario.unit_manager.units[1][-1]
    farm = scenario.unit_manager.units[2][-1]
    assert mark.unit_const == PLAIN_UNIT_CONST and farm.unit_const == FAKE_FARM_CONST
    with composite_backend.use_backend("native"):
        cache = _edit_cache(style, scenario, sprites)
        for m in (0, -1) if style == "stepped" else (0,):
            cache.render_rect(0, 0, *cache.canvas_dims(m), mip=m)
    assert _pack_of(cache, style) is not None
    refreshed = []

    def elevate(x, y, what):
        tile = mm.terrain[y * mm.map_width + x]
        tile.elevation += 1 if tile.elevation < iso_geometry.MAX_ELEVATION else -1
        pack, headroom = _pack_of(cache, style), getattr(cache, "_headroom", None)
        with composite_backend.use_backend("native"):
            _patch_tiles(cache, style, scenario, [(x, y)], sprites)
        _assert_matches_fresh(cache, style, scenario, sprites, what)
        # A Sloped headroom change takes the wholesale path, which must drop the pack.
        if getattr(cache, "_headroom", None) == headroom:
            assert _pack_of(cache, style) is pack, f"{what} rebuilt the unit pack instead of refreshing it"
            refreshed.append(what)

    painted = [(EDIT_X + dx, EDIT_Y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    painted += [(int(mark.x), int(mark.y)), (FARM_X - 1, FARM_Y)]
    for x, y in painted:
        mm.terrain[y * mm.map_width + x].terrain_id = NEW_TERRAIN_ID
    with composite_backend.use_backend("native"):
        _patch_tiles(cache, style, scenario, painted, sprites)
    _assert_matches_fresh(cache, style, scenario, sprites, "a paint of a new terrain id")

    # (x-1, y) owns its x_high grid edge only while the two heights differ.
    elevate(EDIT_X - 3, EDIT_Y + 2, "an elevation edit flipping a -x neighbour's grid edge")
    elevate(EDIT_X - 2, EDIT_Y - 4, "an elevation edit flipping a -y neighbour's grid edge")
    elevate(int(mark.x), int(mark.y), "an elevation edit under a mark")
    elevate(EDIT_X, EDIT_Y, "an elevation edit under the sprite unit")
    elevate(FARM_X, FARM_Y, "an elevation edit under a farm")
    if style == "sloped":
        elevate(int(mark.x) + 1, int(mark.y) - 1, "an elevation edit moving a mark's corner")
    assert len(refreshed) >= 4, refreshed
    pack = _pack_of(cache, style)

    for unit, player_id, (nx, ny), what in (
        (mark, 1, (EDIT_X - 12.25, EDIT_Y + 7.75), "an off-centre mark move"),
        (farm, 2, (FARM_X + 4.5, FARM_Y - 3.5), "a farm move"),
    ):
        index = scenario.unit_manager.units[player_id].index(unit)
        old_own = (int(unit.x), int(unit.y))
        old_tiles = tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height))
        unit.x, unit.y = nx, ny
        scenario.unit_gen += 1
        splice = UnitSplice(
            player_id, index, unit, old_own, (int(nx), int(ny)), old_tiles,
            tuple(render.unit_occupied_tiles(unit, mm.map_width, mm.map_height)),
        )
        with composite_backend.use_backend("native"):
            assert cache.can_splice([splice])
            cache.invalidate_units([splice])
            cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        _assert_matches_fresh(cache, style, scenario, sprites, what)
    assert _pack_of(cache, style) is pack, "a unit splice rebuilt the unit pack instead of refreshing it"


def _pack_rows(pack) -> dict[int, list[tuple]]:
    idx = pack.index.reshape(-1, 2)
    return {
        t: [tuple(r) for r in pack.rows[s : s + n].tolist()]
        for t, (s, n) in enumerate(idx.tolist()) if n
    }


def test_unit_pack_compaction_keeps_every_tiles_rows(sprite_install, fake_farm):  # noqa: F811
    """refresh() orphans a rewritten tile's old rows; _compact() must drop
    exactly those, keeping every live tile's rows in order."""
    scenario = _sloped_scenario()
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    lvl = cache._level(0)
    pack = native_composite.UnitPack(
        False, mm.map_width, mm.map_height, lvl.proj, cache.units_by_tile, lvl.sprites, 0, cache.elevations,
    )
    everything = np.arange(mm.map_width * mm.map_height)
    pack.ensure(everything)
    before = _pack_rows(pack)
    assert before, "fixture has no unit rows"
    tiles = {(t % mm.map_width, t // mm.map_width) for t in before}
    for _ in range(3):
        pack.refresh(tiles, cache.units_by_tile, lvl.sprites, 0, cache.elevations)
        assert not pack.ready[list(before)].any()
        pack.ensure(everything)
    assert pack.garbage > 0
    assert _pack_rows(pack) == before
    pack._compact()
    assert pack.garbage == 0 and pack.n == sum(len(v) for v in before.values())
    assert _pack_rows(pack) == before


# --- worker-thread chunk jobs (Batch F T2) ----------------------------------


def _chunk_of_tile(cache, x: int, y: int) -> tuple[int, int, int]:
    ox, oy = iso_geometry.tile_screen_origin(x, y, int(cache.elevations[y, x]), cache.proj)
    return 0, ox // cache.chunk_px, oy // cache.chunk_px


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_worker_threads_warm_every_chunk_byte_identical(
    native_kernel, sprite_install, fake_farm, synthetic_textures, style,  # noqa: F811
):
    """A whole grid warmed through MarginWarmer on the real pool equals a
    synchronous get_chunk(), chunk for chunk, and the pooled path was taken."""
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import margin_warm

    import conftest

    conftest.ensure_qapp()
    scenario = _cliff_scenario() if style == "stepped" else _sloped_scenario()
    with composite_backend.use_backend("native"):
        warm = _edit_cache(style, scenario, sprites=True)
        sync = _edit_cache(style, scenario, sprites=True)
        installed = []
        real_install = warm.install_chunk

        def install(job, painted):
            landed = real_install(job, painted)
            if landed:
                installed.append(job.key)
            return landed

        warm.install_chunk = install
        for mip in [m for m in warm.mip_levels() if m in (0, -1)]:
            if style == "stepped":
                warm._level(mip)
            gw, gh = warm.canvas_dims(mip)
            keys = [(mip, cx, cy) for cy in range(-(-gh // warm.chunk_px)) for cx in range(-(-gw // warm.chunk_px))]
            warmer = margin_warm.MarginWarmer()
            warmer.start(warm, mip, [key[1:] for key in keys])
            deadline = time.monotonic() + 60
            while warmer.is_active:
                assert time.monotonic() < deadline, "the pooled warm never drained"
                QApplication.processEvents()
                time.sleep(0.001)
            assert any(key[0] == mip for key in installed), f"mip {mip}: no chunk came from a worker -- vacuous"
            for key in keys:
                assert warm.has_chunk(*key), f"{key} was never warmed"
                if not np.array_equal(warm._cache[key], sync.get_chunk(*key)):
                    pytest.fail(f"{style} chunk {key} from a worker differs from a synchronous composite")


def test_a_job_reads_the_elevations_it_was_prepared_with(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """Prepared, then an elevation edit, then run: the pixels are the
    pre-edit chunk (the job's snapshot), and install refuses them."""
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache("stepped", scenario, sprites=True)
        key = _chunk_of_tile(cache, EDIT_X, EDIT_Y)
        before = _edit_cache("stepped", scenario, sprites=True).get_chunk(*key).copy()
        job = cache.prepare_chunk_job(*key)
        assert job is not None

        mm = scenario.map_manager
        for x, y in ((EDIT_X, EDIT_Y), (EDIT_X + 1, EDIT_Y)):
            tile = mm.terrain[y * mm.map_width + x]
            tile.elevation = (tile.elevation + 7) % (iso_geometry.MAX_ELEVATION + 1)
        _patch_tiles(cache, "stepped", scenario, [(EDIT_X, EDIT_Y), (EDIT_X + 1, EDIT_Y)], True)
        after = _edit_cache("stepped", scenario, sprites=True).get_chunk(*key)
        assert not np.array_equal(before, after), "the edit didn't change this chunk -- vacuous"

        assert job.run()
        assert np.array_equal(job.scratch, before), "the job read the live elevations, not its snapshot"
        assert not cache.install_chunk(job, True), "a job prepared before a patch() was installed"
        assert not cache.has_chunk(*key)

        fresh = cache.prepare_chunk_job(*key)
        assert fresh.run() and cache.install_chunk(fresh, True)
        assert np.array_equal(cache.get_chunk(*key), after)


def test_a_mutation_entry_point_rejects_a_prepared_job(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """Every funnel that can change chunk pixels bumps the epoch."""
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache("stepped", scenario, sprites=True)
        key = _chunk_of_tile(cache, EDIT_X, EDIT_Y)
        mutations = {
            "invalidate_region": lambda: cache.invalidate_region((0, 0, 1, 1)),
            "invalidate_units": cache.invalidate_units,
            "set_grid": lambda: cache.set_grid(grid_bake(False, 60, 2)),
            "set_layers": lambda: cache.set_layers(LayerState(terrain_textures=not cache.layers.terrain_textures)),
            "set_sprites_enabled": lambda: cache.set_sprites_enabled(not cache.sprites_enabled),
            "patch": lambda: cache.patch((0, 0, 1, 1), elevation_changed=set()),
        }
        for name, mutate in mutations.items():
            job = cache.prepare_chunk_job(*key)
            assert job is not None and job.run()
            mutate()
            assert not cache.install_chunk(job, True), f"{name} did not reject an in-flight job"
        job = cache.prepare_chunk_job(*key)
        assert job.run() and cache.install_chunk(job, True), "control: an unmutated job must install"
        assert not cache.install_chunk(job, True), "an already-cached chunk was replaced"


def test_a_shared_unit_pack_is_copied_before_it_is_written(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """A later chunk's lazy derivation must not rewrite the index a worker
    may still be reading: a torn read there indexes past its rows."""
    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache("stepped", scenario, sprites=True)
        first = cache.prepare_chunk_job(*_chunk_of_tile(cache, EDIT_X, EDIT_Y))
        second_key = _chunk_of_tile(cache, FARM_X, FARM_Y)
        assert second_key != first.key, "both units share a chunk -- vacuous"
        shared = first.args[-1]
        index, mask, rgb = shared[0], shared[4], shared[5]
        frozen = [a.copy() for a in (index, mask, rgb)]

        cache.prepare_chunk_job(*second_key)
        pack = cache._levels[0].unit_pack
        assert pack.farm_mask.any() and not frozen[1].any(), "the farm wasn't derived by the second chunk -- vacuous"
        for name, held, was in zip(("index", "farm_mask", "farm_rgb"), (index, mask, rgb), frozen, strict=True):
            assert np.array_equal(held, was), f"{name} was written in place while shared"
        assert first.run()


def test_a_job_keeps_its_unit_pack_alive_after_the_cache_drops_it(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    """The kernel blits sprites by raw address into memory only the pack
    owns, and the GUI thread drops packs (a rebuild, a level install) while
    a worker may still be running."""
    import gc
    import weakref

    scenario = _cliff_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache("stepped", scenario, sprites=True)
        key = _chunk_of_tile(cache, EDIT_X, EDIT_Y)
        want = _edit_cache("stepped", scenario, sprites=True).get_chunk(*key).copy()
        job = cache.prepare_chunk_job(*key)
        lvl = cache._levels[0]
        pack = weakref.ref(lvl.unit_pack)
        assert pack()._rgbas, "no sprite in this chunk's pack -- vacuous"
        lvl.unit_pack = lvl.sprites = None
        del cache, lvl
        gc.collect()

        assert pack() is not None, "the job let its pack be freed under the kernel"
        assert job.run() and np.array_equal(job.scratch, want)
        del job
        gc.collect()
        assert pack() is None, "control: nothing else should hold the pack"


# --- unit-pack pre-derive (maintainer plan 2026-09-27) ----------------------


def _warm_mips(style: str, cache) -> list[int]:
    return [m for m in cache.mip_levels() if m in (0, -1, -2)] if style == "stepped" else [0]


def _drain_pack_warm(cache, mip: int) -> None:
    job = cache.pack_warm_job(mip)
    assert job is not None, f"mip {mip}: nothing to pre-derive -- vacuous"
    for _ in job.gen:
        pass


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pack_warm_composites_byte_identical(
    native_kernel, sprite_install, fake_farm, synthetic_textures, monkeypatch, style,  # noqa: F811
):
    """A completed pack warm, then the whole canvas natively, equals a fresh
    numpy render and a lazy native one, using the pack the warm derived into.
    Small slices so slice boundaries land mid-map."""
    from descape import render_cache

    monkeypatch.setattr(render_cache, "PACK_WARM_TILES", 3)
    scenario = _sloped_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache(style, scenario, sprites=True)
        lazy = _edit_cache(style, scenario, sprites=True)
        for m in _warm_mips(style, cache):
            if style == "stepped":
                cache._level(m)
            assert cache.pack_warm_job(m, after_level_warm=False) is not None
            _drain_pack_warm(cache, m)
            pack = cache._unit_pack_of(m, create=False)
            assert pack is not None and pack.ready[pack.has].all(), f"mip {m}: tiles left pending"
            got = cache.render_rect(0, 0, *cache.canvas_dims(m), mip=m)
            assert cache._unit_pack_of(m, create=False) is pack, f"mip {m}: the composite rebuilt the warmed pack"
            want_lazy = lazy.render_rect(0, 0, *lazy.canvas_dims(m), mip=m)
            with composite_backend.use_backend("numpy"):
                fresh = _edit_cache(style, scenario, sprites=True)
                want = fresh.render_rect(0, 0, *fresh.canvas_dims(m), mip=m)
            assert np.array_equal(want_lazy, want), f"control: lazy native differs from numpy at mip {m}"
            if not np.array_equal(got, want):
                ys, xs = np.nonzero(np.any(got != want, axis=-1))
                pytest.fail(f"{style} mip {m} after a pack warm: {ys.size} px differ, first at (y={ys[0]}, x={xs[0]})")
            assert pack.n > 0, f"mip {m}: no rows derived -- vacuous"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_pack_warm_then_splices_matches_a_fresh_render(
    native_kernel, sprite_install, fake_farm, synthetic_textures, style,  # noqa: F811
):
    """Pre-derived, then an Elevate under a mark and a unit move through the
    real funnels: the refreshed pack still composites like a fresh render."""
    scenario = _sloped_scenario()
    mm = scenario.map_manager
    mark = scenario.unit_manager.units[1][-1]
    assert mark.unit_const == PLAIN_UNIT_CONST
    with composite_backend.use_backend("native"):
        cache = _edit_cache(style, scenario, sprites=True)
        for m in (0, -1) if style == "stepped" else (0,):
            if style == "stepped":
                cache._level(m)
            _drain_pack_warm(cache, m)
            cache.render_rect(0, 0, *cache.canvas_dims(m), mip=m)
        pack = _pack_of(cache, style)

        tile = mm.terrain[int(mark.y) * mm.map_width + int(mark.x)]
        tile.elevation += 1 if tile.elevation < iso_geometry.MAX_ELEVATION else -1
        _patch_tiles(cache, style, scenario, [(int(mark.x), int(mark.y))], True)
    _assert_matches_fresh(cache, style, scenario, True, "an elevation edit under a mark after a pack warm")

    index = scenario.unit_manager.units[1].index(mark)
    old_own = (int(mark.x), int(mark.y))
    old_tiles = tuple(render.unit_occupied_tiles(mark, mm.map_width, mm.map_height))
    mark.x, mark.y = EDIT_X - 12.25, EDIT_Y + 7.75
    scenario.unit_gen += 1
    splice = UnitSplice(
        1, index, mark, old_own, (int(mark.x), int(mark.y)), old_tiles,
        tuple(render.unit_occupied_tiles(mark, mm.map_width, mm.map_height)),
    )
    with composite_backend.use_backend("native"):
        assert cache.can_splice([splice])
        cache.invalidate_units([splice])
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
    _assert_matches_fresh(cache, style, scenario, True, "a mark move after a pack warm")
    assert _pack_of(cache, style) is pack, "a splice rebuilt the warmed pack instead of refreshing it"


def _started_pack_warm(cache, mip: int):
    """A pack warm past its first step (pack fetched, pending snapshotted)."""
    job = cache.pack_warm_job(mip)
    assert job is not None
    next(job.gen)
    pack = cache._unit_pack_of(mip, create=False)
    assert pack is not None and pack.pending().size, "nothing pending -- vacuous"
    return job, pack


@pytest.mark.parametrize(
    "case", ["stepped-wholesale", "stepped-rebuilt-level", "sloped-wholesale"],
)
def test_a_pack_warm_stops_when_its_pack_is_replaced(native_kernel, sprite_install, fake_farm, synthetic_textures, case):  # noqa: F811
    """No _cancel_warms() (the second line of defence): a wholesale mutation
    between steps stops the walk without deriving into the stale pack. The
    rebuilt-level case is resident again by the next step, so only the pack
    identity check can stop it; Sloped is always resident."""
    style = case.split("-")[0]
    scenario = _sloped_scenario()
    with composite_backend.use_backend("native"):
        cache = _edit_cache(style, scenario, sprites=True)
        if style == "stepped":
            cache._level(0)
        job, pack = _started_pack_warm(cache, 0)
        ready = pack.ready.copy()

        cache.invalidate_units()
        if case == "stepped-rebuilt-level":
            cache._level(0)
            assert cache.is_level_resident(0)

        with pytest.raises(StopIteration):
            next(job.gen)
        assert np.array_equal(pack.ready, ready), "the walk derived into a pack its level no longer uses"
        replacement = cache._unit_pack_of(0, create=False)
        assert replacement is None or not replacement.ready.any(), "the walk wrote into the replacement pack"


def test_pack_warm_has_nothing_to_do(native_kernel, sprite_install, fake_farm, synthetic_textures):  # noqa: F811
    scenario = _sloped_scenario()
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = render.elevations_and_proj(scenario)
    with composite_backend.use_backend("native"):
        stepped = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True)
        stepped._level(0)
        assert stepped.pack_warm_job(0) is not None, "control: a resident level with pending tiles"
        assert stepped.pack_warm_job(-1) is None, "a standalone warm of a not-resident level"
        assert stepped.pack_warm_job(-1, after_level_warm=True) is not None
        _drain_pack_warm(stepped, 0)
        assert stepped.pack_warm_job(0) is None, "a fully derived level"
        no_units = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True, with_units=False)
        no_units._level(0)
        assert no_units.pack_warm_job(0) is None
        assert FlatChunkCache(scenario, tile_px, sprites=True).pack_warm_job(0) is None
        s_elev, corner_rise, s_proj = render.sloped_elevations_and_proj(scenario)
        sloped = SlopedChunkCache(scenario, s_elev, corner_rise, s_proj, tile_px, sprites=True)
        _drain_pack_warm(sloped, 0)
        assert sloped.pack_warm_job(0) is None, "a fully derived Sloped level"
    with composite_backend.use_backend("numpy"):
        assert IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True).pack_warm_job(0, True) is None
        assert SlopedChunkCache(scenario, s_elev, corner_rise, s_proj, tile_px, sprites=True).pack_warm_job(0) is None


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
