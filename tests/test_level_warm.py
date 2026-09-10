"""Incremental level warm -- the sliced walks and the QTimer driver behind
them (maintainer plan 2026-09-04).

**The regression risk this file exists for is Step 1, not the driver.**
`sprite_draws_by_anchor` and `_flat_icon_layer` were each split into a
resumable generator plus a thin drain, so that a warm can advance a few
milliseconds at a time. A split that changes iteration ORDER (rather than
output) is invisible to every existing test and silently wrong in exactly one
place: `farm_by_tile`'s documented "later unit wins on overlap" semantics.
Hence the equality checks below compare the sliced walk against the whole one
on a fixture that deliberately contains overlapping farms.

Comparison is STRUCTURAL, never `==` on the layer itself: `SpriteLayer` is a
plain dataclass whose `by_anchor` holds `SpriteDraw`s wrapping numpy arrays,
so `==` either raises "truth value of an array is ambiguous" or passes only
by LRU object identity -- i.e. it would go green for the wrong reason.

Synthetic bytes and a tmp install for the default tier, matching the suite's
standing posture; the corpus-marked checks re-run the same equality on real
files, where the unit mix is one nobody wrote a fixture for.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import asset_source, level_warm, render, unit_sprites
from descape.scenario_io import load_map_and_units
from descape.terrain_palette import PLAYER_COLORS
from test_unit_sprites import CONST, FILE_NAME, build_sld

MAP_W = MAP_H = 12
# A farm-family const with a real foundation_terrain_id and no sprite, so it
# takes sprite_draws_by_anchor's farm branch rather than its sprite branch --
# see tests/test_farm_terrain.py, which pins this same table.
FARM_CONST = 50


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
    """tests/test_sprite_chunks.py's duck-type, kept as its own copy for the
    reason that module already records."""

    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player):
    tiles = [Tile(x, y, 0) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """tests/test_sprite_chunks.py's fixture -- one synthetic sprite big
    enough to reach past its own tile."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4, canvas=4 * unit_sprites.NATIVE_TILE_W))
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


@pytest.fixture
def mixed_scenario():
    """Sprites, overlapping farms, an off-map unit and two players -- one of
    each branch the per-unit body can take, so a slice boundary landing on any
    of them is covered.

    The two farms at (2,2) and (3,3) OVERLAP (3x3 footprints), which is what
    makes farm_by_tile's later-wins rule observable at all: on a
    non-overlapping fixture an order-reversing split still produces an
    identical dict."""
    return _scenario([
        [Unit(2.5, 2.5, FARM_CONST), Unit(3.5, 3.5, FARM_CONST)],       # GAIA
        [Unit(6.5, 6.5, CONST), Unit(8.5, 4.5, CONST, rotation=1.5)],   # player 1
        [Unit(-40.0, -40.0, CONST), Unit(9.5, 9.5, CONST)],             # player 2
    ])


def _elevations_and_proj(scn):
    """render_terrain_iso_with_proj's own two outputs -- the same pair the
    live IsoChunkCache feeds sprite_draws_by_anchor, rather than a
    hand-built projection that could drift from it."""
    _img, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    return elevations, proj


def _draw_key(draw):
    """A SpriteDraw as comparable plain data -- pixels included, so this can't
    pass by object identity out of unit_sprites' own LRU."""
    return (draw.rgba.shape, draw.rgba.tobytes(), draw.hotspot_x, draw.hotspot_y)


def _layer_key(layer):
    return (
        {anchor: [(_draw_key(draw), px, py) for draw, px, py in slot] for anchor, slot in layer.by_anchor.items()},
        layer.bboxes,
        layer.skip_ids,
        layer.farm_by_tile,
    )


def _icons_key(icons_and_rows):
    icons, rows = icons_and_rows
    return ({row: _draw_key(draw) for row, draw in icons.items()}, rows)


def _step_all(gen):
    """Drives a sliced walk one yield at a time, the way LevelWarmer's budget
    loop does, and returns (payload, yields)."""
    yields = 0
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value, yields
        yields += 1


# --- Step 1: the sliced walks must equal the whole ones --------------------


def test_sliced_sprite_walk_equals_the_whole_one(sprite_install, mixed_scenario) -> None:
    elevations, proj = _elevations_and_proj(mixed_scenario)
    whole = render.sprite_draws_by_anchor(mixed_scenario, proj, elevations)
    sliced, _ = _step_all(render.sprite_draws_by_anchor_sliced(mixed_scenario, proj, elevations))

    assert whole.by_anchor, "vacuous: the fixture resolved no sprites at all"
    assert whole.farm_by_tile, "vacuous: the fixture resolved no farm overrides"
    assert _layer_key(sliced) == _layer_key(whole)


def test_the_sprite_walk_yields_once_per_unit(sprite_install, mixed_scenario) -> None:
    """Granularity, not output: a yield placed after the filter/bounds
    `continue`s would still produce the right layer while letting a run of
    skipped units blow straight past the tick budget."""
    elevations, proj = _elevations_and_proj(mixed_scenario)
    _, yields = _step_all(render.sprite_draws_by_anchor_sliced(mixed_scenario, proj, elevations))
    assert yields == sum(len(units) for units in mixed_scenario.unit_manager.units)


def test_sliced_flat_icon_walk_equals_the_whole_one(sprite_install, mixed_scenario) -> None:
    whole = render._flat_icon_layer(mixed_scenario, 32)
    sliced, yields = _step_all(render._flat_icon_layer_sliced(mixed_scenario, 32))

    assert whole[0], "vacuous: the fixture resolved no icons at all"
    assert _icons_key(sliced) == _icons_key(whole)
    # The ROW COUNT is half the payload -- FlatChunkCache._level_icons asserts
    # on it, so a sliced walk that returned only the dict would disarm that
    # guard on the warm path alone.
    assert sliced[1] == whole[1]
    assert yields == sum(len(units) for units in mixed_scenario.unit_manager.units)


def test_a_partial_sprite_walk_yields_no_layer(sprite_install, mixed_scenario) -> None:
    """The payload rides StopIteration, never a yield, so there is no way for
    a caller to observe (and install) a half-built layer."""
    elevations, proj = _elevations_and_proj(mixed_scenario)
    gen = render.sprite_draws_by_anchor_sliced(mixed_scenario, proj, elevations)
    assert next(gen) is None
    assert next(gen) is None


@pytest.mark.corpus
def test_sliced_walks_equal_the_whole_ones_on_real_files(scenario_path) -> None:
    """The default-tier fixture above is three hand-written units; a real file
    is ~14k with a unit mix nobody designed for this test."""
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    scenario = load_map_and_units(scenario_path)
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    _img, elevations, proj = render.render_terrain_iso_with_proj(scenario, with_sprites=True)

    whole = render.sprite_draws_by_anchor(scenario, proj, elevations)
    sliced, yields = _step_all(render.sprite_draws_by_anchor_sliced(scenario, proj, elevations))
    assert yields == sum(len(units) for units in scenario.unit_manager.units)
    assert _layer_key(sliced) == _layer_key(whole)

    whole_icons = render._flat_icon_layer(scenario, tile_px)
    sliced_icons, _ = _step_all(render._flat_icon_layer_sliced(scenario, tile_px))
    assert _icons_key(sliced_icons) == _icons_key(whole_icons)


# --- Step 2: the driver installs, revalidates and cancels ------------------


def _iso_cache(scn):
    from descape.render_cache import IsoChunkCache

    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    elevations, proj = render.elevations_and_proj(scn)
    return IsoChunkCache(scn, elevations, proj, tile_px, sprites=True)


def _flat_cache(scn):
    from descape.render_cache import FlatChunkCache

    return FlatChunkCache(scn, render.tile_pixels_for_map(MAP_W, MAP_H), sprites=True)


def _warm(cache, mips):
    """start() + run_to_completion(): the whole warm, synchronously, with no
    QApplication and no timer -- LevelWarmer._schedule() no-ops without one,
    which is what keeps these checks in the suite's usual shape."""
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, mips)
    return warmer


def test_a_warm_installs_a_stepped_level(sprite_install, mixed_scenario) -> None:
    cache = _iso_cache(mixed_scenario)
    assert cache._levels[1].gen != cache._source_gen, "level 1 was already built -- vacuous"

    _warm(cache, [1]).run_to_completion()

    assert cache._levels[1].gen == cache._source_gen
    assert cache._levels[1].sprites is not None
    assert cache._levels[1].sprites.by_anchor, "installed an empty layer -- vacuous"
    assert cache._levels[1].building_bboxes is not None


def test_a_warm_installs_a_flat_icon_layer(sprite_install, mixed_scenario) -> None:
    cache = _flat_cache(mixed_scenario)
    assert 1 not in cache._level_icon_layers, "level 1 was already built -- vacuous"

    _warm(cache, [1]).run_to_completion()

    assert cache._level_icon_layers[1], "installed an empty icon layer -- vacuous"


def test_nothing_is_queued_with_sprites_off(mixed_scenario) -> None:
    """With sprites off a level costs only _building_bboxes_iso (3.9ms
    measured, against the sprite walk's 676ms), so there is nothing worth
    ticking for -- and level_warm_job() saying so is what leaves is_active
    False rather than scheduling a timer to do nothing."""
    from descape.render_cache import FlatChunkCache, IsoChunkCache

    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    elevations, proj = render.elevations_and_proj(mixed_scenario)
    iso = IsoChunkCache(mixed_scenario, elevations, proj, tile_px, sprites=False)
    flat = FlatChunkCache(mixed_scenario, tile_px, sprites=False)

    assert iso.level_warm_job(1) is None
    assert flat.level_warm_job(1) is None
    assert not _warm(iso, [1]).is_active
    assert not _warm(flat, [1]).is_active


def test_sloped_has_nothing_to_warm(mixed_scenario) -> None:
    """One mip level means no not-yet-visited level exists -- the base
    class's None is Sloped's real answer, not an unimplemented stub."""
    from descape import iso_geometry
    from descape.render_cache import SlopedChunkCache

    tile_px = render.tile_pixels_for_map(MAP_W, MAP_H)
    elevations, proj = render.elevations_and_proj(mixed_scenario)
    corner_rise = iso_geometry.corner_rise_px(elevations, proj)
    cache = SlopedChunkCache(mixed_scenario, elevations, corner_rise, proj, tile_px, sprites=True)
    assert cache.mip_levels() == [0]
    assert cache.level_warm_job(0) is None


def test_a_unit_edit_mid_warm_drops_the_stepped_result(sprite_install, mixed_scenario) -> None:
    """The revalidation half of the design: even with every cancel call site
    removed, a layer built across a mutation is never installed."""
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    warmer.tick()

    cache.invalidate_units()
    warmer.run_to_completion()

    assert cache._levels[1].sprites is None
    assert cache._levels[1].gen != cache._source_gen


def test_a_unit_edit_mid_warm_drops_the_flat_result(sprite_install, mixed_scenario) -> None:
    """Flat's own predicate, and the one that needed _unit_gen: after
    invalidate_units() the target mip is absent from _level_icon_layers
    exactly as it was before the warm started, so "still absent" cannot tell
    the two apart on its own."""
    cache = _flat_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    warmer.tick()

    cache.invalidate_units()
    warmer.run_to_completion()

    assert 1 not in cache._level_icon_layers


def test_cancel_drops_the_queue_without_installing(sprite_install, mixed_scenario) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [-1, 1])
    warmer.tick()

    warmer.cancel()
    warmer.run_to_completion()

    assert not warmer.is_active
    assert cache._levels[1].sprites is None
    assert cache._levels[-1].sprites is None


def test_a_paint_that_wins_the_race_keeps_its_own_layer(sprite_install, mixed_scenario) -> None:
    """The non-obvious third predicate: the warm's result is EQUIVALENT to
    what the paint built, so installing it would be harmless-looking and
    still wrong -- it replaces a layer already wired into composited chunks
    with a fresh copy of the same thing."""
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    warmer.tick()

    painted = cache._level(1).sprites
    assert painted is not None, "the paint built no layer -- vacuous"
    warmer.run_to_completion()

    assert cache._levels[1].sprites is painted


def test_neighbour_mips_excludes_the_opening_level(sprite_install, mixed_scenario) -> None:
    cache = _iso_cache(mixed_scenario)
    levels = cache.mip_levels()
    assert len(levels) > 1, "single-level ladder -- vacuous"
    opening = cache.mip_for_scale(1.0)

    mips = level_warm.neighbour_mips(cache, 1.0)

    assert opening not in mips
    assert set(mips) <= {opening - 1, opening + 1}
    assert all(levels[0] <= mip <= levels[-1] for mip in mips)


def test_on_job_done_fires_once_per_queued_mip_in_order(sprite_install, mixed_scenario) -> None:
    """2026-09-07 plan's load-time margin warm, Step 2: on_job_done is the
    hook the load-time chunk warm chains off of. Jobs run strictly one at a
    time (_start_next_job pops the queue), so the callback must fire in the
    same order the mips were queued, once each."""
    cache = _iso_cache(mixed_scenario)
    warmer = level_warm.LevelWarmer()
    done = []
    warmer.start(cache, [-1, 1], on_job_done=done.append)
    warmer.run_to_completion()
    assert done == [-1, 1]


def test_on_job_done_does_not_fire_for_a_mip_with_nothing_to_warm(mixed_scenario) -> None:
    """With sprites off, level_warm_job() returns None for every mip -- no
    job is ever queued for it, so on_job_done must never fire either. A
    caller that also needs those mips covered checks is_level_resident()
    itself; see _queue_load_warm's own docstring."""
    cache = _iso_cache(mixed_scenario)
    cache.set_sprites_enabled(False)
    warmer = level_warm.LevelWarmer()
    done = []
    warmer.start(cache, [1], on_job_done=done.append)
    assert not warmer.is_active
    warmer.run_to_completion()
    assert done == []


def test_on_job_done_does_not_fire_for_a_cancelled_job(sprite_install, mixed_scenario) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = level_warm.LevelWarmer()
    done = []
    warmer.start(cache, [1], on_job_done=done.append)
    warmer.tick()
    warmer.cancel()
    warmer.run_to_completion()
    assert done == []


# --- Step 3/5: the viewer hook and the setting that gates it ---------------


def _shown_window():
    """A real window, SHOWN BEFORE the load. Order matters: the warm's target
    levels come from MapView._fit_baseline_scale(), which returns None for a
    zero-size viewport -- so loading into a never-shown window (every other
    gui test's shape) correctly warms nothing, and would make the checks
    below pass vacuously. Caller must mark_saved() + close()."""
    from PyQt5.QtWidgets import QApplication

    import conftest
    conftest.ensure_qapp()
    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(800, 600)
    window.show()
    QApplication.processEvents()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


@pytest.mark.gui
def test_a_load_queues_a_warm_and_it_installs(monkeypatch) -> None:
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = _shown_window()
    try:
        assert window._level_warmer.is_active, "the load queued nothing -- vacuous"
        cache = window._cache
        fit = window.map_view._fit_baseline_scale() * window.map_view.devicePixelRatioF()
        targets = level_warm.neighbour_mips(cache, fit)
        assert targets, "the fit level has no neighbours on this ladder -- vacuous"
        assert any(cache._levels[mip].gen != cache._source_gen for mip in targets)

        window._level_warmer.run_to_completion()

        assert not window._level_warmer.is_active
        assert all(cache._levels[mip].gen == cache._source_gen for mip in targets)
    finally:
        _close(window)


@pytest.mark.gui
def test_no_warm_is_scheduled_with_the_setting_off(monkeypatch) -> None:
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", False)
    window = _shown_window()
    try:
        assert not window._level_warmer.is_active
    finally:
        _close(window)


@pytest.mark.gui
def test_an_edit_cancels_an_in_flight_warm(monkeypatch) -> None:
    """The cancel side of Step 4, through a real viewer path rather than by
    calling the driver directly: _after_unit_mutation is one of the mutating
    call sites, and the warm must be gone before it returns."""
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = _shown_window()
    try:
        assert window._level_warmer.is_active, "the load queued nothing -- vacuous"
        window._after_unit_mutation()
        assert not window._level_warmer.is_active
    finally:
        _close(window)
