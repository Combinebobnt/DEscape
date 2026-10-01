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
from pathlib import Path

import pytest
from test_native_composite import native_kernel  # noqa: F401 -- a fixture

from descape import asset_source, composite_backend, level_warm, render, unit_sprites
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


@pytest.fixture
def one_step_ticks(monkeypatch):
    """A tick makes exactly one step at BUDGET_MS 0. At 12ms a fast sprite
    decode can finish this fixture's whole level in the first tick, leaving
    nothing mid-warm to test."""
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)


def _first_step(warmer) -> None:
    assert warmer.tick(), "the warm finished in one step -- vacuous"


def test_a_unit_edit_mid_warm_drops_the_stepped_result(sprite_install, mixed_scenario, one_step_ticks) -> None:
    """The revalidation half of the design: even with every cancel call site
    removed, a layer built across a mutation is never installed."""
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    _first_step(warmer)

    cache.invalidate_units()
    warmer.run_to_completion()

    assert cache._levels[1].sprites is None
    assert cache._levels[1].gen != cache._source_gen


def test_a_unit_edit_mid_warm_drops_the_flat_result(sprite_install, mixed_scenario, one_step_ticks) -> None:
    """Flat's own predicate, and the one that needed _unit_gen: after
    invalidate_units() the target mip is absent from _level_icon_layers
    exactly as it was before the warm started, so "still absent" cannot tell
    the two apart on its own."""
    cache = _flat_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    _first_step(warmer)

    cache.invalidate_units()
    warmer.run_to_completion()

    assert 1 not in cache._level_icon_layers


def test_cancel_drops_the_queue_without_installing(sprite_install, mixed_scenario, one_step_ticks) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [-1, 1])
    _first_step(warmer)

    warmer.cancel()
    warmer.run_to_completion()

    assert not warmer.is_active
    assert cache._levels[1].sprites is None
    assert cache._levels[-1].sprites is None


def test_a_paint_that_wins_the_race_keeps_its_own_layer(sprite_install, mixed_scenario, one_step_ticks) -> None:
    """The non-obvious third predicate: the warm's result is EQUIVALENT to
    what the paint built, so installing it would be harmless-looking and
    still wrong -- it replaces a layer already wired into composited chunks
    with a fresh copy of the same thing."""
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    _first_step(warmer)

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


def test_neighbour_mips_of_clamps_at_both_ladder_ends(sprite_install, mixed_scenario) -> None:
    cache = _iso_cache(mixed_scenario)
    levels = cache.mip_levels()
    assert len(levels) >= 3, "no interior level on this ladder -- vacuous"

    assert level_warm.neighbour_mips_of(cache, levels[0]) == [levels[0] + 1]
    assert level_warm.neighbour_mips_of(cache, levels[-1]) == [levels[-1] - 1]
    assert level_warm.neighbour_mips_of(cache, levels[1]) == [levels[0], levels[2]]
    for scale in (0.1, 0.5, 1.0, 2.0, 8.0):
        assert level_warm.neighbour_mips(cache, scale) == level_warm.neighbour_mips_of(
            cache, cache.mip_for_scale(scale)
        )


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


def test_on_job_done_does_not_fire_for_a_cancelled_job(sprite_install, mixed_scenario, one_step_ticks) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = level_warm.LevelWarmer()
    done = []
    warmer.start(cache, [1], on_job_done=done.append)
    _first_step(warmer)
    warmer.cancel()
    warmer.run_to_completion()
    assert done == []


def _hold_mouse(monkeypatch) -> dict:
    """QApplication.mouseButtons() reads LeftButton while held["value"]."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )
    return held


def test_the_level_warmer_pauses_while_a_mouse_button_is_held(sprite_install, mixed_scenario, monkeypatch) -> None:
    """MarginWarmer's item-23 backoff, shared (2026-09-29 warm-tick plan): a
    held tick advances nothing and backs the timer off; release drains."""
    held = _hold_mouse(monkeypatch)
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    intervals: list = []
    monkeypatch.setattr(warmer, "_set_interval", intervals.append)

    for _ in range(3):
        assert warmer.tick() is True
    assert cache._levels[1].gen != cache._source_gen, "warmed while held"
    assert warmer.is_active and warmer._job is None, "a job started while held"
    assert intervals and set(intervals) == {level_warm.LevelWarmer.HELD_INTERVAL_MS}

    held["value"] = False
    warmer.run_to_completion()
    assert intervals[-1] == 0
    assert cache._levels[1].gen == cache._source_gen


@pytest.mark.gui
def test_the_level_warmer_backs_off_the_timer_interval_while_held(sprite_install, mixed_scenario, monkeypatch) -> None:
    """The real QTimer's interval, as margin_warm's gui counterpart checks it."""
    import conftest

    conftest.ensure_qapp()
    held = _hold_mouse(monkeypatch)
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    try:
        warmer.tick()
        assert warmer._timer.interval() == level_warm.LevelWarmer.HELD_INTERVAL_MS
        held["value"] = False
        warmer.tick()
        assert warmer._timer.interval() == 0
    finally:
        warmer.cancel()


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


def _in_flight_jobs(warmer) -> list:
    """The LevelWarmJob objects a warmer is currently holding, queued or
    running. Returns the objects themselves, not their ids: the caller
    compares them by identity across a cancel, and an id of a dropped job can
    legitimately be reused by the job that replaces it."""
    jobs = [job for _, job, _notify in warmer._queue]
    if warmer._job is not None:
        jobs.append(warmer._job)
    return jobs


@pytest.mark.gui
def test_an_edit_drops_the_in_flight_warm_and_re_arms(monkeypatch) -> None:
    """The cancel side of Step 4, through a real viewer path rather than by
    calling the driver directly: _after_unit_mutation is one of the mutating
    call sites, and every job in flight when it starts must be gone before it
    returns.

    The assertion is job IDENTITY, not `not is_active`, because Batch B step
    B1 re-arms a fresh warm at that method's tail. Identity is the invariant
    the cancel actually carries: a job built before the edit walks the
    pre-edit unit list, and one built after the invalidation cannot. Both
    halves are pinned here, so neither the cancel nor the re-arm can go
    missing without a failure.
    """
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = _shown_window()
    try:
        assert window._level_warmer.is_active, "the load queued nothing -- vacuous"
        stale = _in_flight_jobs(window._level_warmer)
        assert stale, "vacuous: nothing was in flight"

        window._after_unit_mutation()

        live = _in_flight_jobs(window._level_warmer)
        assert not [job for job in live if any(job is old for old in stale)], (
            "a job that started before the edit is still in flight after it"
        )
        assert window._level_warmer.is_active, "B1's tail re-arm queued nothing"
    finally:
        _close(window)


@pytest.mark.gui
def test_a_terrain_edit_re_arms_the_level_warm(monkeypatch) -> None:
    """B1's other half. _apply_dirty() cancels the warms at its top, and
    before B1 nothing started them again, so the first stroke, paste or undo
    of a session left every neighbour mip cold until the next file open and
    the next zoom paid the whole level build inside paint()."""
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = _shown_window()
    try:
        assert window._level_warmer.is_active, "vacuous: the load queued nothing"
        stale = _in_flight_jobs(window._level_warmer)
        assert stale, "vacuous: nothing was in flight"

        window._apply_dirty([0])

        live = _in_flight_jobs(window._level_warmer)
        assert not [job for job in live if any(job is old for old in stale)]
        assert window._level_warmer.is_active, "B1's tail re-arm queued nothing"
    finally:
        _close(window)


# --- unit-pack pre-derive chaining (maintainer plan 2026-09-27) -------------

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _pack_pending(cache, mip: int) -> int | None:
    """None while the level has no current pack, else its pending tile count."""
    pack = cache._unit_pack_of(mip, create=False)
    return None if pack is None else int(pack.pending().size)


def test_on_job_done_fires_after_each_neighbours_pack_derive(native_kernel, sprite_install, mixed_scenario) -> None:  # noqa: F811
    """Each neighbour's pack derive runs right after its level warm and
    carries the notify; a pack_only mip is derived last and never notifies."""
    with composite_backend.use_backend("native"):
        cache = _iso_cache(mixed_scenario)
        cache._level(0)
        warmer = level_warm.LevelWarmer()
        seen = []
        notifying = warmer.start(
            cache, [-1, 1], pack_only=[0], on_job_done=lambda m: seen.append((m, _pack_pending(cache, m))),
        )
        assert notifying == {-1, 1}
        assert [(m, notify) for m, _job, notify in warmer._queue] == [
            (-1, False), (-1, True), (1, False), (1, True), (0, False),
        ]
        warmer.run_to_completion()
    assert seen == [(-1, 0), (1, 0)]
    assert _pack_pending(cache, 0) == 0


def test_on_job_done_does_not_fire_for_a_cancelled_pack_derive(
    native_kernel, sprite_install, mixed_scenario, monkeypatch,  # noqa: F811
) -> None:
    from descape import render_cache

    monkeypatch.setattr(render_cache, "PACK_WARM_TILES", 1)
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    with composite_backend.use_backend("native"):
        cache = _iso_cache(mixed_scenario)
        warmer = level_warm.LevelWarmer()
        done = []
        warmer.start(cache, [1], on_job_done=done.append)

        def mid_derive() -> bool:
            pack = cache._unit_pack_of(1, create=False)
            return pack is not None and pack.ready[pack.has].any() and pack.pending().size > 0

        while not mid_derive():
            assert warmer.tick(), "the derive finished without a mid-walk point -- vacuous"
        warmer.cancel()
        warmer.run_to_completion()
    assert done == []


@pytest.mark.gui
def test_a_neighbours_load_warm_starts_only_after_its_pack_derive(native_kernel, monkeypatch) -> None:  # noqa: F811
    """Resident neighbours whose packs were dropped have only a derive left:
    their load warm must wait for it, and be queued once."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    with composite_backend.use_backend("native"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            cache = window._cache
            window._level_warmer.run_to_completion()
            fit = window.map_view._fit_baseline_scale() * window.map_view.devicePixelRatioF()
            mips = level_warm.neighbour_mips(cache, fit)
            assert mips, "the fit level has no neighbours on this ladder -- vacuous"
            for m in mips:
                cache._level(m)
                cache._levels[m].unit_pack = None
            calls = []
            real = window._queue_load_warm

            def spy(mip):
                calls.append((mip, _pack_pending(cache, mip)))
                real(mip)

            monkeypatch.setattr(window, "_queue_load_warm", spy)
            window._start_level_warm()
            window._level_warmer.run_to_completion()
            assert sorted(calls) == sorted((m, 0) for m in mips)
        finally:
            conftest.close_window(window)


@pytest.mark.gui
def test_the_viewport_mip_is_pre_derived_and_an_idle_re_arm_queues_nothing(native_kernel, monkeypatch) -> None:  # noqa: F811
    """A resident viewport level whose pack was dropped (a splice onto a stale
    pack does that) gets a pack-only derive; once nothing is pending anywhere,
    the per-stroke-step re-arm leaves the LevelWarmer idle."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    with composite_backend.use_backend("native"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            cache = window._cache
            QApplication.processEvents()
            window._level_warmer.run_to_completion()
            target = window.map_view.viewport_chunk_target()
            assert target is not None
            mip = target[0]
            assert cache.is_level_resident(mip), "the first paint never built the viewport's level -- vacuous"
            cache._levels[mip].unit_pack = None

            window._start_level_warm()
            assert (mip, False) in [(m, notify) for m, _job, notify in window._level_warmer._queue]
            window._level_warmer.run_to_completion()
            assert _pack_pending(cache, mip) == 0

            window._apply_dirty([0])
            assert not window._level_warmer.is_active, "a re-arm with nothing pending started a timer"
        finally:
            conftest.close_window(window)


# --- 2026-09-29 warm-tick plan, C: sliced assembly and the fresh-tick rule ----


def _assembly_key(built):
    from test_bystander_grid_patch import grid_state

    bboxes, grid = built
    return dict(bboxes), grid_state(grid)


def _assembled(cache, mip: int, monkeypatch, slice_units: int):
    """(bboxes, grid) and the yield count of cache._assemble_level() at
    render.ASSEMBLY_SLICE = slice_units, over the level's own sprite layer."""
    lvl = cache._levels[mip]
    sprites, _memo = render._drain(cache._sprite_walk(lvl))
    monkeypatch.setattr(render, "ASSEMBLY_SLICE", slice_units)
    built, yields = _step_all(cache._assemble_level(lvl, sprites))
    return built, yields, sprites


def test_the_sliced_assembly_equals_the_whole_one(sprite_install, mixed_scenario, monkeypatch) -> None:
    """Slice 1 puts a boundary between every unit and key; the whole side is
    the pre-split composition of the three public functions."""
    cache = _iso_cache(mixed_scenario)
    sliced, yields, sprites = _assembled(cache, 1, monkeypatch, 1)
    monkeypatch.setattr(render, "ASSEMBLY_SLICE", 10**9)
    mm, lvl = mixed_scenario.map_manager, cache._levels[1]
    whole_bboxes = render.merge_sprite_bboxes(
        render._building_bboxes_iso(mixed_scenario, mm.map_width, mm.map_height, lvl.proj, cache.elevations), sprites
    )
    whole = (whole_bboxes, render.build_bystander_grid(whole_bboxes, cache.chunk_px))

    assert sprites.bboxes and whole_bboxes, "vacuous: nothing to assemble"
    assert _assembly_key(sliced) == _assembly_key(whole)
    units = sum(len(u) for u in mixed_scenario.unit_manager.units)
    assert yields >= units + len(sprites.bboxes), f"assembly barely sliced: {yields} yields"


@pytest.mark.corpus
def test_the_sliced_assembly_equals_the_whole_one_on_real_files(scenario_path, monkeypatch) -> None:
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    from descape.render_cache import IsoChunkCache

    scenario = load_map_and_units(scenario_path)
    mm = scenario.map_manager
    elevations, proj = render.elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, render.tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    mip = cache.mip_levels()[0]
    sliced, yields, _sprites = _assembled(cache, mip, monkeypatch, 7)
    whole, _, _ = _assembled(cache, mip, monkeypatch, 10**9)
    assert _assembly_key(sliced) == _assembly_key(whole)
    assert yields > sum(len(u) for u in scenario.unit_manager.units) // 7


def test_a_warm_assembles_a_level_a_slice_per_step(sprite_install, mixed_scenario, monkeypatch) -> None:
    """Granularity in the driver, not just in the generator: at one step per
    tick no tick does more than one unit's bbox or one key's grid cells."""
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    monkeypatch.setattr(render, "ASSEMBLY_SLICE", 1)
    per_tick = {"_building_bbox_for": 0, "_bbox_cells": 0}
    peaks = dict(per_tick)
    totals = dict(per_tick)
    for name in per_tick:
        real = getattr(render, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            per_tick[_name] += 1
            totals[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(render, name, counted)
    cache = _iso_cache(mixed_scenario)
    warmer = _warm(cache, [1])
    while True:
        more = warmer.tick()
        for name, n in per_tick.items():
            peaks[name] = max(peaks[name], n)
            per_tick[name] = 0
        if not more:
            break
    assert cache._levels[1].gen == cache._source_gen, "the warm installed nothing"
    assert totals["_building_bbox_for"] == sum(len(u) for u in mixed_scenario.unit_manager.units)
    assert totals["_bbox_cells"] >= len(cache._levels[1].building_bboxes) > 1
    assert peaks == {"_building_bbox_for": 1, "_bbox_cells": 1}, f"a tick assembled unsliced: {peaks}"


def _mid_assembly(cache, monkeypatch):
    """A one-step-per-tick warm of level 1, stopped once its assembly has begun."""
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    monkeypatch.setattr(render, "ASSEMBLY_SLICE", 1)
    seen = []
    real = render._building_bbox_for
    monkeypatch.setattr(render, "_building_bbox_for", lambda *a, **k: seen.append(1) or real(*a, **k))
    warmer = _warm(cache, [1])
    while not seen:
        assert warmer.tick(), "the warm finished before its assembly began -- vacuous"
    assert cache._levels[1].gen != cache._source_gen, "the assembly finished in the step it began -- vacuous"
    return warmer


@pytest.mark.parametrize("action", ["unit-edit", "cancel"])
def test_an_edit_or_cancel_mid_assembly_installs_nothing(action, sprite_install, mixed_scenario, monkeypatch) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = _mid_assembly(cache, monkeypatch)
    if action == "unit-edit":
        cache.invalidate_units()
    else:
        warmer.cancel()
    warmer.run_to_completion()
    assert cache._levels[1].sprites is None
    assert cache._levels[1].gen != cache._source_gen


def test_a_paint_that_wins_mid_assembly_keeps_its_own_layer(sprite_install, mixed_scenario, monkeypatch) -> None:
    cache = _iso_cache(mixed_scenario)
    warmer = _mid_assembly(cache, monkeypatch)
    painted = cache._level(1)
    layer, grid = painted.sprites, painted.bystander_grid
    assert layer is not None, "the paint built no layer -- vacuous"
    warmer.run_to_completion()
    assert cache._levels[1].sprites is layer and cache._levels[1].bystander_grid is grid


class _StepClock:
    """level_warm._now's stand-in: time moves only when a fake job steps."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class _FakeJobsCache:
    """level_warm_job(mip): a job of `steps` 1ms steps logging (mip, step, tick)."""

    def __init__(self, clock, steps: int) -> None:
        self.clock, self.steps, self.tick = clock, steps, 0
        self.log: list[tuple[int, int, int]] = []

    def level_warm_job(self, mip: int):
        from descape.render_cache import LevelWarmJob

        def gen():
            for step in range(self.steps):
                self.log.append((mip, step, self.tick))
                self.clock.t += 0.001
                yield

        return LevelWarmJob(gen=gen(), install=lambda _payload: True)

    def pack_warm_job(self, mip: int, after_level_warm: bool = False):
        return None


def test_a_job_starts_only_on_a_fresh_tick(monkeypatch) -> None:
    """The first job ends 3ms into a 12ms budget; the second, whose first step
    is unsliced setup in the real jobs, still waits for the next tick. The
    finished job's on_job_done also ends that tick."""
    clock = _StepClock()
    monkeypatch.setattr(level_warm, "_now", clock)
    cache = _FakeJobsCache(clock, steps=3)
    warmer = level_warm.LevelWarmer()
    done = []
    warmer.start(cache, [1, 2], on_job_done=lambda mip: done.append((mip, cache.tick)))
    ticks = 0
    while True:
        cache.tick = ticks
        more = warmer.tick()
        ticks += 1
        if not more:
            break
    assert [(mip, step) for mip, step, _tick in cache.log] == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert {mip: tick for mip, step, tick in cache.log if step == 0} == {1: 0, 2: 1}, cache.log
    assert done == [(1, 0), (2, 1)]
    assert ticks == 2


def test_a_traced_tick_reports_its_steps_by_job_and_its_gc(monkeypatch) -> None:
    """Perf Trace's worst-tick attribution: each tick hands level_warm_tick()
    its ms per step label (a job's first step apart, as `setup`) and the gc
    ms that landed inside it."""
    from descape import perf_trace

    clock = _StepClock()
    monkeypatch.setattr(level_warm, "_now", clock)
    monkeypatch.setattr(perf_trace, "_enabled", True)
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "time", type("T", (), {"perf_counter": staticmethod(clock)}))
    ticks = []
    monkeypatch.setattr(
        perf_trace, "level_warm_tick", lambda ms, installs, split, gc_ms: ticks.append((dict(split), gc_ms))
    )
    cache = _FakeJobsCache(clock, steps=3)
    real_job = cache.level_warm_job

    def job_with_a_collection(mip):
        job = real_job(mip)
        inner = job.gen

        def gen():
            yield next(inner)
            perf_trace._on_gc("start", {"generation": 2})
            clock.t += 0.025
            perf_trace._on_gc("stop", {"generation": 2})
            yield from inner

        if mip == 2:
            job.kind = "flush"
        else:
            job.gen = gen()
        return job

    cache.level_warm_job = job_with_a_collection
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [1, 2], on_job_done=lambda mip: None)
    # Real collections meanwhile cost 0 ms on the fake clock.
    perf_trace.set_gc_hook(True)
    try:
        while warmer.tick():
            pass
    finally:
        perf_trace.set_gc_hook(False)
    # The collection spends the 12 ms budget, so job 1 finishes on the next tick.
    assert ticks[0] == (pytest.approx({"walk 1 setup": 1.0, "walk 1": 26.0}), pytest.approx(25.0))
    assert set(ticks[1][0]) == {"walk 1", "install", "done"}
    assert ticks[2] == (pytest.approx({"flush 2 setup": 1.0, "flush 2": 2.0, "install": 0.0, "done": 0.0}), 0.0)
    assert len(ticks) == 3


# --- the warm set follows the view (zoom plan 2026-09-29, Step 2) ----------


def _spy_level_warm_starts(window, monkeypatch) -> list:
    """Each LevelWarmer.start()'s mips, in call order; forwards to the real one."""
    starts = []
    real = window._level_warmer.start

    def spy(cache, mips, **kwargs):
        starts.append(list(mips))
        return real(cache, mips, **kwargs)

    monkeypatch.setattr(window._level_warmer, "start", spy)
    return starts


def _fake_viewport_mip(window, monkeypatch) -> dict:
    """viewport_chunk_target() reads fake["mip"], with that level's real chunk
    range for the current scene rect, so the margin warm still gets a valid ring."""
    view = window.map_view
    fake = {}
    monkeypatch.setattr(view, "viewport_chunk_target", lambda: (fake["mip"], *view.viewport_chunk_target_at(fake["mip"])))
    return fake


def _stale_residents(cache, visible: int) -> set[int]:
    return {m for m in cache.resident_levels() if m != visible and not cache.is_level_resident(m)}


@pytest.mark.gui
def test_a_viewport_mip_change_re_anchors_the_level_warm(monkeypatch) -> None:
    """Zooming used to leave the warm set on the fit level's neighbours. A
    viewport fire at a new mip re-arms it on that mip's neighbours (plus any
    stale residents); a fire at the same mip does not."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    try:
        cache = window._cache
        assert {-2, -1, 0, 1} <= set(cache.mip_levels()), "ladder too short -- vacuous"
        fake = _fake_viewport_mip(window, monkeypatch)
        starts = _spy_level_warm_starts(window, monkeypatch)
        window._last_viewport_chunk_target = (-1, *window.map_view.viewport_chunk_target_at(-1))

        fake["mip"] = 0
        window._on_viewport_changed()
        assert len(starts) == 1, "a mip change did not re-arm the level warm"
        assert {-1, 1} <= set(starts[0])
        assert set(starts[0]) - {-1, 1} <= _stale_residents(cache, 0)

        window._on_viewport_changed()
        assert len(starts) == 1, "a same-mip viewport change re-armed the level warm"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_style_switch_with_a_poll_pending_leaves_no_warm_on_the_old_cache(native_kernel, monkeypatch) -> None:  # noqa: F811
    """_render_current cancels the warms, then runs processEvents() BEFORE it
    swaps self._cache. A viewport poll firing there must not re-arm the level
    warm on the outgoing cache: its job would finish against the new one and
    ask _queue_load_warm for a mip the new ladder does not have."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    with composite_backend.use_backend("native"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            old = window._cache
            # Zoomed in to mip 0, so a re-arm there has a cold neighbour (1) to queue.
            fake = _fake_viewport_mip(window, monkeypatch)
            fake["mip"] = 0
            assert not old.is_level_resident(1), "mip 1 is already warm -- vacuous"
            fires = []
            real_fire = window._on_viewport_changed

            def spy() -> None:
                fires.append(window._cache is old)
                real_fire()

            view = window.map_view
            view.on_viewport_changed = spy
            real_cancel = window._cancel_warms

            def cancel_with_a_poll_pending() -> None:
                # The load-dependent case made certain: a 0 ms poll that the
                # next processEvents() fires, and a target it has not seen.
                real_cancel()
                view._last_viewport_target = None
                view._viewport_poll_timer.setInterval(0)
                view._viewport_poll_timer.start()

            monkeypatch.setattr(window, "_cancel_warms", cancel_with_a_poll_pending)
            window.terrain_style_combo.setCurrentText("Sloped")
            view._viewport_poll_timer.setInterval(view.VIEWPORT_POLL_MS)
            assert window._cache is not old
            assert True in fires, "no poll fired before the cache swap -- vacuous"

            assert window._level_warmer._cache is None or window._level_warmer._cache is window._cache
            window._level_warmer.run_to_completion()
        finally:
            conftest.close_window(window)


@pytest.mark.gui
def test_a_post_edit_re_arm_anchors_on_the_viewport_mip(monkeypatch) -> None:
    """B1's tail re-arm while zoomed in warms around the level in view, not
    the fit level: that is the level the next zoom leaves from."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    try:
        cache = window._cache
        view = window.map_view
        fit_mip = cache.mip_for_scale(view._fit_baseline_scale() * view.devicePixelRatioF())
        fit_set = level_warm.neighbour_mips_of(cache, fit_mip)
        anchor = next(
            (m for m in reversed(cache.mip_levels()) if level_warm.neighbour_mips_of(cache, m) != fit_set), None,
        )
        assert anchor is not None, "every level has the fit level's neighbours -- vacuous"
        fake = _fake_viewport_mip(window, monkeypatch)
        fake["mip"] = anchor
        starts = _spy_level_warm_starts(window, monkeypatch)

        window._apply_dirty([0])

        assert starts, "B1's tail re-arm never started the level warm"
        want = level_warm.neighbour_mips_of(cache, anchor)
        assert starts[-1][:len(want)] == want
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_mip_change_queues_no_load_warm_for_an_already_warm_level(native_kernel, monkeypatch) -> None:  # noqa: F811
    """The plan's check on re-arming at every mip change. _queue_load_warm's
    residency check lets a resident level through; what keeps it from
    re-queueing is load_warm_chunks dropping chunks already cached."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    with composite_backend.use_backend("native"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            cache = window._cache
            QApplication.processEvents()
            real_target = window.map_view.viewport_chunk_target()
            assert real_target is not None
            fit_mip = real_target[0]
            window._level_warmer.run_to_completion()
            while window._load_warmer.is_active or window._load_warm_queue:
                window._pump_load_warm()
                window._load_warmer.run_to_completion()
            neighbours = level_warm.neighbour_mips_of(cache, fit_mip)
            assert neighbours and all(cache.is_level_resident(m) for m in neighbours), "vacuous"
            assert cache.is_level_resident(fit_mip), "the first paint never built the fit level -- vacuous"

            calls = []
            real_queue = window._queue_load_warm

            def spy(mip: int) -> None:
                calls.append(mip)
                real_queue(mip)

            monkeypatch.setattr(window, "_queue_load_warm", spy)
            fake = _fake_viewport_mip(window, monkeypatch)
            fake["mip"] = neighbours[0]
            window._last_viewport_chunk_target = real_target
            window._on_viewport_changed()

            assert fit_mip in calls, "the resident fit level never reached _queue_load_warm -- vacuous"
            assert window._load_warm_queue == [] and not window._load_warmer.is_active
        finally:
            conftest.close_window(window)
