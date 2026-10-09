"""A Stepped Elevate stroke splices only the visible level; every other current
level defers its re-anchor to one flush on its next read (2026-09-28
stepped-defer plan). See IsoChunkCache._splice_levels, _flush_pending and
_ChunkCacheBase.patch's _defers_patch evict.

Every stroke step goes through patch(..., rebuild_levels=(0,)), the shape the
viewer passes on a Stepped terrain edit, with levels -2/-1/0 resident. The
existing elevation-splice oracles call patch() without rebuild_levels and so
exercise only the every-level path.

Levels are compared with a fresh cache after a read through _level(), the
production reader, which is where the flush runs. Same fixture posture as
tests/test_elevation_unit_splice.py: duck-typed units appended before the
cache is built, and unit_gen bumped by hand wherever a real edit would.

The 2026-09-29 warm-tick plan moves the flush out of every chunk read a warm
makes: a pending level is not resident, a chunk warm refuses it, and
LevelWarmer runs its flush job (paused while a mouse button is held).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_bystander_grid_patch import grid_state
from test_convert_splice_corpus import _level_state, _pack_state
from test_elevation_unit_splice import BASE_ELEVATION, UNIT_TILE, _place
from test_invalidate_units_splice import _call_counts, _occupied, _own_tile
from test_native_composite import native_kernel  # noqa: F401 -- fixture
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- fixture

from descape import composite_backend, level_warm, margin_warm, render_cache
from descape.elevation_tools import set_tiles_elevation
from descape.render import dirty_screen_bbox_iso, elevations_and_proj, tile_pixels_for_map
from descape.render_cache import REACH_FALLBACK, IsoChunkCache, UnitSplice
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter

import conftest
from test_unit_sprites import CONST as SPRITE_CONST

VISIBLE = 0
OFFSCREEN = (-2, -1)
MIPS = (*OFFSCREEN, VISIBLE)
TILE_A, TILE_B, TILE_C = (40, 40), (44, 40), (48, 40)
EMPTY_TILE = (36, 40)  # raised with A, so it is pending with no unit on it
STROKE = ([TILE_A, EMPTY_TILE], [TILE_B], [TILE_C])


def _scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = BASE_ELEVATION
    units = [
        _place(scenario, 1, SPRITE_CONST, x + 0.5, y + 0.5) for x, y in (TILE_A, TILE_B, TILE_C)
    ]
    _place(scenario, 2, SPRITE_CONST, 30.5, 40.5)  # off the stroke, moved onto it later
    return scenario, units


def _new_cache(scenario, unit_filter: UnitFilter = UnitFilter(), sprites: bool = True) -> IsoChunkCache:
    mm = scenario.map_manager
    elevations, proj = elevations_and_proj(scenario)
    return IsoChunkCache(
        scenario, elevations, proj, tile_pixels_for_map(mm.map_width, mm.map_height),
        sprites=sprites, unit_filter=unit_filter,
    )


def _resident_cache(scenario, sprites: bool = True) -> IsoChunkCache:
    """Every chunk of -2 and -1 cached, and level 0 around the stroke."""
    cache = _new_cache(scenario, sprites=sprites)
    for mip in OFFSCREEN:
        cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
    cache.render_rect(0, 0, 4096, 4096, mip=VISIBLE)
    return cache


def _step(cache, scenario, tiles, rebuild_levels=(VISIBLE,)) -> tuple:
    """One Elevate +1 over `tiles`, through the viewer's calls; returns the bbox."""
    mm = scenario.map_manager
    before = [t.elevation for t in mm.terrain]
    set_tiles_elevation(mm, [(x, y, mm.get_tile(x, y).elevation + 1) for x, y in tiles])
    dirty = [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]
    changed: set = set()
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, cache.elevations, cache.proj, with_units=True, with_sprites=True,
        elevation_changed=changed,
    )
    cache.patch(bbox, elevation_changed=changed, rebuild_levels=rebuild_levels)
    return bbox


def _stroke(cache, scenario) -> list:
    return [_step(cache, scenario, tiles) for tiles in STROKE]


def _bbox_chunks(cache, mip: int, bboxes) -> set:
    keys = set()
    for bbox in bboxes:
        lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, bbox)
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, lx0, ly0, lx1, ly1)
        keys.update((mip, cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1))
    return keys


def _spliced_levels(cache, monkeypatch) -> list[int]:
    """The mip of every _splice_level() call, in order."""
    by_id = {id(lvl): mip for mip, lvl in cache._levels.items()}
    calls = []
    real = IsoChunkCache._splice_level

    def recorded(self, lvl, *args, **kwargs):
        calls.append(by_id[id(lvl)])
        return real(self, lvl, *args, **kwargs)

    monkeypatch.setattr(IsoChunkCache, "_splice_level", recorded)
    return calls


def _assert_matches_fresh(cache, scenario, unit_filter: UnitFilter = UnitFilter(), rebuilt=()) -> None:
    """rebuilt: levels a wholesale rebuild reinstalled, which drops the pack by design."""
    keys: dict = {}
    fresh = _new_cache(scenario, unit_filter)
    for mip in MIPS:
        assert _level_state(cache, mip, False, keys) == _level_state(fresh, mip, False, keys), f"mip {mip}"
        assert not cache._defers_patch(mip), f"mip {mip} still pending after a read"
        if composite_backend.native is None:
            continue
        pack = cache._unit_pack_of(mip, create=mip in rebuilt)
        assert pack is not None, f"mip {mip} lost its unit pack instead of refreshing it"
        assert _pack_state(pack, keys) == _pack_state(fresh._unit_pack_of(mip, create=True), keys), f"mip {mip}"


def test_offscreen_levels_defer_their_splice_and_evict_the_stroke(sprite_install, monkeypatch):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    gen = cache._source_gen
    spliced = _spliced_levels(cache, monkeypatch)

    bboxes = _stroke(cache, scenario)

    assert cache._source_gen == gen, "a stroke step fell back to the wholesale path"
    assert spliced and set(spliced) == {VISIBLE}, f"off-screen levels spliced mid-stroke: {spliced}"
    for mip in OFFSCREEN:
        assert cache._levels[mip].pending_elev == {TILE_A, EMPTY_TILE, TILE_B, TILE_C}
        covered = _bbox_chunks(cache, mip, bboxes)
        assert covered and not covered & set(cache._cache), f"mip {mip} kept a chunk under the stroke"
        if composite_backend.native is not None:
            assert cache.pack_warm_job(mip) is None, f"mip {mip} would read its pack before a flush"
    _assert_matches_fresh(cache, scenario)
    assert set(spliced) == set(MIPS), "the reads did not flush the off-screen levels"


def test_a_pending_chunk_composites_like_a_fresh_cache(sprite_install):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    bboxes = _stroke(cache, scenario)
    fresh = _new_cache(scenario)
    for mip in OFFSCREEN:
        for key in sorted(_bbox_chunks(cache, mip, bboxes)):
            assert np.array_equal(cache.get_chunk(*key), fresh.get_chunk(*key)), f"chunk {key}"


def test_a_unit_move_onto_and_off_pending_tiles_matches_a_fresh_cache(sprite_install):  # noqa: F811
    """invalidate_units() runs after the unit moved, with pre-edit buckets, so
    it must not flush: a flush there re-adds the mover at its new tile and the
    move splice then adds it again."""
    scenario, units = _scenario()
    cache = _resident_cache(scenario)
    _stroke(cache, scenario)
    mover = scenario.unit_manager.units[2][-1]
    batches = [
        (2, len(scenario.unit_manager.units[2]) - 1, mover, EMPTY_TILE),  # onto a pending tile
        (1, 0, units[0], (52, 40)),  # off one
    ]
    for player_id, index, unit, (tx, ty) in batches:
        old_own, old_tiles = _own_tile(unit), _occupied(scenario, unit)
        unit.x, unit.y = tx + 0.5, ty + 0.5
        scenario.unit_gen += 1
        splice = UnitSplice(player_id, index, unit, old_own, _own_tile(unit), old_tiles, _occupied(scenario, unit))
        assert cache.can_splice([splice]), "the move fell back, so this proves nothing"
        cache.invalidate_units([splice])
    _assert_matches_fresh(cache, scenario)


def test_a_union_past_the_cap_rebuilds_the_level_wholesale(sprite_install, monkeypatch):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 2)
    gen = cache._source_gen
    _stroke(cache, scenario)
    assert cache._source_gen == gen, "a single step went over the cap, so the union case is untested"
    counts = _call_counts(monkeypatch)
    cache._level(-1)
    assert counts["building_bboxes"] == 1, "the over-cap flush did not rebuild the level"
    _assert_matches_fresh(cache, scenario, rebuilt=OFFSCREEN)


def test_a_gen_bump_while_pending_clears_it(sprite_install):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    _stroke(cache, scenario)
    hidden = UnitFilter(players=frozenset({0, 1}))
    cache.set_unit_filter(hidden)
    for mip in OFFSCREEN:
        cache._level(mip)
        assert cache._levels[mip].pending_elev == set(), f"mip {mip}: the rebuild kept a moot pending set"
    _assert_matches_fresh(cache, scenario, hidden, rebuilt=MIPS)


def test_sprite_extent_before_falls_back_while_pending(sprite_install):  # noqa: F811
    """A pending level's layer is behind, so the edit is sized with the reach
    fallback; once read it answers like a fresh cache."""
    scenario, units = _scenario()
    cache = _resident_cache(scenario)
    _stroke(cache, scenario)
    unit = units[0]
    occ = _occupied(scenario, unit)
    changed = [UnitSplice(1, 0, unit, _own_tile(unit), _own_tile(unit), occ, occ)]
    assert cache.sprite_extent_before(changed, -1) is REACH_FALLBACK
    cache._level(-1)
    fresh = _new_cache(scenario)
    fresh.render_rect(0, 0, 256, 256, mip=-1)
    got = cache.sprite_extent_before(changed, -1)
    assert got is not REACH_FALLBACK and got is not None
    assert got == fresh.sprite_extent_before(changed, -1)


@pytest.mark.parametrize("rebuild_levels", [None, (-2, -1, 0)])
def test_levels_allowed_to_splice_never_defer(rebuild_levels, sprite_install):  # noqa: F811
    """None (a caller with no viewport target; undo/redo pass (visible,) like a
    stroke step) keeps the every-level splice."""
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    for tiles in STROKE:
        _step(cache, scenario, tiles, rebuild_levels)
    assert not any(cache._levels[mip].pending_elev for mip in MIPS)
    _assert_matches_fresh(cache, scenario)


# --- 2026-09-29 warm-tick plan: pending levels go through LevelWarmer ------


def _level_work(monkeypatch, names=("_flush_pending", "_assemble_level", "_commit_level")) -> list:
    """Every call to `names` (level flush, assembly, install) from here on."""
    calls = []
    for name in names:
        real = getattr(IsoChunkCache, name)

        def recorded(self, *args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(IsoChunkCache, name, recorded)
    return calls


def _pack_pending(cache, mip: int) -> int | None:
    """None while the level has no current pack, else its pending tile count."""
    pack = cache._unit_pack_of(mip, create=False)
    return None if pack is None else int(pack.pending().size)


def _stroke_chunks(cache, mip: int, bboxes) -> list:
    return sorted(key[1:] for key in _bbox_chunks(cache, mip, bboxes))


@pytest.mark.parametrize("cap", [None, 2], ids=["under-cap", "over-cap"])
def test_a_pending_level_is_not_resident_and_a_chunk_warm_refuses_it(cap, sprite_install, monkeypatch):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    if cap is not None:
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", cap)
    bboxes = _stroke(cache, scenario)
    calls = _level_work(monkeypatch)
    for mip in OFFSCREEN:
        assert cache._levels[mip].gen == cache._source_gen and cache._levels[mip].pending_elev, "vacuous"
        assert not cache.is_level_resident(mip)
        assert cache.level_warm_job(mip) is not None, f"mip {mip}: no warm would flush it"
        warmer = margin_warm.MarginWarmer()
        warmer.start(cache, mip, _stroke_chunks(cache, mip, bboxes))
        assert not warmer.is_active, f"mip {mip}: a chunk warm queued a pending level"
        warmer.run_to_completion()
    assert calls == [], "a chunk warm flushed or rebuilt a level"


@pytest.mark.parametrize("cap", [None, 2], ids=["under-cap", "over-cap"])
def test_a_level_warm_flushes_a_pending_level_then_its_load_warm_runs(
    cap, native_kernel, sprite_install, monkeypatch,  # noqa: F811
):
    """The flush job's pack derive is chained (pack_warm_job's after_level_warm),
    so on_job_done sees the level resident with nothing left to derive, and the
    load warm it starts does no level work."""
    with composite_backend.use_backend("native"):
        scenario, _units = _scenario()
        cache = _resident_cache(scenario)
        if cap is not None:
            monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", cap)
        gen = cache._source_gen
        bboxes = _stroke(cache, scenario)
        assert cache._source_gen == gen, "a stroke step fell back to the wholesale path"
        seen, load_warms = [], []

        def on_job_done(mip: int) -> None:
            seen.append((mip, cache.is_level_resident(mip), _pack_pending(cache, mip)))
            load = margin_warm.MarginWarmer()
            load.start(cache, mip, _stroke_chunks(cache, mip, bboxes))
            load_warms.append(load)

        counts = _call_counts(monkeypatch)
        warmer = level_warm.LevelWarmer()
        assert warmer.start(cache, list(OFFSCREEN), on_job_done=on_job_done) == set(OFFSCREEN)
        warmer.run_to_completion()

        assert seen == [(mip, True, 0) for mip in OFFSCREEN]
        assert counts["building_bboxes"] == (len(OFFSCREEN) if cap else 0), "wrong flush path taken"
        calls = _level_work(monkeypatch)
        for load in load_warms:
            assert load.is_active, "the load warm refused a flushed level"
            load.run_to_completion()
        assert calls == [], "the load warm did level work"
        _assert_matches_fresh(cache, scenario)


def test_an_over_cap_flush_job_rebuilds_the_bboxes_with_sprites_off(sprite_install, monkeypatch):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario, sprites=False)
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 2)
    _stroke(cache, scenario)
    assert all(cache._levels[mip].pending_elev for mip in OFFSCREEN), "vacuous"
    counts = _call_counts(monkeypatch)
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, list(OFFSCREEN))
    warmer.run_to_completion()
    assert counts["building_bboxes"] == len(OFFSCREEN)
    fresh = _new_cache(scenario, sprites=False)
    for mip in OFFSCREEN:
        assert cache.is_level_resident(mip)
        want = fresh._level(mip)
        got = cache._levels[mip]
        assert got.building_bboxes == want.building_bboxes, f"mip {mip}"
        assert grid_state(got.bystander_grid) == grid_state(want.bystander_grid), f"mip {mip}"


def test_a_flush_job_whose_level_a_paint_already_flushed_does_nothing(sprite_install, monkeypatch):  # noqa: F811
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    _stroke(cache, scenario)
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [-1])
    cache._level(-1)
    calls = _level_work(monkeypatch)
    warmer.run_to_completion()
    assert calls == []
    _assert_matches_fresh(cache, scenario)


def test_a_unit_edit_mid_rebuild_drops_the_flush_jobs_result(sprite_install, monkeypatch):  # noqa: F811
    """Over cap the flush marks the level stale and the job rebuilds it across
    ticks; a mutation after the flush step must not get that rebuild installed."""
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 2)
    monkeypatch.setattr(level_warm, "BUDGET_MS", 0)
    _stroke(cache, scenario)
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [-1])
    assert warmer.tick(), "the flush job finished in one step -- vacuous"
    assert cache._levels[-1].gen != cache._source_gen, "the flush left the level current -- vacuous"
    cache.invalidate_units()
    calls = _level_work(monkeypatch, ("_commit_level",))
    warmer.run_to_completion()
    assert calls == [], "a rebuild started before the edit was installed"
    assert cache._levels[-1].gen != cache._source_gen


def test_a_held_mouse_button_pauses_the_flush_until_release(sprite_install, monkeypatch):  # noqa: F811
    """The per-step re-arm queues a flush job every stroke step; the held
    backoff is what keeps it from running in the gaps between move events."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )
    scenario, _units = _scenario()
    cache = _resident_cache(scenario)
    _stroke(cache, scenario)
    pending = {mip: set(cache._levels[mip].pending_elev) for mip in OFFSCREEN}
    warmer = level_warm.LevelWarmer()
    intervals: list = []
    monkeypatch.setattr(warmer, "_set_interval", intervals.append)
    warmer.start(cache, list(OFFSCREEN))

    for _ in range(3):
        assert warmer.tick() is True
    assert {mip: cache._levels[mip].pending_elev for mip in OFFSCREEN} == pending, "flushed while held"
    assert intervals and set(intervals) == {level_warm.LevelWarmer.HELD_INTERVAL_MS}

    held["value"] = False
    warmer.run_to_completion()
    assert intervals[-1] == 0
    _assert_matches_fresh(cache, scenario)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_an_elevate_stroke_end_queues_a_flush_job_and_the_load_warm_waits_for_it(
    sprite_install, monkeypatch,  # noqa: F811
):
    """The viewer wiring: _start_level_warm's stale-resident list now includes
    a pending level, and _queue_load_warm runs for it only once it is flushed."""
    from descape import settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.shown_terrain_window()
    try:
        scenario = window.scenario
        window._ensure_unit_edits().add(1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
        window.terrain_style_combo.setCurrentText("Sloped")
        window.terrain_style_combo.setCurrentText("Stepped")
        cache = window._cache
        visible = window.map_view.viewport_chunk_target()[0]
        other = visible + 1
        for mip in (visible, other):
            cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
        gen = cache._source_gen
        calls = []
        real = window._queue_load_warm

        def spy(mip: int) -> None:
            calls.append((mip, cache.is_level_resident(mip)))
            real(mip)

        monkeypatch.setattr(window, "_queue_load_warm", spy)
        window._on_tool_selected("elevation")
        window.brush_size_spin.setValue(1)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*UNIT_TILE, 0)
        window.on_edit_stroke_end()

        assert cache._source_gen == gen, "the splice fell back -- vacuous"
        assert cache._levels[other].pending_elev, "the other level deferred nothing -- vacuous"
        assert other in [mip for mip, _job, _notify in window._level_warmer._queue], "no flush job queued"
        assert not [c for c in calls if c[0] == other], "a load warm was queued for a pending level"
        assert window._load_warmer._mip != other

        window._level_warmer.run_to_completion()
        assert cache.is_level_resident(other)
        assert (other, True) in calls, "the flushed level's load warm never ran"
        mm = scenario.map_manager
        fresh = IsoChunkCache(
            scenario, window._iso_elevations.copy(), window._iso_proj,
            tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True,
        )
        dims = cache.canvas_dims(other)
        assert np.array_equal(cache.render_rect(0, 0, *dims, mip=other), fresh.render_rect(0, 0, *dims, mip=other))
    finally:
        conftest.close_window(window)
