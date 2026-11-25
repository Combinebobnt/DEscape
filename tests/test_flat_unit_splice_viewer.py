"""Flat's unit row splice (FlatChunkCache.invalidate_units(changed)) driven by
the real viewer edits, sprites on: Convert, Place Unit, a Move (V) drag and a
single Delete. tests/test_invalidate_units_splice.py pins the splice at the
cache level; this checks that what the viewer hands it, and its own repaint
(patch_rects), leave no level stale.

A level index grows finer (render_cache_mip._init_mip_levels), so a zoom out one
level from level 0 lands on level -1. Each check renders level -1, then 0,
then 1 over the tiles the edits touch (the chunks a zoom out one level and
back, then one in, would ask for) and compares each with a fresh
FlatChunkCache built from the window's live scenario, filter, layers and grid.
All three are made resident before every edit, so a level the edit should
have repaired but left alone shows as a pixel mismatch rather than
recompositing fresh.

The synthetic install is tests/test_unit_edit_sprite_bbox.py's `art` fixture:
a Hero with a full team-colour mask (so Convert changes its pixels) and a
Villager.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_unit_edit_sprite_bbox import art  # noqa: F401 -- pytest fixture, imported for its name

from descape import render_flat
from descape.render_cache import FlatChunkCache
from testkit.render_family import set_render_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

HERO = 160
VILLAGER = 83
LEVELS = (-1, 0, 1)  # zoomed out one level, back, and one level in
VIEW_TILES = (14, 14, 46, 30)  # x0, y0, x1, y1: every unit below and every edit stays inside
# P1 heroes, a P1 villager, a P2 hero: distinct tiles, every block with neighbours.
_SETUP = [(1, HERO, 20.5, 20.5), (1, HERO, 24.5, 20.5), (1, VILLAGER, 28.5, 20.5), (1, HERO, 32.5, 20.5),
          (2, HERO, 36.5, 20.5)]


def _window():
    """The blank template, Flat with Isometric View off, Units mode, sprites on."""
    window = conftest.blank_window()
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    assert isinstance(window._cache, FlatChunkCache)
    assert window._cache.sprites_enabled
    return window


def _setup_units(window) -> list:
    model = window._ensure_unit_edits()
    assert model is not None
    units = []
    for player, const, x, y in _SETUP:
        with window._unit_edit(model, "Setup", [player]):
            units.append(model.add(player, const, x, y))
    return units


def _rect(cache, mip: int) -> tuple[int, int, int, int]:
    tp = cache.mip_tile_px(mip)
    x0, y0, x1, y1 = VIEW_TILES
    return x0 * tp, y0 * tp, x1 * tp, y1 * tp


def _level(window, mip: int) -> np.ndarray:
    cache = window._cache
    return cache.render_rect(*_rect(cache, mip), mip=mip).copy()


def _fresh(window, mip: int) -> np.ndarray:
    cache = window._cache
    fresh = FlatChunkCache(
        window.scenario, cache.tile_px, sprites=True, unit_filter=cache.unit_filter, layers=cache.layers
    )
    fresh.set_grid(cache.grid)
    return fresh.render_rect(*_rect(fresh, mip), mip=mip)


def _shown(window) -> dict[int, np.ndarray]:
    """The view's chunks resident at every checked level."""
    return {mip: _level(window, mip) for mip in LEVELS}


def _warm(window) -> dict[int, np.ndarray]:
    """The view's chunks resident at every checked level, plus their draws and icon layers."""
    shown = _shown(window)
    cache = window._cache
    assert all(cache._level_icon_layers.get(mip) for mip in LEVELS), "icons are not resolving at every level"
    return shown


def _check(window, what: str, before: dict[int, np.ndarray]) -> None:
    for mip in LEVELS:
        now = _level(window, mip)
        fresh = _fresh(window, mip)
        if not np.array_equal(now, fresh):
            ys, xs = np.nonzero((now != fresh).any(axis=2))
            pytest.fail(f"{what}: {len(xs)} stale px at level {mip}, x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")
        assert not np.array_equal(now, before[mip]), f"{what} changed nothing at level {mip}, so this proves nothing"


def _count_paths(window, monkeypatch) -> dict[str, int]:
    """Row splices taken, and wholesale level-0 row walks. A wholesale
    invalidate_units() re-walks the rows at once; LevelWarmer's icon walk for a
    level nobody has shown yet is not a fallback, so icons are not counted."""
    cache = window._cache
    counts = {"splices": 0, "rows": 0}
    real_rows, real_splice = render_flat._flat_unit_base_rows, cache._splice_rows

    def rows(*args, **kwargs):
        counts["rows"] += 1
        return real_rows(*args, **kwargs)

    def splice(changed):
        counts["splices"] += 1
        return real_splice(changed)

    set_render_global(monkeypatch, "_flat_unit_base_rows", rows)
    monkeypatch.setattr(cache, "_splice_rows", splice)
    return counts


def _gesture(window, from_tile, to_tile=None, *, tool: str) -> None:
    """A real press(-move)-release through MapView under `tool`."""
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QMouseEvent

    getattr(window, f"{tool}_action").setChecked(True)
    view = window.map_view
    press = conftest.viewport_pos(view, *from_tile)
    end = press if to_tile is None else conftest.viewport_pos(view, *to_tile)
    view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    if end != press:
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


def _select(window, player: int, unit) -> None:
    window._selection = [(player, unit.reference_id)]
    window._refresh_selection_view()


def test_flat_viewer_edits_splice_and_leave_every_level_like_a_fresh_cache(art, monkeypatch) -> None:  # noqa: F811
    """TASK-031.02 step 1. Pass conditions, for each of Convert (a stroke on
    a P1 hero's tile, to Player 2), Place Unit (a hero on an empty tile), a
    Move tool drag of one hero by two tiles, and a single Delete:
    - the edit took the row splice, and no wholesale row walk ran;
    - levels -1, 0 and 1, each rendered from the live cache over the edited
      tiles, equal a fresh cache's render of the same rect, byte for byte;
    - each level changed, so the comparison is not vacuous."""
    from PyQt5.QtCore import QPointF, Qt

    window = _window()
    try:
        units = _setup_units(window)
        tp = window.map_view._tile_pixels
        # Zoomed in, so a two-tile drag clears UNIT_DRAG_THRESHOLD_PX.
        window.map_view.scale(4.0, 4.0)
        window.map_view.centerOn(QPointF(28 * tp, 22 * tp))
        counts = _count_paths(window, monkeypatch)

        def edit(what: str, run) -> None:
            before = _warm(window)
            counts.update(splices=0, rows=0)
            run()
            assert counts["splices"] >= 1 and counts["rows"] == 0, f"{what} did not splice: {counts}"
            _check(window, what, before)

        def convert() -> None:
            window.units_panel.select_owner(2)
            window.brush_size_spin.setValue(1)
            _gesture(window, (24, 20), tool="convert")
            assert units[1] in window.scenario.unit_manager.units[2]

        def place() -> None:
            window.units_panel.select_object(HERO)
            window.units_panel.select_owner(1)
            window.place_unit_action.setChecked(True)
            window.on_unit_place(QPointF(30.5 * tp, 24.5 * tp), Qt.NoModifier)
            assert window.scenario.unit_manager.units[1][-1].unit_const == HERO

        def move() -> None:
            _select(window, 1, units[0])
            _gesture(window, (20, 20), (22, 22), tool="move")
            assert (int(units[0].x), int(units[0].y)) == (22, 22)

        def delete() -> None:
            _select(window, 1, units[3])
            window.on_unit_delete(Qt.NoModifier)
            assert units[3] not in window.scenario.unit_manager.units[1]

        for what, run in (("Convert", convert), ("Place", place), ("Move drag", move), ("Delete", delete)):
            edit(what, run)
    finally:
        conftest.close_window(window)


# --- Visible-level patch and the batch evict rule (TASK-442) ---

FOREST_OAK = 10  # density 1000: one tree per tile
FAR_TILES = (80, 80, 100, 100)  # a rect no edit below reaches, at every level
STROKE_A = [(18 + i, 26) for i in range(8)]
STROKE_B = [(18 + i, 28) for i in range(8)]


def _pin_view(window, monkeypatch, mip=0) -> None:
    """viewport_chunk_target() pinned to the whole of level `mip`, or to None."""
    cache = window._cache
    cache.max_bytes *= 8  # every checked level resident at once, plus FAR_TILES
    if mip is None:
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: None)
        return
    rng = cache.chunk_index_range(mip, 0, 0, *cache.canvas_dims(mip))
    monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: (mip, *rng))


def _spy_patches(window, monkeypatch) -> list:
    """Every patch_rects() call from here on, as (rects, levels)."""
    cache = window._cache
    handed = []
    real = cache.patch_rects

    def patch_rects(rects, levels=None):
        handed.append((list(rects), levels))
        return real(rects, levels=levels)

    monkeypatch.setattr(cache, "patch_rects", patch_rects)
    return handed


def _warm_far(window) -> dict:
    """FAR_TILES resident at every checked level; {key: id(chunk)} of every chunk then cached."""
    cache = window._cache
    for mip in LEVELS:
        tp = cache.mip_tile_px(mip)
        cache.render_rect(*(v * tp for v in FAR_TILES), mip=mip)
    return {key: id(chunk) for key, chunk in cache._cache.items()}


def _keys_under(cache, rects, mip) -> set:
    keys = set()
    for rect in rects:
        rng = cache.chunk_range_under(rect, mip)
        if rng is not None:
            keys |= {(mip, cx, cy) for cy in range(rng[1], rng[3] + 1) for cx in range(rng[0], rng[2] + 1)}
    return keys


def _assert_visible_patch(window, what: str, handed: list, ids: dict) -> None:
    """One patch_rects(levels=(0,)) call; mip 0 patched in place; mips -1 and 1
    dropped their chunks under the rects and kept one off them."""
    cache = window._cache
    assert [levels for _rects, levels in handed] == [(0,)], f"{what}: {handed}"
    rects = handed[0][0]
    now = {key: id(chunk) for key, chunk in cache._cache.items()}
    under0 = _keys_under(cache, rects, 0)
    assert under0 & set(ids), f"{what}: no mip-0 chunk was cached under the rects"
    for key, ident in ids.items():
        if key[0] == 0:
            assert now.get(key) == ident, f"{what}: mip-0 chunk {key} was replaced or evicted"
    for mip in (-1, 1):
        under = _keys_under(cache, rects, mip)
        assert under & set(ids), f"{what}: no mip-{mip} chunk was cached under the rects"
        assert not under & set(now), f"{what}: mip {mip} kept {sorted(under & set(now))} under the rects"
        kept = [key for key in ids if key[0] == mip and key not in under and now.get(key) == ids[key]]
        assert kept, f"{what}: mip {mip} kept no chunk off the rects"


def test_flat_single_unit_edits_patch_the_visible_level_and_drop_the_others(art, monkeypatch) -> None:  # noqa: F811
    """T1: Place, a Move drag and a Delete each hand patch_rects() levels=(0,)
    with mip 0 in view; mip 0's chunks are patched in place, mips -1 and 1 drop
    theirs under the rects and keep one off them, and every level equals a fresh cache."""
    from PyQt5.QtCore import QPointF, Qt

    window = _window()
    try:
        units = _setup_units(window)
        tp = window.map_view._tile_pixels
        window.map_view.scale(4.0, 4.0)
        window.map_view.centerOn(QPointF(28 * tp, 22 * tp))
        _pin_view(window, monkeypatch, 0)

        def place() -> None:
            window.units_panel.select_object(HERO)
            window.units_panel.select_owner(1)
            window.place_unit_action.setChecked(True)
            window.on_unit_place(QPointF(30.5 * tp, 24.5 * tp), Qt.NoModifier)
            assert window.scenario.unit_manager.units[1][-1].unit_const == HERO

        def move() -> None:
            _select(window, 1, units[0])
            _gesture(window, (20, 20), (22, 22), tool="move")
            assert (int(units[0].x), int(units[0].y)) == (22, 22)

        def delete() -> None:
            _select(window, 1, units[3])
            window.on_unit_delete(Qt.NoModifier)
            assert units[3] not in window.scenario.unit_manager.units[1]

        for what, run in (("Place", place), ("Move drag", move), ("Delete", delete)):
            before = _warm(window)
            ids = _warm_far(window)
            handed = _spy_patches(window, monkeypatch)
            run()
            _assert_visible_patch(window, what, handed, ids)
            _check(window, what, before)
    finally:
        conftest.close_window(window)


def test_a_wholesale_flat_unit_edit_rebuilds_only_the_visible_icon_layer(art, monkeypatch) -> None:  # noqa: F811
    """T2: a single-unit Nudge the cache cannot splice takes the wholesale
    source path; afterwards only mip 0 holds an icon layer (none rebuilt for
    -1 or 1 inside the handler), and every level equals a fresh cache."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        units = _setup_units(window)
        _pin_view(window, monkeypatch, 0)
        cache = window._cache
        monkeypatch.setattr(cache, "_flat_splice_eligible", lambda changed: False)
        counts = _count_paths(window, monkeypatch)
        _select(window, 1, units[0])
        before = _warm(window)
        handed = _spy_patches(window, monkeypatch)

        # A whole tile: a sub-tile Nudge's icon spills past the one tile it patches.
        assert window.on_unit_nudge(1, 0, Qt.ShiftModifier)

        assert counts["rows"] >= 1, f"the Nudge did not take the wholesale source path: {counts}"
        assert [levels for _rects, levels in handed] == [(0,)], handed
        icons = cache._level_icon_layers
        assert icons.get(0), "the visible level's icon layer was not rebuilt"
        assert -1 not in icons and 1 not in icons, f"icon layers rebuilt in the handler: {sorted(icons)}"
        _check(window, "wholesale Nudge", before)
    finally:
        conftest.close_window(window)


def _draw_window(monkeypatch):
    """The blank template, top-down Flat, Terrain mode, Draw with Trees on, mip 0 pinned in view."""
    from testkit.viewer_family import set_viewer_global

    window = conftest.terrain_edit_window()
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window._on_tool_selected("draw")
    window.paint_trees_check.setChecked(True)
    window.paint_eye_candy_check.setChecked(False)
    window.terrain_panel.set_terrain(FOREST_OAK)
    assert isinstance(window._cache, FlatChunkCache) and window._cache.sprites_enabled
    # The blank template has too few units for a batch to beat Flat's wholesale rebuild.
    set_viewer_global(monkeypatch, "_SPLICE_COST_RATIO", 0)
    _pin_view(window, monkeypatch, 0)
    return window


def _spy_region_evictions(window, monkeypatch) -> list:
    """Every cache invalidate_region() call from here on, as (bbox, levels, terrain_changed)."""
    cache = window._cache
    calls = []
    real = cache.invalidate_region

    def invalidate(bbox, levels=None, terrain_changed=True):
        calls.append((bbox, levels, terrain_changed))
        return real(bbox, levels=levels, terrain_changed=terrain_changed)

    monkeypatch.setattr(cache, "invalidate_region", invalidate)
    return calls


def _stroke_end_under(window, monkeypatch, tiles, ratio: float):
    """A live-painted Draw stroke over `tiles` at the default ratio, every
    level rendered over it, then `ratio` set and the spies armed just before
    on_edit_stroke_end(). Returns (patches, evictions, before)."""
    from testkit.viewer_family import set_viewer_global

    cache = window._cache
    set_viewer_global(monkeypatch, "_FLAT_EVICT_RATIO", 1.0)
    before = _shown(window)
    window.on_edit_stroke_start()
    for x, y in tiles:
        window.on_edit_stroke_tile(x, y, 0)
    xs, ys = [x for x, _y in tiles], [y for _x, y in tiles]
    bbox = (min(xs) * cache.tile_px, min(ys) * cache.tile_px, (max(xs) + 1) * cache.tile_px, (max(ys) + 1) * cache.tile_px)
    _shown(window)
    for mip in LEVELS:
        assert _keys_under(cache, [bbox], mip) & set(cache._cache), f"mip {mip}: nothing cached under the stroke"
    set_viewer_global(monkeypatch, "_FLAT_EVICT_RATIO", ratio)
    patches = _spy_patches(window, monkeypatch)
    evictions = _spy_region_evictions(window, monkeypatch)
    window.on_edit_stroke_end()
    return patches, evictions, before


def test_a_flat_batch_unit_edit_evicts_its_bbox_past_the_ratio_and_patches_under_it(art, monkeypatch) -> None:  # noqa: F811
    """T3: a spliced Draw stroke end with Trees on. At _FLAT_EVICT_RATIO 0 it
    hands patch_rects() nothing and every level's chunks under the bbox are
    gone; at inf it patches levels=(0,); a non-batch Nudge at ratio 0 still
    patches. Every level equals a fresh cache after each."""
    from PyQt5.QtCore import Qt

    from testkit.viewer_family import set_viewer_global

    window = _draw_window(monkeypatch)
    try:
        cache = window._cache
        after_calls = []
        real_after = window._after_unit_mutation

        def after(changed=None, **kwargs):
            after_calls.append((changed, kwargs.get("batch", False)))
            return real_after(changed, **kwargs)

        monkeypatch.setattr(window, "_after_unit_mutation", after)

        patches, evictions, before = _stroke_end_under(window, monkeypatch, STROKE_A, 0.0)
        changed, batch = after_calls[-1]
        assert changed and batch, f"the stroke end was not a spliced batch: {after_calls[-1]}"
        assert patches == [], patches
        ((bbox, levels, terrain_changed),) = evictions
        assert levels is None and terrain_changed is False, evictions
        for mip in LEVELS:
            assert not _keys_under(cache, [bbox], mip) & set(cache._cache), f"ratio 0: mip {mip} kept chunks"
        _check(window, "ratio 0 stroke end", before)

        patches, evictions, before = _stroke_end_under(window, monkeypatch, STROKE_B, float("inf"))
        changed, batch = after_calls[-1]
        assert changed and batch, f"the stroke end was not a spliced batch: {after_calls[-1]}"
        assert [levels for _rects, levels in patches] == [(0,)], patches
        assert evictions == [], evictions
        _check(window, "ratio inf stroke end", before)

        set_viewer_global(monkeypatch, "_FLAT_EVICT_RATIO", 0.0)
        window.mode_combo.setCurrentText("Units")
        tree = next(u for u in window.scenario.unit_manager.units[0] if (int(u.x), int(u.y)) == STROKE_B[0])
        _select(window, 0, tree)
        before = _warm(window)
        patches = _spy_patches(window, monkeypatch)
        assert window.on_unit_nudge(0, 1, Qt.ShiftModifier)
        assert not after_calls[-1][1], "the Nudge was a batch"
        assert [levels for _rects, levels in patches] == [(0,)], patches
        _check(window, "Nudge at ratio 0", before)
    finally:
        window.edit_history.mark_saved()
        conftest.close_window(window)


def test_a_flat_unit_edit_with_no_viewport_target_patches_every_level(art, monkeypatch) -> None:  # noqa: F811
    """T4: viewport_chunk_target() None: a Place hands patch_rects() levels=None
    and every level equals a fresh cache."""
    from PyQt5.QtCore import QPointF, Qt

    window = _window()
    try:
        _setup_units(window)
        _pin_view(window, monkeypatch, None)
        tp = window.map_view._tile_pixels
        before = _warm(window)
        handed = _spy_patches(window, monkeypatch)
        window.units_panel.select_object(HERO)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        window.on_unit_place(QPointF(30.5 * tp, 24.5 * tp), Qt.NoModifier)
        assert window.scenario.unit_manager.units[1][-1].unit_const == HERO
        assert [levels for _rects, levels in handed] == [None], handed
        _check(window, "Place with no target", before)
    finally:
        conftest.close_window(window)


def test_a_unit_brush_flush_and_stroke_end_patch_the_visible_level(art, monkeypatch) -> None:  # noqa: F811
    """T5: a Unit Brush stroke at 100% density over two tiles, flushed after
    the first: the flush and the stroke end (its pending list non-empty) each
    hand patch_rects() levels=(0,), and every level equals a fresh cache after the end."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        _setup_units(window)
        _pin_view(window, monkeypatch, 0)
        assert window.units_panel.select_owner(1)
        assert window.units_panel.pick_object(HERO)
        window.unit_brush_action.setChecked(True)
        window.brush_size_spin.setValue(1)
        window.unit_brush_density_spin.setValue(100)
        before = _warm(window)
        handed = _spy_patches(window, monkeypatch)
        window.map_view._stroke_button = Qt.LeftButton
        window.on_edit_stroke_start()
        window.on_edit_stroke_tiles([(22, 24)], Qt.NoModifier)
        window._flush_unit_stroke_refresh()
        assert [levels for _rects, levels in handed] == [(0,)], f"the flush: {handed}"
        window.on_edit_stroke_tiles([(26, 24)], Qt.NoModifier)
        assert window._unit_stroke_pending_splices, "the second tile queued nothing for the stroke end"
        window.on_edit_stroke_end()
        window.map_view._stroke_button = None
        assert [levels for _rects, levels in handed] == [(0,), (0,)], f"the stroke end: {handed}"
        _check(window, "Unit Brush", before)
    finally:
        conftest.close_window(window)


def test_flat_convert_to_a_player_filters_hide_then_unhide_shows_the_units(art) -> None:  # noqa: F811
    """TASK-031.02 step 2, view level. Pass conditions:
    - with Player 2 unticked in Filters, a Convert stroke over a P1 hero hands
      it to Player 2, and every checked level matches a fresh cache under that
      filter (the hero is gone from view, so each level changed);
    - ticking Player 2 back on: every checked level matches a fresh cache
      again and differs from the hidden state, so the converted hero reappears."""
    window = _window()
    try:
        units = _setup_units(window)
        window.player_actions[2].setChecked(False)
        window.units_panel.select_owner(2)
        window.brush_size_spin.setValue(1)
        before = _warm(window)
        _gesture(window, (24, 20), tool="convert")
        assert units[1] in window.scenario.unit_manager.units[2]
        _check(window, "Convert to a hidden player", before)

        hidden = _warm(window)
        window.player_actions[2].setChecked(True)
        _check(window, "Un-hide", hidden)
    finally:
        conftest.close_window(window)
