"""Flat's unit row splice (FlatChunkCache.invalidate_units(changed)) driven by
the real viewer edits, sprites on: Convert, Place Unit, a Move (V) drag and a
single Delete. tests/test_invalidate_units_splice.py pins the splice at the
cache level; this checks that what the viewer hands it, and its own repaint
(patch_rects), leave no level stale.

A level index grows finer (render_cache._init_mip_levels), so a zoom out one
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

from descape import render
from descape.render_cache import FlatChunkCache

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


def _warm(window) -> dict[int, np.ndarray]:
    """The view's chunks resident at every checked level, plus their draws and icon layers."""
    shown = {mip: _level(window, mip) for mip in LEVELS}
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
    real_rows, real_splice = render._flat_unit_rows, cache._splice_rows

    def rows(*args, **kwargs):
        counts["rows"] += 1
        return real_rows(*args, **kwargs)

    def splice(changed):
        counts["splices"] += 1
        return real_splice(changed)

    monkeypatch.setattr(render, "_flat_unit_rows", rows)
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
