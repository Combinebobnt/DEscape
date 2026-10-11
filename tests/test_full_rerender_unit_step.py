"""TASK-031.61 A2b: a tile + unit history move whose tile patch takes a full
branch (past STEPPED_FULL_RERENDER_THRESHOLD, a new cache built from the
already-restored units) runs its unit step wholesale. Applying the move's
splices on top of that cache inserted an add a second time. Since A5 that
branch refreshes the cache in place instead, and the same checks hold.

Each step re-renders the whole canvas right before it (GOTCHAS: an evicted
chunk redraws fresh and hides a stale one), then checks units_by_tile and the
pixels against fresh builds, failing as `<step>: ...`. Every window here ends
with edit_history.mark_saved() before close()."""

from __future__ import annotations

import numpy as np
import pytest
from test_region_paste_render import _block_window, _paste

from descape import render

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FOREST_OAK = 10  # density 1000: one tree per tile
FOREST_PALM = 13  # density 1000, a different tree
BEACH = 2
REGION = (20, 20, 32, 32)  # 144 tiles: past the pinned threshold, under the 1000-unit cap


def _canvas(window) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    return window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=0).copy()


def _fresh(window, style: str) -> np.ndarray:
    canvas_w, canvas_h = window._cache.canvas_dims(0)
    sprites = window._cache.sprites_enabled
    if style == "Stepped":
        full = render.render_terrain_iso_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    else:
        full = render.render_terrain_sloped_with_proj(window.scenario, with_units=True, with_sprites=sprites)[0]
    return full[:canvas_h, :canvas_w]


def _ids(by_tile: dict) -> dict:
    return {tile: [(id(u), color) for u, color in entries] for tile, entries in by_tile.items() if entries}


def _check(window, style: str, step: str) -> None:
    cache = window._cache
    fresh = render._units_by_tile(window.scenario, cache.unit_filter)
    assert _ids(cache.units_by_tile) == _ids(fresh), f"{step}: units_by_tile differs from a fresh build"
    assert np.array_equal(_canvas(window), _fresh(window, style)), f"{step}: differs from a fresh render"


def _pin_threshold(monkeypatch, value: int) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", value)


def _fill(window, terrain_id: int) -> int:
    """A Trees-on Paint Can of `terrain_id` into REGION; returns the GAIA units removed + added."""
    before = list(window.scenario.unit_manager.units[0])
    window.terrain_panel.set_terrain(terrain_id)
    window.on_fill(REGION[0], REGION[1], 0)
    after = window.scenario.unit_manager.units[0]
    ids_before, ids_after = {id(u) for u in before}, {id(u) for u in after}
    return len(ids_before - ids_after) + len(ids_after - ids_before)


def _filled_window(style: str, monkeypatch):
    """The blank template with a BEACH patch over REGION (Trees off), then two
    Trees-on Paint Cans into it: FOREST_OAK (144 trees added), then FOREST_PALM
    over it (each oak removed, a palm added), so both the undo and the redo of
    the second bring units back. 144 tiles, 288 unit changes."""
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 100_000)
    window = conftest.terrain_edit_window()
    if style != "Stepped":
        window.terrain_style_combo.setCurrentText(style)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window._on_tool_selected("draw")
    window.terrain_panel.set_terrain(BEACH)
    x0, y0, x1, y1 = REGION
    window.on_edit_stroke_start()
    for y in range(y0, y1):
        for x in range(x0, x1):
            window.on_edit_stroke_tile(x, y, 0)
    window.on_edit_stroke_end()
    _pin_threshold(monkeypatch, 50)
    window._on_tool_selected("fill")
    window.paint_trees_check.setChecked(True)
    assert _fill(window, FOREST_OAK) == 144
    assert (x1 - x0) * (y1 - y0) > 50 and _fill(window, FOREST_PALM) == 288
    return window


def _undo_redo(window, style: str, before_step=None) -> None:
    for step, act in (("undo", window.undo), ("redo", window.redo), ("undo 2", window.undo)):
        _canvas(window)
        if before_step is not None:
            before_step(step)
        act()
        _check(window, style, step)


def _refuse_refresh(monkeypatch) -> None:
    """Every cache refuses A5's in-place refresh, so the threshold branch
    re-renders (the out-of-range fallback): a new cache built from the
    restored units, the case A2b's wholesale unit step exists for."""
    from descape.render_cache import IsoChunkCache, SlopedChunkCache

    for cls in (IsoChunkCache, SlopedChunkCache):
        monkeypatch.setattr(cls, "refresh_wholesale", lambda self, *a, **k: None)


@pytest.mark.parametrize("full", ["refresh", "rerender"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_trees_fill_past_the_threshold_undoes_and_redoes_like_a_fresh_build(style, full, monkeypatch) -> None:
    if full == "rerender":
        _refuse_refresh(monkeypatch)
    window = _filled_window(style, monkeypatch)
    try:
        _check(window, style, "fill")
        _undo_redo(window, style)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("full", ["refresh", "rerender"])
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_the_bench_reads_the_threshold_branch_status_line_as_its_path(style, full, monkeypatch) -> None:
    """tools/bench_gui_drag.py's G4 path comes from the undo's status line: a
    refused refresh's re-render must not read as `patch`."""
    from tools import bench_gui_drag

    if full == "rerender":
        _refuse_refresh(monkeypatch)
    window = _filled_window(style, monkeypatch)
    try:
        before = len(window.status_log.toPlainText().splitlines())
        window.undo()
        sections = {"undo2": window.status_log.toPlainText().splitlines()[before:]}
        stalls = dict.fromkeys(("draw", "undo1", "undo2"), (0.0, 0.0, 0.0))
        assert f" G4_undo2_path={full}:144 " in bench_gui_drag.gh180_gates(sections, stalls) + " "
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("full", ["refresh", "rerender"])
@pytest.mark.parametrize("block", ["no_elevation", "plain"])
def test_a_paste_shaped_record_past_the_threshold_undoes_and_redoes_like_a_fresh_build(
    block, full, monkeypatch
) -> None:
    """Paste Region with units, the threshold pinned under the block so each
    tile patch takes the full branch. Without elevation its undo and redo take
    the tiles-first branch; with it (`plain`), the sources-first one."""
    if full == "rerender":
        _refuse_refresh(monkeypatch)
    window = _block_window(block, monkeypatch)
    try:
        _pin_threshold(monkeypatch, 5)
        _canvas(window)
        _paste(window)
        _check(window, "Stepped", "paste")
        _undo_redo(window, "Stepped")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("broken", ["undo", "redo", "undo 2"])
def test_the_sequence_check_names_the_one_step_whose_unit_step_went_stale(broken, monkeypatch) -> None:
    """Control (GOTCHAS: a step-k break fails naming step k): on step `broken`
    only, _apply_dirty() reports its full re-render as a patch, so that step
    applies its splices on top (the pre-fix behaviour), and the check must
    fail there. The re-render is forced (_refuse_refresh()): since A5 a fill
    past the threshold refreshes in place, keeping the pre-move sources (it
    moves no elevation), so splicing after it is harmless."""
    _refuse_refresh(monkeypatch)
    window = _filled_window("Stepped", monkeypatch)
    try:
        current = {}
        real_dirty = window._apply_dirty

        def apply_dirty(dirty, **kwargs):
            full = real_dirty(dirty, **kwargs)
            return False if current.get("step") == broken else full

        monkeypatch.setattr(window, "_apply_dirty", apply_dirty)
        with pytest.raises(AssertionError, match=f"{broken}: "):
            _undo_redo(window, "Stepped", before_step=lambda step: current.update(step=step))
    finally:
        window.edit_history.mark_saved()
        window.close()
