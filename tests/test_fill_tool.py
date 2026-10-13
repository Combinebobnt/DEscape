"""Paint Can wiring, driven through a real offscreen ViewerWindow -- the
Qt-registry and click-routing half of the feature that tests/test_fill_
tools.py's fake-tile unit tests can't reach (QAction/keybind wiring, real
_update_tool_enabled gating, and the actual MapView.mousePressEvent click
branch vs. the drag-stroke path).

Same technique tools/verify_copy_paste.py already established
(QT_QPA_PLATFORM=offscreen, one shared QApplication via conftest.
ensure_qapp()), but written natively as pytest per tests/README.md's stated
convention ("New tools/verify_*.py scripts are not the convention going
forward") -- mirroring tests/test_new_map.py/test_lazy_viewport.py, not the
corpus-marked legacy-adapter pattern. Runs against the shipped blank
120x120 template only -- no examples/ corpus needed, so this stays in the
default tier.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- close() runs _confirm_discard_changes(), which pops a
modal QMessageBox on a dirty document and would hang an offscreen run (same
requirement tests/test_new_map.py's own docstring states).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_toggle_viewer import sprite_install  # noqa: F401 -- fixture, requested by name

# blend_install is requested by name and requests the next two itself;
# blend_keys_match_the_mirrors is autouse, so every composite here checks the cache's blend key map.
from test_terrain_blend_render import (  # noqa: F401
    blend_install,
    blend_keys_match_the_mirrors,
    fake_farm,
    synthetic_textures,
)

from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# Any two fixed TerrainId values present in every DE structure version (same
# pair tools/verify_copy_paste.py and tools/verify_write_path.py use, for the
# same reason) -- both distinct from the blank template's own terrain_id=0,
# so a fill is actually observable. A second value is needed by the
# mid-drag-switch test below: once a fill covers (nearly) the whole map with
# _FILL_TERRAIN, painting a single tile with that same terrain_id again would
# be a real no-op (commit_stroke() pushes no record for it) -- not the
# "another edit still works" case that test needs to observe.
_FILL_TERRAIN = 15  # GRASS_1
_OTHER_TERRAIN = 2  # BEACH


def _edit_window(tool: str = "fill"):
    """Loads the blank template, switches to Terrain mode with `tool` active,
    and selects _FILL_TERRAIN in the terrain panel. Caller must
    edit_history.mark_saved() + close()."""
    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_FILL_TERRAIN)
    # This file tests Paint Can's own mechanics, not descape/terrain_units.py
    # -- Trees defaults on and would otherwise pop an unpatched large-fill
    # confirm QMessageBox for the whole-map fills below, hanging offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    return window


def _shown_flat(window) -> None:
    """Forces Flat style (a plain int-division tile lookup, unlike Stepped's
    screen_to_tile) and pumps a real show()/processEvents() cycle so
    map_view's transform is the actual one fitInView would compute --
    required before mapFromScene() below can be trusted, and the same
    "must actually show the top-level window" requirement tests/
    test_lazy_viewport.py's own _show_and_settle() documents."""
    from PyQt5.QtWidgets import QApplication

    # Unchecked BEFORE the style switch, while still in Stepped -- iso_action
    # defaults checked (MapView._isometric's own default), so Flat would
    # otherwise render through the Flat+Isometric plan's real-iso path
    # instead of the plain int-division canvas this module means to exercise.
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.show()
    QApplication.processEvents()


def test_fill_registered_as_keybind_and_toolbar_action() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    assert ("tool_fill", "Paint Can Tool", "P") in settings.REBINDABLE_ACTIONS

    window = ViewerWindow()
    try:
        # Proves ViewerWindow._build_keybind_actions's
        # getattr(self, f"{tool_id}_action") wiring
        # (no default -- an AttributeError there would already have failed
        # window construction, but this pins the specific mapping too).
        assert window._keybind_actions["tool_fill"] is window.fill_action
        assert window.fill_action.shortcut() == QKeySequence("P")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_fill_enable_gating_matches_draw_tool() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.fill_action.isEnabled()  # no map loaded yet

        window.load_scenario(BLANK_TEMPLATE_PATH)
        # Mode-inapplicable tools hide rather than grey now (ToolDef.modes),
        # so View mode is asserted on visibility, not enabled state -- write_ok
        # alone already makes fill_action enabled once a map is loaded.
        assert not window.fill_action.isVisible()  # still View mode

        window.mode_combo.setCurrentText("Terrain")
        assert window.fill_action.isVisible()
        assert window.fill_action.isEnabled()
        # Paint Can shares Draw's gate (write_ok only), not Elevate's
        # (write_ok and map_is_square) -- assert the two stay equal rather
        # than hardcoding True, so a future gating change to either tool
        # can't silently diverge them unnoticed.
        assert window.fill_action.isEnabled() == window.draw_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_selecting_fill_configures_map_view() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QGraphicsView

    from descape.viewer_common import CLICK_TOOLS, EDIT_TOOLS

    window = _edit_window("fill")
    try:
        assert window._current_tool == "fill"
        assert window.map_view._tool == "fill"
        assert window.map_view.dragMode() == QGraphicsView.NoDrag
        assert window.map_view.cursor().shape() == Qt.CrossCursor
        assert "fill" in EDIT_TOOLS
        assert "fill" in CLICK_TOOLS
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
def test_fill_end_to_end_and_undo(style: str) -> None:
    window = _edit_window("fill")
    try:
        window.terrain_style_combo.setCurrentText(style)
        mm = window.scenario.map_manager

        window.on_fill(0, 0, 0)

        assert all(t.terrain_id == _FILL_TERRAIN for t in mm.terrain)
        assert all(t.layer == -1 for t in mm.terrain)
        assert len(window.edit_history.records) == 1
        assert window.edit_history.records[-1].label == "Fill terrain"
        assert window.edit_history.can_undo

        window.undo()
        assert all(t.terrain_id == 0 for t in mm.terrain)
        assert not window.edit_history.can_undo
        assert window.edit_history.can_redo
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_drag_after_click_fills_only_once() -> None:
    """The load-bearing regression test: a fill routed through the ordinary
    stroke chain instead of the dedicated CLICK_TOOLS branch would re-fire
    once per drag-entered tile. Synthesizes real QMouseEvents into map_view
    (press, several moves with the button held, release) rather than
    calling on_fill()/on_click_edit() directly, so it actually exercises
    MapView.mousePressEvent's routing, not just ViewerWindow's handler."""
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("fill")
    try:
        _shown_flat(window)
        map_view = window.map_view

        press_pos = conftest.viewport_pos(map_view, 0, 0)
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press_pos, Qt.LeftButton, Qt.LeftButton))
        assert len(window.edit_history.records) == 1
        assert map_view._stroke_active is False

        for tx, ty in [(1, 0), (2, 0), (0, 1)]:
            move_pos = conftest.viewport_pos(map_view, tx, ty)
            map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, move_pos, Qt.NoButton, Qt.LeftButton))
            assert len(window.edit_history.records) == 1

        map_view.mouseReleaseEvent(
            conftest.mouse_event(QEvent.MouseButtonRelease, press_pos, Qt.LeftButton, Qt.NoButton)
        )
        assert len(window.edit_history.records) == 1
        assert map_view._stroke_active is False
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_right_click_is_inert() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("fill")
    try:
        _shown_flat(window)
        map_view = window.map_view
        pos = conftest.viewport_pos(map_view, 0, 0)

        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.RightButton, Qt.RightButton))

        assert window.edit_history.records == []
        assert map_view._stroke_active is False
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_repeat_fill_at_same_tile_is_a_logged_no_op() -> None:
    window = _edit_window("fill")
    try:
        window.on_fill(0, 0, 0)
        assert len(window.edit_history.records) == 1

        window.on_fill(0, 0, 0)
        assert len(window.edit_history.records) == 1  # still just one -- no phantom record
        assert "no change" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def _layer_tiles(window, layers: dict[tuple[int, int], int]) -> None:
    """Gives the named tiles a layered terrain directly, as a loaded file would."""
    mm = window.scenario.map_manager
    for (x, y), layer_id in layers.items():
        mm.terrain[y * mm.map_width + x].layer = layer_id


def _layers(window) -> dict[tuple[int, int], int]:
    mm = window.scenario.map_manager
    width = mm.map_width
    return {(i % width, i // width): t.layer for i, t in enumerate(mm.terrain) if t.layer != -1}


def test_same_terrain_fill_clears_the_regions_layers_in_one_undo_step(monkeypatch) -> None:
    import descape.viewer as viewer_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a small layer clear must not ask")

    window = _edit_window("fill")
    try:
        mm = window.scenario.map_manager
        window.on_fill(0, 0, 0)  # the whole map is _FILL_TERRAIN now
        layered = {(3, 5): 9, (4, 5): 9, (40, 41): 11}
        _layer_tiles(window, layered)
        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))
        # Trees on: a layer clear changes no terrain_id, so the unit plan must not even run.
        window.paint_trees_check.setChecked(True)
        monkeypatch.setattr(window, "_apply_terrain_unit_plan", fail_if_called)

        window.on_fill(60, 60, 0)

        assert _layers(window) == {}
        assert all(t.terrain_id == _FILL_TERRAIN for t in mm.terrain)
        assert len(window.edit_history.records) == 2
        record = window.edit_history.records[-1]
        assert record.label == "Fill terrain"
        assert len(record.changes) == 3
        assert "Cleared layered terrain on 3 tiles from (60, 60)" in window.status_log.toPlainText()

        window.undo()
        assert _layers(window) == layered
        assert all(t.terrain_id == _FILL_TERRAIN for t in mm.terrain)
        window.redo()
        assert _layers(window) == {}

        window.on_fill(60, 60, 0)  # nothing left to clear
        assert len(window.edit_history.records) == 2
        assert "no change" in window.status_log.toPlainText().splitlines()[-1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_large_same_terrain_fill_asks_and_cancel_leaves_everything(monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 10)
    window = _edit_window("fill")
    try:
        mm = window.scenario.map_manager
        window.on_fill(0, 0, 0)  # trees and eye candy are off, so this one does not ask
        layered = {(x, 7): 9 for x in range(20, 31)}  # 11 tiles, one past the threshold
        _layer_tiles(window, layered)
        calls = []

        def cancel(*args, **kwargs):
            calls.append(args)
            return QMessageBox.Cancel

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(cancel))
        window.on_fill(60, 60, 0)

        assert len(calls) == 1
        assert "layered terrain on 11 tiles" in calls[0][2]
        assert _layers(window) == layered
        assert len(window.edit_history.records) == 1
        assert not window._busy

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
        window.on_fill(60, 60, 0)
        assert _layers(window) == {}
        assert len(window.edit_history.records) == 2
        assert all(t.terrain_id == _FILL_TERRAIN for t in mm.terrain)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_same_terrain_fill_at_the_threshold_does_not_ask(monkeypatch) -> None:
    import descape.viewer as viewer_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a clear of exactly the threshold must not ask")

    monkeypatch.setattr(viewer_module, "TERRAIN_UNIT_CONFIRM_THRESHOLD", 10)
    window = _edit_window("fill")
    try:
        window.on_fill(0, 0, 0)
        _layer_tiles(window, {(x, 7): 9 for x in range(20, 30)})  # exactly 10
        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))
        window.on_fill(60, 60, 0)
        assert _layers(window) == {}
        assert len(window.edit_history.records) == 2
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_mid_drag_tool_switch_to_fill_does_not_wedge_history() -> None:
    """Regression guard for the stroke-chain shape this feature deliberately
    avoided (see MapView.mousePressEvent's own comment): if Paint Can were
    ever moved back into the drag-stroke path, a mid-drag switch away from
    Draw could leave an EditHistory stroke snapshot open forever.

    Drives a real mouse press to open the Draw stroke (rather than
    calling ViewerWindow.on_edit_stroke_start()/on_edit_stroke_tile()
    directly) so map_view._stroke_active is genuinely True when the tool
    switches -- that's what makes MapView.set_tool()'s dangling-stroke-close
    branch actually run, the real mechanism this test exists to pin. Kept
    even though the CLICK_TOOLS shape makes it pass trivially today, so a
    future refactor regressing that shape gets caught here.

    Asserts the committed record's label: found while writing this test that
    _on_tool_selected() set self._current_tool to the NEW tool before calling
    map_view.set_tool() (which can synchronously close the OLD tool's
    dangling stroke via _end_stroke()/on_edit_stroke_end(), which reads
    self._current_tool for the label) -- so the closed Draw stroke used to
    commit labelled "Fill terrain", not "Paint terrain". Confirmed
    pre-existing and not specific to Paint Can (Draw -> Elevate mid-drag hit
    the same mislabel); fixed by reordering _on_tool_selected() to call
    set_tool() first. The diff content itself was never affected either way
    (commit_stroke() diffs against begin_stroke()'s snapshot regardless of
    label), which is what this test's earlier revision actually needed to
    hold before the label fix landed."""
    from PyQt5.QtCore import QEvent, Qt

    window = _edit_window("draw")
    try:
        _shown_flat(window)
        map_view = window.map_view
        mm = window.scenario.map_manager
        pos = conftest.viewport_pos(map_view, 0, 0)
        map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        assert map_view._stroke_active is True
        assert mm.terrain[0].terrain_id == _FILL_TERRAIN  # the Draw stroke actually painted (0, 0)

        window._on_tool_selected("fill")  # set_tool() closes the outgoing Draw stroke
        assert map_view._stroke_active is False
        assert len(window.edit_history.records) == 1
        assert window.edit_history.can_undo
        assert window.edit_history.records[0].label == "Paint terrain"  # the OUTGOING tool, not "Fill terrain"

        window.on_fill(1, 1, 0)  # must not raise RuntimeError -- fills the rest of the map
        assert len(window.edit_history.records) == 2
        assert mm.terrain[2 * mm.map_width + 2].terrain_id == _FILL_TERRAIN

        window._on_tool_selected("draw")
        window.terrain_panel.set_terrain(_OTHER_TERRAIN)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(2, 2, 0)
        window.on_edit_stroke_end()
        assert len(window.edit_history.records) == 3
        assert mm.terrain[2 * mm.map_width + 2].terrain_id == _OTHER_TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_paste_no_longer_depends_on_active_tool() -> None:
    """Phase 2.8 retired the tool-scoped clipboard kind gating this test used
    to cover: Copy/Paste Region are enabled purely off self._region/
    self._region_clipboard now, regardless of which tool (Paint Can here) is
    active -- see _update_tool_enabled()'s own comment."""
    window = _edit_window("fill")
    try:
        window.on_region_selected((0, 0, 2, 2))
        assert window.copy_action.isEnabled()

        window.copy_region()
        assert window._region_clipboard is not None
        assert len(window._region_clipboard.terrain_ids) == 4
        assert window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_region_does_not_touch_the_terrain_picker() -> None:
    """The old tool-scoped copy_tile()'s terrain branch doubled as an
    eyedropper, loading the picked terrain into the terrain picker -- retired in
    favour of the real Eyedropper tool (pick_tile_value()), which covers the
    same case for Draw/Fill AND Elevate/Set Elevation. Copy Region must not
    revive that side effect."""
    window = _edit_window("draw")
    try:
        mm = window.scenario.map_manager
        mm.get_tile(0, 0).terrain_id = _OTHER_TERRAIN
        assert window.terrain_panel.terrain_id() == _FILL_TERRAIN  # sanity: still the starting selection

        window.on_region_selected((0, 0, 1, 1))
        window.copy_region()
        assert window.terrain_panel.terrain_id() == _FILL_TERRAIN  # unchanged
        assert window._region_clipboard.terrain_ids == (_OTHER_TERRAIN,)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_stepped_full_rerender_fallback_actually_fires(monkeypatch) -> None:
    """_apply_dirty()'s STEPPED_FULL_RERENDER_THRESHOLD branch (measured
    with tools/bench_fill_latency.py -- see that constant's own comment) is
    otherwise untested: the real threshold (20,000) is well above the blank
    120x120 template's full-map worst case (14,400 tiles), so no other test
    in this module ever exercises it. Lowers the threshold instead of
    filling a 480x480 map here, to keep this test in the default tier.

    Also confirms no wait cursor leaks (on_fill() pushes its own; the branch
    pushed a nested one via _render_current() until TASK-031.61 A5 made it
    an in-place refresh): a stuck wait cursor would be a real, visible UI
    bug that nothing else here would catch."""
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 100)

    window = _edit_window("fill")
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.on_fill(0, 0, 0)  # fills all 14,400 tiles -- well past the patched threshold

        assert "refreshed the full map in place instead of patching" in window.status_log.toPlainText()
        assert QApplication.overrideCursor() is None
    finally:
        window.edit_history.mark_saved()
        window.close()


# -- top-down Flat: block rects, or an eviction once the patch outgrows the view (TASK-324) --

# A closed _OTHER_TERRAIN outline, inclusive tile corners. Each side sits one tile outside a mip-0
# chunk edge, so the outline's own tiles share no chunk with the tiles a fill inside it dirties.
_RING = (23, 23, 56, 48)
_INSIDE = (30, 30)
_INSIDE_TILES = 32 * 24
_GRASS_3 = 9  # outranks _OTHER_TERRAIN in blend_install's table: a fill inside re-blends the outline's tiles
_EVICTED = "evicted the cached region instead of patching"


def _blended_flat_window(levels):
    """A shown top-down Flat window at 1:1, so mip 0 is the visible level,
    with the _RING outline drawn, Paint Can holding _GRASS_3 and the whole
    canvas resident at each of `levels`. Returns (window, bbox): the
    outline's rect in canvas pixels, the fill's region grown by its blend
    ring. Caller must conftest.close_window()."""
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QApplication

    from descape import render

    window = conftest.shown_terrain_window()
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    # Unchecked before the style switch: see _shown_flat().
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    x0, y0, x1, y1 = _RING
    outline = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1) if x in (x0, x1) or y in (y0, y1)]
    window._on_tool_selected("draw_rect")
    assert window.terrain_panel.set_terrain(_OTHER_TERRAIN)
    window.on_shape_commit(outline)
    window._on_tool_selected("fill")
    assert window.terrain_panel.set_terrain(_GRASS_3)
    view = window.map_view
    view.setTransform(QTransform())
    view.center_on_tile((x0 + x1) // 2, (y0 + y1) // 2)
    QApplication.processEvents()
    cache = window._cache
    assert render.blend_effective(cache.layers), "blending is off, so the outline would prove nothing"
    assert view.viewport_chunk_target()[0] == 0, "mip 0 is not the visible level"
    cache.max_bytes *= 4  # room for every level at once, and for whatever a warm tick adds
    for mip in levels:
        cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
    t = cache.tile_px
    return window, (x0 * t, y0 * t, (x1 + 1) * t, (y1 + 1) * t)


def _chunk_keys_under(cache, bbox) -> set:
    """Every cached chunk key, at any level, whose chunk the REFERENCE bbox reaches."""
    keys = set()
    for mip in cache.resident_levels():
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, *cache._bbox_to_level(mip, bbox))
        keys.update(k for k in cache._cache if k[0] == mip and cx0 <= k[1] <= cx1 and cy0 <= k[2] <= cy1)
    return keys


def _levels(cache, levels) -> dict:
    return {mip: cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip).copy() for mip in levels}


def _fresh_levels(window, levels) -> dict:
    from descape.render_cache import FlatChunkCache

    cache = window._cache
    fresh = FlatChunkCache(
        window.scenario, cache.tile_px, sprites=cache.sprites_enabled, unit_filter=cache.unit_filter, layers=cache.layers
    )
    return _levels(fresh, levels)


def _count_patch_rects(cache, monkeypatch) -> list:
    """Every rect list patch_rects() is handed from here on, through a fake that delegates."""
    handed = []
    real = cache.patch_rects

    def patch_rects(rects):
        handed.append(list(rects))
        return real(rects)

    monkeypatch.setattr(cache, "patch_rects", patch_rects)
    return handed


@pytest.mark.parametrize("levels", [(0,), (0, -1)], ids=["visible-level", "and-a-coarser-one"])
@pytest.mark.parametrize("rule", ["count", "ratio"])
def test_a_flat_fill_too_large_to_patch_evicts_its_region(rule, levels, blend_install, monkeypatch) -> None:  # noqa: F811
    """A fill bounded by a second terrain, past the dirty-count threshold
    (`count`, lowered to 100) or under it but past the visible-tiles ratio
    (`ratio`): nothing is patched, every chunk under the region and its
    blend ring is dropped at every resident level, every other chunk stays
    the same object, and the canvas read back equals a fresh cache. The
    same again for its undo."""
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module

    def no_dilation(*args, **kwargs):
        raise AssertionError("the count pre-check dilated the dirty set")

    window, bbox = _blended_flat_window(levels)
    try:
        cache = window._cache
        if rule == "count":
            # Only the dirty count may evict here, before any rect is built.
            monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 100)
            monkeypatch.setattr(viewer_module, "_FLAT_EVICT_RATIO", float("inf"))
            monkeypatch.setattr(viewer_module, "flat_dirty_rects", no_dilation)
        else:
            monkeypatch.setattr(viewer_module, "_FLAT_EVICT_RATIO", 0.01)
        handed = _count_patch_rects(cache, monkeypatch)
        original = _levels(cache, levels)
        line = f"Flat view: {_INSIDE_TILES} tiles dirty, {_EVICTED}"

        for n, (step, act) in enumerate((("fill", lambda: window.on_fill(*_INSIDE, 0)), ("undo", window.undo)), start=1):
            before = _levels(cache, levels)  # every chunk of every level resident again
            under = _chunk_keys_under(cache, bbox)
            assert {k[0] for k in under} >= set(levels) and len(under) >= 30, f"{step}: too few chunks under the bbox"
            outside = {k: id(chunk) for k, chunk in cache._cache.items() if k not in under}
            assert outside, f"{step}: no chunk outside the bbox (vacuous)"

            act()

            assert not handed, f"{step}: patch_rects() ran"
            assert not under & cache._cache.keys(), f"{step}: chunks under the bbox are still cached"
            assert all(k in cache._cache and id(cache._cache[k]) == v for k, v in outside.items()), (
                f"{step}: a chunk outside the bbox was dropped or replaced"
            )
            assert window.status_log.toPlainText().count(line) == n, f"{step}: no status line"
            assert QApplication.overrideCursor() is None
            after = _levels(cache, levels)
            fresh = _fresh_levels(window, levels)
            for mip in levels:
                assert np.array_equal(after[mip], fresh[mip]), f"{step}: mip {mip} differs from a fresh render"
                assert not np.array_equal(after[mip], before[mip]), f"{step}: mip {mip} did not change (vacuous)"
        for mip in levels:
            assert np.array_equal(after[mip], original[mip]), f"mip {mip}: the undo did not restore the canvas"
    finally:
        conftest.close_window(window)


def test_a_small_flat_edit_patches_block_rects_and_evicts_nothing(blend_install, monkeypatch) -> None:  # noqa: F811
    """At the default ratio and threshold a few contiguous tiles are patched
    in place, one rect a step, and no chunk is dropped."""
    window, _bbox = _blended_flat_window((0,))
    try:
        cache = window._cache
        handed = _count_patch_rects(cache, monkeypatch)
        before = _levels(cache, (0,))
        warm = {k: id(chunk) for k, chunk in cache._cache.items()}

        window._on_tool_selected("draw")
        window.on_edit_stroke_start()
        for tile in ((36, 36), (37, 36), (36, 37), (37, 37)):  # each ring stays inside block (2, 2)
            window.on_edit_stroke_tile(*tile, 0)
        window.on_edit_stroke_end()

        assert len(handed) == 4 and all(len(rects) == 1 for rects in handed), handed
        assert all(k in cache._cache and id(cache._cache[k]) == v for k, v in warm.items()), "a small edit evicted a chunk"
        assert _EVICTED not in window.status_log.toPlainText()
        after = _levels(cache, (0,))
        assert not np.array_equal(after[0], before[0]), "the edit changed no pixel (vacuous)"
        assert np.array_equal(after[0], _fresh_levels(window, (0,))[0]), "the patched canvas differs from a fresh render"
    finally:
        conftest.close_window(window)


def test_the_flat_evict_rule_weighs_covered_tiles_per_level_against_visible_tiles(monkeypatch) -> None:
    """_flat_patch_exceeds_viewport() at its boundary: rect tiles times
    resident levels against _FLAT_EVICT_RATIO times the tiles in the visible
    chunks (a ragged edge chunk counted at its real size), strictly greater,
    and never with no viewport target."""
    import descape.viewer as viewer_module

    window = _edit_window("draw")
    try:
        window.iso_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Flat")
        cache = window._cache
        t = cache.tile_px
        assert cache.chunk_px == 8 * t and cache.mip_tile_px(-1) * 2 == t
        exceeds = window._flat_patch_exceeds_viewport
        target = [(0, 0, 0, 1, 1)]  # 2x2 chunks of 8x8 tiles: 256 visible
        monkeypatch.setattr(window.map_view, "viewport_chunk_target", lambda: target[0])

        assert not exceeds([(0, 0, 16 * t, 16 * t)], t), "an empty cache has nothing to patch, so nothing to evict"
        cache.get_chunk(0, 0, 0)
        assert cache.resident_levels() == [0]
        assert not exceeds([(0, 0, 16 * t, 16 * t)], t), "256 tiles on one level is not past 256 visible"
        assert exceeds([(0, 0, 16 * t, 16 * t), (16 * t, 0, 17 * t, t)], t)
        cache.get_chunk(-1, 0, 0)
        assert cache.resident_levels() == [-1, 0]
        assert not exceeds([(0, 0, 16 * t, 8 * t)], t), "128 tiles on two levels is not past 256 visible"
        assert exceeds([(0, 0, 16 * t, 8 * t), (16 * t, 0, 17 * t, t)], t)

        # Mip -1 holds 16 tiles a chunk; the 120-tile map's last chunk is 8 wide and high: 64 visible.
        target[0] = (-1, 7, 7, 7, 7)
        assert not exceeds([(0, 0, 8 * t, 4 * t)], t)
        assert exceeds([(0, 0, 8 * t, 4 * t), (16 * t, 0, 17 * t, t)], t)

        monkeypatch.setattr(viewer_module, "_FLAT_EVICT_RATIO", float("inf"))
        assert not exceeds([(0, 0, 120 * t, 120 * t)], t)
        monkeypatch.setattr(viewer_module, "_FLAT_EVICT_RATIO", 0.0)
        assert exceeds([(0, 0, t, t)], t)
        target[0] = None
        assert not exceeds([(0, 0, 120 * t, 120 * t)], t), "no viewport target, yet it priced an eviction"
    finally:
        window.edit_history.mark_saved()
        window.close()
