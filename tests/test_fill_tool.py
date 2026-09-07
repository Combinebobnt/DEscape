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

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_PATH

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
    and selects _FILL_TERRAIN on terrain_combo. Caller must
    edit_history.mark_saved() + close()."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected(tool)
    window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_FILL_TERRAIN))
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


def _mouse_event(kind, pos, button, buttons):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)


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
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, press_pos, Qt.LeftButton, Qt.LeftButton))
        assert len(window.edit_history.records) == 1
        assert map_view._stroke_active is False

        for tx, ty in [(1, 0), (2, 0), (0, 1)]:
            move_pos = conftest.viewport_pos(map_view, tx, ty)
            map_view.mouseMoveEvent(_mouse_event(QEvent.MouseMove, move_pos, Qt.NoButton, Qt.LeftButton))
            assert len(window.edit_history.records) == 1

        map_view.mouseReleaseEvent(
            _mouse_event(QEvent.MouseButtonRelease, press_pos, Qt.LeftButton, Qt.NoButton)
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

        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, pos, Qt.RightButton, Qt.RightButton))

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
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
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
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(_OTHER_TERRAIN))
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(2, 2, 0)
        window.on_edit_stroke_end()
        assert len(window.edit_history.records) == 3
        assert mm.terrain[2 * mm.map_width + 2].terrain_id == _OTHER_TERRAIN
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_paste_gating_treats_fill_as_terrain_kind() -> None:
    window = _edit_window("fill")
    try:
        window.on_hover((0, 0))
        assert window.copy_action.isEnabled()

        window.copy_tile()
        assert window._clipboard["kind"] == "terrain"
        assert window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_terrain_selects_it_for_drawing() -> None:
    """copy_tile()'s terrain branch must load the picked terrain into
    terrain_combo, not just the clipboard -- Draw/Fill both paint from
    terrain_combo.currentData() (on_edit_stroke_tile/on_fill), never from
    the clipboard, so a Copy that only fills the clipboard leaves a
    subsequent drag-painted stroke using whatever terrain_combo already
    showed. _edit_window() starts terrain_combo on _FILL_TERRAIN; this sets
    the hovered tile to the other value directly (no stroke needed) so
    Copy's effect on the combo is observable."""
    window = _edit_window("draw")
    try:
        mm = window.scenario.map_manager
        mm.get_tile(0, 0).terrain_id = _OTHER_TERRAIN
        assert window.terrain_combo.currentData() == _FILL_TERRAIN  # sanity: still the starting selection

        window.on_hover((0, 0))
        window.copy_tile()
        assert window.terrain_combo.currentData() == _OTHER_TERRAIN
        assert window._clipboard == {"kind": "terrain", "terrain_id": _OTHER_TERRAIN, "layer": -1}
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

    Also confirms the wait cursor _apply_dirty's fallback pushes via
    _render_current() (nested inside on_fill()'s own identical wrap --
    Qt's override-cursor is a stack, so this should unwind cleanly) doesn't
    leak: a stuck wait cursor would be a real, visible UI bug that nothing
    else here would catch."""
    from PyQt5.QtWidgets import QApplication

    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "STEPPED_FULL_RERENDER_THRESHOLD", 100)

    window = _edit_window("fill")
    try:
        window.terrain_style_combo.setCurrentText("Stepped")
        window.on_fill(0, 0, 0)  # fills all 14,400 tiles -- well past the patched threshold

        assert "re-rendered full map instead of patching" in window.status_log.toPlainText()
        assert QApplication.overrideCursor() is None
    finally:
        window.edit_history.mark_saved()
        window.close()
