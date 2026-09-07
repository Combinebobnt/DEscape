"""Ruler tool coverage that needs a real ViewerWindow: the toolbar gating, the
mouse routing in MapView, the overlay's lifecycle, and the two properties the
Qt-free tests/test_ruler.py cannot reach at all (that a measurement never
enters EditHistory, and that its label survives Flat+Isometric's transform).

Same offscreen technique tests/test_fill_tool.py documents: QT_QPA_PLATFORM=
offscreen, one shared QApplication via conftest.ensure_qapp(), and every
window edit_history.mark_saved()'d before close() so closeEvent's discard
prompt can't block forever offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import ruler
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# The label's fill, straight off MapView.RULER_LABEL_COLOR. Distinct from the
# line/outline pen so an ink measurement can isolate the text.
_LABEL_RGB = (255, 190, 110)


def _ruler_window(style: str = "Flat"):
    """A shown window with the blank template loaded and the Ruler active.
    Caller must edit_history.mark_saved() + close()."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window.scenario is not None, "blank template failed to load"
    window.terrain_style_combo.setCurrentText(style)
    window._on_tool_selected("ruler")
    # A real show()/processEvents() cycle, so map_view's transform is the one
    # fitInView actually computed and mapFromScene() below can be trusted.
    window.show()
    QApplication.processEvents()
    return window


def _viewport_pos(map_view, tile_x: int, tile_y: int):
    """Where to click to hit a given tile, in every Elevation View.

    Goes through _tile_polygon rather than Flat's (tile + 0.5) * tile_px,
    which is only correct for Flat and lands on the wrong tile entirely in
    Stepped and Sloped. Not circular with the code under test: Stepped
    resolves the resulting press through screen_to_tile and Sloped through its
    own pick plane, both independent inverses, so a click placed here and a
    tile asserted below is a real round trip."""
    from PyQt5.QtCore import QPointF

    polygon = map_view._tile_polygon(tile_x, tile_y)
    assert polygon is not None, f"no footprint for ({tile_x}, {tile_y})"
    return QPointF(map_view.mapFromScene(polygon.boundingRect().center()))


def _mouse_event(kind, pos, button, buttons):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QMouseEvent

    return QMouseEvent(kind, pos, button, buttons, Qt.NoModifier)


def _drag(map_view, a: tuple[int, int], b: tuple[int, int]) -> None:
    """A full press, move, release with the left button, through MapView's own
    handlers rather than the session, so the routing is what is under test."""
    from PyQt5.QtCore import QEvent, Qt

    map_view.mousePressEvent(
        _mouse_event(QEvent.MouseButtonPress, _viewport_pos(map_view, *a), Qt.LeftButton, Qt.LeftButton)
    )
    map_view.mouseMoveEvent(
        _mouse_event(QEvent.MouseMove, _viewport_pos(map_view, *b), Qt.NoButton, Qt.LeftButton)
    )
    map_view.mouseReleaseEvent(
        _mouse_event(QEvent.MouseButtonRelease, _viewport_pos(map_view, *b), Qt.LeftButton, Qt.NoButton)
    )


def _label_ink_bbox(map_view) -> tuple[int, int]:
    """Width and height of the label's drawn ink, in device pixels, measured
    off a real render rather than read back off the item. Reading
    boundingRect() would pass with ItemIgnoresTransformations removed, because
    that rect is the item's own untransformed size either way."""
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    pixmap = map_view.viewport().grab()
    array = qimage_rgb888_to_array(pixmap.toImage().convertToFormat(QImage.Format_RGB888))
    hit = (np.abs(array.astype(np.int16) - np.array(_LABEL_RGB, dtype=np.int16)) <= 24).all(axis=2)
    assert hit.any(), "no label ink found in the capture"
    ys, xs = np.nonzero(hit)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def test_ruler_registered_as_keybind_and_toolbar_action() -> None:
    from PyQt5.QtGui import QKeySequence

    from descape import settings
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    assert ("tool_ruler", "Ruler Tool", "R") in settings.REBINDABLE_ACTIONS

    window = ViewerWindow()
    try:
        assert window._keybind_actions["tool_ruler"] is window.ruler_action
        assert window.ruler_action.shortcut() == QKeySequence("R")
        assert window.elevation_action.shortcut() == QKeySequence("E")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ruler_is_enabled_in_every_mode_and_style() -> None:
    """"Enabled in all modes, unlike the mode-specific edit tools" is the
    backlog item's own wording, and it means gated on has_map alone the way
    Pan is. Sloped force-modes out of Units, so that pair is skipped."""
    window = _ruler_window()
    try:
        assert window.ruler_action.isEnabled()
        for style in ("Flat", "Stepped", "Sloped"):
            window.terrain_style_combo.setCurrentText(style)
            for mode in ("View", "Terrain", "Units", "Triggers", "Map Options"):
                if mode == "Units" and style == "Sloped":
                    continue
                window.mode_combo.setCurrentText(mode)
                assert window.ruler_action.isEnabled(), f"{mode}/{style}"
                assert window._current_tool == "ruler", f"forced off the Ruler in {mode}/{style}"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ruler_is_disabled_with_no_map() -> None:
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert not window.ruler_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_left_drag_with_ruler_active_does_not_pan() -> None:
    """Without the Ruler named in _apply_drag_mode, a left drag pans the view
    and no measurement is ever made. The dragMode assertion is what actually
    catches that here, and it has to be asserted directly: these events are
    delivered straight to the handler, whose ruler branch returns before
    super().mousePressEvent, so ScrollHandDrag never gets the chance to move
    the scrollbars that a real user's drag would. Verified by reverting the
    fix: this is the only test in the file that fails."""
    from PyQt5.QtWidgets import QGraphicsView

    window = _ruler_window()
    try:
        map_view = window.map_view
        assert map_view.dragMode() == QGraphicsView.NoDrag
        before = (map_view.horizontalScrollBar().value(), map_view.verticalScrollBar().value())
        _drag(map_view, (10, 10), (30, 22))
        after = (map_view.horizontalScrollBar().value(), map_view.verticalScrollBar().value())
        assert after == before, "the view scrolled, so the drag was routed to a pan"
        assert map_view._ruler.endpoints == ((10, 10), (30, 22))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_ruler_press_in_units_mode_does_not_select() -> None:
    """Pins the branch order in mousePressEvent. Asserting only that a
    measurement exists passes even if the Units branch ALSO ran, so the empty
    selection is the half that matters."""
    window = _ruler_window()
    try:
        window.mode_combo.setCurrentText("Units")
        map_view = window.map_view
        _drag(map_view, (12, 12), (20, 20))
        assert map_view._ruler.endpoints == ((12, 12), (20, 20))
        assert window._selection == []
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_completed_measurement_leaves_edit_history_untouched() -> None:
    """"Does not alter the map", pinned rather than assumed. EditHistory has
    no read-only mode to opt into: a tool that mutates nothing simply never
    calls begin_stroke."""
    window = _ruler_window()
    try:
        assert not window.edit_history.records
        _drag(window.map_view, (5, 5), (25, 15))
        assert not window.edit_history.records
        assert not window.edit_history.can_undo
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_label_is_unsheared_under_flat_isometric() -> None:
    """Flat+Isometric applies scale(1, 0.5) then rotate(-45) to the whole
    scene, so scene-space text would come out rotated, squashed, and at a
    different ink size. ItemIgnoresTransformations is what keeps the two
    captures identical."""
    from PyQt5.QtWidgets import QApplication

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (40, 30))
        map_view.set_isometric(False)
        QApplication.processEvents()
        upright = _label_ink_bbox(map_view)
        map_view.set_isometric(True)
        QApplication.processEvents()
        rotated = _label_ink_bbox(map_view)
        # Within a couple pixels, not exactly equal: the label's scene-space
        # anchor lands on a different subpixel phase in the two transforms,
        # so antialiasing alone moves the ink bbox slightly, a bit more at
        # RULER_LABEL_FONT_PX's larger sizes (measured 2px at 18). That
        # tolerance is nowhere near enough to admit a sheared label, which
        # rotate(-45) plus scale(1, 0.5) would leave a fraction of this width.
        assert abs(upright[0] - rotated[0]) <= 2, f"width changed: {upright} vs {rotated}"
        assert abs(upright[1] - rotated[1]) <= 2, f"height changed: {upright} vs {rotated}"
        # A horizontal strip of text in both, never a diagonal smear.
        assert upright[0] > 6 * upright[1]
        assert rotated[0] > 6 * rotated[1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_press_out_in_the_overscroll_void_starts_nothing() -> None:
    """The reason the press guards on _pos_on_map rather than "_pick_tile is
    not None": Flat's integer division answers for every pixel, including the
    40% void, so the None check alone would let an off-map tile through."""
    from PyQt5.QtCore import QEvent, QPointF, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        void = QPointF(map_view.mapFromScene(QPointF(-500.0, -500.0)))
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, void, Qt.LeftButton, Qt.LeftButton))
        assert map_view._ruler.endpoints is None
        assert map_view._ruler_line_item is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("style", ["Flat", "Stepped", "Sloped"])
def test_the_overlay_is_built_in_every_elevation_view(style: str) -> None:
    window = _ruler_window(style)
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        assert map_view._ruler_line_item is not None
        assert len(map_view._ruler_end_items) == 2
        assert map_view._ruler_label_item.text() == "24.4 tiles  (dx +20, dy +14)"
        assert map_view._ruler_line_item.line().length() > 0
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_endpoints_follow_real_elevation_while_the_distance_stays_flat() -> None:
    """The deliberate split in descape.ruler's docstring, pinned so a later
    pass can't quietly reconcile the two. Raising a tile moves its marker up
    the screen without changing the measured distance at all."""
    window = _ruler_window("Stepped")
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        distance_before = map_view._ruler.measurement.distance
        anchor_before = map_view._ruler_anchor((30, 24))

        raised = map_view._iso_elevations.copy()
        raised[24, 30] = raised[24, 30] + 4
        map_view._iso_elevations = raised
        anchor_after = map_view._ruler_anchor((30, 24))

        assert anchor_after.y() < anchor_before.y(), "marker ignored the tile's elevation"
        assert map_view._ruler.measurement.distance == distance_before
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.parametrize("clear_by", ["escape", "right_click", "tool_change", "mode_change"])
def test_the_measurement_is_cleared_by_every_documented_gesture(clear_by: str) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        assert map_view._ruler_line_item is not None

        if clear_by == "escape":
            map_view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        elif clear_by == "right_click":
            pos = _viewport_pos(map_view, 15, 15)
            map_view.mousePressEvent(
                _mouse_event(QEvent.MouseButtonPress, pos, Qt.RightButton, Qt.RightButton)
            )
        elif clear_by == "tool_change":
            window._on_tool_selected("pan")
        else:
            window.mode_combo.setCurrentText("Terrain")

        assert map_view._ruler.endpoints is None
        assert map_view._ruler_line_item is None
        assert map_view._ruler_label_item is None
        assert window.ruler_status_label.text() == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_on_map_label_anchors_to_the_live_endpoint_not_the_midpoint() -> None:
    """The fix for a small-window complaint: a long measurement's midpoint
    can sit off screen with no readout visible at all, where the point the
    mouse is on (or last set the measurement to) always was on screen a
    moment ago."""
    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        b_anchor = map_view._ruler_anchor((30, 24))
        assert map_view._ruler_label_item.pos() == b_anchor
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_status_bar_shows_the_live_measurement_mid_drag() -> None:
    """The status bar readout is what stays visible when the on-map label
    scrolls off screen, so it has to track a PENDING drag, not just a
    completed measurement -- unlike on_ruler_measured, which only fires once
    on completion (see _report_ruler)."""
    from PyQt5.QtCore import QEvent, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        map_view.mousePressEvent(
            _mouse_event(QEvent.MouseButtonPress, _viewport_pos(map_view, 10, 10), Qt.LeftButton, Qt.LeftButton)
        )
        map_view.mouseMoveEvent(
            _mouse_event(QEvent.MouseMove, _viewport_pos(map_view, 30, 24), Qt.NoButton, Qt.LeftButton)
        )
        assert map_view._ruler.state == ruler.STATE_PENDING
        assert ruler.format_measurement(map_view._ruler.measurement) in window.ruler_status_label.text()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_close_then_resize_does_not_touch_deleted_items() -> None:
    """scene().clear() destroys the C++ objects, so a surviving Python
    reference would dangle. Mirrors the same hazard the edge-tick item's null
    in clear_image() exists for."""
    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        window.close_scenario()
        assert map_view._ruler_line_item is None
        assert map_view._ruler.endpoints is None
        assert window.ruler_status_label.text() == ""
        window.resize(820, 620)
        map_view.set_isometric(True)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_opening_a_new_map_over_an_active_measurement_clears_the_status_bar() -> None:
    """set_source()'s own _ruler.clear()/_forget_ruler_items(), a separate
    code path from _clear_ruler() since scene().clear() already destroyed
    the C++ items -- covered here rather than assumed identical to the
    clear_image() path above."""
    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        assert window.ruler_status_label.text() != ""
        window.load_scenario(BLANK_TEMPLATE_PATH, untitled=True)
        assert window.ruler_status_label.text() == ""
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_copy_and_paste_stay_disabled_while_the_ruler_is_active() -> None:
    """Falls out of _update_tool_enabled's current_tool_action dict having no
    entry for a non-edit tool, exactly as it already does for Pan. Asserted
    rather than assumed, since nothing else would notice if that dict grew a
    ruler row by accident."""
    window = _ruler_window()
    try:
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("ruler")
        assert not window.copy_action.isEnabled()
        assert not window.paste_action.isEnabled()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_cross_cursor_wins_over_units_modes_pointing_hand() -> None:
    from PyQt5.QtCore import Qt

    window = _ruler_window()
    try:
        window.mode_combo.setCurrentText("Units")
        assert window.map_view.cursor().shape() == Qt.CrossCursor
    finally:
        window.edit_history.mark_saved()
        window.close()


def _status_lines(window) -> list[str]:
    return window.status_log.toPlainText().splitlines()


def test_a_completed_measurement_is_logged_once() -> None:
    """The log is the measurement's only durable record: the on-map label
    always shows the latest one, so comparing two readings needs this."""
    window = _ruler_window()
    try:
        _drag(window.map_view, (10, 10), (30, 24))
        ruler_lines = [line for line in _status_lines(window) if line.startswith("Ruler:")]
        assert ruler_lines == ["Ruler: 24.4 tiles  (dx +20, dy +14) from (10, 10) to (30, 24)"]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_pending_measurement_is_not_logged() -> None:
    """A press and release inside one tile leaves the session pending, which
    is a half-made measurement with no result to report yet."""
    from PyQt5.QtCore import QEvent, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        pos = _viewport_pos(map_view, 10, 10)
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
        map_view.mouseReleaseEvent(_mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))
        assert not [line for line in _status_lines(window) if line.startswith("Ruler:")]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_dragging_logs_once_not_once_per_frame() -> None:
    """_report_ruler fires on the edge into DONE, not on every state read. A
    handler called from mouseMoveEvent instead would pass every other test in
    this file and flood the log."""
    from PyQt5.QtCore import QEvent, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        map_view.mousePressEvent(
            _mouse_event(QEvent.MouseButtonPress, _viewport_pos(map_view, 5, 5), Qt.LeftButton, Qt.LeftButton)
        )
        for tile_x in range(6, 20):
            map_view.mouseMoveEvent(
                _mouse_event(QEvent.MouseMove, _viewport_pos(map_view, tile_x, 5), Qt.NoButton, Qt.LeftButton)
            )
        map_view.mouseReleaseEvent(
            _mouse_event(
                QEvent.MouseButtonRelease, _viewport_pos(map_view, 19, 5), Qt.LeftButton, Qt.NoButton
            )
        )
        assert len([line for line in _status_lines(window) if line.startswith("Ruler:")]) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_starting_a_new_measurement_does_not_relog_the_old_one() -> None:
    """press() from DONE goes back to PENDING, an edge _report_ruler must not
    treat as a completion."""
    from PyQt5.QtCore import QEvent, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        map_view.mousePressEvent(
            _mouse_event(
                QEvent.MouseButtonPress, _viewport_pos(map_view, 2, 2), Qt.LeftButton, Qt.LeftButton
            )
        )
        assert len([line for line in _status_lines(window) if line.startswith("Ruler:")]) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_void_click_after_a_measurement_does_not_relog_it() -> None:
    """The case _report_ruler's "previous was not already DONE" half exists
    for, and the only one that reaches it: an off-map press is ignored so the
    session stays DONE, and then the release arrives with the state unchanged.
    Without that half the finished measurement logs again on every stray click
    in the void."""
    from PyQt5.QtCore import QEvent, QPointF, Qt

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (30, 24))
        void = QPointF(map_view.mapFromScene(QPointF(-500.0, -500.0)))
        map_view.mousePressEvent(_mouse_event(QEvent.MouseButtonPress, void, Qt.LeftButton, Qt.LeftButton))
        map_view.mouseReleaseEvent(
            _mouse_event(QEvent.MouseButtonRelease, void, Qt.LeftButton, Qt.NoButton)
        )
        assert len([line for line in _status_lines(window) if line.startswith("Ruler:")]) == 1
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_keybind_migration_reaches_the_real_actions(tmp_path) -> None:
    """The Qt-free cases in tests/test_settings.py stop at get_keybind. This
    one runs the whole chain a first launch does: a pre-Ruler config.yaml ->
    _load_keybinds -> _build_keybind_actions -> apply_keybind -> the QActions
    themselves. Without it the migration's only coverage would be of the layer
    below the one users actually feel."""
    from PyQt5.QtGui import QKeySequence

    from descape.viewer import ViewerWindow

    # The autouse _isolated_settings fixture has already pointed
    # settings.CONFIG_PATH at this same tmp_path and nulled _keybinds, so
    # writing here is what the app will read.
    (tmp_path / "config.yaml").write_text("keybinds:\n  tool_elevation: R\n  tool_pan: M\n")

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        assert window.elevation_action.shortcut() == QKeySequence("E")
        assert window.ruler_action.shortcut() == QKeySequence("R")
        assert window.pan_action.shortcut() == QKeySequence("M")
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_hard_scrolling_leaves_no_stale_label_fragments() -> None:
    """The one risk with no in-repo precedent: an ItemIgnoresTransformations
    item sits awkwardly in the
    scene's BSP index, and the documented failure mode is paint artifacts left
    behind on scroll.

    Measured the way the distance ticks closed their own boundingRect symptom.
    Stale copies would leave label-coloured ink scattered across the viewport,
    and _label_ink_bbox spans EVERY matching pixel, so a single leftover copy
    a hundred pixels away would roughly double the reported width.

    Both captures re-centre on the label first. Without that the test passes
    vacuously: after six hard scrolls the label sits almost entirely outside
    the viewport, and a 5-pixel sliver trivially satisfies any upper bound."""
    from PyQt5.QtWidgets import QApplication

    window = _ruler_window()
    try:
        map_view = window.map_view
        _drag(map_view, (10, 10), (40, 30))
        map_view.scale(3.0, 3.0)
        anchor = map_view._ruler_label_item.pos()
        map_view.centerOn(anchor)
        QApplication.processEvents()
        clean = _label_ink_bbox(map_view)
        # The whole label really is on screen, so the comparison below has
        # something to bite on.
        assert clean[0] > 100, f"label not fully visible to begin with: {clean}"

        bar = map_view.horizontalScrollBar()
        vbar = map_view.verticalScrollBar()
        for step in range(6):
            bar.setValue(bar.minimum() if step % 2 else bar.maximum())
            vbar.setValue(vbar.maximum() if step % 2 else vbar.minimum())
            QApplication.processEvents()
        map_view.centerOn(anchor)
        QApplication.processEvents()

        scrolled = _label_ink_bbox(map_view)
        assert scrolled[0] <= clean[0] + 2, f"label ink smeared horizontally: {clean} -> {scrolled}"
        assert scrolled[1] <= clean[1] + 2, f"label ink smeared vertically: {clean} -> {scrolled}"
    finally:
        window.edit_history.mark_saved()
        window.close()
