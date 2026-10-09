"""GH #179: MapView.viewportEvent coalesces mouse moves to at most one
handled move per paint, so a native Wayland session (whose Qt5 plugin
delivers every motion event) stops queueing stroke steps faster than they
drain.

Everything here goes through QApplication.sendEvent() on the viewport, the
only route that reaches viewportEvent; direct mouseMoveEvent() calls (the rest
of the suite) bypass the coalescer by design and stay synchronous.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import time

import pytest

from descape.brush import line_tiles

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TERRAIN = 15  # GRASS_1, distinct from the blank template's own terrain_id=0


def _window(style: str = "Stepped", tool: str = "draw"):
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    window._on_tool_selected(tool)
    window.terrain_panel.set_terrain(_TERRAIN)
    # Trees default on and would pop a large-edit confirm modal offscreen.
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    if style == "Flat":
        # Unchecked before the style switch -- see tests/test_brush_stroke.py's _shown_flat.
        window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText(style)
    window.resize(1200, 800)
    window.show()
    QApplication.processEvents()
    _pump(window.map_view)
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _idle(view) -> bool:
    return view._move_stash is None and not view._move_flush_timer.isActive() and not view._move_fallback_timer.isActive()


def _pump(view, timeout_s: float = 2.0) -> None:
    """Processes events until the coalescer has nothing pending."""
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    QApplication.processEvents()
    while not _idle(view):
        assert time.monotonic() < deadline, "the coalescer never went idle"
        time.sleep(0.005)
        QApplication.processEvents()


def _pump_for(ms: float) -> None:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + ms / 1000
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.002)
    QApplication.processEvents()


def _send(view, kind, pos, button=None, buttons=None) -> None:
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtWidgets import QApplication

    event = conftest.mouse_event(
        kind,
        QPointF(pos),
        Qt.NoButton if button is None else button,
        Qt.NoButton if buttons is None else buttons,
    )
    QApplication.sendEvent(view.viewport(), event)


def _tile_pos(view, tile):
    return conftest.polygon_viewport_pos(view, *tile)


def _record_deliveries(view) -> list[list[tuple[int, int]]]:
    """Swaps MapView's stroke callbacks for fakes, so only its routing runs."""
    delivered: list[list[tuple[int, int]]] = []
    view._on_stroke_start = lambda: None
    view._on_stroke_end = lambda: None
    view._on_stroke_tiles = lambda tiles, modifiers: delivered.append(list(tiles))
    return delivered


class _PaintCounter:
    """Counts the viewport's paint events, to prove a step painted nothing."""

    def __init__(self, view) -> None:
        from PyQt5.QtCore import QEvent, QObject

        counter = self

        class _Filter(QObject):
            def eventFilter(self, obj, event) -> bool:
                if event.type() == QEvent.Paint:
                    counter.count += 1
                return False

        self.count = 0
        self._filter = _Filter()
        view.viewport().installEventFilter(self._filter)


@pytest.mark.parametrize("style", ["Flat", "Stepped"])
def test_a_burst_of_moves_lands_as_one_gap_filled_step(style: str) -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _window(style)
    try:
        view = window.map_view
        delivered = _record_deliveries(view)
        a, b = (20, 20), (31, 26)
        _send(view, QEvent.MouseButtonPress, _tile_pos(view, a), Qt.LeftButton, Qt.LeftButton)
        assert delivered == [[a]]
        delivered.clear()
        path = line_tiles(*a, *b)  # excludes a, the press's own tile
        for tile in path:
            _send(view, QEvent.MouseMove, _tile_pos(view, tile), Qt.NoButton, Qt.LeftButton)
        assert delivered == [], "a stashed move must not be handled synchronously"
        _pump(view)
        assert len(delivered) == 1, f"{len(path)} moves ran {len(delivered)} steps"
        assert view._stroke_touched == {a, *path}
        _send(view, QEvent.MouseButtonRelease, _tile_pos(view, b), Qt.LeftButton, Qt.NoButton)
    finally:
        _close(window)


def test_a_release_sees_the_last_move_first() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _window()
    try:
        view = window.map_view
        delivered = _record_deliveries(view)
        a, b = (20, 20), (26, 20)
        _send(view, QEvent.MouseButtonPress, _tile_pos(view, a), Qt.LeftButton, Qt.LeftButton)
        _send(view, QEvent.MouseMove, _tile_pos(view, (23, 20)), Qt.NoButton, Qt.LeftButton)
        _send(view, QEvent.MouseMove, _tile_pos(view, b), Qt.NoButton, Qt.LeftButton)
        _send(view, QEvent.MouseButtonRelease, _tile_pos(view, b), Qt.LeftButton, Qt.NoButton)
        assert delivered == [[a], line_tiles(*a, *b)]
        assert view._move_stash is None
        _pump(view)
    finally:
        _close(window)


def test_a_pending_hover_is_handled_before_the_press() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = _window()
    try:
        view = window.map_view
        order: list[str] = []
        original_hover = view._on_hover
        view._on_hover = lambda tile: (order.append("hover"), original_hover(tile))
        view._on_stroke_start = lambda: order.append("press")
        view._on_stroke_tiles = lambda tiles, modifiers: None
        view._on_stroke_end = lambda: None
        _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
        assert order == []
        _send(view, QEvent.MouseButtonPress, _tile_pos(view, (22, 20)), Qt.LeftButton, Qt.LeftButton)
        assert order[:2] == ["hover", "press"], order
        _send(view, QEvent.MouseButtonRelease, _tile_pos(view, (22, 20)), Qt.LeftButton, Qt.NoButton)
        _pump(view)
    finally:
        _close(window)


def _off_map_pos(view):
    from PyQt5.QtCore import QPoint

    pos = QPoint(4, 4)
    assert view._pick_tile(view.mapToScene(pos)) is None, "the corner must be off-map for a no-paint hover"
    return pos


def test_the_gate_holds_a_move_until_a_paint_or_the_fallback() -> None:
    """Pan tool, so the edit highlight's pulse paints nothing. An off-map
    hover with no highlight up changes nothing on screen, so it paints
    nothing and leaves the gate shut."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    window = _window(tool="pan")
    try:
        view = window.map_view
        handled: list[object] = []
        original_hover = view._on_hover
        view._on_hover = lambda tile: (handled.append(tile), original_hover(tile))
        off_map = _off_map_pos(view)
        _send(view, QEvent.MouseMove, off_map)
        _pump(view)
        paints = _PaintCounter(view)
        # Long enough that a loaded machine can't fire it inside one processEvents().
        view._move_fallback_timer.setInterval(60_000)

        handled.clear()
        _send(view, QEvent.MouseMove, off_map)
        QApplication.processEvents()
        assert handled == [None], "an open gate hands the move over on the next turn"
        assert paints.count == 0, "the off-map hover painted, so the gate test below is vacuous"
        assert not view._move_gate_open

        _send(view, QEvent.MouseMove, off_map)
        QApplication.processEvents()
        QApplication.processEvents()
        assert handled == [None], "a second move went through with no paint in between"
        view.viewport().repaint()
        QApplication.processEvents()
        assert handled == [None, None], "a paint reopens the gate"

        view._move_fallback_timer.setInterval(view.MOVE_PAINT_WAIT_MS)
        _pump_for(view.MOVE_PAINT_WAIT_MS * 3)
        handled.clear()
        paints.count = 0
        _send(view, QEvent.MouseMove, off_map)
        QApplication.processEvents()
        _send(view, QEvent.MouseMove, off_map)
        _pump(view)
        assert paints.count == 0
        assert handled == [None, None], "with no paint, the fallback hands the move over"
    finally:
        _close(window)


def test_a_disabled_view_drops_moves() -> None:
    from PyQt5.QtCore import QEvent

    window = _window()
    try:
        view = window.map_view
        handled: list[object] = []
        view._on_hover = handled.append
        window.setEnabled(False)
        try:
            _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
            assert view._move_stash is None
            _pump_for(view.MOVE_PAINT_WAIT_MS * 3)
        finally:
            window.setEnabled(True)
        assert handled == []

        # Disabled after the stash: the flush drops it rather than handling it.
        _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
        window.setEnabled(False)
        try:
            _pump_for(view.MOVE_PAINT_WAIT_MS * 3)
        finally:
            window.setEnabled(True)
        assert handled == []
        assert view._move_stash is None
    finally:
        _close(window)


def test_a_key_press_handles_the_pending_move_first() -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    window = _window()
    try:
        view = window.map_view
        handled: list[object] = []
        view._on_hover = handled.append
        _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
        assert handled == []
        view.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Shift, Qt.ShiftModifier))
        assert handled == [(20, 20)]
        view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Shift, Qt.NoModifier))
        _pump(view)
    finally:
        _close(window)


def _wheel_event(view, pos):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    return QWheelEvent(
        QPointF(pos),
        QPointF(view.viewport().mapToGlobal(pos.toPoint())),
        QPoint(0, 0),
        QPoint(0, 0),  # no notch: the flush is under test, not the zoom
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


@pytest.mark.parametrize("kind", ["wheel", "double_click", "key_release", "shortcut_override", "public"])
def test_every_flush_first_path_handles_the_pending_move(kind: str) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication

    window = _window(tool="pan")
    try:
        view = window.map_view
        handled: list[object] = []
        view._on_hover = handled.append
        pos = _tile_pos(view, (20, 20))
        _send(view, QEvent.MouseMove, pos)
        assert handled == []
        if kind == "wheel":
            QApplication.sendEvent(view.viewport(), _wheel_event(view, pos))
        elif kind == "double_click":
            _send(view, QEvent.MouseButtonDblClick, pos, Qt.LeftButton, Qt.LeftButton)
        elif kind == "key_release":
            view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Shift, Qt.NoModifier))
        elif kind == "shortcut_override":
            QApplication.sendEvent(view, QKeyEvent(QEvent.ShortcutOverride, Qt.Key_V, Qt.ControlModifier))
        else:
            view.flush_pending_move()
        assert handled[:1] == [(20, 20)], handled
        assert view._move_stash is None
        if kind == "double_click":
            _send(view, QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton)
        _pump(view)
    finally:
        _close(window)


def test_paste_region_anchors_on_the_pending_move_not_the_last_handled_one() -> None:
    """Ctrl+V is a window-level QAction, so no MapView key handler flushes
    for it; paste_region() must, or it pastes one hover behind the cursor."""
    from PyQt5.QtCore import QEvent

    window = _window()
    try:
        view = window.map_view
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(0, 0, 0)
        window.on_edit_stroke_end()
        window._on_tool_selected("select")
        window.on_region_selected((0, 0, 1, 1))
        window.copy_region()
        _send(view, QEvent.MouseMove, _tile_pos(view, (10, 12)))
        _pump(view)
        assert window._hover_tile == (10, 12)
        _send(view, QEvent.MouseMove, _tile_pos(view, (30, 32)))
        window.paste_region()
        mm = window.scenario.map_manager
        assert mm.get_tile(30, 32).terrain_id == _TERRAIN
        assert mm.get_tile(10, 12).terrain_id != _TERRAIN
    finally:
        _close(window)


def test_a_leave_lands_the_pending_hover_before_clearing_the_highlight() -> None:
    """Otherwise the 0 ms flush redraws the gold highlight after leaveEvent
    cleared it, and it sits there with the cursor outside the view."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        view = window.map_view
        pos = _tile_pos(view, (20, 20))
        _send(view, QEvent.MouseMove, pos)
        _pump(view)
        assert view._highlight_outline_item is not None, "a handled hover draws no highlight -- vacuous"
        _send(view, QEvent.MouseMove, _tile_pos(view, (22, 20)))
        QApplication.sendEvent(view.viewport(), QEvent(QEvent.Leave))
        QApplication.sendEvent(view, QEvent(QEvent.Leave))
        _pump(view)
        assert view._highlight_outline_item is None
    finally:
        _close(window)


@pytest.mark.parametrize("how", ["hide", "set_source"])
def test_a_hide_or_a_new_source_drops_a_pending_move(how: str) -> None:
    from PyQt5.QtCore import QEvent
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        view = window.map_view
        handled: list[object] = []
        view._on_hover = handled.append
        _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
        view._move_gate_open = False
        if how == "hide":
            QApplication.sendEvent(view.viewport(), QEvent(QEvent.Hide))
        else:
            # Re-stashed at set_source() itself: refresh_map() may pump a disabled
            # window first, which drops the stash on its own and would hide this path.
            stash = view._move_stash
            original = view.set_source

            def restash_then_set_source(*args, **kwargs):
                view._move_stash = stash
                view._move_gate_open = False
                original(*args, **kwargs)

            view.set_source = restash_then_set_source
            try:
                window.refresh_map()
            finally:
                del view.set_source
        assert view._move_stash is None
        assert view._move_gate_open
        _pump_for(view.MOVE_PAINT_WAIT_MS * 3)
        assert handled == []
    finally:
        _close(window)


def test_closing_the_scenario_drops_a_pending_move() -> None:
    from PyQt5.QtCore import QEvent

    window = _window()
    try:
        view = window.map_view
        handled: list[object] = []
        view._on_hover = handled.append
        _send(view, QEvent.MouseMove, _tile_pos(view, (20, 20)))
        view._move_gate_open = False
        window.edit_history.mark_saved()
        window.close_scenario()
        assert view._move_stash is None
        assert view._move_gate_open
        _pump_for(view.MOVE_PAINT_WAIT_MS * 3)
        assert handled == []
    finally:
        _close(window)


def test_a_coalesced_middle_drag_scrolls_as_far_as_an_uncoalesced_one() -> None:
    from PyQt5.QtCore import QEvent, QPoint, Qt

    window = _window()
    try:
        view = window.map_view
        view.scale(4, 4)
        _pump(view)
        h, v = view.horizontalScrollBar(), view.verticalScrollBar()
        h.setValue((h.minimum() + h.maximum()) // 2)
        v.setValue((v.minimum() + v.maximum()) // 2)
        start = (h.value(), v.value())
        a, b = QPoint(300, 300), QPoint(700, 520)
        steps = [a + (b - a) * (i / 12) for i in range(1, 13)]

        _send(view, QEvent.MouseButtonPress, a, Qt.MiddleButton, Qt.MiddleButton)
        for p in steps:
            _send(view, QEvent.MouseMove, p, Qt.NoButton, Qt.MiddleButton)
        _send(view, QEvent.MouseButtonRelease, b, Qt.MiddleButton, Qt.NoButton)
        _pump(view)
        coalesced = (h.value(), v.value())
        assert coalesced != start, "the drag scrolled nothing, so the comparison is vacuous"

        h.setValue(start[0])
        v.setValue(start[1])
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, a, Qt.MiddleButton, Qt.MiddleButton))
        for p in steps:
            view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, p, Qt.NoButton, Qt.MiddleButton))
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, b, Qt.MiddleButton, Qt.NoButton))
        assert (h.value(), v.value()) == coalesced
    finally:
        _close(window)
