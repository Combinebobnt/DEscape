"""GH #146: a wheel notch right after a pan, mouse not moved since, must zoom
around the cursor, not jump the view to a scene-rect corner.

Qt's own AnchorUnderMouse anchors on the scene point it recorded at the last
mouse move it processed, which a middle-drag pan (MapView consumes those
events) and a hand-drag pushed into the overscroll clamp both leave stale.

Events go through QApplication.sendEvent() on the viewport, and every event
first moves QCursor and sets WA_UnderMouse: Qt's anchor path reads
QCursor::pos() and underMouse(), not the event's position, so without them a
test takes the view-centre fallback and passes whether or not the bug is
there.
"""

from __future__ import annotations

import pytest
from test_zoom_status import _window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _put_cursor(view, p) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QCursor

    QCursor.setPos(view.viewport().mapToGlobal(p))
    view.setAttribute(Qt.WA_UnderMouse, True)
    view.viewport().setAttribute(Qt.WA_UnderMouse, True)


def _mouse(view, kind, p, button=None, buttons=None) -> None:
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtWidgets import QApplication

    _put_cursor(view, p)
    event = conftest.mouse_event(
        kind, QPointF(p), Qt.NoButton if button is None else button, Qt.NoButton if buttons is None else buttons
    )
    QApplication.sendEvent(view.viewport(), event)


def _wheel(view, p) -> None:
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    _put_cursor(view, p)
    view._end_wheel_gesture()
    event = QWheelEvent(
        QPointF(p),
        QPointF(view.viewport().mapToGlobal(p)),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view.viewport(), event)
    QApplication.processEvents()


def _drag(view, button, a, b) -> None:
    from PyQt5.QtCore import QEvent, Qt

    _mouse(view, QEvent.MouseMove, a)
    _mouse(view, QEvent.MouseButtonPress, a, button, button)
    for i in range(1, 11):
        _mouse(view, QEvent.MouseMove, a + (b - a) * (i / 10), Qt.NoButton, button)
    _mouse(view, QEvent.MouseButtonRelease, b, button, Qt.NoButton)


def _zoomed_in_window(centered_on_cursor: bool):
    from PyQt5.QtCore import QEvent

    window = _window()
    view = window.map_view
    view.set_zoom_anchor_mode(centered_on_cursor)
    center = view.viewport().rect().center()
    _mouse(view, QEvent.MouseMove, center)
    for _ in range(4):
        _wheel(view, center)
    return window, view


def _cursor_drift(view, p) -> int:
    """Wheels one notch in at `p` and returns how far, in viewport px, the
    scene point that was under `p` ended up from it."""
    before = view.mapToScene(p)
    _wheel(view, p)
    return (view.mapFromScene(before) - p).manhattanLength()


@pytest.mark.parametrize(("start", "end"), [((300, 300), (900, 700)), ((1100, 800), (300, 200))])
def test_wheel_after_middle_drag_zooms_under_cursor(start, end) -> None:
    from PyQt5.QtCore import QPoint, Qt

    window, view = _zoomed_in_window(True)
    try:
        b = QPoint(*end)
        _drag(view, Qt.MiddleButton, QPoint(*start), b)
        assert _cursor_drift(view, b) <= 3
    finally:
        conftest.close_window(window)


def test_wheel_after_hand_drag_into_clamp_zooms_under_cursor() -> None:
    from PyQt5.QtCore import QPoint, Qt

    window, view = _zoomed_in_window(True)
    try:
        a, b = QPoint(200, 200), QPoint(1300, 900)
        # Repeated drags so the pan runs into the overscroll clamp.
        for _ in range(7):
            _drag(view, Qt.LeftButton, a, b)
        assert _cursor_drift(view, b) <= 3
    finally:
        conftest.close_window(window)


def test_view_centre_mode_keeps_the_centre_after_a_pan() -> None:
    from PyQt5.QtCore import QPoint, Qt

    window, view = _zoomed_in_window(False)
    try:
        b = QPoint(900, 700)
        _drag(view, Qt.MiddleButton, QPoint(300, 300), b)
        centre = view.viewport().rect().center()
        before = view.mapToScene(centre)
        _wheel(view, b)
        assert (view.mapFromScene(before) - centre).manhattanLength() <= 3
    finally:
        conftest.close_window(window)
