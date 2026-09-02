"""Main-window geometry: the left column runs full height, the system log
sits beside it rather than under it, and the log's height is draggable and
persisted.

Nothing pinned main-window geometry before this file. The layout it covers is
easy to regress silently -- the log was a fixed-height full-width strip for
most of the project's life, and reverting to that shape breaks no other test.

Same offscreen-ViewerWindow technique as tests/test_trigger_panel.py, and its
_window() fixture's show() + processEvents() is load-bearing here for the same
reason: QSplitter.setSizes() is renormalized against real geometry, so on an
unshown window every size assertion tests Qt's layout fallback instead.
"""

from __future__ import annotations

import pytest

import conftest
from descape import settings

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _window():
    """A shown, fixed-size offscreen ViewerWindow."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.resize(1500, 900)
    window.show()
    QApplication.processEvents()
    return window


def test_left_column_runs_full_height_with_the_log_beside_it() -> None:
    from PyQt5.QtCore import QPoint

    window = _window()
    try:
        column = window.left_stack
        log = window.status_log
        column_bottom = column.mapTo(window, QPoint(0, column.height())).y()
        log_bottom = log.mapTo(window, QPoint(0, log.height())).y()
        assert column_bottom == log_bottom, "left column must reach the log's bottom edge"

        column_right = column.mapTo(window, QPoint(column.width(), 0)).x()
        log_left = log.mapTo(window, QPoint(0, 0)).x()
        # Flush against the handle, not merely somewhere to the right of it.
        assert log_left - column_right == window.content_splitter.handleWidth()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_log_height_is_draggable_and_holds_across_a_window_resize() -> None:
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.log_splitter.setSizes([10_000, 200])
        QApplication.processEvents()
        assert window.log_splitter.sizes()[1] == 200
        window.resize(1500, 1200)
        QApplication.processEvents()
        # Stretch factor 0 on the log pane: the map absorbs the resize, so this
        # is an exact height and not a ratio that drifts with the window.
        assert window.log_splitter.sizes()[1] == 200
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_persisted_log_height_is_restored_on_the_next_launch() -> None:
    """The path a user actually experiences, and the one the other tests
    structurally cannot see: ViewerWindow calls setSizes() during construction,
    on an UNSHOWN splitter, where Qt renormalizes sizes against the layout
    fallback rather than real geometry (see _window()'s note in
    test_trigger_panel.py). Setting a height on an already-shown window proves
    nothing about that."""
    settings.set_log_height(250)
    window = _window()
    try:
        assert window.log_splitter.sizes()[1] == 250
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_dragging_the_log_to_its_extreme_still_leaves_a_usable_map() -> None:
    """setChildrenCollapsible(False) on its own bottoms out at QGraphicsView's
    70 px minimumSizeHint, which on a letterboxed render shows only background.
    The explicit floor is what keeps the map a map at the extreme."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.log_splitter.setSizes([0, 10_000])
        QApplication.processEvents()
        assert window.map_view.height() >= settings.MIN_MAP_PANE
        # The other extreme: the log never shrinks below its own two-line floor.
        window.log_splitter.setSizes([10_000, 0])
        QApplication.processEvents()
        assert window.status_log.height() >= settings.MIN_LOG_PANE
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_log_height_round_trips_through_the_config_file(monkeypatch) -> None:
    """closeEvent persists the dragged height, and get_log_height reads it
    back. The manual "close and reopen the app" check, automated."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    try:
        window.log_splitter.setSizes([10_000, 200])
        QApplication.processEvents()
        window.edit_history.mark_saved()
    finally:
        window.close()

    assert "log_height: 200" in settings.CONFIG_PATH.read_text()
    # Clear the memo so this reads the file rather than the value closeEvent
    # just cached, which is what a fresh process would do.
    monkeypatch.setattr(settings, "_log_height", None)
    assert settings.get_log_height() == 200
