"""Perf Trace's drag bracketing and idle `perf view` flush, wired through a
real MapView: a stroke's press and release reach begin_drag()/end_drag(),
and a pan with no drag prints its own line once the canvas goes quiet
(viewer_canvas's idle timer). The logic itself is pinned Qt-free in
tests/test_perf_trace.py."""

from __future__ import annotations

import time

import pytest

from descape import debug_log, perf_trace

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _spin_until(predicate, timeout_s: float) -> bool:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def traced_window():
    from PyQt5.QtWidgets import QApplication

    window = conftest.terrain_edit_window()
    window._on_tool_selected("draw")
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.resize(600, 600)
    window.show()
    QApplication.processEvents()
    perf_trace.enable(True)
    debug_log.clear()
    yield window
    perf_trace.enable(False)
    perf_trace._current_step = {}
    perf_trace._step_totals = []
    perf_trace._phase_sums = {}
    perf_trace._phase_order = []
    perf_trace._repaint_durations = []
    perf_trace._armed_label = None
    perf_trace._drag_active = False
    debug_log.clear()
    conftest.close_window(window)


def test_a_real_stroke_is_bracketed_and_drops_hover_phases(traced_window):
    from PyQt5.QtCore import QEvent, Qt

    map_view = traced_window.map_view
    pos = conftest.polygon_viewport_pos(map_view, 30, 30)
    map_view.mouseMoveEvent(conftest.mouse_event(QEvent.MouseMove, pos, Qt.NoButton, Qt.NoButton))
    assert perf_trace._current_step, "hover no longer records pick -- this test proves nothing"

    map_view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, pos, Qt.LeftButton, Qt.LeftButton))
    assert perf_trace._drag_active
    assert "pick" not in perf_trace._phase_order or perf_trace._step_totals, "hover pick leaked into the drag"
    map_view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pos, Qt.LeftButton, Qt.NoButton))

    assert not perf_trace._drag_active
    assert "perf drag paint-terrain:" in debug_log.get_log_text()


def test_a_pan_with_no_drag_prints_its_own_view_line(traced_window):
    map_view = traced_window.map_view
    debug_log.clear()
    perf_trace._repaint_durations = []
    map_view.scale(1.25, 1.25)
    assert _spin_until(lambda: "perf view: repaint:" in debug_log.get_log_text(), timeout_s=5.0)
    assert "perf drag" not in debug_log.get_log_text()
