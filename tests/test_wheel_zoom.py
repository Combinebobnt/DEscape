"""MapView.wheelEvent's notch accumulator and its one-mip-per-gesture clamp
(2026-09-09 perf batch B, item 22).

Before this, every wheel event was one full 1.25x zoom step and one
full-viewport repaint no matter how many angleDelta units it carried: a
high-resolution wheel or a trackpad paid a step per tiny event, and a hard
flick could cross two mip levels in one go, landing outside
level_warm.neighbour_mips()'s +/-1 warm set and paying a synchronous level
build inside paint().

Events are delivered straight to view.wheelEvent(), the same way
test_zoom_status._wheel does and for the same reason, but deliberately NOT
through that helper, which ends the gesture before each call. Everything
here is about what a run of events does, so the run has to stay one gesture.
"""

from __future__ import annotations

import pytest

import conftest
from test_zoom_status import _window

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _wheel_units(view, units: int) -> None:
    """One wheel event carrying `units` of vertical angleDelta. 120 is one
    ordinary detent, less is a high-resolution wheel's fraction of one, more
    is a flick coalesced into a single event. No gesture reset."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    center = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        center,
        center,
        QPoint(0, 0),
        QPoint(0, units),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    view.wheelEvent(event)


def _scale(view) -> float:
    return abs(view.transform().determinant()) ** 0.5


def test_one_full_notch_is_still_exactly_one_zoom_step() -> None:
    """The coarse-wheel case: unchanged from before the accumulator."""
    window = _window()
    try:
        view = window.map_view
        before = _scale(view)
        _wheel_units(view, 120)
        assert _scale(view) == pytest.approx(before * 1.25, rel=1e-9)
    finally:
        conftest.close_window(window)


def test_sub_notch_events_accumulate_into_one_step() -> None:
    """Three 40-unit events are one detent's worth, so they cost one zoom
    step and two repaint-free no-ops, not three steps."""
    window = _window()
    try:
        view = window.map_view
        before = _scale(view)
        _wheel_units(view, 40)
        assert _scale(view) == pytest.approx(before, rel=1e-9)
        _wheel_units(view, 40)
        assert _scale(view) == pytest.approx(before, rel=1e-9)
        _wheel_units(view, 40)
        assert _scale(view) == pytest.approx(before * 1.25, rel=1e-9)
    finally:
        conftest.close_window(window)


def test_a_hard_flick_crosses_at_most_one_mip_level() -> None:
    """1200 units is ten notches, i.e. 9.3x unclamped, more than three mip
    levels' worth. The gesture is allowed exactly one."""
    window = _window()
    try:
        view = window.map_view
        target = view.viewport_chunk_target()
        assert target is not None, "no canvas item, so nothing selects a mip"
        before_mip = target[0]
        levels = view._canvas_item._cache.mip_levels()
        if before_mip + 1 > levels[-1]:
            pytest.skip("view already sits on the finest enumerated level")
        before = _scale(view)
        _wheel_units(view, 1200)
        after_mip = view.viewport_chunk_target()[0]
        ratio = _scale(view) / before
        assert after_mip - before_mip == 1
        # Whole notches, and fewer than the ~3.05 notches a second mip
        # boundary would need (1.25**k < 4.0).
        assert 1.25 - 1e-9 <= ratio < 4.0
    finally:
        conftest.close_window(window)


def test_a_pause_starts_a_fresh_mip_budget() -> None:
    """The clamp is per gesture, not a hard stop: pausing lets the next flick
    move another level. The pause is staged by backdating the last-event
    timestamp, which exercises wheelEvent's own gap branch rather than the
    reset method a real sleep would reach through."""
    window = _window()
    try:
        view = window.map_view
        levels = view._canvas_item._cache.mip_levels()
        first_mip = view.viewport_chunk_target()[0]
        if first_mip + 2 > levels[-1]:
            pytest.skip("ladder has no room for two levels of zoom-in")
        _wheel_units(view, 1200)
        mid_mip = view.viewport_chunk_target()[0]
        _wheel_units(view, 1200)
        assert view.viewport_chunk_target()[0] == mid_mip, "same gesture must not cross a second level"
        view._wheel_last_t -= 1.0
        _wheel_units(view, 1200)
        assert view.viewport_chunk_target()[0] == mid_mip + 1
    finally:
        conftest.close_window(window)


def test_a_reversal_after_a_whole_detent_starts_a_new_gesture() -> None:
    """Reversing is the other way a gesture ends, and it has to work off the
    running direction rather than the residual: whole detents leave a
    residual of exactly 0, so flicking out and straight back in would
    otherwise stay anchored on the level the FIRST flick started at and
    travel two levels from where the view actually sits."""
    window = _window()
    try:
        view = window.map_view
        levels = view._canvas_item._cache.mip_levels()
        # Park somewhere with a level on either side, via view.scale() so the
        # setup itself is not a wheel gesture.
        for _ in range(60):
            mip = view.viewport_chunk_target()[0]
            if levels[0] < mip < levels[-1]:
                break
            view.scale(1.25, 1.25)
        else:
            pytest.skip("no mip level with a neighbour on both sides is reachable")
        start_mip = view.viewport_chunk_target()[0]
        _wheel_units(view, -1200)
        assert view.viewport_chunk_target()[0] == start_mip - 1
        _wheel_units(view, 1200)
        assert view.viewport_chunk_target()[0] == start_mip
    finally:
        conftest.close_window(window)


def test_a_direction_flip_drops_the_residual() -> None:
    """+60 then -120 is one zoom-out step, not none: the flip throws the
    stale +60 away rather than netting it against the new direction."""
    window = _window()
    try:
        view = window.map_view
        before = _scale(view)
        _wheel_units(view, 60)
        assert _scale(view) == pytest.approx(before, rel=1e-9)
        _wheel_units(view, -120)
        assert _scale(view) == pytest.approx(before * 0.8, rel=1e-9)
    finally:
        conftest.close_window(window)
