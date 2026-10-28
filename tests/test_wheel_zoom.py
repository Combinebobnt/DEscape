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
from test_zoom_status import _window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _wheel_units(view, units: int, modifiers=None) -> None:
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
        Qt.NoModifier if modifiers is None else modifiers,
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


def test_a_roll_that_keeps_going_crosses_one_level_per_gap() -> None:
    """A rate limit, not a stop: a roll with no pause re-anchors its budget
    once WHEEL_GESTURE_GAP_S has passed since the last crossing (2026-09-30
    stress log: continuous rolling stuck at 603% of fit until the wheel idled).
    The time since the crossing is staged by backdating the budget stamp only;
    the last-event stamp stays fresh, so the gesture itself never ends."""
    window = _window()
    try:
        view = window.map_view
        levels = view._canvas_item._cache.mip_levels()
        first_mip = view.viewport_chunk_target()[0]
        if first_mip + 2 > levels[-1]:
            pytest.skip("ladder has no room for two levels of zoom-in")
        _wheel_units(view, 1200)
        mid_mip = view.viewport_chunk_target()[0]
        assert mid_mip == first_mip + 1
        _wheel_units(view, 1200)
        assert view.viewport_chunk_target()[0] == mid_mip, "a second crossing inside the gap"
        view._wheel_budget_t -= view.WHEEL_GESTURE_GAP_S
        gesture_before = view._wheel_dir, view._wheel_last_t
        _wheel_units(view, 1200)
        assert view.viewport_chunk_target()[0] == mid_mip + 1, "a continuing roll stayed refused past the gap"
        assert gesture_before[0] == view._wheel_dir and view._wheel_last_t >= gesture_before[1]
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("up", [True, False])
def test_the_last_detent_lands_exactly_on_the_zoom_bound(up) -> None:
    """A detent that would overshoot the ceiling (floor) is clamped onto it,
    so the readout reaches 6400% (50%) instead of stopping one step short."""
    window = _window()
    try:
        view = window.map_view
        for _ in range(80):
            before = _scale(view)
            view._end_wheel_gesture()
            _wheel_units(view, 120 if up else -120)
            if _scale(view) == before:
                break
        else:
            pytest.fail("never reached the zoom bound")
        want = view.MAX_ZOOM_MULTIPLE_OF_FIT if up else view.MIN_ZOOM_FRACTION_OF_FIT
        assert view.zoom_percent_of_fit() == pytest.approx(want * 100, rel=1e-6)
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


# --- GH #172: Ctrl+wheel turns Place Unit's ghost --------------------------------

_PLACE_CONST = 83  # a villager: ANGLE, 16 frames


def _place_window(const: int = _PLACE_CONST):
    """_window() with Place Unit armed on `const` and the cursor hovering the
    viewport centre, where _wheel_units() points, so the ghost is showing."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    window = _window()
    window.mode_combo.setCurrentText("Units")
    window.units_panel._set_pending_object(const)
    window.units_panel.select_owner(1)
    window.place_unit_action.setChecked(True)
    view = window.map_view
    center = QPointF(view.viewport().rect().center())
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, center, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
    return window


def _ctrl():
    from PyQt5.QtCore import Qt

    return Qt.ControlModifier


def _gesture(view) -> tuple:
    return (view._wheel_accum, view._wheel_dir, view._wheel_gesture_mip, view._wheel_last_t, view._wheel_budget_t)


def _facing(window) -> int:
    from descape import place_rotation, unit_rotation

    value = place_rotation.rotation_for(window._place_rotation, _PLACE_CONST, 1)
    return unit_rotation.rotation_to_facing(value, 16)


def test_ctrl_wheel_over_the_ghost_turns_it_and_leaves_the_zoom_alone() -> None:
    window = _place_window()
    try:
        view = window.map_view
        assert view._unit_ghost_item is not None
        scale, gesture = _scale(view), _gesture(view)
        _wheel_units(view, 120, _ctrl())
        assert _facing(window) == 1, "scroll up is clockwise, like '.'"
        _wheel_units(view, -240, _ctrl())
        assert _facing(window) == 15
        assert _scale(view) == scale
        assert _gesture(view) == gesture, "a turn must not spend the zoom gesture's budget"
    finally:
        conftest.close_window(window)


def test_two_half_notches_make_one_step() -> None:
    window = _place_window()
    try:
        view = window.map_view
        _wheel_units(view, 60, _ctrl())
        assert _facing(window) == 0
        _wheel_units(view, 60, _ctrl())
        assert _facing(window) == 1
    finally:
        conftest.close_window(window)


def test_plain_wheel_still_zooms_with_place_unit_armed() -> None:
    window = _place_window()
    try:
        view = window.map_view
        before = _scale(view)
        _wheel_units(view, 120)
        assert _scale(view) == pytest.approx(before * 1.25, rel=1e-9)
        assert _facing(window) == 0
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("case", ["wall_pick", "off_map", "other_tool"])
def test_ctrl_wheel_with_no_ghost_does_nothing(case) -> None:
    """The default Ctrl row is "tool": with no ghost and no brush it has
    nothing to act on, so it no longer zooms (GH #245, user 2026-10-25)."""
    from PyQt5.QtCore import QEvent

    from descape.viewer import _wall_consts

    window = _place_window()
    try:
        view = window.map_view
        if case == "wall_pick":
            window.units_panel._set_pending_object(min(_wall_consts()))
        elif case == "off_map":
            view.leaveEvent(QEvent(QEvent.Leave))
        else:
            window.pan_action.setChecked(True)
        assert view._unit_ghost_item is None
        before = _scale(view)
        _wheel_units(view, 120, _ctrl())
        assert _scale(view) == before
        assert window._place_rotation.facing == 0.0
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("case", ["leave", "tool_switch", "catalog_pick"])
def test_a_leftover_half_notch_is_dropped_when_the_ghost_goes(case) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    window = _place_window()
    try:
        view = window.map_view
        _wheel_units(view, 60, _ctrl())
        if case == "leave":
            view.leaveEvent(QEvent(QEvent.Leave))
        elif case == "tool_switch":
            window.pan_action.setChecked(True)
            window.place_unit_action.setChecked(True)
        else:
            window.units_panel._set_pending_object(4)  # an archer, then back
            window.units_panel._set_pending_object(_PLACE_CONST)
        center = QPointF(view.viewport().rect().center())
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, center, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        assert view._unit_ghost_item is not None
        _wheel_units(view, 60, _ctrl())
        assert _facing(window) == 0, "the first half notch must not have survived"
        _wheel_units(view, 60, _ctrl())
        assert _facing(window) == 1
    finally:
        conftest.close_window(window)


def test_a_reversed_ctrl_wheel_drops_the_stale_half_notch() -> None:
    """+60 then -120 is one counter-clockwise step, the zoom's own flip rule."""
    window = _place_window()
    try:
        view = window.map_view
        _wheel_units(view, 60, _ctrl())
        _wheel_units(view, -120, _ctrl())
        assert _facing(window) == 15
    finally:
        conftest.close_window(window)
