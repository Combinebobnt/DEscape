"""The rebindable mouse wheel (Settings > Keybinds, GH #172 and GH #210):
MapView.wheelEvent runs the action bound to the event's modifier slot.

tests/test_wheel_zoom.py is the pin that the shipped defaults keep the old
behaviour; this file is about the binding layer itself. Events go straight to
view.wheelEvent(), as there, and for the same reason: a run of them has to
stay one gesture. Alt + wheel arrives transposed on X11 and Windows (Qt 5
delivers it as a sideways delta), so the Alt slot runs with both shapes.
"""

from __future__ import annotations

import pytest
from test_wheel_zoom import _facing, _gesture, _place_window, _scale
from test_zoom_status import _window

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

# One wheel-up notch per slot, as (slot, modifier name, (x, y) angleDelta).
_SLOT_EVENTS = [
    pytest.param("plain", "NoModifier", (0, 120), id="plain"),
    pytest.param("ctrl", "ControlModifier", (0, 120), id="ctrl"),
    pytest.param("shift", "ShiftModifier", (0, 120), id="shift"),
    pytest.param("alt", "AltModifier", (0, 120), id="alt"),
    pytest.param("alt", "AltModifier", (120, 0), id="alt-transposed"),
]


def _mods(*names: str):
    from PyQt5.QtCore import Qt

    modifiers = Qt.KeyboardModifiers(Qt.NoModifier)
    for name in names:
        modifiers |= getattr(Qt, name)
    return modifiers


def _wheel_xy(view, delta: tuple[int, int], modifiers=None):
    """One wheel event carrying an (x, y) angleDelta, returned so a test can
    read isAccepted(). No gesture reset."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    center = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        center,
        center,
        QPoint(0, 0),
        QPoint(*delta),
        Qt.NoButton,
        Qt.NoModifier if modifiers is None else modifiers,
        Qt.NoScrollPhase,
        False,
    )
    view.wheelEvent(event)
    return event


def _hover_centre(view) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    center = QPointF(view.viewport().rect().center())
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, center, Qt.NoButton, Qt.NoButton, Qt.NoModifier))


def _terrain_window(tool: str = "draw"):
    """_window() in Terrain mode with `tool` active and the cursor hovering
    the viewport centre, so a brush tool shows its footprint there."""
    window = _window()
    window.mode_combo.setCurrentText("Terrain")
    getattr(window, f"{tool}_action").setChecked(True)
    assert window._current_tool == tool
    _hover_centre(window.map_view)
    return window


def _bind(window, **slots: str) -> None:
    """The Keybinds tab's own path: persist, then push the whole table down."""
    from descape import settings

    for slot, action in slots.items():
        settings.set_wheel_binding(slot, action)
    window.map_view.set_wheel_bindings(settings.get_wheel_bindings())


def _scrolls(view) -> tuple[int, int]:
    return view.horizontalScrollBar().value(), view.verticalScrollBar().value()


def _room_to_pan(view) -> None:
    """Zooms in by hand (not a wheel gesture) and parks both scrollbars mid-range."""
    view.scale(4.0, 4.0)
    for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
        bar.setValue((bar.minimum() + bar.maximum()) // 2)
        assert bar.minimum() + 500 < bar.value() < bar.maximum() - 500


def _footprint(view):
    assert view._highlight_outline_item is not None, "no hover footprint is showing"
    return view._highlight_outline_item.path().boundingRect()


def test_the_view_knows_a_modifier_set_for_every_settings_slot() -> None:
    from descape import settings
    from descape.map_view import MapView

    assert set(MapView._WHEEL_SLOT_MODIFIERS) == set(settings.WHEEL_SLOTS)
    assert len(set(MapView._WHEEL_SLOT_MODIFIERS.values())) == len(settings.WHEEL_SLOTS)


# --- the slot by action matrix ---------------------------------------------------


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
@pytest.mark.parametrize("action", ["tool", "rotate"])
def test_a_slot_bound_to_a_rotating_action_turns_the_ghost(slot, modifier, delta, action) -> None:
    window = _place_window()
    try:
        view = window.map_view
        assert view._unit_ghost_item is not None
        _bind(window, **{slot: action})
        scale, gesture = _scale(view), _gesture(view)
        event = _wheel_xy(view, delta, _mods(modifier))
        assert _facing(window) == 1, "scroll up is clockwise, like '.'"
        assert _scale(view) == scale
        assert _gesture(view) == gesture, "a turn must not spend the zoom gesture's budget"
        assert event.isAccepted()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
@pytest.mark.parametrize("action", ["tool", "brush_size"])
def test_a_slot_bound_to_a_brush_action_steps_the_brush_size(slot, modifier, delta, action) -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        _bind(window, **{slot: action})
        size, scale, gesture = window.brush_size_spin.value(), _scale(view), _gesture(view)
        event = _wheel_xy(view, delta, _mods(modifier))
        assert window.brush_size_spin.value() == size + 1, "scroll up is one size bigger"
        assert _scale(view) == scale
        assert _gesture(view) == gesture, "a brush step must not spend the zoom gesture's budget"
        assert event.isAccepted()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
@pytest.mark.parametrize("action", ["pan_vertical", "pan_horizontal"])
def test_a_slot_bound_to_a_pan_scrolls_the_view_up_or_left(slot, modifier, delta, action) -> None:
    from descape.map_view import MapView

    window = _terrain_window("draw")
    try:
        view = window.map_view
        _room_to_pan(view)
        _bind(window, **{slot: action})
        size, scale, gesture = window.brush_size_spin.value(), _scale(view), _gesture(view)
        h, v = _scrolls(view)
        event = _wheel_xy(view, delta, _mods(modifier))
        step = MapView.WHEEL_PAN_NOTCH_PX
        want = (h, v - step) if action == "pan_vertical" else (h - step, v)
        assert _scrolls(view) == want, "the action picks the axis, not the delta's own direction"
        assert _scale(view) == scale and _gesture(view) == gesture
        assert window.brush_size_spin.value() == size
        assert event.isAccepted()

        _wheel_xy(view, (-delta[0], -delta[1]), _mods(modifier))
        assert _scrolls(view) == (h, v), "wheel down did not pan back"
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
@pytest.mark.parametrize("state", ["draw", "ghost"])
def test_a_slot_bound_to_nothing_swallows_the_event(slot, modifier, delta, state) -> None:
    window = _terrain_window("draw") if state == "draw" else _place_window()
    try:
        view = window.map_view
        _room_to_pan(view)
        _bind(window, **{slot: "none"})
        before = (_scale(view), _gesture(view), _scrolls(view), window.brush_size_spin.value())
        event = _wheel_xy(view, delta, _mods(modifier))
        assert (_scale(view), _gesture(view), _scrolls(view), window.brush_size_spin.value()) == before
        assert window._place_rotation.facing == 0.0
        assert event.isAccepted()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
def test_a_slot_bound_to_zoom_zooms_over_the_ghost_and_in_draw(slot, modifier, delta) -> None:
    """The zoom column, where the default (Ctrl) would otherwise act on the tool."""
    for build in (_place_window, _terrain_window):
        window = build()
        try:
            view = window.map_view
            _bind(window, **{slot: "zoom"})
            size, scale = window.brush_size_spin.value(), _scale(view)
            _wheel_xy(view, delta, _mods(modifier))
            assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
            assert window.brush_size_spin.value() == size
            assert window._place_rotation.facing == 0.0
        finally:
            conftest.close_window(window)


# --- a tool action with nothing to act on zooms -----------------------------------


def test_the_tool_action_zooms_on_pan() -> None:
    window = _terrain_window("pan")
    try:
        view = window.map_view
        _bind(window, plain="tool")
        size, level, scale = window.brush_size_spin.value(), window.elevation_level_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert (window.brush_size_spin.value(), window.elevation_level_spin.value()) == (size, level)
    finally:
        conftest.close_window(window)


def test_the_brush_size_action_zooms_over_the_ghost() -> None:
    window = _place_window()
    try:
        view = window.map_view
        assert view._unit_ghost_item is not None
        _bind(window, ctrl="brush_size")
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size
        assert _facing(window) == 0
    finally:
        conftest.close_window(window)


def test_the_rotate_action_zooms_with_draw_active() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        _bind(window, ctrl="rotate")
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size
        assert window._place_rotation.facing == 0.0
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("action", ["tool", "brush_size"])
def test_a_brush_action_in_the_eyedropper_zooms_and_leaves_the_level_alone(action) -> None:
    """Brush size only: "]" falls through to the Level there, the wheel does not."""
    window = _terrain_window("eyedropper")
    try:
        view = window.map_view
        assert window.elevation_level_spin.isEnabled() and not window.brush_size_spin.isEnabled()
        window.elevation_level_spin.setValue(3)
        _bind(window, ctrl=action)
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.elevation_level_spin.value() == 3
        assert window.brush_size_spin.value() == size
    finally:
        conftest.close_window(window)


# --- the shipped defaults, and the two tester setups -------------------------------


def test_by_default_ctrl_wheel_steps_the_brush_in_draw_and_plain_wheel_zooms() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size + 1
        assert _scale(view) == scale
        _wheel_xy(view, (0, -120), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size

        _wheel_xy(view, (0, 120))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize(
    "modifier, delta",
    [
        pytest.param("ShiftModifier", (0, 120), id="shift"),
        pytest.param("AltModifier", (0, 120), id="alt"),
        pytest.param("AltModifier", (120, 0), id="alt-transposed"),
    ],
)
def test_by_default_shift_and_alt_wheel_zoom(modifier, delta) -> None:
    """Alt + wheel was dead where the platform transposes it; Shift was never read."""
    window = _terrain_window("draw")
    try:
        view = window.map_view
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, delta, _mods(modifier))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size
    finally:
        conftest.close_window(window)


def test_wheel_on_tool_and_ctrl_on_zoom_over_the_ghost() -> None:
    """GH #172's follow-up: plain wheel turns the preview, Ctrl + wheel zooms."""
    window = _place_window()
    try:
        view = window.map_view
        _bind(window, plain="tool", ctrl="zoom")
        scale = _scale(view)
        _wheel_xy(view, (0, 120))
        assert _facing(window) == 1
        _wheel_xy(view, (0, -240))
        assert _facing(window) == 15
        assert _scale(view) == scale

        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert _facing(window) == 15
    finally:
        conftest.close_window(window)


def test_wheel_on_tool_and_ctrl_on_zoom_with_draw_active() -> None:
    """GH #210: plain wheel sizes the brush, its footprint follows, the bounds
    clamp without a zoom, and Ctrl + wheel zooms."""
    from descape import brush

    window = _terrain_window("draw")
    try:
        view = window.map_view
        _bind(window, plain="tool", ctrl="zoom")
        window.brush_size_spin.setValue(3)
        scale, three = _scale(view), _footprint(view)

        _wheel_xy(view, (0, 240))
        assert window.brush_size_spin.value() == 5
        five = _footprint(view)
        assert five.width() > three.width() and five.height() > three.height(), "the footprint kept its old size"
        _wheel_xy(view, (0, -240))
        assert window.brush_size_spin.value() == 3
        assert _footprint(view) == three

        window.brush_size_spin.setValue(brush.BRUSH_SIZE_MAX - 1)
        _wheel_xy(view, (0, 360))
        assert window.brush_size_spin.value() == brush.BRUSH_SIZE_MAX
        _wheel_xy(view, (0, 120))
        assert window.brush_size_spin.value() == brush.BRUSH_SIZE_MAX
        window.brush_size_spin.setValue(brush.BRUSH_SIZE_MIN + 1)
        _wheel_xy(view, (0, -360))
        assert window.brush_size_spin.value() == brush.BRUSH_SIZE_MIN
        _wheel_xy(view, (0, -120))
        assert window.brush_size_spin.value() == brush.BRUSH_SIZE_MIN
        assert _scale(view) == scale, "a step refused at a bound fell through to the zoom"

        size = window.brush_size_spin.value()
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size
    finally:
        conftest.close_window(window)


# --- the brush accumulator --------------------------------------------------------


def test_two_half_notches_make_one_brush_step() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        size, gesture = window.brush_size_spin.value(), _gesture(view)
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size + 1
        assert _gesture(view) == gesture, "a brush step must not spend the zoom gesture's budget"
    finally:
        conftest.close_window(window)


def test_a_reversed_brush_wheel_drops_the_stale_half_notch() -> None:
    """+60 then -120 is one step down, the zoom's own flip rule."""
    window = _terrain_window("draw")
    try:
        view = window.map_view
        window.brush_size_spin.setValue(4)
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        _wheel_xy(view, (0, -120), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == 3
        _wheel_xy(view, (0, -60), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == 3, "the reversal banked the stale half notch"
    finally:
        conftest.close_window(window)


def test_a_tool_switch_drops_a_leftover_brush_half_notch() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        size = window.brush_size_spin.value()
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        window.elevation_action.setChecked(True)
        assert window._current_tool == "elevation" and window.brush_size_spin.isEnabled()
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size, "the first half notch must not have survived"
        _wheel_xy(view, (0, 60), _mods("ControlModifier"))
        assert window.brush_size_spin.value() == size + 1
    finally:
        conftest.close_window(window)


def test_a_wheel_pan_stands_down_during_a_middle_drag() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        _room_to_pan(view)
        _bind(window, plain="pan_vertical")
        before = _scrolls(view)
        view._middle_drag_active = True
        _wheel_xy(view, (0, 120))
        assert _scrolls(view) == before
        view._middle_drag_active = False
        _wheel_xy(view, (0, 120))
        assert _scrolls(view) != before
    finally:
        conftest.close_window(window)


# --- exact-match slots ------------------------------------------------------------


@pytest.mark.parametrize(
    "modifiers, delta",
    [
        pytest.param(("ControlModifier", "ShiftModifier"), (0, 120), id="ctrl-shift"),
        pytest.param(("ControlModifier", "AltModifier"), (0, 120), id="ctrl-alt"),
        pytest.param(("ControlModifier", "AltModifier"), (120, 0), id="ctrl-alt-transposed"),
        pytest.param(("ControlModifier", "MetaModifier"), (0, 120), id="ctrl-meta"),
        pytest.param(("MetaModifier",), (0, 120), id="meta"),
    ],
)
def test_an_unlisted_modifier_combination_zooms_over_the_ghost(modifiers, delta) -> None:
    """Run where Ctrl alone would turn the ghost, so a zoom here is the rule and not a fallback."""
    window = _place_window()
    try:
        view = window.map_view
        assert view._unit_ghost_item is not None
        scale = _scale(view)
        _wheel_xy(view, delta, _mods(*modifiers))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert _facing(window) == 0
    finally:
        conftest.close_window(window)


def test_the_keypad_modifier_does_not_break_the_slot_lookup() -> None:
    window = _place_window()
    try:
        view = window.map_view
        scale = _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier", "KeypadModifier"))
        assert _facing(window) == 1
        assert _scale(view) == scale
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("modifier", ["NoModifier", "ControlModifier", "ShiftModifier"])
def test_a_sideways_tilt_without_alt_still_does_nothing(modifier) -> None:
    window = _place_window()
    try:
        view = window.map_view
        _bind(window, plain="tool", shift="tool")
        scale, gesture = _scale(view), _gesture(view)
        _wheel_xy(view, (120, 0), _mods(modifier))
        assert _scale(view) == scale and _gesture(view) == gesture
        assert _facing(window) == 0
    finally:
        conftest.close_window(window)


def test_a_binding_saved_before_the_window_is_built_is_live_on_the_first_event() -> None:
    from descape import settings

    settings.set_wheel_binding("plain", "brush_size")
    settings.set_wheel_binding("ctrl", "zoom")
    window = _terrain_window("draw")
    try:
        view = window.map_view
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120))
        assert window.brush_size_spin.value() == size + 1
        assert _scale(view) == scale
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == pytest.approx(scale * 1.25, rel=1e-9)
        assert window.brush_size_spin.value() == size + 1
    finally:
        conftest.close_window(window)
