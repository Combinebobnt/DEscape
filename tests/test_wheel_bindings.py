"""The rebindable mouse wheel (Settings > Keybinds, GH #172 and GH #210):
MapView.wheelEvent runs the action bound to the event's modifier slot.

tests/test_wheel_zoom.py pins the shipped defaults; this file is about the
binding layer itself. Events go straight to
view.wheelEvent(), as there, and for the same reason: a run of them has to
stay one gesture. Alt + wheel arrives transposed on X11 and Windows (Qt 5
delivers it as a sideways delta), so the Alt slot runs with both shapes.

That leaves the step from QApplication to wheelEvent() untested, so one
test sends single notches through QApplication.sendEvent() on the viewport
instead (_send_wheel, GH #245). A single notch has no gesture to keep.

The paste-turn tests (GH #245) reuse tests/test_region_rotate.py's blank
template, block and paste.
"""

from __future__ import annotations

import pytest
import test_unit_drag_preview as drag_preview
from test_region_rotate import _add, _close, _live, _paste, _setup, _tiles, _units
from test_region_select import _select_window
from test_wheel_zoom import _PLACE_CONST, _facing, _gesture, _place_window, _scale
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


def _send_wheel(view, delta: tuple[int, int], modifiers=None):
    """_wheel_xy() through QApplication.sendEvent() on the viewport: Qt's own
    delivery path down to wheelEvent(), all but the platform plugin."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    center = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        center,
        QPointF(view.viewport().mapToGlobal(center.toPoint())),
        QPoint(0, 0),
        QPoint(*delta),
        Qt.NoButton,
        Qt.NoModifier if modifiers is None else modifiers,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view.viewport(), event)
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


# --- a tool action with nothing to act on does nothing (GH #245) -------------------


def test_the_tool_action_does_nothing_on_pan() -> None:
    """User, 2026-10-25: only a row set to Zoom zooms, so a tool action with
    no brush, ghost or paste swallows the notch instead of zooming."""
    window = _terrain_window("pan")
    try:
        view = window.map_view
        _bind(window, plain="tool")
        before = (window.brush_size_spin.value(), window.elevation_level_spin.value(), _scale(view), _gesture(view))
        event = _wheel_xy(view, (0, 120))
        assert (window.brush_size_spin.value(), window.elevation_level_spin.value(), _scale(view), _gesture(view)) == before
        assert event.isAccepted()
    finally:
        conftest.close_window(window)


def test_the_brush_size_action_does_nothing_over_the_ghost() -> None:
    window = _place_window()
    try:
        view = window.map_view
        assert view._unit_ghost_item is not None
        _bind(window, ctrl="brush_size")
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert (_scale(view), window.brush_size_spin.value(), _facing(window)) == (scale, size, 0)
    finally:
        conftest.close_window(window)


def test_the_rotate_action_does_nothing_with_draw_active() -> None:
    window = _terrain_window("draw")
    try:
        view = window.map_view
        _bind(window, ctrl="rotate")
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert (_scale(view), window.brush_size_spin.value()) == (scale, size)
        assert window._place_rotation.facing == 0.0
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("action", ["tool", "brush_size"])
def test_a_brush_action_in_the_eyedropper_does_nothing_and_leaves_the_level_alone(action) -> None:
    """Brush size only: "]" falls through to the Level there, the wheel does not."""
    window = _terrain_window("eyedropper")
    try:
        view = window.map_view
        assert window.elevation_level_spin.isEnabled() and not window.brush_size_spin.isEnabled()
        window.elevation_level_spin.setValue(3)
        _bind(window, ctrl=action)
        size, scale = window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert (_scale(view), window.elevation_level_spin.value(), window.brush_size_spin.value()) == (scale, 3, size)
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


# --- real delivery (GH #245) --------------------------------------------------------


@pytest.mark.parametrize("slot, modifier, delta", _SLOT_EVENTS)
@pytest.mark.parametrize("tool", ["draw", "select"])
@pytest.mark.parametrize("action", ["pan_vertical", "brush_size"])
def test_a_wheel_event_sent_through_the_application_runs_its_binding(slot, modifier, delta, tool, action) -> None:
    """Brush size has no brush to step in Select, so it does nothing there."""
    from descape.map_view import MapView

    window = _terrain_window(tool)
    try:
        view = window.map_view
        _room_to_pan(view)
        _bind(window, **{slot: action})
        has_brush = window.brush_size_spin.isEnabled()
        assert has_brush == (tool == "draw")
        size, scale, (h, v) = window.brush_size_spin.value(), _scale(view), _scrolls(view)
        event = _send_wheel(view, delta, _mods(modifier))
        if action == "pan_vertical":
            assert _scrolls(view) == (h, v - MapView.WHEEL_PAN_NOTCH_PX)
            assert (_scale(view), window.brush_size_spin.value()) == (scale, size)
            assert event.isAccepted()
        elif has_brush:
            assert window.brush_size_spin.value() == size + 1
            assert _scale(view) == scale
            assert event.isAccepted()
        else:
            assert (_scale(view), window.brush_size_spin.value()) == (scale, size)
            assert event.isAccepted()
    finally:
        conftest.close_window(window)


# --- the rotate binding turns a just-pasted region (GH #245) --------------------------


def _paste_window(tool: str = "select"):
    """_select_window() with test_region_rotate's block copied and pasted, then `tool` active."""
    window = _select_window()
    _setup(window)
    _paste(window)
    if tool != "select":
        getattr(window, f"{tool}_action").setChecked(True)
        assert window._current_tool == tool
    assert window._paste_armed() and window._region_turn_target()
    return window


def _turn(window) -> str:
    return window._paste_move.element


def test_one_wheel_notch_turns_the_pasted_region_as_the_rotate_key_does() -> None:
    from descape import region_transform

    window = _paste_window()
    try:
        view = window.map_view
        _bind(window, plain="rotate")
        depth = len(window.edit_history.records)
        scale, gesture = _scale(view), _gesture(view)
        event = _wheel_xy(view, (0, 120))
        assert event.isAccepted()
        assert _turn(window) == region_transform.ROTATE_CW, "scroll up turns the ghost's way round"
        assert len(window.edit_history.records) == depth, "a turn is folded into the paste's own record"
        assert _scale(view) == scale and _gesture(view) == gesture, "a turn must not zoom or spend the budget"
        assert window._region == (41, 39, 44, 44)
        wheeled = (_tiles(window), _units(window), window._region)
    finally:
        _close(window)

    window = _paste_window()
    try:
        window.on_region_transform(region_transform.ROTATE_CW)
        assert (_tiles(window), _units(window), window._region) == wheeled
    finally:
        _close(window)


def test_wheel_up_then_down_gives_back_the_plain_paste_and_one_undo_clears_it() -> None:
    window = _select_window()
    try:
        _setup(window)
        before = (_tiles(window), _units(window))
        depth = len(window.edit_history.records)
        _paste(window)
        plain = (_tiles(window), _units(window))
        _bind(window, plain="rotate")
        _wheel_xy(window.map_view, (0, 120))
        assert (_tiles(window), _units(window)) != plain
        _wheel_xy(window.map_view, (0, -120))
        assert _turn(window) == "id"
        assert (_tiles(window), _units(window)) == plain
        assert len(window.edit_history.records) == depth + 1
        window.undo()
        assert (_tiles(window), _units(window)) == before
    finally:
        _close(window)


def test_the_wheel_turn_goes_through_on_region_transform_once_per_notch(monkeypatch) -> None:
    """The hard rule: the wheel writes nothing itself; the Rotate keys' path does."""
    from descape import region_transform

    window = _paste_window()
    try:
        calls: list[str] = []
        monkeypatch.setattr(window, "on_region_transform", calls.append)
        _bind(window, plain="rotate")
        _wheel_xy(window.map_view, (0, 240))
        _wheel_xy(window.map_view, (0, -120))
        assert calls == [region_transform.ROTATE_CW] * 2 + [region_transform.ROTATE_CCW]
    finally:
        _close(window)


@pytest.mark.parametrize("busy", ["_stroke_in_progress", "region_drag_live"])
def test_rotate_mid_stroke_or_mid_drag_does_nothing_and_leaves_the_paste(monkeypatch, busy) -> None:
    """on_region_transform() refuses then, so the paste has no turn to take
    and the wheel does nothing, as any tool action with nothing to act on."""
    window = _paste_window()
    try:
        target = window if busy == "_stroke_in_progress" else window.map_view
        monkeypatch.setattr(target, busy, lambda: True)
        _bind(window, plain="rotate")
        tiles, scale = _tiles(window), _scale(window.map_view)
        _wheel_xy(window.map_view, (0, 120))
        assert _turn(window) == "id" and _tiles(window) == tiles
        assert _scale(window.map_view) == scale
    finally:
        _close(window)


def test_tool_in_draw_with_a_paste_armed_steps_the_brush_and_leaves_the_paste() -> None:
    window = _paste_window("draw")
    try:
        _bind(window, plain="tool")
        tiles, size = _tiles(window), window.brush_size_spin.value()
        _wheel_xy(window.map_view, (0, 120))
        assert window.brush_size_spin.value() == size + 1
        assert _turn(window) == "id" and _tiles(window) == tiles
    finally:
        _close(window)


def test_tool_in_select_with_a_paste_armed_turns_it() -> None:
    from descape import region_transform

    window = _paste_window()
    try:
        assert not window.brush_size_spin.isEnabled()
        _bind(window, plain="tool")
        size, scale = window.brush_size_spin.value(), _scale(window.map_view)
        _wheel_xy(window.map_view, (0, 120))
        assert _turn(window) == region_transform.ROTATE_CW
        assert window.brush_size_spin.value() == size and _scale(window.map_view) == scale
    finally:
        _close(window)


def test_rotate_in_draw_with_a_paste_armed_turns_it_and_leaves_the_brush() -> None:
    """The documented disagreement: under Rotate preview the paste wins over the brush."""
    from descape import region_transform

    window = _paste_window("draw")
    try:
        _bind(window, plain="rotate")
        size = window.brush_size_spin.value()
        _wheel_xy(window.map_view, (0, 120))
        assert _turn(window) == region_transform.ROTATE_CW
        assert window.brush_size_spin.value() == size
    finally:
        _close(window)


@pytest.mark.parametrize("tool", ["draw", "select"])
def test_brush_size_with_a_paste_armed_never_turns_it(tool) -> None:
    window = _paste_window(tool)
    try:
        view = window.map_view
        _bind(window, plain="brush_size")
        tiles, size, scale = _tiles(window), window.brush_size_spin.value(), _scale(view)
        _wheel_xy(view, (0, 120))
        assert _turn(window) == "id" and _tiles(window) == tiles
        assert _scale(view) == scale
        assert window.brush_size_spin.value() == (size + 1 if tool == "draw" else size)
    finally:
        _close(window)


def test_two_half_notches_make_one_paste_turn_and_a_reversal_drops_the_stale_half() -> None:
    from descape import region_transform

    window = _paste_window()
    try:
        _bind(window, plain="rotate")
        view = window.map_view
        _wheel_xy(view, (0, 60))
        assert _turn(window) == "id"
        _wheel_xy(view, (0, 60))
        assert _turn(window) == region_transform.ROTATE_CW
        _wheel_xy(view, (0, 60))
        _wheel_xy(view, (0, -120))
        assert _turn(window) == "id", "+60 then -120 is one turn back"
        _wheel_xy(view, (0, -60))
        assert _turn(window) == "id", "the reversal banked the stale half notch"
    finally:
        _close(window)


def test_a_tool_switch_drops_a_leftover_paste_half_notch() -> None:
    from descape import region_transform

    window = _paste_window()
    try:
        _bind(window, plain="rotate")
        view = window.map_view
        _wheel_xy(view, (0, 60))
        window._on_tool_selected("pan")  # _select_window() left the Pan action checked
        assert window._current_tool == "pan" and window._region_turn_target()
        _wheel_xy(view, (0, 60))
        assert _turn(window) == "id", "the first half notch must not have survived"
        _wheel_xy(view, (0, 60))
        assert _turn(window) == region_transform.ROTATE_CW
    finally:
        _close(window)


def test_rotate_with_a_units_selection_and_a_paste_armed_does_nothing_and_edits_no_unit() -> None:
    """_region_turn_target()'s refusal: a Units-mode selection keeps Rotate for itself."""
    window = _select_window()
    try:
        (loner,) = _add(window, [(1, 4, 80.5, 80.5, 0.0, None)])  # an archer, ANGLE
        _setup(window)
        _paste(window)
        window.mode_combo.setCurrentText("Units")
        window._selection = [(1, loner.reference_id)]
        window._refresh_selection_view()
        assert window._paste_armed() and not window._region_turn_target()
        _bind(window, plain="rotate", ctrl="tool")
        view = window.map_view
        everything = _units(window, 0, 0, 120, 120)
        tiles, scale = _tiles(window), _scale(view)
        _wheel_xy(view, (0, 120))
        _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert _scale(view) == scale
        assert _turn(window) == "id" and _tiles(window) == tiles
        assert _units(window, 0, 0, 120, 120) == everything
        assert [float(u.rotation) for u in _live(window, 4) if (u.x, u.y) == (80.5, 80.5)] == [0.0]
    finally:
        _close(window)


@pytest.mark.parametrize("action", ["tool", "rotate"])
def test_place_units_ghost_wins_over_an_armed_paste(action) -> None:
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    window = _select_window()
    try:
        _setup(window)
        _paste(window)
        tiles = _tiles(window)
        window.mode_combo.setCurrentText("Units")
        window.units_panel._set_pending_object(_PLACE_CONST)
        window.units_panel.select_owner(1)
        window.place_unit_action.setChecked(True)
        view = window.map_view
        center = QPointF(view.viewport().rect().center())
        view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, center, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        assert view._unit_ghost_item is not None and window._paste_armed()
        _bind(window, plain=action)
        _wheel_xy(view, (0, 120))
        assert _facing(window) == 1
        assert _turn(window) == "id" and _tiles(window) == tiles
    finally:
        _close(window)


# --- where each action applies (GH #245) ----------------------------------------------


def test_every_wheel_dropdown_item_says_where_its_action_applies() -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    assert set(settings.WHEEL_ACTION_HINTS) == set(settings.WHEEL_ACTIONS)
    for action in ("tool", "brush_size", "rotate"):
        assert settings.WHEEL_ACTION_HINTS[action].endswith("Does nothing anywhere else."), action
    assert "Brush size box" in settings.WHEEL_ACTION_HINTS["brush_size"]
    for action in ("tool", "rotate"):
        hint = settings.WHEEL_ACTION_HINTS[action]
        assert "Place Unit's preview" in hint and "just-pasted region" in hint, action
    dialog, window = conftest.dialog_and_window()
    try:
        for slot, combo in dialog._wheel_combos.items():
            for i in range(combo.count()):
                tip = combo.itemData(i, Qt.ToolTipRole)
                assert tip, f"{slot}: {combo.itemText(i)!r} has no tooltip"
                assert tip == settings.WHEEL_ACTION_HINTS[combo.itemData(i)]
        assert dialog._wheel_note.text().endswith("Brush size and Rotate preview do nothing in a tool they don't apply to.")
    finally:
        dialog.close()
        conftest.close_window(window)


# --- GH #280: the wheel's rotate binding turns a live Move drag ----------------


def _live_drag():
    """(window, unit): an Archer on open ground, dragged past the threshold, button held."""
    window = drag_preview._window()
    (unit,) = drag_preview._add(window, drag_preview._ARCHER, [(60, 60)])
    drag_preview._start(window, (60, 60), (66, 66))
    assert window.map_view.unit_drag_live()
    return window, unit


@pytest.mark.parametrize("action", ["rotate", "tool"])
def test_a_rotate_or_tool_slot_turns_a_live_drag_and_spends_no_zoom_state(action) -> None:
    window, unit = _live_drag()
    try:
        view = window.map_view
        _bind(window, ctrl=action)
        scale, gesture = _scale(view), _gesture(view)
        event = _wheel_xy(view, (0, 120), _mods("ControlModifier"))
        assert event.isAccepted()
        assert window._drag_turn.facing == 1, "scroll up is clockwise, as under Place Unit"
        assert unit.rotation == 0.0, "pending, as a key"
        assert (_scale(view), _gesture(view)) == (scale, gesture)
        drag_preview._release(window, (66, 66))
        assert unit.rotation == pytest.approx(drag_preview._TURN / 16)
    finally:
        _close(window)


def test_a_half_notch_accumulates_on_a_live_drag() -> None:
    window, _unit = _live_drag()
    try:
        _bind(window, plain="rotate")
        _wheel_xy(window.map_view, (0, 60))
        assert not window._drag_turn
        _wheel_xy(window.map_view, (0, 60))
        assert window._drag_turn.facing == 1
    finally:
        drag_preview._escape(window)
        _close(window)


def test_a_notch_mid_drag_goes_through_on_unit_rotate(monkeypatch) -> None:
    """The hard rule: one router for the keys and the wheel, and it writes nothing itself."""
    window, _unit = _live_drag()
    try:
        turns: list = []
        monkeypatch.setattr(window, "_turn_unit_drag", turns.append)
        _bind(window, plain="rotate")
        _wheel_xy(window.map_view, (0, 120))
        _wheel_xy(window.map_view, (0, -240))
        assert [turn.facing for turn in turns] == [1, -2]
    finally:
        drag_preview._escape(window)
        _close(window)


def test_plain_wheel_still_zooms_mid_drag_and_a_turn_after_it_rereads_the_cursor() -> None:
    from PyQt5.QtCore import QEvent, Qt

    from descape import render

    window, unit = _live_drag()
    try:
        view = window.map_view
        scale = _scale(view)
        pixel = view._unit_drag_hover[0]
        _wheel_xy(view, (0, 120))
        assert _scale(view) > scale
        assert not window._drag_turn

        drag_preview._cw(window)
        ghost = view._unit_ghost_item.sceneBoundingRect()
        # The release reads the same viewport pixel through the new transform.
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, pixel, Qt.LeftButton, Qt.NoButton))
        mm = window.scenario.map_manager
        x0, _x1, y0, _y1 = render.unit_tile_bounds(unit, mm.map_width, mm.map_height)
        tp = view._tile_pixels
        assert (ghost.x(), ghost.y()) == (x0 * tp, y0 * tp)
    finally:
        _close(window)


def test_a_rotate_slot_under_move_with_no_live_drag_stays_none() -> None:
    from PyQt5.QtCore import QEvent, Qt

    window = drag_preview._window()
    try:
        view = window.map_view
        _bind(window, ctrl="rotate")
        drag_preview._add(window, drag_preview._ARCHER, [(60, 60)])
        ctrl = _mods("ControlModifier")
        assert view._wheel_action(ctrl) == "none"
        press = drag_preview._viewport_pos(window, 60, 60)
        view.mousePressEvent(conftest.mouse_event(QEvent.MouseButtonPress, press, Qt.LeftButton, Qt.LeftButton))
        assert view._wheel_action(ctrl) == "none", "a held press that never moved is not a move"
        view.mouseReleaseEvent(conftest.mouse_event(QEvent.MouseButtonRelease, press, Qt.LeftButton, Qt.NoButton))
    finally:
        _close(window)
