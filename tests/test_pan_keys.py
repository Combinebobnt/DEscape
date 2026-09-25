"""Held-key smooth pan (view_pan_*), and its arbitration with Units mode's
arrow-key nudge.

Key events are synthesized straight at MapView.keyPressEvent/keyReleaseEvent
the way test_ruler_viewer.py does, rather than through QTest.keyClick: these
four ids deliberately never reach a QAction, so there is no shortcut for Qt to
dispatch and a real key press would only arrive via the widget anyway.

_pan_step(dt) is driven directly wherever motion is asserted. The timer's own
dt comes from a QElapsedTimer, which an offscreen test cannot make
deterministic; _pan_timer.isActive() is what the timer itself is checked on.

The ownership tests at the bottom need a real scenario-loaded window in Units
mode, so they borrow test_unit_edit_viewer.py's fixture and Flat-style setup
rather than the bare ViewerWindow the pan tests use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    return ViewerWindow()


def _units_window():
    """A loaded scenario in Units mode -- see test_unit_edit_viewer.py's
    _window() for why Isometric is unchecked before the style switch."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(FIXTURE_PATH)
    assert window.scenario is not None, "fixture failed to load"
    window.iso_action.setChecked(False)
    window.terrain_style_combo.setCurrentText("Flat")
    window.mode_combo.setCurrentText("Units")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _press(view, key, modifiers=None, autorep=False) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    if modifiers is None:
        modifiers = Qt.NoModifier
    view.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, modifiers, "", autorep))


def _release(view, key, modifiers=None, autorep=False) -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent

    if modifiers is None:
        modifiers = Qt.NoModifier
    view.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, key, modifiers, "", autorep))


def _room_to_scroll(view) -> None:
    """Gives both scrollbars a range with the handle parked in the middle, so
    a pan in any direction has somewhere to go. A bare MapView has no scene
    rect of its own, which would leave every value clamped at 0 and make a
    'did it move' assertion vacuous."""
    for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
        bar.setRange(0, 2000)
        bar.setValue(1000)


# --- the hold cycle ----------------------------------------------------------


def test_a_bound_press_starts_the_timer_and_a_release_stops_it() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        before = view.horizontalScrollBar().value()

        _press(view, Qt.Key_Right)
        assert view._pan_timer.isActive()

        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() > before

        _release(view, Qt.Key_Right)
        assert not view._pan_timer.isActive()
    finally:
        _close(window)


def test_auto_repeat_neither_restarts_nor_stops_a_hold() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        _press(view, Qt.Key_Left)
        held = dict(view._pan_held)

        _release(view, Qt.Key_Left, autorep=True)
        assert view._pan_timer.isActive(), "an auto-repeat release must not end the hold"
        _press(view, Qt.Key_Left, autorep=True)
        assert view._pan_timer.isActive()
        assert dict(view._pan_held) == held
    finally:
        _close(window)


def test_both_keys_of_an_axis_pair_then_releasing_one_leaves_the_other_panning() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        _press(view, Qt.Key_Left)
        _press(view, Qt.Key_Right)
        before = view.horizontalScrollBar().value()
        # Opposed directions cancel while both are down.
        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() == before

        _release(view, Qt.Key_Left)
        assert view._pan_timer.isActive()
        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() > before
    finally:
        _close(window)


def test_a_diagonal_hold_advances_both_axes_and_a_speed_change_applies_next_tick() -> None:
    """GH #93: Right + Down held move both scrollbars on every tick, and the
    pan speed is read per tick, so a Settings change needs no re-press."""
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        h, v = view.horizontalScrollBar(), view.verticalScrollBar()
        settings.set_pan_speed(settings.PAN_SPEED_MIN)
        _press(view, Qt.Key_Right)
        _press(view, Qt.Key_Down)

        # 100 ms at 200 px/s is exactly 20 px per axis, no diagonal normalization.
        for _ in range(2):
            start = (h.value(), v.value())
            view._pan_step(100.0)
            assert (h.value(), v.value()) == (start[0] + 20, start[1] + 20)

        settings.set_pan_speed(settings.PAN_SPEED_MAX)
        start = (h.value(), v.value())
        view._pan_step(100.0)
        assert (h.value(), v.value()) == (start[0] + 200, start[1] + 200)
        assert view._pan_timer.isActive()
    finally:
        _close(window)


def test_focus_out_stops_a_held_pan() -> None:
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QFocusEvent

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        _press(view, Qt.Key_Down)
        assert view._pan_timer.isActive()

        view.focusOutEvent(QFocusEvent(QEvent.FocusOut))
        assert not view._pan_timer.isActive()
        assert view._pan_held == {}
    finally:
        _close(window)


def test_the_pan_stops_when_the_scrollbar_is_clamped() -> None:
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        view.horizontalScrollBar().setValue(view.horizontalScrollBar().maximum())
        _press(view, Qt.Key_Right)
        assert view._pan_timer.isActive()

        view._pan_step(100.0)
        assert not view._pan_timer.isActive(), "a clamped pan must not keep ticking forever"
    finally:
        _close(window)


def test_sub_pixel_ticks_accumulate_instead_of_truncating_to_zero() -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        settings.set_pan_speed(settings.PAN_SPEED_MIN)
        _press(view, Qt.Key_Right)
        before = view.horizontalScrollBar().value()

        # 1 ms at 200 px/s is 0.2 px: int() alone would discard every tick.
        view._pan_step(1.0)
        assert view.horizontalScrollBar().value() == before
        for _ in range(4):
            view._pan_step(1.0)
        assert view.horizontalScrollBar().value() > before
    finally:
        _close(window)


def test_middle_drag_suppresses_the_key_pan() -> None:
    """Two mechanisms, one pair of scrollbars."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        _press(view, Qt.Key_Right)
        before = view.horizontalScrollBar().value()

        view._middle_drag_active = True
        view._pan_step(100.0)
        assert view.horizontalScrollBar().value() == before
    finally:
        _close(window)


# --- binding ----------------------------------------------------------------


def test_rebinding_pan_to_w_leaves_a_bare_arrow_inert() -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        settings.set_keybind("view_pan_right", "W")
        window.apply_keybind("view_pan_right")

        before = view.horizontalScrollBar().value()
        _press(view, Qt.Key_Right)
        assert not view._pan_timer.isActive(), "the native single-step jump must stay suppressed"
        assert view.horizontalScrollBar().value() == before

        _press(view, Qt.Key_W)
        assert view._pan_timer.isActive()
        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() > before
    finally:
        _close(window)


def test_clearing_a_pan_binding_leaves_its_key_inert() -> None:
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        settings.set_keybind("view_pan_right", "")
        window.apply_keybind("view_pan_right")

        _press(view, Qt.Key_Right)
        assert not view._pan_timer.isActive()
    finally:
        _close(window)


def test_a_modifier_binding_matches_only_with_that_modifier_held() -> None:
    """QKeySequence packs the key WITH its modifier bits into one int, so an
    unmasked comparison against event.key() never matches a combo."""
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        settings.set_keybind("view_pan_up", "Ctrl+Up")
        window.apply_keybind("view_pan_up")

        _press(view, Qt.Key_Up)
        assert not view._pan_timer.isActive(), "a bare Up must not fire a Ctrl+Up binding"

        _press(view, Qt.Key_Up, Qt.ControlModifier)
        assert view._pan_timer.isActive()
    finally:
        _close(window)


def test_a_release_that_drops_the_modifier_first_still_ends_the_hold() -> None:
    """Letting go of Ctrl before Up delivers the Up release with the modifier
    already gone -- matching on (key, modifiers) would miss it and leave the
    timer running forever."""
    from PyQt5.QtCore import Qt

    from descape import settings

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        settings.set_keybind("view_pan_up", "Ctrl+Up")
        window.apply_keybind("view_pan_up")
        _press(view, Qt.Key_Up, Qt.ControlModifier)
        assert view._pan_timer.isActive()

        _release(view, Qt.Key_Up, Qt.NoModifier)
        assert not view._pan_timer.isActive()
    finally:
        _close(window)


def test_a_multi_chord_sequence_is_treated_as_unbound() -> None:
    """A chord cannot be held."""
    from descape import settings

    window = _window()
    try:
        view = window.map_view
        settings.set_keybind("view_pan_down", "Ctrl+K, Ctrl+D")
        window.apply_keybind("view_pan_down")

        assert not any(d == (0, 1) for d in view._pan_bindings.values())
    finally:
        _close(window)


# --- nudge vs pan ownership --------------------------------------------------


def test_an_arrow_nudges_in_units_mode_with_a_selection_and_never_pans() -> None:
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        entry = view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        view.set_unit_selection([entry])
        start_x = entry.unit.x
        before = view.horizontalScrollBar().value()

        _press(view, Qt.Key_Right)

        assert view._unit_index.entry_for_key(key).unit.x > start_x
        assert view.horizontalScrollBar().value() == before
        assert not view._pan_timer.isActive()
    finally:
        _close(window)


def test_an_arrow_pans_in_units_mode_with_nothing_selected() -> None:
    """A deliberate behaviour change: before the pan feature this path
    returned early and did nothing at all."""
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        window._selection = []
        view.set_unit_selection(None)
        before = view.horizontalScrollBar().value()

        _press(view, Qt.Key_Right)
        assert view._pan_timer.isActive()
        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() > before
    finally:
        _close(window)


def test_shift_arrow_with_a_selection_is_still_the_whole_tile_nudge() -> None:
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        entry = view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        view.set_unit_selection([entry])
        start_x = entry.unit.x

        _press(view, Qt.Key_Right, Qt.ShiftModifier)

        moved = view._unit_index.entry_for_key(key).unit.x - start_x
        assert moved == pytest.approx(1.0)
        assert not view._pan_timer.isActive()
    finally:
        _close(window)


def test_held_arrow_nudging_still_fires_once_per_auto_repeat() -> None:
    """The regression the conditional auto-repeat short-circuit exists to
    prevent: an unconditional 'isAutoRepeat -> return' would freeze a held
    nudge after its first press."""
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        entry = view._unit_index.entries[0]
        key = (entry.player_id, entry.unit.reference_id)
        window._selection = [key]
        view.set_unit_selection([entry])
        start_x = entry.unit.x

        _press(view, Qt.Key_Right)
        first = view._unit_index.entry_for_key(key).unit.x
        _press(view, Qt.Key_Right, autorep=True)
        second = view._unit_index.entry_for_key(key).unit.x

        assert first > start_x
        assert second > first
    finally:
        _close(window)


def test_a_release_for_a_nudged_arrow_is_a_silent_no_op() -> None:
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        entry = view._unit_index.entries[0]
        window._selection = [(entry.player_id, entry.unit.reference_id)]
        view.set_unit_selection([entry])

        _press(view, Qt.Key_Right)
        _release(view, Qt.Key_Right)
        assert view._pan_held == {}
    finally:
        _close(window)


def test_selecting_a_unit_mid_hold_ends_the_pan_rather_than_leaving_it_running() -> None:
    """Clicking a unit while an arrow is held moves ownership of that key
    from pan to nudge; the pan must end there, not at a release the pan path
    no longer owns."""
    from PyQt5.QtCore import Qt

    window = _units_window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        window._selection = []
        view.set_unit_selection(None)
        _press(view, Qt.Key_Right)
        assert view._pan_timer.isActive()

        entry = view._unit_index.entries[0]
        window._selection = [(entry.player_id, entry.unit.reference_id)]
        view.set_unit_selection([entry])
        _press(view, Qt.Key_Right, autorep=True)

        assert not view._pan_timer.isActive()
        assert view._pan_held == {}
    finally:
        _close(window)


def test_the_keypad_modifier_does_not_break_a_bound_arrow() -> None:
    """macOS sets Qt.KeypadModifier on the ARROW keys themselves, so an
    unmasked modifier comparison would leave the shipped defaults matching
    nothing there. Also covers a numpad arrow on any platform."""
    from PyQt5.QtCore import Qt

    window = _window()
    try:
        view = window.map_view
        _room_to_scroll(view)
        before = view.horizontalScrollBar().value()

        _press(view, Qt.Key_Right, Qt.KeypadModifier)
        assert view._pan_timer.isActive()
        view._pan_step(16.0)
        assert view.horizontalScrollBar().value() > before

        _release(view, Qt.Key_Right, Qt.KeypadModifier)
        assert not view._pan_timer.isActive()
    finally:
        _close(window)
