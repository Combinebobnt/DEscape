"""Units mode: the rotation dial's wheel, clicks and compass drags."""

from __future__ import annotations

import pytest
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import QApplication
from unit_edit_viewer_helpers import (
    _GATE_E,
    _GATE_NE,
    _GATE_SE,
    _REF_ARCHER_P1,
    _REF_ARCHER_P2,
    _REF_TREE_OAK,
    _REF_TREE_PINE,
    _REF_WALL,
    _close,
    _last_status_lines,
    _make_gate,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _select,
    _warnings,
    _window,
)

from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# --- TASK-548: Rotation on a group with mixed rotation ranges -----------------

_TREBUCHET = 42  # 32 directions; the fixture's archers have 16


_TWO_DIRECTIONS = 2238


_TWO_VARIANTS = 454


_GATE_N = 667


def _rotation_spin(window):
    return window.units_panel.unit_field_editors["rotation"]


def _dial(window):
    return window.units_panel.unit_rotation_dial


def _wheel(widget, units: int, modifiers=Qt.NoModifier) -> None:
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QWheelEvent

    centre = QPointF(widget.rect().center())
    event = QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, units), Qt.NoButton, modifiers, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def _mouse(widget, kind: str, point, buttons=Qt.LeftButton) -> None:
    """A real QMouseEvent: QTest.mouseMove carries no held button on a QWidget."""
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent

    event_type = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}[kind]
    button = Qt.NoButton if kind == "move" else Qt.LeftButton
    local = QPointF(point)
    event = QMouseEvent(event_type, local, local, widget.mapToGlobal(point.toPoint()), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def _click_spoke(window, spoke: int) -> None:
    dial = _dial(window)
    _mouse(dial, "press", dial.spoke_point(spoke))
    _mouse(dial, "release", dial.spoke_point(spoke), Qt.NoButton)


def _archer_and_trebuchet(window, archer_rotation: float = 0.0, trebuchet_rotation: float = 0.0):
    """The fixture's two archers, the second turned into a trebuchet, selected in that order."""
    archer, trebuchet = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)
    trebuchet.unit.unit_const = _TREBUCHET
    archer.unit.rotation, trebuchet.unit.rotation = archer_rotation, trebuchet_rotation
    window._after_unit_mutation()
    return _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2)


def _facing(unit) -> int:
    from descape import unit_rotation

    return unit_rotation.rotation_to_facing(unit.rotation, unit_rotation.angle_count_for(unit.unit_const))


def _two_gates(window, first=(10.0, 5.5), second=(_GATE_E, 20.0, 20.0)):
    """Gates at different orientations: the wall at `first`, the P1 archer as `second`."""
    _make_gate(window, *first)
    (archer,) = _select(window, _REF_ARCHER_P1)
    archer.unit.unit_const, archer.unit.x, archer.unit.y = second
    window._after_unit_mutation()
    return _select(window, _REF_WALL, _REF_ARCHER_P1)


@pytest.mark.parametrize("via_dial", [False, True], ids=["spinbox", "dial"])
def test_a_mixed_range_group_wheels_a_whole_turn_on_the_finest_grid(via_dial: bool) -> None:
    """GH #71: from 0 the group read "(mixed)", 0, "(mixed)", 0 and never got further."""
    from descape import unit_rotation

    window = _window()
    try:
        archer, trebuchet = _archer_and_trebuchet(window)
        spin = _rotation_spin(window)
        target = _dial(window) if via_dial else spin
        records_before = len(window.edit_history.records)
        read = []
        for notch in range(1, 34):
            _wheel(target, 120)
            read.append(spin.text())
            assert _facing(trebuchet.unit) == notch % 32
            assert _facing(archer.unit) == unit_rotation.snap_facing(notch, 32, 16)
        assert read == [str(notch % 32) for notch in range(1, 34)]
        assert len(window.edit_history.records) == records_before + 33
    finally:
        _close(window)


def test_a_mixed_range_group_wheels_down_through_the_wrap() -> None:
    window = _window()
    try:
        _archer_and_trebuchet(window)
        spin = _rotation_spin(window)
        records_before = len(window.edit_history.records)
        read = []
        for _ in range(3):
            _wheel(spin, -120)
            read.append(spin.text())
        assert read == ["31", "30", "29"]
        assert len(window.edit_history.records) == records_before + 3
    finally:
        _close(window)


@pytest.mark.parametrize("spoke", [5, 7])
def test_a_dial_click_on_an_odd_spoke_keeps_the_pointer(spoke: int) -> None:
    window = _window()
    try:
        _archer_and_trebuchet(window)
        _click_spoke(window, spoke)
        assert _rotation_spin(window).text() == str(spoke)
        assert _dial(window).value == spoke
    finally:
        _close(window)


def test_a_notch_on_an_unaligned_group_keeps_every_offset_and_skips_the_tree() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        archer_at_9 = unit_rotation.facing_to_rotation(9, 16)
        archer, trebuchet = _archer_and_trebuchet(window, archer_at_9, unit_rotation.facing_to_rotation(2, 32))
        entries = _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_TREE_OAK)
        tree = next(e for e in entries if e.unit.reference_id == _REF_TREE_OAK)
        before = [e.unit.rotation for e in entries]
        records_before = len(window.edit_history.records)
        spin = _rotation_spin(window)
        assert spin.text() == "(mixed)"

        _wheel(spin, 120)

        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (3, 10)
        assert tree.unit.rotation == 7.0
        assert spin.text() == "(mixed)"
        assert len(window.edit_history.records) == records_before + 1
        assert window.edit_history.peek_undo().label == "Turn 2 units"
        assert _last_status_lines(window, 1) == ["Turned 2 units by +1 (1 skipped: not rotatable)"]

        _wheel(spin, 120)  # facings 3 and 4 of 32 snap to one frame of 16: only the trebuchet moves
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (4, 10)
        assert window.edit_history.peek_undo().label == "Turn 1 unit"

        window.undo()
        window.undo()
        assert [e.unit.rotation for e in entries] == before
    finally:
        _close(window)


def test_a_notch_across_the_wrap_turns_a_coarse_member_one_frame_up() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        archer, trebuchet = _archer_and_trebuchet(
            window, unit_rotation.facing_to_rotation(3, 16), unit_rotation.facing_to_rotation(30, 32)
        )
        _wheel(_rotation_spin(window), 120)  # the coarse member's turn is 0 - 15, one frame up
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (31, 4)
        _wheel(_rotation_spin(window), -120)
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (30, 3)
    finally:
        _close(window)


def test_a_notch_moves_an_off_grid_facing_along_as_rotate_does() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        archer, _trebuchet = _archer_and_trebuchet(window, 7.0, unit_rotation.facing_to_rotation(2, 32))
        assert _rotation_spin(window).text() == "(mixed)"
        _wheel(_rotation_spin(window), 120)
        assert archer.unit.rotation == pytest.approx(unit_rotation.rotate_step(7.0, 16, 1))
    finally:
        _close(window)


def test_a_notch_on_a_mixed_variant_group_steps_each_tree_in_one_record() -> None:
    window = _window()
    try:
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        entries[0].unit.rotation, entries[1].unit.rotation = 7.0, 9.0
        window._after_unit_mutation()
        entries = _select(window, _REF_TREE_OAK, _REF_TREE_PINE)
        spin = _rotation_spin(window)
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        _wheel(spin, 120)

        assert [e.unit.rotation for e in entries] == [8.0, 10.0]
        assert spin.text() == "(mixed)"
        assert len(window.edit_history.records) == records_before + 1
        window.undo()
        assert [e.unit.rotation for e in entries] == [7.0, 9.0]
    finally:
        _close(window)


def test_a_notch_on_a_mixed_gate_group_turns_each_gate_in_one_record() -> None:
    window = _window()
    try:
        entries = _two_gates(window)
        spin = _rotation_spin(window)
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        _wheel(spin, 120)

        assert [e.unit.unit_const for e in entries] == [_GATE_E, _GATE_SE]
        assert spin.text() == "(mixed)"
        assert len(window.edit_history.records) == records_before + 1
        window.undo()
        assert [e.unit.unit_const for e in entries] == [_GATE_NE, _GATE_E]
    finally:
        _close(window)


def test_a_notch_skips_a_gate_that_would_hang_off_the_map_and_says_so() -> None:
    window = _window()
    try:
        entries = _two_gates(window, first=(10.0, 117.5))
        records_before = len(window.edit_history.records)

        _wheel(_rotation_spin(window), 120)

        assert [e.unit.unit_const for e in entries] == [_GATE_NE, _GATE_SE]
        assert len(window.edit_history.records) == records_before + 1
        assert _last_status_lines(window, 1) == ["Set Rotation: 1 gate(s) skipped (would hang off the map edge)"]
    finally:
        _close(window)


def test_an_all_blocked_gate_group_adds_no_record_and_still_says_so() -> None:
    """(4, 1) off the bottom edge and (1, 4) off the right one: a notch fits neither."""
    window = _window()
    try:
        entries = _two_gates(window, first=(10.0, 117.5), second=(_GATE_N, 118.5, 22.0))
        before = [(e.unit.unit_const, e.unit.x, e.unit.y) for e in entries]
        records_before = len(window.edit_history.records)

        _wheel(_rotation_spin(window), 120)

        assert [(e.unit.unit_const, e.unit.x, e.unit.y) for e in entries] == before
        assert len(window.edit_history.records) == records_before
        assert _last_status_lines(window, 1) == ["Set Rotation: 2 gate(s) skipped (would hang off the map edge)"]
        assert _rotation_spin(window).text() == "(mixed)"
    finally:
        _close(window)


@pytest.mark.parametrize(
    ("refs", "const", "rotations"),
    [
        pytest.param((_REF_ARCHER_P1, _REF_ARCHER_P2), _TWO_DIRECTIONS, (0.0, 3.14), id="two-directions"),
        pytest.param((_REF_TREE_OAK, _REF_TREE_PINE), _TWO_VARIANTS, (0.0, 1.0), id="two-variants"),
    ],
)
def test_a_step_of_ten_on_a_group_with_two_frames_adds_no_record(refs, const: int, rotations) -> None:
    """Ctrl+wheel is a whole number of cycles here, and _unit_edit asserts on an empty splice list."""
    window = _window()
    try:
        entries = _select(window, *refs)
        for entry, rotation in zip(entries, rotations, strict=True):
            entry.unit.unit_const, entry.unit.rotation = const, rotation
        window._after_unit_mutation()
        entries = _select(window, *refs)
        spin = _rotation_spin(window)
        assert spin.text() == "(mixed)"
        records_before = len(window.edit_history.records)

        _wheel(spin, 120, Qt.ControlModifier)

        assert [e.unit.rotation for e in entries] == list(rotations)
        assert len(window.edit_history.records) == records_before
    finally:
        _close(window)


# --- TASK-548: the compass turns the unit as it moves -------------------------


def _press(window, spoke: int) -> None:
    _mouse(_dial(window), "press", _dial(window).spoke_point(spoke))


def _move(window, spoke: int) -> None:
    _mouse(_dial(window), "move", _dial(window).spoke_point(spoke))


def _release(window, spoke: int) -> None:
    _mouse(_dial(window), "release", _dial(window).spoke_point(spoke), Qt.NoButton)


def _one_archer(window, rotation: float = 0.0):
    (entry,) = _select(window, _REF_ARCHER_P1)
    entry.unit.rotation = rotation
    window._after_unit_mutation()
    return _select(window, _REF_ARCHER_P1)[0]


def test_a_compass_drag_turns_the_unit_at_each_crossing_and_is_one_undo_step() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        entry = _one_archer(window)
        history = window.edit_history
        records_before = len(history.records)

        _press(window, 2)
        assert _facing(entry.unit) == 2, "the press commits the spoke under the cursor"
        _move(window, 3)
        assert _facing(entry.unit) == 3, "the unit turns with the mouse still down"
        _move(window, 4)
        assert _facing(entry.unit) == 4
        assert len(history.records) == records_before + 1
        _release(window, 4)

        assert len(history.records) == records_before + 1
        assert _rotation_spin(window).value() == 4
        assert _last_status_lines(window, 1) == ["Set unit Rotation"]
        window.undo()
        assert entry.unit.rotation == 0.0
        assert not history.is_dirty
        window.redo()
        assert entry.unit.rotation == unit_rotation.facing_to_rotation(4, 16)
    finally:
        _close(window)


def test_a_drag_on_a_coarser_member_folds_every_key_any_spoke_touched() -> None:
    """Spokes 2 and 4 turn only the trebuchet (the archer's snap holds), spoke 3 both."""
    from descape import unit_rotation

    window = _window()
    try:
        archer, trebuchet = _archer_and_trebuchet(
            window, unit_rotation.facing_to_rotation(1, 16), unit_rotation.facing_to_rotation(1, 32)
        )
        start = (archer.unit.rotation, trebuchet.unit.rotation)
        assert _rotation_spin(window).text() == "1"
        records_before = len(window.edit_history.records)

        _press(window, 2)
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (2, 1)
        _move(window, 3)
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (3, 2)
        _move(window, 4)
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (4, 2)
        _release(window, 4)

        record = window.edit_history.peek_undo()
        assert len(window.edit_history.records) == records_before + 1
        assert len(record.unit_field_entries) == 2
        assert record.label == "Set rotation on 2 units", "counts the group, not the last spoke's one unit"
        window.undo()
        assert (archer.unit.rotation, trebuchet.unit.rotation) == start
        window.redo()
        assert (_facing(trebuchet.unit), _facing(archer.unit)) == (4, 2)
    finally:
        _close(window)


def test_a_drag_on_a_mixed_group_aligns_it_and_stays_one_step() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        archer, trebuchet = _archer_and_trebuchet(window, trebuchet_rotation=unit_rotation.facing_to_rotation(9, 32))
        start = (archer.unit.rotation, trebuchet.unit.rotation)
        records_before = len(window.edit_history.records)
        assert _rotation_spin(window).text() == "(mixed)"

        _press(window, 4)
        _move(window, 6)
        _release(window, 6)

        assert _rotation_spin(window).text() == "6"
        assert len(window.edit_history.records) == records_before + 1
        window.undo()
        assert (archer.unit.rotation, trebuchet.unit.rotation) == start
        assert _rotation_spin(window).text() == "(mixed)"
    finally:
        _close(window)


def test_a_drag_with_a_tree_in_the_group_logs_one_line_with_the_skip() -> None:
    window = _window()
    try:
        _archer_and_trebuchet(window)
        _select(window, _REF_ARCHER_P1, _REF_ARCHER_P2, _REF_TREE_OAK)
        lines_before = len(window.status_log.toPlainText().splitlines())

        _press(window, 2)
        _move(window, 3)
        assert len(window.status_log.toPlainText().splitlines()) == lines_before, "no line per spoke"
        _release(window, 3)

        assert window.status_log.toPlainText().splitlines()[lines_before:] == [
            "Set rotation on 2 units (1 skipped: not rotatable)"
        ]
    finally:
        _close(window)


@pytest.mark.parametrize("rotation", [0.0, 7.0], ids=["on-grid", "off-grid"])
def test_a_drag_back_to_the_start_leaves_no_record_and_the_stored_value(rotation: float) -> None:
    window = _window()
    try:
        entry = _one_archer(window, rotation)
        start = _dial(window).value
        history = window.edit_history
        records_before = len(history.records)

        _press(window, (start + 3) % 16)
        assert entry.unit.rotation != rotation
        _move(window, start)
        _release(window, start)

        assert entry.unit.rotation == rotation
        assert len(history.records) == records_before
        assert not history.can_redo
        assert not history.is_dirty
        assert _rotation_spin(window).value() == start
        assert not window.undo_action.isEnabled()
    finally:
        _close(window)


def test_a_drag_back_to_the_start_at_the_record_cap_undoes_its_own_record() -> None:
    """The cursor stays flat at the cap, so a cursor comparison would miss this."""
    from descape.edit_history import DEFAULT_MAX_RECORDS

    window = _window()
    try:
        entry = _one_archer(window, 7.0)
        history = window.edit_history
        for _ in range(DEFAULT_MAX_RECORDS):
            window.on_unit_rotate(1)
        assert len(history.records) == history.max_records
        history.mark_saved()
        before_drag, rotation = history.peek_undo(), entry.unit.rotation
        start = _dial(window).value

        _press(window, (start + 3) % 16)
        _move(window, start)
        _release(window, start)

        assert history.peek_undo() is before_drag
        assert entry.unit.rotation == rotation
        assert not history.can_redo
        assert not history.is_dirty
    finally:
        _close(window)


def test_an_undo_during_a_drag_survives_the_release() -> None:
    window = _window()
    try:
        entry = _one_archer(window)
        records_before = len(window.edit_history.records)
        _press(window, 3)
        _move(window, 4)

        window.undo()
        assert entry.unit.rotation == 0.0
        _release(window, 4)

        assert entry.unit.rotation == 0.0
        assert _rotation_spin(window).value() == 0
        assert window.edit_history.can_redo, "the release must not act on an undone record"
        assert len(window.edit_history.records) == records_before + 1
    finally:
        _close(window)


def test_an_undo_during_a_drag_then_another_spoke_pushes_a_record_the_drag_follows() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        entry = _one_archer(window)
        records_before = len(window.edit_history.records)
        _press(window, 3)
        window.undo()
        _move(window, 5)
        _move(window, 6)
        _release(window, 6)

        assert len(window.edit_history.records) == records_before + 1
        assert entry.unit.rotation == unit_rotation.facing_to_rotation(6, 16)
        window.undo()
        assert entry.unit.rotation == 0.0
    finally:
        _close(window)


def test_a_save_during_a_drag_keeps_its_record_and_the_next_spokes_start_another() -> None:
    from descape import unit_rotation

    window = _window()
    try:
        entry = _one_archer(window)
        history = window.edit_history
        records_before = len(history.records)
        _press(window, 3)
        history.mark_saved()
        saved_record = history.peek_undo()
        _move(window, 5)
        assert history.peek_undo() is not saved_record
        assert history.is_dirty
        _move(window, 3)
        _release(window, 3)

        assert len(history.records) == records_before + 2
        window.undo()
        assert entry.unit.rotation == unit_rotation.facing_to_rotation(3, 16)
        assert not history.is_dirty, "back on the saved state"
        window.undo()
        assert entry.unit.rotation == 0.0
    finally:
        _close(window)


def test_a_save_during_a_drag_then_back_to_the_start_keeps_both_records() -> None:
    """The file on disk holds the spoke, so the return is a real edit, not a drag that never happened."""
    from descape import unit_rotation

    window = _window()
    try:
        entry = _one_archer(window)
        history = window.edit_history
        records_before = len(history.records)
        _press(window, 3)
        history.mark_saved()
        _move(window, 0)
        _release(window, 0)

        assert len(history.records) == records_before + 2
        assert entry.unit.rotation == 0.0
        assert history.is_dirty
        window.undo()
        assert entry.unit.rotation == unit_rotation.facing_to_rotation(3, 16)
    finally:
        _close(window)


def test_a_rotate_key_during_a_drag_stands_alone_and_the_release_keeps_it() -> None:
    window = _window()
    try:
        entry = _one_archer(window)
        records_before = len(window.edit_history.records)
        _press(window, 3)
        window.on_unit_rotate(1)
        assert _facing(entry.unit) == 4
        _release(window, 3)

        assert _facing(entry.unit) == 4, "the release's spoke is already committed"
        assert len(window.edit_history.records) == records_before + 2
    finally:
        _close(window)


def test_a_value_typed_while_a_drag_is_left_open_is_its_own_record() -> None:
    """A release lost to a modal leaves the drag open; only dial commits may read it."""
    window = _window()
    try:
        _one_archer(window)
        records_before = len(window.edit_history.records)
        _press(window, 3)
        _rotation_spin(window).setValue(7)
        assert len(window.edit_history.records) == records_before + 2
        assert window.edit_history.peek_undo().label == "Set unit Rotation"
    finally:
        _close(window)


def test_a_click_on_the_current_spoke_or_the_dead_zone_of_a_fresh_document_adds_nothing() -> None:
    window = _window()
    try:
        _one_archer(window)
        assert not window.edit_history.records
        _click_spoke(window, 0)
        dial = _dial(window)
        centre = QPointF(dial.width() / 2, dial.height() / 2)
        _mouse(dial, "press", centre)
        _mouse(dial, "release", centre, Qt.NoButton)
        assert not window.edit_history.records
        assert window._rotation_drag is None
    finally:
        _close(window)


def test_a_read_only_file_warns_once_and_the_drag_commits_nothing(monkeypatch) -> None:
    from descape.unit_model import UnitEditsUnavailableError

    def _refuse(_scenario):
        raise UnitEditsUnavailableError("synthetic refusal")

    window = _window()
    try:
        entry = _one_archer(window)
        set_viewer_global(monkeypatch, "UnitEditModel", _refuse)
        window.unit_edits = None
        warned = _warnings(window)

        _press(window, 3)
        _move(window, 4)
        _release(window, 4)

        assert len(warned) == 1
        assert entry.unit.rotation == 0.0
        assert _rotation_spin(window).value() == 0
        assert window._rotation_drag is None
        assert not window.edit_history.records
    finally:
        _close(window)


def test_a_selection_change_through_a_wall_closes_the_drag_as_hidden() -> None:
    """Needs a shown window: no Hide reaches the dial otherwise."""
    window = _window()
    try:
        window.show()
        QApplication.processEvents()
        _one_archer(window)
        dial = _dial(window)
        assert dial.isVisible()
        calls = []
        real = dial.on_drag
        dial.on_drag = lambda active, hidden=False: calls.append((active, hidden)) or real(active, hidden)
        _press(window, 3)
        kept = window.edit_history.peek_undo()

        _select(window, _REF_WALL)

        assert not dial.isVisible()
        assert calls == [(True, False), (False, True)]
        assert window._rotation_drag is None
        assert window.edit_history.peek_undo() is kept, "a hidden close leaves the record standing"
        _release(window, 5)
        assert window.edit_history.peek_undo() is kept

        (entry,) = _select(window, _REF_ARCHER_P1)
        records_before = len(window.edit_history.records)
        _rotation_spin(window).setValue(9)
        assert _facing(entry.unit) == 9
        assert len(window.edit_history.records) == records_before + 1
    finally:
        window.hide()
        _close(window)


def test_a_hidden_close_back_on_the_start_leaves_the_record_standing() -> None:
    """That close runs inside the panel's populate, where the back-to-start undo would nest."""
    window = _window()
    try:
        window.show()
        QApplication.processEvents()
        _one_archer(window)
        assert _dial(window).isVisible()
        _press(window, 3)
        _move(window, 0)
        kept = window.edit_history.peek_undo()

        _select(window, _REF_WALL)

        assert window._rotation_drag is None
        assert window.edit_history.peek_undo() is kept
        assert not window.edit_history.can_redo
    finally:
        window.hide()
        _close(window)
