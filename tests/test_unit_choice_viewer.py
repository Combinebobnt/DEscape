"""GH #186 end to end: a pick in the inspector's Capture Flag or Unit State
combo, through ViewerWindow._on_unit_field_changed and UnitEditModel, to one
undo record and the saved file.

Two real fixtures, because the two fields differ by file version: Unit State
is in every one, Capture Flag is stored from 1.59 only. units_120x120 is 1.58
(Capture Flag disabled), v159_units_triggers is 1.59.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import unit_fields
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import capture_flag_of

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

V158 = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
V159 = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"

_CAPTURE = unit_fields.FIELDS_BY_ID["capture_flag"]
_STATE = unit_fields.FIELDS_BY_ID["status"]

# 1.59 fixture: (player, reference_id) with each unit's stored capture flag.
_GAIA_ONCE = (0, 100)  # 1
_P1_ARCHER = (1, 200)  # 2
_P1_VILLAGER = (1, 201)  # -1
_P2_VILLAGER = (2, 301)  # 3
# 1.58 fixture: a House holding villager 203, and a loose archer.
_HOUSE_158 = (1, 200)
_ARCHER_158 = (1, 201)
_GARRISONED_158 = 203


def _window(path: Path):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.load_scenario(path)
    assert window.scenario is not None, "fixture failed to load"
    window.mode_combo.setCurrentText("Units")
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def _select(window, *keys) -> list:
    window._selection = list(keys)
    window._refresh_selection_view()
    entries = [window.map_view._unit_index.entry_for_key(key) for key in keys]
    assert None not in entries
    return [entry.unit for entry in entries]


def _combo(window, field_id: str):
    return window.units_panel.unit_field_editors[field_id]


def _pick(window, field_id: str, value: int) -> None:
    """A user's pick: move the combo to the item holding `value`."""
    combo = _combo(window, field_id)
    index = combo.findData(value)
    assert index >= 0, f"{field_id} offers no item for {value}"
    combo.setCurrentIndex(index)


def _records(window) -> int:
    return len(window.edit_history.records)


def _saved_fields(window, tmp_path: Path) -> dict:
    out = tmp_path / "saved.aoe2scenario"
    write_scenario(window.scenario, out, backup=False, **window._edit_model_kwargs())
    reloaded = load_map_and_units(out)
    return {
        (player, unit.reference_id): (unit.status, capture_flag_of(unit))
        for player, units in enumerate(reloaded.unit_manager.units)
        for unit in units
    }


def _live_fields(window) -> dict:
    return {
        (player, unit.reference_id): (unit.status, capture_flag_of(unit))
        for player, units in enumerate(window.scenario.unit_manager.units)
        for unit in units
    }


# -- one unit ------------------------------------------------------------------


def test_the_inspector_shows_what_the_1_59_file_stores() -> None:
    window = _window(V159)
    try:
        _select(window, _GAIA_ONCE)
        assert _combo(window, "capture_flag").isEnabled()
        assert _combo(window, "capture_flag").currentText() == "Convert Once"
        assert _combo(window, "status").currentText() == "Alive"
        _select(window, _P2_VILLAGER)
        assert _combo(window, "capture_flag").currentText() == "Convert Never (Gaia Aggressive)"
        assert _records(window) == 0, "populating is not an edit"
        assert window.unit_edits is None, "populating must not build the edit model"
    finally:
        _close(window)


def test_capture_flag_round_trips_through_one_undo_record_and_a_save(tmp_path: Path) -> None:
    window = _window(V159)
    try:
        before = _live_fields(window)
        (villager,) = _select(window, _P1_VILLAGER)
        assert villager.capture_flag == -1

        _pick(window, "capture_flag", 1)

        assert _records(window) == 1
        assert window.edit_history.records[-1].label == "Set Capture Flag"
        assert villager.capture_flag == 1
        assert _combo(window, "capture_flag").currentText() == "Convert Once"
        expected = dict(before)
        expected[_P1_VILLAGER] = (2, 1)
        assert _saved_fields(window, tmp_path) == expected

        window.undo()
        assert villager.capture_flag == -1
        assert _combo(window, "capture_flag").currentText() == "Default"
        assert _saved_fields(window, tmp_path) == before

        window.redo()
        assert villager.capture_flag == 1
        assert _combo(window, "capture_flag").currentText() == "Convert Once"
    finally:
        _close(window)


@pytest.mark.parametrize(("path", "key"), [(V158, _ARCHER_158), (V159, _P1_ARCHER)], ids=["1.58", "1.59"])
def test_unit_state_round_trips_through_one_undo_record_and_a_save(path: Path, key, tmp_path: Path) -> None:
    window = _window(path)
    try:
        before = _live_fields(window)
        (archer,) = _select(window, key)
        assert archer.status == 2

        _pick(window, "status", 0)

        assert _records(window) == 1
        assert window.edit_history.records[-1].label == "Set Unit State"
        assert archer.status == 0
        assert _combo(window, "status").currentText() == "Foundation"
        expected = dict(before)
        expected[key] = (0, before[key][1])
        assert _saved_fields(window, tmp_path) == expected

        window.undo()
        assert archer.status == 2
        assert _combo(window, "status").currentText() == "Alive"
        assert _saved_fields(window, tmp_path) == before
    finally:
        _close(window)


def test_reporting_the_value_a_unit_already_has_records_nothing() -> None:
    window = _window(V159)
    try:
        (archer,) = _select(window, _P1_ARCHER)
        window._on_unit_field_changed(_CAPTURE, 2)
        window._on_unit_field_changed(_STATE, 2)
        assert _records(window) == 0
        assert (archer.status, archer.capture_flag) == (2, 2)
        assert window.unit_edits is None or not window.unit_edits.has_edits
    finally:
        _close(window)


def test_an_unlabelled_stored_value_is_kept_until_the_user_picks_another(tmp_path: Path) -> None:
    """"Other (N)" is selected at populate, so picking it again is no edit,
    and the raw value reaches the file untouched by an unrelated edit."""
    window = _window(V159)
    try:
        model = window._ensure_unit_edits()
        (archer,) = _select(window, _P1_ARCHER)
        with window._unit_edit(model, "Script edit", [1]):
            model.set_status(archer, 9)
        _select(window, _P1_ARCHER)
        combo = _combo(window, "status")
        assert (combo.currentText(), combo.currentData()) == ("Other (9)", 9)
        records = _records(window)

        window._on_unit_field_changed(_STATE, 9)
        assert _records(window) == records
        _pick(window, "capture_flag", 0)
        assert _records(window) == records + 1
        assert _combo(window, "status").currentText() == "Other (9)"
        assert _saved_fields(window, tmp_path)[_P1_ARCHER] == (9, 0)
    finally:
        _close(window)


# -- a group -------------------------------------------------------------------


def test_a_group_edit_is_one_undo_record(tmp_path: Path) -> None:
    window = _window(V159)
    try:
        before = _live_fields(window)
        archer, villager, gaia = _select(window, _P1_ARCHER, _P1_VILLAGER, _GAIA_ONCE)
        assert _combo(window, "capture_flag").currentText() == "(mixed)"

        _pick(window, "capture_flag", 0)

        assert _records(window) == 1
        assert window.edit_history.records[-1].label == "Set Capture Flag on 3 units"
        assert (archer.capture_flag, villager.capture_flag, gaia.capture_flag) == (0, 0, 0)
        combo = _combo(window, "capture_flag")
        assert combo.currentText() == "Convert Never"
        assert combo.findText("(mixed)") < 0
        expected = dict(before)
        for key in (_P1_ARCHER, _P1_VILLAGER, _GAIA_ONCE):
            expected[key] = (2, 0)
        assert _saved_fields(window, tmp_path) == expected

        window.undo()
        assert (archer.capture_flag, villager.capture_flag, gaia.capture_flag) == (2, -1, 1)
        assert _records(window) == 1 and not window.edit_history.can_undo
        assert _combo(window, "capture_flag").currentText() == "(mixed)"
        assert _saved_fields(window, tmp_path) == before
    finally:
        _close(window)


def test_a_group_edit_writes_only_the_members_that_differ() -> None:
    window = _window(V159)
    try:
        model = window._ensure_unit_edits()
        archer, villager = _select(window, _P1_ARCHER, _P1_VILLAGER)

        _pick(window, "capture_flag", 2)  # the archer's own value

        assert window.edit_history.records[-1].label == "Set Capture Flag on 1 units"
        assert (archer.capture_flag, villager.capture_flag) == (2, 2)
        dirty = [index for index, blob in enumerate(model._blobs[1]) if blob is None]
        assert dirty == [window.scenario.unit_manager.units[1].index(villager)]
    finally:
        _close(window)


def test_a_group_unit_state_edit_is_one_undo_record() -> None:
    window = _window(V158)
    try:
        house, archer = _select(window, _HOUSE_158, _ARCHER_158)

        _pick(window, "status", 7)

        assert _records(window) == 1
        assert window.edit_history.records[-1].label == "Set Unit State on 2 units"
        assert (house.status, archer.status) == (7, 7)
        window.undo()
        assert (house.status, archer.status) == (2, 2)
    finally:
        _close(window)


def test_a_host_s_unit_state_does_not_carry_to_its_garrison() -> None:
    """Unlike a typed coordinate (GH #42), the setting applies to exactly
    the selected units."""
    window = _window(V158)
    try:
        (house,) = _select(window, _HOUSE_158)
        inside = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == _GARRISONED_158)
        assert inside.garrisoned_in_id == house.reference_id

        _pick(window, "status", 0)

        assert house.status == 0
        assert inside.status == 2
    finally:
        _close(window)


# -- before 1.59 ---------------------------------------------------------------


def test_capture_flag_is_disabled_on_a_pre_1_59_file() -> None:
    window = _window(V158)
    try:
        _select(window, _ARCHER_158)
        combo = _combo(window, "capture_flag")
        assert not combo.isEnabled()
        assert combo.currentText() == "Default"
        assert "1.59" in combo.toolTip()
        assert _combo(window, "status").isEnabled()
        # After an edit depoisons the classes the unit has no such attribute at all.
        _pick(window, "status", 3)
        assert not _combo(window, "capture_flag").isEnabled()
        assert _combo(window, "capture_flag").currentText() == "Default"
    finally:
        _close(window)


def test_a_capture_flag_report_on_a_pre_1_59_file_writes_nothing() -> None:
    """The row is disabled there, so this is belt and braces: a report that
    arrives anyway must not reach set_capture_flag(), which raises."""
    window = _window(V158)
    try:
        (archer,) = _select(window, _ARCHER_158)
        window._on_unit_field_changed(_CAPTURE, 1)
        assert _records(window) == 0
        assert "capture_flag" not in vars(archer)
        assert not window.unit_edits.has_edits
    finally:
        _close(window)


def test_opening_a_1_59_file_after_a_1_58_one_enables_the_row() -> None:
    window = _window(V158)
    try:
        _select(window, _ARCHER_158)
        assert not _combo(window, "capture_flag").isEnabled()
        window.edit_history.mark_saved()
        window.load_scenario(V159)
        window.mode_combo.setCurrentText("Units")
        _select(window, _GAIA_ONCE)
        combo = _combo(window, "capture_flag")
        assert combo.isEnabled()
        assert combo.currentText() == "Convert Once"
        assert "1.59" not in combo.toolTip()
    finally:
        _close(window)


# -- the mouse wheel (TASK-344) ------------------------------------------------


def _wheel(combo) -> None:
    """One notch towards an item the combo has, delivered as Qt would."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    units = -120 if combo.currentIndex() < combo.count() - 1 else 120
    centre = QPointF(combo.rect().center())
    event = QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, units), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(combo, event)


@pytest.mark.parametrize(
    ("path", "key", "field_id"),
    [(V158, _ARCHER_158, "status"), (V159, _P1_ARCHER, "capture_flag"), (V158, _ARCHER_158, "player")],
    ids=["unit-state", "capture-flag", "owner"],
)
def test_the_wheel_edits_an_inspector_combo_only_once_it_has_focus(path: Path, key, field_id: str) -> None:
    """A wheel notch over the inspector, which scrolls, must not edit the selection by passing over a combo."""
    from PyQt5.QtWidgets import QApplication

    window = _window(path)
    try:
        window.show()
        QApplication.setActiveWindow(window)
        _select(window, key)
        combo = _combo(window, field_id)
        assert combo.isEnabled()
        window.map_view.setFocus()
        QApplication.processEvents()
        assert not combo.hasFocus()
        before = combo.currentData()

        _wheel(combo)
        assert combo.currentData() == before
        assert _records(window) == 0
        assert not combo.hasFocus(), "the wheel must not hand the combo focus either"

        combo.setFocus()
        QApplication.processEvents()
        assert combo.hasFocus(), "offscreen focus failed -- the control below would pass vacuously"
        _wheel(combo)
        assert combo.currentData() != before
        assert _records(window) == 1
    finally:
        _close(window)
