"""GH #186's inspector rows: the Capture Flag and Unit State combos of
descape/units_panel.py, standalone like tests/test_units_panel.py (no
ViewerWindow, no loaded scenario).

The panel's own contract only: what a populate shows for a labelled value,
an unlabelled one and a mixed group, that a populate never reports as an
edit while a real pick does, and that Capture Flag is disabled in a file
that does not store it. The write path is tests/test_unit_choice_viewer.py's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import unit_fields
from descape.unit_pick import UnitEntry
from descape.units_panel import UnitsPanel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ARCHER_CONST = 4
_CAPTURE_LABELS = [label for _value, label in unit_fields.CAPTURE_FLAG_CHOICES]
_STATE_LABELS = [label for _value, label in unit_fields.UNIT_STATE_CHOICES]


@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1
    status: int = 2
    capture_flag: int = -1


@dataclass
class OldFakeUnit:
    """A fake from before GH #186, with neither field."""

    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0


def _entry(player_id: int = 1, reference_id: int = 101, **kwargs) -> UnitEntry:
    unit = SyntheticUnit(x=4.5, y=8.5, unit_const=_ARCHER_CONST, reference_id=reference_id, **kwargs)
    return UnitEntry(player_id=player_id, unit=unit, own_x=4, own_y=8, order=0)


def _panel(**kwargs):
    """A panel plus the list its on_unit_field callback appends (field_id, value) to."""
    conftest.ensure_qapp()
    reported: list = []
    panel = UnitsPanel(on_unit_field=lambda spec, value: reported.append((spec.field_id, value)), **kwargs)
    return panel, reported


def _items(combo) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def _capture(panel):
    return panel.unit_field_editors["capture_flag"]


def _state(panel):
    return panel.unit_field_editors["status"]


# -- one unit ------------------------------------------------------------------


def test_the_two_rows_follow_rotation_in_the_grid() -> None:
    panel, _reported = _panel()
    grid = panel.unit_inspector_grid.layout()
    rows = {
        field_id: grid.getItemPosition(grid.indexOf(panel.unit_field_captions[field_id]))[0]
        for field_id in ("rotation", "capture_flag", "status", "reference_id")
    }
    assert rows["capture_flag"] == rows["rotation"] + 1
    assert rows["status"] == rows["rotation"] + 2
    assert rows["reference_id"] == rows["rotation"] + 3
    assert panel.unit_field_captions["capture_flag"].text() == "Capture Flag:"
    assert panel.unit_field_captions["status"].text() == "Unit State:"


def test_show_unit_populates_both_combos_without_reporting_an_edit() -> None:
    panel, reported = _panel()
    panel.show_unit(_entry(status=0, capture_flag=1))
    assert (_state(panel).currentText(), _state(panel).currentData()) == ("Foundation", 0)
    assert (_capture(panel).currentText(), _capture(panel).currentData()) == ("Convert Once", 1)
    assert _items(_state(panel)) == _STATE_LABELS
    assert _items(_capture(panel)) == _CAPTURE_LABELS
    assert _state(panel).isVisibleTo(panel) and _capture(panel).isVisibleTo(panel)
    assert _state(panel).isEnabled() and _capture(panel).isEnabled()
    assert reported == []


def test_every_labelled_value_shows_its_own_label() -> None:
    panel, reported = _panel()
    for value, label in unit_fields.UNIT_STATE_CHOICES:
        panel.show_unit(_entry(status=value))
        assert (_state(panel).currentText(), _state(panel).currentData()) == (label, value)
    for value, label in unit_fields.CAPTURE_FLAG_CHOICES:
        panel.show_unit(_entry(capture_flag=value))
        assert (_capture(panel).currentText(), _capture(panel).currentData()) == (label, value)
    assert reported == []


def test_a_raw_7_shows_as_remove() -> None:
    """773 corpus units carry status 7: it is a labelled value, not "Other"."""
    panel, _reported = _panel()
    panel.show_unit(_entry(status=7))
    assert _state(panel).currentText() == "Remove"
    assert _items(_state(panel)) == _STATE_LABELS


def test_an_unlabelled_value_shows_as_other_and_survives_a_repopulate() -> None:
    panel, reported = _panel()
    entry = _entry(status=9, capture_flag=-3)
    for _ in range(3):
        panel.show_unit(entry)
        assert (_state(panel).currentText(), _state(panel).currentData()) == ("Other (9)", 9)
        assert _items(_state(panel)) == [*_STATE_LABELS, "Other (9)"]
        assert (_capture(panel).currentText(), _capture(panel).currentData()) == ("Other (-3)", -3)
        assert _items(_capture(panel)) == [*_CAPTURE_LABELS, "Other (-3)"]
    assert reported == []


def test_the_other_item_goes_when_the_next_unit_has_a_labelled_value() -> None:
    panel, reported = _panel()
    panel.show_unit(_entry(status=9))
    panel.show_unit(_entry(status=200))
    assert _items(_state(panel)) == [*_STATE_LABELS, "Other (200)"]
    panel.show_unit(_entry(status=3))
    assert _items(_state(panel)) == _STATE_LABELS
    assert _state(panel).currentText() == "Dead"
    assert reported == []


def test_a_unit_with_neither_field_shows_alive_and_default() -> None:
    """Older test fakes have no `status` and no `capture_flag`, and a
    pre-1.59 unit has no capture flag: both read as the field's default."""
    panel, reported = _panel()
    unit = OldFakeUnit(x=4.5, y=8.5, unit_const=_ARCHER_CONST, reference_id=7)
    panel.show_unit(UnitEntry(player_id=1, unit=unit, own_x=4, own_y=8, order=0))
    assert _state(panel).currentText() == "Alive"
    assert _capture(panel).currentText() == "Default"
    assert reported == []


def test_picking_a_value_reports_its_raw_int() -> None:
    panel, reported = _panel()
    panel.show_unit(_entry())
    _state(panel).setCurrentIndex(_state(panel).findData(3))
    _capture(panel).setCurrentIndex(_capture(panel).findData(2))
    assert reported == [("status", 3), ("capture_flag", 2)]
    assert all(type(value) is int for _field_id, value in reported)


def test_picking_default_on_one_unit_reports_minus_one() -> None:
    panel, reported = _panel()
    panel.show_unit(_entry(capture_flag=1))
    _capture(panel).setCurrentIndex(_capture(panel).findData(-1))
    assert reported == [("capture_flag", -1)]


def test_show_unit_none_strips_an_other_item() -> None:
    panel, reported = _panel()
    panel.show_unit(_entry(status=9))
    panel.show_unit(None)
    assert _items(_state(panel)) == _STATE_LABELS
    assert reported == []


# -- a group -------------------------------------------------------------------


def test_a_group_shows_the_common_value() -> None:
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, status=0, capture_flag=2), _entry(reference_id=2, status=0, capture_flag=2)])
    assert _state(panel).currentText() == "Foundation"
    assert _capture(panel).currentText() == "Convert Multiple Times"
    assert _items(_state(panel)) == _STATE_LABELS
    assert _items(_capture(panel)) == _CAPTURE_LABELS
    assert reported == []


def test_a_group_that_disagrees_shows_mixed() -> None:
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, status=2, capture_flag=0), _entry(reference_id=2, status=7, capture_flag=1)])
    for combo, labels in ((_state(panel), _STATE_LABELS), (_capture(panel), _CAPTURE_LABELS)):
        assert combo.currentText() == "(mixed)"
        assert combo.currentData() is None
        assert _items(combo) == ["(mixed)", *labels]
    assert reported == []


def test_only_the_field_that_disagrees_is_mixed() -> None:
    panel, _reported = _panel()
    panel.show_group([_entry(reference_id=1, status=2, capture_flag=0), _entry(reference_id=2, status=2, capture_flag=1)])
    assert _state(panel).currentText() == "Alive"
    assert _capture(panel).currentText() == "(mixed)"


def test_a_group_sharing_an_unlabelled_value_shows_other() -> None:
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, status=9), _entry(reference_id=2, status=9)])
    assert (_state(panel).currentText(), _state(panel).currentData()) == ("Other (9)", 9)
    assert _items(_state(panel)) == [*_STATE_LABELS, "Other (9)"]
    assert reported == []


def test_picking_default_in_a_mixed_group_reports_minus_one() -> None:
    """A CHOICE row is never in _mixed_sentinels: a FACING-style -1 sentinel
    would swallow Capture Flag's Default, whose raw value is -1."""
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, capture_flag=0), _entry(reference_id=2, capture_flag=1)])
    assert not {"capture_flag", "status"} & set(panel._mixed_sentinels)
    _capture(panel).setCurrentIndex(_capture(panel).findData(-1))
    assert reported == [("capture_flag", -1)]


def test_landing_on_mixed_again_reports_nothing() -> None:
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, status=2), _entry(reference_id=2, status=7)])
    combo = _state(panel)
    combo.setCurrentIndex(combo.findData(3))
    assert reported == [("status", 3)]
    combo.setCurrentIndex(0)
    assert combo.currentText() == "(mixed)"
    assert reported == [("status", 3)]


def test_repopulating_a_mixed_group_never_stacks_mixed_items() -> None:
    panel, reported = _panel()
    group = [_entry(reference_id=1, status=2), _entry(reference_id=2, status=7)]
    for _ in range(3):
        panel.show_group(group)
        assert _items(_state(panel)) == ["(mixed)", *_STATE_LABELS]
    panel.show_group([_entry(reference_id=1, status=7), _entry(reference_id=2, status=7)])
    assert _items(_state(panel)) == _STATE_LABELS
    assert _state(panel).currentText() == "Remove"
    assert reported == []


def test_leaving_group_mode_strips_mixed() -> None:
    panel, reported = _panel()
    panel.show_group([_entry(reference_id=1, status=2, capture_flag=0), _entry(reference_id=2, status=7, capture_flag=1)])
    panel.show_unit(_entry(status=7, capture_flag=1))
    assert _items(_state(panel)) == _STATE_LABELS
    assert _items(_capture(panel)) == _CAPTURE_LABELS
    assert (_state(panel).currentText(), _capture(panel).currentText()) == ("Remove", "Convert Once")
    panel.show_group([_entry(reference_id=1, status=2), _entry(reference_id=2, status=7)])
    panel.show_unit(None)
    assert _items(_state(panel)) == _STATE_LABELS
    assert reported == []


# -- the per-file gate ---------------------------------------------------------


def test_capture_flag_is_disabled_with_a_tooltip_in_a_file_that_does_not_store_it() -> None:
    asked: list = []

    def supported(name: str) -> bool:
        asked.append(name)
        return False

    panel, reported = _panel(codec_field_supported=supported)
    panel.show_unit(_entry(status=7))
    capture = _capture(panel)
    assert not capture.isEnabled()
    assert capture.currentText() == "Default"
    assert "1.59" in capture.toolTip() and "September 2026" in capture.toolTip()
    assert panel.unit_field_captions["capture_flag"].toolTip() == capture.toolTip()
    assert capture.isVisibleTo(panel), "disabled, not hidden"
    assert set(asked) == {"capture_flag"}
    # Unit State is in every version: never gated.
    assert _state(panel).isEnabled()
    assert _state(panel).currentText() == "Remove"
    assert "1.59" not in _state(panel).toolTip()
    assert panel.unit_field_captions["status"].toolTip() == ""
    assert reported == []


def test_a_disabled_capture_flag_shows_default_for_a_group_too() -> None:
    panel, reported = _panel(codec_field_supported=lambda _name: False)
    panel.show_group([_entry(reference_id=1, capture_flag=0), _entry(reference_id=2, capture_flag=1)])
    capture = _capture(panel)
    assert not capture.isEnabled()
    assert capture.currentText() == "Default"
    assert _items(capture) == _CAPTURE_LABELS
    assert reported == []


def test_the_gate_is_asked_again_on_every_populate() -> None:
    """A pull, like map_size: opening a 1.59 file after a 1.58 one needs no push."""
    answer = {"supported": False}
    panel, reported = _panel(codec_field_supported=lambda _name: answer["supported"])
    panel.show_unit(_entry(capture_flag=1))
    assert not _capture(panel).isEnabled()
    answer["supported"] = True
    panel.show_unit(_entry(capture_flag=1))
    capture = _capture(panel)
    assert capture.isEnabled()
    assert capture.currentText() == "Convert Once"
    assert capture.toolTip() == "Convert Once"
    assert panel.unit_field_captions["capture_flag"].toolTip() == ""
    assert reported == []


def test_with_no_callback_both_rows_are_live() -> None:
    """The panel stays constructible on its own: no gate means supported."""
    panel, _reported = _panel()
    panel.show_unit(_entry(capture_flag=3))
    assert _capture(panel).isEnabled()
    assert _capture(panel).currentText() == "Convert Never (Gaia Aggressive)"


# -- size ----------------------------------------------------------------------


@pytest.mark.font_sensitive
def test_the_longest_label_does_not_raise_the_panel_floor() -> None:
    """"Convert Never (Gaia Aggressive)" sized by its own text would push the
    page past MIN_USEFUL_WIDTH, and with it the whole window's minimum."""
    panel, _reported = _panel()
    panel.show_unit(_entry(capture_flag=3))
    assert panel.minimumSizeHint().width() <= UnitsPanel.MIN_USEFUL_WIDTH
    widest = max(combo.minimumSizeHint().width() for combo in (_capture(panel), _state(panel)))
    assert widest <= _capture(panel).fontMetrics().horizontalAdvance("Convert Never (Gaia Aggressive)")


@pytest.mark.font_sensitive
def test_extra_pane_width_goes_to_the_editors_not_the_captions() -> None:
    """TASK-247: with no column stretch the grid widened the captions first
    (113 px to 280 px), and a 500 px pane still cut the longest label."""
    from PyQt5.QtWidgets import QApplication, QStyle, QStyleOptionComboBox

    panel, _reported = _panel()
    capture = _capture(panel)
    caption = panel.unit_field_captions["capture_flag"]
    floor = UnitsPanel.MIN_USEFUL_WIDTH
    measured = {}
    panel.show()
    try:
        panel.show_unit(_entry(capture_flag=3))
        # The pane widths at the test font, scaled like the floor at a larger one.
        for base_px in (380, 500, 600):
            panel.resize(floor * base_px // 380, 900)
            QApplication.processEvents()
            option = QStyleOptionComboBox()
            capture.initStyleOption(option)
            field = capture.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, capture)
            measured[base_px] = (caption.width(), capture.width(), field.width())
    finally:
        panel.close()
    need = capture.fontMetrics().horizontalAdvance("Convert Never (Gaia Aggressive)")
    assert measured[500][2] - 2 >= need, f"a {measured[500][2]} px field cuts {need} px of text"
    assert measured[600][0] == measured[380][0], f"the caption column grew: {measured}"
    assert measured[380][1] < measured[500][1] < measured[600][1], f"the editor column did not grow: {measured}"


def test_a_live_row_repeats_its_value_as_the_tooltip() -> None:
    """The price of that floor: the closed combo cuts the longest label, so
    the full text is on the tooltip (and in the popup), whatever is shown."""
    panel, _reported = _panel()
    panel.show_unit(_entry(capture_flag=3, status=9))
    assert _capture(panel).toolTip() == "Convert Never (Gaia Aggressive)"
    assert _state(panel).toolTip() == "Other (9)"
    panel.show_group([_entry(reference_id=1, capture_flag=0), _entry(reference_id=2, capture_flag=1)])
    assert _capture(panel).toolTip() == "(mixed)"
    assert _state(panel).toolTip() == "Alive"
