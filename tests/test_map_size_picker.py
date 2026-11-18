"""descape/map_size_picker.py: the size row New Map and Resize Map share
(TASK-620), offscreen, typed into the way a user types."""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def picker_for():
    conftest.ensure_qapp()
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QSpinBox, QVBoxLayout, QWidget

    from descape.map_size_picker import MapSizePicker

    class Host(QWidget):
        """Records the keys the picker passed on, as a dialog would see them."""

        def __init__(self) -> None:
            super().__init__()
            self.keys = []

        def keyPressEvent(self, event) -> None:
            self.keys.append(event.key())

    made = []

    def make(tiles=240, **kwargs):
        host = Host()
        layout = QVBoxLayout(host)
        picker = MapSizePicker(tiles, **kwargs)
        picker.other = QSpinBox()  # somewhere for Tab to go
        layout.addWidget(picker)
        layout.addWidget(picker.other)
        host.show()
        QTest.qWaitForWindowExposed(host)
        picker.host = host
        picker.emitted, picker.validity = [], []
        picker.changed.connect(picker.emitted.append)
        picker.validity_changed.connect(picker.validity.append)
        made.append(host)
        return picker

    yield make
    for host in made:
        host.close()
        host.deleteLater()


def _type(picker, text: str) -> None:
    """Selects the spin's number, as Tab focus does, and types `text` over it."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    picker.spin.setFocus(Qt.TabFocusReason)
    picker.spin.selectAll()
    line_edit = picker.spin.lineEdit()
    QTest.keyClick(line_edit, Qt.Key_Delete)
    QTest.keyClicks(line_edit, text)


def _key(picker, key, target: str = "line edit") -> None:
    """A key press on the spin's line edit, or on the spin itself (where real focus sends it)."""
    from PyQt5.QtTest import QTest

    QTest.keyClick({"line edit": picker.spin.lineEdit(), "spin": picker.spin, "combo": picker.combo}[target], key)


def test_a_preset_fills_the_spin(picker_for) -> None:
    picker = picker_for()
    picker.combo.setCurrentIndex(picker.combo.findData(144))
    assert picker.tiles == 144 and picker.spin.value() == 144
    assert picker.emitted == [144]


def test_typing_selects_the_matching_preset_or_custom(picker_for) -> None:
    from descape.map_size_picker import CUSTOM_LABEL

    picker = picker_for()
    _type(picker, "200")
    assert picker.combo.currentText() == "Normal (200×200)"
    _type(picker, "150")
    assert picker.combo.currentText() == CUSTOM_LABEL and picker.is_custom
    assert picker.emitted == [200, 150]


def test_custom_keeps_the_size_until_a_game_size_is_retyped(picker_for) -> None:
    from PyQt5.QtCore import Qt

    picker = picker_for()
    picker.combo.setCurrentIndex(picker.combo.count() - 1)
    assert picker.tiles == 240 and picker.emitted == [] and picker.is_custom
    _type(picker, "240")  # the same size: no valueChanged
    _key(picker, Qt.Key_Tab)
    assert picker.combo.currentText() == "Giant (240×240)"


@pytest.mark.parametrize("target", ["line edit", "spin"])
@pytest.mark.parametrize("enter_applies", [False, True])
def test_enter_on_text_under_the_minimum_keeps_it_and_emits_nothing(picker_for, enter_applies, target) -> None:
    from PyQt5.QtCore import Qt

    picker = picker_for(enter_applies=enter_applies)
    _type(picker, "60")
    assert not picker.acceptable and picker.validity == [False]
    _key(picker, Qt.Key_Return, target)
    assert picker.spin.lineEdit().text().startswith("60")
    assert picker.emitted == [] and picker.tiles == 240
    assert picker.host.keys == []


@pytest.mark.parametrize("target", ["line edit", "spin"])
@pytest.mark.parametrize("enter_applies", [False, True])
def test_enter_on_a_valid_size_passes_on_only_without_enter_applies(picker_for, enter_applies, target) -> None:
    from PyQt5.QtCore import Qt

    picker = picker_for(enter_applies=enter_applies)
    _type(picker, "168")
    _key(picker, Qt.Key_Enter, target)
    assert picker.emitted == [168] and picker.tiles == 168
    assert picker.host.keys == ([] if enter_applies else [Qt.Key_Enter])


@pytest.mark.parametrize("enter_applies", [False, True])
def test_enter_on_the_combo_passes_on_only_without_enter_applies(picker_for, enter_applies) -> None:
    from PyQt5.QtCore import Qt

    picker = picker_for(enter_applies=enter_applies)
    picker.combo.setFocus(Qt.TabFocusReason)
    _key(picker, Qt.Key_Return, "combo")
    assert picker.host.keys == ([] if enter_applies else [Qt.Key_Return])


def test_leaving_bad_text_or_retyping_the_size_ends_acceptable(picker_for) -> None:
    from PyQt5.QtCore import Qt

    picker = picker_for()
    _type(picker, "60")
    _key(picker, Qt.Key_Tab)  # Qt puts 240 back, with no textChanged or valueChanged
    assert picker.acceptable and picker.tiles == 240
    assert picker.validity == [False, True]
    _type(picker, "")
    assert picker.validity == [False, True, False]
    _type(picker, "240")
    assert picker.acceptable and picker.validity == [False, True, False, True]
    assert picker.emitted == []


def test_the_current_size_is_marked(picker_for) -> None:
    from descape.map_size_picker import CURRENT_MARK

    picker = picker_for(240, current=240)
    assert picker.combo.currentText() == "Giant (240×240)" + CURRENT_MARK
    odd = picker_for(150, current=150)
    assert odd.combo.currentText() == "Current (150×150)" and not odd.is_custom
    sizes = [odd.combo.itemData(i) for i in range(odd.combo.count() - 1)]
    assert sizes == sorted(sizes)
    # A map below the minimum gets no entry the spin could not hold.
    small = picker_for(61, current=40)
    assert small.combo.findData(40) < 0 and small.combo.count() == odd.combo.count() - 1
