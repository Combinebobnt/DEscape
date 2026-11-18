"""descape/spin_boxes.py: a click at the end of a suffixed spin (TASK-620)
leaves the cursor in the suffix, where Qt drops every Backspace and digit."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def spin_pair():
    """A focusable neighbour, then a 61..480 spin at 240 with a " tiles" suffix."""
    conftest.ensure_qapp()
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QSpinBox, QVBoxLayout, QWidget

    made = []

    def make(helper: bool):
        from descape.spin_boxes import keep_cursor_before_suffix

        host = QWidget()
        layout = QVBoxLayout(host)
        other, spin = QSpinBox(), QSpinBox()
        spin.setRange(61, 480)
        spin.setValue(240)
        spin.setSuffix(" tiles")
        if helper:
            keep_cursor_before_suffix(spin)
        layout.addWidget(other)
        layout.addWidget(spin)
        host.resize(220, 90)
        host.show()
        QTest.qWaitForWindowExposed(host)
        made.append(host)
        return other, spin

    yield make
    for host in made:
        host.close()
        host.deleteLater()


def _click_and_retype(spin, x: int, typed: str) -> None:
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest

    line_edit = spin.lineEdit()
    QTest.mouseClick(line_edit, Qt.LeftButton, pos=QPoint(x, line_edit.height() // 2))
    for _ in range(3):
        QTest.keyClick(line_edit, Qt.Key_Backspace)
    QTest.keyClicks(line_edit, typed)


@pytest.mark.parametrize("where", ["past the end", "inside the suffix"])
@pytest.mark.parametrize("helper", [True, False])
def test_a_click_in_the_suffix_still_takes_typing(spin_pair, where, helper) -> None:
    _other, spin = spin_pair(helper)
    line_edit = spin.lineEdit()
    x = line_edit.width() - 4 if where == "past the end" else line_edit.fontMetrics().horizontalAdvance("240 ti") + 4
    _click_and_retype(spin, x, "168")
    # Without the helper the text never changes: the control that this test can fail.
    assert spin.value() == (168 if helper else 240)


def test_tab_focus_still_selects_the_number(spin_pair) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    other, spin = spin_pair(True)
    other.setFocus()
    QTest.keyClick(other, Qt.Key_Tab)
    assert spin.lineEdit().selectedText() == "240"
    QTest.keyClicks(spin.lineEdit(), "200")
    assert spin.value() == 200


def test_every_suffixed_spin_calls_the_helper() -> None:
    """A new setSuffix() with no keep_cursor_before_suffix() within 3 lines brings the trap back."""
    missing = []
    for path in sorted((ROOT / "descape").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if re.search(r"\.setSuffix\(", line):
                nearby = "\n".join(lines[number : number + 4])
                if "keep_cursor_before_suffix(" not in nearby:
                    missing.append(f"{path.relative_to(ROOT)}:{number + 1}")
    assert missing == []
