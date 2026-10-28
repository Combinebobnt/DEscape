"""TASK-362: in Units mode the left pane cannot be dragged below the Units
page's own minimumSizeHint width (user decision 2026-10-14); every other mode
keeps settings.MIN_SPLIT_PANE as its drag floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _drag_left_to(window, width: int) -> int:
    from PyQt5.QtWidgets import QApplication

    total = sum(window.content_splitter.sizes())
    window.content_splitter.setSizes([width, total - width])
    QApplication.processEvents()
    return window.content_splitter.sizes()[0]


def test_units_mode_floors_the_left_pane_at_the_units_page_and_leaving_restores_it() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Units")
        QApplication.processEvents()
        needed = window.units_panel.minimumSizeHint().width()
        assert needed > settings.MIN_SPLIT_PANE, "the Units page asks for no more than the old floor"

        assert _drag_left_to(window, settings.MIN_SPLIT_PANE) >= needed

        for mode in ("View", "Terrain", "Triggers"):
            window.mode_combo.setCurrentText(mode)
            QApplication.processEvents()
            assert _drag_left_to(window, settings.MIN_SPLIT_PANE) == settings.MIN_SPLIT_PANE, mode

        window.mode_combo.setCurrentText("Units")
        QApplication.processEvents()
        assert window.content_splitter.sizes()[0] >= needed, "re-entering Units must lift a narrow pane"
    finally:
        conftest.close_window(window)
