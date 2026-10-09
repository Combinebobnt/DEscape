"""GH #142: View mode's info page gets a minimum pane width like the panels.

The left stack's own minimum is settings.MIN_SPLIT_PANE, not the page's
minimumSizeHint, so a pane dragged narrow in another mode kept clipping the
player-stats table at 150% scaling (144 DPI) once View was entered.
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
_NARROW_PX = 150


@pytest.mark.font_sensitive
def test_entering_view_from_a_narrow_pane_fits_the_info_page() -> None:
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import _LEFT_PAGE_INFO

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Units")
        total = sum(window.content_splitter.sizes())
        window.content_splitter.setSizes([_NARROW_PX, total - _NARROW_PX])
        QApplication.processEvents()
        assert window.content_splitter.sizes()[0] == _NARROW_PX, "the narrow setup did not take"

        window.mode_combo.setCurrentText("View")
        QApplication.processEvents()
        page = window.left_stack.widget(_LEFT_PAGE_INFO)
        assert window.left_stack.currentWidget() is page
        assert window.player_stats_rows, "no stats table shown -- the page would measure too narrow"
        left = window.content_splitter.sizes()[0]
        needed = page.minimumSizeHint().width()
        assert left >= needed, f"View pane {left} px, info page needs {needed} px"
    finally:
        conftest.close_window(window)


def test_a_pane_already_wider_than_the_floor_is_left_alone() -> None:
    from PyQt5.QtWidgets import QApplication

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(BLANK_FIXTURE)
        window.mode_combo.setCurrentText("Units")
        total = sum(window.content_splitter.sizes())
        window.content_splitter.setSizes([500, total - 500])
        window.mode_combo.setCurrentText("View")
        QApplication.processEvents()
        assert window.content_splitter.sizes()[0] == 500
    finally:
        conftest.close_window(window)
