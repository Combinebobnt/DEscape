"""MIN_USEFUL_WIDTH follows the app font (viewer_common.FontScaledWidth).

At Windows 150% scaling Qt 5 reports 144 DPI, so every label and field grows
by half while a fixed pixel pane width did not: the Map Options status line
crowded out the form and a trigger field overflowed the pane.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def app_font():
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    saved = app.font()
    yield app
    app.setFont(saved)


def test_the_pinned_test_font_gives_the_measured_widths(app_font) -> None:
    from descape.map_options_panel import MapOptionsPanel
    from descape.trigger_panel import TriggerPanel

    assert MapOptionsPanel.MIN_USEFUL_WIDTH == 300
    assert TriggerPanel.MIN_USEFUL_WIDTH == 340


def test_a_larger_font_widens_and_a_smaller_one_does_not_narrow(app_font) -> None:
    from PyQt5.QtGui import QFont

    from descape.map_options_panel import MapOptionsPanel
    from descape.trigger_panel import TriggerPanel

    family = app_font.font().family()
    app_font.setFont(QFont(family, 18))
    assert MapOptionsPanel.MIN_USEFUL_WIDTH >= 440
    assert TriggerPanel.MIN_USEFUL_WIDTH >= 500
    assert MapOptionsPanel.MIN_USEFUL_WIDTH < TriggerPanel.MIN_USEFUL_WIDTH

    app_font.setFont(QFont(family, 8))
    assert MapOptionsPanel.MIN_USEFUL_WIDTH == 300
    assert TriggerPanel.MIN_USEFUL_WIDTH == 340


def test_an_instance_reads_the_same_width_as_the_class(app_font) -> None:
    from descape.map_options_panel import MapOptionsPanel

    panel = MapOptionsPanel()
    try:
        assert panel.MIN_USEFUL_WIDTH == MapOptionsPanel.MIN_USEFUL_WIDTH
    finally:
        panel.deleteLater()
