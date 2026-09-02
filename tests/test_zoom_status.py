"""The bottom status bar's live "Zoom: N%" readout (MapView.zoom_percent_of_fit /
ViewerWindow._update_zoom_status), added so the user could measure where
zooming in stopped being useful and use that reading to calibrate
MAX_ZOOM_MULTIPLE_OF_FIT (now 64.0, alongside a 50% zoom-out floor).

"At fit" is not exactly 100% on a single fit -- see test_zoom_bounds.py's
_window() docstring on why applying the fit transform can change
viewport().rect() afterwards, leaving the first fit a few percent off. Assert
with a tolerance, not exact percentages, except for the "--" no-map cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _window(width: int = 1400, height: int = 1000):
    """Same shape as test_zoom_bounds.py's _window(): a shown ViewerWindow
    fitted at its final size, with the deliberate double set_isometric() that
    file's docstring explains (the first fit can be a few percent off the
    viewport that actually ends up on screen; the second converges exactly)."""
    from PyQt5.QtWidgets import QApplication

    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    window.load_scenario(FIXTURE)
    if window.scenario is None:
        window.close()
        pytest.skip(f"{FIXTURE.name} failed to load")
    window.resize(width, height)
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    window.map_view.set_isometric(False)
    QApplication.processEvents()
    window.map_view.set_isometric(False)
    QApplication.processEvents()
    return window


def _wheel(view, up: bool = True) -> None:
    """A real QWheelEvent, not view.scale(...) -- this is the one case that
    bypasses wheelEvent() and so cannot exercise it. Delivered directly to
    view.wheelEvent() rather than through QApplication.sendEvent(), since
    MapView.wheelEvent only reads event.angleDelta().y(); no other event
    plumbing (position, viewport picking) is involved."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent

    center = QPointF(view.viewport().rect().center())
    angle = QPoint(0, 120 if up else -120)
    event = QWheelEvent(
        center, center, QPoint(0, 0), angle, Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False
    )
    view.wheelEvent(event)


def test_no_map_loaded_shows_a_dash() -> None:
    window = conftest.shown_window()
    try:
        assert window.zoom_status_label.text() == "  Zoom: --  "
        assert window.map_view.zoom_percent_of_fit() is None
    finally:
        conftest.close_window(window)


def test_at_fit_reads_roughly_100_percent() -> None:
    window = _window()
    try:
        pct = window.map_view.zoom_percent_of_fit()
        assert pct is not None
        assert 95 <= pct <= 105
        assert window.zoom_status_label.text() == f"  Zoom: {pct:.0f}%  "
    finally:
        conftest.close_window(window)


def test_a_wheel_tick_rises_by_the_step_factor() -> None:
    window = _window()
    try:
        before = window.map_view.zoom_percent_of_fit()
        _wheel(window.map_view, up=True)
        after = window.map_view.zoom_percent_of_fit()
        assert after == pytest.approx(before * 1.25, rel=1e-6)
        assert window.zoom_status_label.text() == f"  Zoom: {after:.0f}%  "
    finally:
        conftest.close_window(window)


def test_a_resize_with_no_zoom_action_changes_the_reading() -> None:
    """The stale-label discriminator: a resize alone changes
    _fit_baseline_scale()'s viewport-derived answer with no scale() call at
    all, so a fix that only hooks wheelEvent would leave this label wrong."""
    from PyQt5.QtWidgets import QApplication

    window = _window(1400, 1000)
    try:
        before = window.zoom_status_label.text()
        window.resize(900, 1400)
        QApplication.processEvents()
        after = window.zoom_status_label.text()
        assert after != before
    finally:
        conftest.close_window(window)


def test_closing_the_map_resets_the_readout_to_a_dash() -> None:
    window = _window()
    try:
        assert window.map_view.zoom_percent_of_fit() is not None
        window.map_view.clear_image()
        assert window.zoom_status_label.text() == "  Zoom: --  "
        assert window.map_view.zoom_percent_of_fit() is None
    finally:
        conftest.close_window(window)


def test_a_wheel_tick_refused_by_the_ceiling_leaves_the_label_unchanged() -> None:
    window = _window()
    try:
        view = window.map_view
        # Wheel in with real events until wheelEvent's own ceiling guard
        # starts refusing -- i.e. the label stops moving between ticks.
        before = window.zoom_status_label.text()
        for _ in range(60):
            _wheel(view, up=True)
            after = window.zoom_status_label.text()
            if after == before:
                break
            before = after
        else:
            pytest.fail("never reached the zoom-in ceiling")
        _wheel(view, up=True)
        assert window.zoom_status_label.text() == before
    finally:
        conftest.close_window(window)
