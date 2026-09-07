"""Honest render-timing messages: the status/log "in X.XXs" numbers after a
load or re-render used to cover only synchronous prep, with the deferred
MapCanvasItem.paint() composite excluded entirely and, for load_scenario
specifically, the file parse excluded too -- so the visible number could
read ~250x under the real wait. This file pins the reworded wording rather
than re-measuring the timings themselves (tools/ probes already did that).

Driven through a real offscreen ViewerWindow against the shipped blank
template, same technique as tests/test_fill_tool.py -- no examples/ corpus
needed, so this stays in the default tier."""

from __future__ import annotations

import pytest

import conftest
from descape import debug_log, perf_trace
from descape.scenario_io import BLANK_TEMPLATE_PATH

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    return ViewerWindow()


def test_load_message_reports_total_with_parse_and_prepare_breakdown():
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        text = window.status_log.toPlainText()
        assert "in " in text and "(parse " in text and "prepare " in text
        # The whole point of the fix: the visible total is parse + prepare,
        # not just the (much smaller) prepare span alone.
        import re

        m = re.search(
            r"in (\d+\.\d+)s \(parse (\d+\.\d+)s, prepare (\d+\.\d+)s\)", text
        )
        assert m, text
        total, parse, prepare = (float(g) for g in m.groups())
        # Each of the three numbers is independently rounded to 2dp, so the
        # printed total can be off from parse+prepare by up to a cent's
        # worth of rounding on each side.
        assert total == pytest.approx(parse + prepare, abs=0.02)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_style_switch_reports_prepared_in_not_bare_in():
    """on_terrain_style_changed only ever times _render_current()'s
    synchronous prep -- the deferred composite happens later in paint() --
    so its message must say so rather than reading as a total."""
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.terrain_style_combo.setCurrentText("Flat")
        text = window.status_log.toPlainText()
        assert "Elevation view: Flat" in text
        assert "prepared in" in text
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_refresh_map_reports_prepared_in():
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.refresh_map()
        text = window.status_log.toPlainText()
        assert "Re-rendered map" in text
        assert "prepared in" in text
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_fill_message_says_applied_in_not_prepared_in():
    """on_fill's number covers the edit actually being applied
    (edit_history.apply() + _apply_dirty()), not preparation for a render --
    "prepared in" would misdescribe it (split out of the main plan as its
    own TODO item)."""
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("fill")
        window.terrain_combo.setCurrentIndex(window.terrain_combo.findData(15))  # GRASS_1
        window.on_fill(0, 0, 0)
        text = window.status_log.toPlainText()
        assert "applied in" in text
        assert "prepared in" not in text.splitlines()[-1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_perf_trace_on_emits_load_repaint_line_off_emits_nothing():
    """Step 4's closed gap: with Perf Trace on, a real first paint after
    load flushes a repaint-only debug_log line under the "load" label; off
    (the default), nothing does and the visible status/log messages are
    exactly what the tests above already pin."""
    from PyQt5.QtWidgets import QApplication

    perf_trace.enable(True)
    debug_log.clear()
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        QApplication.processEvents()
        QApplication.processEvents()
        assert "perf load: repaint:" in debug_log.get_log_text()
    finally:
        perf_trace.enable(False)
        perf_trace._repaint_durations = []
        perf_trace._armed_label = None
        debug_log.clear()
        window.edit_history.mark_saved()
        window.close()

    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        QApplication.processEvents()
        QApplication.processEvents()
        assert "perf" not in debug_log.get_log_text()
        assert "Loaded" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()
