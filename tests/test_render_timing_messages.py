"""Honest render-timing messages: the status/log "in X.XXs" numbers after a
load or re-render used to cover only synchronous prep, with the deferred
MapCanvasItem.paint() composite excluded entirely and, for load_scenario
specifically, the file parse excluded too -- so the visible number could
read ~250x under the real wait. This file pins the reworded wording rather
than re-measuring the timings themselves (tools/ probes already did that).
The 2026-09-08 follow-up adds the composite back as its own second line
after a load, always on rather than only under Help > Perf Trace.

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
        # Not testing descape/terrain_units.py -- Trees defaults on and would
        # otherwise pop an unpatched large-fill confirm QMessageBox for this
        # whole-map fill, hanging offscreen.
        window.paint_trees_check.setChecked(False)
        window.on_fill(0, 0, 0)
        text = window.status_log.toPlainText()
        assert "applied in" in text
        assert "prepared in" not in text.splitlines()[-1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def _spin(times: int = 5) -> None:
    """Event-loop spins with margin: the composite lands on one spin and the
    zero-delay drain timer on a later one, and neither is promised a
    particular spin."""
    from PyQt5.QtWidgets import QApplication

    for _ in range(times):
        QApplication.processEvents()


def test_first_paint_line_reports_composite_and_a_real_total():
    """The deferred composite the Loaded line can't see gets its own
    follow-up line, with Perf Trace off (the whole point, since it used to
    be visible only under tracing), and its total is
    parse + prepare + first paint."""
    import re

    # Perf Trace off is this test's precondition, not an assumption:
    # test_perf_trace.py resets that module in setup_function, so its last
    # test leaves tracing enabled for whatever runs next.
    perf_trace.enable(False)
    perf_trace._repaint_durations = []
    perf_trace._armed_label = None
    debug_log.clear()
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        _spin()
        text = window.status_log.toPlainText()
        assert "First paint composited in " in text, text
        loaded = re.search(
            r"in (\d+\.\d+)s \(parse (\d+\.\d+)s, prepare (\d+\.\d+)s\)", text
        )
        assert loaded, text
        painted = re.search(
            r"First paint composited in (\d+\.\d+)s \(total (\d+\.\d+)s, mip (-?\d+)", text
        )
        assert painted, text
        _, parse, prepare = (float(g) for g in loaded.groups())
        paint, total = float(painted.group(1)), float(painted.group(2))
        # Four independently 2dp-rounded numbers, so allow a cent each way
        # per term, same reasoning as the parse/prepare check above.
        assert total == pytest.approx(parse + prepare + paint, abs=0.03)
        # The perf-trace pair below asserts on this substring being absent
        # with tracing off; the new line must not reintroduce it.
        assert "perf" not in debug_log.get_log_text()
    finally:
        debug_log.clear()
        window.edit_history.mark_saved()
        window.close()


def test_first_paint_line_sums_a_sliver_and_the_real_composite():
    """Nothing is filtered out as "not the real paint": an 8px sliver
    repaint arriving before the composite must not be reported INSTEAD of
    it. Both numbers are driven in directly, so this pins the arithmetic
    rather than whatever paints the offscreen platform happened to issue."""
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        # Zero the accumulators first, so the arithmetic below is pinned
        # whether or not the offscreen platform issued a paint of its own.
        report = window._pending_paint_report
        assert report is not None, "load left no pending first-paint report"
        report.update(paint=0.0, paints=0, max_paint=-1.0)
        window._on_canvas_paint_timed(0.01, -2)
        window._on_canvas_paint_timed(5.10, -2)
        _spin()
        line = window.status_log.toPlainText().splitlines()[-1]
        assert line.startswith("First paint composited in 5.11s"), line
        assert "2 paints" in line, line
        assert "mip -2" in line, line
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_paint_with_no_load_pending_reports_nothing():
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        _spin()
        before = window.status_log.toPlainText()
        assert "First paint composited in " in before
        # The report is one-shot: a later repaint (pan, zoom, an edit) must
        # not emit a second line.
        window._on_canvas_paint_timed(3.0, -2)
        _spin()
        assert window.status_log.toPlainText() == before
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_first_paint_line_has_the_same_shape_switched_to_flat():
    """1.2: "re-open the same file in Flat: a much smaller first-paint
    figure, same line shape." load_scenario() renders at whichever Terrain
    Style was already selected (its own comment), so switching the combo
    after the first load and then re-opening is what actually renders Flat --
    switching alone only emits a "prepared in" line (see
    test_style_switch_reports_prepared_in_not_bare_in), never a fresh
    first-paint report."""
    import re

    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        _spin()
        window.terrain_style_combo.setCurrentText("Flat")
        window.load_scenario(BLANK_TEMPLATE_PATH)
        _spin()
        lines = [
            line for line in window.status_log.toPlainText().splitlines()
            if "First paint composited in " in line
        ]
        assert len(lines) == 2, lines
        assert re.match(
            r"First paint composited in \d+\.\d+s \(total \d+\.\d+s, mip -?\d+", lines[-1]
        ), lines[-1]
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_new_map_gets_the_same_first_paint_follow_up():
    """1.3: "File > New Map: the `Created ...` line gets the same
    follow-up." on_new_map's own real name is new_map() -- load_scenario()
    handles both, untitled=True is the only difference from an Open."""
    window = _window()
    try:
        window.new_map()
        assert window.scenario is not None, "New Map failed to produce a scenario"
        window.resize(400, 400)
        window.show()
        _spin()
        text = window.status_log.toPlainText()
        assert "Created " in text, text
        assert "First paint composited in " in text, text
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_two_loads_each_get_exactly_one_first_paint_line():
    """1.4: "open a file, then immediately open a second one: exactly one
    first-paint line per load, attributed to the right file." Only one real
    file is available in the default tier (no examples/ corpus), so "the
    right file" is checked by PAIRING -- each Loaded line must be followed by
    exactly one First paint line before the next Loaded line -- rather than
    by two distinct filenames."""
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        _spin()
        window.load_scenario(BLANK_TEMPLATE_PATH)
        _spin()
        lines = window.status_log.toPlainText().splitlines()
        loaded_idx = [i for i, line in enumerate(lines) if line.startswith("Loaded ")]
        paint_idx = [i for i, line in enumerate(lines) if "First paint composited in " in line]
        assert len(loaded_idx) == 2 and len(paint_idx) == 2, lines
        for load_i, paint_i, next_load_i in zip(loaded_idx, paint_idx, loaded_idx[1:] + [len(lines)]):
            assert load_i < paint_i < next_load_i, lines
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_pan_or_zoom_after_first_paint_emits_no_further_line():
    """1.6: "pan/zoom around afterwards: no further first-paint lines." A
    real QGraphicsView.scale() zoom, not a direct _on_canvas_paint_timed()
    call (test_paint_with_no_load_pending_reports_nothing already drives that
    one) -- this is the closer-to-the-checklist real-UI path."""
    window = _window()
    try:
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None, "blank template failed to load"
        window.resize(400, 400)
        window.show()
        _spin()
        before = window.status_log.toPlainText()
        assert "First paint composited in " in before
        window.map_view.scale(1.1, 1.1)
        _spin()
        window.map_view.scale(1 / 1.1, 1 / 1.1)
        _spin()
        assert window.status_log.toPlainText() == before
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
