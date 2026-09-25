"""draw-perf plan Step 1: perf_trace is off by default, a true no-op while
disabled, and aggregates+flushes exactly one line per drag once enabled.
Pure-module test, no Qt needed."""

from __future__ import annotations

from descape import debug_log, perf_trace


def setup_function():
    perf_trace.enable(False)
    debug_log.clear()
    # Reset module state directly rather than via flush() (which is itself
    # under test and a no-op while disabled) -- a leftover step from a test
    # that forgot to flush must not leak into the next one.
    perf_trace._current_step = {}
    perf_trace._step_totals = []
    perf_trace._phase_sums = {}
    perf_trace._phase_order = []
    perf_trace._repaint_durations = []
    perf_trace._armed_label = None
    perf_trace._drag_active = False


def test_disabled_by_default():
    assert perf_trace.is_enabled() is False


def test_phase_is_a_true_no_op_while_disabled():
    with perf_trace.phase("bbox"):
        pass
    perf_trace.step()
    perf_trace.flush("paint-terrain")
    assert debug_log.get_log_text() == "(empty)"
    assert perf_trace._current_step == {}
    assert perf_trace._step_totals == []


def test_step_with_no_recorded_phase_is_a_no_op():
    perf_trace.enable(True)
    perf_trace.step()
    assert perf_trace._step_totals == []


def test_phases_accumulate_across_steps_and_flush_emits_one_line():
    perf_trace.enable(True)
    with perf_trace.phase("bbox"):
        pass
    with perf_trace.phase("patch"):
        pass
    perf_trace.step()
    with perf_trace.phase("bbox"):
        pass
    perf_trace.step()
    with perf_trace.phase("repaint"):
        pass
    with perf_trace.phase("repaint"):
        pass

    perf_trace.flush("paint-terrain")

    text = debug_log.get_log_text()
    assert text.count("\n[") == 0, "flush() must emit exactly one debug_log line per drag"
    assert "perf drag paint-terrain: 2 steps" in text
    assert "bbox" in text and "patch" in text
    assert "repaint: 2 calls" in text

    # State resets for the next drag.
    assert perf_trace._current_step == {}
    assert perf_trace._step_totals == []
    assert perf_trace._repaint_durations == []


def test_flush_with_no_steps_is_a_no_op():
    perf_trace.enable(True)
    perf_trace.flush("paint-terrain")
    assert debug_log.get_log_text() == "(empty)"


def test_repaint_never_appears_in_the_step_phase_line():
    """repaint is reported on its own line -- Qt's paint() fires after
    invalidate_region() schedules one, not synchronously inside a step, so
    it must never be folded into the ms/step phase breakdown."""
    perf_trace.enable(True)
    with perf_trace.phase("repaint"):
        pass
    with perf_trace.phase("bbox"):
        pass
    perf_trace.step()
    perf_trace.flush("paint-terrain")

    lines = debug_log.get_log_text().splitlines()
    phase_line = lines[1]
    assert "repaint" not in phase_line
    assert "bbox" in phase_line


def test_disabling_mid_drag_does_not_crash_and_flush_stays_a_no_op():
    perf_trace.enable(True)
    with perf_trace.phase("bbox"):
        pass
    perf_trace.step()
    perf_trace.enable(False)
    perf_trace.step()  # no-op while disabled, previous step's data untouched
    perf_trace.flush("paint-terrain")
    assert debug_log.get_log_text() == "(empty)"


def test_arm_and_first_paint_done_flush_a_repaint_only_line():
    """The load path's hook: no drag ever ran (no step() call at all), just
    a repaint recorded under an armed label."""
    perf_trace.enable(True)
    perf_trace.arm("load")
    with perf_trace.phase("repaint"):
        pass
    perf_trace.first_paint_done()

    text = debug_log.get_log_text()
    assert "perf load: repaint: 1 calls" in text
    assert perf_trace._armed_label is None
    assert perf_trace._repaint_durations == []


def test_first_paint_done_without_arm_is_a_no_op():
    perf_trace.enable(True)
    with perf_trace.phase("repaint"):
        pass
    perf_trace.first_paint_done()
    assert debug_log.get_log_text() == "(empty)"
    # The unflushed repaint duration is still sitting there for a later
    # drag's flush() to pick up -- first_paint_done() only acts when armed.
    assert len(perf_trace._repaint_durations) == 1


def test_arm_is_a_no_op_while_disabled():
    perf_trace.arm("load")
    assert perf_trace._armed_label is None


def test_flush_emits_repaint_only_line_with_no_steps_recorded():
    """flush()'s own no-steps-is-a-no-op guard (test_flush_with_no_steps_is_
    a_no_op above) must not swallow a repaint-only flush -- the gap step 4
    of the honest-render-timing plan closes."""
    perf_trace.enable(True)
    with perf_trace.phase("repaint"):
        pass
    perf_trace.flush("load")

    text = debug_log.get_log_text()
    assert text.count("\n[") == 0
    assert "perf load: repaint: 1 calls" in text
    assert perf_trace._repaint_durations == []


def _repaint():
    with perf_trace.phase("repaint"):
        pass


def test_flush_idle_logs_repaints_outside_a_drag_as_a_view_line():
    perf_trace.enable(True)
    _repaint()
    _repaint()
    perf_trace.flush_idle()
    text = debug_log.get_log_text()
    assert "perf view: repaint: 2 calls" in text
    assert perf_trace._repaint_durations == []


def test_flush_idle_is_a_no_op_during_a_drag_or_an_armed_load():
    perf_trace.enable(True)
    perf_trace.begin_drag()
    _repaint()
    perf_trace.flush_idle()
    assert debug_log.get_log_text() == "(empty)"
    perf_trace.end_drag("draw")
    perf_trace._repaint_durations = []

    perf_trace.arm("load")
    _repaint()
    perf_trace.flush_idle()
    assert debug_log.get_log_text() == "(empty)"


def test_begin_drag_splits_off_earlier_repaints_and_drops_hover_phases():
    """Pan repaints before the drag get their own line, and hover pick time
    (recorded on every mouse move, drag or not) never reaches the first step."""
    perf_trace.enable(True)
    _repaint()
    perf_trace._record("pick", 500.0)  # an afternoon of hovering
    perf_trace.begin_drag()
    perf_trace._record("pick", 1.0)
    perf_trace.step()
    perf_trace.flush("paint-terrain")
    perf_trace.end_drag("draw")

    lines = debug_log.get_log_text().splitlines()
    assert "perf view: repaint: 1 calls" in lines[0]
    assert "perf drag paint-terrain: 1 steps, 1ms total" in lines[1]
    assert "repaint" not in "\n".join(lines[1:])


def test_end_drag_flushes_a_stroke_the_viewer_did_not():
    perf_trace.enable(True)
    perf_trace.begin_drag()
    perf_trace._record("patch", 2.0)
    perf_trace.step()
    perf_trace.end_drag("convert")
    assert "perf drag convert: 1 steps" in debug_log.get_log_text()
    assert perf_trace._drag_active is False


def test_end_drag_after_the_viewer_flushed_logs_nothing_more():
    perf_trace.enable(True)
    perf_trace.begin_drag()
    perf_trace._record("patch", 2.0)
    perf_trace.step()
    perf_trace.flush("paint-terrain")
    perf_trace.end_drag("draw")
    assert debug_log.get_log_text().count("perf drag") == 1


def test_the_idle_scheduler_runs_on_repaints_outside_a_drag_only(monkeypatch):
    calls = []
    monkeypatch.setattr(perf_trace, "_idle_scheduler", lambda: calls.append(1))
    perf_trace.enable(True)
    _repaint()
    assert len(calls) == 1
    perf_trace.begin_drag()
    _repaint()
    assert len(calls) == 1
