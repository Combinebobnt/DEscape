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
