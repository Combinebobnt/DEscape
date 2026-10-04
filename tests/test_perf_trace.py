"""draw-perf plan Step 1: perf_trace is off by default, a true no-op while
disabled, and aggregates+flushes exactly one line per drag once enabled.
Pure-module test, no Qt needed."""

from __future__ import annotations

import gc
import threading

import pytest

from descape import debug_log, perf_trace


@pytest.fixture(autouse=True)
def _fresh_trace_state(monkeypatch):
    debug_log.clear()
    # Reset module state directly rather than via flush() (which is itself
    # under test and a no-op while disabled) -- a leftover step from a test
    # that forgot to flush must not leak into the next one. monkeypatch also
    # restores it afterwards, so tracing left on here can't leak into the
    # next file on the same worker.
    monkeypatch.setattr(perf_trace, "_enabled", False)
    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    monkeypatch.setattr(perf_trace, "_sprite_counter", lambda: (0, 0))
    # A real collection under the fake clock would add a 0.0ms gc part to an exact-line assert.
    assert perf_trace._on_gc not in gc.callbacks, "a test left the gc hook installed"
    yield
    perf_trace.set_gc_hook(False)


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


def test_warm_tick_is_a_no_op_while_disabled():
    perf_trace.warm_tick(5.0, 2, 3.0)
    assert perf_trace._warm_ticks == []


def test_warm_ticks_join_repaints_on_the_view_line_with_max_tick_and_max_chunk():
    """Max chunk is its own figure: under a budget, max tick alone can't say
    whether one chunk is too slow to fit (Batch F T2's trigger)."""
    perf_trace.enable(True)
    _repaint()
    perf_trace.warm_tick(7.5, 3, 2.5)
    perf_trace.warm_tick(9.0, 1, 9.0)
    perf_trace.flush_idle()
    text = debug_log.get_log_text()
    assert "perf view: repaint: 1 calls" in text
    assert "warm: 2 ticks, 4 chunks, 16ms total (max tick 9.0, max chunk 9.0)" in text
    assert perf_trace._warm_ticks == []


def test_warm_ticks_alone_still_flush_a_view_line():
    perf_trace.enable(True)
    perf_trace.warm_tick(4.0, 2, 2.2)
    perf_trace.flush_idle()
    assert "perf view: warm: 1 ticks, 2 chunks, 4ms total (max tick 4.0, max chunk 2.2), composite" in (
        debug_log.get_log_text()
    )


def test_worker_kernels_join_the_warm_bucket_and_can_flush_alone():
    """A pooled chunk's kernel lands after its tick, so it may be the only
    warm figure left at the next flush."""
    perf_trace.enable(True)
    perf_trace.warm_tick(3.0, 2, 1.6)
    perf_trace.warm_worker(1.2)
    perf_trace.warm_worker(2.4)
    perf_trace.flush_idle()
    perf_trace.warm_worker(5.0)
    perf_trace.flush_idle()
    text = debug_log.get_log_text()
    assert "warm: 1 ticks, 2 chunks, 3ms total (max tick 3.0, max chunk 1.6), worker 2 chunks, 4ms total (max 2.4)" in text
    assert "perf view: warm: worker 1 chunks, 5ms total (max 5.0), composite" in text
    assert perf_trace._warm_worker == []


def test_the_idle_scheduler_runs_on_warm_ticks_outside_a_drag_only(monkeypatch):
    calls = []
    monkeypatch.setattr(perf_trace, "_idle_scheduler", lambda: calls.append(1))
    perf_trace.enable(True)
    perf_trace.warm_tick(1.0, 1, 1.0)
    assert len(calls) == 1
    perf_trace.begin_drag()
    perf_trace.warm_tick(1.0, 1, 1.0)
    assert len(calls) == 1


def test_the_idle_scheduler_runs_on_repaints_outside_a_drag_only(monkeypatch):
    calls = []
    monkeypatch.setattr(perf_trace, "_idle_scheduler", lambda: calls.append(1))
    perf_trace.enable(True)
    _repaint()
    assert len(calls) == 1
    perf_trace.begin_drag()
    _repaint()
    assert len(calls) == 1


def test_phases_after_the_last_step_print_as_the_stroke_end_line():
    """The stroke-end handler records after MapView's last step(); those
    phases get their own line instead of being dropped or averaged in."""
    perf_trace.enable(True)
    perf_trace._record("patch", 2.0)
    perf_trace.step()
    perf_trace._record("unit_plan", 7.0)
    perf_trace._record("unit_sources", 3.0)
    perf_trace.flush("paint-terrain")

    lines = debug_log.get_log_text().splitlines()
    assert "unit_plan" not in lines[1], "a stroke-end phase leaked into the per-step line"
    assert "| end: unit_plan 7.0 unit_sources 3.0" in lines[2]


# --- perf op, level events, sprites, stalls (perf-trace-coverage plan) -------


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def perf_counter(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(perf_trace, "time", fake)
    perf_trace.enable(True)
    return fake


def _lines() -> list[str]:
    # debug_log prefixes each entry with a timestamp; entries may span lines.
    return [line.split("] ", 1)[-1] for line in debug_log.get_log_text().splitlines()]


def _timed_phase(clock, name: str, ms: float) -> None:
    with perf_trace.phase(name):
        clock.advance(ms)


def test_disabled_op_level_and_where_are_the_shared_null_context_and_record_nothing():
    assert perf_trace.op("place-unit") is perf_trace._NULL_PHASE
    assert perf_trace.level("build", -1) is perf_trace._NULL_PHASE
    assert perf_trace.where("margin-warm") is perf_trace._NULL_PHASE
    assert perf_trace.span("press") is perf_trace._NULL_PHASE
    with perf_trace.level("build", -1) as ev:
        assert ev is None
    perf_trace.level_event("build", -1, 5.0)
    perf_trace.level_warm_tick(5.0, [1.0])
    perf_trace.view_mip(-1)
    perf_trace.view_mip(-2)
    perf_trace.stall(500.0)
    perf_trace.flush_idle()
    assert debug_log.get_log_text() == "(empty)"
    assert not perf_trace._recent and not perf_trace._level_events and not perf_trace._mip_changes
    assert perf_trace._pending_op is None and perf_trace._level_warm_ticks == []


def test_op_line_has_total_phases_and_untimed_and_drops_hover_phases(clock):
    perf_trace._record("pick", 50.0)  # hovering before the click
    with perf_trace.op("Place unit"):
        clock.advance(2)
        _timed_phase(clock, "unit_model_build", 380)
        _timed_phase(clock, "unit_sources", 6)
        clock.advance(4)
    assert debug_log.get_log_text() == "(empty)", "an op line waits for the next op or the idle flush"
    perf_trace.flush_idle()
    assert _lines() == ["perf op place-unit: 392ms | unit_model_build 380.0 unit_sources 6.0 untimed 6.0"]
    assert perf_trace._current_step == {}


def test_a_nested_op_is_a_phase_and_a_same_label_one_is_transparent(clock):
    with perf_trace.op("place-unit"):
        with perf_trace.op("unit_model_build"):
            clock.advance(300)
        with perf_trace.op("Place unit"):
            _timed_phase(clock, "unit_patch", 10)
    perf_trace.flush_idle()
    assert _lines() == ["perf op place-unit: 310ms | unit_model_build 300.0 unit_patch 10.0 untimed 0.0"]


def test_an_op_inside_a_drag_is_a_phase_of_the_stroke(clock):
    perf_trace.begin_drag()
    _timed_phase(clock, "patch", 2)
    perf_trace.step()
    with perf_trace.op("re-render"):
        clock.advance(40)
    perf_trace.flush("draw")
    perf_trace.end_drag("draw")
    text = "\n".join(_lines())
    assert "perf op" not in text
    assert "| end: re-render 40.0" in text


def test_an_op_that_raises_leaves_no_open_op_or_where_tag(clock):
    with pytest.raises(RuntimeError), perf_trace.op("place-unit"):
        raise RuntimeError("abort_unit_edit path")
    assert perf_trace._op_stack == [] and perf_trace._where_stack == [] and perf_trace._top_op is None
    with perf_trace.op("undo"):
        clock.advance(1)
    perf_trace.flush_idle()
    assert _lines()[-1].startswith("perf op undo: 1ms")


def test_same_label_ops_coalesce_across_their_repaints_and_print_before_them(clock):
    for ms in (3, 5, 4):
        with perf_trace.op("Nudge unit"):
            _timed_phase(clock, "unit_patch", ms - 1)
            clock.advance(1)
        _timed_phase(clock, "repaint", 2)
    perf_trace.flush_idle()
    lines = _lines()
    assert lines[0] == "perf op nudge-unit x3: 12ms total, 4.0ms mean (max 5.0) | unit_patch 3.0 untimed 1.0"
    assert lines[1].startswith("perf view: repaint: 3 calls")
    assert len(lines) == 2


def test_a_different_label_writes_the_pending_run_first(clock):
    for _ in range(2):
        with perf_trace.op("nudge"):
            clock.advance(2)
    with perf_trace.op("undo"):
        clock.advance(7)
    assert _lines() == ["perf op nudge x2: 4ms total, 2.0ms mean (max 2.0) | untimed 2.0"]
    perf_trace.flush_idle()
    assert _lines()[-1] == "perf op undo: 7ms | untimed 7.0"


def test_begin_drag_and_a_load_flush_write_the_pending_op_first(clock):
    with perf_trace.op("re-render"):
        perf_trace.arm("re-render")
        clock.advance(20)
    _timed_phase(clock, "repaint", 30)
    perf_trace.first_paint_done()
    lines = _lines()
    assert lines[0].startswith("perf op re-render: 20ms")
    assert lines[1].startswith("perf re-render: repaint: 1 calls")

    with perf_trace.op("copy"):
        clock.advance(1)
    perf_trace.begin_drag()
    assert _lines()[-1].startswith("perf op copy: 1ms")


def test_flush_pending_op_writes_the_line_with_no_idle_tick(clock):
    for _ in range(4):
        with perf_trace.op("paste"):
            clock.advance(250)
    assert debug_log.get_log_text() == "(empty)"
    perf_trace.flush_pending_op()
    assert _lines() == ["perf op paste x4: 1000ms total, 250.0ms mean (max 250.0) | untimed 250.0"]
    assert perf_trace._pending_op is None
    perf_trace.flush_pending_op()
    assert len(_lines()) == 1, "a second flush must not repeat the line"


def test_op_exit_restarts_the_idle_flush_outside_a_drag(clock, monkeypatch):
    calls = []
    monkeypatch.setattr(perf_trace, "_idle_scheduler", lambda: calls.append(1))
    with perf_trace.op("copy"):
        pass
    assert calls == [1], "an op that repaints nothing must still get its line written"


def test_level_events_aggregate_per_kind_mip_and_where(clock):
    with perf_trace.phase("repaint"), perf_trace.level("build", -1) as ev:
        clock.advance(900)
        ev.mark("walk")
        clock.advance(80)
        ev.mark("install")
    with perf_trace.where("margin-warm"):
        perf_trace.level_event("flush", -2, 1.0, tiles=3)
        perf_trace.level_event("flush", -2, 1.1, tiles=4)
    perf_trace.level_event("pack", 0, 0.5)
    perf_trace.flush_idle()
    [line] = _lines()
    assert "levels: -1 build 980.0ms [walk 900.0 install 80.0] (repaint), " in line
    assert "-2 flush 2.1ms x2 7 tiles (margin-warm), 0 pack 0.5ms (other)" in line
    assert perf_trace._level_events == {}


def test_level_events_inside_an_op_print_on_its_line_and_alone_still_flush_a_view_line(clock):
    with perf_trace.op("zoom"), perf_trace.level("flush>rebuild", -1):
        clock.advance(3)
    perf_trace.level_event("build", -2, 40.0)
    perf_trace.flush_idle()
    lines = _lines()
    assert lines[0].endswith("| levels: -1 flush>rebuild 3.0ms (op:zoom)")
    assert lines[1].startswith("perf view: levels: -2 build 40.0ms (other)")


def test_level_warm_ticks_join_the_warm_bucket(clock):
    perf_trace.level_warm_tick(8.0, [3.0])
    perf_trace.level_warm_tick(12.5, [])
    perf_trace.flush_idle()
    assert "warm: level 2 ticks, max tick 12.5 [other 12.5], installs 1 (max 3.0)" in _lines()[0], "gc unmeasured"


def test_the_worst_level_warm_tick_prints_its_split_and_gc(clock):
    perf_trace.level_warm_tick(30.0, [], {"walk -1": 29.0}, 0.0)
    perf_trace.level_warm_tick(152.6, [0.4], {"walk -1 setup": 120.0, "walk -1": 20.0, "install": 0.4}, 55.2)
    perf_trace.level_warm_tick(12.0, [], {"pack 0": 11.9}, 0.0)
    perf_trace.flush_idle()
    assert (
        "max tick 152.6 [walk -1 setup 120.0; walk -1 20.0; install 0.4; other 12.2; gc 55.2], installs 1"
        in _lines()[0]
    )
    perf_trace.level_warm_tick(5.0, [], {"pack 0": 4.0}, 0.0)
    perf_trace.flush_idle()
    assert "max tick 5.0 [pack 0 4.0; other 1.0; gc 0.0]" in _lines()[1], "the worst tick resets per line"


def test_cold_sprite_deltas_split_between_the_op_and_the_view_line(clock, monkeypatch):
    counts = [10, 100]
    monkeypatch.setattr(perf_trace, "_sprite_counter", lambda: tuple(counts))
    with perf_trace.op("place-unit"):
        counts[0] += 3
        counts[1] += 12
    counts[1] += 5  # decoded by the repaint that follows
    _timed_phase(clock, "repaint", 1)
    perf_trace.flush_idle()
    lines = _lines()
    assert lines[0].endswith("| sprites +3 files +12 frames")
    assert "sprites +0 files +5 frames" in lines[1]
    perf_trace.flush_idle()
    _timed_phase(clock, "repaint", 1)
    perf_trace.flush_idle()
    assert "sprites" not in _lines()[-1], "a warm repaint must not print a zero delta"


def test_a_mip_change_shows_on_the_view_line(clock):
    perf_trace.view_mip(-2)
    perf_trace.view_mip(-2)
    perf_trace.view_mip(-1)
    _timed_phase(clock, "repaint", 1)
    perf_trace.flush_idle()
    assert "mip -2->-1" in _lines()[0]


def test_a_stall_names_the_op_and_its_covering_phase(clock):
    with perf_trace.op("place-unit"):
        _timed_phase(clock, "unit_sources", 5)
        _timed_phase(clock, "unit_model_build", 900)
        clock.advance(45)
    perf_trace.stall(950.0)
    assert _lines()[-1] == (
        "perf stall 950ms (in op place-unit: unit_model_build; covered 905 of 950: unit_model_build 900, unit_sources 5)"
    )


def test_a_stall_in_a_repaint_names_the_level_build(clock):
    with perf_trace.phase("repaint"), perf_trace.level("build", -1):
        clock.advance(980)
    clock.advance(20)
    perf_trace.stall(1000.0)
    assert _lines()[-1] == (
        "perf stall 1000ms (in level -1 build (repaint); covered 980 of 1000: level -1 build (repaint) 980, repaint 980)"
    )


def test_an_uncovered_stall_is_untimed_with_the_last_op(clock):
    perf_trace.stall(300.0)
    assert _lines()[-1] == "perf stall 300ms (untimed, no op yet; covered 0 of 300)"
    with perf_trace.op("nudge"):
        clock.advance(3)
    clock.advance(1000)
    perf_trace.stall(800.0)
    assert _lines()[-1] == "perf stall 800ms (untimed, last op nudge 1000ms ago; covered 0 of 800)"


def test_an_op_nested_in_a_drag_is_never_the_stalls_last_op(clock):
    with perf_trace.op("nudge"):
        clock.advance(3)
    perf_trace.begin_drag()
    _timed_phase(clock, "patch", 2)
    perf_trace.step()
    # Ends after the drag's last step, as unit_model_build did in the stress log.
    with perf_trace.op("unit_model_build"):
        clock.advance(5)
    perf_trace.end_drag("draw")
    clock.advance(1000)
    perf_trace.stall(800.0)
    assert _lines()[-1] == "perf stall 800ms (untimed, last op nudge 1007ms ago; covered 0 of 800)"


# --- repaint phases (post-wave stress log follow-up) -------------------------


def test_repaint_phases_split_the_repaint_and_other_excludes_level_events(clock):
    perf_trace.arm("load")
    with perf_trace.phase("repaint"):
        with perf_trace.level("build", -2):
            clock.advance(350)
        for _ in range(3):
            with perf_trace.repaint_phase("composite"):
                clock.advance(300)
        with perf_trace.repaint_phase("blit"):
            clock.advance(6)
        clock.advance(4)
    perf_trace.first_paint_done()
    assert _lines()[-1].startswith(
        "perf load: repaint: 1 calls, 1260ms total (max 1260.0) [composite 900.0 x3; blit 6.0 x1; other 4.0], "
        "levels: -2 build 350.0ms (repaint)"
    )
    with perf_trace.phase("repaint"):
        clock.advance(1)
    perf_trace.flush_idle()
    assert "[" not in _lines()[-1], "the split resets with the line it printed on"


def test_a_repaint_phase_outside_a_repaint_records_nothing(clock):
    perf_trace.begin_drag()
    with perf_trace.phase("patch"), perf_trace.repaint_phase("composite"):
        clock.advance(5)
    perf_trace.step()
    with perf_trace.op("undo"), perf_trace.repaint_phase("composite"):
        clock.advance(5)
    assert perf_trace.repaint_phase("composite") is perf_trace._NULL_PHASE
    assert perf_trace._repaint_phases == {} and perf_trace._phase_sums == {"patch": pytest.approx(5.0)}


# --- garbage collection (post-wave stress log follow-up) ---------------------


def _collect(clock, generation: int, ms: float) -> None:
    """One collection as gc.callbacks would report it, under the fake clock."""
    perf_trace._on_gc("start", {"generation": generation})
    clock.advance(ms)
    perf_trace._on_gc("stop", {"generation": generation})


def test_a_gen2_collection_prints_on_the_op_line_it_landed_in(clock):
    _collect(clock, 2, 7)  # before the op: not its collection
    with perf_trace.op("undo"):
        with perf_trace.phase("unit_sources"):
            _collect(clock, 2, 55)
            clock.advance(5)
        clock.advance(1)
    perf_trace.flush_pending_op()
    assert _lines()[-1] == "perf op undo: 61ms | unit_sources 60.0 untimed 1.0 | gc gen2 55.0ms"


def test_young_collections_count_in_gc_ms_but_print_nowhere(clock):
    with perf_trace.op("undo"):
        _collect(clock, 0, 2)
        _collect(clock, 1, 3)
    perf_trace.flush_pending_op()
    assert "gc" not in _lines()[-1]
    assert perf_trace._gc_total_ms == pytest.approx(5.0)
    assert perf_trace.gc_ms() is None, "with the hook out gc is unmeasured, not zero"


def test_gen2_collections_in_a_drag_print_on_its_line(clock):
    perf_trace.begin_drag()
    for _ in range(2):
        with perf_trace.phase("patch"):
            _collect(clock, 2, 20)
        perf_trace.step()
    _collect(clock, 2, 30)  # stroke-end handler, after the last step
    perf_trace.end_drag("draw")
    assert _lines()[-1] == "  | gc gen2 70.0ms x3 (max 30.0)"


def test_a_lone_gen2_collection_still_flushes_a_view_line(clock):
    _collect(clock, 2, 40)
    perf_trace.flush_idle()
    assert _lines()[-1].startswith("perf view: gc gen2 40.0ms")
    perf_trace.flush_idle()
    assert len(_lines()) == 1, "a collection is reported once"


def test_a_stall_names_a_covering_gen2_collection_as_its_cause(clock):
    with perf_trace.op("undo"):
        with perf_trace.phase("unit_sources"):
            clock.advance(10)
            _collect(clock, 2, 150)
        clock.advance(40)
    perf_trace.stall(200.0)
    assert _lines()[-1] == (
        "perf stall 200ms (in op undo: gc gen2; covered 160 of 200: unit_sources 160, gc gen2 150)"
    )


def test_a_stall_lists_a_smaller_gen2_collection_beside_its_cause(clock):
    with perf_trace.op("undo"):
        with perf_trace.phase("unit_sources"):
            _collect(clock, 2, 40)
            clock.advance(150)
        clock.advance(10)
    perf_trace.stall(200.0)
    assert _lines()[-1] == (
        "perf stall 200ms (in op undo: unit_sources; gc gen2 40.0ms; covered 190 of 200: unit_sources 190, gc gen2 40)"
    )


def test_an_untimed_stall_during_a_drag_says_so(clock):
    with perf_trace.op("tool-switch"):
        clock.advance(1)
    perf_trace.begin_drag()
    clock.advance(400)
    perf_trace.stall(300.0)
    assert _lines()[-1] == "perf stall 300ms (untimed, during a drag; covered 0 of 300)"


def test_the_gc_hook_does_nothing_while_disabled(clock):
    perf_trace.enable(False)
    _collect(clock, 2, 50)
    assert not perf_trace._gc_raw and perf_trace._gc_total_ms == 0.0


def test_the_hook_times_a_real_collection_and_comes_out_again():
    perf_trace.enable(True)
    perf_trace.set_gc_hook(True)
    perf_trace.set_gc_hook(True)
    assert gc.callbacks.count(perf_trace._on_gc) == 1, "installing twice must not double-count"
    gc.collect()
    perf_trace.set_gc_hook(False)
    assert perf_trace._on_gc not in gc.callbacks
    assert len(perf_trace._gc_raw) == 1 and perf_trace._gc_total_ms > 0.0
    gc.collect()
    assert len(perf_trace._gc_raw) == 1, "a removed hook records nothing"


def test_a_collection_on_another_thread_waits_for_the_gui_thread_drain():
    perf_trace.enable(True)
    perf_trace.set_gc_hook(True)
    worker = threading.Thread(target=gc.collect)
    worker.start()
    worker.join()
    perf_trace.set_gc_hook(False)
    assert len(perf_trace._gc_raw) == 1 and not perf_trace._recent
    perf_trace.flush_idle()
    assert "gc gen2" in _lines()[-1] and not perf_trace._gc_raw


# --- terrain texture loads (first-paint-terrain-textures plan, Step 1) --------


def _texture_load(clock, ms: float) -> None:
    with perf_trace.texture_load():
        clock.advance(ms)


def test_texture_loads_print_on_the_op_line_they_ran_in_and_only_when_non_zero(clock):
    with perf_trace.op("re-render"):
        _texture_load(clock, 30)
        _texture_load(clock, 12)
    perf_trace.flush_pending_op()
    assert _lines()[-1] == "perf op re-render: 42ms | untimed 42.0 | textures +2 files 42.0ms"
    with perf_trace.op("undo"):
        clock.advance(1)
    perf_trace.flush_pending_op()
    assert "textures" not in _lines()[-1], "an op that loaded nothing must not print a zero field"


def test_texture_loads_in_a_first_paint_print_on_the_load_line(clock):
    perf_trace.arm("load")
    with perf_trace.phase("repaint"):
        _texture_load(clock, 700)
        clock.advance(300)
    perf_trace.first_paint_done()
    assert _lines()[-1].startswith("perf load: repaint: 1 calls, 1000ms total (max 1000.0), textures +1 files 700.0ms")
    _timed_phase(clock, "repaint", 1)
    perf_trace.flush_idle()
    assert "textures" not in _lines()[-1], "a load is reported once"


def test_texture_load_is_the_shared_null_context_and_records_nothing_while_disabled():
    assert perf_trace.texture_load() is perf_trace._NULL_PHASE
    with perf_trace.texture_load():
        pass
    assert not perf_trace._texture_raw
    assert perf_trace._textures_pending.files == 0


def test_a_texture_load_on_another_thread_waits_for_the_gui_thread_drain(clock):
    worker = threading.Thread(target=_texture_load, args=(clock, 5))
    worker.start()
    worker.join()
    assert len(perf_trace._texture_raw) == 1 and perf_trace._textures_pending.files == 0
    _timed_phase(clock, "repaint", 1)
    perf_trace.flush_idle()
    assert "textures +1 files 5.0ms" in _lines()[-1] and not perf_trace._texture_raw


def test_prefetched_textures_count_as_files_and_only_waits_and_own_loads_as_blocked_ms(clock):
    with perf_trace.op("load"):
        for _ in range(3):
            with perf_trace.texture_prefetch():
                clock.advance(200)
    perf_trace.arm("load")
    with perf_trace.phase("repaint"):
        with perf_trace.texture_wait():
            clock.advance(40)
        _texture_load(clock, 25)  # a file the prefetch missed
        clock.advance(100)
    perf_trace.first_paint_done()
    op_line, load_line = _lines()
    assert op_line.endswith("| textures +3 files 0.0ms (prefetched 3 in 600.0ms)"), op_line
    assert "textures +1 files 65.0ms (wait 40.0)" in load_line, load_line


def test_texture_wait_and_prefetch_record_nothing_while_disabled():
    assert perf_trace.texture_wait() is perf_trace._NULL_PHASE
    assert perf_trace.texture_prefetch() is perf_trace._NULL_PHASE


# --- stall coverage, drag window, spans, young gc (set-elevation-untimed-stalls plan, Step 1) ---


def test_a_stall_prints_its_covered_time_once_and_its_top_three_entries(clock):
    """Covered is the union, so the gc inside stroke_commit counts once (89, not 97)."""
    perf_trace.begin_drag()
    _timed_phase(clock, "patch", 14)
    perf_trace.step()
    clock.advance(100)
    with perf_trace.phase("stroke_commit"):
        _collect(clock, 2, 8)
        clock.advance(22)
    _timed_phase(clock, "footprint_refresh", 42)
    _timed_phase(clock, "overlays", 3)
    clock.advance(87)
    perf_trace.stall(276.0)
    assert _lines()[-1] == (
        "perf stall 276ms (untimed, during a drag; gc gen2 8.0ms; "
        "covered 89 of 276: footprint_refresh 42, stroke_commit 30, patch 14)"
    )


def test_a_stall_that_blocked_past_the_release_still_says_during_a_drag(clock):
    with perf_trace.op("undo"):
        clock.advance(2)
    perf_trace.begin_drag()
    _timed_phase(clock, "patch", 10)
    perf_trace.step()
    clock.advance(150)  # the release blocks, then the drag ends
    perf_trace.end_drag("set-elevation")
    clock.advance(50)
    perf_trace.stall(250.0)
    assert _lines()[-1] == "perf stall 250ms (untimed, during a drag; covered 10 of 250: patch 10)"


def test_a_stall_window_longer_than_the_recent_buffer_prints_covered_as_a_lower_bound(clock):
    """600 1 ms phases in a 700 ms window: the buffer keeps the last 512, so
    the figure is a floor. Fewer entries, or a full buffer whose evictions all
    precede the window, print the plain figure."""
    for _ in range(12):
        _timed_phase(clock, "patch", 1)
    perf_trace.stall(700.0)
    assert _lines()[-1] == "perf stall 700ms (untimed, no op yet; covered 12 of 700: patch 12)"
    for _ in range(600):
        _timed_phase(clock, "patch", 1)
    perf_trace.stall(700.0)
    assert _lines()[-1] == "perf stall 700ms (untimed, no op yet; covered >= 512 of 700: patch 512)"
    clock.advance(50)
    perf_trace.stall(40.0)
    assert _lines()[-1] == "perf stall 40ms (untimed, no op yet; covered 0 of 40)"


def test_spans_stay_out_of_step_totals_and_print_wall_and_untimed(clock):
    """Wall: press 20 (5 of it before begin_drag) + step 10 + release up to the
    flush 106. Untimed: minus the 29 ms of phases inside spans; footprint_refresh
    ran between events, so it is a step phase but in neither figure."""
    with perf_trace.span("press"):
        clock.advance(5)
        perf_trace.begin_drag()
        _timed_phase(clock, "stroke_snapshot", 3)
        with perf_trace.span("step"):
            _timed_phase(clock, "patch", 10)
            clock.advance(2)
        perf_trace.step()
    _timed_phase(clock, "footprint_refresh", 40)
    with perf_trace.span("step"):
        _timed_phase(clock, "patch", 10)
    perf_trace.step()
    with perf_trace.span("release"):
        clock.advance(100)
        _timed_phase(clock, "stroke_commit", 6)
        perf_trace.flush("set-elevation")
        clock.advance(7)
        perf_trace.end_drag("set-elevation")
        clock.advance(9)
    header, phases, end = _lines()
    assert header.startswith(
        "perf drag set-elevation: 2 steps, 63ms total, 31.5ms/step (max 50.0), wall 136ms, untimed 107ms, composite"
    ), header
    assert "span" not in phases and "press" not in phases and end == "  | end: stroke_commit 6.0"
    assert perf_trace._span_stack == []

    # The release's tail after the flush must not leak into the next drag.
    with perf_trace.span("press"):
        perf_trace.begin_drag()
        with perf_trace.span("step"):
            _timed_phase(clock, "patch", 4)
        perf_trace.step()
    perf_trace.end_drag("set-elevation")
    assert ", wall 4ms, untimed 0ms," in _lines()[-2]


def test_a_long_young_collection_names_a_stall_and_prints_nowhere_else(clock):
    with perf_trace.op("undo"):
        _collect(clock, 0, 3)  # under YOUNG_GC_MIN_MS
        _collect(clock, 1, 150)
        clock.advance(10)
    perf_trace.stall(200.0)
    assert _lines()[-1] == "perf stall 200ms (in op undo: gc gen1; covered 150 of 200: gc gen1 150)"
    perf_trace.flush_pending_op()
    assert _lines()[-1] == "perf op undo: 163ms | untimed 163.0"
