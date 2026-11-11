"""tools/bench_gui_zoom.py's bookkeeping: log-line parsing, the per-click
latency split, the G1-G3 gates and the signature routing, all Qt-free, plus
the canvas paint wrapper's per-paint reset. The log lines are real ones from
the first old-allies capture (TASK-031.26); the zoom pass itself needs a real
X server and runs only by hand."""

from __future__ import annotations

import pytest

from tools import bench_gui_zoom as bz

import conftest

# Stepped, sprites on, cold roll: two walk-dominant builds inside paints.
VIEW_COLD_ROLL = (
    "[06:10:58] perf view: repaint: 34 calls, 888ms total (max 413.5) [blit 140.7 x88; prepare 63.8 x36;"
    " join 22.7 x5; composite 11.8 x2; other 52.7], warm: 33 ticks, 93 chunks, 83ms total (max tick 8.8,"
    " max chunk 6.8), worker 83 chunks, 326ms total (max 7.6), level 23 ticks, max tick 13.2 [pack 0 13.2;"
    " other 0.0; gc 0.1; max step 2.2 (pack 0)], installs 2 (max 0.0), levels: 0 build 207.6ms [walk 184.3"
    " install 23.3] (repaint), 0 pack 5.4ms (repaint), 1 build 377.4ms [walk 351.8 install 25.6] (repaint),"
    " 1 pack 5.5ms (repaint), 0 install 0.0ms (level-warm), 1 install 0.0ms (level-warm), mip -2->-1 -1->0"
    " 0->1 1->0 0->-1 -1->-2, gc gen2 94.1ms, composite native, platform xcb"
)
# Stepped, sprites off, first visit: an assembly-only build inside the paint.
VIEW_SPRITES_OFF = (
    "[06:12:22] perf view: repaint: 1 calls, 118ms total (max 118.4) [prepare 73.4 x26; join 7.1 x3;"
    " blit 3.9 x4; composite 6.1 x1; other 2.8], warm: 6 ticks, 24 chunks, 22ms total (max tick 5.5,"
    " max chunk 2.1), worker 23 chunks, 46ms total (max 3.8), level 3 ticks, max tick 14.0 [pack -1 14.0;"
    " other 0.0; gc 0.0; max step 3.2 (pack -1)], installs 1 (max 0.0), levels: -1 build 20.4ms [walk 0.0"
    " install 20.4] (repaint), -1 pack 4.7ms (repaint), -1 install 0.0ms (level-warm), mip -2->-1,"
    " gc gen2 1.8ms, composite native, platform xcb"
)
VIEW_RESIDENT = (
    "[06:09:14] perf view: repaint: 1 calls, 23ms total (max 23.0) [prepare 6.8 x9; join 9.4 x2; blit 4.2 x6;"
    " other 2.5], warm: 6 ticks, 15 chunks, 15ms total (max tick 6.1, max chunk 5.5), worker 15 chunks,"
    " 65ms total (max 6.0), mip 0->1, composite native, platform xcb"
)
OP_ZOOM_ROLL = "[06:21:52] perf op zoom x41: 3ms total, 0.1ms mean (max 0.1) | untimed 0.1"
OP_ZOOM_ONE = "[00:37:48] perf op zoom: 0ms | untimed 0.1"  # GH #78 tester log 4
STALL = (
    "[06:10:55] perf stall 255ms (in level 0 build (repaint); covered 253 of 255: repaint 241,"
    " level 0 build (repaint) 186, level-warm tick 12)"
)


def test_timestamps_strip():
    assert bz.strip_ts(OP_ZOOM_ONE) == "perf op zoom: 0ms | untimed 0.1"
    assert bz.strip_ts("no stamp") == "no stamp"


def test_level_events_parse_with_split_and_where():
    events = bz.parse_levels(bz.strip_ts(VIEW_COLD_ROLL))
    assert [(e.mip, e.kind, e.ms, e.where) for e in events] == [
        (0, "build", 207.6, "repaint"), (0, "pack", 5.4, "repaint"), (1, "build", 377.4, "repaint"),
        (1, "pack", 5.5, "repaint"), (0, "install", 0.0, "level-warm"), (1, "install", 0.0, "level-warm"),
    ]
    assert events[2].split == {"walk": 351.8, "install": 25.6}
    assert events[1].split == {}


def test_the_levels_section_stops_before_the_mip_changes():
    """`mip -2->-1` and `gc gen2` follow the levels and must not read as entries."""
    events = bz.parse_levels(VIEW_SPRITES_OFF)
    assert [(e.mip, e.kind) for e in events] == [(-1, "build"), (-1, "pack"), (-1, "install")]
    assert bz.parse_levels(VIEW_RESIDENT) == []


def test_the_logged_counts_read_repaint_builds_and_stalls_but_not_the_load_line():
    load = "[06:10:48] perf load: repaint: 1 calls, 770ms total (max 770.4), levels: -2 build 519.7ms [walk 495.9 install 23.8] (repaint)"
    assert bz.logged_counts([VIEW_COLD_ROLL, STALL, load, OP_ZOOM_ROLL]) == (2, 1)
    assert bz.logged_counts([VIEW_SPRITES_OFF]) == (1, 0)
    assert bz.logged_counts([VIEW_RESIDENT]) == (0, 0)


def test_a_cross_check_flags_wrappers_and_log_disagreeing():
    assert bz.cross_check("notch #5", (1, 0), [VIEW_SPRITES_OFF]).endswith("stalls 0 seen, 0 logged")
    assert bz.cross_check("notch #5", (0, 0), [VIEW_SPRITES_OFF]).endswith("MISMATCH")
    assert bz.cross_check("notch #6", (0, 0), [VIEW_RESIDENT]) is None


def test_cross_checks_group_a_roll_and_count_disagreements():
    """A roll drains per burst, so its builds are compared over the whole pass."""
    a, b = _click("roll", 0.0, end=0.03, idx=1), _click("roll", 0.03, end=1.0, idx=2)
    b.lines = [VIEW_COLD_ROLL, STALL]
    builds = [_build(0.05, VIEW_COLD_ROLL, 0), _build(0.3, VIEW_COLD_ROLL, 2)]
    stall = bz.Activity(0.3, 0.55, "stall", "255ms")
    lines = bz.cross_checks([a, b], [], [*builds, stall])
    assert lines == [
        "  cross-check roll (2 clicks): builds in paint 2 seen, 2 logged; stalls 1 seen, 1 logged",
        "  cross-check: 0 disagreements",
    ]
    assert bz.cross_checks([a, b], [], builds)[-1] == "  cross-check: 1 disagreements"


def test_the_op_zoom_count_sums_merged_and_single_lines():
    lines = [bz.strip_ts(x) for x in (OP_ZOOM_ROLL, VIEW_RESIDENT, OP_ZOOM_ONE)]
    assert bz.op_zoom_count(lines) == 42


def test_a_stall_line_parses():
    ms, cause = bz.parse_stall(bz.strip_ts(STALL))
    assert ms == 255
    assert cause.startswith("in level 0 build (repaint); covered 253 of 255")
    assert bz.parse_stall(OP_ZOOM_ONE) is None


def test_stall_parser_keeps_thread_suffix_and_old_logs_stay_unclassified():
    raw = bz.strip_ts(STALL)[:-1] + "; thread fault: cpu 3.0ms runq 2.0ms maj 1 min 7 vcsw 1 ivcsw 2)"
    ms, cause = bz.parse_stall(raw)
    assert ms == 255 and cause.endswith("ivcsw 2")
    assert bz.parse_thread(cause) == {
        "kind": "fault", "text": "cpu 3.0ms runq 2.0ms maj 1 min 7 vcsw 1 ivcsw 2", "cpu_ms": 3.0,
        "runq_ms": 2.0, "maj": 1, "min": 7, "vcsw": 1, "ivcsw": 2,
    }
    assert bz.parse_thread(bz.parse_stall(bz.strip_ts(STALL))[1]) is None


def test_recorder_exports_each_slow_preframe_once_not_view_maxima_or_click_duplicates(monkeypatch):
    import json

    from descape import debug_log, perf_trace
    from descape.thread_stats import Delta

    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_enabled", True)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    rec = bz._Recorder()
    rec.install()
    try:
        stats = Delta(50, 2, 40, 0, 4, 1, 2)
        fetch = {"render_ms": 47, "convert_ms": 2, "k": 1, "width": 32, "height": 64}
        perf_trace.preframe_tick(50, "fetch", stats, fetch)
        perf_trace.preframe_tick(49.999, "strip", stats)
        perf_trace.flush_idle()  # the max tick line must not become another raw event
        perf_trace.stall(250, Delta(300, 3, 250, 0))
        got = json.loads(json.dumps(rec.diagnostics()))
        assert len(got["slow_preframe_ticks"]) == 1 and len(got["stalls"]) == 1
        tick = got["slow_preframe_ticks"][0]
        assert tick["detail"] == "fetch" and tick["fetch"] == fetch and tick["thread"] == bz.thread_record(stats)
        assert tick["raw"].startswith("perf slow tick preframe fetch 50.0 ms: runq;")
        stall = got["stalls"][0]
        assert stall["thread"]["kind"] == "runq" and "; thread runq:" in stall["detail"]
        assert stall["raw"].startswith("perf stall 250ms (")
        click = _click(t=rec.activities[-1].start, end=rec.activities[-1].end + 1)
        row = bz.format_row(click, bz.click_stats(click, [], rec.activities))
        assert "thread runq: cpu 3.0ms runq 250.0ms maj 0" in row
        assert bz.gates([click], [], rec.activities)[1].met is False, "Step 1 must not exempt env stalls"
        assert sum("perf slow tick" in entry.text() for entry in debug_log.entries()) >= 1
    finally:
        rec.restore()


def test_runtime_metadata_reads_actual_loaded_versions_and_interpreter():
    import sys
    from pathlib import Path

    import numpy
    from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, qVersion

    try:
        flag = Path("/proc/sys/kernel/sched_schedstats").read_text().strip()
    except OSError:
        flag = "unavailable"
    assert bz.runtime_metadata() == {"qt": qVersion(), "qt_compiled": QT_VERSION_STR, "pyqt": PYQT_VERSION_STR,
                                     "numpy": numpy.__version__, "interpreter": sys.executable, "python": sys.version,
                                     "sched_schedstats": flag}


@pytest.mark.parametrize("flag", ["0\n", "1\n", "malformed\n", FileNotFoundError(), PermissionError()])
def test_runtime_metadata_preserves_sysctl_context_without_disabling_sample(monkeypatch, flag):
    from pathlib import Path
    from types import SimpleNamespace

    from descape import thread_stats

    reads = []

    def read_text(path):
        reads.append(path)
        if path == thread_stats.SCHEDSTAT:
            return "1000000 0 1\n"
        assert path == Path("/proc/sys/kernel/sched_schedstats")
        if isinstance(flag, OSError):
            raise flag
        return flag

    usage = SimpleNamespace(ru_majflt=0, ru_minflt=0, ru_nvcsw=0, ru_nivcsw=0)
    monkeypatch.setattr(thread_stats.sys, "platform", "linux")
    monkeypatch.setattr(thread_stats, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: usage))
    monkeypatch.setattr(bz.Path, "read_text", read_text)
    expected = "unavailable" if isinstance(flag, OSError) else flag.strip()
    assert bz.runtime_metadata()["sched_schedstats"] == expected
    sampled = thread_stats.sample()
    assert sampled.cpu_ns == 1000000 and sampled.runq_ns == 0
    assert reads == [Path("/proc/sys/kernel/sched_schedstats"), thread_stats.SCHEDSTAT]


def test_recorder_accepts_baseline_stall_and_preframe_callback_signatures(monkeypatch):
    from descape import debug_log, perf_trace

    calls = []
    monkeypatch.setattr(perf_trace, "preframe_tick", lambda ms, kind: calls.append((ms, kind)))

    def stall(ms):
        calls.append((ms,))
        debug_log.log(f"perf stall {ms}ms (untimed, no op yet)")

    monkeypatch.setattr(perf_trace, "stall", stall)
    monkeypatch.delattr(perf_trace, "SLOW_TICK_MS")
    rec = bz._Recorder()
    rec.install()
    try:
        perf_trace.stall(250)
        perf_trace.preframe_tick(100, "fetch")
        assert calls == [(250,), (100, "fetch")]
        assert rec.diagnostics()["stalls"][0]["thread"] is None
        assert rec.diagnostics()["slow_preframe_ticks"][0]["raw"] == "", "baseline reused a stall as a slow tick"
    finally:
        rec.restore()


@pytest.fixture
def d1_delta():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Delta:
        """The e128d3d Delta shape and nominal classifier, with no confidence API."""

        wall_ms: float
        cpu_ms: float | None
        runq_ms: float | None = None
        maj: int | None = None
        min: int | None = None
        vcsw: int | None = None
        ivcsw: int | None = None

        def kind(self):
            if self.wall_ms <= 0 or self.cpu_ms is None:
                return "mixed"
            half = self.wall_ms * 0.5
            if self.cpu_ms >= half:
                return "cpu"
            if self.runq_ms is None:
                return "off-cpu"
            if self.runq_ms >= half:
                return "runq"
            if max(0.0, self.wall_ms - self.cpu_ms - self.runq_ms) >= half:
                if self.maj is None:
                    return "off-cpu"
                return "fault" if self.maj > 0 else "blocked"
            return "mixed"

        def text(self):
            parts = [f"{name} {value:.1f}ms" for name, value in (("cpu", self.cpu_ms), ("runq", self.runq_ms))
                     if value is not None]
            parts += [f"{name} {getattr(self, name)}" for name in ("maj", "min", "vcsw", "ivcsw")
                      if getattr(self, name) is not None]
            return " ".join(parts)

    return Delta


@pytest.mark.parametrize("callback", ["stall", "preframe"])
@pytest.mark.parametrize(("runq", "maj", "nominal"), [(250, 0, "runq"), (2, 1, "fault")])
def test_wrapped_callbacks_keep_d1_delta_unqualified(monkeypatch, d1_delta, callback, runq, maj, nominal):
    import json
    from dataclasses import asdict

    from descape import perf_trace

    for name, value in perf_trace._fresh_state().items():
        monkeypatch.setattr(perf_trace, name, value)
    monkeypatch.setattr(perf_trace, "_enabled", True)
    monkeypatch.setattr(perf_trace, "_idle_scheduler", None)
    stats = d1_delta(300, 3, runq, maj, 7, 1, 2)
    rec = bz._Recorder()
    rec.install()
    try:
        if callback == "stall":
            perf_trace.stall(250, stats)
            group = "stalls"
        else:
            perf_trace.preframe_tick(300, "fetch", stats)
            group = "slow_preframe_ticks"
        data = json.loads(json.dumps(rec.diagnostics()))
        assert data["counts"][group] == {"raw": 1, "eligible": 0}
        activity = data[group][0]
        assert activity["thread"] == {
            **asdict(stats), "kind": "unqualified", "nominal_kind": nominal,
            "qualified": False, "eligible": False, "text": stats.text(),
        }
        assert nominal in activity["raw"]
        assert all(name not in activity["thread"] for name in
                   ("sampling_ms", "maj_bounds", "possible_kinds", "wall_start", "wall_end"))
    finally:
        rec.restore()


def test_thread_record_keeps_extended_native_confidence_metadata():
    from dataclasses import asdict

    from descape.thread_stats import Delta

    stats = Delta(200, 190, 0, 0, sampling_ms=2, maj_bounds=(0, 0), wall_start=1, wall_end=1.2)
    assert bz.thread_record(stats) == {
        **asdict(stats), "kind": "cpu", "nominal_kind": "cpu", "possible_kinds": ["cpu"],
        "qualified": True, "eligible": True, "text": stats.text(),
    }


def _build(start: float, line: str, index: int) -> bz.Activity:
    """An Activity for the `index`th level entry of a real line, starting at `start` s."""
    e = bz.parse_levels(line)[index]
    return bz.Activity(start, start + e.ms / 1000, "level", f"{e.mip} {e.kind} ({e.where})", e.split, e.where)


def _click(sweep="notch", t=0.0, *, accepted=True, mip=(0, 0), resident=True, end=None, idx=1) -> bz.Click:
    c = bz.Click(sweep, idx, 1, accepted, 100.0, 125.0 if accepted else 100.0, mip[0], mip[1], resident, 0.7,
                 t, t + 0.001, t + 0.002)
    c.window_end = t + 0.5 if end is None else end
    return c


def test_a_click_queued_behind_a_long_paint_is_felt_until_its_own_frame_ends():
    """Roll #13-#19 shape: the handler ran 300 ms after the click, and its first
    frame starts after the next click's handler; latency still runs to its end."""
    first = bz.Click("roll", 1, 1, True, 100, 125, -1, 0, False, 0.6, 0.000, 0.001, 0.002)
    queued = bz.Click("roll", 2, 1, True, 125, 156, 0, 0, True, 0.7, 0.030, 0.400, 0.401)
    last = bz.Click("roll", 3, 1, True, 156, 195, 0, 0, True, 0.9, 0.060, 0.402, 0.403)
    first.window_end, queued.window_end, last.window_end = queued.t_entry, last.t_entry, 1.0
    frames = [bz.Frame(0.003, 0.390, {"levels": 300.0}, 0), bz.Frame(0.404, 0.420, {"blit": 10.0}, 0)]
    st = bz.click_stats(queued, frames, [])
    assert st.first is frames[1]
    assert st.queue_ms == pytest.approx(370.0)
    assert st.to_frame_ms == pytest.approx(390.0)
    assert bz.click_stats(first, frames, []).to_frame_ms == pytest.approx(390.0)


def test_g1_is_each_resident_revisit_notch_s_worst_canvas_repaint_not_its_slowest_frame():
    """The plan's G1 is the canvas repaint; a frame's flush and other items are context only,
    and the worst canvas can sit in a frame other than the slowest one."""
    clicks = [_click("revisit", 0.0, idx=1), _click("revisit", 1.0, idx=2), _click("revisit", 2.0, resident=False, idx=3)]
    frames = [
        bz.Frame(0.003, 0.0102, {"blit": 3.0, "other": 1.0}, 1, 4.1),  # 7.2 ms frame, 4.0 canvas
        bz.Frame(1.003, 1.0110, {"blit": 1.0}, 1, 1.2),  # notch 2's slowest frame, mostly flush
        bz.Frame(1.020, 1.0244, {"blit": 3.2, "other": 1.0}, 1, 4.3),  # notch 2's worst canvas, 4.2
        bz.Frame(2.003, 2.050, {"blit": 40.0}, 0, 40.0),  # not built and warm: not G1's
    ]
    g1 = bz.gates(clicks, frames, [])[0]
    assert g1.met is True
    assert g1.dev_ms == pytest.approx(4.2)
    assert "whole frame max 8.0" in g1.line
    frames[2] = bz.Frame(1.020, 1.0244, {"blit": 3.3, "other": 1.0}, 1, 4.4)
    g1 = bz.gates(clicks, frames, [])[0]
    assert g1.met is False
    assert g1.signatures[0].startswith("revisit #2 mip 0->0 resid 0.700: blit at mip 1 (no task) [blit 3.3")


def test_g2_counts_one_build_once_however_many_windows_overlap_it():
    a, b = _click("notch", 0.0, end=0.3), _click("notch", 0.3, end=1.0, idx=2)
    build = _build(0.1, VIEW_SPRITES_OFF, 0)  # 20.4 ms, ends inside a's window
    spanning = bz.Activity(0.25, 0.45, "level", "0 build (repaint)", {"walk": 150.0}, "repaint")
    g2 = bz.gates([a, b], [], [build, spanning])[1]
    assert g2.met is False
    assert g2.line.startswith("gate G2: 2 builds in paint, 0 stalls over 2 notches, worst 200 ms")
    assert "(no task)" in g2.signatures[1] and "TASK-031.24" in g2.signatures[0]


def test_top_down_flat_s_lazy_layers_count_as_builds_in_paint():
    """Top-down Flat builds a level as `flat-draws`/`flat-icons`, never `build`."""
    draws = bz.Activity(0.1, 0.1396, "level", "-1 flat-draws (repaint)", {}, "repaint")
    g2 = bz.gates([_click("notch")], [], [draws])[1]
    assert g2.met is False
    assert "level build in paint, flat-draws dominant (no task)" in g2.signatures[0]
    warm = bz.Activity(0.1, 0.12, "level", "-1 flat-icons (level-warm)", {}, "level-warm")
    assert bz.gates([_click("notch")], [], [warm])[1].met is True


def test_g2_is_met_with_only_level_warm_installs():
    install = _build(0.1, VIEW_SPRITES_OFF, 2)
    assert bz.gates([_click("notch")], [], [install])[1].met is True


def test_g3_skips_refused_clicks_and_reads_the_accepted_worst():
    clicks = [_click("roll", 0.0, end=0.03, idx=1), _click("roll", 0.03, accepted=False, end=0.06, idx=2)]
    frames = [bz.Frame(0.003, 0.020, {"blit": 10.0}, -2, 12.0)]
    g3 = bz.gates(clicks, frames, [])[2]
    assert g3.met is True
    assert "(1 refused," in g3.line
    assert g3.dev_ms == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("sweep", "index", "want"),
    [("roll", 2, "TASK-031.25"), ("notch", 2, "TASK-031.24"), ("roll", 0, "TASK-031.25")],
)
def test_walk_dominant_builds_route_to_the_two_candidate_tasks(sweep, index, want):
    assert bz.build_signature(_build(0.0, VIEW_COLD_ROLL, index), sweep)[0].startswith(want)


def test_an_assembly_only_build_has_no_task():
    cand, phase, ms = bz.build_signature(_build(0.0, VIEW_SPRITES_OFF, 0), "notch")
    assert (cand, phase, ms) == ("level build in paint, install dominant (no task)", "build install", 20.4)


@pytest.mark.parametrize(
    ("bracket", "mip", "view_ms", "frame_ms", "want"),
    [
        ({"blit": 10.8, "other": 1.7}, 0, 12.6, 15.1, "blit at mip 0 (no task)"),
        ({"blit": 15.6, "other": 2.7}, -2, 18.5, 22.2, "blit at mip -2 (no task)"),
        ({"blit": 1.2, "other": 0.2}, 1, 1.5, 3.5, "Qt backing-store flush"),
        ({"prepare": 4.9, "join": 7.0, "blit": 3.9, "other": 2.7}, 1, 18.6, 20.9, "TASK-031.10/031.11"),
        ({"blit": 1.0, "other": 6.0}, 1, 7.2, 8.0, "untraced code in the canvas paint"),
        ({"blit": 1.0, "other": 0.5}, 1, 6.0, 7.0, "untraced: other scene items"),
        ({"blit": 1.9, "other": 2.8, "reduce": 9.0}, -2, 14.0, 16.0, "the coarsest level's box reduce"),
    ],
    ids=["level0-blit", "coarse-blit", "flush", "chunk", "canvas-other", "other-items", "reduce"],
)
def test_a_slow_frame_routes_by_its_bracket(bracket, mip, view_ms, frame_ms, want):
    frame = bz.Frame(0.0, frame_ms / 1000, bracket, mip, view_ms)
    assert bz.bracket_signature(frame, mip)[0].startswith(want)


def test_a_frame_with_a_build_inside_routes_by_the_build():
    frame = bz.Frame(0.0, 0.5, {"levels": 377.4, "blit": 5.0}, 1, 400.0)
    build = _build(0.01, VIEW_COLD_ROLL, 2)
    assert bz.frame_signature(frame, [build], "roll")[0].startswith("TASK-031.25")


def test_a_queue_held_by_a_level_warm_walk_step_over_budget_is_the_031_80_step_tail():
    """The felt sub-150 ms pause: a click waits on one long level-warm tick."""
    click = _click("roll", 0.0, end=1.0)
    click.t_entry, click.t_exit = 0.060, 0.061
    frames = [bz.Frame(0.0625, 0.0665, {"blit": 2.0}, 1, 2.5)]
    tick = bz.Activity(0.002, 0.060, "level-warm", "", {"walk 1": 57.8, "other": 0.2})
    st = bz.click_stats(click, frames, [tick])
    cand, phase, ms = bz.latency_signature(click, st, frames, [tick])
    assert cand == bz.LEVEL_WARM_STEP_TAIL
    assert phase == "queue: level-warm walk"
    assert ms == pytest.approx(58.0)


def _walk_tick(ms: float, budget_ms: float | None) -> bz.Activity:
    return bz.Activity(0.0, ms / 1000, "level-warm", "", {"walk 1": ms - 0.1}, budget_ms=budget_ms)


@pytest.mark.parametrize(
    ("ms", "budget", "sweep", "want"),
    [
        (4.5, 3.0, "roll", bz.LEVEL_WARM_STEP_TAIL),  # over its own 3 ms budget + 1
        (13.5, 12.0, "roll", bz.LEVEL_WARM_STEP_TAIL),
        (13.5, None, "roll", bz.LEVEL_WARM_STEP_TAIL),  # no budget recorded: the idle one
        (12.4, 12.0, "roll", bz.LEVEL_WARM_FIRST_NOTCH),
        (12.4, None, "roll", bz.LEVEL_WARM_FIRST_NOTCH),
        (3.6, 3.0, "roll", bz.LEVEL_WARM_TICK),
        (12.4, 12.0, "notch", bz.LEVEL_WARM_TICK),  # outside a roll the idle budget is expected
    ],
    ids=["step-tail-input", "step-tail-idle", "step-tail-unrecorded", "first-notch", "first-notch-unrecorded",
         "input-tick", "notch-idle-tick"],
)
def test_a_walk_tick_is_judged_against_its_own_budget(ms, budget, sweep, want):
    assert bz.activity_signature(_walk_tick(ms, budget), sweep)[0] == want


def test_level_warm_totals_count_a_pass_by_budget():
    acts = [
        bz.Activity(0.1, 0.112, "level-warm", budget_ms=12.0),
        bz.Activity(0.2, 0.203, "level-warm", budget_ms=3.0),
        bz.Activity(0.3, 0.312, "level-warm"),  # unrecorded: the idle budget
        bz.Activity(0.4, 0.45, "gc", "gen2"),
        bz.Activity(1.5, 1.512, "level-warm", budget_ms=12.0),  # outside the pass
    ]
    got = bz.level_warm_totals(acts, 0.0, 1.0)
    assert got == {"ticks": 3, "ms": 27.0, "by_budget": {"12": 2, "3": 1}}


def test_a_level_warm_tick_that_is_not_walk_dominant_stays_a_plain_tick():
    tick = bz.Activity(0.0, 0.020, "level-warm", "", {"pack 1": 19.0}, budget_ms=3.0)
    assert bz.activity_signature(tick, "roll")[0] == bz.LEVEL_WARM_TICK


def test_an_over_gate_clicks_queue_names_every_holder():
    """A 12 ms tick plus a 5 ms frame in a 23 ms queue: both named, and the 6 ms
    neither covers is untraced; a gc nested in the tick is not counted twice."""
    click = _click("roll", 0.0, end=1.0)
    click.t_entry, click.t_exit = 0.023, 0.024
    frames = [bz.Frame(0.001, 0.006, {"blit": 4.0}, 1, 4.5), bz.Frame(0.025, 0.040, {"blit": 14.0}, 1, 14.5)]
    tick = _walk_tick(12.0, 12.0)
    tick.start, tick.end = 0.010, 0.022
    gc2 = bz.Activity(0.012, 0.014, "gc", "gen1")
    acts = [tick, gc2]
    st = bz.click_stats(click, frames, acts)
    text = bz.queue_holders_text(click, st, frames, acts)
    assert text.startswith("roll #1 queue 23.0 ms: ")
    assert f"{bz.LEVEL_WARM_FIRST_NOTCH} [level-warm walk 12.0]" in text
    assert "blit at mip 1 (no task) [blit 5.0]" in text
    assert "gc gen1 [gc 2.0]" in text
    assert text.endswith("; untraced 6.0")
    g3 = bz.gates([click], frames, acts)[2]
    assert g3.met is False and g3.holders == [text]


def test_no_input_budget_parses_and_reaches_the_child():
    args = bz.build_parser().parse_args(["--no-input-budget"])
    assert args.no_input_budget and not args.no_reach
    assert "--no-input-budget" in bz.child_command(args, "Stepped", "on", ["roll"])
    assert "--no-input-budget" not in bz.child_command(bz.build_parser().parse_args([]), "Flat", "off", ["notch"])


def test_no_margin_fix_parses_and_reaches_the_child():
    """TASK-031.85's A arm (render_cache.WARM_NATIVE_INPUTS off) is forwarded like the other A arms."""
    args = bz.build_parser().parse_args(["--no-margin-fix"])
    assert args.no_margin_fix and not args.no_input_budget
    assert "--no-margin-fix" in bz.child_command(args, "Sloped", "off", ["notch"])
    assert "--no-margin-fix" not in bz.child_command(bz.build_parser().parse_args([]), "Stepped", "on", ["notch"])


@pytest.mark.parametrize("flag", [None, "--no-margin-fix", "--no-input-budget"])
def test_module_arms_set_each_switch_only_for_its_flag(monkeypatch, flag):
    """--no-margin-fix clears the native prime and its re-queue (WARM_NATIVE_INPUTS) and the
    cold-first defer (DEFER_COLD_FIRST, TASK-031.82 D3); --no-input-budget pins the budget."""
    from descape import level_warm, margin_warm, render_cache

    for module, name in ((render_cache, "WARM_NATIVE_INPUTS"), (margin_warm, "DEFER_COLD_FIRST"),
                         (level_warm, "INPUT_BUDGET_MS")):
        monkeypatch.setattr(module, name, getattr(module, name))
    bz.apply_module_arms(bz.build_parser().parse_args([flag] if flag else []))
    fix_on = flag != "--no-margin-fix"
    assert render_cache.WARM_NATIVE_INPUTS is fix_on and margin_warm.DEFER_COLD_FIRST is fix_on
    assert (level_warm.INPUT_BUDGET_MS == level_warm.BUDGET_MS) is (flag == "--no-input-budget")


def test_a_frame_dominated_by_gc_gen2_is_031_81():
    click = _click("roll", 0.0, end=1.0)
    frames = [bz.Frame(0.003, 0.103, {"other": 95.0}, 0, 97.0)]
    gc2 = bz.Activity(0.010, 0.090, "gc", "gen2")
    st = bz.click_stats(click, frames, [gc2])
    assert bz.latency_signature(click, st, frames, [gc2])[0] == "gc gen2 (TASK-031.81)"


def test_a_queue_held_by_gc_gen2_is_031_81():
    assert bz.activity_signature(bz.Activity(0.0, 0.1, "gc", "gen2"))[0] == "gc gen2 (TASK-031.81)"


def test_a_row_prints_the_worst_canvas_paint_s_reductions_beside_the_real_level_residual():
    click = _click("revisit", mip=(-2, -2))
    frames = [bz.Frame(0.003, 0.010, {"blit": 1.0}, -2, 1.2, 0), bz.Frame(0.020, 0.040, {"blit": 2.0, "reduce": 9.0}, -2, 11.5, 1)]
    frames[1].fmt, frames[1].preframe = "RGB888", "miss key"
    row = bz.format_row(click, bz.click_stats(click, frames, []))
    assert " resid 0.700 k 1 fmt RGB888 pf miss-key pct " in row
    assert "| canvas 11.0 mip -2 [blit 2.0 reduce 9.0]" in row
    bare = _click("revisit")
    assert " k - fmt - pf - pct " in bz.format_row(bare, bz.click_stats(bare, [], []))


def test_the_preframe_summary_counts_first_frame_hits_and_names_over_gate_misses():
    """TASK-031.71's M check: a notch hits when its FIRST frame did; an over-G1 notch is named with
    its worst canvas paint's outcome, and a notch outside G1's population is not counted."""
    clicks = [_click("revisit", float(i), idx=i + 1) for i in range(3)] + [_click("revisit", 3.0, resident=False, idx=4)]
    frames = [
        bz.Frame(0.003, 0.004, {"blit": 0.2}, 1, preframe="hit", fmt="RGB32"),
        bz.Frame(1.003, 1.010, {"blit": 6.0}, 1, preframe="miss stale", fmt="RGB32"),
        bz.Frame(2.003, 2.004, {"blit": 0.3}, 1, preframe="hit", fmt="RGB32"),
        bz.Frame(3.003, 3.010, {"blit": 9.0}, 1, preframe="miss none", fmt="RGB32"),
    ]
    lines, data = bz.preframe_summary(clicks, frames, [])
    assert data == {"hits": 2, "notches": 3, "worst_hit_ms": pytest.approx(0.3), "misses": {"miss stale": 1},
                    "over": 1, "over_hits": 0}
    assert lines[0].startswith("preframes revisit: 2/3 built-and-warm notches' first frame hit (67%)")
    assert lines[1] == "  over G1: #2 mip 0->0 resid 0.700 canvas 6.0 fmt RGB32 pf miss stale"


def test_a_gate_line_reads_dev_and_x4():
    assert bz.gate_line(3, "worst 21.7 ms", 21.7, True) == "gate G3: worst 21.7 ms dev (x4 86.8 ms) met"
    assert bz.gate_line(1, "no revisit pass", None, None) == "gate G1: no revisit pass dev (x4 -) n/a"


# --- placeholder paints (TASK-031.25) ---------------------------------------

# Stepped, sprites on, cold roll, placeholders on (the lane's smoke run, 2026-10-24).
VIEW_PLACEHOLDER = (
    "[22:49:56] perf view: repaint: 43 calls, 206ms total (max 24.2) [blit 124.5 x106; prepare 6.4 x7; join 9.2 x2;"
    " composite 5.1 x1; reduce 4.0 x2; other 57.0], warm: 75 ticks, 169 chunks, 184ms total (max tick 21.0, max chunk"
    " 21.0), worker 133 chunks, 646ms total (max 14.6), level 102 ticks, max tick 26.0 [walk 1 25.9; other 0.1; gc 0.1;"
    " max step 18.2 (walk 1 p0 #886 const 2423)], installs 4 (max 0.0), levels: 0 placeholder<-1 0.0ms x6 (repaint),"
    " 1 placeholder<-1 0.0ms x9 (repaint), 1 install 0.0ms x2 (level-warm), 1 pack 9.0ms (level-warm), 1 refine 0.0ms"
    " (other), 0 install 0.0ms x2 (level-warm), 0 pack 11.4ms (level-warm), 0 refine 0.0ms (other), mip -2->-1 -1->0"
    " 0->1 1->0 0->-1 -1->-2, gc gen2 176.2ms, composite native, platform xcb"
)
# The same line's shape at a negative level from a negative one (the old kind pattern dropped both).
VIEW_PLACEHOLDER_NEGATIVE = VIEW_PLACEHOLDER.replace("0 placeholder<-1 0.0ms x6", "-1 placeholder<-2 0.0ms x6")
FORCE_SETTLED = "[22:51:02] placeholder: force-settled level 1"


def test_placeholder_and_refine_entries_parse_and_never_count_as_builds():
    events = bz.parse_levels(bz.strip_ts(VIEW_PLACEHOLDER))
    assert [(e.mip, e.kind, e.count, e.where) for e in events[:2]] == [
        (0, "placeholder<-1", 6, "repaint"), (1, "placeholder<-1", 9, "repaint"),
    ]
    assert [(e.mip, e.kind) for e in events if e.kind == "refine"] == [(1, "refine"), (0, "refine")]
    assert len(events) == 8
    negative = bz.parse_levels(VIEW_PLACEHOLDER_NEGATIVE)[0]
    assert (negative.mip, negative.kind, negative.count) == (-1, "placeholder<-2", 6)
    assert bz.logged_counts([VIEW_PLACEHOLDER]) == (0, 0)
    assert bz.logged_placeholders([VIEW_PLACEHOLDER, VIEW_COLD_ROLL]) == 15
    assert bz.logged_placeholders([VIEW_PLACEHOLDER_NEGATIVE]) == 15


def test_force_settles_count_the_watchdog_lines():
    assert bz.force_settles([VIEW_PLACEHOLDER, FORCE_SETTLED, bz.strip_ts(FORCE_SETTLED)]) == 2
    assert bz.force_settles([VIEW_PLACEHOLDER, STALL]) == 0


def _ph_frame(t: float, mip: int, drawn: int, *, full: bool = True, canvas: float = 3.0) -> bz.Frame:
    return bz.Frame(t, t + 0.01, {"blit": canvas}, mip, canvas, 0, drawn, drawn != mip, full and drawn == mip)


def test_time_to_sharp_runs_to_the_first_real_full_view_frame():
    frames = [
        _ph_frame(0.0, 0, -1), _ph_frame(0.1, 0, -1),
        _ph_frame(0.2, 0, 0, full=False),  # a partial real paint: still waiting
        _ph_frame(0.4, 0, 0),
        _ph_frame(1.0, 1, 0),  # a wait the next level supersedes
        _ph_frame(1.1, 2, 1),
        bz.Frame(1.2, 1.21, {}, None),  # no canvas paint in it
    ]
    done, superseded, still_open = bz.sharp_waits(frames)
    assert [(lv, start.start, end.end) for lv, start, end in done] == [(0, 0.0, pytest.approx(0.41))]
    assert (superseded, still_open) == (1, 1)


def test_placeholder_lines_read_p1_p2_p4_p5():
    cold = _click("notch", 0.0, mip=(-1, 0), end=0.5, idx=1)
    warm = _click("notch", 0.5, mip=(0, 1), end=1.0, idx=2)
    warm.warm_before = True
    frames = [
        _ph_frame(0.003, 0, 0),  # cold crossing: sharp at once
        _ph_frame(0.503, 1, 0, canvas=30.0),  # beat its warm: a placeholder over the G3 budget
        _ph_frame(0.8, 1, 1),
    ]
    p = bz.placeholder_stats("notch", [cold, warm], frames, [], [VIEW_PLACEHOLDER, FORCE_SETTLED], 2)
    p.pending_at_end = "none"
    lines = bz.placeholder_lines(p)
    assert lines[0] == ("placeholders notch: 1 frames, worst canvas 30.0 ms dev (x4 120.0) at mip 1 from 0;"
                        " 1 clicks' first frame")
    assert "worst 307.0 ms dev (x4 1228.0) over 1 waits (0 superseded by another level, 0 still open);" in lines[1]
    assert "force-settles 1; stalls 0; pending at pass end none" in lines[1]
    assert lines[2] == "  cross-check placeholders notch: 2 paints seen, 15 logged  MISMATCH"
    assert lines[3] == ("  D3b: 1 crossing notches clicked with no warm active, 0 with a placeholder frame, 0 with"
                        " chunk work in a paint; 1 crossing notches beat their warm, 1 with a placeholder frame")
    assert lines[4] == "  over G3: placeholder frame canvas 30.0 ms dev at mip 1 from 0"
    roll = bz.placeholder_stats("roll", [], [], [], [], 0)
    assert roll.crossing_cold is None and len(bz.placeholder_lines(roll)) == 3


def test_placeholder_markers_leave_g1_s_population_alone_and_print_apart():
    """A 0 ms `placeholder<F` or `refine` event is not a build: a revisit notch with one stays in G1."""
    click = _click("revisit", 0.0)
    frames = [bz.Frame(0.003, 0.006, {"blit": 2.0}, 0, 2.2, 0, -1, True)]
    marker = bz.Activity(0.004, 0.004, "level", "0 placeholder<-1 (repaint)", {}, "repaint")
    refine = bz.Activity(0.2, 0.2, "level", "0 refine (other)", {}, "other")
    g1 = bz.gates([click], frames, [marker, refine])[0]
    assert g1.met is True and "over 1 built-and-warm notches" in g1.line
    st = bz.click_stats(click, frames, [marker, refine])
    assert st.levels == [] and len(st.markers) == 2
    row = bz.format_row(click, st)
    assert " PLACEHOLDER from -1 " in row
    assert row.endswith("markers 0 placeholder<-1 x1, 0 refine x1")


def test_the_reach_line_counts_same_level_zoom_outs_that_composite_apart_from_the_coarsest_and_pending():
    """TASK-031.11 Step 4: only accepted zoom-outs that stay on one level count;
    a first frame with chunk phases over 1 ms is a hit, a placeholder first
    frame is counted apart (R1 leaves pending levels out), the coarsest level
    has its own count, and an over-G3 click held by a margin-warm tick is R3's."""
    def out(t, mip, bracket, *, placeholder=False, accepted=True, direction=-1):
        c = _click("roll", t, accepted=accepted, mip=mip, end=t + 0.5, idx=int(t))
        c.direction = direction
        return c, bz.Frame(t + 0.003, t + 0.006, bracket, mip[1], 3.0, 0, -1 if placeholder else mip[1], placeholder)

    rows = [
        out(0.0, (0, 0), {"prepare": 0.6, "join": 0.6, "blit": 1.0}),  # 1.2 ms of chunk work: a hit
        out(1.0, (0, 0), {"blit": 3.0, "composite": 0.4}),
        out(2.0, (0, 0), {"blit": 2.0}, placeholder=True),
        out(3.0, (-2, -2), {"join": 1.5}),
        out(4.0, (-2, -2), {"blit": 1.0}),
        out(5.0, (0, -1), {"composite": 9.0}),  # a crossing
        out(6.0, (0, 0), {"composite": 9.0}, direction=1),  # a zoom-in
        out(7.0, (0, 0), {"composite": 9.0}, accepted=False),
    ]
    clicks, frames = [c for c, _f in rows], [f for _c, f in rows]
    held = _click("roll", 8.0, mip=(0, 0), end=8.5, idx=8)
    held.t_entry, held.t_exit = 8.060, 8.061
    clicks.append(held)
    frames.append(bz.Frame(8.0625, 8.0665, {"blit": 2.0}, 0, 2.5))
    tick = bz.Activity(8.002, 8.060, "margin-warm", "")
    r = bz.reach_stats("roll", clicks, frames, [tick], -2)
    assert (r.same_level, r.pending, r.coarsest, r.margin_tick_over) == ((2, 1), 1, (2, 1), 1)
    assert bz.reach_line(r) == (
        "reach roll: 1 of 2 same-level zoom-out clicks' first frame with chunk work over 1 ms dev (1 more from a"
        " pending level); coarsest level 1 of 2; 1 over-G3 clicks with a margin-warm tick signature"
    )


def test_no_reach_parses_and_reaches_the_child():
    """The A arm's flag (TASK-031.11 D9) is forwarded to each child process, like the other A arms."""
    args = bz.build_parser().parse_args(["--no-reach", "--no-preframes"])
    assert args.no_reach and args.no_preframes and not args.no_placeholders
    cmd = bz.child_command(args, "Stepped", "on", ["roll"])
    assert "--no-reach" in cmd and "--no-preframes" in cmd and "--no-placeholders" not in cmd
    assert cmd[cmd.index("--child") + 1] == "Stepped,on" and cmd[cmd.index("--pass") + 1] == "roll"
    assert "--no-reach" not in bz.child_command(bz.build_parser().parse_args([]), "Flat", "off", ["notch"])


@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_canvas_paint_that_returns_early_reads_no_format_or_preframe_from_the_paint_before(monkeypatch):
    """The recorder's canvas paint wrapper clears _last_blit_format and
    _last_preframe before each paint: one that returns before blitting
    (an empty exposed rect) reads `-`, not the last paint's values."""
    from types import SimpleNamespace

    from PyQt5.QtGui import QImage

    from descape import viewer_canvas

    def paint(item, painter, option, widget=None):
        if item.blits:
            item._last_blit_format, item._last_preframe = QImage.Format_RGB32, "hit"

    monkeypatch.setattr(viewer_canvas.MapCanvasItem, "paint", paint)
    rec = bz._Recorder()
    rec.install()
    try:
        item = SimpleNamespace(scene=lambda: None, blits=True)
        viewer_canvas.MapCanvasItem.paint(item, None, None, object())
        assert (rec._frame_fmt, rec._frame_preframe) == ("RGB32", "hit")
        item.blits = False
        viewer_canvas.MapCanvasItem.paint(item, None, None, object())
        assert (rec._frame_fmt, rec._frame_preframe) == (None, None)
    finally:
        rec.restore()


@pytest.fixture
def margin_recorder(monkeypatch):
    from collections import deque
    from types import SimpleNamespace

    from descape import perf_trace, render_cache

    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(bz.time, "perf_counter", lambda: clock.t)
    monkeypatch.setattr(perf_trace, "_enabled", True)
    monkeypatch.setattr(perf_trace, "_where_stack", [])
    monkeypatch.setattr(perf_trace, "_texture_raw", deque())
    monkeypatch.setattr(perf_trace, "warm_tick", lambda *a: None)
    monkeypatch.setattr(perf_trace, "warm_deferred", lambda *a: None)
    monkeypatch.setattr(perf_trace, "level_event", lambda *a: None)

    def get_chunk(cache, mip, cx, cy):
        clock.t += cache.get_ms / 1000
        if cache.nested == "get":
            cache.prepare_chunk_job(mip, cx, cy)
        if cache.events:
            perf_trace._texture_raw.extend([("load", 1.5), ("wait", 0.5)])
            perf_trace.level_event("pack", mip, 2.0, split={"walk": 1.0})
        return "pixels"

    def prepare_chunk_job(cache, mip, cx, cy, heights=None):
        clock.t += 0.003
        if cache.nested == "prepare":
            cache.get_chunk(mip, cx, cy)
        return cache.job

    monkeypatch.setattr(render_cache._ChunkCacheBase, "get_chunk", get_chunk)
    monkeypatch.setattr(render_cache._ChunkCacheBase, "prepare_chunk_job", prepare_chunk_job)
    cache = render_cache.IsoChunkCache.__new__(render_cache.IsoChunkCache)
    cache.get_ms, cache.nested, cache.events, cache.job = 5.0, "", False, None
    cache._composited_mips, cache._native_primed, cache._source_gen = set(), {}, 0
    rec = bz._Recorder(profile_paint=1, profile_warm=2)
    rec.install()
    try:
        yield rec, cache, clock
    finally:
        rec.restore()


@pytest.mark.parametrize(
    ("mode", "path", "ms"),
    [("pooled", "pooled", 3.0), ("sync", "sync", 8.0), ("flat", "flat", 5.0),
     ("flat-prepare", "flat", 8.0), ("prepare-nested", "sync", 8.0), ("get-nested", "sync", 8.0)],
)
def test_margin_chunks_record_each_path_and_combine_prepare_get_once(margin_recorder, mode, path, ms):
    from descape import perf_trace, render_cache

    rec, cache, _clock = margin_recorder
    if mode.startswith("flat"):
        flat = render_cache.FlatChunkCache.__new__(render_cache.FlatChunkCache)
        flat.__dict__.update(cache.__dict__)
        cache = flat
    cache.job = object() if mode == "pooled" else None
    cache.nested = {"prepare-nested": "prepare", "get-nested": "get"}.get(mode, "")
    with perf_trace.where("margin-warm"):
        if mode in ("pooled", "sync", "flat-prepare", "prepare-nested"):
            job = cache.prepare_chunk_job(-1, 4, 7)
            if mode in ("sync", "flat-prepare"):
                assert job is None and cache.get_chunk(-1, 4, 7) == "pixels"
        else:
            assert cache.get_chunk(-1, 4, 7) == "pixels"
    assert len(rec.margin_chunks) == 1
    chunk = rec.margin_chunks[0]
    assert (chunk.mip, chunk.cx, chunk.cy, chunk.path, chunk.first_at_level) == (-1, 4, 7, path, True)
    assert chunk.ms == pytest.approx(ms)
    assert (chunk.end - chunk.start) * 1000 == pytest.approx(ms)
    assert len(rec.warm_profiles) == 1 and rec.warm_profiles[0][0] is chunk


@pytest.mark.parametrize("tags", [[], ["repaint"], ["level-warm"], ["margin-warm", "repaint"]])
def test_only_the_innermost_margin_tag_records_chunks(margin_recorder, tags):
    from descape import perf_trace

    rec, cache, _clock = margin_recorder
    perf_trace._where_stack[:] = tags
    assert cache.prepare_chunk_job(0, 1, 2) is None
    assert cache.get_chunk(0, 1, 2) == "pixels"
    assert rec.margin_chunks == [] and rec.warm_profiles == [] and rec._margin_levels == set()


def test_margin_ticks_attach_warmer_chunks_textures_levels_and_raw_json(margin_recorder):
    import json
    from types import SimpleNamespace

    from descape import margin_warm, perf_trace

    rec, cache, clock = margin_recorder
    cache.events = True
    perf_trace._texture_raw.append(("prefetch", 99.0))  # not inside either tick

    def load():
        with perf_trace.where("margin-warm"):
            cache.prepare_chunk_job(0, 2, 3)
            cache.get_chunk(0, 2, 3)
        perf_trace.warm_tick(8.0, 1, 8.0)
        return True

    def navigation():
        with perf_trace.where("margin-warm"):
            cache.get_chunk(0, 3, 4)
            cache.get_chunk(1, 4, 5)
        perf_trace.warm_tick(10.0, 2, 5.0)
        return False

    window = SimpleNamespace(_load_warmer=SimpleNamespace(tick=load), _margin_warmer=SimpleNamespace(tick=navigation))
    rec.watch_margin_warmers(window)
    assert window._load_warmer.tick() is True
    clock.t += 0.01
    assert window._margin_warmer.tick() is False
    ticks = [a for a in rec.activities if a.kind == "margin-warm"]
    assert [a.warmer for a in ticks] == ["_load_warmer", "_margin_warmer"]
    assert [c.first_at_level for c in rec.margin_chunks] == [True, False, True]
    assert ticks[0].chunks == rec.margin_chunks[:1] and ticks[1].chunks == rec.margin_chunks[1:]
    assert [a.ms for a in ticks] == pytest.approx([8.0, 10.0])
    assert all(a.budget_ms == margin_warm.BUDGET_MS for a in ticks)
    assert ticks[0].textures == [("load", 1.5), ("wait", 0.5)]
    assert len(ticks[1].textures) == 4
    assert [(e.mip, e.kind, e.where, e.split) for e in ticks[0].levels] == [(0, "pack", "margin-warm", {"walk": 1.0})]
    assert ticks[0].detail == "1 chunks _load_warmer; slowest mip 0 (2,3) sync _load_warmer 8.0 ms; first at level True"
    assert rec._margin_tick is None and rec._chunk_pending is None
    click = _click("notch", 0.001)
    data = json.loads(json.dumps(rec.margin_data([click], {"notch": (0.001, 1.0, "none")})))
    assert len(data["chunks"]) == 3  # all chunks, not only each tick's slowest
    assert len(data["ticks"][1]["chunks"]) == 2
    assert data["chunks"][0]["warmer"] == "_load_warmer" and data["chunks"][0]["first_at_level"]
    assert data["ticks"][0]["textures"] == [["load", 1.5], ["wait", 0.5]]
    assert data["ticks"][0]["levels"][0]["mip"] == 0
    assert data["clicks"][0]["index"] == 1 and data["passes"]["notch"] == {"start": 0.001, "end": 1.0}
    row = bz.format_row(click, bz.click_stats(click, [], ticks[:1]))
    assert ticks[0].detail in row


def test_margin_chunks_record_warmed_and_primed_at_their_start_and_ticks_their_deferrals(margin_recorder, monkeypatch):
    """TASK-031.82: each chunk says whether its level was warmed and primed when it
    started, so D6 (primed) and D3 (moved) read apart; deferrals land in GATES."""
    import json
    from types import SimpleNamespace

    from descape import margin_warm, perf_trace

    rec, cache, _clock = margin_recorder
    cache._native_primed = {-1: 0}
    monkeypatch.setattr(margin_warm, "deferrals", 7)

    def tick():
        with perf_trace.where("margin-warm"):
            perf_trace.warm_deferred(2)
            for mip in (-1, 0, 1):
                cache.get_chunk(mip, 0, 0)
                cache._composited_mips.add(mip)  # what the real get_chunk's _insert() does
            cache.get_chunk(0, 1, 0)
        return False

    window = SimpleNamespace(_load_warmer=SimpleNamespace(tick=tick), _margin_warmer=SimpleNamespace(tick=tick))
    rec.watch_margin_warmers(window)
    window._margin_warmer.tick()
    fields = [(c.mip, c.cx, c.level_warmed, c.primed) for c in rec.margin_chunks]
    assert fields == [(-1, 0, True, True), (0, 0, False, False), (1, 0, False, False), (0, 1, True, False)]
    data = json.loads(json.dumps(rec.margin_data([], {})))
    assert [(d["mip"], d["warmer"]) for d in data["deferrals"]] == [(2, "_margin_warmer")]
    assert data["deferral_count"] == 7
    assert data["chunks"][0]["primed"] is True and data["chunks"][1]["level_warmed"] is False


def test_warm_profiles_keep_the_slowest_combined_chunks_separate_from_paints(margin_recorder):
    import cProfile

    from descape import perf_trace

    rec, cache, _clock = margin_recorder
    paint = cProfile.Profile()
    paint.runcall(sum, [1, 2])
    rec.keep_profile(100.0, paint, rec.profile_paint)
    with perf_trace.where("margin-warm"):
        for cx, cost in enumerate([6.0, 1.0, 4.0]):
            cache.get_ms = cost
            cache.prepare_chunk_job(1, cx, 7)
            cache.get_chunk(1, cx, 7)
    assert [(c.cx, c.ms) for c, _p in rec.warm_profiles] == [(0, pytest.approx(9.0)), (2, pytest.approx(7.0))]
    assert rec.profiles == [(100.0, paint)] and rec.all_profiles == [paint]
    texts = rec.profile_texts(warm=True)
    assert len(texts) == 2 and texts[0][0] == pytest.approx(9.0)
    assert texts[0][1].startswith("mip 1 (0,7) sync; first at level True\n")
    assert "prepare_chunk_job" in texts[0][1] and "get_chunk" in texts[0][1]
    assert "all 1 summed" in rec.profile_texts()[0][1]


@pytest.mark.parametrize("chunk_ms", [7.999, 8.0, 8.001])
def test_margin_slow_chunk_signature_is_strictly_over_the_chunk_budget(chunk_ms):
    from descape import margin_warm

    chunk = bz.WarmChunk(0.0, chunk_ms / 1000, -1, 2, 3, "sync", chunk_ms)
    act = bz.Activity(0.0, 0.025, "margin-warm", budget_ms=margin_warm.BUDGET_MS, chunks=[chunk])
    signature = bz.activity_signature(act)[0]
    assert bz.MARGIN_TICK in signature
    assert ("TASK-031.82" in signature) == (chunk_ms > margin_warm.BUDGET_MS)
    if chunk_ms > margin_warm.BUDGET_MS:
        assert signature == f"{bz.MARGIN_TICK} (TASK-031.82 slow chunk, mip -1 sync)"
    act.chunks = []
    assert bz.activity_signature(act)[0] == bz.MARGIN_TICK


def test_reach_counts_the_slow_chunk_signature_too():
    click = _click("roll", 0.0, end=0.5)
    click.t_entry, click.t_exit = 0.06, 0.061
    frame = bz.Frame(0.062, 0.066, {"blit": 2.0}, 0)
    chunk = bz.WarmChunk(0.0, 0.06, 0, 2, 3, "pooled", 60.0)
    act = bz.Activity(0.0, 0.06, "margin-warm", chunks=[chunk])
    assert bz.reach_stats("roll", [click], [frame], [act], -2).margin_tick_over == 1


def test_profile_warm_parses_and_reaches_children_alongside_paint_or_frame():
    for other in ("--profile-paint", "--profile-frame"):
        args = bz.build_parser().parse_args(["--profile-warm", "6", other, "2", "--json", "raw.json"])
        assert args.profile_warm == 6 and str(args.json) == "raw.json"
        cmd = bz.child_command(args, "Stepped", "off", ["notch", "roll"])
        assert cmd[cmd.index("--profile-warm") + 1] == "6"
        assert cmd[cmd.index(other) + 1] == "2" and "--json" not in cmd  # only the parent writes the combined file
    assert bz.build_parser().parse_args([]).profile_warm == 0


def test_json_export_retains_all_process_chunk_records(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    rec = bz._Recorder()
    rec.margin_chunks = [bz.WarmChunk(0.0, 0.001, 0, cx, 2, "flat", 1.0) for cx in range(3)]
    gate = {"gates": {str(n): {"met": None} for n in (1, 2, 3)}, "margin_warm": rec.margin_data([], {})}
    monkeypatch.setenv("DISPLAY", ":test")
    dest = tmp_path / "raw.json"
    monkeypatch.setattr(bz.sys, "argv", ["bench_gui_zoom.py", "--style", "Flat", "--sprites", "off", "--pass", "roll",
                                      "--profile-warm", "6", "--json", str(dest)])
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(stdout="GATES " + json.dumps(gate), returncode=0)

    monkeypatch.setattr(bz.subprocess, "run", run)
    bz.main()
    data = json.loads(dest.read_text())
    assert len(data) == 1 and [c["cx"] for c in data[0]["margin_warm"]["chunks"]] == [0, 1, 2]
    assert commands[0][commands[0].index("--profile-warm") + 1] == "6"


@pytest.mark.parametrize("flag", ["--json", "--out-dir"])
@pytest.mark.parametrize("kind", ["direct", "mixed-case", "resolved-link", "loop"])
def test_output_guard_refuses_before_spawn_or_truncation(monkeypatch, tmp_path, flag, kind):
    protected = tmp_path / ("MoDs" if kind == "mixed-case" else "mods") / (
        "SuBsCrIbEd" if kind == "mixed-case" else "subscribed"
    )
    if kind == "loop":
        dest = tmp_path / "loop"
        dest.symlink_to("loop", target_is_directory=True)
    elif kind == "resolved-link":
        protected.mkdir(parents=True)
        dest = tmp_path / "alias"
        dest.symlink_to(protected, target_is_directory=True)
    else:
        dest = protected
    if flag == "--json":
        dest = dest / "raw.json"
    sentinel = tmp_path / "existing.json"
    sentinel.write_text("do not truncate\n")
    argv = ["bench_gui_zoom.py", "--style", "Flat", "--sprites", "off", "--pass", "notch", flag, str(dest)]
    if flag == "--out-dir":
        argv += ["--json", str(sentinel)]
    monkeypatch.setattr(bz.sys, "argv", argv)
    monkeypatch.setenv("DISPLAY", ":test")

    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark spawned before output refusal")

    monkeypatch.setattr(bz.subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as exc:
        bz.main()
    assert exc.value.code == 2 and sentinel.read_text() == "do not truncate\n"
    if protected.exists():
        assert not list(protected.iterdir()), "the guard test must never write protected output"


def test_output_guard_checks_derived_report_paths_before_spawn(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(bz.sys, "argv", ["bench_gui_zoom.py", "--out-dir", str(tmp_path),
                                       "--style", "mods/subscribed/Flat", "--sprites", "off"])

    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark spawned for a protected derived report path")

    monkeypatch.setattr(bz.subprocess, "run", forbidden)
    with pytest.raises(SystemExit):
        bz.main()


def test_output_guard_rechecks_a_destination_changed_during_the_child(monkeypatch, tmp_path):
    from types import SimpleNamespace

    safe = tmp_path / "safe"
    safe.mkdir()
    protected = tmp_path / "mods" / "subscribed"
    protected.mkdir(parents=True)
    alias = tmp_path / "output"
    alias.symlink_to(safe, target_is_directory=True)
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(bz.sys, "argv", ["bench_gui_zoom.py", "--style", "Flat", "--sprites", "off",
                                       "--pass", "notch", "--out-dir", str(alias)])

    def run(*args, **kwargs):
        alias.unlink()
        alias.symlink_to(protected, target_is_directory=True)
        return SimpleNamespace(stdout="", returncode=0)

    def no_write(*args, **kwargs):
        raise AssertionError("write attempted before revalidating the changed destination")

    monkeypatch.setattr(bz.subprocess, "run", run)
    monkeypatch.setattr(bz.Path, "write_text", no_write)
    with pytest.raises(SystemExit):
        bz.main()
    assert not list(safe.iterdir()) and not list(protected.iterdir())


def test_output_guard_uses_the_selected_root_before_any_child_import(tmp_path):
    import os
    import subprocess
    import sys

    root = tmp_path / "selected"
    package = root / "descape"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "scenario_io.py").write_text(
        "def is_workshop_subscribed(path):\n    raise RuntimeError('SELECTED_ROOT_GUARD')\n"
    )
    testkit = root / "testkit"
    testkit.mkdir()
    (testkit / "__init__.py").write_text("raise RuntimeError('CHILD_IMPORT_REACHED')\n")
    dest = tmp_path / "raw.json"
    proc = subprocess.run([sys.executable, bz.__file__, "--root", str(root), "--json", str(dest),
                           "--child", "Flat,off"], capture_output=True, text=True, timeout=15,
                          env={**os.environ, "DISPLAY": ":test"}, check=False)
    assert proc.returncode == 2 and "SELECTED_ROOT_GUARD" in proc.stderr
    assert "CHILD_IMPORT_REACHED" not in proc.stderr and not dest.exists()


def test_output_guard_refuses_a_preimported_wrong_checkout(monkeypatch, tmp_path):
    args = bz.build_parser().parse_args(["--root", str(tmp_path), "--json", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        bz._validate_outputs(args, [], bz.build_parser())


def test_quiet_tail_raw_is_once_after_boundary_without_rescoring_frozen_fields(monkeypatch):
    import copy
    import json
    from types import SimpleNamespace

    rec = bz._Recorder()
    frozen = {
        "style": "Flat", "sprites": False, "passes": ["notch"], "placeholders": True,
        "gates": {str(n): {"met": True, "dev_ms": n, "line": f"G{n} frozen", "signatures": [], "holders": []}
                  for n in (1, 2, 3)},
        "placeholder_passes": {"notch": {"stalls": 0}}, "preframes": True, "preframe_g1": {"hits": 1},
        "reach": True, "reach_passes": {}, "input_budget": True, "level_warm_passes": {},
        "runtime": {"sched_schedstats": "0"},
    }
    expected = copy.deepcopy(frozen)
    monkeypatch.setattr(bz.time, "perf_counter", lambda: 1.0)

    def pump(seconds):
        assert seconds > 1
        rec.activities.append(bz.Activity(2.0, 2.2, "stall", raw="tail stall"))
        pf = bz.Activity(2.2, 2.3, "preframe", "fetch", raw="tail slow preframe")
        rec.activities.append(pf)
        rec.slow_preframe_ticks.append(pf)
        rec.activities.append(bz.Activity(2.4, 2.401, "margin-warm", warmer="tail"))
        rec.margin_chunks.append(bz.WarmChunk(2.4, 2.401, 0, 1, 2, "flat", 1.0))
        frozen["gates"]["2"].update(met=False, dev_ms=200, line="tail must not alter G2")
        frozen["placeholder_passes"]["notch"]["stalls"] = 1

    def no_rescore(*args):
        raise AssertionError("quiet-tail evidence must not rescore the gates")

    monkeypatch.setattr(bz, "_pump", pump)
    monkeypatch.setattr(bz, "gates", no_rescore)
    bench = SimpleNamespace(clicks=[], flush=lambda: ["tail logs"])
    output = []
    bz._finish_report(output, frozen, rec, bench, {"notch": (.1, .9, "none")})
    records = [json.loads(line[6:]) for line in output if line.startswith("GATES ")]
    assert len(records) == 1
    gate = records[0]
    assert {name: gate[name] for name in expected} == expected
    assert gate["gate_boundary"] == 1.0
    assert len(gate["diagnostics"]["stalls"]) == len(gate["diagnostics"]["slow_preframe_ticks"]) == 1
    assert gate["diagnostics"]["counts"] == {"stalls": {"raw": 1, "eligible": 0},
                                            "slow_preframe_ticks": {"raw": 1, "eligible": 0}}
    assert len(gate["margin_warm"]["ticks"]) == len(gate["margin_warm"]["chunks"]) == 1
    for row in [*gate["diagnostics"]["stalls"], *gate["diagnostics"]["slow_preframe_ticks"],
                *gate["margin_warm"]["ticks"], *gate["margin_warm"]["chunks"]]:
        assert row["start"] > gate["gate_boundary"]
    assert output[-1].startswith("GATES ") and "  tail logs" in output


def test_thread_json_and_stall_parser_keep_uncertainty_sampling_and_fault_bounds():
    import json

    from descape.thread_stats import Delta

    stats = Delta(200, 2, 0, 1, sampling_ms=2, maj_bounds=(0, 1), wall_start=1, wall_end=1.2)
    raw = json.loads(json.dumps(bz.thread_record(stats)))
    assert raw["kind"] == "uncertain" and raw["nominal_kind"] == "fault" and raw["eligible"] is False
    assert raw["sampling_ms"] == 2 and raw["maj_bounds"] == [0, 1]
    assert raw["possible_kinds"] == ["blocked", "fault"]
    assert raw["wall_start"] == 1 and raw["wall_end"] == 1.2
    parsed = bz.parse_thread(f"covered 200; thread uncertain: {stats.text()}")
    assert parsed["sampling_ms"] == 2 and parsed["maj_bounds"] == [0, 1]
    assert parsed["possible_kinds"] == ["blocked", "fault"]
    rec = bz._Recorder()
    rec.slow_preframe_ticks = [bz.Activity(1, 1.2, "preframe", "fetch", thread=bz.thread_record(stats)),
                              bz.Activity(2, 2.2, "preframe", "fetch", thread=bz.thread_record(Delta(200, 190)))]
    assert rec.diagnostics()["counts"]["slow_preframe_ticks"] == {"raw": 2, "eligible": 1}
