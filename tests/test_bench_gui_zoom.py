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


def test_a_queue_held_by_a_level_warm_walk_slice_over_budget_is_031_40():
    """The felt sub-150 ms pause: a click waits on one long level-warm tick."""
    click = _click("roll", 0.0, end=1.0)
    click.t_entry, click.t_exit = 0.060, 0.061
    frames = [bz.Frame(0.0625, 0.0665, {"blit": 2.0}, 1, 2.5)]
    tick = bz.Activity(0.002, 0.060, "level-warm", "", {"walk 1": 57.8, "other": 0.2})
    st = bz.click_stats(click, frames, [tick])
    cand, phase, ms = bz.latency_signature(click, st, frames, [tick])
    assert cand.startswith("TASK-031.40 recurrence")
    assert phase == "queue: level-warm walk"
    assert ms == pytest.approx(58.0)


def test_a_frame_dominated_by_gc_gen2_is_031_18():
    click = _click("roll", 0.0, end=1.0)
    frames = [bz.Frame(0.003, 0.103, {"other": 95.0}, 0, 97.0)]
    gc2 = bz.Activity(0.010, 0.090, "gc", "gen2")
    st = bz.click_stats(click, frames, [gc2])
    assert bz.latency_signature(click, st, frames, [gc2])[0] == "TASK-031.18 gc gen2"


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
