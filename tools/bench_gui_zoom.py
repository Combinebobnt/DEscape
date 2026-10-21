#!/usr/bin/env python3
"""Zoom-stutter capture (TASK-031.26, GH #181) through REAL X wheel clicks.
Informational: it prints gates G1-G3 as met or missed, it never exits non-zero
on a miss.

Why real events: a QWheelEvent sent with sendEvent() skips the X queue, so a
click that waits behind a long paint or a warm tick would read as instant.
xdotool on Xvfb puts the clicks in the queue where the user's wheel puts them,
and each event's X server timestamp says when it was clicked.

One process per (style, sprites) combination, so every first visit is a real
first visit with cold process sprite caches. The parent spawns them in order;
each loads --scenario, settles --settle seconds, then runs --pass in order:
  - notch: fit to the zoom-in ceiling one click at a time, then to the floor,
    then back to fit, waiting --notch-gap plus until the paint queue is quiet
    after each click. G2 (first visit) reads this pass.
  - revisit: the same sweep again, every level built by now. G1 (resident
    notch) reads this pass.
  - roll: `xdotool click --repeat 10 --delay <roll-delay>` bursts, fit to the
    ceiling, then to the floor, each leg until a whole burst leaves the zoom
    readout unchanged. G3 (fast roll) reads this pass. With --cold-roll on
    (default) a second process per combination runs the roll first, cold.
Display settings stay the isolated config's defaults (trees and eye candy as a
fresh install has them); only Show sprites is forced, per --sprites. --style
Flat is the default Flat + Isometric View (an IsoChunkCache on zero
elevations); Flat-topdown unticks Isometric View for the top-down FlatChunkCache.

Per click it prints one row: accepted or refused (the zoom readout changed or
not), mip and zoom % before and after, whether the level was built and warm
before the click, the residual scale, click-to-first-frame and click-to-quiet
latency split into queue / handler / wait / frame, the worst canvas paint in
the click's window with its repaint bracket, then that frame's other items and
flush as context, level events, stalls, gc gen2 and warm ticks in the window,
then the debug-log lines drained after it (notch passes; a roll drains per
burst). A frame is one top-level UpdateRequest: every item's paint plus the
backing-store flush. Clicks are timed from their X timestamp, mapped to the
process clock by the smallest event-to-handler offset seen (mouse moves at rest
before the passes calibrate it).

Gates (dev ms; x4 must fit the budget):
  G1 resident notch: revisit notches landing on a level built and warm before
     the click, with no level build in their window: each notch's worst canvas
     repaint (MapCanvasItem.paint, Perf Trace's `repaint`) <= 4.2 ms. The whole
     frame and its flush are printed beside it, not gated.
  G2 first visit: notch pass, no `build (repaint)` level event and no stall.
  G3 fast roll: accepted roll clicks, worst click-to-frame (to the END of the
     first frame after the handler) <= 25 ms, and no `build (repaint)`.
     Refused clicks are counted, not failed.
Each missed gate prints its offending clicks' signatures. A `cross-check` line
compares the builds inside paints the wrappers saw with the `(repaint)` level
builds Perf Trace logged, per notch and per roll, and flags a disagreement.

--profile-paint N wraps MapCanvasItem.paint in cProfile and prints the N
slowest paints' top 25 by cumtime (cProfile inflates every figure; read it for
shares, never for gates). --profile-frame N does the same for whole frames and
adds their top 15 by tottime, where Qt's own C++ (item traversal, the backing
store flush) shows as QMainWindow.event's self time. The two are exclusive.

One shell command, Xvfb on TCP (works where Unix-socket X access isn't
available), killed by its own PID. Never `pkill -f "Xvfb :N"`: that also
matches the invoking shell's own command line and kills it.

    Xvfb :71 -listen tcp -nolisten unix -screen 0 1920x1200x24 & XPID=$!; sleep 1; \\
    DISPLAY=127.0.0.1:71 .venv/bin/python3 tools/bench_gui_zoom.py --style Stepped --sprites on; \\
    kill $XPID; wait $XPID

The `wait` matters: without it the call can end before Xvfb removes
/tmp/.X71-lock, and that stale lock holds a small PID which exists again in
the next sandboxed call, so Xvfb refuses the display as "already active".
If it does, pick a display with no /tmp/.X<N>-lock.

xdotool wheel clicks are 120-unit detents: a high-resolution wheel or trackpad
sends more, smaller events for the same notches, which this does not simulate.

--root DIR imports descape and testkit from another checkout, so one copy of
this script drives a baseline worktree and a lane branch alike.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

try:  # imported as tools.bench_gui_zoom (the tests)
    from tools.bench_gui_drag import _csv, _loadavg, _size, _xdotool
except ModuleNotFoundError:  # run as a script: tools/ is sys.path[0]
    from bench_gui_drag import _csv, _loadavg, _size, _xdotool

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO = ROOT / "examples" / "old-allies-final-v2.aoe2scenario"

G1_DEV_MS = 4.2
G3_DEV_MS = 25.0
X4 = 4
# A frame shorter than this is a hover trickle, not the zoom's repaint (click-to-quiet).
QUIET_FRAME_MS = 2.0
# level_warm.BUDGET_MS, restated so this module stays importable without descape.
LEVEL_WARM_BUDGET_MS = 12.0
DEBUG_LOG_LINES = 1000
CHUNK_PHASES = ("composite", "prepare", "join")


# --- Qt-free bookkeeping (tests/test_bench_gui_zoom.py) ---------------------

_TIMESTAMP = re.compile(r"^\[\d\d:\d\d:\d\d\] ")


def strip_ts(line: str) -> str:
    return _TIMESTAMP.sub("", line)


@dataclass
class LevelEvent:
    mip: int
    kind: str
    ms: float
    count: int = 1
    split: dict[str, float] = field(default_factory=dict)
    where: str = "other"


_LEVEL_ENTRY = re.compile(
    r"(-?\d+) ([a-z][\w-]*) ([\d.]+)ms(?: x(\d+))?(?: \d+ tiles)?(?: \[([^\]]*)\])? \(([^)]+)\)"
)
_LEVELS_END = re.compile(r", (?:mip |gc gen|composite |sprites |textures )")


def parse_levels(line: str) -> list[LevelEvent]:
    """The `levels:` entries of a perf view or op line."""
    start = line.find("levels: ")
    if start < 0:
        return []
    tail = line[start + len("levels: "):]
    end = _LEVELS_END.search(tail)
    if end is not None:
        tail = tail[: end.start()]
    out = []
    for m in _LEVEL_ENTRY.finditer(tail):
        split = {}
        for part in (m.group(5) or "").split():
            if split.get("_name") is None and not re.fullmatch(r"[\d.]+", part):
                split["_name"] = part
            elif "_name" in split:
                split[split.pop("_name")] = float(part)
        split.pop("_name", None)
        out.append(LevelEvent(int(m.group(1)), m.group(2), float(m.group(3)), int(m.group(4) or 1), split, m.group(6)))
    return out


_OP_ZOOM = re.compile(r"perf op zoom(?: x(\d+))?: (\d+)ms")


def op_zoom_count(lines: list[str]) -> int:
    """Accepted notches by Perf Trace's count: the `perf op zoom [xN]` lines, summed."""
    n = 0
    for line in lines:
        m = _OP_ZOOM.search(line)
        if m:
            n += int(m.group(1) or 1)
    return n


_STALL = re.compile(r"perf stall (\d+)ms \((.*)\)$")


def parse_stall(line: str) -> tuple[int, str] | None:
    m = _STALL.search(line)
    return (int(m.group(1)), m.group(2)) if m else None


def logged_counts(lines: list[str]) -> tuple[int, int]:
    """(level builds tagged `(repaint)`, stalls) in drained debug-log lines; the load line excluded."""
    builds = stalls = 0
    for raw in lines:
        line = strip_ts(raw)
        if line.startswith("perf load:"):
            continue
        builds += sum(e.count for e in parse_levels(line) if e.where == "repaint" and e.kind in BUILD_KINDS)
        stalls += parse_stall(line) is not None
    return builds, stalls


def cross_check(label: str, seen: tuple[int, int], lines: list[str]) -> str | None:
    """The `cross-check` line for one notch or roll: the wrappers' builds in paint and stalls
    against Perf Trace's own log. None when both sides saw nothing."""
    logged = logged_counts(lines)
    if seen == logged == (0, 0):
        return None
    flag = "" if seen == logged else "  MISMATCH"
    return f"  cross-check {label}: builds in paint {seen[0]} seen, {logged[0]} logged; stalls {seen[1]} seen, {logged[1]} logged{flag}"


@dataclass
class Frame:
    """One top-level UpdateRequest, with the canvas paint bracket summed inside it."""

    start: float
    end: float
    bracket: dict[str, float] = field(default_factory=dict)
    mip: int | None = None
    view_ms: float = 0.0  # the viewport's own Paint event: every scene item, not the flush

    @property
    def ms(self) -> float:
        return (self.end - self.start) * 1000

    @property
    def canvas_ms(self) -> float:
        """The canvas item's paint time in this frame: G1's quantity."""
        return sum(self.bracket.values())

    def outside(self) -> tuple[float, float]:
        """(other scene items, flush and the rest): the frame's time outside the canvas paint, split."""
        canvas = self.canvas_ms
        items = max(0.0, self.view_ms - canvas)
        return items, max(0.0, self.ms - canvas - items)


@dataclass
class Activity:
    """Something that held the GUI thread: kind is level, level-warm,
    margin-warm, gc or stall; detail is its text."""

    start: float
    end: float
    kind: str
    detail: str = ""
    split: dict[str, float] = field(default_factory=dict)
    where: str = ""

    @property
    def ms(self) -> float:
        return (self.end - self.start) * 1000


@dataclass
class Click:
    sweep: str
    index: int
    direction: int
    accepted: bool
    pct_before: float | None
    pct_after: float | None
    mip_before: int | None
    mip_after: int | None
    resident_before: bool
    residual: float | None
    t_click: float  # X timestamp on the process clock, seconds
    t_entry: float
    t_exit: float
    window_end: float = 0.0
    lines: list[str] = field(default_factory=list)


@dataclass
class ClickStats:
    queue_ms: float
    handler_ms: float
    wait_ms: float | None
    frame_ms: float | None
    to_frame_ms: float | None
    to_quiet_ms: float | None
    first: Frame | None
    slowest: Frame | None
    worst_canvas: Frame | None  # the frame with the longest canvas paint in the window: G1's
    levels: list[Activity]
    stalls: list[Activity]
    gc2_ms: float
    warm: list[Activity]

    @property
    def builds_in_paint(self) -> list[Activity]:
        return [a for a in self.levels if is_build_in_paint(a)]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# A level built lazily inside a paint: Stepped/Sloped's `build`, top-down Flat's two layers.
BUILD_KINDS = ("build", "flat-draws", "flat-icons")


def is_build_in_paint(act: Activity) -> bool:
    words = act.detail.split()
    return act.kind == "level" and act.where == "repaint" and len(words) > 1 and words[1] in BUILD_KINDS


def frame_builds(frame: Frame, activities: list[Activity]) -> list[Activity]:
    """The level builds that ran inside `frame` (1 ms slack for the two clocks' read order)."""
    return [a for a in activities if is_build_in_paint(a) and a.start >= frame.start - 1e-3 and a.end <= frame.end + 1e-3]


def click_stats(click: Click, frames: list[Frame], activities: list[Activity]) -> ClickStats:
    """A click's latency split and what landed in its window, handler entry to the
    next click's handler entry. Its first frame is the first to start after its
    handler returned, however late: a click queued behind a long paint is felt
    until that frame ends."""
    lo, hi = click.t_entry, click.window_end
    inside = [f for f in frames if lo <= f.start < hi]
    first = next((f for f in frames if f.start >= click.t_exit), None) if click.accepted else None
    if first is not None and first not in inside:
        inside.append(first)
    quiet = [f for f in inside if f.ms >= QUIET_FRAME_MS]
    acts = [a for a in activities if a.end > lo and a.start < hi]
    return ClickStats(
        queue_ms=max(0.0, (click.t_entry - click.t_click) * 1000),
        handler_ms=(click.t_exit - click.t_entry) * 1000,
        wait_ms=None if first is None else (first.start - click.t_exit) * 1000,
        frame_ms=None if first is None else first.ms,
        to_frame_ms=None if first is None else (first.end - click.t_click) * 1000,
        to_quiet_ms=None if not quiet else (quiet[-1].end - click.t_click) * 1000,
        first=first,
        slowest=max(inside, key=lambda f: f.ms, default=None),
        worst_canvas=max(inside, key=lambda f: f.canvas_ms, default=None),
        levels=[a for a in acts if a.kind == "level"],
        stalls=[a for a in acts if a.kind == "stall"],
        gc2_ms=sum(a.ms for a in acts if a.kind == "gc" and a.detail == "gen2"),
        warm=[a for a in acts if a.kind in ("level-warm", "margin-warm")],
    )


def bracket_signature(frame: Frame, mip: int | None, canvas_only: bool = False) -> tuple[str, str, float]:
    """(candidate, phase, ms) for a slow frame, from its canvas bracket; canvas_only leaves
    the other items and flush out (G1 gates the canvas paint alone)."""
    b = frame.bracket
    chunk = sum(b.get(k, 0.0) for k in CHUNK_PHASES)
    parts = {"chunk": chunk, "blit": b.get("blit", 0.0), "levels": b.get("levels", 0.0)}
    parts["other"] = b.get("other", 0.0)
    if not canvas_only:
        parts["other items"], parts["flush"] = frame.outside()
    name, ms = max(parts.items(), key=lambda kv: kv[1])
    if name == "chunk":
        top = max(CHUNK_PHASES, key=lambda k: b.get(k, 0.0))
        return ("TASK-031.10/031.11 margin warm not covering the new viewport", top, ms)
    if name == "blit":
        if mip == 0:
            return ("TASK-031.68 level 0 RGB888 point overload", "blit", ms)
        return (f"blit at mip {mip} (no task)", "blit", ms)
    if name == "levels":
        return ("level event inside the paint", "levels", ms)
    if name == "other":
        return ("untraced code in the canvas paint: instrument (Step 2)", "other", ms)
    if name == "other items":
        return ("untraced: other scene items in the viewport paint", "other items", ms)
    return ("Qt backing-store flush and window repaint outside the viewport paint (C++)", "flush", ms)


def activity_signature(act: Activity) -> tuple[str, str, float]:
    if act.kind == "gc":
        return ("TASK-031.18 gc gen2" if act.detail == "gen2" else f"gc {act.detail}", "gc", act.ms)
    if act.kind == "level-warm":
        walk = sum(ms for k, ms in act.split.items() if k.startswith("walk"))
        if act.ms > LEVEL_WARM_BUDGET_MS and walk >= act.ms / 2:
            return ("TASK-031.40 recurrence (level-warm walk slice over budget): new TASK-031 child", "level-warm walk", act.ms)
        return ("level-warm tick", "level-warm", act.ms)
    if act.kind == "margin-warm":
        return ("TASK-031.10/031.11 margin-warm tick", "margin-warm", act.ms)
    if act.kind == "level":
        return (f"level event {act.detail}", act.detail, act.ms)
    return (act.kind, act.detail, act.ms)


def build_signature(act: Activity, sweep: str) -> tuple[str, str, float]:
    """A level build inside a paint: the walk-dominant ones are the two candidate tasks."""
    walk = sum(ms for k, ms in act.split.items() if k.startswith("walk"))
    if walk >= act.ms / 2:
        if sweep == "roll":
            return ("TASK-031.25 placeholder paint / async install (fast roll)", "build walk", act.ms)
        return ("TASK-031.24 per-unit sprite walk floor", "build walk", act.ms)
    kind = act.detail.split()[1] if len(act.detail.split()) > 1 else "untimed"
    top = max(act.split.items(), key=lambda kv: kv[1], default=(kind, act.ms))[0]
    return (f"level build in paint, {top} dominant (no task)", f"build {top}", act.ms)


def frame_signature(frame: Frame, activities: list[Activity], sweep: str) -> tuple[str, str, float]:
    """A slow frame's signature: a level build inside it wins, else its bracket."""
    builds = frame_builds(frame, activities)
    if builds:
        return build_signature(max(builds, key=lambda a: a.ms), sweep)
    return bracket_signature(frame, frame.mip)


def _blocker(t0: float, t1: float, frames: list[Frame], activities: list[Activity], sweep: str = ""):
    """What held the GUI thread most of [t0, t1): (signature, overlap ms) or None."""
    best = None
    for f in frames:
        ov = _overlap(t0, t1, f.start, f.end) * 1000
        if ov > 0 and (best is None or ov > best[1]):
            best = (frame_signature(f, activities, sweep), ov)
    for a in activities:
        if a.kind == "stall" or (a.kind == "level" and a.where == "repaint"):
            continue  # a stall is a report, not a holder; a level in a paint is inside a frame
        ov = _overlap(t0, t1, a.start, a.end) * 1000
        if ov > 0 and (best is None or ov > best[1]):
            best = (activity_signature(a), ov)
    return best


def latency_signature(click: Click, st: ClickStats, frames, activities) -> tuple[str, str, float]:
    """The dominant part of a slow click-to-frame: queue, handler, wait or the frame itself."""
    parts = {"queue": st.queue_ms, "handler": st.handler_ms, "wait": st.wait_ms or 0.0, "frame": st.frame_ms or 0.0}
    name = max(parts, key=parts.get)
    if name == "frame" and st.first is not None:
        gc2 = sum(a.ms for a in activities if a.kind == "gc" and a.detail == "gen2"
                  and _overlap(st.first.start, st.first.end, a.start, a.end) > 0)
        if gc2 >= st.first.ms / 2:
            return ("TASK-031.18 gc gen2", "frame: gc", gc2)
        cand, phase, ms = frame_signature(st.first, activities, click.sweep)
        return (cand, f"frame: {phase}", ms)
    if name == "handler":
        return ("untraced: wheel handler (untimed on perf op zoom)", "handler", parts[name])
    t0, t1 = (click.t_click, click.t_entry) if name == "queue" else (click.t_exit, st.first.start if st.first else click.t_exit)
    found = _blocker(t0, t1, frames, activities, click.sweep)
    if found is None:
        return (f"untimed {name}: nothing traced held the thread", name, parts[name])
    (cand, phase, _ms), ov = found
    return (cand, f"{name}: {phase}", ov)


def gate_line(n: int, value: str, dev_ms: float | None, met: bool | None) -> str:
    x4 = "-" if dev_ms is None else f"{dev_ms * X4:.1f} ms"
    verdict = "n/a" if met is None else "met" if met else "missed"
    return f"gate G{n}: {value} dev (x4 {x4}) {verdict}"


@dataclass
class GateResult:
    n: int
    line: str
    met: bool | None
    dev_ms: float | None
    signatures: list[str]


def _sig_text(click: Click, sig: tuple[str, str, float]) -> str:
    cand, phase, ms = sig
    resid = "-" if click.residual is None else f"{click.residual:.2f}"
    return (f"{click.sweep} #{click.index} mip {click.mip_before}->{click.mip_after} resid {resid}: {cand}"
            f" [{phase} {ms:.1f} ms dev, x4 {ms * X4:.1f}]")


def _unique(pairs) -> list[tuple[Click, Activity]]:
    """(click, activity) pairs with each activity once, kept on its first click."""
    seen: set[int] = set()
    out = []
    for c, a in pairs:
        if id(a) not in seen:
            seen.add(id(a))
            out.append((c, a))
    return out


def gates(clicks: list[Click], frames: list[Frame], activities: list[Activity]) -> list[GateResult]:
    stats = {id(c): click_stats(c, frames, activities) for c in clicks}
    out = []

    g1 = [c for c in clicks if c.sweep == "revisit" and c.accepted and c.resident_before
          and not stats[id(c)].levels]
    if g1:
        def canvas(c: Click) -> float:
            f = stats[id(c)].worst_canvas
            return f.canvas_ms if f is not None else 0.0

        ms = max(canvas(c) for c in g1)
        over = sorted((c for c in g1 if canvas(c) > G1_DEV_MS), key=lambda c: -canvas(c))
        all_sigs = [bracket_signature(stats[id(c)].worst_canvas, stats[id(c)].worst_canvas.mip, canvas_only=True)
                    for c in over]
        sigs = [_sig_text(c, s) for c, s in zip(over[:5], all_sigs[:5], strict=True)]
        counts: dict[str, int] = {}
        for s in all_sigs:
            counts[s[0]] = counts.get(s[0], 0) + 1
        sigs += [f"{n} of {len(over)} over-gate notches: {cand}" for cand, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        skipped = sum(1 for c in clicks if c.sweep == "revisit" and c.accepted) - len(g1)
        frames_ms = [stats[id(c)].slowest.ms for c in g1 if stats[id(c)].slowest is not None]
        flushes = [stats[id(c)].slowest.outside()[1] for c in g1 if stats[id(c)].slowest is not None]
        value = (f"worst canvas repaint {ms:.1f} ms over {len(g1)} built-and-warm notches ({len(over)} over {G1_DEV_MS},"
                 f" {skipped} not built and warm, or building); context: whole frame max {max(frames_ms, default=0):.1f},"
                 f" flush {min(flushes, default=0):.1f}-{max(flushes, default=0):.1f}")
        out.append(GateResult(1, gate_line(1, value, ms, ms <= G1_DEV_MS), ms <= G1_DEV_MS, ms, sigs))
    else:
        out.append(GateResult(1, gate_line(1, "no revisit pass", None, None), None, None, []))

    notch = [c for c in clicks if c.sweep == "notch"]
    if notch:
        builds = _unique((c, a) for c in notch for a in stats[id(c)].builds_in_paint)
        stalls = _unique((c, a) for c in notch for a in stats[id(c)].stalls)
        worst = max([a.ms for _, a in builds] + [a.ms for _, a in stalls], default=0.0)
        met = not builds and not stalls
        sigs = [_sig_text(c, build_signature(a, c.sweep)) for c, a in sorted(builds, key=lambda ca: -ca[1].ms)[:5]]
        for c, a in sorted(stalls, key=lambda ca: -ca[1].ms)[:5]:
            sigs.append(_sig_text(c, ("stall", a.detail, a.ms)))
        value = f"{len(builds)} builds in paint, {len(stalls)} stalls over {len(notch)} notches, worst {worst:.0f} ms"
        out.append(GateResult(2, gate_line(2, value, worst if not met else 0.0, met), met, worst, sigs))
    else:
        out.append(GateResult(2, gate_line(2, "no notch pass", None, None), None, None, []))

    roll = [c for c in clicks if c.sweep == "roll"]
    if roll:
        acc = [c for c in roll if c.accepted and stats[id(c)].to_frame_ms is not None]
        builds = _unique((c, a) for c in roll for a in stats[id(c)].builds_in_paint)
        worst_c = max(acc, key=lambda c: stats[id(c)].to_frame_ms, default=None)
        ms = stats[id(worst_c)].to_frame_ms if worst_c else 0.0
        met = ms <= G3_DEV_MS and not builds
        over = sorted((c for c in acc if stats[id(c)].to_frame_ms > G3_DEV_MS), key=lambda c: -stats[id(c)].to_frame_ms)
        all_sigs = [latency_signature(c, stats[id(c)], frames, activities) for c in over]
        sigs = [_sig_text(c, s) for c, s in zip(over[:5], all_sigs[:5], strict=True)]
        shown = {id(c) for c in over[:5]}
        sigs += [_sig_text(c, build_signature(a, "roll")) for c, a in builds if id(c) not in shown][:5]
        counts: dict[str, int] = {}
        for s in all_sigs:
            counts[s[0]] = counts.get(s[0], 0) + 1
        sigs += [f"{n} of {len(over)} over-gate clicks: {cand}" for cand, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        refused = sum(1 for c in roll if not c.accepted)
        unpainted = sum(1 for c in roll if c.accepted) - len(acc)
        value = (f"worst click-to-frame {ms:.1f} ms over {len(acc)} accepted ({refused} refused, {unpainted} accepted"
                 f" with no frame after, {len(over)} over {G3_DEV_MS:g}), {len(builds)} builds in paint")
        out.append(GateResult(3, gate_line(3, value, ms, met), met, ms, sigs))
    else:
        out.append(GateResult(3, gate_line(3, "no roll pass", None, None), None, None, []))
    return out


def cross_checks(clicks: list[Click], frames: list[Frame], activities: list[Activity]) -> list[str]:
    """`cross-check` lines: per notch (each drains its own log), and per roll (drained per burst,
    so only the pass total lines up), plus one count of disagreements."""
    out, groups = [], []
    for c in clicks:
        if c.sweep == "roll" and groups and groups[-1][0].sweep == "roll":
            groups[-1].append(c)
        else:
            groups.append([c])
    for group in groups:
        stats = [click_stats(c, frames, activities) for c in group]
        builds = _unique((c, a) for c, st in zip(group, stats, strict=True) for a in st.builds_in_paint)
        stalls = _unique((c, a) for c, st in zip(group, stats, strict=True) for a in st.stalls)
        label = f"{group[0].sweep} #{group[0].index}" if len(group) == 1 else f"{group[0].sweep} ({len(group)} clicks)"
        line = cross_check(label, (len(builds), len(stalls)), [ln for c in group for ln in c.lines])
        if line:
            out.append(line)
    out.append(f"  cross-check: {sum('MISMATCH' in ln for ln in out)} disagreements")
    return out


def _f(v: float | None, fmt: str = ".1f") -> str:
    return "-" if v is None else format(v, fmt)


def format_row(click: Click, st: ClickStats) -> str:
    acc = "acc" if click.accepted else "REF"
    head = (
        f"  {click.sweep} #{click.index:02d} {'+' if click.direction > 0 else '-'} {acc}"
        f" mip {click.mip_before}->{click.mip_after} res {'Y' if click.resident_before else 'n'}"
        f" resid {_f(click.residual, '.2f')} pct {_f(click.pct_before, '.0f')}->{_f(click.pct_after, '.0f')}"
    )
    lat = (
        f"click->frame {_f(st.to_frame_ms)} (q {st.queue_ms:.1f} h {st.handler_ms:.1f} w {_f(st.wait_ms)}"
        f" f {_f(st.frame_ms)}) quiet {_f(st.to_quiet_ms)}"
    )
    slow = "canvas -"
    wc = st.worst_canvas
    if wc is not None:
        br = " ".join(f"{k} {v:.1f}" for k, v in wc.bracket.items() if v >= 0.05)
        items, flush = wc.outside()
        slow = (f"canvas {wc.canvas_ms:.1f} mip {wc.mip} [{br}] in frame {wc.ms:.1f} (other items {items:.1f};"
                f" flush {flush:.1f}); slowest frame {st.slowest.ms:.1f}")
    extras = []
    if st.levels:
        extras.append("levels " + ", ".join(f"{a.detail} {a.ms:.1f}" for a in st.levels))
    if st.stalls:
        extras.append("stalls " + " ".join(f"{a.ms:.0f}" for a in st.stalls))
    if st.gc2_ms:
        extras.append(f"gc2 {st.gc2_ms:.1f}")
    if st.warm:
        w = max(st.warm, key=lambda a: a.ms)
        extras.append(f"warm {len(st.warm)} ticks max {w.ms:.1f} ({w.kind})")
    return " | ".join([head, lat, slow, *extras])


# --- Qt side -----------------------------------------------------------------


def _pump(seconds: float) -> None:
    """bench_gui_drag._pump with a 2 ms sleep, not 10: a click arriving mid-sleep would read as queue latency."""
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.002)


class _Recorder:
    """Class-level wrappers installed before the window exists: wheel handling,
    top-level frames, canvas paint brackets, level events, warm ticks, stalls, gc."""

    def __init__(self, profile_paint: int = 0, profile_frame: int = 0) -> None:
        self.frames: list[Frame] = []
        self.activities: list[Activity] = []
        self.wheel: list[dict] = []
        self.offsets: list[float] = []  # handler entry minus X timestamp, ms
        self._canvas_acc: dict[str, float] = {}
        self._frame_mip: int | None = None
        self._view_ms = 0.0
        self._saved: list[tuple[object, str, object]] = []
        self._gc_t0: dict[int, float] = {}
        self.profile_paint = profile_paint
        self.profile_frame = profile_frame
        self.profiles: list[tuple[float, object]] = []
        self.all_profiles: list[object] = []

    def _wrap(self, owner, name: str, wrapper) -> None:
        own = not isinstance(owner, type) or name in owner.__dict__
        original = getattr(owner, name)
        self._saved.append((owner, name, original if own else None))
        setattr(owner, name, wrapper(original))

    def restore(self) -> None:
        import gc

        for owner, name, original in reversed(self._saved):
            # An inherited sip method set back as a class attribute aborts on its next virtual call.
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)

    def x_to_clock(self, x_ms: int) -> float:
        """An X timestamp on the perf_counter clock (seconds), by the smallest offset seen."""
        return (x_ms + min(self.offsets)) / 1000

    def keep_profile(self, ms: float, prof, keep: int) -> None:
        self.all_profiles.append(prof)
        self.profiles.append((ms, prof))
        self.profiles.sort(key=lambda p: -p[0])
        del self.profiles[keep:]

    def profile_texts(self) -> list[tuple[float, str]]:
        """The kept profiles, top 25 by cumtime, plus top 15 by tottime for a frame
        (Qt's own C++ painting and flush show as QMainWindow.event's tottime).
        The first entry (ms -1) is every profiled paint or frame summed."""
        import io
        import pstats

        out = []
        if self.all_profiles:
            total = pstats.Stats(self.all_profiles[0])
            for prof in self.all_profiles[1:]:
                total.add(prof)
            buf = io.StringIO()
            total.stream = buf
            total.sort_stats("tottime").print_stats(30)
            out.append((-1.0, f"(all {len(self.all_profiles)} summed, by tottime)\n{buf.getvalue()}"))
        for ms, prof in self.profiles:
            buf = io.StringIO()
            stats = pstats.Stats(prof, stream=buf)
            stats.sort_stats("cumulative").print_stats(25)
            if self.profile_frame:
                stats.sort_stats("tottime").print_stats(15)
            out.append((ms, buf.getvalue()))
        return out

    def install(self) -> None:
        import cProfile
        import gc

        from PyQt5.QtCore import QEvent

        from descape import map_view, perf_trace, viewer, viewer_canvas

        rec = self
        now = time.perf_counter

        def wheel(original):
            def wrapper(view, event):
                t_entry = now()
                rec.offsets.append(t_entry * 1000 - event.timestamp())
                pct0 = view.zoom_percent_of_fit()
                mip0 = view._mip_for_zoom_factor(1.0)
                cache = view._canvas_item._cache if view._canvas_item is not None else None
                # Built (is_level_resident) and warm (chunks cached at that level): G1's population.
                resident = set()
                if cache is not None:
                    resident = {m for m in cache.resident_levels() if cache.is_level_resident(m)}
                try:
                    return original(view, event)
                finally:
                    t_exit = now()
                    rec.wheel.append({
                        "ts": event.timestamp(), "entry": t_entry, "exit": t_exit, "delta": event.angleDelta().y(),
                        "pct0": pct0, "pct1": view.zoom_percent_of_fit(), "mip0": mip0,
                        "mip1": view._mip_for_zoom_factor(1.0), "resident": resident, "residual": rec.residual(view),
                    })
            return wrapper

        def window_event(original):
            def wrapper(win, event):
                if event.type() != QEvent.UpdateRequest:
                    return original(win, event)
                rec._canvas_acc = {}
                rec._frame_mip = None
                rec._view_ms = 0.0
                prof = cProfile.Profile() if rec.profile_frame else None
                t0 = now()
                try:
                    if prof is not None:
                        prof.enable()
                    return original(win, event)
                finally:
                    if prof is not None:
                        prof.disable()
                    frame = Frame(t0, now(), dict(rec._canvas_acc), rec._frame_mip, rec._view_ms)
                    rec.frames.append(frame)
                    if prof is not None:
                        rec.keep_profile(frame.ms, prof, rec.profile_frame)
            return wrapper

        def viewport_event(original):
            def wrapper(view, event):
                if event.type() != QEvent.Paint:
                    return original(view, event)
                t0 = now()
                try:
                    return original(view, event)
                finally:
                    rec._view_ms += (now() - t0) * 1000
            return wrapper

        def canvas_paint(original):
            def wrapper(item, *a, **k):
                phases0 = {n: v[1] for n, v in perf_trace._repaint_phases.items()}
                level0 = perf_trace._repaint_level_ms
                prof = cProfile.Profile() if rec.profile_paint else None
                t0 = now()
                try:
                    if prof is not None:
                        prof.enable()
                    return original(item, *a, **k)
                finally:
                    if prof is not None:
                        prof.disable()
                    ms = (now() - t0) * 1000
                    acc = rec._canvas_acc
                    timed = 0.0
                    for n, v in perf_trace._repaint_phases.items():
                        d = v[1] - phases0.get(n, 0.0)
                        if d > 0:
                            acc[n] = acc.get(n, 0.0) + d
                            timed += d
                    lv = perf_trace._repaint_level_ms - level0
                    if lv > 0:
                        acc["levels"] = acc.get("levels", 0.0) + lv
                    acc["other"] = acc.get("other", 0.0) + max(0.0, ms - timed - max(lv, 0.0))
                    rec._frame_mip = getattr(item, "_last_mip", None)
                    if prof is not None:
                        rec.keep_profile(ms, prof, rec.profile_paint)
            return wrapper

        def level_event(original):
            def wrapper(kind, mip, ms, tiles=None, split=None):
                where = perf_trace._where_stack[-1] if perf_trace._where_stack else "other"
                t = now()
                rec.activities.append(Activity(t - ms / 1000, t, "level", f"{mip} {kind} ({where})", dict(split or {}), where))
                return original(kind, mip, ms, tiles, split)
            return wrapper

        def level_warm_tick(original):
            def wrapper(tick_ms, install_ms, split=None, gc_ms=None, max_step=None):
                t = now()
                rec.activities.append(Activity(t - tick_ms / 1000, t, "level-warm", "", dict(split or {})))
                return original(tick_ms, install_ms, split, gc_ms, max_step)
            return wrapper

        def warm_tick(original):
            def wrapper(tick_ms, chunks, slowest_chunk_ms):
                t = now()
                rec.activities.append(Activity(t - tick_ms / 1000, t, "margin-warm", f"{chunks} chunks"))
                return original(tick_ms, chunks, slowest_chunk_ms)
            return wrapper

        def stall(original):
            def wrapper(late_ms):
                t = now()
                rec.activities.append(Activity(t - late_ms / 1000, t, "stall", ""))
                original(late_ms)
                text = perf_trace.debug_log._lines[-1] if perf_trace.debug_log._lines else ""
                rec.activities[-1].detail = strip_ts(text).removeprefix("perf stall ")
            return wrapper

        self._wrap(map_view.MapView, "wheelEvent", wheel)
        self._wrap(viewer.ViewerWindow, "event", window_event)
        self._wrap(map_view.MapView, "viewportEvent", viewport_event)
        self._wrap(viewer_canvas.MapCanvasItem, "paint", canvas_paint)
        self._wrap(perf_trace, "level_event", level_event)
        self._wrap(perf_trace, "level_warm_tick", level_warm_tick)
        self._wrap(perf_trace, "warm_tick", warm_tick)
        self._wrap(perf_trace, "stall", stall)
        gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase: str, info: dict) -> None:
        gen = info["generation"]
        if phase == "start":
            self._gc_t0[gen] = time.perf_counter()
        elif gen in self._gc_t0:
            t0 = self._gc_t0.pop(gen)
            self.activities.append(Activity(t0, time.perf_counter(), "gc", f"gen{gen}"))

    @staticmethod
    def residual(view) -> float | None:
        """Device pixels per level pixel at the selected mip: (0.5, 1.0] inside the ladder."""
        from descape.viewer_canvas import _max_axis_scale

        item = view._canvas_item
        if item is None:
            return None
        scale = _max_axis_scale(item.deviceTransform(view.viewportTransform())) * view.devicePixelRatioF()
        mip = item._cache.mip_for_scale(scale)
        return scale * item._cache.mip_scale(mip)


class _Bench:
    def __init__(self, args, rec: _Recorder, window) -> None:
        self.args = args
        self.rec = rec
        self.window = window
        self.view = window.map_view
        self.clicks: list[Click] = []
        self.output: list[str] = []
        self.full_drains = 0

    def drain(self) -> list[str]:
        from descape import debug_log

        text = debug_log.get_log_text()
        debug_log.clear()
        lines = [] if text == "(empty)" else text.splitlines()
        if len(lines) >= DEBUG_LOG_LINES:
            self.full_drains += 1
        return lines

    def flush(self) -> list[str]:
        from descape import perf_trace

        perf_trace.flush_idle()
        return self.drain()

    def quiet(self, quiet_s: float = 0.3, cap_s: float = 20.0) -> None:
        """Pumps until no frame of QUIET_FRAME_MS or more has landed for quiet_s."""
        from PyQt5.QtWidgets import QApplication

        start = time.perf_counter()
        frames = self.rec.frames
        seen, last = len(frames), start
        while time.perf_counter() - start < cap_s:
            QApplication.processEvents()
            time.sleep(0.002)
            if len(frames) != seen:
                if any(f.ms >= QUIET_FRAME_MS for f in frames[seen:]):
                    last = time.perf_counter()
                seen = len(frames)
            elif time.perf_counter() - last > quiet_s:
                return

    def centre(self) -> tuple[int, int]:
        from PyQt5.QtCore import QPoint

        vp = self.view.viewport()
        p = vp.mapToGlobal(QPoint(vp.width() // 2, vp.height() // 2))
        return p.x(), p.y()

    def calibrate(self) -> None:
        """Mouse moves at rest: their X timestamps seed the clock offset."""
        from PyQt5.QtCore import QEvent, QObject

        rec = self.rec

        class _Moves(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.MouseMove:
                    rec.offsets.append(time.perf_counter() * 1000 - event.timestamp())
                return False

        filt = _Moves()
        self.view.viewport().installEventFilter(filt)
        x, y = self.centre()
        for i in range(8):
            _xdotool(["mousemove", str(x + (i % 2) * 3), str(y)])
            _pump(0.1)
        _xdotool(["mousemove", str(x), str(y)])
        _pump(0.2)
        self.view.viewport().removeEventFilter(filt)
        if not rec.offsets:
            raise SystemExit("calibration: no MouseMove reached the viewport (is the window on screen?)")

    def _take_clicks(self, sweep: str, direction: int, start: int, until: float) -> list[Click]:
        """Clicks from the wheel events recorded since index `start`."""
        out = []
        events = self.rec.wheel[start:]
        for ev in events:
            accepted = ev["pct0"] is not None and ev["pct1"] is not None and abs(ev["pct1"] - ev["pct0"]) > 1e-6
            out.append(Click(
                sweep, 0, 1 if ev["delta"] > 0 else -1, accepted, ev["pct0"], ev["pct1"], ev["mip0"], ev["mip1"],
                ev["mip1"] in ev["resident"], ev["residual"], self.rec.x_to_clock(ev["ts"]), ev["entry"], ev["exit"],
            ))
        for i, c in enumerate(out):
            c.window_end = out[i + 1].t_entry if i + 1 < len(out) else until
        return out

    def notch_sweep(self, sweep: str) -> None:
        legs = [(4, "ceiling"), (5, "floor"), (4, "fit")]
        n = 0
        x, y = self.centre()
        _xdotool(["mousemove", str(x), str(y)])
        _pump(0.2)
        self.flush()
        for button, goal in legs:
            for _ in range(80):
                start = len(self.rec.wheel)
                _xdotool(["click", str(button)])
                _pump(self.args.notch_gap)
                self.quiet()
                lines = self.flush()
                clicks = self._take_clicks(sweep, 0, start, time.perf_counter())
                if not clicks:
                    raise SystemExit(f"{sweep}: a wheel click reached no wheelEvent")
                for c in clicks:
                    n += 1
                    c.index = n
                clicks[-1].lines = lines
                self.clicks += clicks
                last = clicks[-1]
                if not last.accepted:
                    break
                if goal == "fit" and last.pct_after is not None and last.pct_after >= 99.0:
                    break

    def roll(self, sweep: str = "roll") -> None:
        x, y = self.centre()
        _xdotool(["mousemove", str(x), str(y)])
        _pump(0.2)
        self.quiet(quiet_s=1.0)
        self.flush()
        start = len(self.rec.wheel)
        drained: list[str] = []
        for button in (4, 5):
            for _ in range(40):
                pct0 = self.view.zoom_percent_of_fit()
                _xdotool(["click", "--repeat", "10", "--delay", str(self.args.roll_delay), str(button)])
                _pump(0.02)
                drained += self.drain()
                if self.view.zoom_percent_of_fit() == pct0:
                    break
        self.quiet(quiet_s=1.0)
        drained += self.flush()
        clicks = self._take_clicks(sweep, 0, start, time.perf_counter())
        for i, c in enumerate(clicks, 1):
            c.index = i
        if clicks:
            clicks[-1].lines = drained
        self.clicks += clicks
        accepted = sum(c.accepted for c in clicks)
        ops = op_zoom_count([strip_ts(ln) for ln in drained])
        self.output.append(f"--- roll: {len(clicks)} clicks, {accepted} accepted (perf op zoom count {ops})")


def _run_combo(args, style: str, sprites: bool, passes: list[str]) -> list[str]:
    from PyQt5.QtWidgets import QApplication

    from descape import composite_backend, debug_log, perf_trace, viewer
    from testkit.qt_window import ScenarioLoadError

    if composite_backend.active_backend() != "native" and not args.allow_numpy:
        raise SystemExit(f"composite backend is {composite_backend.describe()}; build the native kernel or --allow-numpy")
    perf_trace.enable(True)  # before the window: its constructor starts the stall watchdog and gc hook
    rec = _Recorder(args.profile_paint, args.profile_frame)
    rec.install()
    window = viewer.ViewerWindow()
    output: list[str] = []
    try:
        if style != "Stepped":
            window.terrain_style_combo.setCurrentText(style.removesuffix("-topdown"))
        if style == "Flat-topdown":
            window.iso_action.setChecked(False)  # FlatChunkCache; plain "Flat" is Flat + Isometric View
        window.show_sprites_action.setChecked(sprites)
        window.move(0, 0)
        window.resize(*args.window)
        window.show()
        _pump(0.5)
        debug_log.clear()
        t_load = time.perf_counter()
        window.load_scenario(Path(args.scenario))
        if window.scenario is None:
            raise ScenarioLoadError(f"{args.scenario} failed to load")
        load_s = time.perf_counter() - t_load
        _pump(args.settle)
        perf_trace.flush_idle()
        load_lines = debug_log.get_log_text().splitlines()
        debug_log.clear()
        bench = _Bench(args, rec, window)
        bench.calibrate()
        bench.drain()
        view = window.map_view
        cache = window._cache
        vp = view.viewport()
        mm = window.scenario.map_manager
        output.append(
            f"{style}, sprites {'on' if sprites else 'off'} (cache {type(cache).__name__},"
            f" sprites {getattr(cache, 'sprites_enabled', None)}),"
            f" trees {window.paint_trees_check.isChecked()}, eye candy {window.paint_eye_candy_check.isChecked()},"
            f" mode {window.mode}, passes {','.join(passes)}"
        )
        output.append(
            f"map {mm.map_width}x{mm.map_height}, units {sum(len(u) for u in window.scenario.unit_manager.units)},"
            f" window {window.width()}x{window.height()}, viewport {vp.width()}x{vp.height()},"
            f" screen {QApplication.primaryScreen().size().width()}x{QApplication.primaryScreen().size().height()},"
            f" mips {cache.mip_levels()}, load {load_s:.2f}s, clock offset spread"
            f" {max(rec.offsets) - min(rec.offsets):.1f} ms over {len(rec.offsets)} calibration events"
        )
        output.append(composite_backend.describe())
        output.append("--- load log (context)")
        output += [f"  {ln}" for ln in load_lines]
        for name in passes:
            t0 = len(bench.clicks)
            if name == "roll":
                bench.roll()
            else:
                bench.notch_sweep(name)
                output.append(f"--- {name}: {len(bench.clicks) - t0} clicks")
            bench.quiet(quiet_s=1.0)
            bench.flush()
        frames, acts = rec.frames, rec.activities
        current = None
        for c in bench.clicks:
            if c.sweep != current:
                current = c.sweep
                output.append(f"=== rows: {current}")
            output.append(format_row(c, click_stats(c, frames, acts)))
            output += [f"      {ln}" for ln in c.lines]
        if bench.output:
            output += bench.output
        if bench.full_drains:
            output.append(f"WARNING: {bench.full_drains} drains came back exactly {DEBUG_LOG_LINES} lines (ring overflow)")
        per_pass = {}
        for c in bench.clicks:
            per_pass.setdefault(c.sweep, []).append(c)
        output.append("=== cross-check (wrappers vs Perf Trace's log)")
        output += cross_checks(bench.clicks, frames, acts)
        output.append("=== summary")
        for name, cl in per_pass.items():
            acc = [c for c in cl if c.accepted]
            ops = op_zoom_count([strip_ts(ln) for c in cl for ln in c.lines])
            output.append(f"  {name}: {len(cl)} clicks, {len(acc)} accepted, {len(cl) - len(acc)} refused, perf op zoom count {ops}")
        results = gates(bench.clicks, frames, acts)
        for g in results:
            output.append(g.line)
            output += [f"    signature: {s}" for s in g.signatures]
        if rec.profiles:
            what = "frames" if args.profile_frame else "canvas paints"
            output.append(f"=== profile: {len(rec.profiles)} slowest {what} (cProfile-inflated)")
            for ms, text in rec.profile_texts():
                output.append(f"--- {what[:-1]} {ms:.1f} ms")
                output.append(text)
        output.append("GATES " + json.dumps({
            "style": style, "sprites": sprites, "passes": passes,
            "gates": {g.n: {"met": g.met, "dev_ms": g.dev_ms, "line": g.line, "signatures": g.signatures} for g in results},
        }))
    finally:
        if window.scenario is not None:
            window.edit_history.mark_saved()
        window.close()
        rec.restore()
    return output


def _child(args) -> None:
    root = (args.root or ROOT).resolve()
    sys.path.insert(0, str(root))
    from testkit import qt_window, settings_isolation

    os.environ["QT_QPA_PLATFORM"] = "xcb"
    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    style, sprites = args.child.split(",")
    with tempfile.TemporaryDirectory(prefix="bench_gui_zoom_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        output = _run_combo(args, style, sprites == "on", args.passes)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    print(f"descape from {root} ({sha.stdout.strip() or '?'}); loadavg {_loadavg()}")
    print("\n".join(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--style", type=_csv(str), default=["Stepped", "Sloped", "Flat"], help="comma-separated styles")
    parser.add_argument("--sprites", type=_csv(str), default=["on", "off"], help="comma-separated on/off")
    parser.add_argument("--pass", dest="passes", type=_csv(str), default=["notch", "revisit", "roll"], help="passes, in order")
    parser.add_argument("--cold-roll", choices=["on", "off"], default="on", help="a second roll-first process per combination")
    parser.add_argument("--window", type=_size, default=(1920, 1080), help="main window size WxH (default 1920x1080)")
    parser.add_argument("--notch-gap", type=float, default=0.6, help="seconds after each notch before the quiet wait")
    parser.add_argument("--roll-delay", type=int, default=30, help="ms between clicks in a roll burst")
    parser.add_argument("--settle", type=float, default=5.0, help="seconds after the load before the passes")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--profile-paint", type=int, default=0, metavar="N", help="cProfile the N slowest canvas paints")
    profile.add_argument("--profile-frame", type=int, default=0, metavar="N", help="cProfile the N slowest whole frames")
    parser.add_argument("--out-dir", type=Path, help="also write each process's output to <dir>/<style>_<sprites>_<passes>.txt")
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape and testkit from")
    parser.add_argument("--allow-numpy", action="store_true", help="run on the numpy fallback (figures not comparable)")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for p in args.passes:
        if p not in ("notch", "revisit", "roll"):
            parser.error(f"unknown pass {p!r}")
    if not os.environ.get("DISPLAY"):
        sys.exit("needs a real X display (see the Xvfb recipe in --help)")
    if args.child:
        _child(args)
        return
    runs = []
    for style in args.style:
        for sprites in args.sprites:
            runs.append((style, sprites, args.passes))
            if args.cold_roll == "on" and "roll" in args.passes and args.passes[0] != "roll":
                runs.append((style, sprites, ["roll"]))
    table = []
    for style, sprites, passes in runs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", f"{style},{sprites}", "--pass", ",".join(passes)]
        for flag in ("scenario", "notch_gap", "roll_delay", "settle", "profile_paint", "profile_frame"):
            cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        cmd += ["--window", f"{args.window[0]}x{args.window[1]}"]
        if args.root:
            cmd += ["--root", str(args.root)]
        if args.allow_numpy:
            cmd.append("--allow-numpy")
        env = {**os.environ, "DESCAPE_REQUIRE_NATIVE": "1"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        text = proc.stdout + (f"\n--- stderr (exit {proc.returncode})\n{proc.stderr}" if proc.returncode else "")
        print(text, flush=True)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{style}_{sprites}_{'-'.join(passes)}.txt").write_text(text)
        gate = next((ln for ln in proc.stdout.splitlines() if ln.startswith("GATES ")), None)
        table.append((style, sprites, passes, json.loads(gate[6:]) if gate else None, proc.returncode))
    print(f"=== gate table ({Path(args.scenario).name}; loadavg {_loadavg()})")
    for style, sprites, passes, gate, code in table:
        if gate is None:
            print(f"{style:8} sprites {sprites:3} {','.join(passes):20} FAILED (exit {code})")
            continue
        cells = []
        for n in ("1", "2", "3"):
            g = gate["gates"][n]
            if g["met"] is None:
                continue
            cells.append(f"G{n} {g['dev_ms']:.1f} (x4 {g['dev_ms'] * X4:.1f}) {'met' if g['met'] else 'MISSED'}")
        print(f"{style:8} sprites {sprites:3} {','.join(passes):20} " + "; ".join(cells))


if __name__ == "__main__":
    main()
