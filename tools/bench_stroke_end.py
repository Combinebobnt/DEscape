#!/usr/bin/env python3
"""Draw stroke-end cost with Trees/Eye candy on: the release handler, split
by phase, plus the next canvas paint. Informational only, never pass/fail,
matching bench_fill_latency.py's convention.

Graduated from a scratch probe. A Draw stroke that adds or removes trees or
eye candy ends in _apply_terrain_unit_plan -> _push_terrain_unit_record ->
_after_unit_mutation, and what that tail costs depends on whether the unit
caches take the splice path or the wholesale rebuild. This reports which one
ran, so a row is never read as a splice number when it wasn't.

Drives a real offscreen ViewerWindow. Strokes go through
on_edit_stroke_start/on_edit_stroke_tiles/on_edit_stroke_end directly, one
cursor tile per event with perf_trace.step() between events, exactly as
MapView._on_stroke_tiles does -- not QTest, because a ~150-step diagonal at
1:1 zoom runs far off any viewport. Every stroke passes through the map
centre, where the view is centred, so the next paint is a real visible one.

Per case, two strokes over the same path: A paints FOREST_OAK (grass or
whatever is there becomes forest), then B paints a different forest over A
(forest over forest: every tile removes a tree and rolls a new one, the case
the per-splice guard rejects). Then B and A are undone, each timed the same
way, which also puts the map back for the next case. The planner's rng is
seeded per stroke so runs are comparable.

--strokes rows is GH #180's drag instead of one line: --rows screen-horizontal
rows joined into one serpentine stroke, whose union bbox is most of the view
at fit zoom while the tiles it changes are thin strips. --terrains picks what
A and B paint: 109,13 makes A remove the trees under it and plant none, as
that drag does, and its undo put them back with the terrain. A stroke that
pushes no record has no undo row.

--resident viewport (default) leaves only what the view has painted
resident. --resident canvas renders the whole mip-0 canvas first, the worst
case for patch(bbox)'s eager recomposite of every resident chunk.

--fit runs every case at fit zoom (the zoom a load opens at) instead of 1:1,
so the next paint recomposites the whole visible map: at 240x240 that is all
480 Sloped mip-0 chunks, where --resident canvas makes them resident but
leaves the 1:1 viewport showing a few. The strokes are the same tiles.

--tool fill replaces the Draw strokes with one Paint Can over a small
region with Trees/Eye candy on, then its undo: a non-Draw wholesale caller.
--fill-side N[,N...] fills an N x N tile square at the map centre instead,
one case per size. --trees off keeps Trees and Eye candy unchecked for the
timed edits, so a fill is terrain only: with --styles flat2d that is
top-down Flat's own choice between patching its block rects and evicting
the region (viewer._FLAT_EVICT_RATIO, overridden by --flat-ratio: inf never
evicts by the ratio, 0 always does; past STEPPED_FULL_RERENDER_THRESHOLD
dirty tiles it evicts whatever the ratio). With --resident canvas a fill case
re-renders the mip-0 canvas and repaints before each timed call, so an
eviction by the setup or the fill leaves the next call no less to do.

--blend on|off sets View > Layers > Terrain Blending (on needs the game
install's blendomatic and fails without it).

--force-wholesale sends _after_unit_mutation's `changed` to None, i.e. the
pre-splice path, for a same-build comparison row. --cap overrides
render_cache.UNIT_SPLICE_MAX_UNITS, for finding the splice/wholesale crossover;
with sprites on, a row past it takes the visible-only splice unless
--visible-cap 0 is also passed.
--visible-cap overrides render_cache.VISIBLE_SPLICE_MAX_UNITS, the bound on the
visible-only splice of a batch past both 1000 caps (0: never, today's stale
level; inf: always). A row that took it logs `splice_visible <splices>`.
--area-ratio overrides viewer._TIGHT_PATCH_AREA_RATIO and, on an older build
that still has it, the retired _SCOPED_PATCH_AREA_RATIO, for every style (inf:
always patch the bbox eagerly, 0: always evict it), for finding the crossover.
--rect-ratio overrides viewer._RECT_PATCH_AREA_RATIO, Stepped's second choice
once a batch is past --area-ratio: patch one rect per visible-level chunk
(inf: always, 0: never, i.e. evict as before the rects existed). With
--area-ratio 0 every batch reaches that choice, whatever its bbox.

--styles takes stepped, sloped, flat (the default Flat + Isometric View, an
IsoChunkCache) and flat2d (Isometric View off: FlatChunkCache).

The sprite-bbox plan's split repaint (ViewerWindow._repaint_unit_edit_split:
the visible level sized from real sprite extents, other resident levels
evicting the reach-padded bbox) is reported per row. --bbox reach forces its
REACH_FALLBACK, i.e. the pre-split code path, for a same-build comparison.
The bench also runs on a build without the split, where every unit edit
reports R-base. --zoom-cycle zooms out one mip and back before the strokes,
leaving a second level resident.

Columns:
  path       lazy: the handler evicted the whole canvas (the next paint
             recomposites the viewport); evict: it evicted only a bbox's
             chunks at the visible level; rects: it patched one rect per
             visible-level chunk instead (Stepped, past --area-ratio and
             within --rect-ratio); patch: no unit edit ran and the terrain
             was patched in place; otherwise splice or scoped
             (invalidate_units() spliced, or rebuilt its sources: a
             _refresh_source_caches() call from inside it) with the bbox
             patched eagerly. An evict or rects row's invalidate_units() may have
             spliced or rebuilt. Suffix for a unit edit: :T the tight split
             ran; :R-<why> the reach bbox did (wall, visible: no resident
             chunks at the visible mip, unbuilt, off: sprites off, flat:
             Flat 2D, forced: --bbox reach, base: a build without the split)
  units      unit count before -> after
  handler    the release (or undo) handler's own wall time
  paint      sum of MapCanvasItem.paint time until the canvas goes quiet
  | ...      named sub-timings inside the handler (nested: patch and
             invalidate_units are also counted inside apply_dirty/after_units;
             unit_patch is the patch/evict time inside _patch_unit_edit_cache
             alone; rebuild_in_patch the stale Stepped level rebuild inside
             it; evict_region the time of the bbox or whole-canvas evictions),
             then rebuild_in_paint (the same inside the next paint), evict_others (evictions of non-visible levels only),
             and `area` when the viewer priced a batch patch: patch_area() over
             the visible chunks' area, the ratio _patch_area_exceeds_viewport() tests;
             `rect_area` the same for the per-chunk rects it priced (`rects` of them),
             whether it then patched them (path rects) or evicted; rect_bbox the
             sizing of the block rects behind them, rect_patch the
             patch_composite_rects() time, also inside unit_patch;
             splice_visible the splices invalidate_units() put into the visible
             level alone (a batch past both 1000 caps)
  bbox line  (Stepped/Sloped unit edits) tight / reach: dims, patch_area()
             over every resident level and resident chunks per level
             (mip:count) each would touch; sub: whether tight lies inside
             reach (it must); probe: ms spent on these diagnostics, excluded
             from handler
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import _tight_bbox

from testkit import qt_window, settings_isolation

FILES = ["old-allies-final-v2.aoe2scenario", "2_Joan_coop_1_v0_13.aoe2scenario"]
# flat is the app default (Flat + Isometric View, an IsoChunkCache); flat2d unchecks
# Isometric View for the top-down FlatChunkCache.
STYLES = {"stepped": "Stepped", "sloped": "Sloped", "flat": "Flat", "flat2d": "Flat"}
FOREST_A = 10  # FOREST_OAK, density 1000
FOREST_B = 13  # FOREST_PALM_DESERT, density 1000
SHORT_STEPS = 10
LONG_STEPS = 150

# (label, attribute path on the window) timed inside each handler.
_WINDOW_TIMERS = (
    ("record", "edit_history.build_stroke_record"),
    ("plan", "_apply_terrain_unit_plan"),
    ("apply_dirty", "_apply_dirty"),
    ("after_units", "_after_unit_mutation"),
    ("index", "_rebuild_unit_index"),
    ("refs", "_on_unit_references_moved"),
    ("stats", "_repopulate_stats_players"),
    # The block-rect sizing behind Stepped's per-chunk rects; absent on a build without them.
    ("rect_bbox", "_unit_edit_chunk_rects"),
)
_CACHE_TIMERS = (
    ("invalidate_units", "invalidate_units"), ("patch", "patch"), ("patch_rects", "patch_rects"),
    ("rect_patch", "patch_composite_rects"),
)
# Cache calls whose time inside _patch_unit_edit_cache() is also summed as unit_patch.
_UNIT_PATCH_ATTRS = ("patch", "patch_composite_rects")


def _pump(seconds: float) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _pump_until_quiet(paints: list, quiet_s: float = 0.4, cap_s: float = 8.0) -> None:
    """Pumps until no paint has landed for quiet_s, so a lazy rebuild inside
    the first paint is counted however long it takes."""
    from PyQt5.QtWidgets import QApplication

    start = time.perf_counter()
    last_n, last_change = len(paints), start
    while time.perf_counter() - start < cap_s:
        QApplication.processEvents()
        time.sleep(0.005)
        if len(paints) != last_n:
            last_n, last_change = len(paints), time.perf_counter()
        elif time.perf_counter() - last_change > quiet_s:
            return


_MISSING = object()


# True while MapCanvasItem.paint runs, so a Stepped level rebuild is timed only inside the paint.
_IN_PAINT = [False]


def _fallback_kind(cache, changed, mip) -> str:
    """Why sprite_extent_before()/after() answered REACH_FALLBACK, mirroring
    _ChunkCacheBase._extent_fallback()'s checks in order."""
    from descape.render_cache import _const_splice_eligible

    if any(not _const_splice_eligible(s.unit) for s in changed):
        return "wall"
    if mip is None or mip not in cache.resident_levels():
        return "visible"
    return "unbuilt"


class _Timers:
    """Instance-attribute wrappers around the handler's named sub-steps.
    Installed per window/cache and only recording while `armed`.

    Runs against a build with or without the sprite-bbox split repaint
    (ViewerWindow._repaint_unit_edit_split): without it every unit edit
    reports the reach path as `R-base`."""

    def __init__(self, force_reach: bool = False) -> None:
        self.armed = False
        self.force_reach = force_reach
        self.ms: dict[str, float] = {}
        self.wholesale = 0
        self.lazy = 0
        self.evict = 0
        self.evict_others = 0
        self.area: float | None = None
        # The per-chunk rects the viewer priced (Stepped): their area ratio, count, and whether it patched them.
        self.rect_area: float | None = None
        self.rect_count = 0
        self.rects = 0
        self.unit_edits = 0
        self.tight_ran = 0
        self.fallback: str | None = None
        self.bbox_lines: list[str] = []
        # Bench diagnostics only (bbox descriptions, the comparison reach bbox), excluded from handler.
        self.probe_ms = 0.0
        self._in_unit_edit = False
        self._pricing_rects = False
        self._level_depth = 0
        self.rebuild_in_paint = 0.0
        # Splices in each visible-only splice (cache._splice_visible() answering True).
        self.splice_visible: list[int] = []
        # (owner, attr, the instance attribute it replaced or _MISSING), in install order.
        self._installed: list[tuple[object, str, object]] = []

    def _replace(self, owner, attr: str, value) -> None:
        self._installed.append((owner, attr, vars(owner).get(attr, _MISSING)))
        setattr(owner, attr, value)

    def _add(self, label: str, t0: float) -> None:
        self.ms[label] = self.ms.get(label, 0.0) + (time.perf_counter() - t0) * 1000

    def _wrap(self, owner, attr: str, label: str, unit_label: str | None = None) -> None:
        """Times `attr` under `label`; with unit_label, time spent inside
        _patch_unit_edit_cache() is also summed under it."""
        inner = getattr(owner, attr)

        def timed(*args, **kwargs):
            if not self.armed:
                return inner(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                self._add(label, t0)
                if unit_label and self._in_unit_edit:
                    self._add(unit_label, t0)

        self._replace(owner, attr, timed)

    def install(self, window) -> None:
        for label, path in _WINDOW_TIMERS:
            owner = window
            *parents, attr = path.split(".")
            for p in parents:
                owner = getattr(owner, p)
            if hasattr(owner, attr):
                self._wrap(owner, attr, label)
        self.install_cache(window._cache, window.map_view)
        self._install_unit_edit_probe(window)

    def _install_unit_edit_probe(self, window) -> None:
        """Marks _patch_unit_edit_cache() (for unit_patch), whether the tight
        split ran, why not when it didn't, and the bbox(es) it was sized with."""
        from descape import render_cache

        # Absent on a build without the split; then no extent wrapper is installed either.
        REACH_FALLBACK = getattr(render_cache, "REACH_FALLBACK", object())
        cache = window._cache
        inner_edit = window._patch_unit_edit_cache
        split = getattr(window, "_repaint_unit_edit_split", None)
        inner_bbox = window._unit_edit_bbox

        def patch_unit_edit_cache(changed, *args, **kwargs):
            if not self.armed:
                return inner_edit(changed, *args, **kwargs)
            self.unit_edits += 1
            self._in_unit_edit = True
            try:
                return inner_edit(changed, *args, **kwargs)
            finally:
                self._in_unit_edit = False

        self._replace(window, "_patch_unit_edit_cache", patch_unit_edit_cache)

        def unit_edit_bbox(changed, *args, **kwargs):
            result = inner_bbox(changed, *args, **kwargs)
            if not self.armed or result[0] is None:
                return result
            t0 = time.perf_counter()
            tight = bool(args or kwargs) and all(a is not REACH_FALLBACK for a in (*args, *kwargs.values()))
            if tight:
                reach, _ = inner_bbox(changed)
                self.bbox_lines.append(
                    f"tight {_tight_bbox.describe(cache, result[0])} | reach {_tight_bbox.describe(cache, reach)} | "
                    f"sub {_tight_bbox.contains(reach, result[0])}"
                )
            elif not self.bbox_lines or not self.bbox_lines[-1].startswith("tight"):
                # The split's own reach call (for the other levels) is already described above.
                self.bbox_lines.append(f"reach {_tight_bbox.describe(cache, result[0])}")
            self.probe_ms += (time.perf_counter() - t0) * 1000
            return result

        self._replace(window, "_unit_edit_bbox", unit_edit_bbox)

        if split is not None:
            def repaint_split(*args, **kwargs):
                if self.armed:
                    self.tight_ran += 1
                return split(*args, **kwargs)

            self._replace(window, "_repaint_unit_edit_split", repaint_split)

        inner_exceeds = getattr(window, "_patch_area_exceeds_viewport", None)
        if inner_exceeds is not None:
            def exceeds(bbox, *args, **kwargs):
                rects = kwargs.get("rects")
                if not self.armed or rects is None:
                    return inner_exceeds(bbox, *args, **kwargs)
                # The rects are priced one patch_area() call each: keep `area` the bbox's.
                self._pricing_rects = True
                try:
                    result = inner_exceeds(bbox, *args, **kwargs)
                finally:
                    self._pricing_rects = False
                t0 = time.perf_counter()
                target = window.map_view.viewport_chunk_target()
                if target is not None:
                    mip, cx0, cy0, cx1, cy1 = target
                    area = sum(type(cache).patch_area(cache, rect, (mip,)) for rect in rects)
                    self.rect_area = area / ((cx1 - cx0 + 1) * (cy1 - cy0 + 1) * cache.chunk_px**2)
                    self.rect_count = len(rects)
                self.probe_ms += (time.perf_counter() - t0) * 1000
                return result

            self._replace(window, "_patch_area_exceeds_viewport", exceeds)

        for attr in ("sprite_extent_before", "sprite_extent_after"):
            inner_extent = getattr(cache, attr, None)
            if inner_extent is None:
                continue

            def extent(changed, mip, *args, _inner=inner_extent, **kwargs):
                result = _inner(changed, mip, *args, **kwargs)
                if self.force_reach:
                    result = REACH_FALLBACK
                if self.armed and result is REACH_FALLBACK and self.fallback in (None, "none"):
                    t0 = time.perf_counter()
                    self.fallback = "forced" if self.force_reach else _fallback_kind(cache, changed, mip)
                    self.probe_ms += (time.perf_counter() - t0) * 1000
                elif self.armed and self.fallback is None:
                    self.fallback = "none"
                return result

            self._replace(cache, attr, extent)

    def install_cache(self, cache, view) -> None:
        for label, attr in _CACHE_TIMERS:
            if hasattr(cache, attr):
                self._wrap(cache, attr, label, unit_label="unit_patch" if attr in _UNIT_PATCH_ATTRS else None)
        inner_rects = getattr(cache, "patch_composite_rects", None)
        if inner_rects is not None:
            # Over the timer just installed, so the count is outside its time.
            def patch_rects(*args, **kwargs):
                self.rects += self.armed
                return inner_rects(*args, **kwargs)

            self._replace(cache, "patch_composite_rects", patch_rects)
        # Wholesale = a source rebuild from inside invalidate_units(); patch()
        # and Flat's patch_rects() call _refresh_source_caches() too.
        inner_refresh, inner_invalidate = cache._refresh_source_caches, cache.invalidate_units
        inside = [False]

        def spy(elevation_changed=None, **kwargs):
            if self.armed and inside[0]:
                self.wholesale += 1
            return inner_refresh(elevation_changed, **kwargs)

        def invalidate(changed=None, **kwargs):
            inside[0] = True
            try:
                return inner_invalidate(changed, **kwargs)
            finally:
                inside[0] = False

        self._replace(cache, "_refresh_source_caches", spy)
        self._replace(cache, "invalidate_units", invalidate)
        inner_visible = getattr(cache, "_splice_visible", None)
        if inner_visible is not None:
            from descape import perf_trace

            def splice_visible(*args, **kwargs):
                spliced = inner_visible(*args, **kwargs)
                if self.armed and spliced and perf_trace.is_enabled():
                    # Its last act is the zero-length `splice_visible=<splices>` phase.
                    name = perf_trace._recent[-1][3] if perf_trace._recent else ""
                    if name.startswith("splice_visible="):
                        self.splice_visible.append(int(name.partition("=")[2]))
                return spliced

            self._replace(cache, "_splice_visible", splice_visible)
        inner_evict = cache.invalidate_region

        def evict(bbox, *args, **kwargs):
            if not self.armed:
                return inner_evict(bbox, *args, **kwargs)
            levels = args[0] if args else kwargs.get("levels")
            target = view.viewport_chunk_target()
            others = levels is not None and (target is None or target[0] not in levels)
            if others:
                self.evict_others += 1
            elif tuple(bbox) == (0, 0, *cache.canvas_dims(0)):
                self.lazy += 1
            else:
                self.evict += 1
            t0 = time.perf_counter()
            try:
                return inner_evict(bbox, *args, **kwargs)
            finally:
                if not others:
                    self._add("evict_region", t0)
                if self._in_unit_edit:
                    self._add("unit_patch", t0)

        self._replace(cache, "invalidate_region", evict)
        inner_area = cache.patch_area

        def area(bbox, *args, **kwargs):
            result = inner_area(bbox, *args, **kwargs)
            target = view.viewport_chunk_target()
            if self.armed and target is not None and not self._pricing_rects:
                _mip, cx0, cy0, cx1, cy1 = target
                self.area = result / ((cx1 - cx0 + 1) * (cy1 - cy0 + 1) * cache.chunk_px**2)
            return result

        self._replace(cache, "patch_area", area)
        inner_level = getattr(cache, "_level", None)
        if inner_level is not None:
            def level(mip, *args, **kwargs):
                # A stale level's rebuild, outermost call only, inside the next paint or the unit patch.
                lvl = cache._levels.get(mip)
                where = "paint" if _IN_PAINT[0] else "patch" if self.armed and self._in_unit_edit else None
                if self._level_depth or where is None or lvl is None or lvl.gen == cache._source_gen:
                    where = None
                self._level_depth += 1
                t0 = time.perf_counter()
                try:
                    return inner_level(mip, *args, **kwargs)
                finally:
                    self._level_depth -= 1
                    if where == "paint":
                        self.rebuild_in_paint += (time.perf_counter() - t0) * 1000
                    elif where == "patch":
                        self._add("rebuild_in_patch", t0)

            self._replace(cache, "_level", level)

    def uninstall(self) -> None:
        # Reverse order, restoring what each wrapper replaced (e.g. --force-wholesale's).
        for owner, attr, previous in reversed(self._installed):
            if previous is _MISSING:
                with contextlib.suppress(AttributeError):
                    delattr(owner, attr)
            else:
                setattr(owner, attr, previous)
        self._installed = []

    def reset(self) -> None:
        self.ms = {}
        self.wholesale = 0
        self.lazy = 0
        self.evict = 0
        self.evict_others = 0
        self.area = None
        self.rect_area = None
        self.rect_count = 0
        self.rects = 0
        self.unit_edits = 0
        self.tight_ran = 0
        self.fallback = None
        self.bbox_lines = []
        self.probe_ms = 0.0
        self.rebuild_in_paint = 0.0
        self.splice_visible = []

    def bbox_kind(self, cache) -> str:
        """The path column's suffix: T when the tight split ran, else R-<why>."""
        if not self.unit_edits:
            return ""
        if self.tight_ran:
            return ":T"
        if self.fallback not in (None, "none"):
            return f":R-{self.fallback}"
        if type(cache).__name__ == "FlatChunkCache":
            return ":R-flat"
        if not cache.sprites_enabled:
            return ":R-off"
        return ":R-base" if self.fallback is None else ":R-nobbox"


def _serpentine_tiles(w: int, h: int, n_rows: int) -> list[tuple[int, int]]:
    """GH #180's Draw drag (bench_gui_drag.py --flow gh180) in tile space:
    n_rows screen-horizontal rows (the tiles of one y - x), spread over +-0.6
    of the map diamond's half height and 90% of its width there, joined end to
    end into one drag that alternates direction. Its union bbox is most of the
    view while the tiles it changes are a few thin strips."""
    from descape import brush

    path: list[tuple[int, int]] = []
    for r in range(n_rows):
        frac = -0.6 + 1.2 * r / max(1, n_rows - 1)
        d = round(frac * (min(w, h) - 1))
        lo, hi = max(0, -d), min(w, h - d) - 1
        inset = (hi - lo) // 20
        xs = list(range(lo + inset, hi - inset + 1))
        if r % 2:
            xs.reverse()
        row = [(x, x + d) for x in xs]
        if path:
            path.extend(brush.line_tiles(*path[-1], *row[0])[:-1])
        path.extend(row)
    return path


def _stroke_tiles(kind: str, w: int, h: int, rows: int = 4) -> list[tuple[int, int]]:
    cx, cy = w // 2, h // 2
    if kind == "short":
        half = SHORT_STEPS // 2
        return [(cx - half + i, cy) for i in range(SHORT_STEPS)]
    if kind == "rows":
        return _serpentine_tiles(w, h, rows)
    half = LONG_STEPS // 2
    return [
        (x, y) for i in range(LONG_STEPS)
        if 0 <= (x := cx - half + i) < w and 0 <= (y := cy - half + i) < h
    ]


def _unit_count(window) -> int:
    return sum(len(units) for units in window.scenario.unit_manager.units)


def _visible_tiles(cache, target) -> int:
    """Tiles in the visible chunks, a ragged edge chunk at its real size:
    what top-down Flat's evict rule prices a patch against."""
    mip, cx0, cy0, cx1, cy1 = target
    canvas_w, canvas_h = cache.canvas_dims(mip)
    size = cache.chunk_px
    px = (min((cx1 + 1) * size, canvas_w) - cx0 * size) * (min((cy1 + 1) * size, canvas_h) - cy0 * size)
    return round(px / cache.mip_tile_px(mip) ** 2)


def _timed(window, timers: _Timers, paints: list, action) -> tuple[float, float, int, dict, str, list[str]]:
    """Runs `action` (the handler under test), then pumps the next paint(s)."""
    _pump(0.2)
    # A gen-2 pass over ~13k units otherwise lands at random inside a handler (~300ms).
    gc.collect()
    del paints[:]
    timers.reset()
    timers.armed = True
    t0 = time.perf_counter()
    action()
    handler_ms = (time.perf_counter() - t0) * 1000 - timers.probe_ms
    timers.armed = False
    _pump_until_quiet(paints)
    path = (
        "lazy" if timers.lazy else "evict" if timers.evict else "rects" if timers.rects
        else "scoped" if timers.wholesale else "splice" if timers.unit_edits else "patch"
    )
    path += timers.bbox_kind(window._cache)
    ms = dict(timers.ms)
    if timers.splice_visible:
        ms["splice_visible"] = sum(timers.splice_visible)
    if timers.rebuild_in_paint:
        ms["rebuild_in_paint"] = timers.rebuild_in_paint
    if timers.evict_others:
        ms["evict_others"] = timers.evict_others
    if timers.area is not None:
        ms["area"] = timers.area
    if timers.rect_area is not None:
        ms["rect_area"] = timers.rect_area
        ms["rects"] = timers.rect_count
    if timers.bbox_lines:
        ms["probe"] = timers.probe_ms
    return handler_ms, sum(paints), len(paints), ms, path, list(timers.bbox_lines)


def _row(label: str, result, before: int, after: int) -> str:
    handler_ms, paint_ms, n_paints, ms, path, bbox_lines = result
    subs = " ".join(
        f"{k} {v:.2f}" if k in ("area", "rect_area") else f"{k} {v}" if k in ("evict_others", "rects", "splice_visible")
        else f"{k} {v:.1f}"
        for k, v in ms.items()
    )
    line = (
        f"      {label:8s} {path:14s} units {before}->{after} "
        f"handler {handler_ms:7.1f}ms  paint {paint_ms:7.1f}ms ({n_paints})  "
        f"total {handler_ms + paint_ms:7.1f}ms | {subs}"
    )
    return "\n".join([line, *(f"        bbox {b}" for b in bbox_lines)])


def _run_case(
    window, timers: _Timers, paints: list, kind: str, brush: int, seed: int, rows: int = 4,
    terrains: tuple[int, int] = (FOREST_A, FOREST_B),
) -> list[str]:
    from PyQt5.QtCore import Qt

    from descape import perf_trace, terrain_units

    mm = window.scenario.map_manager
    tiles = _stroke_tiles(kind, mm.map_width, mm.map_height, rows)
    window.brush_size_spin.setValue(brush)
    lines = []
    real_plan = terrain_units.plan_terrain_units
    # Strokes that pushed a record, newest last: a stroke that changed nothing has no undo.
    pushed = []
    for stroke_label, terrain_id, stroke_seed in (("A", terrains[0], seed), ("B", terrains[1], seed + 1)):
        window.terrain_panel.set_terrain(terrain_id)
        cursor = window.edit_history.cursor
        window.on_edit_stroke_start()
        for tile in tiles:
            window.on_edit_stroke_tiles([tile], Qt.NoModifier)
            perf_trace.step()

        def seeded(*args, _seed=stroke_seed, **kwargs):
            kwargs["rng"] = random.Random(_seed)
            return real_plan(*args, **kwargs)

        terrain_units.plan_terrain_units = seeded
        try:
            before = _unit_count(window)
            result = _timed(window, timers, paints, window.on_edit_stroke_end)
            lines.append(_row(f"stroke {stroke_label}", result, before, _unit_count(window)))
        finally:
            terrain_units.plan_terrain_units = real_plan
        if window.edit_history.cursor != cursor:
            pushed.append(stroke_label)
    for stroke_label in reversed(pushed):
        before = _unit_count(window)
        result = _timed(window, timers, paints, window.undo)
        lines.append(_row(f"undo {stroke_label}", result, before, _unit_count(window)))
    return lines


def _run_fill_case(
    window, timers: _Timers, paints: list, seed: int, side: int | None = None, units: bool = True, prime=None
) -> list[str]:
    """Paint Can over a small region, then its undo, with Trees/Eye candy on
    unless `units` is off. The region is a brush-9 short stroke of a terrain
    absent from the map, or with `side` a side x side square of it at the
    map centre, painted untimed with both boxes off, so the fill can't
    escape into a same-terrain neighbour (a big fill with the boxes on would
    open the modal Large-fill box). `prime`, called before each timed call,
    puts back whatever the call before it evicted."""
    from PyQt5.QtCore import Qt

    from descape import terrain_catalog, terrain_units
    from descape.fill_tools import contiguous_region

    mm = window.scenario.map_manager
    cx, cy = mm.map_width // 2, mm.map_height // 2
    present = {tile.terrain_id for tile in mm.terrain}
    patch = next(e.id for e in terrain_catalog.terrains() if e.id not in present)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.terrain_panel.set_terrain(patch)
    if side is None:
        window.brush_size_spin.setValue(9)
        window.on_edit_stroke_start()
        for tile in _stroke_tiles("short", mm.map_width, mm.map_height):
            window.on_edit_stroke_tiles([tile], Qt.NoModifier)
        window.on_edit_stroke_end()
    else:
        x0, y0 = cx - side // 2, cy - side // 2
        window.on_shape_commit([(x, y) for y in range(y0, y0 + side) for x in range(x0, x0 + side)])
    _pump(0.5)
    window.paint_trees_check.setChecked(units)
    window.paint_eye_candy_check.setChecked(units)
    window.terrain_panel.set_terrain(FOREST_A)
    region = len(contiguous_region(mm, cx, cy))
    real_plan = terrain_units.plan_terrain_units

    def seeded(*args, **kwargs):
        kwargs["rng"] = random.Random(seed)
        return real_plan(*args, **kwargs)

    lines = []
    terrain_units.plan_terrain_units = seeded
    try:
        if prime is not None:
            prime()
            chunks = Counter(key[0] for key in window._cache._cache)
            lines.append("      resident chunks " + " ".join(f"mip {mip}: {n}" for mip, n in sorted(chunks.items())))
        before = _unit_count(window)
        result = _timed(window, timers, paints, lambda: window.on_fill(cx, cy, Qt.NoModifier))
        lines.append(_row(f"fill {region}", result, before, _unit_count(window)))
    finally:
        terrain_units.plan_terrain_units = real_plan
    if prime is not None:
        prime()
    before = _unit_count(window)
    result = _timed(window, timers, paints, window.undo)
    lines.append(_row("undo fill", result, before, _unit_count(window)))
    window.undo()  # the untimed patch stroke
    _pump(0.5)
    return lines


def _set_sprites(window, on: bool) -> None:
    action = window.show_sprites_action
    if action.isChecked() != on:
        action.setChecked(on)
        _pump(3.0)
    assert window._cache.sprites_enabled == on, "sprite toggle did not reach the cache"


def _zoom_cycle(window) -> None:
    """Zooms out to half scale (one mip coarser) and back, leaving that
    level's chunks resident beside mip 0's, the way a user who zooms does."""
    from PyQt5.QtGui import QTransform

    view = window.map_view
    mm = window.scenario.map_manager
    for scale in (0.5, 1.0):
        view.setTransform(QTransform.fromScale(scale, scale))
        view.center_on_tile(mm.map_width // 2, mm.map_height // 2)
        _pump(4.0)


def _bench_file(
    path: Path, styles, brushes, kinds, sprite_modes, resident: str, force_wholesale: bool, tool: str = "draw",
    force_reach: bool = False, zoom_cycle: bool = False, fit: bool = False, rows: int = 4,
    terrains: tuple[int, int] = (FOREST_A, FOREST_B), fill_sides: list[int] | None = None, units: bool = True,
) -> list[str]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QTransform

    from descape import perf_trace, settings, viewer_canvas

    settings._distance_ticks = False
    perf_trace.enable(True)
    paints: list[float] = []
    orig_paint = viewer_canvas.MapCanvasItem.paint

    def timed_paint(self, *a, **k):
        t0 = time.perf_counter()
        _IN_PAINT[0] = True
        try:
            orig_paint(self, *a, **k)
        finally:
            _IN_PAINT[0] = False
        paints.append((time.perf_counter() - t0) * 1000)

    viewer_canvas.MapCanvasItem.paint = timed_paint
    out: list[str] = []

    def emit(line: str) -> None:
        out.append(line)
        print(line, flush=True)

    emit(f"  {path.name}")
    window = qt_window.stepped_window(path, show=False)
    try:
        window.mode_combo.setCurrentText("Terrain")
        window._on_tool_selected("draw")
        window.paint_trees_check.setChecked(units)
        window.paint_eye_candy_check.setChecked(units)
        if force_wholesale:
            real_after = window._after_unit_mutation

            def wholesale_after(changed=None, **kwargs):
                return real_after(None, **kwargs)

            window._after_unit_mutation = wholesale_after
        window.move(0, 0)
        window.resize(1600, 1000)
        window.show()
        _pump(0.5)
        for style in styles:
            if STYLES[style] != window.terrain_style_combo.currentText():
                window.terrain_style_combo.setCurrentText(STYLES[style])
            if style.startswith("flat"):
                window.iso_action.setChecked(style == "flat")
            view = window.map_view
            mm = window.scenario.map_manager
            if fit:
                view.fitInView(view._map_rect if view._map_rect is not None else view.sceneRect(), Qt.KeepAspectRatio)
            else:
                view.setTransform(QTransform())
                view.center_on_tile(mm.map_width // 2, mm.map_height // 2)
            _pump(4.0)
            for sprites in sprite_modes:
                _set_sprites(window, sprites)
                if resident == "canvas":
                    canvas_w, canvas_h = window._cache.canvas_dims(0)
                    window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
                if zoom_cycle:
                    _zoom_cycle(window)
                timers = _Timers(force_reach)
                timers.install(window)

                def prime(view=view) -> None:
                    """Every mip-0 chunk resident again, then a repaint, so the visible level is too."""
                    cache = window._cache
                    cx0, cy0, cx1, cy1 = cache.chunk_index_range(0, 0, 0, *cache.canvas_dims(0))
                    for cy in range(cy0, cy1 + 1):
                        for cx in range(cx0, cx1 + 1):
                            cache.get_chunk(0, cx, cy)
                    view.viewport().update()
                    _pump_until_quiet(paints)

                vp = view.viewport()
                target = view.viewport_chunk_target()
                visible_tiles = target and _visible_tiles(window._cache, target)
                emit(
                    f"    style={style} sprites={sprites} resident={resident} zoom={'fit' if fit else '1:1'} "
                    f"viewport {vp.width()}x{vp.height()} units {_unit_count(window)} "
                    f"cache {type(window._cache).__name__} "
                    f"mips {_tight_bbox.resident_levels(window._cache)} visible {target and target[0]} "
                    f"visible_tiles {visible_tiles}"
                )
                try:
                    for side in (fill_sides or [None]) if tool == "fill" else ():
                        emit("     fill" + (f" side={side}" if side else ""))
                        for line in _run_fill_case(
                            window, timers, paints, seed=7, side=side, units=units,
                            prime=prime if resident == "canvas" else None,
                        ):
                            emit(line)
                    for kind in kinds if tool == "draw" else ():
                        for brush in brushes:
                            emit(f"     {kind} brush={brush}" + (f" rows={rows}" if kind == "rows" else ""))
                            for line in _run_case(
                                window, timers, paints, kind, brush, seed=1000 * brush + len(kind), rows=rows,
                                terrains=terrains,
                            ):
                                emit(line)
                finally:
                    timers.uninstall()
    finally:
        viewer_canvas.MapCanvasItem.paint = orig_paint
        window.edit_history.mark_saved()
        window.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario_dir", type=Path, nargs="?", default=ROOT / "examples")
    parser.add_argument("--files", help="Comma-separated file names, overriding the built-in list")
    parser.add_argument("--styles", default="stepped,sloped,flat")
    parser.add_argument("--brush", default="1,9", help="Comma-separated brush sizes")
    parser.add_argument("--strokes", default="short,long", help="Comma-separated: short,long,rows")
    parser.add_argument("--rows", type=int, default=4, help="Rows in a `rows` stroke (GH #180's drag has 4)")
    parser.add_argument(
        "--terrains", default=f"{FOREST_A},{FOREST_B}",
        help="Terrain ids strokes A and B paint (109,13: A only removes trees, as GH #180's drag does)",
    )
    parser.add_argument("--sprites", default="on,off", help="Comma-separated: on,off")
    parser.add_argument("--resident", choices=("viewport", "canvas"), default="viewport")
    parser.add_argument("--tool", choices=("draw", "fill"), default="draw")
    parser.add_argument("--force-wholesale", action="store_true")
    parser.add_argument("--cap", type=int, help="Override render_cache.UNIT_SPLICE_MAX_UNITS")
    parser.add_argument(
        "--visible-cap", help="Override render_cache.VISIBLE_SPLICE_MAX_UNITS (0: never, inf: always)",
    )
    parser.add_argument("--area-ratio", type=float, help="Override viewer._TIGHT_PATCH_AREA_RATIO")
    parser.add_argument(
        "--rect-ratio", type=float,
        help="Override viewer._RECT_PATCH_AREA_RATIO (Stepped's per-chunk rects; 0: never, inf: always)",
    )
    parser.add_argument(
        "--bbox", choices=("auto", "reach"), default="auto",
        help="auto: whatever the viewer picks; reach: force the reach-padded fallback",
    )
    parser.add_argument("--zoom-cycle", action="store_true", help="Zoom out one mip and back before the strokes")
    parser.add_argument("--fit", action="store_true", help="Fit zoom (the zoom a load opens at), not 1:1")
    parser.add_argument(
        "--flat-ratio", type=float,
        help="Override viewer._FLAT_EVICT_RATIO (top-down Flat's terrain patch; 0: always evict, inf: never by the ratio)",
    )
    parser.add_argument("--fill-side", help="--tool fill: comma-separated sides of square fill regions, one case each")
    parser.add_argument(
        "--trees", choices=("on", "off"), default="on",
        help="Trees and Eye candy for the timed edits (off: terrain only)",
    )
    parser.add_argument(
        "--blend", choices=("on", "off"), default="on",
        help="View > Layers > Terrain Blending (on needs the game install)",
    )
    args = parser.parse_args()
    fill_sides = [int(s) for s in args.fill_side.split(",")] if args.fill_side else None
    if fill_sides and args.trees == "on":
        from descape import viewer

        if max(fill_sides) ** 2 > viewer.TERRAIN_UNIT_CONFIRM_THRESHOLD:
            raise SystemExit("--fill-side: a fill this large with --trees on opens the modal Large-fill box. Pass --trees off.")
    if args.flat_ratio is not None:
        from descape import viewer

        # Absent on a build that patches one rect a tile and never evicts.
        if hasattr(viewer, "_FLAT_EVICT_RATIO"):
            viewer._FLAT_EVICT_RATIO = args.flat_ratio
    if args.area_ratio is not None:
        from descape import viewer

        for name in ("_TIGHT_PATCH_AREA_RATIO", "_SCOPED_PATCH_AREA_RATIO"):
            if hasattr(viewer, name):
                setattr(viewer, name, dict.fromkeys(getattr(viewer, name), args.area_ratio))
    if args.rect_ratio is not None:
        from descape import viewer

        # Absent on a build without the rects, where every over-ratio batch evicts.
        if hasattr(viewer, "_RECT_PATCH_AREA_RATIO"):
            viewer._RECT_PATCH_AREA_RATIO = args.rect_ratio
    if args.cap is not None:
        from descape import render_cache

        render_cache.UNIT_SPLICE_MAX_UNITS = args.cap
    if args.visible_cap is not None:
        from descape import render_cache

        # Absent on a build without the visible-only splice.
        if not hasattr(render_cache, "VISIBLE_SPLICE_MAX_UNITS"):
            raise SystemExit("--visible-cap: this build has no VISIBLE_SPLICE_MAX_UNITS.")
        render_cache.VISIBLE_SPLICE_MAX_UNITS = 10**9 if args.visible_cap == "inf" else int(args.visible_cap)

    from descape import composite_backend, render, settings

    names = [n.strip() for n in args.files.split(",")] if args.files else FILES
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    brushes = [int(b) for b in args.brush.split(",")]
    kinds = [k.strip() for k in args.strokes.split(",")]
    sprite_modes = [s.strip() == "on" for s in args.sprites.split(",")]
    terrain_a, terrain_b = (int(t) for t in args.terrains.split(","))

    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    print(composite_backend.describe(), flush=True)
    with tempfile.TemporaryDirectory(prefix="bench_stroke_end_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        # Before any window: ViewerWindow reads its layers from settings at construction.
        settings.set_view_layer("terrain_blend", args.blend == "on")
        effective = render.blend_effective(settings.get_view_layers())
        if args.blend == "on" and not effective:
            raise SystemExit("--blend on: no blendomatic readable (no AoE2:DE install configured). Pass --blend off.")
        print(f"terrain blending {args.blend}", flush=True)
        for name in names:
            path = args.scenario_dir / name
            if not path.exists():
                print(f"  {name} skipped (not found in {args.scenario_dir})")
                continue
            _bench_file(
                path, styles, brushes, kinds, sprite_modes, args.resident, args.force_wholesale, args.tool,
                force_reach=args.bbox == "reach", zoom_cycle=args.zoom_cycle, fit=args.fit, rows=args.rows,
                terrains=(terrain_a, terrain_b), fill_sides=fill_sides, units=args.trees == "on",
            )


if __name__ == "__main__":
    main()
