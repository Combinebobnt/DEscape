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

--resident viewport (default) leaves only what the view has painted
resident. --resident canvas renders the whole mip-0 canvas first, the worst
case for patch(bbox)'s eager recomposite of every resident chunk.

--tool fill replaces the Draw strokes with one Paint Can over a small
region with Trees/Eye candy on, then its undo: a non-Draw wholesale caller.

--force-wholesale sends _after_unit_mutation's `changed` to None, i.e. the
pre-splice path, for a same-build comparison row. --cap overrides
render_cache.UNIT_SPLICE_MAX_UNITS, for finding the splice/wholesale crossover.
--area-ratio overrides viewer._SCOPED_PATCH_AREA_RATIO for every style (inf:
always patch the bbox eagerly, 0: always evict it), for finding the crossover.

--styles takes stepped, sloped, flat (the default Flat + Isometric View, an
IsoChunkCache) and flat2d (Isometric View off: FlatChunkCache).

Columns:
  path       lazy: the handler evicted the whole canvas (the next paint
             recomposites the viewport); evict: it evicted only a bbox's
             chunks; otherwise splice or scoped (invalidate_units() spliced,
             or rebuilt its sources: a _refresh_source_caches() call from
             inside it) with the bbox patched eagerly. An evict row's
             invalidate_units() may have spliced or rebuilt.
  units      unit count before -> after
  handler    the release (or undo) handler's own wall time
  paint      sum of MapCanvasItem.paint time until the canvas goes quiet
  | ...      named sub-timings inside the handler (nested: patch and
             invalidate_units are also counted inside apply_dirty/after_units),
             then `area` when the viewer priced a batch patch: patch_area()
             over the visible chunks' area, the ratio _if_spliceable() tests
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
)
_CACHE_TIMERS = (("invalidate_units", "invalidate_units"), ("patch", "patch"), ("patch_rects", "patch_rects"))


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


class _Timers:
    """Instance-attribute wrappers around the handler's named sub-steps.
    Installed per window/cache and only recording while `armed`."""

    def __init__(self) -> None:
        self.armed = False
        self.ms: dict[str, float] = {}
        self.wholesale = 0
        self.lazy = 0
        self.evict = 0
        self.area: float | None = None
        # (owner, attr, the instance attribute it replaced or _MISSING), in install order.
        self._installed: list[tuple[object, str, object]] = []

    def _replace(self, owner, attr: str, value) -> None:
        self._installed.append((owner, attr, vars(owner).get(attr, _MISSING)))
        setattr(owner, attr, value)

    def _wrap(self, owner, attr: str, label: str) -> None:
        inner = getattr(owner, attr)

        def timed(*args, **kwargs):
            if not self.armed:
                return inner(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                self.ms[label] = self.ms.get(label, 0.0) + (time.perf_counter() - t0) * 1000

        self._replace(owner, attr, timed)

    def install(self, window) -> None:
        for label, path in _WINDOW_TIMERS:
            owner = window
            *parents, attr = path.split(".")
            for p in parents:
                owner = getattr(owner, p)
            self._wrap(owner, attr, label)
        self.install_cache(window._cache, window.map_view)

    def install_cache(self, cache, view) -> None:
        for label, attr in _CACHE_TIMERS:
            if hasattr(cache, attr):
                self._wrap(cache, attr, label)
        # Wholesale = a source rebuild from inside invalidate_units(); patch()
        # and Flat's patch_rects() call _refresh_source_caches() too.
        inner_refresh, inner_invalidate = cache._refresh_source_caches, cache.invalidate_units
        inside = [False]

        def spy(elevation_changed=None):
            if self.armed and inside[0]:
                self.wholesale += 1
            return inner_refresh(elevation_changed)

        def invalidate(changed=None):
            inside[0] = True
            try:
                return inner_invalidate(changed)
            finally:
                inside[0] = False

        self._replace(cache, "_refresh_source_caches", spy)
        self._replace(cache, "invalidate_units", invalidate)
        inner_evict = cache.invalidate_region

        def evict(bbox):
            if self.armed:
                if tuple(bbox) == (0, 0, *cache.canvas_dims(0)):
                    self.lazy += 1
                else:
                    self.evict += 1
            return inner_evict(bbox)

        self._replace(cache, "invalidate_region", evict)
        inner_area = cache.patch_area

        def area(bbox):
            result = inner_area(bbox)
            target = view.viewport_chunk_target()
            if self.armed and target is not None:
                _mip, cx0, cy0, cx1, cy1 = target
                self.area = result / ((cx1 - cx0 + 1) * (cy1 - cy0 + 1) * cache.chunk_px**2)
            return result

        self._replace(cache, "patch_area", area)

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
        self.area = None


def _stroke_tiles(kind: str, w: int, h: int) -> list[tuple[int, int]]:
    cx, cy = w // 2, h // 2
    if kind == "short":
        half = SHORT_STEPS // 2
        return [(cx - half + i, cy) for i in range(SHORT_STEPS)]
    half = LONG_STEPS // 2
    return [
        (x, y) for i in range(LONG_STEPS)
        if 0 <= (x := cx - half + i) < w and 0 <= (y := cy - half + i) < h
    ]


def _unit_count(window) -> int:
    return sum(len(units) for units in window.scenario.unit_manager.units)


def _timed(window, timers: _Timers, paints: list, action) -> tuple[float, float, int, dict, str]:
    """Runs `action` (the handler under test), then pumps the next paint(s)."""
    _pump(0.2)
    # A gen-2 pass over ~13k units otherwise lands at random inside a handler (~300ms).
    gc.collect()
    del paints[:]
    timers.reset()
    timers.armed = True
    t0 = time.perf_counter()
    action()
    handler_ms = (time.perf_counter() - t0) * 1000
    timers.armed = False
    _pump_until_quiet(paints)
    path = "lazy" if timers.lazy else "evict" if timers.evict else "scoped" if timers.wholesale else "splice"
    ms = dict(timers.ms)
    if timers.area is not None:
        ms["area"] = timers.area
    return handler_ms, sum(paints), len(paints), ms, path


def _row(label: str, result, before: int, after: int) -> str:
    handler_ms, paint_ms, n_paints, ms, path = result
    subs = " ".join(f"{k} {v:.2f}" if k == "area" else f"{k} {v:.1f}" for k, v in ms.items())
    return (
        f"      {label:8s} {path:9s} units {before}->{after} "
        f"handler {handler_ms:7.1f}ms  paint {paint_ms:7.1f}ms ({n_paints})  "
        f"total {handler_ms + paint_ms:7.1f}ms | {subs}"
    )


def _run_case(window, timers: _Timers, paints: list, kind: str, brush: int, seed: int) -> list[str]:
    from PyQt5.QtCore import Qt

    from descape import perf_trace, terrain_units

    mm = window.scenario.map_manager
    tiles = _stroke_tiles(kind, mm.map_width, mm.map_height)
    window.brush_size_spin.setValue(brush)
    lines = []
    real_plan = terrain_units.plan_terrain_units
    for stroke_label, terrain_id, stroke_seed in (("A", FOREST_A, seed), ("B", FOREST_B, seed + 1)):
        window.terrain_panel.set_terrain(terrain_id)
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
    for undo_label in ("undo B", "undo A"):
        before = _unit_count(window)
        result = _timed(window, timers, paints, window.undo)
        lines.append(_row(undo_label, result, before, _unit_count(window)))
    return lines


def _run_fill_case(window, timers: _Timers, paints: list, seed: int) -> list[str]:
    """Paint Can with Trees/Eye candy on over a small region, then its undo.
    The region is a brush-9 short stroke of a terrain absent from the map,
    painted untimed with both boxes off, so the fill can't escape into a
    same-terrain neighbour (a big fill would open the modal Large-fill box)."""
    from PyQt5.QtCore import Qt

    from descape import terrain_catalog, terrain_units
    from descape.fill_tools import contiguous_region

    mm = window.scenario.map_manager
    present = {tile.terrain_id for tile in mm.terrain}
    patch = next(e.id for e in terrain_catalog.terrains() if e.id not in present)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)
    window.brush_size_spin.setValue(9)
    window.terrain_panel.set_terrain(patch)
    window.on_edit_stroke_start()
    for tile in _stroke_tiles("short", mm.map_width, mm.map_height):
        window.on_edit_stroke_tiles([tile], Qt.NoModifier)
    window.on_edit_stroke_end()
    _pump(0.5)
    window.paint_trees_check.setChecked(True)
    window.paint_eye_candy_check.setChecked(True)
    window.terrain_panel.set_terrain(FOREST_A)
    cx, cy = mm.map_width // 2, mm.map_height // 2
    region = len(contiguous_region(mm, cx, cy))
    real_plan = terrain_units.plan_terrain_units

    def seeded(*args, **kwargs):
        kwargs["rng"] = random.Random(seed)
        return real_plan(*args, **kwargs)

    lines = []
    terrain_units.plan_terrain_units = seeded
    try:
        before = _unit_count(window)
        result = _timed(window, timers, paints, lambda: window.on_fill(cx, cy, Qt.NoModifier))
        lines.append(_row(f"fill {region}", result, before, _unit_count(window)))
    finally:
        terrain_units.plan_terrain_units = real_plan
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


def _bench_file(
    path: Path, styles, brushes, kinds, sprite_modes, resident: str, force_wholesale: bool, tool: str = "draw"
) -> list[str]:
    from PyQt5.QtGui import QTransform

    from descape import perf_trace, settings, viewer_canvas

    settings._distance_ticks = False
    perf_trace.enable(True)
    paints: list[float] = []
    orig_paint = viewer_canvas.MapCanvasItem.paint

    def timed_paint(self, *a, **k):
        t0 = time.perf_counter()
        orig_paint(self, *a, **k)
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
        window.paint_trees_check.setChecked(True)
        window.paint_eye_candy_check.setChecked(True)
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
            view.setTransform(QTransform())
            mm = window.scenario.map_manager
            view.center_on_tile(mm.map_width // 2, mm.map_height // 2)
            _pump(4.0)
            for sprites in sprite_modes:
                _set_sprites(window, sprites)
                if resident == "canvas":
                    canvas_w, canvas_h = window._cache.canvas_dims(0)
                    window._cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
                timers = _Timers()
                timers.install(window)
                vp = view.viewport()
                emit(
                    f"    style={style} sprites={sprites} resident={resident} "
                    f"viewport {vp.width()}x{vp.height()} units {_unit_count(window)} "
                    f"cache {type(window._cache).__name__}"
                )
                try:
                    if tool == "fill":
                        emit("     fill")
                        for line in _run_fill_case(window, timers, paints, seed=7):
                            emit(line)
                    for kind in kinds if tool == "draw" else ():
                        for brush in brushes:
                            emit(f"     {kind} brush={brush}")
                            for line in _run_case(window, timers, paints, kind, brush, seed=1000 * brush + len(kind)):
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
    parser.add_argument("--strokes", default="short,long", help="Comma-separated: short,long")
    parser.add_argument("--sprites", default="on,off", help="Comma-separated: on,off")
    parser.add_argument("--resident", choices=("viewport", "canvas"), default="viewport")
    parser.add_argument("--tool", choices=("draw", "fill"), default="draw")
    parser.add_argument("--force-wholesale", action="store_true")
    parser.add_argument("--cap", type=int, help="Override render_cache.UNIT_SPLICE_MAX_UNITS")
    parser.add_argument("--area-ratio", type=float, help="Override viewer._SCOPED_PATCH_AREA_RATIO")
    args = parser.parse_args()
    if args.area_ratio is not None:
        from descape import viewer

        viewer._SCOPED_PATCH_AREA_RATIO = dict.fromkeys(viewer._SCOPED_PATCH_AREA_RATIO, args.area_ratio)
    if args.cap is not None:
        from descape import render_cache

        render_cache.UNIT_SPLICE_MAX_UNITS = args.cap

    from descape import composite_backend

    names = [n.strip() for n in args.files.split(",")] if args.files else FILES
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    brushes = [int(b) for b in args.brush.split(",")]
    kinds = [k.strip() for k in args.strokes.split(",")]
    sprite_modes = [s.strip() == "on" for s in args.sprites.split(",")]

    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    print(composite_backend.describe(), flush=True)
    with tempfile.TemporaryDirectory(prefix="bench_stroke_end_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        for name in names:
            path = args.scenario_dir / name
            if not path.exists():
                print(f"  {name} skipped (not found in {args.scenario_dir})")
                continue
            _bench_file(path, styles, brushes, kinds, sprite_modes, args.resident, args.force_wholesale, args.tool)


if __name__ == "__main__":
    main()
