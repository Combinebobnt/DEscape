#!/usr/bin/env python3
"""Load-path cost: scenario_io's Map/Units parse, and File > New Map's
generate-then-parse, the way the GUI runs them (GC on, the previous document
still alive while the next one loads).

Per run it reports:

- **total**: the whole load_map_and_units() call.
- **deferred**: an explicit gen0 collection right after it. A load that
  pauses GC leaves the whole parse uncollected, and the first allocation
  after it would otherwise run that pass wherever it happens to land (in
  the GUI, inside "prepare"). ~0 on a build that does not pause GC, so
  compare **total+def** across builds, not total.
- **map**:Options' end to Units' start. Timed as that gap, not by wrapping
  the Map section's own load, because the terrain fast path walks Map
  without AoE2Scenario._create_and_load_section(); the gap is the same span
  on both paths.
- **units**: the Units section's _create_and_load_section(), or on the
  units fast path scenario_io._load_units_fast().
- **mm/um**: MapManager.construct() / UnitManager.construct(), or on the
  units fast path scenario_io._fast_unit_manager().
- **gc**: time spent in collections during the call (gc.callbacks), and how
  many ran per generation, not counting the deferred pass.

After the runs, the GC-tracked object count with only the last document
alive (after a full collection), against the count before the first load.

`--new-map N` adds File > New Map at N x N: the donor template load, the
splice + recompress (blank_scenario_bytes), then the parse of the result
(load_map_and_units_from_bytes). The donor is loaded here and passed in so
`--library-terrain` reaches it too; without the flag this is exactly what
blank_scenario_bytes(N) does on its own.

`--library-terrain` forces the library's per-tile TerrainStruct walk for
every load, for an A/B against the fast path. On a build without the fast
path (no `fast_terrain` keyword on load_map_and_units) the library walk is
the only path and the flag changes nothing.

`--library-units` does the same for the Units section: fast_units=False,
the library's per-unit UnitStruct walk and UnitManager.construct().

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail. This machine drifts 15-20% run to run; read medians.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AoE2ScenarioParser.objects.aoe2_object import AoE2Object
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager
from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

from descape import scenario_io
from descape.scenario_new import blank_scenario_bytes

HAS_FAST_TERRAIN = "fast_terrain" in inspect.signature(scenario_io.load_map_and_units).parameters
HAS_FAST_UNITS = "fast_units" in inspect.signature(scenario_io.load_map_and_units).parameters


class _GcClock:
    """Accumulates time spent inside collections, via gc.callbacks."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self.passes = [0, 0, 0]
        self._t0 = 0.0

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
        else:
            self.seconds += time.perf_counter() - self._t0
            self.passes[info["generation"]] += 1

    def snapshot(self) -> tuple[float, tuple[int, int, int]]:
        return self.seconds, tuple(self.passes)


class _Probe:
    """Wraps the library entry points the load goes through, recording
    per-section start/end times and the two manager constructs."""

    def __init__(self) -> None:
        self.sections: dict[str, tuple[float, float]] = {}
        self.construct: dict[str, float] = {}
        self._depth = 0
        self._orig_section = AoE2Scenario._create_and_load_section
        self._orig_construct = AoE2Object.__dict__["construct"]
        self._orig_units_fast = getattr(scenario_io, "_load_units_fast", None)
        self._orig_unit_manager = getattr(scenario_io, "_fast_unit_manager", None)

    def install(self) -> None:
        probe = self
        orig_section = self._orig_section
        orig_construct = self._orig_construct.__func__
        orig_units_fast = self._orig_units_fast
        orig_unit_manager = self._orig_unit_manager

        # The loader resolves both as module globals at call time.
        def units_fast(scenario, igen):
            t0 = time.perf_counter()
            try:
                return orig_units_fast(scenario, igen)
            finally:
                probe.sections["Units"] = (t0, time.perf_counter())

        def unit_manager(scenario, fast):
            t0 = time.perf_counter()
            try:
                return orig_unit_manager(scenario, fast)
            finally:
                probe.construct["UnitManager"] = time.perf_counter() - t0

        if orig_units_fast is not None:
            scenario_io._load_units_fast = units_fast
            scenario_io._fast_unit_manager = unit_manager

        def section(scenario, name, igen):
            t0 = time.perf_counter()
            try:
                return orig_section(scenario, name, igen)
            finally:
                probe.sections[name] = (t0, time.perf_counter())

        # On AoE2Object, not MapManager: depoison() strips any attribute
        # added to a POISONED_CLASSES class at the start of every load.
        def construct(cls, *args, **kwargs):
            timed = cls in (MapManager, UnitManager) and probe._depth == 0
            if not timed:
                return orig_construct(cls, *args, **kwargs)
            probe._depth += 1
            t0 = time.perf_counter()
            try:
                return orig_construct(cls, *args, **kwargs)
            finally:
                probe._depth -= 1
                probe.construct[cls.__name__] = time.perf_counter() - t0

        AoE2Scenario._create_and_load_section = section
        AoE2Object.construct = classmethod(construct)

    def uninstall(self) -> None:
        AoE2Scenario._create_and_load_section = self._orig_section
        AoE2Object.construct = self._orig_construct
        if self._orig_units_fast is not None:
            scenario_io._load_units_fast = self._orig_units_fast
            scenario_io._fast_unit_manager = self._orig_unit_manager

    def reset(self) -> None:
        self.sections.clear()
        self.construct.clear()

    def split(self) -> tuple[float, float]:
        """(map, units) seconds for the load just run."""
        units = self.sections.get("Units")
        options = self.sections.get("Options")
        if units is None or options is None:
            return float("nan"), float("nan")
        return units[0] - options[1], units[1] - units[0]


def _load_kwargs(library_terrain: bool, library_units: bool = False) -> dict:
    kwargs = {"fast_terrain": not library_terrain} if HAS_FAST_TERRAIN else {}
    if HAS_FAST_UNITS:
        kwargs["fast_units"] = not library_units
    return kwargs


def _timed_load(probe: _Probe, gc_clock: _GcClock, loader, *args, **kwargs):
    probe.reset()
    gc_before, passes_before = gc_clock.snapshot()
    t0 = time.perf_counter()
    loaded = loader(*args, **kwargs)
    total = time.perf_counter() - t0
    # Read with no tracked allocation first: after a GC-paused load, the
    # next one runs the deferred gen0 pass, which is timed on its own below.
    gc_after = gc_clock.seconds
    p0, p1, p2 = gc_clock.passes
    t1 = time.perf_counter()
    gc.collect(0)
    deferred = time.perf_counter() - t1
    map_s, units_s = probe.split()
    row = {
        "total": total,
        "deferred": deferred,
        "total+def": total + deferred,
        "map": map_s,
        "units": units_s,
        "mm": probe.construct.get("MapManager", float("nan")),
        "um": probe.construct.get("UnitManager", float("nan")),
        "gc": gc_after - gc_before,
        "passes": tuple(a - b for a, b in zip((p0, p1, p2), passes_before, strict=True)),
    }
    return loaded, row


def _fmt_row(label: str, row: dict) -> str:
    passes = "/".join(str(p) for p in row["passes"])
    extra = "".join(f" {k} {row[k]:.3f}" for k in ("generate", "donor") if k in row)
    return (
        f"  {label:<8}{extra} total {row['total']:.3f}  deferred {row['deferred']:.3f}"
        f"  map {row['map']:.3f}  units {row['units']:.3f}"
        f"  mm {row['mm']:.3f}  um {row['um']:.3f}  gc {row['gc']:.3f} ({passes} gen0/1/2)"
    )


def _summary(rows: list[dict]) -> str:
    keys = ("generate", "donor", "total", "deferred", "total+def", "map", "units", "mm", "um", "gc")
    keys = [k for k in keys if k in rows[0]]
    parts = []
    for key in keys:
        values = [r[key] for r in rows]
        parts.append(f"{key} {statistics.median(values):.3f} [{min(values):.3f}-{max(values):.3f}]")
    return "  median " + "  ".join(parts)


def _bench_file(path: Path, runs: int, kwargs: dict, probe: _Probe, gc_clock: _GcClock):
    print(f"\n{path.name}")
    rows = []
    previous = None
    for i in range(runs):
        loaded, row = _timed_load(probe, gc_clock, scenario_io.load_map_and_units, path, **kwargs)
        # Replaced only after the next load returns, like ViewerWindow.scenario.
        previous = loaded
        rows.append(row)
        print(_fmt_row(f"run {i + 1}", row))
    mm = previous.map_manager
    units = sum(len(u) for u in previous.unit_manager.units)
    print(f"  {mm.map_width}x{mm.map_height}, {units:,} units, terrain_write_supported={previous.terrain_write_supported}")
    print(_summary(rows))
    return previous


def _bench_new_map(tiles: int, runs: int, kwargs: dict, probe: _Probe, gc_clock: _GcClock) -> None:
    print(f"\nNew Map {tiles}x{tiles}")
    rows = []
    previous = None
    for i in range(runs):
        gc_before, passes_before = gc_clock.snapshot()
        t0 = time.perf_counter()
        donor = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH, **kwargs)
        t_donor = time.perf_counter() - t0
        data = blank_scenario_bytes(tiles, donor)
        generate = time.perf_counter() - t0
        del donor
        gc_generate, passes_generate = gc_clock.snapshot()
        loaded, row = _timed_load(
            probe,
            gc_clock,
            scenario_io.load_map_and_units_from_bytes,
            data,
            f"blank_{tiles}x{tiles}.aoe2scenario",
            **kwargs,
        )
        previous = loaded
        row["generate"] = generate
        row["donor"] = t_donor
        # gc and passes cover generate + parse, the whole File > New wait.
        row["gc"] += gc_generate - gc_before
        row["passes"] = tuple(p + g - b for p, g, b in zip(row["passes"], passes_generate, passes_before, strict=True))
        rows.append(row)
        print(_fmt_row(f"run {i + 1}", row))
    del previous
    print(_summary(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help=".aoe2scenario files to load")
    parser.add_argument("--runs", type=int, default=5, help="Loads per file (default 5)")
    parser.add_argument("--new-map", type=int, metavar="N", help="Also time File > New Map at N x N")
    parser.add_argument("--library-terrain", action="store_true", help="Force the library's per-tile terrain walk (A/B)")
    parser.add_argument("--library-units", action="store_true", help="Force the library's per-unit walk (A/B)")
    args = parser.parse_args()
    if not args.files and args.new_map is None:
        parser.error("give at least one file or --new-map N")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    kwargs = _load_kwargs(args.library_terrain, args.library_units)
    if HAS_FAST_TERRAIN:
        path_label = "library (forced)" if args.library_terrain else "fast"
    else:
        path_label = "library (this build has no fast path)"
    if HAS_FAST_UNITS:
        units_label = "library (forced)" if args.library_units else "fast"
    else:
        units_label = "library (this build has no fast path)"
    print(
        f"terrain path: {path_label}; units path: {units_label}; GC {'on' if gc.isenabled() else 'off'}; {args.runs} runs"
    )

    gc.collect()
    baseline = len(gc.get_objects())
    gc_clock = _GcClock()
    probe = _Probe()
    gc.callbacks.append(gc_clock)
    probe.install()
    try:
        for path in args.files:
            last = _bench_file(path, args.runs, kwargs, probe, gc_clock)
            # Tracked-object count with exactly this file's last load alive.
            gc.collect()
            print(f"  GC-tracked objects: {len(gc.get_objects()):,} with it loaded, {baseline:,} before any load")
            del last
        if args.new_map is not None:
            _bench_new_map(args.new_map, args.runs, kwargs, probe, gc_clock)
    finally:
        probe.uninstall()
        gc.callbacks.remove(gc_clock)


if __name__ == "__main__":
    main()
