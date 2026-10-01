#!/usr/bin/env python3
"""First-unit-edit cost: what the GUI pays the first time a document's units
are edited, after the load has already returned.

Per run, on a fresh load of each file (GC on, as in the GUI):

- **model**: UnitEditModel(loaded), the viewer's lazy construction on the
  first unit edit, including its per-file reproduce gate.
- **edit**: one set_position() on the file's first unit.
- **save**: serialize() of the Units section after that edit (the
  encoder a dirty save runs, or the library commit on a build without one).

The load itself is not timed here; tools/bench_load.py is the load bench.

`--root DIR` imports descape from another checkout, so one copy of this
script times a baseline worktree and a lane branch alike. Without it, the
checkout this script lives in.

`--library-units` loads with fast_units=False (A/B), where the build has
that keyword; otherwise it changes nothing.

`--library-commit` times the save through _serialize_via_commit() (A/B), the
commit serialize() falls back to, where the build has it; otherwise it
changes nothing.

Informational only, matching the other tools/bench_*.py: always runs, never
pass/fail. Read medians.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help=".aoe2scenario files to load")
    parser.add_argument("--runs", type=int, default=5, help="Runs per file (default 5)")
    parser.add_argument("--root", type=Path, help="DEscape checkout to import descape from")
    parser.add_argument("--library-units", action="store_true", help="Load with fast_units=False (A/B)")
    parser.add_argument("--library-commit", action="store_true", help="Save through the library commit (A/B)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    sys.path.insert(0, str(root))
    from descape import scenario_io
    from descape.unit_model import UnitEditModel

    has_fast_units = "fast_units" in inspect.signature(scenario_io.load_map_and_units).parameters
    kwargs = {"fast_units": not args.library_units} if has_fast_units else {}
    label = ("library (forced)" if args.library_units else "fast") if has_fast_units else "library (no fast path)"
    has_via_commit = hasattr(UnitEditModel, "_serialize_via_commit")
    save_label = ("library commit (forced)" if args.library_commit else "encoder") if has_via_commit else "library commit (no encoder)"
    use_commit = args.library_commit and has_via_commit
    print(f"descape from {root}; units path: {label}; save: {save_label}; GC {'on' if gc.isenabled() else 'off'}; {args.runs} runs")

    for path in args.files:
        print(f"\n{path.name}")
        rows = []
        for i in range(args.runs):
            loaded = scenario_io.load_map_and_units(path, **kwargs)
            gc.collect()
            t0 = time.perf_counter()
            model = UnitEditModel(loaded)
            t1 = time.perf_counter()
            unit = next(u for units in loaded.unit_manager.units for u in units)
            model.set_position(unit, unit.x, unit.y, unit.z)
            t2 = time.perf_counter()
            model._serialize_via_commit() if use_commit else model.serialize()
            t3 = time.perf_counter()
            row = {"model": t1 - t0, "edit": t2 - t1, "save": t3 - t2}
            rows.append(row)
            print(f"  run {i + 1}  model {row['model']:.4f}  edit {row['edit']:.4f}  save {row['save']:.4f}")
            del model, loaded
        parts = []
        for key in ("model", "edit", "save"):
            values = [r[key] for r in rows]
            parts.append(f"{key} {statistics.median(values):.4f} [{min(values):.4f}-{max(values):.4f}]")
        print("  median " + "  ".join(parts))


if __name__ == "__main__":
    main()
