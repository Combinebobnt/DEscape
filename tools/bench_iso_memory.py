#!/usr/bin/env python3
"""Scenario load time and peak memory footprint, across the flat renderer
and the two Phase 0 iso prototypes (numpy, Cython), on the four real
example files used elsewhere in this session's Phase 0 benchmarking.

**Not part of the Phase 0 numpy-vs-Cython backend decision** (that's
settled, decided from
tools/bench_iso_backend.py's per-tile and per-touched-tile numbers) --
load time and memory are both orthogonal to that choice. Load time is
pure AoE2ScenarioParser disk-parse/decompress cost, paid before any
rendering starts, identical regardless of which backend renders
afterward. Memory is backend-*invariant* here specifically: the numpy and
Cython kernels produce byte-identical output at the same array size
(verified in iso_kernel.pyx), so their peak RSS is the same by
construction -- it can't distinguish the two.

What this script actually answers: "how long does opening a file take, and
how much memory does each render mode hold onto, given what actually
recurs during a session." Confirmed directly against descape/viewer.py
before writing this: undo/redo (edit_history.py's apply/undo/redo +
ViewerWindow._apply_dirty) do NOT reload from disk and do NOT do a full
re-render -- they're the already-fast incremental
refresh_tiles()/refresh_units_over() path, already covered by
bench_iso_backend.py's "per-touched-tile incremental" numbers. Opening a
file (ViewerWindow.load_scenario) is the only thing that pays the load-from-disk
cost, paired with exactly one full render. A future Terrain Style switch
(the plan's Phase 3, not built) would pay a full RE-RENDER cost per
refresh_map()'s existing settings-change precedent, but still not a reload
-- so "load time" and "full re-render time" are two different numbers, and
this script measures both, not just one.

Kept for later, specifically: the plan's own Performance impact table
flags an open Phase 3 question -- whether to discard the inactive render
mode's canvas on a Terrain Style toggle, or accept holding both live (the
flat_plus_iso_numpy stage below exists to measure exactly that transient).
Not acted on now, per explicit direction to stop after Phase 0 -- this
script is just where the real numbers for that future call will come from.

Each (file, stage) combination runs in its own subprocess so peak RSS
(resource.getrusage().ru_maxrss, KB on Linux) reflects only that stage, not
accumulated state from earlier measurements sharing the same process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISO_DIR = ROOT / "native" / "iso_render"

FILES = [
    "2_Joan_coop_2_v0_15.aoe2scenario",
    "C2_ElCid_coop_4_v0_13.aoe2scenario",
    "F7_3_York (865).aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
]

# flat_plus_iso_numpy stands in for the plan's flagged "peak memory if both
# a flat and a baked iso canvas must coexist" transient (e.g. switching
# Terrain Style without discarding the previous mode's canvas).
STAGES = ["load_only", "flat", "iso_numpy", "iso_cython", "flat_plus_iso_numpy"]


def _child_main(args: argparse.Namespace) -> None:
    import resource

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ISO_DIR))
    from descape.render import render_terrain, tile_pixels_for_map
    from descape.scenario_io import load_map_and_units

    path = ROOT / "examples" / args.file
    t0 = time.perf_counter()
    scenario = load_map_and_units(path)
    load_s = time.perf_counter() - t0

    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    tiles = list(mm.terrain)

    # Results are bound to real local variables (not discarded) for
    # flat_plus_iso_numpy specifically -- that stage exists to measure both
    # canvases genuinely coexisting (the plan's flagged "up to ~746MB
    # transient" concern), and CPython would otherwise free the flat array
    # the instant render_terrain() returns with nothing holding its result,
    # letting the allocator reuse that memory for the iso array and
    # understating the true combined peak. ru_maxrss is a high-water mark
    # that never decreases mid-process, so the single-canvas stages below
    # don't need this -- their peak is captured at allocation time
    # regardless of what happens to the reference afterward.
    render_s = None
    flat_img = None
    if args.stage in ("flat", "flat_plus_iso_numpy"):
        t0 = time.perf_counter()
        flat_img = render_terrain(scenario)  # noqa: F841 -- held for RSS measurement, see comment above
        render_s = time.perf_counter() - t0
    if args.stage in ("iso_numpy", "flat_plus_iso_numpy"):
        from iso_bench_numpy import render_terrain_iso_numpy

        t0 = time.perf_counter()
        iso_img = render_terrain_iso_numpy(tiles, mm.map_width, mm.map_height, tile_px)  # noqa: F841 -- held for RSS measurement, see comment above
        iso_s = time.perf_counter() - t0
        render_s = iso_s if render_s is None else render_s + iso_s
    if args.stage == "iso_cython":
        from iso_kernel import render_terrain_iso_cython

        t0 = time.perf_counter()
        cython_img = render_terrain_iso_cython(tiles, mm.map_width, mm.map_height, tile_px)  # noqa: F841 -- held for RSS measurement, see comment above
        render_s = time.perf_counter() - t0

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "load_s": load_s,
                "render_s": render_s,
                "peak_rss_kb": peak_rss_kb,
                "w": mm.map_width,
                "h": mm.map_height,
                "tile_px": tile_px,
            }
        )
    )


def run_child(fname: str, stage: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--file", fname, "--stage", stage],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(ROOT),
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--file")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()

    if args.child:
        _child_main(args)
        return

    rows = []
    for fname in FILES:
        for stage in STAGES:
            r = run_child(fname, stage)
            r["file"] = fname
            r["stage"] = stage
            rows.append(r)
            render_ms = f"{r['render_s']*1000:8.1f}ms" if r["render_s"] is not None else "     n/a"
            print(
                f"{fname:38s} {stage:20s} load={r['load_s']*1000:8.1f}ms  "
                f"render={render_ms}  peak_rss={r['peak_rss_kb']/1024:8.1f}MB",
                file=sys.stderr,
            )

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
