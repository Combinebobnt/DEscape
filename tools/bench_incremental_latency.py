#!/usr/bin/env python3
"""Per-touched-tile incremental redraw latency (descape.render.
refresh_region_iso), across every real example file. Informational only,
matching this project's own bench_iso_backend.py/bench_iso_memory.py
convention -- always runs, never pass/fail. One representative single-tile
elevation raise per file, timed with perf_counter, best-of-5.

Extracted from tools/verify_iso_incremental.py's bench_incremental_latency()
(the pytest migration plan's Ordering step 4) -- it never had a check_
prefix or a pass/fail verdict, so it belongs here as a benchmark tool, not
as a migrated test. This is the number Phase 3's edit-disable scope cut
(decision #6) was explicitly conditioned on Phase 4 eventually producing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation
from descape.render import refresh_region_iso, render_terrain_iso_with_proj, tile_pixels_for_map
from descape.scenario_io import load_map_and_units


def _editable(scenario) -> bool:
    return scenario.terrain_write_supported and scenario.map_is_square


def bench_incremental_latency(files: list[Path]) -> str:
    lines = []
    for path in sorted(files):
        scenario = load_map_and_units(path)
        if not _editable(scenario):
            lines.append(f"  {path.name:38s} skipped (no elevation editing)")
            continue
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        img, elevations, proj = render_terrain_iso_with_proj(scenario)
        cx, cy = mm.map_width // 2, mm.map_height // 2

        samples = []
        for _ in range(5):
            tile = mm.get_tile(cx, cy)
            target = proj.min_elev if tile.elevation > proj.min_elev else proj.max_elev
            hist = EditHistory()
            hist.begin_stroke(mm.terrain)
            set_tile_elevation(mm, cx, cy, target)
            dirty = hist.commit_stroke("bench", mm.terrain)
            t0 = time.perf_counter()
            refresh_region_iso(img, scenario, dirty, elevations, proj, tile_px)
            samples.append((time.perf_counter() - t0) * 1000)

        best = min(samples)
        lines.append(
            f"  {path.name:38s} {mm.map_width}x{mm.map_height} tile_px={tile_px}  "
            f"best-of-5: {best:.2f}ms  (all: {', '.join(f'{s:.2f}' for s in samples)})"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario_dir", type=Path, nargs="?", default=ROOT / "examples", help="Directory of .aoe2scenario files"
    )
    args = parser.parse_args()

    files = sorted(args.scenario_dir.glob("*.aoe2scenario"))
    if not files:
        raise SystemExit(f"No .aoe2scenario files found in {args.scenario_dir}")

    print(bench_incremental_latency(files))


if __name__ == "__main__":
    main()
