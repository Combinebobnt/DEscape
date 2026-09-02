#!/usr/bin/env python3
"""Times descape.render_cache.IsoChunkCache at CHUNK_PX in (256, 512, 1024) on the
largest real map: cold full-canvas assembly (proxy for panning into unseen
territory) and a warm-cache single-tile edit patch (proxy for live-editing
stutter). Informational only, matching this project's own bench_iso_backend.py/
bench_iso_memory.py convention -- always runs, never pass/fail. The number
this exists to inform is which value DEFAULT_CHUNK_PX should be.

Extracted from tools/verify_iso_chunks.py's bench_chunk_px() (the pytest
migration plan's Ordering step 4) -- it never had a check_ prefix or a
pass/fail verdict, so it belongs here as a benchmark tool, not as a
migrated test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import iso_geometry
from descape.edit_history import EditHistory
from descape.elevation_tools import set_tile_elevation
from descape.render import dirty_screen_bbox_iso, render_terrain_iso_with_proj, tile_pixels_for_map
from descape.render_cache import IsoChunkCache
from descape.scenario_io import load_map_and_units


def _editable(scenario) -> bool:
    return scenario.terrain_write_supported and scenario.map_is_square


def bench_chunk_px(files: list[Path]) -> str:
    largest = max(files, key=lambda p: load_map_and_units(p).map_manager.map_width)
    scenario = load_map_and_units(largest)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    lines = [f"  {largest.name} ({mm.map_width}x{mm.map_height}, tile_px={tile_px}):"]

    for chunk_px in (256, 512, 1024):
        _img, elevations, proj = render_terrain_iso_with_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, chunk_px=chunk_px, max_chunks=100000)
        canvas_w, canvas_h = cache.canvas_dims()

        t0 = time.perf_counter()
        cache.render_rect(0, 0, canvas_w, canvas_h)
        cold_ms = (time.perf_counter() - t0) * 1000

        if _editable(scenario):
            cx, cy = mm.map_width // 2, mm.map_height // 2
            hist = EditHistory()
            hist.begin_stroke(mm.terrain)
            tile = mm.get_tile(cx, cy)
            set_tile_elevation(
                mm, cx, cy, iso_geometry.MIN_ELEVATION if tile.elevation > iso_geometry.MIN_ELEVATION else iso_geometry.MAX_ELEVATION
            )
            dirty = hist.commit_stroke("bench", mm.terrain)
            bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
            t0 = time.perf_counter()
            if bbox is not None:
                cache.patch(bbox)
            patch_ms = (time.perf_counter() - t0) * 1000
        else:
            patch_ms = float("nan")

        n_chunks = ((canvas_w + chunk_px - 1) // chunk_px) * ((canvas_h + chunk_px - 1) // chunk_px)
        lines.append(
            f"    CHUNK_PX={chunk_px:5d}  {n_chunks:4d} chunks  cold full assembly: {cold_ms:7.2f}ms  "
            f"warm single-tile patch: {patch_ms:6.2f}ms"
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

    print(bench_chunk_px(files))


if __name__ == "__main__":
    main()
