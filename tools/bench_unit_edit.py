#!/usr/bin/env python3
"""Times ViewerWindow._after_unit_mutation()'s current whole-canvas sequence
for one `set_position` unit move, per style -- a unit-editing performance
batch's decision gate: whether a per-edit splice of the chunk caches' unit
sources is worth building, or whether the recomposite cost dominates enough
that a bbox-scoped repaint alone already captures the win.

**No ViewerWindow**: drives UnitEditModel and each chunk cache directly,
reproducing _after_unit_mutation()'s sequence (cache.invalidate_units() ->
cache.invalidate_region(whole canvas) -> map_view repaint -> unit_pick.
build_index()) rather than importing viewer.py, whose ViewerWindow needs a
real QApplication and, per tools/bench_fill_latency.py's own on-record
warning, can write the user's real config.yaml and SIGABRT on close.

Phases, timed separately so D4's splice can be judged against each one
rather than the total:

  - _capture x2 (UnitEditModel.begin_unit_edit()/commit_unit_edit()'s own
    before/after snapshot, unit_model.py:788)
  - invalidate_units() -- the wholesale per-style source rebuild
  - canvas eviction (invalidate_region over the whole canvas) and the
    recomposite of just the chunks a real viewport would actually
    re-request (render_rect over a viewport-sized rect, not the whole
    canvas -- the viewport is what MapView actually asks for after a
    repaint, per _after_unit_mutation's own map_view.invalidate_region()
    call feeding a real paint event, not a full-canvas render_rect)
  - _rebuild_unit_index() (unit_pick.build_index(), viewer.py:1805)
  - build_bystander_grid() (Sloped only -- IsoChunkCache/FlatChunkCache
    have no bystander grid)

Informational only, matching tools/bench_pick_plane_patch.py's convention:
always runs, never pass/fail. Read the ratio each phase takes of the total,
not the absolute ms -- this machine drifts 15-20% across a run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from descape import render, unit_pick
from descape.edit_history import EditHistory
from descape.render import elevations_and_proj, sloped_elevations_and_proj, tile_pixels_for_map
from descape.render_cache import DEFAULT_CHUNK_PX, FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units
from descape.unit_model import UnitEditModel

CORPUS_FILE = "F7_2_Dos Pilas (648).aoe2scenario"
REPEATS = 5
# A 1920x1080 viewport is what MapView actually re-requests after a repaint
# (_after_unit_mutation's map_view.invalidate_region() feeds a real paint
# event next, never a full-canvas render_rect) -- see the module docstring.
VIEWPORT_W, VIEWPORT_H = 1920, 1080


def _time_ms(fn, repeats: int = REPEATS) -> float:
    """Mean wall-clock milliseconds per call, one untimed warm-up first."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) * 1000 / repeats


def _viewport_rect(canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
    w, h = min(VIEWPORT_W, canvas_w), min(VIEWPORT_H, canvas_h)
    x0, y0 = (canvas_w - w) // 2, (canvas_h - h) // 2
    return x0, y0, x0 + w, y0 + h


def _pick_movable_unit(scenario):
    """The first unit any player owns -- GAIA (player 0) can carry ids that
    the model refuses to relocate for no such reason, but any concrete unit
    works for a bare set_position timing, so just take the first one found
    scanning players 1..8 first (real buildings/units), falling back to
    GAIA."""
    manager = scenario.unit_manager
    for player in [*range(1, len(manager.units)), 0]:
        units = manager.units[player]
        if units:
            return player, units[0]
    raise RuntimeError("corpus file has no units at all")


def _run_phases(cache, scenario, model: UnitEditModel, player: int, unit, viewport) -> list[str]:
    lines: list[str] = []
    history = EditHistory()
    orig_x, orig_y, orig_z = unit.x, unit.y, unit.z

    def _capture_pair():
        model.begin_unit_edit([player])
        model.set_position(unit, orig_x + 1, orig_y, orig_z)
        model.commit_unit_edit("bench move", history, push=False)
        model.begin_unit_edit([player])
        model.set_position(unit, orig_x, orig_y, orig_z)
        model.commit_unit_edit("bench move back", history, push=False)

    lines.append(f"    {'_capture x2':>28}  {_time_ms(_capture_pair):7.3f}ms")

    # Warm the viewport before timing eviction+recomposite, mirroring a real
    # session that has already painted once.
    cache.render_rect(*viewport)

    canvas_w, canvas_h = cache.canvas_dims(0)

    def _move_and_settle():
        unit.x, unit.y = orig_x + 1, orig_y
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)
        unit_pick.build_index(scenario, cache.unit_filter)
        unit.x, unit.y = orig_x, orig_y
        # Undo left the cache mid-edit-state for the next repeat's warm-up.
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    def _invalidate_units_only():
        unit.x, unit.y = orig_x + 1, orig_y
        cache.invalidate_units()
        unit.x, unit.y = orig_x, orig_y
        cache.invalidate_units()

    lines.append(f"    {'invalidate_units()':>28}  {_time_ms(_invalidate_units_only):7.3f}ms")

    def _evict_and_recomposite():
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    lines.append(f"    {'evict + recomposite visible':>28}  {_time_ms(_evict_and_recomposite):7.3f}ms")

    def _rebuild_index():
        unit_pick.build_index(scenario, cache.unit_filter)

    lines.append(f"    {'_rebuild_unit_index()':>28}  {_time_ms(_rebuild_index):7.3f}ms")

    if cache.style == "sloped":
        def _bystander():
            render.build_bystander_grid(cache.building_bboxes, cache.chunk_px)

        lines.append(f"    {'build_bystander_grid()':>28}  {_time_ms(_bystander):7.3f}ms")

    lines.append(f"    {'--- whole sequence, once ---':>28}")
    lines.append(f"    {'set_position + full sequence':>28}  {_time_ms(_move_and_settle):7.3f}ms")
    return lines


def bench(path: Path, chunk_px: int = DEFAULT_CHUNK_PX) -> str:
    lines = [f"  {path.name}"]

    for style in ("stepped", "sloped", "flat"):
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        model = UnitEditModel(scenario)
        player, unit = _pick_movable_unit(scenario)

        if style == "stepped":
            elevations, proj = elevations_and_proj(scenario)
            cache = IsoChunkCache(scenario, elevations, proj, tile_px, chunk_px=chunk_px)
        elif style == "sloped":
            elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
            cache = SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px)
        else:
            cache = FlatChunkCache(scenario, tile_px, chunk_px=chunk_px)

        canvas_w, canvas_h = cache.canvas_dims(0)
        viewport = _viewport_rect(canvas_w, canvas_h)
        lines.append(
            f"  [{style}] canvas {canvas_w}x{canvas_h}, unit player={player} "
            f"const={unit.unit_const} at ({unit.x:.1f}, {unit.y:.1f})"
        )
        lines += _run_phases(cache, scenario, model, player, unit, viewport)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", type=Path, nargs="?", default=ROOT / "examples" / CORPUS_FILE, help="A .aoe2scenario file"
    )
    parser.add_argument("--chunk-px", type=int, default=DEFAULT_CHUNK_PX)
    args = parser.parse_args()

    if not args.scenario.exists():
        print(f"no such file: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    print(bench(args.scenario, args.chunk_px))


if __name__ == "__main__":
    main()
