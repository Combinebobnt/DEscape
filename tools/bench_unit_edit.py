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
real QApplication and settings isolation (see tools/bench_fill_latency.py
for the testkit pattern that keeps it off the user's real config.yaml).

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

Two further sections, both with real sprites on (the configured AoE2:DE
install) and mips 0 and 1 resident before each edit:

  - Flat's lazy per-level rebuild. FlatChunkCache.invalidate_units() only
    drops unit_draws and the icon layers; the rebuild lands on the next
    composite, which the phases above never time. Reported per resident
    level (draws, icons), plus the first viewport recomposite after the
    invalidate. The draws+icons sum is the Flat row splice's decision gate.
  - A Convert-shaped batch: reassign CONVERT_K units inside one
    begin/commit_unit_edit, per style, then invalidate_units() either
    wholesale (changed=None) or with one UnitSplice per unit, plus the
    lazy source rebuild each style owes on its resident levels. Three batch
    kinds (_convert_batch()): `alone` (every unit alone on its tiles, no
    walls), `walls` (a quarter are wall/gate consts) and `shared` (a quarter
    share a tile with a unit outside the batch). Each row says whether
    can_splice() accepted the batch, so a fallback row reads as one.
    Stepped/Sloped splice rows add the flush's repaint half: the bbox the
    viewer sizes (tight from sprite_extent_before()/after(), else the reach
    bbox) patched at mip 0, with mip 0 resident over the batch, and the
    share of the flush it takes. --convert-only runs just this section.

--fallback-rate runs a frequency proxy instead, per example file: each
non-GAIA player's units bucketed into WINDOW_W x WINDOW_H tile windows (a
brush-5 drag's one flush), each window checked as one Convert batch against
the real units_by_tile with the guard can_splice() runs. It reports the share
of windows rejected, split by what the window holds (a wall/gate, a shared
tile, both, neither). No sprites and no cache build.

--bbox-only runs just the sprite-bbox plan's Step 0 row instead: a Move of
the largest movable building, patching today's MAX_SPRITE_REACH-padded bbox
against the one tools/_tight_bbox.py sizes from real sprite extents, and, on
a build that has them, the one the caches' own sprite_extent_before()/after()
size (the `lane` row, what the viewer's tight path runs).

--group-move runs the group-move-wall plan's rows instead: a ~115-unit
one-player window of old-allies (one holding a wall and a gate, one holding
neither) group-moved by the stress log's three deltas through
ViewerWindow._patch_unit_edit_cache() itself, borrowed onto a stand-in, so
the rows follow whichever build is on the path. Stepped with mips -2/-1/0
resident and Sloped, sprites on. Per row: the median `unit_sources`,
`unit_patch` and per-level in-op build ms from Perf Trace's own phases, and
the `splice_refused` tokens seen. --counts prints the tokens and build
counts only, no ms, for a no-timing check. DESCAPE_BENCH_ROOT points the
import at another checkout's descape/, so one script drives base and head.

Informational only, matching tools/bench_pick_plane_patch.py's convention:
always runs, never pass/fail. Read the ratio each phase takes of the total,
not the absolute ms -- this machine drifts 15-20% across a run.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The --group-move A/B: this script, another checkout's descape/ (read before the import below).
CODE_ROOT = Path(os.environ.get("DESCAPE_BENCH_ROOT") or ROOT).resolve()
sys.path.insert(0, str(CODE_ROOT))

from descape import render, unit_pick, unit_sprites
from descape.edit_history import EditHistory
from descape.render import elevations_and_proj, sloped_elevations_and_proj, tile_pixels_for_map
from descape.render_cache import DEFAULT_CHUNK_PX, FlatChunkCache, IsoChunkCache, SlopedChunkCache, UnitSplice
from descape.scenario_io import load_map_and_units
from descape.unit_filter import UnitFilter
from descape.unit_model import UnitEditModel

CORPUS_FILE = "F7_2_Dos Pilas (648).aoe2scenario"
REPEATS = 5
# The sprite sections pay an untimed wholesale restore per repeat, so fewer.
SPRITE_REPEATS = 3
CONVERT_K = 20
CONVERT_KINDS = ("alone", "walls", "shared")
# One Convert flush in --fallback-rate: brush 5 wide, ~100ms of drag long.
WINDOW_W, WINDOW_H = 5, 8
RESIDENT_MIPS = (0, 1)
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


def _make_cache(style: str, scenario, chunk_px: int, sprites: bool = False):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        return IsoChunkCache(scenario, elevations, proj, tile_px, chunk_px=chunk_px, sprites=sprites)
    if style == "sloped":
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, chunk_px=chunk_px, sprites=sprites)
    return FlatChunkCache(scenario, tile_px, chunk_px=chunk_px, sprites=sprites)


def _resident_mips(cache) -> tuple[int, ...]:
    return tuple(m for m in RESIDENT_MIPS if m in cache.mip_levels())


def _warm_resident(cache) -> None:
    """Paints a viewport at every resident mip, so each level's sources exist."""
    for mip in _resident_mips(cache):
        cache.render_rect(*_viewport_rect(*cache.canvas_dims(mip)), mip=mip)


def _settle(cache) -> None:
    """Forces the lazy source rebuild each style owes after invalidate_units(),
    without compositing. Sloped rebuilds eagerly inside the call, so nothing."""
    if cache.style == "stepped":
        for mip in _resident_mips(cache):
            cache._level(mip)
    elif cache.style == "flat":
        for mip in _resident_mips(cache):
            cache._level_unit_draws(mip)
            cache._level_icons(mip)


def _footprint(scenario, unit) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    mm = scenario.map_manager
    tiles = render.unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ()
    return (int(unit.x), int(unit.y)), tuple(tiles)


def _flat_rebuild_section(path: Path, chunk_px: int) -> list[str]:
    """Flat, sprites on: what a wholesale invalidate_units() costs once the
    next paint pays for it, per resident level."""
    scenario = load_map_and_units(path)
    cache = _make_cache("flat", scenario, chunk_px, sprites=True)
    _warm_resident(cache)
    _player, unit = _pick_movable_unit(scenario)
    orig_x, orig_y = unit.x, unit.y
    canvas_w, canvas_h = cache.canvas_dims(0)
    viewport = _viewport_rect(canvas_w, canvas_h)
    mips = _resident_mips(cache)
    totals: dict[str, float] = {}

    def _add(key: str, t0: float) -> float:
        now = time.perf_counter()
        totals[key] = totals.get(key, 0.0) + (now - t0) * 1000
        return now

    for repeat in range(SPRITE_REPEATS + 1):
        # Repeat 0 is the untimed warm-up; the edit alternates so each is real.
        unit.x = orig_x + 1 if repeat % 2 == 0 else orig_x
        t = time.perf_counter()
        cache.invalidate_units()
        t = _add("invalidate_units()", t)
        for mip in mips:
            cache._level_unit_draws(mip)
            t = _add(f"mip {mip} draws", t)
            cache._level_icons(mip)
            t = _add(f"mip {mip} icons", t)
        if repeat == 0:
            totals.clear()
    unit.x, unit.y = orig_x, orig_y

    def _first_recomposite():
        cache.invalidate_units()
        cache.invalidate_region((0, 0, canvas_w, canvas_h))
        cache.render_rect(*viewport)

    rows = cache.unit_draws[0].shape[0] if cache.unit_draws is not None else 0
    icons = len(cache._level_icons(0) or {})
    lines = [f"  [flat, sprites on] {rows} rows, {icons} icons at mip 0, resident mips {list(mips)}"]
    for key, total in totals.items():
        lines.append(f"    {key:>28}  {total / SPRITE_REPEATS:7.3f}ms")
    # invalidate_units() rebuilds mip 0's draws eagerly, so it counts too.
    gate = sum(totals.values()) / SPRITE_REPEATS
    lines.append(f"    {'GATE draws+icons, resident':>28}  {gate:7.3f}ms  (Step 2 builds if >= ~33ms)")
    lines.append(
        f"    {'first viewport recomposite':>28}  {_time_ms(_first_recomposite, SPRITE_REPEATS):7.3f}ms"
        "  (invalidate_units + evict + mip 0 viewport)"
    )
    return lines


def _is_wall(unit) -> bool:
    """A wall/connector or rotation-variant const: the consts whose splice has
    always been refused because their shape reads their neighbours."""
    return unit.unit_const in unit_sprites.wall_connector_consts() or unit_sprites.rotation_variant_eligible(
        unit.unit_const
    )


def _convert_batch(scenario, k: int, kind: str = "alone") -> tuple[int, int, list, int]:
    """(source, destination, batch, special): k units of one non-GAIA player,
    a destination player that isn't the source, and how many of the batch are
    `kind`'s special units.

    `alone` is the original picker: the most populous player, every unit alone
    on its tiles and not a wall, so the splice path is actually exercised.
    `walls` and `shared` take ceil(k/4) special units first, from the player
    with the most of them, and fill the rest as `alone` does. A `walls` pick is
    a wall alone on its tiles, a `shared` pick a non-wall sharing a tile with
    a unit outside the batch, so each kind carries one fallback cause only."""
    manager = scenario.unit_manager
    mm = scenario.map_manager
    occupancy: dict[tuple[int, int], list] = {}
    footprints: dict[int, list | None] = {}
    for units in manager.units:
        for u in units:
            tiles = render.unit_occupied_tiles(u, mm.map_width, mm.map_height)
            footprints[id(u)] = tiles
            for tile in tiles or ():
                occupancy.setdefault(tile, []).append(u)

    def alone(u) -> bool:
        tiles = footprints[id(u)]
        return bool(tiles) and all(len(occupancy[t]) == 1 for t in tiles)

    def special(u) -> bool:
        if kind == "walls":
            return _is_wall(u) and alone(u)
        if kind == "shared":
            return not _is_wall(u) and bool(footprints[id(u)]) and not alone(u)
        return False

    players = range(1, len(manager.units))
    if kind == "alone":
        source = max(players, key=lambda p: len(manager.units[p]))
    else:
        source = max(players, key=lambda p: sum(1 for u in manager.units[p] if special(u)))
    destination = next(p for p in players if p != source)
    batch: list = []
    picked: set[int] = set()
    blocked: set[int] = set()
    want = -(-k // 4) if kind != "alone" else 0
    for u in manager.units[source]:
        if len(batch) == want:
            break
        if id(u) in blocked or not special(u):
            continue
        co_occupants = {id(o) for t in footprints[id(u)] for o in occupancy[t] if o is not u}
        if kind == "shared" and co_occupants <= picked:
            continue
        batch.append(u)
        picked.add(id(u))
        # A co-occupant picked later would make the pair share only with each other.
        blocked |= co_occupants
    n_special = len(batch)
    for u in manager.units[source]:
        if len(batch) == k:
            break
        if id(u) in picked or id(u) in blocked or _is_wall(u) or not alone(u):
            continue
        batch.append(u)
    return source, destination, batch, n_special


def _convert_section(path: Path, style: str, chunk_px: int) -> list[str]:
    scenario = load_map_and_units(path)
    cache = _make_cache(style, scenario, chunk_px, sprites=True)
    _warm_resident(cache)
    model = UnitEditModel(scenario)
    history = EditHistory()
    lines: list[str] = []
    for kind in CONVERT_KINDS:
        lines += _convert_kind_rows(scenario, cache, model, history, style, kind)
    return lines


MODEL_KEY = "model: begin + reassign + commit"


def _repaint_bbox(cache, splices, pre):
    """(bbox, label): what ViewerWindow._patch_unit_edit_cache() repaints at
    the visible level (mip 0 here) once invalidate_units() has run: the
    tight bbox from the cache's own sprite extents, or the reach-padded one
    when either extent is REACH_FALLBACK (or the build has no extents)."""
    import _tight_bbox

    from descape import render_cache

    fallback = getattr(render_cache, "REACH_FALLBACK", None)
    if pre is not None and pre is not fallback:
        post = cache.sprite_extent_after(splices, 0)
        if post is not fallback:
            return _tight_bbox.tight_bbox(cache, splices, pre, post), "tight"
    return _tight_bbox.dirty_bbox(cache, splices, with_sprites=True), "reach"


def _convert_kind_rows(scenario, cache, model, history, style: str, kind: str) -> list[str]:
    source, destination, batch, n_special = _convert_batch(scenario, CONVERT_K, kind)
    header = f"  [{style}, sprites on] Convert {kind}"
    if not batch:
        return [f"{header}: no candidate units in this file"]
    dest_list = scenario.unit_manager.units[destination]
    totals: dict[str, float] = {}
    spliced: list[bool] = []
    repaint: dict[str, str] = {}
    # Stepped/Sloped only: Flat repaints per-tile rects, which step 3 leaves alone.
    with_repaint = style != "flat" and hasattr(cache, "sprite_extent_before")
    if with_repaint:
        import _tight_bbox

        # Mip 0 resident over the batch, as if the view sat on the converted town.
        in_place = []
        for u in batch:
            own, tiles = _footprint(scenario, u)
            in_place.append(UnitSplice(source, 0, u, own, own, tiles, tiles))
        reach = _tight_bbox.dirty_bbox(cache, in_place, with_sprites=True)
        if reach is not None:
            cache.render_rect(*reach, mip=0)

    def _run(use_splice: bool) -> None:
        t0 = time.perf_counter()
        model.begin_unit_edit([source, destination])
        splices = []
        for u in batch:
            own, tiles = _footprint(scenario, u)
            model.reassign(u, destination)
            splices.append(
                UnitSplice(destination, len(dest_list) - 1, u, own, own, tiles, tiles, old_player_id=source)
            )
        model.commit_unit_edit("bench convert", history)
        t_model = time.perf_counter() - t0
        if use_splice:
            # Untimed: whether invalidate_units() below splices or falls back.
            spliced.append(cache.can_splice(splices))
        pre = None
        t_pre = 0.0
        if use_splice and with_repaint:
            t = time.perf_counter()
            pre = cache.sprite_extent_before(splices, 0)
            t_pre = time.perf_counter() - t
        t1 = time.perf_counter()
        cache.invalidate_units(splices if use_splice else None)
        t2 = time.perf_counter()
        _settle(cache)
        t3 = time.perf_counter()
        mode = "splice" if use_splice else "wholesale"
        rows = [
            (MODEL_KEY, t_model),
            (f"{mode}: invalidate_units()", t2 - t1),
            (f"{mode}: lazy source rebuild", t3 - t2),
            (f"{mode}: TOTAL cache", t3 - t1),
        ]
        if use_splice and with_repaint:
            bbox, label = _repaint_bbox(cache, splices, pre)
            t4 = time.perf_counter()
            if bbox is not None:
                cache.patch(bbox, elevation_changed=set(), levels=(0,))
            t5 = time.perf_counter()
            sizing = t_pre + (t4 - t3)
            rows += [
                (f"{mode}: repaint bbox sizing", sizing),
                (f"{mode}: repaint patch mip 0", t5 - t4),
                (f"{mode}: TOTAL flush", t3 - t1 + sizing + (t5 - t4)),
            ]
            repaint[label] = _tight_bbox.describe(cache, bbox)
        for key, value in rows:
            totals[key] = totals.get(key, 0.0) + value * 1000
        # Untimed restore: exact list order back, sources rebuilt from it.
        history.undo(scenario.map_manager.terrain, units=model)
        cache.invalidate_units()
        _settle(cache)

    _run(False)  # warm-up
    totals.clear()
    for use_splice in (False, True):
        for _ in range(SPRITE_REPEATS):
            _run(use_splice)
    taken = "splice" if all(spliced) else "FALLBACK" if not any(spliced) else "mixed"
    lines = [(
        f"{header}: {len(batch)} units ({n_special} {kind}), player {source} -> {destination}, "
        f"resident mips {list(_resident_mips(cache))}, splice mode took: {taken}"
    )]
    for key, total in totals.items():
        count = 2 * SPRITE_REPEATS if key == MODEL_KEY else SPRITE_REPEATS
        lines.append(f"    {key:>36}  {total / count:8.3f}ms")
    flush = totals.get("splice: TOTAL flush")
    if flush:
        share = totals["splice: repaint patch mip 0"] / flush
        lines.append(f"    {'repaint share of the flush':>36}  {share:8.1%}  | " + "; ".join(
            f"{label} {desc}" for label, desc in repaint.items()
        ))
    return lines


def _splice_guard(scenario, units_by_tile: dict, unit_filter: UnitFilter):
    """What Stepped/Sloped can_splice() asks, without building a cache: a
    module-level _splice_plan() where this build has one, else
    _batch_splice_eligible(). Feature-detected so the tool runs on either."""
    from descape import render_cache

    plan = getattr(render_cache, "_splice_plan", None)
    if plan is None:
        return lambda changed: render_cache._batch_splice_eligible(units_by_tile, changed)
    return lambda changed: plan(scenario, units_by_tile, unit_filter, changed) is not None


def _fallback_rate(path: Path) -> str:
    """One line: the share of WINDOW_W x WINDOW_H Convert windows the guard
    rejects, split by what each window holds. A proxy for real drags, not a
    recording of one."""
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    unit_filter = UnitFilter()
    units_by_tile = render._units_by_tile(scenario, unit_filter)
    guard = _splice_guard(scenario, units_by_tile, unit_filter)
    windows: dict[tuple[int, int, int], list[UnitSplice]] = {}
    has_wall: set[tuple[int, int, int]] = set()
    has_shared: set[tuple[int, int, int]] = set()
    for player in range(1, len(scenario.unit_manager.units)):
        destination = 1 if player != 1 else 2
        for i, u in enumerate(scenario.unit_manager.units[player]):
            tiles = render.unit_occupied_tiles(u, mm.map_width, mm.map_height)
            if tiles is None:
                continue
            own, tiles = (int(u.x), int(u.y)), tuple(tiles)
            key = (player, own[0] // WINDOW_W, own[1] // WINDOW_H)
            windows.setdefault(key, []).append(
                UnitSplice(destination, i, u, own, own, tiles, tiles, old_player_id=player)
            )
            if _is_wall(u):
                has_wall.add(key)
            if any(len(units_by_tile.get(t, ())) > 1 for t in tiles):
                has_shared.add(key)
    causes = {"wall": 0, "shared": 0, "both": 0, "neither": 0}
    rejected = 0
    for key, changed in windows.items():
        if guard(changed):
            continue
        rejected += 1
        wall, shared = key in has_wall, key in has_shared
        causes["both" if wall and shared else "wall" if wall else "shared" if shared else "neither"] += 1
    n = len(windows) or 1
    split = " ".join(f"{name} {count / n:6.1%}" for name, count in causes.items())
    return (
        f"  {path.name[:40]:<40} windows {len(windows):5d}  rejected {rejected / n:6.1%}  [{split}]"
        f"  | hold a wall {len(has_wall) / n:6.1%}, a shared tile {len(has_shared) / n:6.1%}"
    )


def _big_movable_building(scenario) -> tuple[int, int, object]:
    """(player, index, unit): the non-GAIA unit with the largest span whose
    footprint, and that footprint shifted +1 in x, hold no other unit, and
    whose const splices, so a Move of it takes the splice path."""
    manager = scenario.unit_manager
    mm = scenario.map_manager
    occupancy: dict[tuple[int, int], list] = {}
    for units in manager.units:
        for u in units:
            for tile in render.unit_occupied_tiles(u, mm.map_width, mm.map_height) or ():
                occupancy.setdefault(tile, []).append(u)
    best = None
    for player in range(1, len(manager.units)):
        for index, u in enumerate(manager.units[player]):
            if u.unit_const in unit_sprites.wall_connector_consts():
                continue
            if unit_sprites.rotation_variant_eligible(u.unit_const):
                continue
            tiles = render.unit_occupied_tiles(u, mm.map_width, mm.map_height)
            if not tiles or int(u.x) + 2 >= mm.map_width:
                continue
            shifted = {(x + 1, y) for x, y in tiles}
            if any(o is not u for t in {*tiles, *shifted} for o in occupancy.get(t, ())):
                continue
            span = len(tiles)
            if best is None or span > best[0]:
                best = (span, player, index, u)
    if best is None:
        raise RuntimeError("no movable building found")
    return best[1], best[2], best[3]


def _bbox_section(path: Path, style: str, chunk_px: int) -> list[str]:
    """Sprite-bbox plan Step 0's single-unit row: Move a large building +1
    tile with sprites on, then patch today's reach-padded bbox or the tight
    one (tools/_tight_bbox.py), timing splice + bbox + patch. Mip 0 only is
    resident and visible, the cache-only analogue of a zoomed-in view."""
    import _tight_bbox

    from descape import render_cache

    REACH_FALLBACK = getattr(render_cache, "REACH_FALLBACK", None)
    scenario = load_map_and_units(path)
    cache = _make_cache(style, scenario, chunk_px, sprites=True)
    model = UnitEditModel(scenario)
    history = EditHistory()
    player, index, unit = _big_movable_building(scenario)
    orig_x, orig_y, orig_z = unit.x, unit.y, unit.z
    if style == "stepped":
        cache._level(0)
    # A viewport centred on the building, so the patch lands in resident chunks.
    own, _tiles = _footprint(scenario, unit)
    bx0, by0, bx1, by1 = _tight_bbox.pre_extent(
        cache, [UnitSplice(player, index, unit, own, own, _tiles, _tiles)], [0]
    )
    canvas_w, canvas_h = cache.canvas_dims(0)
    vw, vh = min(VIEWPORT_W, canvas_w), min(VIEWPORT_H, canvas_h)
    vx0 = max(0, min(canvas_w - vw, (bx0 + bx1 - vw) // 2))
    vy0 = max(0, min(canvas_h - vh, (by0 + by1 - vh) // 2))
    viewport = (vx0, vy0, vx0 + vw, vy0 + vh)
    cache.render_rect(*viewport)
    totals: dict[str, float] = {}
    shown: dict[str, str] = {}

    def run(mode: str, dx: float, timed: bool) -> None:
        old_own, old_tiles = _footprint(scenario, unit)
        model.begin_unit_edit([player])
        model.set_position(unit, orig_x + dx, orig_y, orig_z)
        model.commit_unit_edit("bench bbox move", history, push=False)
        new_own, new_tiles = _footprint(scenario, unit)
        changed = [UnitSplice(player, index, unit, old_own, new_own, old_tiles, new_tiles)]
        t0 = time.perf_counter()
        fallback = "-"
        if mode == "tight":
            fallback = _tight_bbox.fallback_reason(cache, changed, 0)
            pre = _tight_bbox.pre_extent(cache, changed, [0])
        elif mode == "lane":
            pre = cache.sprite_extent_before(changed, 0)
        cache.invalidate_units(changed)
        if mode == "tight" and fallback == _tight_bbox.NONE:
            post = _tight_bbox.post_extent(cache, changed, [0])
            bbox = _tight_bbox.tight_bbox(cache, changed, pre, post)
        elif mode == "lane" and pre is not REACH_FALLBACK:
            fallback = "none"
            post = cache.sprite_extent_after(changed, 0)
            bbox = _tight_bbox.tight_bbox(cache, changed, pre, post)
        else:
            fallback = "reach" if mode == "lane" else fallback
            bbox = _tight_bbox.dirty_bbox(cache, changed, with_sprites=True)
        cache.patch(bbox, elevation_changed=set())
        elapsed = (time.perf_counter() - t0) * 1000
        if timed:
            totals[mode] = totals.get(mode, 0.0) + elapsed
        today = _tight_bbox.dirty_bbox(cache, changed, with_sprites=True)
        boxes[mode] = bbox
        shown[mode] = f"fb {fallback} {_tight_bbox.describe(cache, bbox)} sub {_tight_bbox.contains(today, bbox)}"
        cache.render_rect(*viewport)

    # lane: the shipped cache API (sprite_extent_before/after), on a build that has it.
    modes = ("today", "tight", *(("lane",) if hasattr(cache, "sprite_extent_before") else ()))
    boxes: dict[str, tuple] = {}
    for mode in modes:
        run(mode, 1, False)  # warm-up
        run(mode, 0, False)
        for _ in range(SPRITE_REPEATS):
            run(mode, 1, True)
            run(mode, 0, True)
    lines = [f"  [{style}, sprites on] Move const={unit.unit_const} player={player} +1 tile, mip 0 resident"]
    lines.extend(
        f"    {mode:>6} splice+bbox+patch {totals[mode] / (2 * SPRITE_REPEATS):7.2f}ms | {shown[mode]}"
        for mode in modes
    )
    if "lane" in boxes:
        lines.append(f"    lane bbox == tight bbox (last move back): {boxes['lane'] == boxes['tight']}")
    return lines


GROUP_MOVE_FILE = "old-allies-final-v2.aoe2scenario"
GROUP_MOVE_TARGET = 115
# The 2026-09-30 stress log's three Moves of one selection, in order.
GROUP_MOVE_DELTAS = ((-4, -1), (5, -2), (-1, 5))
GROUP_MOVE_MIPS = (-2, -1, 0)
GROUP_MOVE_REPEATS = 5


def _is_connector(unit) -> bool:
    return unit.unit_const in unit_sprites.wall_connector_consts()


def _group_window(scenario, want_walls: bool) -> tuple[int, list[tuple[int, object]]]:
    """(player, [(index, unit), ...]): one non-GAIA player's units in the
    square tile window whose count lands nearest GROUP_MOVE_TARGET, holding a
    rotation-variant wall and a gate when want_walls, else no wall/connector
    const at all. Not always player 1: old-allies' player 1 has 76 units.
    Units any delta would carry off-map are left out."""
    best = None
    for player in range(1, len(scenario.unit_manager.units)):
        found = _player_window(scenario, player, want_walls)
        if found is not None and (best is None or found[0] < best[0]):
            best = (found[0], player, found[1])
    if best is None:
        raise RuntimeError(f"no {'wall' if want_walls else 'wall-free'} window of non-GAIA units")
    return best[1], best[2]


def _player_window(scenario, player: int, want_walls: bool):
    """_group_window() for one player: (score, members) or None."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    offsets, ox, oy = [(0, 0)], 0, 0
    for dx, dy in GROUP_MOVE_DELTAS:
        ox, oy = ox + dx, oy + dy
        offsets.append((ox, oy))
    by_tile: dict[tuple[int, int], list[tuple[int, object]]] = {}
    for i, u in enumerate(scenario.unit_manager.units[player]):
        if not render.unit_occupied_tiles(u, w, h):
            continue
        if not all(0 <= u.x + dx < w and 0 <= u.y + dy < h for dx, dy in offsets):
            continue
        by_tile.setdefault((int(u.x), int(u.y)), []).append((i, u))
    if want_walls:
        centres = [t for t, us in by_tile.items() if any(_is_connector(u) for _i, u in us)]
    else:
        centres = sorted(by_tile)[:: max(1, len(by_tile) // 400)]
    best = None
    for cx, cy in centres:
        members: list[tuple[int, object]] = []
        for r in range(1, 40):
            members = [
                m for x in range(cx - r, cx + r + 1) for y in range(cy - r, cy + r + 1) for m in by_tile.get((x, y), ())
            ]
            if len(members) >= GROUP_MOVE_TARGET:
                break
        walls = sum(1 for _i, u in members if unit_sprites.rotation_variant_eligible(u.unit_const))
        gates = sum(1 for _i, u in members if _is_connector(u) and not unit_sprites.rotation_variant_eligible(u.unit_const))
        if want_walls and not (walls and gates):
            continue
        if not want_walls and any(_is_connector(u) or unit_sprites.rotation_variant_eligible(u.unit_const)
                                  for _i, u in members):
            continue
        score = abs(len(members) - GROUP_MOVE_TARGET)
        if best is None or score < best[0]:
            best = (score, sorted(members, key=lambda m: m[0]))
    return best


def _viewer_standin(cache, scenario, style: str, target):
    """The ViewerWindow attributes _patch_unit_edit_cache() and its helpers
    read, with those methods borrowed from this build's ViewerWindow."""
    from descape.viewer import ViewerWindow

    class _MapView:
        def viewport_chunk_target(self):
            return target

        def invalidate_region(self, _bbox) -> None:
            pass

    class _Standin:
        _patch_unit_edit_cache = ViewerWindow._patch_unit_edit_cache
        _repaint_unit_edit_split = ViewerWindow._repaint_unit_edit_split
        _unit_edit_bbox = ViewerWindow._unit_edit_bbox
        _patch_area_exceeds_viewport = ViewerWindow._patch_area_exceeds_viewport

    standin = _Standin()
    standin._cache, standin.scenario, standin.map_view = cache, scenario, _MapView()
    standin._render_style = standin._terrain_style = style
    standin._iso_elevations, standin._iso_proj = cache.elevations, cache.proj
    return standin


def _level_viewport(cache, mip: int, centre) -> tuple[int, int, int, int]:
    """A VIEWPORT_W x VIEWPORT_H level-`mip` rect around a reference-pixel point."""
    scale = cache.mip_tile_px(mip) / cache.mip_tile_px(0)
    lw, lh = cache.canvas_dims(mip)
    vw, vh = min(VIEWPORT_W, lw), min(VIEWPORT_H, lh)
    x0 = max(0, min(lw - vw, int(centre[0] * scale) - vw // 2))
    y0 = max(0, min(lh - vh, int(centre[1] * scale) - vh // 2))
    return x0, y0, x0 + vw, y0 + vh


def _group_move_rows(path: Path, style: str, chunk_px: int, want_walls: bool, visible_mip: int, counts: bool):
    from descape import iso_geometry, perf_trace

    scenario = load_map_and_units(path)
    cache = _make_cache(style, scenario, chunk_px, sprites=True)
    model = UnitEditModel(scenario)
    history = EditHistory()
    player, members = _group_window(scenario, want_walls)
    mips = [m for m in GROUP_MOVE_MIPS if m in cache.mip_levels()]
    visible = visible_mip if visible_mip in mips else mips[-1]
    xs = [u.x for _i, u in members]
    ys = [u.y for _i, u in members]
    centre = iso_geometry.tile_screen_origin(int(sum(xs) / len(xs)), int(sum(ys) / len(ys)), 0, cache.proj)
    viewports = {m: _level_viewport(cache, m, centre) for m in mips}
    target = (visible, *cache.chunk_index_range(visible, *viewports[visible]))
    standin = _viewer_standin(cache, scenario, style, target)
    samples: dict[str, list[float]] = {}
    tokens: dict[str, int] = {}
    builds: dict[str, int] = {}
    ops = 0

    def warm() -> None:
        # Untimed, between ops: every level resident around the view and current,
        # as LevelWarmer plus the next paint leave them.
        for m in mips:
            cache.render_rect(*viewports[m], mip=m)

    def move(dx: float, dy: float, timed: bool) -> None:
        nonlocal ops
        warm()
        perf_trace.enable(timed)
        try:
            with perf_trace.op("bench-group-move") as op:
                model.begin_unit_edit([player], fields_only=True)
                splices = []
                for i, u in members:
                    old_own, old_tiles = _footprint(scenario, u)
                    model.set_position(u, u.x + dx, u.y + dy, u.z)
                    new_own, new_tiles = _footprint(scenario, u)
                    splices.append(UnitSplice(player, i, u, old_own, new_own, old_tiles, new_tiles))
                model.commit_unit_edit("bench group move", history, push=False)
                standin._patch_unit_edit_cache(splices)
        finally:
            perf_trace.enable(False)
        if not timed:
            return
        ops += 1
        for name, ms in op.phases.items():
            if name.startswith("splice_refused="):
                tokens[name] = tokens.get(name, 0) + 1
            elif name in ("unit_sources", "unit_patch", "sprite_extent", "unit_bbox"):
                samples.setdefault(name, []).append(ms)
        for (kind, mip, _where), agg in op.levels.items():
            key = f"{mip} {kind}"
            builds[key] = builds.get(key, 0) + agg.count
            samples.setdefault(key, []).append(agg.ms)

    net = [sum(d[0] for d in GROUP_MOVE_DELTAS), sum(d[1] for d in GROUP_MOVE_DELTAS)]
    move(*GROUP_MOVE_DELTAS[0], timed=False)  # warm-up, then back
    move(-GROUP_MOVE_DELTAS[0][0], -GROUP_MOVE_DELTAS[0][1], timed=False)
    for _ in range(1 if counts else GROUP_MOVE_REPEATS):
        for dx, dy in GROUP_MOVE_DELTAS:
            move(dx, dy, timed=True)
        move(-net[0], -net[1], timed=False)
    walls = sum(1 for _i, u in members if unit_sprites.rotation_variant_eligible(u.unit_const))
    gates = sum(1 for _i, u in members if _is_connector(u)) - walls
    row = "wall" if want_walls else "no-wall"
    head = (
        f"  [{style}, sprites on] {row}: player {player}, {len(members)} units ({walls} walls, {gates} gates), "
        f"resident mips {mips}, visible {visible}, {ops} ops"
    )
    token_text = ", ".join(f"{k} x{v}" for k, v in sorted(tokens.items())) or "none"
    lines = [head, f"    splice_refused: {token_text}"]
    build_text = ", ".join(f"{k} x{v}" for k, v in sorted(builds.items())) or "none"
    lines.append(f"    in-op level events: {build_text}")
    if counts:
        return lines
    for name in ("unit_sources", "unit_patch", *sorted(k for k in samples if k[0] in "-0123456789")):
        values = samples.get(name)
        if not values:
            continue
        # A level built in only some ops: the median over every op, zeros included.
        padded = values + [0.0] * (ops - len(values))
        lines.append(f"    {name:>28}  median {statistics.median(padded):7.2f}ms  max {max(values):7.2f}ms  n {len(values)}")
    both = [a + b for a, b in zip(samples.get("unit_sources", []), samples.get("unit_patch", []), strict=False)]
    if both:
        lines.append(f"    {'unit_sources + unit_patch':>28}  median {statistics.median(both):7.2f}ms")
    return lines


def group_move_bench(path: Path, chunk_px: int, visible_mip: int, counts: bool) -> None:
    print(f"  {path.name} (code: {CODE_ROOT})", flush=True)
    for style in ("stepped", "sloped"):
        for want_walls in (True, False):
            print("\n".join(_group_move_rows(path, style, chunk_px, want_walls, visible_mip, counts)), flush=True)


def bench(path: Path, chunk_px: int = DEFAULT_CHUNK_PX) -> str:
    lines = [f"  {path.name}"]

    for style in ("stepped", "sloped", "flat"):
        scenario = load_map_and_units(path)
        model = UnitEditModel(scenario)
        player, unit = _pick_movable_unit(scenario)
        cache = _make_cache(style, scenario, chunk_px)

        canvas_w, canvas_h = cache.canvas_dims(0)
        viewport = _viewport_rect(canvas_w, canvas_h)
        lines.append(
            f"  [{style}] canvas {canvas_w}x{canvas_h}, unit player={player} "
            f"const={unit.unit_const} at ({unit.x:.1f}, {unit.y:.1f})"
        )
        lines += _run_phases(cache, scenario, model, player, unit, viewport)
        lines.append("")

    lines += _flat_rebuild_section(path, chunk_px)
    lines.append("")
    for style in ("stepped", "sloped", "flat"):
        lines += _convert_section(path, style, chunk_px)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scenario", type=Path, nargs="?", default=None,
        help=f"A .aoe2scenario file (default examples/{CORPUS_FILE}; with --fallback-rate, every examples/ file)",
    )
    parser.add_argument("--chunk-px", type=int, default=DEFAULT_CHUNK_PX)
    parser.add_argument("--bbox-only", action="store_true", help="Only the sprite-bbox single-unit rows")
    parser.add_argument("--convert-only", action="store_true", help="Only the Convert batch rows, repaint share included")
    parser.add_argument(
        "--fallback-rate", action="store_true", help="Only the Convert window proxy: the share of flushes that fall back"
    )
    parser.add_argument("--group-move", action="store_true", help=f"Only the group-Move rows (default {GROUP_MOVE_FILE})")
    parser.add_argument("--visible-mip", type=int, default=0, help="--group-move: the level the view shows")
    parser.add_argument("--counts", action="store_true", help="--group-move: tokens and build counts only, no ms")
    args = parser.parse_args()

    if args.group_move:
        path = args.scenario or ROOT / "examples" / GROUP_MOVE_FILE
        group_move_bench(path, args.chunk_px, args.visible_mip, args.counts)
        return

    if args.fallback_rate:
        paths = [args.scenario] if args.scenario else sorted((ROOT / "examples").glob("*.aoe2scenario"))
        print(f"  Convert windows {WINDOW_W}x{WINDOW_H} tiles, per non-GAIA player; shares are of all windows")
        for path in paths:
            try:
                print(_fallback_rate(path), flush=True)
            except Exception as exc:
                print(f"  {path.name}: skipped ({type(exc).__name__}: {exc})", flush=True)
        return

    if args.scenario is None:
        args.scenario = ROOT / "examples" / CORPUS_FILE
    if not args.scenario.exists():
        print(f"no such file: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    if args.bbox_only:
        print(f"  {args.scenario.name}")
        for style in ("stepped", "sloped"):
            print("\n".join(_bbox_section(args.scenario, style, args.chunk_px)), flush=True)
        return
    if args.convert_only:
        print(f"  {args.scenario.name}")
        for style in ("stepped", "sloped", "flat"):
            print("\n".join(_convert_section(args.scenario, style, args.chunk_px)), flush=True)
        return
    print(bench(args.scenario, args.chunk_px))


if __name__ == "__main__":
    main()
