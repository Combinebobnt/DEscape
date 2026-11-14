"""Verifies descape.render_flat.composite_rect_flat() and FlatChunkCache --
v2.6 completion plan's Track B, Phase B-E. Mirrors
tools/verify_iso_chunks.py's seven checks for the Flat counterpart of
IsoChunkCache, plus one Flat-specific check (8) that directly exercises
this phase's own historical hazard.

Same load-bearing bar as the iso checks: a chunk cache is only useful if
it's provably indistinguishable, pixel-for-pixel, from the existing
full-canvas compositor (render_scenario(isometric=False)) -- no "seams are
acceptable" concession, since that full compositor remains a fully
independent implementation (see composite_rect_flat()'s own docstring for
why that duplication is kept on purpose).

Checks:
  1. Stitched chunks == full render, INCLUDING unit stacking order --
     FlatChunkCache.render_rect() over the WHOLE canvas must be
     np.array_equal to an independent render_scenario(isometric=False)
     call. Only meaningful for the order clause if at least one file has a
     real building/unit overlap (a tile more than one unit's draw covers);
     the corpus-tier run asserts that non-vacuously, the fixture-tier run
     (a unit-free map) does not.
  2. Arbitrary non-chunk-aligned rects == full render crop -- check 1 alone
     only ever calls render_rect() at x0=y0=0 with chunk-aligned bounds.
  3. Chunk request order doesn't matter.
  4. patch_rects() after each of a scripted terrain-edit sequence (plus
     undo) re-establishes byte-identity against a fresh full render -- the
     bar ViewerWindow._apply_dirty's Flat branch itself has to clear.
     Unlike the iso version, no elevation ops: Flat has no elevation term
     at all, so an elevation-only edit would trivially produce zero pixel
     difference here.
  5. Eviction round-trips byte-identically.
  6. invalidate_region() round-trips byte-identically.
  7. composite_rect_flat() mutates nothing it reads (unit_draws, terrain
     state).
  8. Flat-specific: a rect containing a real unit/building overlap,
     composited as one call vs. as a 2x2 tiling of sub-rects, matches both
     each other and the full-render crop -- the direct assertion of
     composite_rect_flat()'s own commutation argument, at the exact place
     the historical "building erases a tree" bug (refresh_units_over()'s
     docstring) lived.

Checks 5, 6, and 7 only exercise files[0] even at corpus tier (same
convention tools/verify_iso_chunks.py used) -- they're proving properties of
the shared cache/patch machinery itself, not per-scenario content, so one
representative file is enough. Checks 1, 2, 3, 4, and 8 vary per corpus file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- pytest fixture, imported for its name

from descape.edit_history import EditHistory
from descape.render import render_scenario
from descape.render_cache import FlatChunkCache
from descape.render_flat import _flat_unit_draws, composite_rect_flat
from descape.render_paint import flat_dirty_rects, tile_pixels_for_map
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units

import conftest

RNG_SEED = 20260810


def _pick_different_terrain(current: int) -> int:
    return 15 if current == 2 else 2


def _flat_scripted_ops(mm):
    """Two independent terrain paints, no elevation ops -- see this file's
    own module docstring for why elevation ops would be meaningless here."""
    w, h = mm.map_width, mm.map_height
    tile_a = (min(2, w - 1), 0)
    tile_b = (0, min(2, h - 1))

    def op_paint_a():
        tile = mm.get_tile(*tile_a)
        tile.terrain_id = _pick_different_terrain(tile.terrain_id)
        tile.layer = -1

    def op_paint_b():
        tile = mm.get_tile(*tile_b)
        tile.terrain_id = _pick_different_terrain(tile.terrain_id)
        tile.layer = -1

    return [("terrain paint A", op_paint_a), ("terrain paint B", op_paint_b)]


def _dirty_rects(mm, dirty_indices, tile_px: int, layers) -> list[tuple[int, int, int, int]]:
    """The viewer's own funnel, dilated to the neighbours while blending is effective."""
    coords = [(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices]
    return flat_dirty_rects(coords, mm.map_width, mm.map_height, tile_px, layers)


def _overlapping_tile_count(bboxes: np.ndarray, tile_px: int) -> tuple[int, tuple[int, int] | None]:
    """Number of tiles covered by more than one unit's pixel bbox, and the
    canvas-pixel origin of one such tile (or None) -- the exact condition
    refresh_units_over()'s docstring identifies as where a naive redraw can
    silently erase a unit (a building footprint overlapping a tree)."""
    buckets: dict[tuple[int, int], int] = {}
    first_overlap = None
    for bx0, by0, bx1, by1 in bboxes:
        for tx in range(int(bx0) // tile_px, -(-int(bx1) // tile_px)):
            for ty in range(int(by0) // tile_px, -(-int(by1) // tile_px)):
                key = (tx, ty)
                buckets[key] = buckets.get(key, 0) + 1
                if buckets[key] == 2 and first_overlap is None:
                    first_overlap = (tx * tile_px, ty * tile_px)
    return sum(1 for n in buckets.values() if n > 1), first_overlap


def _check_stitched_matches_full(files: list[Path], require_overlap_evidence: bool) -> tuple[bool, str]:
    problems = []
    total_overlap_tiles = 0
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        cache = FlatChunkCache(scenario, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()

        n_overlap, _ = _overlapping_tile_count(cache.unit_draws[0], tile_px) if cache.unit_draws is not None else (0, None)
        total_overlap_tiles += n_overlap

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_scenario(scenario, isometric=False)
        if stitched.shape != full.shape:
            problems.append(f"{path.name}: shape mismatch, stitched {stitched.shape} vs full {full.shape}")
            continue
        if not np.array_equal(stitched, full):
            diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
            problems.append(f"{path.name}: {diff} pixels differ from a full render")

    if require_overlap_evidence and total_overlap_tiles == 0:
        problems.append(
            "no file in this run had a real unit/building overlap tile -- the order clause above "
            "passed vacuously, not because it's actually correct"
        )

    if problems:
        return False, "; ".join(problems)
    return True, (
        f"OK ({len(files)} files, chunked assembly byte-identical to a full render, "
        f"{total_overlap_tiles} overlapping-unit tiles exercised)"
    )


def _check_order_independence(files: list[Path]) -> tuple[bool, str]:
    rng = np.random.default_rng(RNG_SEED)
    problems = []
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        cache_a = FlatChunkCache(scenario, tile_px)
        cache_b = FlatChunkCache(scenario, tile_px)
        canvas_w, canvas_h = cache_a.canvas_dims()
        n_cx = (canvas_w + cache_a.chunk_px - 1) // cache_a.chunk_px
        n_cy = (canvas_h + cache_a.chunk_px - 1) // cache_a.chunk_px

        keys = [(cx, cy) for cy in range(n_cy) for cx in range(n_cx)]
        keys_shuffled = keys[:]
        rng.shuffle(keys_shuffled)

        for cx, cy in keys:
            cache_a.get_chunk(0, cx, cy)
        for cx, cy in reversed(keys_shuffled):
            cache_b.get_chunk(0, cx, cy)

        rect_a = cache_a.render_rect(0, 0, canvas_w, canvas_h)
        rect_b = cache_b.render_rect(0, 0, canvas_w, canvas_h)
        if not np.array_equal(rect_a, rect_b):
            diff = int(np.count_nonzero(np.any(rect_a != rect_b, axis=2)))
            problems.append(f"{path.name}: {diff} pixels differ between fetch orders")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({len(files)} files, row-major vs shuffled-reversed chunk fetch orders produced identical pixels)"


def _check_patch_after_scripted_ops(files: list[Path]) -> tuple[bool, str]:
    problems = []
    n_checked = 0
    for path in files:
        scenario = load_map_and_units(path)
        if not scenario.terrain_write_supported:
            continue
        n_checked += 1
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        cache = FlatChunkCache(scenario, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()
        cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk so patch() has something to patch

        hist = EditHistory()
        ops = _flat_scripted_ops(mm)

        for label, apply_fn in ops:
            hist.begin_stroke(mm.terrain)
            apply_fn()
            dirty = hist.commit_stroke(label, mm.terrain)
            if not dirty:
                problems.append(f"{path.name} [{label}]: produced no dirty tiles -- expected a real change")
                continue
            cache.patch_rects(_dirty_rects(mm, dirty, tile_px, cache.layers))
            stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
            full = render_scenario(scenario, isometric=False)
            if not np.array_equal(stitched, full):
                diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
                problems.append(f"{path.name} [{label}]: patch_rects() result differs from a full re-composite at {diff} px")

        for _ in ops:
            dirty = hist.undo(mm.terrain)
            if not dirty:
                continue
            cache.patch_rects(_dirty_rects(mm, dirty, tile_px, cache.layers))

        stitched = cache.render_rect(0, 0, canvas_w, canvas_h)
        full = render_scenario(scenario, isometric=False)
        if not np.array_equal(stitched, full):
            diff = int(np.count_nonzero(np.any(stitched != full, axis=2)))
            problems.append(f"{path.name} [undo-to-start]: patch_rects() result differs from a full re-composite at {diff} px")

    if problems:
        return False, "; ".join(problems)
    return True, f"OK ({n_checked} editable files, patch_rects() after every scripted op + full undo matched a fresh full render)"


def _check_eviction_round_trips(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scenario, tile_px, max_chunks=2)
    canvas_w, canvas_h = cache.canvas_dims()
    n_cx = (canvas_w + cache.chunk_px - 1) // cache.chunk_px
    n_cy = (canvas_h + cache.chunk_px - 1) // cache.chunk_px
    if n_cx * n_cy < 3:
        return None, f"{path.name} has only {n_cx * n_cy} chunks at chunk_px={cache.chunk_px}, need >= 3"

    first_key = (0, 0, 0)
    first = cache.get_chunk(*first_key).copy()
    touched = 0
    for cy in range(n_cy):
        for cx in range(n_cx):
            if (cx, cy) == (0, 0):
                continue
            cache.get_chunk(0, cx, cy)
            touched += 1
            if touched >= 3:
                break
        if touched >= 3:
            break
    if first_key in cache._cache:
        return False, f"{path.name}: chunk (0,0,0) was still cached after touching {touched} others at max_chunks=2"

    refetched = cache.get_chunk(*first_key)
    if not np.array_equal(first, refetched):
        diff = int(np.count_nonzero(np.any(first != refetched, axis=2)))
        return False, f"{path.name}: re-fetched evicted chunk differs from its original at {diff} pixels"
    return True, f"OK ({path.name}, chunk (0,0,0) evicted after {touched} others then re-fetched byte-identical)"


def _check_arbitrary_rects_match_full(files: list[Path]) -> tuple[bool, str]:
    rng = np.random.default_rng(RNG_SEED)
    problems = []
    total_rects = 0
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        cache = FlatChunkCache(scenario, tile_px)
        canvas_w, canvas_h = cache.canvas_dims()
        full = render_scenario(scenario, isometric=False)

        for _ in range(30):
            x0 = int(rng.integers(0, canvas_w))
            y0 = int(rng.integers(0, canvas_h))
            rw = int(rng.integers(1, max(2, canvas_w // 3)))
            rh = int(rng.integers(1, max(2, canvas_h // 3)))
            x1, y1 = min(canvas_w, x0 + rw), min(canvas_h, y0 + rh)
            if x1 <= x0 or y1 <= y0:
                continue
            total_rects += 1
            rect = cache.render_rect(x0, y0, x1, y1)
            expected = full[y0:y1, x0:x1]
            if not np.array_equal(rect, expected):
                diff = int(np.count_nonzero(np.any(rect != expected, axis=2)))
                problems.append(f"{path.name} rect=({x0},{y0},{x1},{y1}): {diff} pixels differ")

    if problems:
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        return False, shown + more
    return True, f"OK ({len(files)} files, {total_rects} random non-chunk-aligned rects matched a full render crop)"


def _check_invalidate_region_round_trips(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scenario, tile_px, max_chunks=100000)
    canvas_w, canvas_h = cache.canvas_dims()
    cache.render_rect(0, 0, canvas_w, canvas_h)  # warm every chunk

    bbox = (0, 0, min(cache.chunk_px, canvas_w), min(cache.chunk_px, canvas_h))
    keys_before = set(cache._cache)
    cache.invalidate_region(bbox)
    evicted = keys_before - set(cache._cache)
    if not evicted:
        return False, f"{path.name}: invalidate_region({bbox}) evicted nothing (test doesn't actually exercise it)"

    refetched = cache.render_rect(0, 0, canvas_w, canvas_h)
    full = render_scenario(scenario, isometric=False)
    if not np.array_equal(refetched, full):
        diff = int(np.count_nonzero(np.any(refetched != full, axis=2)))
        return False, f"{path.name}: after invalidate_region + re-fetch, {diff} pixels differ from a full render"
    return True, f"OK ({path.name}, invalidate_region evicted {len(evicted)} chunk(s), re-fetch matched a full render)"


def _check_composite_rect_flat_mutates_nothing(files: list[Path]) -> tuple[bool, str]:
    path = files[0]
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    unit_draws = _flat_unit_draws(scenario, tile_px)
    bboxes_before, colors_before = (a.copy() for a in unit_draws)
    terrain_before = [(t.terrain_id, t.elevation) for t in mm.terrain]

    canvas_w = mm.map_width * tile_px
    canvas_h = mm.map_height * tile_px
    composite_rect_flat(scenario, 0, 0, min(1024, canvas_w), min(1024, canvas_h), tile_px, unit_draws=unit_draws)

    bboxes_after, colors_after = unit_draws
    terrain_after = [(t.terrain_id, t.elevation) for t in mm.terrain]
    if not np.array_equal(bboxes_before, bboxes_after) or not np.array_equal(colors_before, colors_after):
        return False, f"{path.name}: unit_draws changed after composite_rect_flat()"
    if terrain_before != terrain_after:
        n = sum(1 for a, b in zip(terrain_before, terrain_after, strict=True) if a != b)
        return False, f"{path.name}: {n} tiles' terrain/elevation changed after composite_rect_flat()"
    return True, f"OK ({path.name}, unit_draws and terrain state bit-identical before/after composite_rect_flat())"


def _check_unit_overlap_tiling_matches_single_call(files: list[Path]) -> tuple[bool, str]:
    for path in files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        unit_draws = _flat_unit_draws(scenario, tile_px)
        if unit_draws[0].shape[0] == 0:
            continue
        n_overlap, origin = _overlapping_tile_count(unit_draws[0], tile_px)
        if origin is None:
            continue

        cx, cy = origin
        x0, y0 = max(0, cx - 100), max(0, cy - 100)
        x1, y1 = cx + 100, cy + 100
        single = composite_rect_flat(scenario, x0, y0, x1, y1, tile_px, unit_draws=unit_draws)

        mxp, myp = (x0 + x1) // 2, (y0 + y1) // 2
        q1 = composite_rect_flat(scenario, x0, y0, mxp, myp, tile_px, unit_draws=unit_draws)
        q2 = composite_rect_flat(scenario, mxp, y0, x1, myp, tile_px, unit_draws=unit_draws)
        q3 = composite_rect_flat(scenario, x0, myp, mxp, y1, tile_px, unit_draws=unit_draws)
        q4 = composite_rect_flat(scenario, mxp, myp, x1, y1, tile_px, unit_draws=unit_draws)
        tiled = np.zeros_like(single)
        tiled[0 : myp - y0, 0 : mxp - x0] = q1
        tiled[0 : myp - y0, mxp - x0 : x1 - x0] = q2
        tiled[myp - y0 : y1 - y0, 0 : mxp - x0] = q3
        tiled[myp - y0 : y1 - y0, mxp - x0 : x1 - x0] = q4

        full = render_scenario(scenario, isometric=False)
        expected = full[y0:y1, x0:x1]

        if not np.array_equal(tiled, single):
            diff = int(np.count_nonzero(np.any(tiled != single, axis=2)))
            return False, f"{path.name}: 2x2-tiled composite differs from a single-call composite at {diff} px"
        if not np.array_equal(single, expected):
            diff = int(np.count_nonzero(np.any(single != expected, axis=2)))
            return False, f"{path.name}: composite at a real overlap region differs from a full render at {diff} px"
        return True, (
            f"OK ({path.name}, region at ({cx},{cy}) with {n_overlap} overlapping tile(s): "
            f"single-call == 2x2-tiled == full render)"
        )
    return None, "no file in this run has a real unit/building overlap tile to test against"


@pytest.mark.parametrize(
    "check_fn",
    [
        _check_order_independence,
        _check_patch_after_scripted_ops,
        _check_eviction_round_trips,
        _check_arbitrary_rects_match_full,
        _check_invalidate_region_round_trips,
        _check_composite_rect_flat_mutates_nothing,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_check_"),
)
def test_flat_chunks(check_fn) -> None:
    ok, detail = check_fn([FIXTURE_PATH])
    if ok is None:
        pytest.skip(detail)
    assert ok, detail


def test_flat_chunks_stitched_matches_full() -> None:
    ok, detail = _check_stitched_matches_full([FIXTURE_PATH], require_overlap_evidence=False)
    assert ok, detail


def test_flat_chunks_unit_overlap_tiling() -> None:
    """Fixture-tier stand-in: the blank template has no units, so this
    can only ever skip here -- the real exercise of this check is the
    corpus-tier version below. Kept so a reader scanning default-tier
    output sees this check exists at all, not just its corpus sibling."""
    ok, detail = _check_unit_overlap_tiling_matches_single_call([FIXTURE_PATH])
    if ok is None:
        pytest.skip(detail)
    assert ok, detail


@pytest.mark.corpus
@pytest.mark.parametrize(
    "check_fn",
    [
        _check_order_independence,
        _check_patch_after_scripted_ops,
        _check_eviction_round_trips,
        _check_arbitrary_rects_match_full,
        _check_invalidate_region_round_trips,
        _check_composite_rect_flat_mutates_nothing,
        _check_unit_overlap_tiling_matches_single_call,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_check_"),
)
def test_flat_chunks_corpus(check_fn, corpus_files) -> None:
    ok, detail = check_fn(corpus_files)
    if ok is None:
        pytest.skip(detail)
    assert ok, detail


@pytest.mark.corpus
def test_flat_chunks_stitched_matches_full_corpus(corpus_files) -> None:
    ok, detail = _check_stitched_matches_full(corpus_files, require_overlap_evidence=True)
    assert ok, detail


@pytest.mark.test_map
@pytest.mark.parametrize(
    "check_fn",
    [
        pytest.param(fn, id=fn.__name__.removeprefix("_check_"))
        for fn in (
            _check_order_independence,
            _check_patch_after_scripted_ops,
            _check_eviction_round_trips,
            _check_arbitrary_rects_match_full,
            _check_invalidate_region_round_trips,
            _check_composite_rect_flat_mutates_nothing,
            _check_unit_overlap_tiling_matches_single_call,
        )
    ],
)
def test_flat_chunks_on_test_map(check_fn) -> None:
    # The map has units and overlaps, so a None (nothing to check) fails here instead of skipping.
    ok, detail = check_fn([conftest.TEST_MAP_PATH])
    assert ok, detail
    if check_fn is _check_patch_after_scripted_ops:
        assert "(1 editable files" in detail, f"test map assumption: terrain writable ({detail})"


@pytest.mark.test_map
def test_flat_chunks_stitched_matches_full_on_test_map() -> None:
    ok, detail = _check_stitched_matches_full([conftest.TEST_MAP_PATH], require_overlap_evidence=True)
    assert ok, detail


# TASK-031.72: level draws derived from mip-independent row columns.

SPAN1_CONST = 4242  # test_unit_sprites.CONST: no .dat span, so 1x1 and off-centre on both axes


def _ladder(scenario) -> dict[int, int]:
    """{mip: tile_px} of a Flat cache on `scenario`."""
    mm = scenario.map_manager
    probe = FlatChunkCache(scenario, tile_pixels_for_map(mm.map_width, mm.map_height), with_units=False)
    return {mip: probe.mip_tile_px(mip) for mip in probe.mip_levels()}


def _offcentre_scenario(const: int):
    """Span-1 units whose paint offset times each level's tile_px is a half
    (k / (2 * tile_px), k odd, both signs), so the rounding tie is hit at
    every level; plus a Mill (offset 0), a unit clamped at the map edge and
    one off the map."""
    from test_invalidate_units_splice import MILL_CONST, _place, _scenario

    scenario = _scenario()
    for row, tile_px in enumerate(sorted(set(_ladder(scenario).values()))):
        for col, k in enumerate((1, 3, -1, -3, 5)):
            frac = k / (2 * tile_px)
            _place(scenario, 1 + col % 2, const, 10 + 2 * col + 0.5 + frac, 10 + 2 * row + 0.5 - frac)
    _place(scenario, 0, MILL_CONST, 30.0, 30.0)
    _place(scenario, 0, MILL_CONST, 0.5, 0.5)
    _place(scenario, 2, const, -40.0, -40.0)
    return scenario


def test_flat_unit_bboxes_equal_the_scalar_bbox_at_every_mip() -> None:
    from descape import render_flat

    scenario = _offcentre_scenario(SPAN1_CONST)
    tiles, offsets, colors, row_uid, _row_player = render_flat._flat_unit_base_rows(scenario)
    units = {id(u): u for units in scenario.unit_manager.units for u in units}
    for mip, tile_px in _ladder(scenario).items():
        products = offsets * tile_px
        assert (np.abs(products - np.floor(products)) == 0.5).any(), f"mip {mip}: no rounding tie -- vacuous"
        got = render_flat.flat_unit_bboxes(tiles, offsets, tile_px)
        want = [
            render_flat._flat_unit_bbox(tuple(int(v) for v in t), units[uid], tile_px)
            for t, uid in zip(tiles, row_uid.tolist(), strict=True)
        ]
        assert got.dtype == np.int32 and np.array_equal(got, np.array(want, dtype=np.int32)), f"mip {mip}"
        bboxes, walk_colors = _scalar_walk(scenario, tile_px)
        assert np.array_equal(got, bboxes) and np.array_equal(colors, walk_colors), f"mip {mip}"


@pytest.mark.parametrize("sprites", [False, True])
def test_flat_cache_level_draws_equal_the_walk_at_every_mip(sprites, request) -> None:
    from test_invalidate_units_splice import SPRITE_CONST

    if sprites:
        request.getfixturevalue("sprite_install")
    scenario = _offcentre_scenario(SPRITE_CONST)
    mm = scenario.map_manager
    cache = FlatChunkCache(scenario, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=sprites)
    for mip in cache.mip_levels():
        cache._level_icons(mip)  # sprites on: the install path builds the draws first
        bboxes, colors = cache._level_unit_draws(mip)
        want_bboxes, want_colors = _scalar_walk(scenario, cache.mip_tile_px(mip))
        assert np.array_equal(bboxes, want_bboxes) and np.array_equal(colors, want_colors), f"mip {mip}"


@pytest.mark.parametrize("wholesale", ["invalidate_units", "sprites_toggle"])
def test_flat_wholesale_paths_re_walk_the_scenario(wholesale, request) -> None:
    """A wholesale invalidate_units() and the Show sprites toggle re-walk the
    live lists, not the old row table: units appended with no unit_gen bump
    reach every level's draws, and each icon walk agrees with them."""
    from test_invalidate_units_splice import SPRITE_CONST, Unit

    request.getfixturevalue("sprite_install")
    scenario = _offcentre_scenario(SPRITE_CONST)
    mm = scenario.map_manager
    cache = FlatChunkCache(scenario, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True)
    for mip in cache.mip_levels():
        cache._level_icons(mip)
    before = len(cache.unit_draws[0])
    for k in range(3):
        scenario.unit_manager.units[1].append(Unit(40.0 + k, 40.0, SPRITE_CONST, player=1))
    if wholesale == "invalidate_units":
        cache.invalidate_units()
    else:
        cache.set_sprites_enabled(False)
        cache.set_sprites_enabled(True)
    assert len(cache.unit_draws[0]) == before + 3
    for mip in cache.mip_levels():
        cache._level_icons(mip)  # the install's row-count assert fires on a stale row table
        bboxes, colors = cache._level_unit_draws(mip)
        want_bboxes, want_colors = _scalar_walk(scenario, cache.mip_tile_px(mip))
        assert np.array_equal(bboxes, want_bboxes) and np.array_equal(colors, want_colors), f"mip {mip}"


def test_flat_mip0_icons_first_built_inside_a_unit_window(request) -> None:
    """TASK-604: level 0's icon layer first built inside a mutate-now window
    (Eraser, Convert) pairs with live-walk draws, not the lagging row table;
    invalidate_units() drops both, leaving unit_draws the up-to-date rows. A
    level 0 icon layer built before the window keeps the row table's draws."""
    from test_invalidate_units_splice import SPRITE_CONST, _place

    request.getfixturevalue("sprite_install")
    scenario = _offcentre_scenario(SPRITE_CONST)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scenario, tile_px, sprites=True)
    rows = len(cache.unit_draws[0])
    _place(scenario, 1, SPRITE_CONST, 40.0, 40.0)  # the model moves, the cache does not yet
    assert cache._in_unit_window()

    cache._composite_rect(0, 0, 0, 64, 64)  # the paint's order: draws, then icons (the assert)
    want_bboxes, want_colors = _scalar_walk(scenario, cache.mip_tile_px(0))
    bboxes, colors = cache._level_unit_draws(0)
    assert len(bboxes) == rows + 1 and np.array_equal(bboxes, want_bboxes) and np.array_equal(colors, want_colors)
    assert len(cache.unit_draws[0]) == rows, "the row table moved ahead of invalidate_units()"

    cache.invalidate_units()
    assert not cache._in_unit_window() and 0 not in cache._level_icon_layers
    assert cache._level_unit_draws(0) is cache.unit_draws
    assert np.array_equal(cache.unit_draws[0], want_bboxes)
    cache._level_icons(0)

    _place(scenario, 1, SPRITE_CONST, 42.0, 40.0)
    assert cache._in_unit_window()
    assert cache._level_unit_draws(0) is cache.unit_draws, "a resident level 0 left its row table mid-window"
    cache._composite_rect(0, 0, 0, 64, 64)


@pytest.mark.corpus
def test_flat_unit_bboxes_equal_the_walk_on_the_corpus(corpus_files) -> None:
    """Derive equals walk at every level of every corpus map (--corpus-full: all of examples/)."""
    from descape import render_flat

    rows = offcentre = 0
    for path in corpus_files:
        scenario = load_map_and_units(path)
        tiles, offsets, colors, _uid, _player = render_flat._flat_unit_base_rows(scenario)
        rows += len(tiles)
        offcentre += int((offsets != 0).any(axis=1).sum())
        for mip, tile_px in _ladder(scenario).items():
            bboxes, walk_colors = _scalar_walk(scenario, tile_px)
            derived = render_flat.flat_unit_bboxes(tiles, offsets, tile_px)
            assert np.array_equal(derived, bboxes) and np.array_equal(colors, walk_colors), f"{path.name} mip {mip}"
    assert rows and offcentre, f"{rows} rows, {offcentre} off-centre -- vacuous"


def _scalar_walk(scenario, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """The pre-031.72 per-level walk, _flat_unit_bbox() per unit: the oracle
    the derive must equal (_flat_unit_draws() derives now)."""
    from descape import render_flat, render_units

    mm = scenario.map_manager
    bboxes, colors = [], []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            bounds = render_units.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            if bounds is not None:
                bboxes.append(render_flat._flat_unit_bbox(bounds, unit, tile_px))
                colors.append(render_units._unit_color(unit, scenario.player_colors[player_id]))
    return np.array(bboxes, dtype=np.int32).reshape(-1, 4), np.array(colors, dtype=np.uint8).reshape(-1, 3)


def test_top_down_flat_sprites_off_residency() -> None:
    """Every level resident on a fresh sprites-off cache. Inside a mutate-now
    window (the model ahead of the row table), a level first built there is
    not resident, walks the live lists, and becomes resident once
    invalidate_units() drops it; a level built before the window stays."""
    from test_invalidate_units_splice import MILL_CONST, _place

    scenario = _offcentre_scenario(SPAN1_CONST)
    mm = scenario.map_manager
    cache = FlatChunkCache(scenario, tile_pixels_for_map(mm.map_width, mm.map_height), sprites=False)
    mips = cache.mip_levels()
    assert len(mips) >= 3
    assert all(cache.is_level_resident(mip) for mip in mips)
    assert all(cache.paints_without_build(mip) for mip in mips)
    early, late = [m for m in mips if m != 0][:2]
    cache._level_unit_draws(early)

    _place(scenario, 1, MILL_CONST, 40.0, 40.0)  # the model moves, the cache does not yet
    assert cache._in_unit_window()
    assert cache.is_level_resident(0) and cache.is_level_resident(early)
    assert not cache.is_level_resident(late)
    bboxes, _colors = cache._level_unit_draws(late)
    assert np.array_equal(bboxes, _scalar_walk(scenario, cache.mip_tile_px(late))[0]), "not the live walk"
    assert not cache.is_level_resident(late), "a level built inside the window reads resident"

    cache.invalidate_units()
    assert not cache._in_unit_window()
    assert all(cache.is_level_resident(mip) for mip in mips)
    for mip in mips:
        want = _scalar_walk(scenario, cache.mip_tile_px(mip))[0]
        assert np.array_equal(cache._level_unit_draws(mip)[0], want), f"mip {mip}"


def test_top_down_flat_residency_with_sprites_on_or_units_off(request) -> None:
    """Sprites on keeps the icon-layer rule; units off is resident everywhere."""
    from test_invalidate_units_splice import SPRITE_CONST

    request.getfixturevalue("sprite_install")
    scenario = _offcentre_scenario(SPRITE_CONST)
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scenario, tile_px, sprites=True)
    mip = cache.mip_levels()[0]
    assert not cache.is_level_resident(mip) and not cache.paints_without_build(mip)
    cache._level_icons(mip)
    assert cache.is_level_resident(mip) and cache.paints_without_build(mip)

    off = FlatChunkCache(scenario, tile_px, with_units=False, sprites=True)
    assert all(off.is_level_resident(m) and off.paints_without_build(m) for m in off.mip_levels())
