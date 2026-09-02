"""Phase B-D-b of the mip-level plan -- real per-level projections,
materialized headlessly. Neither this phase
nor Phase B-D-a (tests/test_mip_cache.py) changes a single pixel reachable
from the running app: descape/viewer.py still only ever calls with the
default mip=0 until Phase B-D-c wires a real LOD signal.

Structured in three layers:
  1. Pure geometry (default tier, milliseconds) -- the exactness matrix
     itself, field-by-field checks, rejection of inexact candidates, and
     the elev_step_pct self-check's raise/degrade behavior.
  2. The load-bearing non-tautology byte-identity check (fixture tier) --
     see test_level_minus_one_stitch_matches_a_forced_half_tile_px_render's
     own docstring for why this is NOT allowed to be written the obvious
     way.
  3. Corpus-marked checks that need real per-file content the unit-free
     120x120 fixture donor can't provide: building_bboxes genuinely
     differing per level (the real gap this whole mip design exists to
     close -- see IsoChunkCache._level() in descape/render_cache.py), patch()
     rebuilding exactly as many levels as are resident (not enumerated),
     patch() re-establishing byte-identity at a non-reference level after
     scripted edits, and invalidate_region()'s high-edge fix for a finer
     level's larger chunk grid.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from descape import iso_geometry, render
from descape.edit_history import EditHistory
from descape.render import (
    dirty_screen_bbox_iso,
    elevations_and_proj,
    render_scenario,
    render_terrain_iso,
    tile_pixels_for_map,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units

# ---------------------------------------------------------------------------
# Layer 1: pure geometry, no scenario load at all.
# ---------------------------------------------------------------------------

_SHAPES = [(480, 480), (120, 120), (200, 144)]
_BASES = [32, 64]

# The plan's own documented matrix, re-measured against canvas_size_and_origin
# itself (see iso_geometry.mip_projections_for's own docstring) -- identical
# across every (shape, base) pair, so parametrized against all of them rather
# than asserted once.
_EXPECTED_MATRIX = {
    10: {32, 64},
    25: {16, 32, 64, 128},
    75: {16, 32, 64, 128},
    50: {8, 16, 32, 64, 128},
    150: {8, 16, 32, 64, 128},
    100: {4, 8, 16, 32, 64, 128},
    200: {4, 8, 16, 32, 64, 128},
}


@pytest.mark.parametrize("w,h", _SHAPES, ids=lambda t: f"{t}" if isinstance(t, int) else None)
@pytest.mark.parametrize("base_tile_px", _BASES)
@pytest.mark.parametrize("pct", sorted(_EXPECTED_MATRIX))
def test_exactness_matrix(w, h, base_tile_px, pct) -> None:
    base = iso_geometry.canvas_size_and_origin(w, h, base_tile_px, 0, 7, elev_step_pct=pct)
    projs = iso_geometry.mip_projections_for(w, h, base, pct)
    got = {p.tile_px for p in projs.values()}
    # MIP_MIN/MAX_TILE_PIXELS = 16/128 clip the doc's raw table (which
    # includes 4 and 8 at some pct values).
    expected = {tp for tp in _EXPECTED_MATRIX[pct] if iso_geometry.MIP_MIN_TILE_PIXELS <= tp <= iso_geometry.MIP_MAX_TILE_PIXELS}
    assert got == expected


def test_pct10_at_base_16_is_a_singleton() -> None:
    """Base-relative, not absolute: the matrix above is all measured at
    base 32/64. At base 16, elev_step_pct=10 has NO exact neighbor at
    all -- a reachable configuration (render.tile_pixels_for_map returns
    16 on a small map at Potato/Potatest quality). mip_levels() == [0] is
    a normal answer here, not a failure this phase forgot to handle."""
    base = iso_geometry.canvas_size_and_origin(480, 480, 16, 0, 7, elev_step_pct=10)
    projs = iso_geometry.mip_projections_for(480, 480, base, 10)
    assert list(projs) == [0]
    assert projs[0] is base


@pytest.mark.parametrize("w,h", _SHAPES)
def test_every_level_is_field_by_field_exact(w, h) -> None:
    """Checks each scaled field individually (so a failure names the
    field, not just 'not exact'), plus min_elev/max_elev equality, plus
    the canvas_dims (skirt-headroom-inclusive) cross-check -- the value
    the blit actually trusts, per render_cache.IsoChunkCache.__init__'s own
    construction-time assert."""
    base = iso_geometry.canvas_size_and_origin(w, h, 32, 0, 7, elev_step_pct=50)
    projs = iso_geometry.mip_projections_for(w, h, base, 50)
    for level, proj in projs.items():
        if level == 0:
            continue
        for f in ("half_w", "half_h", "elev_step", "origin_x", "origin_y", "canvas_w", "canvas_h", "corner_headroom_px"):
            bv, ov = getattr(base, f), getattr(proj, f)
            assert ov * base.tile_px == bv * proj.tile_px, f"level {level} field {f!r}: {ov} vs base {bv}"
        assert proj.min_elev == base.min_elev
        assert proj.max_elev == base.max_elev

        ref_w, ref_h = render._canvas_pixel_dims(base)
        lvl_w, lvl_h = render._canvas_pixel_dims(proj)
        assert lvl_w * base.tile_px == ref_w * proj.tile_px
        assert lvl_h * base.tile_px == ref_h * proj.tile_px


def test_inexact_levels_are_rejected() -> None:
    """Proves the filter isn't returning everything mip_tile_px_candidates
    offers: at elev_step_pct=10, tile_px=16 is a real candidate of base 32
    (per mip_tile_px_candidates) but not an exact mip of it (per the
    matrix above, 10's exact set is {32,64})."""
    base = iso_geometry.canvas_size_and_origin(480, 480, 32, 0, 7, elev_step_pct=10)
    assert 16 in iso_geometry.mip_tile_px_candidates(32).values()
    assert iso_geometry.mip_projection(480, 480, 16, base, elev_step_pct=10) is None


def test_wrong_elev_step_pct_raises_at_the_identity_level() -> None:
    """mip_projections_for's own self-check: passing an elev_step_pct that
    did NOT build `base` must fail loudly, not silently return a wrong
    level set. 25's own exact set doesn't even contain 32 -- see the
    matrix above."""
    base = iso_geometry.canvas_size_and_origin(480, 480, 32, 0, 7, elev_step_pct=50)
    with pytest.raises(AssertionError):
        iso_geometry.mip_projections_for(480, 480, base, elev_step_pct=25)


def test_elev_step_pct_inside_the_rounding_band_degrades_safely() -> None:
    """55 is inside the same round() band as 50 at this half_h -- the
    identity level still reproduces `base` exactly, so this must NOT
    raise, but the resulting set is shallower than 50's own {8..128}."""
    base = iso_geometry.canvas_size_and_origin(480, 480, 32, 0, 7, elev_step_pct=50)
    projs = iso_geometry.mip_projections_for(480, 480, base, elev_step_pct=55)
    assert projs[0] is base
    assert {p.tile_px for p in projs.values()} == {16, 32}


def test_flat_candidates_are_unfiltered_and_not_pct_dependent() -> None:
    """Flat has no projection -- canvas_dims() is a bare multiply, exact
    at every tile_px, no elev_step term to break exactness -- so
    mip_tile_px_candidates() (used unfiltered by FlatChunkCache) needs no
    exactness check and doesn't vary with elev_step_pct at all."""
    candidates = iso_geometry.mip_tile_px_candidates(32)
    assert set(candidates.values()) == {16, 32, 64, 128}
    assert candidates[0] == 32


# ---------------------------------------------------------------------------
# Layer 2: the load-bearing non-tautology byte-identity check.
# ---------------------------------------------------------------------------


def _check_iso_level_matches_forced_render(scenario, level: int, monkeypatch) -> None:
    """The two sides must NEVER share a forced tile_px. The cache is built
    at the REAL reference first (so its levels are enumerated against the
    genuine tile_pixels_for_map() result, and `level` really is a
    non-reference level), and `stitched` is computed BEFORE any
    monkeypatch -- strictly stronger than merely scoping the patch
    narrowly, since it makes it structurally impossible for the patch to
    have influenced the cache path at all. Only AFTER that does
    render.tile_pixels_for_map get patched, scoped to a single fresh
    ground-truth render_terrain_iso() call. Forcing both sides to the same
    tile_px would make the cache's own reference equal `level`, so
    "level `level`" would secretly be level 0 -- passing against an
    implementation that ignores mip entirely. Sound only because levels
    are enumerated at construction (IsoChunkCache.__init__) and
    _composite_rect() uses the level's OWN stored tile_px/proj rather than
    calling tile_pixels_for_map() itself -- both deliberate invariants of
    this design, not incidental.

    The patch MUST be undone before returning, not just left to the
    fixture's own teardown: the corpus caller loops over several files
    sharing one function-scoped monkeypatch, and elevations_and_proj()
    resolves tile_pixels_for_map as a render module global -- so a patch
    surviving into the next iteration silently feeds it the PREVIOUS
    file's forced tile_px while line 179 below still reads this module's
    own import-time binding, tripping the two apart."""
    mm = scenario.map_manager
    ref_tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = elevations_and_proj(scenario)
    assert proj.tile_px == ref_tile_px
    cache = IsoChunkCache(scenario, elevations, proj, ref_tile_px)
    assert level in cache.mip_levels(), f"level {level} not enumerated (levels={cache.mip_levels()})"

    ref_w, ref_h = cache.canvas_dims(0)
    lvl_w, lvl_h = cache.canvas_dims(level)
    assert (lvl_w, lvl_h) != (ref_w, ref_h)  # negative control
    lvl_tile_px = cache.mip_tile_px(level)
    assert lvl_w * ref_tile_px == ref_w * lvl_tile_px
    assert lvl_h * ref_tile_px == ref_h * lvl_tile_px

    stitched = cache.render_rect(0, 0, lvl_w, lvl_h, mip=level)  # BEFORE any monkeypatch

    monkeypatch.setattr(render, "tile_pixels_for_map", lambda w, h: lvl_tile_px)
    truth = render.render_terrain_iso(scenario)
    monkeypatch.undo()
    assert truth.shape[:2] == (lvl_h, lvl_w)
    assert np.array_equal(stitched, truth)


def test_level_minus_one_stitch_matches_a_forced_half_tile_px_render(monkeypatch) -> None:
    scenario = load_map_and_units(FIXTURE_PATH)
    _check_iso_level_matches_forced_render(scenario, -1, monkeypatch)


def test_flat_level_minus_one_matches_a_forced_half_tile_px_render(monkeypatch) -> None:
    """Flat's counterpart. render_terrain() and overlay_units() both
    resolve tile_pixels_for_map as a module global (same as Stepped's
    render_terrain_iso_with_proj), so one monkeypatch covers the whole
    ground-truth path via render_scenario(isometric=False)."""
    scenario = load_map_and_units(FIXTURE_PATH)
    mm = scenario.map_manager
    ref_tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(scenario, ref_tile_px)
    assert -1 in cache.mip_levels()

    ref_w, ref_h = cache.canvas_dims(0)
    lvl_w, lvl_h = cache.canvas_dims(-1)
    assert (lvl_w, lvl_h) == (ref_w // 2, ref_h // 2)

    stitched = cache.render_rect(0, 0, lvl_w, lvl_h, mip=-1)  # BEFORE any monkeypatch

    monkeypatch.setattr(render, "tile_pixels_for_map", lambda w, h: cache.mip_tile_px(-1))
    truth = render_scenario(scenario, isometric=False)
    assert truth.shape[:2] == (lvl_h, lvl_w)
    assert np.array_equal(stitched, truth)


# ---------------------------------------------------------------------------
# Layer 3: corpus-marked -- need real per-file content the fixture can't give.
# ---------------------------------------------------------------------------


@pytest.mark.corpus
def test_level_minus_one_matches_forced_render_corpus(corpus_files, monkeypatch) -> None:
    checked = 0
    for path in corpus_files:
        scenario = load_map_and_units(path)
        if not scenario.map_is_square:
            continue
        checked += 1
        _check_iso_level_matches_forced_render(scenario, -1, monkeypatch)
    if checked == 0:
        pytest.skip("no square corpus file in this run")


@pytest.mark.corpus
def test_building_bboxes_are_genuinely_per_level(corpus_files) -> None:
    """The fixture donor is unit-free (tests/README.md), so at fixture
    tier building_bboxes == {} at EVERY level and a naive test would pass
    vacuously against a broken 'reuse level 0's bboxes at every level'
    implementation -- exactly the gap this whole per-level design exists
    to close (see render_cache.IsoChunkCache._level()'s own docstring). Only a
    real corpus file with actual buildings can exercise this for real."""
    for path in corpus_files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        if -1 not in cache.mip_levels():
            continue
        level0_bboxes = cache._level(0).building_bboxes
        if not level0_bboxes:
            continue
        level_minus1_bboxes = cache._level(-1).building_bboxes
        assert level_minus1_bboxes, f"{path.name}: level -1 building_bboxes empty despite level 0 having {len(level0_bboxes)}"
        assert level_minus1_bboxes != level0_bboxes
        # Spot-check one shared key's bbox for the exact scale relation.
        shared_key = next(iter(set(level0_bboxes) & set(level_minus1_bboxes)), None)
        if shared_key is not None:
            b0 = level0_bboxes[shared_key]
            b1 = level_minus1_bboxes[shared_key]
            t0, t1 = cache.tile_px, cache.mip_tile_px(-1)
            for v0, v1 in zip(b0, b1):
                assert v1 * t0 == v0 * t1
        return
    pytest.skip("no corpus file in this run has both a real building and an enumerated level -1")


@pytest.mark.corpus
def test_patch_rebuilds_only_resident_levels(corpus_files, monkeypatch) -> None:
    """The gen-counter design's own direct test: with only level 0
    resident, one patch() must trigger exactly 1 building_bboxes rebuild;
    after level -1 is warmed too, one patch() must trigger exactly 2,
    NEVER len(cache.mip_levels()) (5, on this fixture's own elev_step_pct).
    On a real 480x480 map this is the difference between ~15-20ms and
    ~75-100ms per edit -- a performance-correctness assertion, not style."""
    for path in corpus_files:
        scenario = load_map_and_units(path)
        if not scenario.terrain_write_supported or not scenario.map_is_square:
            continue
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        if -1 not in cache.mip_levels():
            continue

        calls = []
        real_fn = render._building_bboxes_iso

        def counting(*args, **kwargs):
            calls.append(1)
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(render, "_building_bboxes_iso", counting)

        tile = mm.get_tile(min(2, mm.map_width - 1), 0)

        def toggle_paint():
            tile.terrain_id = 15 if tile.terrain_id != 15 else 2
            tile.layer = -1

        hist = EditHistory()

        # Warm level 0 only, then edit once.
        canvas_w, canvas_h = cache.canvas_dims(0)
        cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
        hist.begin_stroke(mm.terrain)
        toggle_paint()
        dirty = hist.commit_stroke("terrain paint", mm.terrain)
        if not dirty:
            continue
        bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
        if bbox is None:
            continue
        calls.clear()
        cache.patch(bbox)
        assert len(calls) == 1, f"{path.name}: expected 1 rebuild with only level 0 resident, got {len(calls)}"

        # Warm level -1 too, then edit again -- now 2 levels resident.
        lw, lh = cache.canvas_dims(-1)
        cache.render_rect(0, 0, lw, lh, mip=-1)
        hist.begin_stroke(mm.terrain)
        toggle_paint()
        dirty = hist.commit_stroke("terrain paint 2", mm.terrain)
        if not dirty:
            return
        bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
        if bbox is None:
            return
        calls.clear()
        cache.patch(bbox)
        assert len(calls) == 2, f"{path.name}: expected 2 rebuilds with 2 levels resident, got {len(calls)}"
        assert len(cache.mip_levels()) != 2  # sanity: the level COUNT is not coincidentally 2
        return
    pytest.skip("no editable square corpus file in this run enumerates level -1")


@pytest.mark.corpus
def test_patch_at_a_non_reference_level_re_establishes_equality(corpus_files, monkeypatch) -> None:
    verify_iso_chunks = conftest.load_verify_module("verify_iso_chunks")
    checked = 0
    for path in corpus_files:
        scenario = load_map_and_units(path)
        if not scenario.terrain_write_supported or not scenario.map_is_square:
            continue
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        if -1 not in cache.mip_levels():
            continue
        checked += 1

        ref_w, ref_h = cache.canvas_dims(0)
        lvl_w, lvl_h = cache.canvas_dims(-1)
        cache.render_rect(0, 0, ref_w, ref_h, mip=0)
        cache.render_rect(0, 0, lvl_w, lvl_h, mip=-1)

        hist = EditHistory()
        for label, apply_fn in verify_iso_chunks._scripted_ops(mm):
            hist.begin_stroke(mm.terrain)
            apply_fn()
            dirty = hist.commit_stroke(label, mm.terrain)
            if not dirty:
                continue
            bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, proj, with_units=True)
            if bbox is None:
                continue
            cache.patch(bbox)

        stitched0 = cache.render_rect(0, 0, ref_w, ref_h, mip=0)
        truth0 = render_terrain_iso(scenario)
        assert np.array_equal(stitched0, truth0), f"{path.name}: level 0 diverged after patch()"

        stitched_lvl = cache.render_rect(0, 0, lvl_w, lvl_h, mip=-1)
        monkeypatch.setattr(render, "tile_pixels_for_map", lambda w, h: cache.mip_tile_px(-1))
        truth_lvl = render.render_terrain_iso(scenario)
        monkeypatch.undo()
        assert np.array_equal(stitched_lvl, truth_lvl), f"{path.name}: level -1 diverged after patch()"
        return
    if checked == 0:
        pytest.skip("no editable square corpus file in this run enumerates level -1")


@pytest.mark.corpus
def test_invalidate_region_at_the_canvas_high_edge_evicts_finer_levels(corpus_files) -> None:
    """The case the old mip-agnostic chunk-index match got WRONG: a finer
    level's canvas is LARGER, so its chunk grid extends further, and a
    reference-derived index range used unconverted misses its high-edge
    chunks entirely. Warms only a small high-corner rect at a finer level
    (never the whole finer canvas -- 4x the reference's bytes) to keep
    this affordable.

    The warmed set can NOT be asserted evicted wholesale: the warm rect is
    chunk_px-sized and anchored at the canvas corner, but the canvas is
    not generally a whole number of chunks (measured: 2_Joan at level +1
    is 18432x9440 against chunk_px=512, so the rect straddles two chunk
    ROWS), while the probe bbox is one reference pixel and lands in only
    one of them. So compare against the exact set the bbox really covers,
    derived here by plain multiplication rather than through the
    _bbox_to_level() under test -- sound because a finer level's tile_px
    ratio is an exact integer. Asserting set EQUALITY rather than mere
    containment additionally pins that conversion as precise, not merely
    over-broad; over-eviction is harmless per this track's plan doc, so
    treat that half as a tightening, not a load-bearing invariant."""
    for path in corpus_files:
        scenario = load_map_and_units(path)
        mm = scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px)
        finer = 1  # tile_px doubled
        if finer not in cache.mip_levels():
            continue

        ref_w, ref_h = cache.canvas_dims(0)
        lw, lh = cache.canvas_dims(finer)
        ratio = cache.mip_tile_px(finer) // cache.mip_tile_px(0)
        # What makes the expected set below derivable without _bbox_to_level
        # -- and it covers the skirt-headroom term, not just canvas_h.
        assert (ref_w * ratio, ref_h * ratio) == (lw, lh)

        corner_x0, corner_y0 = max(0, lw - cache.chunk_px), max(0, lh - cache.chunk_px)
        before = cache.render_rect(corner_x0, corner_y0, lw, lh, mip=finer)
        corner_keys = {key for key in cache._cache if key[0] == finer}
        assert corner_keys, f"{path.name}: warming the high corner cached nothing at level {finer}"

        bbox = (ref_w - 1, ref_h - 1, ref_w, ref_h)
        chunk_px = cache.chunk_px
        expected = {
            (finer, cx, cy)
            for cx in range(((ref_w - 1) * ratio) // chunk_px, (ref_w * ratio - 1) // chunk_px + 1)
            for cy in range(((ref_h - 1) * ratio) // chunk_px, (ref_h * ratio - 1) // chunk_px + 1)
        }
        assert corner_keys & expected, (
            f"{path.name}: the probe bbox covers {expected}, none of which the warm rect cached "
            f"({corner_keys}) -- the probe no longer tests anything"
        )

        cache.invalidate_region(bbox)
        assert corner_keys & set(cache._cache) == corner_keys - expected, (
            f"{path.name}: invalidate_region at the reference's high edge should have evicted exactly "
            f"{corner_keys & expected} at level {finer}; still cached: {corner_keys & set(cache._cache)}"
        )

        after = cache.render_rect(corner_x0, corner_y0, lw, lh, mip=finer)
        assert np.array_equal(before, after)
        return
    pytest.skip("no corpus file in this run enumerates a finer level (+1)")
