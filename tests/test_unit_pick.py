"""Verifies descape.unit_pick -- phase 3's P3-c.

The load-bearing check here is the **ID-plane oracle**. Everything else in
this file is a property check; the oracle is the only thing that can catch a
paint-order mismatch, which is the failure mode that actually matters: a
pick that disagrees with what the compositor drew means clicking a unit
selects a different one, and no amount of "the index looks right" testing
finds that.

The oracle works by re-running the compositor's OWN unit loop -- the same
depth_order() walk, the same per-tile bucket order, the same
per-footprint-tile diamond placement -- but painting a unique per-unit ID
instead of a color. That plane is then, by construction, the ground truth
for "which unit is visible at this pixel", and pick_unit() must agree with
it everywhere.

Documented residual, measured rather than assumed away:
iso_geometry.screen_to_tile() returns None on skirt-face pixels (phase 2's
own accepted behavior), where pick_unit() treats the unit as unoccluded. The
Stepped oracle therefore requires every disagreement to be a pixel where
screen_to_tile() returns None -- it does not simply tolerate mismatches.
"""

from __future__ import annotations


import numpy as np
import pytest

from testkit.fakes import (
    FakeScenario,
    SyntheticTile,
    SyntheticUnit,
)

from descape import iso_geometry, render
from descape.scenario_io import load_map_and_units
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter
from descape.unit_pick import build_index, pick_unit, unit_key, units_in_rect, unit_polygons

RNG_SEED = 20260819
SAMPLE_PIXELS = 4000




_TREE_CONST = min(TREE_UNIT_IDS)
_BUILDING_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == 4 and sy == 4)
_PLAIN_CONST = next(uid for uid in range(1, 10_000) if uid not in TREE_UNIT_IDS and uid not in BUILDING_TILE_SPANS)

MAP_W = MAP_H = 20


def _scenario(elevated: bool = False, ramped: bool = False) -> FakeScenario:
    """A 20x20 map with a deliberately awkward unit population: a 4x4
    building, a second building overlapping its footprint edge, stacked
    units on one exact tile, and scattered trees.

    elevated raises a band of tiles so a building's slab sits at a different
    height from the terrain under part of its own footprint -- the case
    asymmetry 1 exists for, and the one a screen_to_tile-based pick would
    get wrong.

    ramped is the Sloped counterpart, and it is a DIFFERENT shape on
    purpose: a west-to-east ramp climbing one level per tile straight
    through where the units stand, so every tile under them has four corner
    rises that genuinely differ. `elevated`'s band is flat on both sides of
    a single sharp edge, so most of the map's tiles are planar there and the
    per-column resample -- the thing a Sloped pick has to agree with -- is
    barely exercised. Both are run against the sloped oracle: the ramp for
    coverage, the band because its 3-level step deliberately violates this
    project's +-1-elevation-neighbour invariant and so stretches the
    candidate loop's own bound further than legal terrain ever would.
    """
    tiles = []
    for y in range(MAP_H):
        for x in range(MAP_W):
            # Band placed so the 4x4 building at (10.5, 10.5) STRADDLES its
            # edge (footprint x/y 9..12): tile 9 stays at 0 while 10-12 rise.
            # A band that merely contained the whole footprint would make the
            # asymmetry-2 check below vacuous.
            if ramped:
                elevation = max(0, min(4, x - 8))
            else:
                elevation = 3 if (elevated and 10 <= x <= 16 and 10 <= y <= 16) else 0
            tiles.append(SyntheticTile(x=x, y=y, elevation=elevation))

    ref = iter(range(1, 10_000))
    units_by_player = [[] for _ in range(9)]
    units_by_player[GAIA_PLAYER_ID] = [
        SyntheticUnit(x=3.5, y=3.5, unit_const=_TREE_CONST, reference_id=next(ref)),
        SyntheticUnit(x=4.5, y=3.5, unit_const=_TREE_CONST, reference_id=next(ref)),
        SyntheticUnit(x=11.5, y=11.5, unit_const=_TREE_CONST, reference_id=next(ref)),
    ]
    units_by_player[1] = [
        # A 4x4 building straddling the raised band's edge when elevated.
        # Deliberately at a half-tile coordinate, so nothing here may assume
        # the footprint is centred on the own tile -- an even span has no
        # centre tile at all.
        SyntheticUnit(x=10.5, y=10.5, unit_const=_BUILDING_CONST, reference_id=next(ref)),
        # Overlaps the building's own footprint -- later in order, so it
        # must win any pixel the two share.
        SyntheticUnit(x=12.5, y=9.5, unit_const=_BUILDING_CONST, reference_id=next(ref)),
    ]
    units_by_player[2] = [
        # Two units on one exact tile: only `order` disambiguates them.
        SyntheticUnit(x=6.5, y=15.5, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=6.5, y=15.5, unit_const=_PLAIN_CONST, reference_id=next(ref)),
    ]
    units_by_player[4] = [SyntheticUnit(x=17.5, y=6.5, unit_const=_PLAIN_CONST, reference_id=next(ref))]
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


def _tile_px() -> int:
    return render.tile_pixels_for_map(MAP_W, MAP_H)


# --- index ------------------------------------------------------------


def test_index_expands_footprints_not_just_own_tiles() -> None:
    scn = _scenario()
    index = build_index(scn)
    building = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST)
    # A multi-tile building must be reachable from a corner of its footprint,
    # not only from its own tile. The corner is read off unit_tile_bounds
    # rather than written as own - 2: that literal is only the true corner for
    # an odd span on an integer coordinate, and this fixture is neither, so a
    # hardcoded offset would silently assert about a tile well inside the
    # footprint and prove less than it looks like it does.
    x0, x1, y0, y1 = render.unit_tile_bounds(building.unit, MAP_W, MAP_H)
    assert (x1 - x0, y1 - y0) == (4, 4)
    for corner in ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        assert building.order in index.by_tile[corner], corner
    assert (building.own_x, building.own_y) != (x0, y0)


def test_index_order_matches_paint_order_and_is_dense() -> None:
    scn = _scenario()
    index = build_index(scn)
    assert [e.order for e in index.entries] == list(range(len(index.entries)))
    # Player-then-unit-list order, exactly as the renderers iterate.
    assert [e.player_id for e in index.entries] == sorted(e.player_id for e in index.entries)


def test_by_key_addresses_units_stably() -> None:
    scn = _scenario()
    index = build_index(scn)
    for entry in index.entries:
        assert index.entry_for_key(unit_key(entry.player_id, entry.unit)) is entry


def test_filtered_units_are_absent_from_the_index_entirely() -> None:
    scn = _scenario()
    index = build_index(scn, UnitFilter(show_trees=False))
    assert all(e.unit.unit_const != _TREE_CONST for e in index.entries)
    # And therefore unpickable: no tile bucket may still reference one.
    referenced = {o for orders in index.by_tile.values() for o in orders}
    assert all(index.entries[o].unit.unit_const != _TREE_CONST for o in referenced)


def test_off_map_units_are_dropped() -> None:
    scn = _scenario()
    scn.unit_manager.units[1].append(
        SyntheticUnit(x=MAP_W + 5.5, y=2.5, unit_const=_PLAIN_CONST, reference_id=9999)
    )
    index = build_index(scn)
    assert all(e.own_x < MAP_W for e in index.entries)


# --- the ID-plane oracle ----------------------------------------------


def _flat_id_plane(scn, index, tile_px: int) -> np.ndarray:
    """Re-runs composite_rect_flat()'s effective unit loop painting order+1.

    Flat blits opaque rects in ascending index order, so painting in the
    same order and letting later writes win reproduces exactly what's
    visible.
    """
    plane = np.zeros((MAP_H * tile_px, MAP_W * tile_px), dtype=np.int32)
    for entry in index.entries:
        bounds = render.unit_tile_bounds(entry.unit, MAP_W, MAP_H)
        tx0, tx1, ty0, ty1 = bounds
        plane[ty0 * tile_px : ty1 * tile_px, tx0 * tile_px : tx1 * tile_px] = entry.order + 1
    return plane


def _stepped_id_plane(scn, index, tile_px: int, elevations, proj, map_w=None, map_h=None) -> np.ndarray:
    """Re-runs _paint_tile_and_units_iso()'s own loop: for each tile in
    depth_order, clear that tile's terrain diamond (terrain occludes any
    unit painted earlier, at a farther tile), then paint ONE diamond as its
    own ID for each unit whose FOOTPRINT covers this tile, in bucket order.

    The two coordinates are deliberately different, and mixing them up is
    the natural typo: the diamond is painted at THIS tile's screen position,
    but at the UNIT'S OWN tile's elevation -- a multi-tile building is still
    one flat slab at a single height (asymmetry 1), it just no longer paints
    that whole slab at its own tile's moment in the walk.

    Skirts are deliberately not painted. pick_unit() treats a
    screen_to_tile()-None pixel (which is what a skirt pixel is) as
    unoccluded, and the assertion below requires every disagreement to be
    exactly such a pixel -- so leaving skirts out keeps the oracle honest
    rather than papering over that residual.
    """
    map_w = MAP_W if map_w is None else map_w
    map_h = MAP_H if map_h is None else map_h
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h
    plane = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)

    by_footprint_tile: dict[tuple[int, int], list] = {}
    for entry in index.entries:
        tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, map_w, map_h)
        for fy in range(ty0, ty1):
            for fx in range(tx0, tx1):
                by_footprint_tile.setdefault((fx, fy), []).append(entry)

    def paint(base_x, base_y, value):
        yy, xx = base_y + dst_y, base_x + dst_x
        ok = (yy >= 0) & (yy < canvas_h) & (xx >= 0) & (xx < canvas_w)
        plane[yy[ok], xx[ok]] = value

    for x, y in iso_geometry.depth_order(map_w, map_h):
        x, y = int(x), int(y)
        e = int(elevations[y, x])
        bx, by = iso_geometry.tile_screen_origin(x, y, e, proj)
        paint(bx, by, 0)
        for entry in by_footprint_tile.get((x, y), ()):
            ue = int(elevations[entry.own_y, entry.own_x])
            ox, oy = iso_geometry.tile_screen_origin(x, y, ue, proj)
            paint(ox, oy, entry.order + 1)
    return plane


def _sloped_geometry(scn, elev_step_pct: int = 50):
    """(elevations, corner_rise, proj) for a synthetic scenario, at a chosen
    elev_step_pct stop.

    Built here rather than through render.sloped_elevations_and_proj()
    purely so the stop is a parameter: that function reads
    settings.get_elev_step_pct(), and the candidate-loop bound below has to
    be exercised at several stops (the persisted value on a real machine is
    50; the Appearance slider reaches 200, which makes every rise -- and so
    the loop -- four times longer). Everything else matches that function
    exactly, corner_headroom_steps=1 included."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    proj = iso_geometry.canvas_size_and_origin(
        mm.map_width,
        mm.map_height,
        tile_px,
        iso_geometry.MIN_ELEVATION,
        iso_geometry.MAX_ELEVATION,
        elev_step_pct=elev_step_pct,
        corner_headroom_steps=1,
    )
    corner_rise = iso_geometry.corner_rise_px(elevations, proj, rule=render.SLOPE_CORNER_RULE)
    return elevations, corner_rise, proj


def _unit_rise(entry, corner_rise) -> int:
    """The rise render._paint_tile_and_units_sloped places this unit at,
    derived the way the RENDERER derives it (iso_geometry.unit_rise_px on
    the unit's own tile and sub-tile fractions) rather than by calling
    unit_pick's own helper -- an oracle that reused the picker's code could
    not catch the two disagreeing."""
    unit = entry.unit
    return iso_geometry.unit_rise_px(
        corner_rise, entry.own_x, entry.own_y, unit.x - entry.own_x, unit.y - entry.own_y
    )


def _sloped_id_plane(index, tile_px, corner_rise, proj, map_w=None, map_h=None) -> np.ndarray:
    """_stepped_id_plane()'s Sloped twin: re-runs
    _paint_tile_and_units_sloped()'s own loop, painting order+1.

    Two differences from the Stepped oracle, both load-bearing. Terrain
    clears its WARPED quad (sloped_quad_indices, based at
    tile_screen_origin(x, y, 0) - d_min, the placement convention
    _render_tile_sloped documents), not a uniform diamond. And a unit's
    diamond sits at its own PIXEL rise -- one height for the whole
    footprint, painted at each covering tile's own screen position.

    No skirts to leave out, unlike the Stepped oracle: Sloped paints none,
    so this plane has no pixel where the pick is allowed to disagree.
    """
    map_w = MAP_W if map_w is None else map_w
    map_h = MAP_H if map_h is None else map_h
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h
    plane = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    d_dst_y, d_dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)

    by_footprint_tile: dict[tuple[int, int], list] = {}
    for entry in index.entries:
        tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, map_w, map_h)
        for fy in range(ty0, ty1):
            for fx in range(tx0, tx1):
                by_footprint_tile.setdefault((fx, fy), []).append(entry)

    def paint(base_x, base_y, dy, dx, value):
        yy, xx = base_y + dy, base_x + dx
        ok = (yy >= 0) & (yy < canvas_h) & (xx >= 0) & (xx < canvas_w)
        plane[yy[ok], xx[ok]] = value

    for x, y in iso_geometry.depth_order(map_w, map_h):
        x, y = int(x), int(y)
        corners = (
            int(corner_rise[y, x]),
            int(corner_rise[y, x + 1]),
            int(corner_rise[y + 1, x]),
            int(corner_rise[y + 1, x + 1]),
        )
        bx, by = iso_geometry.tile_screen_origin(x, y, 0, proj)
        s_dst_y, s_dst_x, _ssy, _ssx, _uv = iso_geometry.sloped_quad_indices(tile_px, *corners)
        paint(bx, by - min(corners), s_dst_y, s_dst_x, 0)
        for entry in by_footprint_tile.get((x, y), ()):
            paint(bx, by - _unit_rise(entry, corner_rise), d_dst_y, d_dst_x, entry.order + 1)
    return plane


def _sloped_terrain_tiles(scn, corner_rise, proj, tile_px) -> np.ndarray:
    """The whole canvas as one tile-id plane -- what
    render_cache.SlopedChunkCache.pick_tile() resolves a click through, built here
    without a cache so this file stays Qt-free and chunk-free."""
    return render.composite_ids_rect_sloped(
        scn, 0, 0, proj.canvas_w, proj.canvas_h, corner_rise, proj, tile_px
    )


def test_flat_pick_agrees_with_the_id_plane() -> None:
    scn = _scenario()
    index = build_index(scn)
    tile_px = _tile_px()
    plane = _flat_id_plane(scn, index, tile_px)

    rng = np.random.default_rng(RNG_SEED)
    ys = rng.integers(0, plane.shape[0], SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], SAMPLE_PIXELS)
    for sx, sy in zip(xs.tolist(), ys.tolist()):
        got = pick_unit(index, "flat", sx, sy, tile_px, MAP_W, MAP_H)
        expected = int(plane[sy, sx])
        assert (0 if got is None else got.order + 1) == expected, f"flat pick disagreed at ({sx}, {sy})"


@pytest.mark.parametrize("elevated", [False, True], ids=["flat-ground", "raised-band"])
def test_stepped_pick_agrees_with_the_id_plane(elevated: bool) -> None:
    scn = _scenario(elevated=elevated)
    index = build_index(scn)
    tile_px = _tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    plane = _stepped_id_plane(scn, index, tile_px, elevations, proj)

    rng = np.random.default_rng(RNG_SEED)
    ys = rng.integers(0, plane.shape[0], SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], SAMPLE_PIXELS)

    disagreements = 0
    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist()):
        got = pick_unit(index, "stepped", sx, sy, tile_px, MAP_W, MAP_H, elevations, proj)
        expected = int(plane[sy, sx])
        actual = 0 if got is None else got.order + 1
        if expected:
            hits += 1
        if actual == expected:
            continue
        disagreements += 1
        # Not a tolerance: every disagreement must be exactly the documented
        # skirt residual, where screen_to_tile() has no analytic inverse.
        assert iso_geometry.screen_to_tile(sx, sy, elevations, proj) is None, (
            f"stepped pick disagreed at ({sx}, {sy}): expected {expected}, got {actual}, "
            "and this pixel is NOT a skirt-face pixel"
        )
    assert hits > 0, "sampled no unit pixels at all -- the oracle would be vacuous"
    # Recorded so a regression that turns the whole plane into skirt-residual
    # excuses shows up as a number change, not a silent pass.
    assert disagreements < hits, f"{disagreements} disagreements against only {hits} unit pixels"


@pytest.mark.corpus
def test_stepped_pick_agrees_with_the_id_plane_on_a_real_file(scenario_path) -> None:
    """The plan's own corpus-tier requirement: run the oracle against a real
    scenario, not just the synthetic fixture.

    Real files bring what no hand-built fixture does -- thousands of GAIA
    objects stacked on shared tiles, real elevation, and real footprint
    overlap between buildings and the clutter around them.

    Size-gated, and the gate is the honest part: the ID plane is a
    full-canvas int32 array, which on the 480x480 corpus outlier is tens of
    GB. Those files skip with a named reason rather than the whole check
    quietly being written to avoid them. The quick-corpus 144x144 file is
    ~0.17 GB and runs.
    """
    scenario = load_map_and_units(scenario_path)
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tile_px = render.tile_pixels_for_map(w, h)
    elevations, proj = render.elevations_and_proj(scenario)

    plane_bytes = proj.canvas_w * proj.canvas_h * 4
    if plane_bytes > 512 * 1024**2:
        pytest.skip(
            f"{scenario_path.name}: a {proj.canvas_w}x{proj.canvas_h} int32 ID plane is "
            f"{plane_bytes / 1024**3:.1f} GB -- the painted-plane oracle does not scale here"
        )

    index = build_index(scenario)
    if not index.entries:
        pytest.skip(f"{scenario_path.name} has no on-map units to pick")
    plane = _stepped_id_plane(scenario, index, tile_px, elevations, proj, w, h)

    rng = np.random.default_rng(RNG_SEED)
    ys = rng.integers(0, proj.canvas_h, SAMPLE_PIXELS)
    xs = rng.integers(0, proj.canvas_w, SAMPLE_PIXELS)

    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist()):
        expected = int(plane[sy, sx])
        got = pick_unit(index, "stepped", sx, sy, tile_px, w, h, elevations, proj)
        actual = 0 if got is None else got.order + 1
        if expected:
            hits += 1
        if actual == expected:
            continue
        assert iso_geometry.screen_to_tile(sx, sy, elevations, proj) is None, (
            f"{scenario_path.name}: pick disagreed at ({sx}, {sy}): expected {expected}, "
            f"got {actual}, and this pixel is NOT a skirt-face pixel"
        )
    assert hits > 0, f"{scenario_path.name}: sampled no unit pixels -- the oracle would be vacuous"


@pytest.mark.parametrize(
    "kwargs", [{"ramped": True}, {"elevated": True}], ids=["ramp", "three-level-step"]
)
def test_sloped_pick_agrees_with_the_id_plane(kwargs) -> None:
    """Track C5's load-bearing check, and the reason a flat fixture would
    not do: on flat ground every sloped_quad_indices call degenerates to a
    plain diamond and the whole per-column resample this pick has to agree
    with never runs.

    Zero disagreements allowed, unlike the Stepped oracle above: Sloped
    paints no skirts, so it has no pixel where screen_to_tile()'s documented
    residual could excuse one.
    """
    scn = _scenario(**kwargs)
    index = build_index(scn)
    tile_px = _tile_px()
    _elevations, corner_rise, proj = _sloped_geometry(scn)
    plane = _sloped_id_plane(index, tile_px, corner_rise, proj)
    terrain = _sloped_terrain_tiles(scn, corner_rise, proj, tile_px)

    rng = np.random.default_rng(RNG_SEED)
    ys = rng.integers(0, plane.shape[0], SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], SAMPLE_PIXELS)

    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist()):
        tile_id = int(terrain[sy, sx])
        tile = None if tile_id == render.PICK_ID_NONE else (tile_id % MAP_W, tile_id // MAP_W)
        got = pick_unit(
            index,
            "sloped",
            sx,
            sy,
            tile_px,
            MAP_W,
            MAP_H,
            None,
            proj,
            corner_rise=corner_rise,
            terrain_tile=tile,
        )
        expected = int(plane[sy, sx])
        if expected:
            hits += 1
        assert (0 if got is None else got.order + 1) == expected, (
            f"sloped pick disagreed at ({sx}, {sy}): expected {expected}, "
            f"got {0 if got is None else got.order + 1}"
        )
    assert hits > 0, "sampled no unit pixels at all -- the oracle would be vacuous"


@pytest.mark.parametrize("pct", [25, 50, 100, 200])
def test_sloped_pick_finds_every_unit_at_every_elev_step_stop(pct: int) -> None:
    """The candidate loop's own bound, at the stops it actually has to
    survive. Under-enumerating d = y - x returns None on a real unit, which
    reads as "nothing there" rather than as a bug -- so this walks EVERY
    unit rather than sampling, and does it at the extreme stops of the
    Appearance slider (this machine persists 50; the slider reaches 200,
    which makes every rise, and so the loop, four times longer).
    """
    scn = _scenario(ramped=True)
    index = build_index(scn)
    tile_px = _tile_px()
    _elevations, corner_rise, proj = _sloped_geometry(scn, elev_step_pct=pct)
    plane = _sloped_id_plane(index, tile_px, corner_rise, proj)
    terrain = _sloped_terrain_tiles(scn, corner_rise, proj, tile_px)

    for entry in index.entries:
        pixels = np.argwhere(plane == entry.order + 1)
        if pixels.size == 0:
            continue  # fully occluded by later terrain; nothing to pick
        sy, sx = (int(v) for v in pixels[len(pixels) // 2])
        tile_id = int(terrain[sy, sx])
        tile = None if tile_id == render.PICK_ID_NONE else (tile_id % MAP_W, tile_id // MAP_W)
        got = pick_unit(
            index, "sloped", sx, sy, tile_px, MAP_W, MAP_H, None, proj,
            corner_rise=corner_rise, terrain_tile=tile,
        )
        assert got is not None and got.order == entry.order, (
            f"unit order={entry.order} is visible at ({sx}, {sy}) at elev_step_pct={pct}, but the "
            f"pick returned {None if got is None else got.order} -- candidate loop under-enumerated?"
        )


def test_stepped_pick_never_returns_a_filtered_unit() -> None:
    scn = _scenario()
    tile_px = _tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    index = build_index(scn, UnitFilter(show_trees=False))

    rng = np.random.default_rng(RNG_SEED)
    ys = rng.integers(0, proj.canvas_h, 1500)
    xs = rng.integers(0, proj.canvas_w, 1500)
    for sx, sy in zip(xs.tolist(), ys.tolist()):
        got = pick_unit(index, "stepped", sx, sy, tile_px, MAP_W, MAP_H, elevations, proj)
        assert got is None or got.unit.unit_const != _TREE_CONST


def test_sloped_pick_returns_none_without_its_own_height_field() -> None:
    """The inverse of the pin C5 removed. Sloped is pickable now, but it
    reads corner_rise, not elevations -- a caller that hands it Stepped's
    arguments must get None rather than a plausible wrong answer off the
    wrong height field."""
    scn = _scenario()
    index = build_index(scn)
    tile_px = _tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    assert pick_unit(index, "sloped", 10, 10, tile_px, MAP_W, MAP_H, elevations, proj) is None


def test_unknown_style_raises() -> None:
    scn = _scenario()
    index = build_index(scn)
    with pytest.raises(ValueError):
        pick_unit(index, "hexagonal", 0, 0, _tile_px(), MAP_W, MAP_H)


# --- highlight geometry -----------------------------------------------


def test_flat_polygon_matches_the_drawn_rect() -> None:
    scn = _scenario()
    index = build_index(scn)
    tile_px = _tile_px()
    entry = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST)
    polygons = unit_polygons(entry, "flat", tile_px, MAP_W, MAP_H)
    assert len(polygons) == 1
    tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, MAP_W, MAP_H)
    assert polygons[0] == [
        (tx0 * tile_px, ty0 * tile_px),
        (tx1 * tile_px, ty0 * tile_px),
        (tx1 * tile_px, ty1 * tile_px),
        (tx0 * tile_px, ty1 * tile_px),
    ]


def test_stepped_polygons_sit_at_the_units_own_elevation_not_each_tiles() -> None:
    """The asymmetry-2 regression. On a map where the building's footprint
    spans tiles at two different terrain heights, every one of its diamonds
    must sit at the unit's OWN tile's elevation -- matching _draw_unit_iso's
    flat slab. Using each footprint tile's own elevation would look correct
    on flat ground and silently drift apart here.
    """
    scn = _scenario(elevated=True)
    index = build_index(scn)
    tile_px = _tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    entry = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST)

    tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, MAP_W, MAP_H)
    footprint_elevs = {int(elevations[fy, fx]) for fy in range(ty0, ty1) for fx in range(tx0, tx1)}
    assert len(footprint_elevs) > 1, "fixture must span two terrain heights or this proves nothing"

    own_elev = int(elevations[entry.own_y, entry.own_x])
    polygons = unit_polygons(entry, "stepped", tile_px, MAP_W, MAP_H, elevations, proj)
    expected = []
    for fy in range(ty0, ty1):
        for fx in range(tx0, tx1):
            ox, oy = iso_geometry.tile_screen_origin(fx, fy, own_elev, proj)
            expected.append(
                [
                    (ox + proj.half_w, oy),
                    (ox + 2 * proj.half_w, oy + proj.half_h),
                    (ox + proj.half_w, oy + 2 * proj.half_h),
                    (ox, oy + proj.half_h),
                ]
            )
    assert polygons == expected


def test_sloped_polygons_return_none_without_corner_rise() -> None:
    scn = _scenario()
    index = build_index(scn)
    entry = index.entries[0]
    assert unit_polygons(entry, "sloped", _tile_px(), MAP_W, MAP_H) is None


def test_sloped_polygons_sit_at_the_units_own_rise_not_each_tiles() -> None:
    """Asymmetry 2 in Sloped's own terms: every diamond of a multi-tile
    building must sit at the unit's own interpolated PIXEL rise, matching
    render._draw_unit_sloped's flat pad -- not at each footprint tile's own
    surface, which would look right on flat ground and drift apart on the
    ramp this fixture puts the building on.
    """
    scn = _scenario(ramped=True)
    index = build_index(scn)
    tile_px = _tile_px()
    _elevations, corner_rise, proj = _sloped_geometry(scn)
    entry = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST)

    tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, MAP_W, MAP_H)
    spread = corner_rise[ty0 : ty1 + 1, tx0 : tx1 + 1]
    assert int(spread.max() - spread.min()) > 0, "fixture went flat -- per-tile heights would agree"

    rise = _unit_rise(entry, corner_rise)
    polygons = unit_polygons(
        entry, "sloped", tile_px, MAP_W, MAP_H, None, proj, corner_rise=corner_rise
    )
    expected = []
    for fy in range(ty0, ty1):
        for fx in range(tx0, tx1):
            ox, oy = iso_geometry.tile_screen_origin(fx, fy, 0, proj)
            expected.append(
                [
                    (ox + proj.half_w, oy - rise),
                    (ox + 2 * proj.half_w, oy - rise + proj.half_h),
                    (ox + proj.half_w, oy - rise + 2 * proj.half_h),
                    (ox, oy - rise + proj.half_h),
                ]
            )
    assert polygons == expected


# --- units_in_rect (b2.1) -----------------------------------------------


def test_units_in_rect_covers_a_small_area() -> None:
    scn = _scenario()
    index = build_index(scn)
    # The two GAIA trees at (3.5, 3.5) and (4.5, 3.5) sit on tiles (3, 3) and
    # (4, 3) -- a rect covering exactly those two tiles must return exactly
    # those two units and nothing from the rest of the fixture's population.
    found = units_in_rect(index, 3, 3, 5, 4)
    assert {(e.player_id, e.unit.reference_id) for e in found} == {
        (GAIA_PLAYER_ID, scn.unit_manager.units[GAIA_PLAYER_ID][0].reference_id),
        (GAIA_PLAYER_ID, scn.unit_manager.units[GAIA_PLAYER_ID][1].reference_id),
    }


def test_units_in_rect_dedupes_a_multi_tile_footprint() -> None:
    scn = _scenario()
    index = build_index(scn)
    building = next(e for e in index.entries if e.unit.unit_const == _BUILDING_CONST and e.player_id == 1)
    tx0, tx1, ty0, ty1 = render.unit_tile_bounds(building.unit, MAP_W, MAP_H)
    assert (tx1 - tx0, ty1 - ty0) == (4, 4)
    # A rect covering the whole 4x4 footprint must return the building
    # exactly once, not once per covered tile.
    found = units_in_rect(index, tx0, ty0, tx1, ty1)
    assert found.count(building) == 1


def test_units_in_rect_is_empty_outside_every_footprint() -> None:
    scn = _scenario()
    index = build_index(scn)
    assert units_in_rect(index, 0, 0, 1, 1) == []


def test_units_in_rect_respects_the_filter() -> None:
    """Filtered units are absent from build_index() entirely (settled fact
    4/hard rule in the plan: honouring the filter means querying the index,
    not the scenario), so a rect over a hidden tree's tile must not surface
    it -- there is no separate visibility flag to check here."""
    scn = _scenario()
    index = build_index(scn, UnitFilter(show_trees=False))
    found = units_in_rect(index, 3, 3, 5, 4)
    assert found == []
