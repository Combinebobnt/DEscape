"""Stage 1 of free (non-snapped) unit placement: a unit's sub-tile position
must actually reach the screen.

Until 2026-09-17 it did not, in any render style. `render._span_start(coord, 1)`
returns `int(coord)`, so a 1x1 unit's sprite anchored at `int(x) + 0.5` whatever
`x` was: 35.2, 35.5, 35.9 and 35.0 all produced the same pixel. That made the
inspector's 0.01 X/Y spinboxes and the 0.1 arrow nudge write coordinates the
render never showed moving, and drew the ~9.2% of real corpus units that sit
off-centre up to half a tile from where their file says they stand.

Pixel-level on purpose, per the plan: a model-only assertion catches none of
this, which is b1's own recorded postmortem.

Two things this module pins that are easy to regress in opposite directions:

- **A span-1 axis is continuous.** Different fractions, different pixels.
- **A span > 1 axis is still half-tile-quantized.** Buildings are drawn as
  slabs deliberately (`_span_start`'s own measured branch), and 88.0% of corpus
  units sit at exactly `(0.5, 0.5)`, so the centred case must stay byte-identical
  or the flat-map byte-identity oracle moves under everything else.

Synthetic bytes and a tmp install throughout -- no game assets, matching the
suite's standing posture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, iso_geometry, render, unit_sprites
from descape.terrain_palette import PLAYER_COLORS

from test_unit_sprites import CONST, FILE_NAME, build_sld

MAP_W = MAP_H = 12


@dataclass
class Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player, *, elevation=0):
    tiles = [Tile(x, y, elevation) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _only_draw(scn):
    """The single (piece, px, py) a one-unit scenario's sprite layer holds."""
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
    sprites = render.sprite_draws_by_anchor(scn, proj, elevations)
    draws = [d for v in sprites.by_anchor.values() for d in v]
    assert len(draws) == 1, f"fixture resolved {len(draws)} draws, not 1"
    return draws[0]


def test_a_sub_tile_offset_moves_a_span_1_sprite(sprite_install):
    _, x_lo, y_lo = _only_draw(_scenario([[], [Unit(5.2, 5.8, CONST)]]))
    _, x_mid, y_mid = _only_draw(_scenario([[], [Unit(5.5, 5.5, CONST)]]))
    _, x_hi, y_hi = _only_draw(_scenario([[], [Unit(5.9, 5.1, CONST)]]))

    assert (x_lo, y_lo) != (x_mid, y_mid)
    assert (x_hi, y_hi) != (x_mid, y_mid)
    assert (x_lo, y_lo) != (x_hi, y_hi)
    # x = origin + (mx + my) * half_w, so the two off-centre points here share
    # a screen column (5.2 + 5.8 == 5.9 + 5.1) and differ only in row. Pinning
    # the axis split, not just "something moved".
    assert x_lo == x_hi
    assert y_lo != y_hi


def test_the_centred_case_is_unchanged(sprite_install):
    """88.0% of corpus units sit at exactly (0.5, 0.5). Every one of them must
    still anchor where it did before Stage 1, or the flat-map byte-identity
    oracle re-baselines for no reason."""
    _, px, py = _only_draw(_scenario([[], [Unit(5.5, 7.5, CONST)]]))
    _, elevations, proj = render.render_terrain_iso_with_proj(
        _scenario([[], [Unit(5.5, 7.5, CONST)]]), with_sprites=True
    )
    # The pre-Stage-1 expression, written out: the span-1 quantized centre.
    rise = int(elevations[7, 5]) * proj.elev_step
    want = iso_geometry.map_point_to_screen(
        render._span_start(5.5, 1) + 0.5, render._span_start(7.5, 1) + 0.5, rise, proj
    )
    assert (px, py) == want


def test_a_building_axis_stays_half_tile_quantized(sprite_install, monkeypatch):
    """span > 1 keeps `_span_start(coord, span) + span / 2`. Buildings are
    drawn as half-tile slabs on purpose, and this is the branch with 27,320
    measured corpus values behind it."""
    monkeypatch.setitem(render.BUILDING_TILE_SPANS, CONST, (3, 3))
    _, px, py = _only_draw(_scenario([[], [Unit(5.5, 5.5, CONST)]]))
    _, elevations, proj = render.render_terrain_iso_with_proj(
        _scenario([[], [Unit(5.5, 5.5, CONST)]]), with_sprites=True
    )
    rise = int(elevations[5, 5]) * proj.elev_step
    want = iso_geometry.map_point_to_screen(
        render._span_start(5.5, 3) + 1.5, render._span_start(5.5, 3) + 1.5, rise, proj
    )
    assert (px, py) == want


def test_sprite_axis_centre_splits_per_axis_not_per_unit():
    """A gate segment is (4, 1) and takes both branches at once."""
    assert render._sprite_axis_centre(10.2, 1) == 10.2
    assert render._sprite_axis_centre(10.0, 4) == render._span_start(10.0, 4) + 2.0
    # The whole point of the span-1 branch: these used to collapse together.
    assert render._sprite_axis_centre(35.2, 1) != render._sprite_axis_centre(35.9, 1)


def test_map_point_to_screen_is_monotone_and_keeps_the_half_h_shift():
    proj = render.render_terrain_iso_with_proj(_scenario([[]]))[2]
    half_w, half_h = proj.half_w, proj.half_h

    # The trailing + half_h, isolated: at the origin tile's own centre the
    # point must sit half a tile BELOW tile_screen_origin's top-left corner.
    ox, oy = iso_geometry.tile_screen_origin(0, 0, 0, proj)
    sx, sy = iso_geometry.map_point_to_screen(0.5, 0.5, 0, proj)
    assert (sx, sy) == (ox + half_w, oy + half_h)

    # Monotone in each axis independently -- the property _unit_screen_bbox_iso
    # takes its four corner extremes from.
    base = iso_geometry.map_point_to_screen(4.0, 4.0, 0, proj)
    assert iso_geometry.map_point_to_screen(4.5, 4.0, 0, proj)[0] > base[0]
    assert iso_geometry.map_point_to_screen(4.0, 4.5, 0, proj)[0] > base[0]
    assert iso_geometry.map_point_to_screen(4.0, 4.5, 0, proj)[1] > base[1]
    assert iso_geometry.map_point_to_screen(4.5, 4.0, 0, proj)[1] < base[1]


def test_off_centre_units_paint_different_pixels(sprite_install):
    """The end-to-end form of the first test: two renders of the same unit at
    two sub-tile positions inside ONE tile must differ on the canvas, not just
    in the draw list."""
    a, _, _ = render.render_terrain_iso_with_proj(
        _scenario([[], [Unit(5.15, 5.85, CONST)]]), with_sprites=True
    )
    b, _, _ = render.render_terrain_iso_with_proj(
        _scenario([[], [Unit(5.5, 5.5, CONST)]]), with_sprites=True
    )
    assert not np.array_equal(a, b)


# --- the off-centre ID-plane oracle -----------------------------------
#
# tests/test_unit_pick.py's oracle is the right shape for this, but its
# fixture places every unit at an exact `.5`, so free placement makes no
# difference to it and it proves nothing here. Its pinned stack counts make
# widening that fixture in place the expensive option, so this is a second,
# deliberately off-centre population run through the same idea: re-run the
# compositor's own unit loop painting IDs, then require the picker to agree
# with it pixel for pixel.

from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS  # noqa: E402
from descape.unit_pick import build_index, pick_unit, unit_polygons  # noqa: E402
from testkit.fakes import FakeScenario, SyntheticTile, SyntheticUnit  # noqa: E402

OC_MAP_W = OC_MAP_H = 20
OC_SAMPLE_PIXELS = 6000
OC_SEED = 20260917

_BUILDING_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == 4 and sy == 4)
_PLAIN_CONST = next(
    uid for uid in range(1, 10_000) if uid not in TREE_UNIT_IDS and uid not in BUILDING_TILE_SPANS
)


def _offcentre_scenario(elevated: bool = False) -> FakeScenario:
    """Units at fractions chosen to straddle a tile boundary in every
    direction, plus the two cases that must NOT move.

    The extremes are the point: 0.02 and 0.98 put a mark almost exactly half
    a tile off its own tile, which is where an under-enumerating pick loop
    starts returning None. An exact 0.5 and a 4x4 building are in the same
    population so the same oracle run proves the unmoved cases too."""
    tiles = [
        SyntheticTile(x=x, y=y, elevation=3 if (elevated and 10 <= x <= 16 and 10 <= y <= 16) else 0)
        for y in range(OC_MAP_H)
        for x in range(OC_MAP_W)
    ]
    ref = iter(range(1, 10_000))
    units_by_player = [[] for _ in range(9)]
    units_by_player[1] = [
        SyntheticUnit(x=4.02, y=4.98, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=6.98, y=6.02, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=8.15, y=8.85, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=14.98, y=14.98, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        # Must not move: exactly centred, and a building on a span > 1 axis.
        SyntheticUnit(x=17.5, y=3.5, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=11.5, y=11.5, unit_const=_BUILDING_CONST, reference_id=next(ref)),
    ]
    units_by_player[2] = [
        # Two on one tile at different sub-tile points -- the case Stage 1
        # visually separates. Only `order` used to tell them apart.
        SyntheticUnit(x=6.2, y=15.2, unit_const=_PLAIN_CONST, reference_id=next(ref)),
        SyntheticUnit(x=6.8, y=15.8, unit_const=_PLAIN_CONST, reference_id=next(ref)),
    ]
    return FakeScenario(OC_MAP_W, OC_MAP_H, tiles, units_by_player)


def _oc_tile_px() -> int:
    return render.tile_pixels_for_map(OC_MAP_W, OC_MAP_H)


def _oc_flat_id_plane(index, tile_px: int) -> np.ndarray:
    """_flat_unit_draws()'s rects, in ascending order, later writes winning --
    with the free-placement shift applied HERE rather than read back out of
    render, so the oracle can catch the painter and the picker agreeing on a
    wrong shift."""
    plane = np.zeros((OC_MAP_H * tile_px, OC_MAP_W * tile_px), dtype=np.int32)
    for entry in index.entries:
        tx0, tx1, ty0, ty1 = render.unit_tile_bounds(entry.unit, OC_MAP_W, OC_MAP_H)
        span_x, span_y = render.tile_span(entry.unit.unit_const, render.NON_BUILDING_SPAN)
        dx = entry.unit.x - int(entry.unit.x) - 0.5 if span_x <= 1 else 0.0
        dy = entry.unit.y - int(entry.unit.y) - 0.5 if span_y <= 1 else 0.0
        ox, oy = round(dx * tile_px), round(dy * tile_px)
        y0, y1 = max(0, ty0 * tile_px + oy), max(0, ty1 * tile_px + oy)
        x0, x1 = max(0, tx0 * tile_px + ox), max(0, tx1 * tile_px + ox)
        plane[y0:y1, x0:x1] = entry.order + 1
    return plane


def _oc_stepped_id_plane(index, tile_px: int, elevations, proj) -> np.ndarray:
    """_paint_tile_and_units_iso()'s loop: clear each tile's terrain diamond in
    depth order, then paint one shifted diamond per covering unit."""
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h
    plane = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    dst_y, dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)

    by_footprint_tile: dict[tuple[int, int], list] = {}
    for entry in index.entries:
        for fx, fy in render.unit_occupied_tiles(entry.unit, OC_MAP_W, OC_MAP_H):
            by_footprint_tile.setdefault((fx, fy), []).append(entry)

    def paint(base_x, base_y, value):
        yy, xx = base_y + dst_y, base_x + dst_x
        ok = (yy >= 0) & (yy < canvas_h) & (xx >= 0) & (xx < canvas_w)
        plane[yy[ok], xx[ok]] = value

    for raw_x, raw_y in iso_geometry.depth_order(OC_MAP_W, OC_MAP_H):
        x, y = int(raw_x), int(raw_y)
        e = int(elevations[y, x])
        bx, by = iso_geometry.tile_screen_origin(x, y, e, proj)
        paint(bx, by, 0)
        for entry in by_footprint_tile.get((x, y), ()):
            ue = int(elevations[entry.own_y, entry.own_x])
            span_x, span_y = render.tile_span(entry.unit.unit_const, render.NON_BUILDING_SPAN)
            dx = entry.unit.x - int(entry.unit.x) - 0.5 if span_x <= 1 else 0.0
            dy = entry.unit.y - int(entry.unit.y) - 0.5 if span_y <= 1 else 0.0
            cx, cy = iso_geometry.map_point_to_screen(
                x + 0.5 + dx, y + 0.5 + dy, ue * proj.elev_step, proj
            )
            paint(cx - proj.half_w, cy - proj.half_h, entry.order + 1)
    return plane


def test_the_oracle_is_not_vacuous():
    """Guards the guard: if some future change re-snapped marks to their tiles,
    every test below would still pass while proving nothing."""
    scn = _offcentre_scenario()
    index = build_index(scn)
    moved = [e for e in index.entries if render.unit_paint_offset(e.unit) != (0.0, 0.0)]
    unmoved = [e for e in index.entries if render.unit_paint_offset(e.unit) == (0.0, 0.0)]
    assert len(moved) >= 6, "the off-centre population stopped being off-centre"
    assert len(unmoved) >= 2, "no centred/building control left in the population"


def test_flat_pick_agrees_with_the_offcentre_id_plane():
    scn = _offcentre_scenario()
    index = build_index(scn)
    tile_px = _oc_tile_px()
    plane = _oc_flat_id_plane(index, tile_px)

    rng = np.random.default_rng(OC_SEED)
    ys = rng.integers(0, plane.shape[0], OC_SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], OC_SAMPLE_PIXELS)
    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist(), strict=True):
        got = pick_unit(index, "flat", sx, sy, tile_px, OC_MAP_W, OC_MAP_H)
        expected = int(plane[sy, sx])
        hits += bool(expected)
        assert (0 if got is None else got.order + 1) == expected, (
            f"flat pick disagreed at ({sx}, {sy})"
        )
    assert hits > 0, "sampled no unit pixels at all -- the oracle would be vacuous"


def test_flat_pick_finds_every_painted_pixel_of_a_straddling_mark():
    """The sweep the plan asks for, exhaustive rather than sampled: an
    under-enumerating pick loop returns None, which reads as 'nothing there'."""
    scn = _offcentre_scenario()
    index = build_index(scn)
    tile_px = _oc_tile_px()
    plane = _oc_flat_id_plane(index, tile_px)
    entry = next(e for e in index.entries if (e.unit.x, e.unit.y) == (4.02, 4.98))
    ys, xs = np.nonzero(plane == entry.order + 1)
    assert ys.size, "this unit paints nothing, so the sweep proves nothing"
    for sx, sy in zip(xs.tolist(), ys.tolist(), strict=True):
        got = pick_unit(index, "flat", sx, sy, tile_px, OC_MAP_W, OC_MAP_H)
        assert got is not None and got.order == entry.order, f"missed at ({sx}, {sy})"


@pytest.mark.parametrize("elevated", [False, True], ids=["flat-ground", "raised-band"])
def test_stepped_pick_agrees_with_the_offcentre_id_plane(elevated: bool):
    scn = _offcentre_scenario(elevated=elevated)
    index = build_index(scn)
    tile_px = _oc_tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    plane = _oc_stepped_id_plane(index, tile_px, elevations, proj)

    rng = np.random.default_rng(OC_SEED)
    ys = rng.integers(0, plane.shape[0], OC_SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], OC_SAMPLE_PIXELS)

    disagreements = 0
    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist(), strict=True):
        got = pick_unit(index, "stepped", sx, sy, tile_px, OC_MAP_W, OC_MAP_H, elevations, proj)
        expected = int(plane[sy, sx])
        actual = 0 if got is None else got.order + 1
        if expected:
            hits += 1
        if actual == expected:
            continue
        disagreements += 1
        # Not a tolerance: every disagreement must be the documented skirt
        # residual, where screen_to_tile() has no analytic inverse.
        assert iso_geometry.screen_to_tile(sx, sy, elevations, proj) is None, (
            f"stepped pick disagreed at ({sx}, {sy}): expected {expected}, got {actual}, "
            "and this pixel is NOT a skirt-face pixel"
        )
    assert hits > 0, "sampled no unit pixels at all -- the oracle would be vacuous"
    assert disagreements < hits, f"{disagreements} disagreements against only {hits} unit pixels"


def test_the_highlight_follows_the_mark():
    """unit_polygons' whole contract is that the outline matches the pixels
    actually painted, so it has to take the same shift."""
    scn = _offcentre_scenario()
    index = build_index(scn)
    tile_px = _oc_tile_px()
    elevations, proj = render.elevations_and_proj(scn)
    entry = next(e for e in index.entries if (e.unit.x, e.unit.y) == (8.15, 8.85))

    (flat_poly,) = unit_polygons(entry, "flat", tile_px, OC_MAP_W, OC_MAP_H)
    xs = [p[0] for p in flat_poly]
    ys = [p[1] for p in flat_poly]
    plane = _oc_flat_id_plane(index, tile_px)
    py, px = np.nonzero(plane == entry.order + 1)
    assert (min(xs), min(ys)) == (int(px.min()), int(py.min()))
    assert (max(xs), max(ys)) == (int(px.max()) + 1, int(py.max()) + 1)

    # Stepped: for a 1x1 unit, `tile + 0.5 + offset` IS the unit's own
    # coordinate, so the diamond's centre must land exactly on the unit's
    # continuous map point. Asserted on the centre rather than on a shift,
    # because this unit's dx and dy happen to cancel in the x term
    # ((dx + dy) * half_w == 0 at 8.15 / 8.85) and a shift check on x alone
    # would read as "snapped" while the outline was perfectly correct.
    (stepped_poly,) = unit_polygons(
        entry, "stepped", tile_px, OC_MAP_W, OC_MAP_H, elevations=elevations, proj=proj
    )
    sxs = [p[0] for p in stepped_poly]
    sys_ = [p[1] for p in stepped_poly]
    centre = ((min(sxs) + max(sxs)) / 2, (min(sys_) + max(sys_)) / 2)
    elev = int(elevations[entry.own_y, entry.own_x])
    want = iso_geometry.map_point_to_screen(
        entry.unit.x, entry.unit.y, elev * proj.elev_step, proj
    )
    assert centre == want

    # And it is genuinely off the tile lattice the centred control sits on.
    centred = next(e for e in index.entries if (e.unit.x, e.unit.y) == (17.5, 3.5))
    (centred_poly,) = unit_polygons(
        centred, "stepped", tile_px, OC_MAP_W, OC_MAP_H, elevations=elevations, proj=proj
    )
    cys = [p[1] for p in centred_poly]
    lattice_y = ((8 - 8) - (17 - 3)) * proj.half_h
    assert centre[1] - (min(cys) + max(cys)) / 2 != lattice_y, (
        "the stepped highlight is still snapped to its tile"
    )


def test_a_tenth_of_a_tile_nudge_moves_the_render():
    """The pre-existing defect this stage fixes: _UNIT_NUDGE_STEP is 0.1 and
    the inspector's spinboxes take two decimals, and neither used to move a
    unit on screen in any style."""
    tile_px = _oc_tile_px()
    for style, kwargs in (("flat", {}), ("stepped", {})):
        before = render.unit_paint_offset(SyntheticUnit(x=8.5, y=8.5, unit_const=_PLAIN_CONST))
        after = render.unit_paint_offset(SyntheticUnit(x=8.6, y=8.5, unit_const=_PLAIN_CONST))
        assert before != after, style
        assert round((after[0] - before[0]) * tile_px) != 0, (
            f"0.1 tile is under half a pixel at tile_px={tile_px}, so {style} "
            "cannot show the nudge; pick a bigger map or a bigger step"
        )
        assert kwargs == {}


# --- the off-centre Sloped ID-plane oracle ----------------------------
#
# The exhaustive positional claim for a Sloped mark, made where it can be
# made correctly: against a plane that reproduces _paint_tile_and_units_sloped's
# own depth walk, so a mark partly painted over by a later tile is handled by
# construction rather than by a tolerance. Encodes the SAME centred/off-centre
# split the renderer uses -- a centred 1x1 unit paints through its tile's
# warped quad (Track C6), an off-centre one paints a diamond at its own point
# (free placement, Stage 1) -- which is what makes this an oracle for that
# split rather than a restatement of it.


def _oc_sloped_geometry(scn, elev_step_pct: int):
    """(corner_rise, proj) at a chosen elev_step_pct stop. Built here rather
    than via render.sloped_elevations_and_proj purely so the stop is a
    parameter; everything else matches that function, corner_headroom_steps
    included."""
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
    for tile in mm.terrain:
        elevations[tile.y, tile.x] = tile.elevation
    proj = iso_geometry.canvas_size_and_origin(
        mm.map_width, mm.map_height, tile_px,
        iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION,
        elev_step_pct=elev_step_pct, corner_headroom_steps=1,
    )
    corner_rise = iso_geometry.corner_rise_px(elevations, proj, rule=render.SLOPE_CORNER_RULE)
    return corner_rise, proj


def _oc_sloped_id_plane(index, tile_px: int, corner_rise, proj) -> np.ndarray:
    canvas_w, canvas_h = proj.canvas_w, proj.canvas_h
    plane = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    d_dst_y, d_dst_x, _sy, _sx = iso_geometry.diamond_indices(tile_px)

    by_footprint_tile: dict[tuple[int, int], list] = {}
    for entry in index.entries:
        for fx, fy in render.unit_occupied_tiles(entry.unit, OC_MAP_W, OC_MAP_H):
            by_footprint_tile.setdefault((fx, fy), []).append(entry)

    def paint(base_x, base_y, dy, dx, value):
        yy, xx = base_y + dy, base_x + dx
        ok = (yy >= 0) & (yy < canvas_h) & (xx >= 0) & (xx < canvas_w)
        plane[yy[ok], xx[ok]] = value

    for raw_x, raw_y in iso_geometry.depth_order(OC_MAP_W, OC_MAP_H):
        x, y = int(raw_x), int(raw_y)
        corners = (
            int(corner_rise[y, x]), int(corner_rise[y, x + 1]),
            int(corner_rise[y + 1, x]), int(corner_rise[y + 1, x + 1]),
        )
        bx, by = iso_geometry.tile_screen_origin(x, y, 0, proj)
        s_dst_y, s_dst_x, _a, _b, _uv = iso_geometry.sloped_quad_indices(tile_px, *corners)
        paint(bx, by - min(corners), s_dst_y, s_dst_x, 0)
        for entry in by_footprint_tile.get((x, y), ()):
            unit = entry.unit
            span_x, span_y = render.tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
            dx = unit.x - int(unit.x) - 0.5 if span_x <= 1 else 0.0
            dy = unit.y - int(unit.y) - 0.5 if span_y <= 1 else 0.0
            if span_x <= 1 and span_y <= 1 and (dx, dy) == (0.0, 0.0):
                paint(bx, by - min(corners), s_dst_y, s_dst_x, entry.order + 1)
                continue
            ux, uy = int(unit.x), int(unit.y)
            rise = iso_geometry.unit_rise_px(corner_rise, ux, uy, unit.x - ux, unit.y - uy)
            cx, cy = iso_geometry.map_point_to_screen(x + 0.5 + dx, y + 0.5 + dy, rise, proj)
            paint(cx - proj.half_w, cy - proj.half_h, d_dst_y, d_dst_x, entry.order + 1)
    return plane


@pytest.mark.parametrize("elev_step_pct", [50, 200])
def test_sloped_pick_agrees_with_the_offcentre_id_plane(elev_step_pct: int):
    """Run at 200 as well as the shipped 50 because every rise is four times
    longer there, which is what actually stretches _pick_unit_sloped's
    widened d bound rather than merely visiting it."""
    scn = _offcentre_scenario()
    # A west-to-east ramp straight through where the units stand, so every
    # tile under them has four genuinely different corner rises.
    for tile in scn.map_manager.terrain:
        tile.elevation = max(0, min(4, tile.x - 4))
    index = build_index(scn)
    tile_px = _oc_tile_px()
    corner_rise, proj = _oc_sloped_geometry(scn, elev_step_pct)
    plane = _oc_sloped_id_plane(index, tile_px, corner_rise, proj)
    terrain = render.composite_ids_rect_sloped(
        scn, 0, 0, proj.canvas_w, proj.canvas_h, corner_rise, proj, tile_px
    )

    rng = np.random.default_rng(OC_SEED)
    ys = rng.integers(0, plane.shape[0], OC_SAMPLE_PIXELS)
    xs = rng.integers(0, plane.shape[1], OC_SAMPLE_PIXELS)
    hits = 0
    for sx, sy in zip(xs.tolist(), ys.tolist(), strict=True):
        tid = int(terrain[sy, sx])
        tile = None if tid == render.PICK_ID_NONE else (tid % OC_MAP_W, tid // OC_MAP_W)
        got = pick_unit(
            index, "sloped", sx, sy, tile_px, OC_MAP_W, OC_MAP_H,
            corner_rise=corner_rise, proj=proj, terrain_tile=tile,
        )
        expected = int(plane[sy, sx])
        hits += bool(expected)
        assert (0 if got is None else got.order + 1) == expected, (
            f"sloped pick disagreed at ({sx}, {sy}) at pct={elev_step_pct}"
        )
    assert hits > 0, "sampled no unit pixels at all -- the oracle would be vacuous"


# --- a straddling mark must not clip at a chunk edge -------------------


def _stepped_chunked(scn, full, elevations, proj, tile_px, bboxes, chunk=128):
    """Every chunk-sized rect composited in isolation, stitched back together
    -- the same shape tests/test_sprite_chunks.py's own stitching check uses,
    and the only place a bbox that under-covers actually shows."""
    units_by_tile = render._units_by_tile(scn)
    out = np.zeros_like(full)
    h, w = full.shape[:2]
    for y0 in range(0, h, chunk):
        for x0 in range(0, w, chunk):
            x1, y1 = min(x0 + chunk, w), min(y0 + chunk, h)
            out[y0:y1, x0:x1] = render.composite_rect_iso(
                scn, x0, y0, x1, y1, elevations, proj, tile_px, units_by_tile, bboxes,
            )
    return out


def _straddling_scenario():
    """One off-centre 1x1 unit, shifted hard toward a neighbour so its mark
    crosses a tile boundary and, at this map size, a chunk boundary too."""
    tiles = [
        SyntheticTile(x=x, y=y, elevation=0) for y in range(OC_MAP_H) for x in range(OC_MAP_W)
    ]
    units_by_player = [[] for _ in range(9)]
    units_by_player[1] = [
        SyntheticUnit(x=x + 0.02, y=x + 0.98, unit_const=_PLAIN_CONST, reference_id=x)
        for x in range(2, OC_MAP_W - 2)
    ]
    return FakeScenario(OC_MAP_W, OC_MAP_H, tiles, units_by_player)


def test_a_straddling_mark_survives_chunked_compositing():
    scn = _straddling_scenario()
    tile_px = _oc_tile_px()
    full, elevations, proj = render.render_terrain_iso_with_proj(scn)
    bboxes = render._building_bboxes_iso(scn, OC_MAP_W, OC_MAP_H, proj, elevations)
    assert bboxes, (
        "no unit earned a bbox, so _building_bboxes_iso' span-1 skip is still swallowing "
        "the off-centre tail and this test would prove nothing"
    )
    got = _stepped_chunked(scn, full, elevations, proj, tile_px, bboxes)
    assert np.array_equal(got, full)


def test_an_off_centre_span_1_unit_earns_a_screen_bbox():
    """_building_bboxes_iso skipped every span-1 unit on the stated grounds
    that it "only ever paints its own tile, so it can't make a neighbour a
    bystander". A straddling mark breaks that premise, so the skip is now
    conditioned on a zero paint offset.

    Asserted on the bbox itself rather than through a stitched-chunk
    mutation, deliberately. The stitch above passes either way at this
    project's chunk size: composite_rect_iso's candidate set comes from
    tiles_in_screen_rect, which is already an accepted strict superset and
    carries enough margin to pull in the one neighbouring tile a mark can
    reach. So a mutation test here would assert nothing -- this is the
    property that actually changed, and the one a future margin change could
    silently start relying on.
    """
    scn = _straddling_scenario()
    _full, elevations, proj = render.render_terrain_iso_with_proj(scn)
    bboxes = render._building_bboxes_iso(scn, OC_MAP_W, OC_MAP_H, proj, elevations)

    unit = scn.unit_manager.units[1][0]
    key = (int(unit.x), int(unit.y))
    assert key in bboxes, "an off-centre span-1 unit is still being skipped"

    # And the bbox really covers the mark, corner to corner.
    dx, dy = render.unit_paint_offset(unit)
    cx, cy = iso_geometry.map_point_to_screen(
        key[0] + 0.5 + dx, key[1] + 0.5 + dy, 0, proj
    )
    sx0, sy0, sx1, sy1 = bboxes[key]
    assert sx0 <= cx - proj.half_w and sx1 >= cx + proj.half_w
    assert sy0 <= cy - proj.half_h and sy1 >= cy + proj.half_h


def test_a_centred_span_1_unit_still_earns_no_bbox():
    """The other half: the 88% must not start paying for a bbox they cannot
    need, which would grow every real file's bystander dict for nothing."""
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(OC_MAP_H) for x in range(OC_MAP_W)]
    units = [[] for _ in range(9)]
    units[1] = [SyntheticUnit(x=5.5, y=5.5, unit_const=_PLAIN_CONST, reference_id=1)]
    scn = FakeScenario(OC_MAP_W, OC_MAP_H, tiles, units)
    _full, elevations, proj = render.render_terrain_iso_with_proj(scn)
    assert render._building_bboxes_iso(scn, OC_MAP_W, OC_MAP_H, proj, elevations) == {}
