"""descape.grid_overlay: the Qt-free half of View > Grid.

The iso endpoints are checked against edge_ticks.iso_corner as the oracle,
and the grid is checked to meet the Distance Ticks ruler, which is the one
invariant invisible to a test of either module alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import edge_ticks, grid_overlay, iso_geometry, unit_pick


def _proj(w: int = 12, h: int = 9, tile_px: int = 32, min_elev: int = 0, max_elev: int = 3):
    return iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)


# --- geometry --------------------------------------------------------------


@pytest.mark.parametrize(("w", "h"), [(12, 9), (1, 1), (40, 24)])
def test_iso_line_counts_and_endpoints_match_iso_corner(w: int, h: int) -> None:
    proj = _proj(w, h)
    x_axis, y_axis = grid_overlay.grid_axes(w, h, proj=proj)
    assert (x_axis.axis, y_axis.axis) == ("x", "y")
    assert len(x_axis.lines) == w + 1
    assert len(y_axis.lines) == h + 1
    for i, (start, end) in enumerate(x_axis.lines):
        assert start == edge_ticks.iso_corner(i, 0, proj)
        assert end == edge_ticks.iso_corner(i, h, proj)
    for j, (start, end) in enumerate(y_axis.lines):
        assert start == edge_ticks.iso_corner(0, j, proj)
        assert end == edge_ticks.iso_corner(w, j, proj)


def test_flat_lines_span_the_canvas_border() -> None:
    w, h, tile_px = 20, 16, 8
    x_axis, y_axis = grid_overlay.grid_axes(w, h, tile_px=tile_px)
    assert x_axis.lines[0] == ((0, 0), (0, h * tile_px))
    assert x_axis.lines[-1] == ((w * tile_px, 0), (w * tile_px, h * tile_px))
    assert y_axis.lines[-1] == ((0, h * tile_px), (w * tile_px, h * tile_px))
    assert x_axis.minor_step == (float(tile_px), 0.0)
    assert y_axis.minor_step == (0.0, float(tile_px))


def test_endpoints_are_exact_integers() -> None:
    for axis in grid_overlay.grid_axes(12, 9, proj=_proj()):
        for line in axis.lines:
            for point in line:
                assert all(type(c) is int for c in point)


def test_the_minor_step_walks_from_one_line_to_the_next() -> None:
    proj = _proj()
    for axis in grid_overlay.grid_axes(12, 9, proj=proj):
        (x0, y0), _ = axis.lines[0]
        (x1, y1), _ = axis.lines[1]
        assert (float(x1 - x0), float(y1 - y0)) == axis.minor_step


@pytest.mark.parametrize("interval", edge_ticks.TICK_INTERVALS)
@pytest.mark.parametrize("style", ["iso", "flat"])
def test_every_ruler_tick_sits_on_the_end_of_a_grid_line(interval: int, style: str) -> None:
    w, h = 40, 24
    kwargs = {"proj": _proj(w, h)} if style == "iso" else {"tile_px": 8}
    endpoints = {point for axis in grid_overlay.grid_axes(w, h, **kwargs) for line in axis.lines for point in line}
    for run in edge_ticks.edge_runs(w, h, interval, **kwargs):
        for anchor in run.anchors:
            assert anchor in endpoints, (run.edge, anchor)


def test_grid_axes_needs_a_projection_or_a_tile_size() -> None:
    with pytest.raises(ValueError):
        grid_overlay.grid_axes(4, 4)


def test_a_projection_wins_over_tile_px() -> None:
    proj = _proj(4, 4)
    assert grid_overlay.grid_axes(4, 4, proj=proj, tile_px=8) == grid_overlay.grid_axes(4, 4, proj=proj)


def test_majors_land_every_fourth_line_from_zero() -> None:
    x_axis, _ = grid_overlay.grid_axes(12, 9, tile_px=8)
    assert [i for i, major in enumerate(x_axis.majors) if major] == [0, 4, 8, 12]


# --- LOD ---------------------------------------------------------------------


def test_minors_drop_before_the_grid_does() -> None:
    seen = [grid_overlay.grid_lod(s) for s in (40.0, 5.0, 4.9, 1.5, 1.49, 0.1)]
    assert seen[0] == grid_overlay.GridLod(True, True)
    assert seen[1] == grid_overlay.GridLod(True, True)
    assert seen[2] == grid_overlay.GridLod(True, False)
    assert seen[3] == grid_overlay.GridLod(True, False)
    assert seen[4] == grid_overlay.GridLod(False, False)


def test_the_ladder_is_monotonic_across_a_sweep() -> None:
    previous = grid_overlay.grid_lod(100.0)
    for tenth in range(1000, -1, -1):
        lod = grid_overlay.grid_lod(tenth / 10.0)
        assert lod.draw_grid <= previous.draw_grid
        assert lod.draw_minors <= previous.draw_minors
        assert lod.draw_minors <= lod.draw_grid
        previous = lod


# --- appearance --------------------------------------------------------------


@pytest.mark.parametrize(("raw", "clamped"), [(-999, -100), (-100, -100), (-7, -7), (0, 0), (43, 43), (100, 100), (999, 100)])
def test_a_blend_clamps_to_the_range(raw: int, clamped: int) -> None:
    assert grid_overlay.clamp_blend(raw) == clamped


@pytest.mark.parametrize(("raw", "snapped"), [(-3, 1), (1, 1), (3, 3), (9, 4)])
def test_thickness_snaps_to_the_nearest_stop(raw: int, snapped: int) -> None:
    assert grid_overlay.snap_thickness(raw) == snapped
    assert grid_overlay.thickness_for_index(grid_overlay.thickness_index(raw)) == snapped


def test_the_defaults_are_in_range() -> None:
    assert grid_overlay.BLEND_MIN <= grid_overlay.BLEND_DEFAULT <= grid_overlay.BLEND_MAX
    assert grid_overlay.THICKNESS_DEFAULT in grid_overlay.THICKNESS_STOPS


@pytest.mark.parametrize("blend", [-100, -63, -1, 1, 37, 100])
def test_majors_are_more_opaque_and_both_ranks_take_the_blend_side(blend: int) -> None:
    minor, major = grid_overlay.grid_colors(blend)
    channel = 255 if blend > 0 else 0
    assert major[3] > minor[3]
    assert minor[:3] == major[:3] == (channel, channel, channel)


def test_a_centred_blend_is_fully_transparent() -> None:
    minor, major = grid_overlay.grid_colors(0)
    assert minor[3] == major[3] == 0


def test_full_deflection_reaches_the_rank_ceilings_on_both_sides() -> None:
    for blend in (grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX):
        minor, major = grid_overlay.grid_colors(blend)
        assert (minor[3], major[3]) == (grid_overlay.MINOR_ALPHA, grid_overlay.MAJOR_ALPHA)


@pytest.mark.parametrize("magnitude", [1, 25, 50, 99, 100])
def test_the_two_sides_are_mirror_images_in_alpha(magnitude: int) -> None:
    """Only the channel flips across the centre; the strength does not."""
    dark, light = grid_overlay.grid_colors(-magnitude), grid_overlay.grid_colors(magnitude)
    assert [c[3] for c in dark] == [c[3] for c in light]


@pytest.mark.parametrize(
    ("lightness", "blend"), [(0, -100), (30, -75), (120, 0), (150, 25), (240, 100), (999, 100)]
)
def test_a_legacy_lightness_maps_onto_the_signed_axis(lightness: int, blend: int) -> None:
    assert grid_overlay.blend_for_lightness(lightness) == blend


# --- Follow Terrain Elevation ------------------------------------------------


def _all_tiles(w: int, h: int) -> np.ndarray:
    return np.array([(x, y) for y in range(h) for x in range(w)], dtype=np.int64)


def _elevated(w: int = 12, h: int = 9, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, (h, w))


def _unit_edges(segments: np.ndarray, half_w: int) -> set[tuple[int, int, int, int]]:
    """Every segment split into its per-tile steps (each one half_w wide on
    screen), direction-normalized, so merged runs compare against per-tile
    oracle edges."""
    edges = set()
    for x0, y0, x1, y1 in segments.tolist():
        steps = abs(x1 - x0) // half_w
        assert steps * half_w == abs(x1 - x0) and (y1 - y0) % steps == 0
        dx, dy = (x1 - x0) // steps, (y1 - y0) // steps
        for k in range(steps):
            a = (x0 + k * dx, y0 + k * dy)
            b = (a[0] + dx, a[1] + dy)
            edges.add((*min(a, b), *max(a, b)))
    return edges


def _oracle_edge(a, b) -> tuple[int, int, int, int]:
    return (*min(a, b), *max(a, b))


def test_corner_point_at_zero_rise_is_iso_corner() -> None:
    proj = _proj()
    for cx in range(13):
        for cy in range(10):
            assert grid_overlay.corner_point(cx, cy, 0, proj) == edge_ticks.iso_corner(cx, cy, proj)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_flat_map_drapes_to_exactly_the_ground_plane_lines(style: str) -> None:
    w, h = 12, 9
    proj = _proj(w, h)
    kwargs = {"elevations": np.zeros((h, w), dtype=np.int64)} if style == "stepped" else {
        "corner_rise": np.zeros((h + 1, w + 1), dtype=np.int64)}
    minor, major = grid_overlay.draped_lines(_all_tiles(w, h), style, proj, **kwargs)
    x_axis, y_axis = grid_overlay.grid_axes(w, h, proj=proj)
    for got, want_major in ((minor, False), (major, True)):
        expected = {
            (a[0], a[1], b[0], b[1])
            for axis in (x_axis, y_axis)
            for (a, b), is_major in zip(axis.lines, axis.majors, strict=True)
            if is_major == want_major
        }
        assert {tuple(s) for s in got.tolist()} == expected


def _diamond_edges(x: int, y: int, elevations: np.ndarray, proj) -> list[tuple[int, int, int, int]]:
    """Tile (x, y)'s four highlight-diamond edges at its own elevation."""
    ox, oy = iso_geometry.tile_screen_origin(x, y, int(elevations[y, x]), proj)
    points = [tuple(map(int, p)) for p in unit_pick.diamond_points(ox, oy, proj.half_w, proj.half_h)]
    return [_oracle_edge(points[i], points[(i + 1) % 4]) for i in range(4)]


def _buried(edge, x: int, y: int, elevations: np.ndarray, proj) -> bool:
    """Would a tile painted AFTER (x, y) cover this edge's midpoint? Each
    candidate's silhouette is its diamond swept down by its own rise -- face
    plus skirt, which is what render.py actually paints -- and the paint order
    is iso_geometry.depth_order's own ascending d = y - x."""
    h, w = elevations.shape
    mx, my = (edge[0] + edge[2]) / 2.0, (edge[1] + edge[3]) / 2.0
    half_w, half_h = proj.half_w, proj.half_h
    for bx, by in iso_geometry.depth_order(w, h).tolist():
        if (by - bx, bx) <= (y - x, x):
            continue
        ox, oy = iso_geometry.tile_screen_origin(bx, by, int(elevations[by, bx]), proj)
        drop = int(elevations[by, bx]) * proj.elev_step
        for dy in range(drop + 1):
            if abs(mx - ox - half_w) * half_h + abs(my - oy - dy - half_h) * half_w < half_w * half_h:
                return True
    return False


def test_stepped_segments_are_highlight_diamond_edges() -> None:
    """unit_pick.diamond_points at each tile's own elevation, the construction
    MapView._tile_polygon uses for the brush highlight, is the oracle. A
    subset, not equality: see the buried-edge test below."""
    w, h = 12, 9
    proj = _proj(w, h)
    elevations = _elevated(w, h)
    minor, major = grid_overlay.draped_lines(_all_tiles(w, h), "stepped", proj, elevations=elevations)
    allowed = {e for y in range(h) for x in range(w) for e in _diamond_edges(x, y, elevations, proj)}
    assert _unit_edges(np.concatenate([minor, major]), proj.half_w) <= allowed


def test_stepped_drops_a_diamond_edge_only_where_a_nearer_tile_buries_it() -> None:
    """The mid-tile-gridline bug: a behind-and-lower tile's copy of a shared
    boundary used to paint inside the front tile's face, since the overlay has
    no depth test. Both directions matter -- dropping one visible edge would
    leave a gap in the lattice."""
    w, h = 9, 7
    proj = _proj(w, h)
    # A stepped pyramid: +-1 between cardinal neighbours, which is all
    # MapManager._elevation_tile_recursion (the in-game brush's own smoothing)
    # ever produces, and it slopes every way at once so each of the four
    # neighbour directions is exercised as both the higher and the lower side.
    xs, ys = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    elevations = np.minimum.reduce([xs, ys, w - 1 - xs, h - 1 - ys]).astype(np.int64)
    assert np.abs(np.diff(elevations, axis=0)).max() <= 1
    assert np.abs(np.diff(elevations, axis=1)).max() <= 1
    minor, major = grid_overlay.draped_lines(_all_tiles(w, h), "stepped", proj, elevations=elevations)
    drawn = _unit_edges(np.concatenate([minor, major]), proj.half_w)
    for y in range(h):
        for x in range(w):
            for edge in _diamond_edges(x, y, elevations, proj):
                buried = _buried(edge, x, y, elevations, proj)
                if buried:
                    continue
                assert edge in drawn, f"visible edge {edge} of tile {(x, y)} was dropped"
    for edge in drawn:
        owners = [
            (x, y)
            for y in range(h)
            for x in range(w)
            if edge in _diamond_edges(x, y, elevations, proj)
        ]
        assert any(not _buried(edge, x, y, elevations, proj) for x, y in owners), (
            f"{edge} is drawn but every tile it belongs to has it buried"
        )


def test_stepped_draws_a_height_step_once_per_side_at_each_sides_height() -> None:
    proj = _proj(2, 1)
    elevations = np.array([[2, 3]], dtype=np.int64)
    minor, major = grid_overlay.draped_lines(_all_tiles(2, 1), "stepped", proj, elevations=elevations)
    edges = _unit_edges(np.concatenate([minor, major]), proj.half_w)
    for elevation in (2, 3):
        rise = elevation * proj.elev_step
        top = grid_overlay.corner_point(1, 0, rise, proj)
        bottom = grid_overlay.corner_point(1, 1, rise, proj)
        assert _oracle_edge(top, bottom) in edges


def test_stepped_draws_a_shared_edge_at_equal_height_only_once() -> None:
    proj = _proj(2, 1)
    minor, major = grid_overlay.draped_lines(
        _all_tiles(2, 1), "stepped", proj, elevations=np.array([[1, 1]], dtype=np.int64)
    )
    segments = [tuple(s) for s in np.concatenate([minor, major]).tolist()]
    assert len(segments) == len(set(segments))
    rise = proj.elev_step
    shared = _oracle_edge(grid_overlay.corner_point(1, 0, rise, proj), grid_overlay.corner_point(1, 1, rise, proj))
    assert sum(1 for s in segments if _unit_edges(np.array([s]), proj.half_w) == {shared}) == 1


def test_sloped_corners_come_from_corner_rise_and_the_lattice_is_shared() -> None:
    w, h = 12, 9
    proj = _proj(w, h)
    corner_rise = iso_geometry.corner_rise_px(_elevated(w, h), proj)
    minor, major = grid_overlay.draped_lines(_all_tiles(w, h), "sloped", proj, corner_rise=corner_rise)
    expected = set()
    for cy in range(h + 1):
        for cx in range(w + 1):
            here = grid_overlay.corner_point(cx, cy, int(corner_rise[cy, cx]), proj)
            if cx < w:
                expected.add(_oracle_edge(here, grid_overlay.corner_point(cx + 1, cy, int(corner_rise[cy, cx + 1]), proj)))
            if cy < h:
                expected.add(_oracle_edge(here, grid_overlay.corner_point(cx, cy + 1, int(corner_rise[cy + 1, cx]), proj)))
    segments = np.concatenate([minor, major])
    assert _unit_edges(segments, proj.half_w) == expected
    assert len({tuple(s) for s in segments.tolist()}) == len(segments)


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_culling_to_a_window_loses_nothing_inside_it(style: str) -> None:
    w, h = 24, 20
    proj = _proj(w, h)
    elevations = _elevated(w, h, seed=7)
    kwargs = {"elevations": elevations} if style == "stepped" else {
        "corner_rise": iso_geometry.corner_rise_px(elevations, proj)}
    full = np.concatenate(grid_overlay.draped_lines(_all_tiles(w, h), style, proj, **kwargs))
    x0, y0, x1, y1 = 300, 150, 700, 450
    window = iso_geometry.tiles_in_screen_rect(x0, y0, x1, y1, w, h, proj)
    culled = np.concatenate(grid_overlay.draped_lines(window, style, proj, **kwargs))

    def inside(edges):
        return {e for e in edges if x0 <= e[0] and e[2] <= x1 and y0 <= min(e[1], e[3]) and max(e[1], e[3]) <= y1}

    assert inside(_unit_edges(full, proj.half_w)) <= _unit_edges(culled, proj.half_w)


def test_draped_endpoints_are_integers() -> None:
    proj = _proj()
    minor, major = grid_overlay.draped_lines(_all_tiles(12, 9), "stepped", proj, elevations=_elevated())
    assert minor.dtype == major.dtype == np.int64


def test_each_style_rejects_the_other_styles_source() -> None:
    proj = _proj()
    tiles = _all_tiles(12, 9)
    with pytest.raises(ValueError):
        grid_overlay.draped_lines(tiles, "stepped", proj)
    with pytest.raises(ValueError):
        grid_overlay.draped_lines(tiles, "sloped", proj)
    with pytest.raises(ValueError):
        grid_overlay.draped_lines(tiles, "stepped", proj, corner_rise=np.zeros((10, 13), dtype=np.int64))
    with pytest.raises(ValueError):
        grid_overlay.draped_lines(tiles, "sloped", proj, elevations=np.zeros((9, 12), dtype=np.int64))
    with pytest.raises(ValueError):
        grid_overlay.draped_lines(tiles, "flat", proj, elevations=np.zeros((9, 12), dtype=np.int64))
