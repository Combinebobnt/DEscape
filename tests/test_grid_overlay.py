"""descape.grid_overlay: the Qt-free half of View > Grid.

The iso endpoints are checked against edge_ticks.iso_corner as the oracle,
and the grid is checked to meet the Distance Ticks ruler, which is the one
invariant invisible to a test of either module alone.
"""

from __future__ import annotations

import pytest

from descape import edge_ticks, grid_overlay, iso_geometry


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


@pytest.mark.parametrize("every", grid_overlay.MAJOR_EVERY_STOPS)
@pytest.mark.parametrize("style", ["iso", "flat"])
def test_majors_land_on_multiples_of_each_spacing_and_only_there(every: int, style: str) -> None:
    w, h = 40, 33
    kwargs = {"proj": _proj(w, h)} if style == "iso" else {"tile_px": 8}
    x_axis, y_axis = grid_overlay.grid_axes(w, h, major_every=every, **kwargs)
    assert [i for i, major in enumerate(x_axis.majors) if major] == list(range(0, w + 1, every))
    assert [j for j, major in enumerate(y_axis.majors) if major] == list(range(0, h + 1, every))
    assert [i for i in range(w + 1) if grid_overlay.is_major(i, every)] == list(range(0, w + 1, every))
    # Geometry does not depend on the spacing, only the major flags do.
    default_x, _ = grid_overlay.grid_axes(w, h, **kwargs)
    assert x_axis.lines == default_x.lines


@pytest.mark.parametrize("every", grid_overlay.MAJOR_EVERY_STOPS)
def test_owned_edges_flag_majors_at_the_spacing(every: int) -> None:
    w = h = 33
    for tx in range(w):
        edges = dict(grid_overlay.owned_edges(tx, tx, w, h, major_every=every))
        assert edges["x_low"] is (tx % every == 0)
        assert edges["y_low"] is (tx % every == 0)
        if tx + 1 == w:
            assert edges["x_high"] is ((tx + 1) % every == 0)


def test_the_default_spacing_is_four_and_a_stop() -> None:
    assert grid_overlay.MAJOR_EVERY_DEFAULT == 4
    assert grid_overlay.MAJOR_EVERY_STOPS == (2, 4, 8, 16)
    assert [i for i in range(17) if grid_overlay.is_major(i)] == [0, 4, 8, 12, 16]
    assert grid_overlay.owned_edges(4, 5, 16, 16) == grid_overlay.owned_edges(4, 5, 16, 16, major_every=4)


@pytest.mark.parametrize(
    ("value", "valid"),
    [(2, True), (4, True), (8, True), (16, True), (3, False), (32, False), (0, False), (True, False),
     ("4", False), (4.0, False), (None, False)],
)
def test_only_an_int_stop_is_a_valid_spacing(value, valid: bool) -> None:
    assert grid_overlay.valid_major_every(value) is valid


def test_the_spacing_is_part_of_the_bake_spec() -> None:
    assert grid_overlay.grid_bake(True, -100, 1) == grid_overlay.grid_bake(True, -100, 1, 4)
    assert grid_overlay.grid_bake(True, -100, 1, major_every=8) != grid_overlay.grid_bake(True, -100, 1)
    assert grid_overlay.grid_bake(True, -100, 1, major_every=8).major_every == 8
    assert grid_overlay.DEFAULT_GRID.major_every == grid_overlay.MAJOR_EVERY_DEFAULT


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


def test_the_default_spacing_keeps_todays_lod_verdicts() -> None:
    """Today's rule written out literally: majors 4 minors apart at least
    6 px, minors at least 5 px."""
    for tenth in range(1001):
        s = tenth / 10.0
        expected = grid_overlay.GridLod(s * 4 >= 6.0, s * 4 >= 6.0 and s >= 5.0)
        assert grid_overlay.grid_lod(s) == expected, s
        assert grid_overlay.grid_lod(s, 4) == expected, s
        assert grid_overlay.bake_lod(s, 4) == expected, s


def test_a_coarse_grid_survives_further_out_and_a_fine_one_drops_sooner() -> None:
    # 1.0 px per tile: today's majors are 4 px apart (dropped); every 16 they are 16 px apart.
    assert not grid_overlay.grid_lod(1.0).draw_grid
    assert grid_overlay.grid_lod(1.0, 16).draw_grid
    assert grid_overlay.grid_lod(1.0, 8).draw_grid
    # 2.0 px per tile: today's majors are 8 px apart (kept); every 2 they are 4 px apart.
    assert grid_overlay.grid_lod(2.0).draw_grid
    assert not grid_overlay.grid_lod(2.0, 2).draw_grid


@pytest.mark.parametrize("every", grid_overlay.MAJOR_EVERY_STOPS)
def test_minors_drop_before_the_grid_at_every_spacing(every: int) -> None:
    previous = grid_overlay.grid_lod(100.0, every)
    assert previous == grid_overlay.GridLod(True, True)
    for tenth in range(1000, -1, -1):
        lod = grid_overlay.grid_lod(tenth / 10.0, every)
        assert lod.draw_grid <= previous.draw_grid
        assert lod.draw_minors <= previous.draw_minors
        assert lod.draw_minors <= lod.draw_grid
        assert lod.draw_grid == (tenth / 10.0 * every >= grid_overlay.MIN_GRID_SPACING_PX)
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
