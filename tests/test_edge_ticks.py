"""descape.edge_ticks: the Qt-free placement half of the map-edge distance
ruler. No QApplication, no PyQt5 import; the item that consumes this is
covered by tests/test_edge_ticks_viewer.py instead.

Three proof obligations get their own tests here, each re-derived rather
than assumed:

1. The anchor formula agrees with iso_geometry.ground_outline_corners at the
   four grid extremes, exactly, in integers. That function is used as the
   ORACLE, never a hand-typed expected list. The two must not be able to
   drift apart silently.
2. scene_pad(s, font_px) * s >= sqrt(2) * device_reach_px(font_px), and
   device_reach_px(font_px) dominates everything the item actually draws,
   at every legal font_px. The sqrt(2) matters: the obvious form of this
   check (>= device_reach_px, no factor) still passes with PAD_SAFETY cut
   to 1.0, which breaks containment under Flat's scale(1, 0.5) +
   rotate(-45) where the smallest singular value is sqrt(0.5) of the scale
   the caller measures.
3. The LOD ladder drops labels strictly before minors, and the edge last.
4. label_box_px/label_center_px/device_reach_px reproduce today's literal
   constants exactly at font_px=12, and stay exact (not merely close) at
   every other legal font_px, since they're multiply-before-divide rather
   than a ratio constant.
"""

from __future__ import annotations

import math

import pytest

from descape import edge_ticks, iso_geometry
from descape.scenario_new import STANDARD_MAP_SIZES


def _proj(w: int = 12, h: int = 9, tile_px: int = 32, min_elev: int = 0, max_elev: int = 3):
    return iso_geometry.canvas_size_and_origin(w, h, tile_px, min_elev, max_elev)


# --- obligation 1: anchors against the ground_outline_corners oracle -------


@pytest.mark.parametrize("w, h", [(12, 9), (1, 1), (120, 120), (168, 200)])
def test_the_four_extreme_anchors_are_ground_outline_corners(w: int, h: int) -> None:
    """west/north/east/south, in that compass order, are corners (0,0),
    (w,0), (w,h), (0,h) of the generalised formula."""
    proj = _proj(w, h)
    west, north, east, south = iso_geometry.ground_outline_corners(w, h, proj)
    assert edge_ticks.iso_corner(0, 0, proj) == west
    assert edge_ticks.iso_corner(w, 0, proj) == north
    assert edge_ticks.iso_corner(w, h, proj) == east
    assert edge_ticks.iso_corner(0, h, proj) == south


def test_anchors_are_exact_integers_not_rounded_floats() -> None:
    """tile_screen_origin is all-integer arithmetic and iso_corner adds one
    more int, so an == against the oracle above is meaningful rather than
    accidentally-close."""
    proj = _proj(120, 120, tile_px=48, min_elev=1, max_elev=7)
    for i in (0, 7, 60, 119, 120):
        x, y = edge_ticks.iso_corner(i, 3, proj)
        assert isinstance(x, int) and isinstance(y, int)


def test_every_run_anchor_matches_iso_corner() -> None:
    """The four EdgeRuns place their anchors with the same formula the
    oracle test above pins, not a second copy of it."""
    w, h, interval = 40, 24, 4
    proj = _proj(w, h)
    runs = {run.edge: run for run in edge_ticks.edge_runs(w, h, interval, proj=proj)}
    assert runs["y0"].anchors == tuple(edge_ticks.iso_corner(i, 0, proj) for i in range(0, w + 1, interval))
    assert runs["y1"].anchors == tuple(edge_ticks.iso_corner(i, h, proj) for i in range(0, w + 1, interval))
    assert runs["x0"].anchors == tuple(edge_ticks.iso_corner(0, j, proj) for j in range(0, h + 1, interval))
    assert runs["x1"].anchors == tuple(edge_ticks.iso_corner(w, j, proj) for j in range(0, h + 1, interval))


# --- outward and step directions -------------------------------------------


def test_iso_outward_directions_point_away_from_the_map_center() -> None:
    """Each edge's outward ray, followed from any of its anchors, must
    increase the distance to the grid's own screen-space center."""
    w, h = 40, 24
    proj = _proj(w, h)
    center_x = (edge_ticks.iso_corner(0, 0, proj)[0] + edge_ticks.iso_corner(w, h, proj)[0]) / 2.0
    center_y = (edge_ticks.iso_corner(w, 0, proj)[1] + edge_ticks.iso_corner(0, h, proj)[1]) / 2.0
    for run in edge_ticks.edge_runs(w, h, 4, proj=proj):
        # The shared corner anchors sit ON a diagonal through the center, so
        # an interior anchor is the honest probe.
        ax, ay = run.anchors[len(run.anchors) // 2]
        before = math.hypot(ax - center_x, ay - center_y)
        after = math.hypot(ax + run.outward[0] - center_x, ay + run.outward[1] - center_y)
        assert after > before, f"{run.edge} outward {run.outward} points inward"


def test_the_minor_step_walks_from_one_anchor_to_the_next() -> None:
    """minor_step is what the item measures LOD spacing with, so it has to
    be the real anchor-to-anchor delta, not a per-tile one."""
    w, h, interval = 40, 24, 5
    proj = _proj(w, h)
    for run in edge_ticks.edge_runs(w, h, interval, proj=proj):
        for k in range(len(run.anchors) - 1):
            dx = run.anchors[k + 1][0] - run.anchors[k][0]
            dy = run.anchors[k + 1][1] - run.anchors[k][1]
            assert (dx, dy) == pytest.approx(run.minor_step)


def test_flat_outward_is_axis_aligned_and_unit_length() -> None:
    """Flat's y0 outward is exactly (0, -1). The Flat + non-isometric case
    is what makes the item's device-length test a countable vertical run of
    pixels with no antialiasing to argue about."""
    runs = {run.edge: run for run in edge_ticks.edge_runs(20, 16, 4, tile_px=8)}
    assert runs["y0"].outward == (0.0, -1.0)
    assert runs["y1"].outward == (0.0, 1.0)
    assert runs["x0"].outward == (-1.0, 0.0)
    assert runs["x1"].outward == (1.0, 0.0)


def test_flat_anchors_sit_on_the_canvas_border() -> None:
    """FlatChunkCache.canvas_dims() is (map_w*tile_px, map_h*tile_px), so
    the flat branch's own anchors must land exactly on that rect."""
    w, h, tile_px = 20, 16, 8
    runs = {run.edge: run for run in edge_ticks.edge_runs(w, h, 4, tile_px=tile_px)}
    assert all(y == 0 for _x, y in runs["y0"].anchors)
    assert all(y == h * tile_px for _x, y in runs["y1"].anchors)
    assert all(x == 0 for x, _y in runs["x0"].anchors)
    assert all(x == w * tile_px for x, _y in runs["x1"].anchors)


def test_each_edge_labels_the_coordinate_that_varies_along_it() -> None:
    """y0/y1 read as an X ruler and x0/x1 as a Y ruler, which is only true
    while `tiles` carries the varying axis."""
    w, h, interval = 40, 24, 4
    runs = {run.edge: run for run in edge_ticks.edge_runs(w, h, interval, proj=_proj(w, h))}
    assert runs["y0"].tiles == runs["y1"].tiles == tuple(range(0, w + 1, interval))
    assert runs["x0"].tiles == runs["x1"].tiles == tuple(range(0, h + 1, interval))


# --- tick indices and major classification ---------------------------------


def test_the_run_ends_flush_only_when_the_interval_divides_the_span() -> None:
    assert edge_ticks.tick_indices(120, 4)[-1] == 120
    assert edge_ticks.tick_indices(144, 5)[-1] == 140  # 144 % 5 != 0
    assert edge_ticks.tick_indices(0, 4) == (0,)


def test_tick_indices_rejects_a_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        edge_ticks.tick_indices(120, 0)


def test_index_zero_is_always_a_major() -> None:
    for interval in edge_ticks.TICK_INTERVALS:
        assert edge_ticks.is_major(0, interval)


@pytest.mark.parametrize("interval", edge_ticks.TICK_INTERVALS)
def test_every_nth_minor_is_a_major(interval: int) -> None:
    indices = edge_ticks.tick_indices(200, interval)
    majors = [n for n, i in enumerate(indices) if edge_ticks.is_major(i, interval)]
    assert majors == list(range(0, len(indices), edge_ticks.MAJORS_PER_MINOR))


def test_the_default_interval_lands_a_major_on_more_standard_sizes() -> None:
    """Why the default is 5 and not 4, measured against the real map sizes
    rather than asserted. Majors land every interval*MAJORS_PER_MINOR tiles."""
    def flush(interval: int) -> set[int]:
        span = interval * edge_ticks.MAJORS_PER_MINOR
        return {size for size in STANDARD_MAP_SIZES if size % span == 0}

    # Measured, not quoted: every-5's 20-tile majors land flush on 5 of the
    # 7 standard sizes; every-4's 16-tile majors land on only 3 (144, 240,
    # 480). 5 of 7 against 3 of 7 is why 5 is the default.
    assert flush(4) == {144, 240, 480}
    assert flush(5) == {120, 200, 220, 240, 480}
    assert len(flush(5)) > len(flush(4))
    assert edge_ticks.TICK_INTERVAL_DEFAULT == 5


def test_a_480_at_the_default_interval_is_388_ticks() -> None:
    """The count the anchor cache exists for. Pinned so a change to the
    interval set or the inclusive endpoint shows up as a number, not as a
    frame-rate complaint."""
    runs = edge_ticks.edge_runs(480, 480, edge_ticks.TICK_INTERVAL_DEFAULT, tile_px=8)
    assert sum(len(run.anchors) for run in runs) == 388


# --- obligation 2: the pad contains everything paint() draws ---------------


# sqrt(0.5) is the smallest singular value of scale(1, 0.5) + rotate(-45)
# relative to that transform's own sqrt-determinant scale, so a pad derived
# from the measured scale must clear this factor to still contain the reach.
_FLAT_ISO_MIN_SINGULAR_RATIO = math.sqrt(0.5)


_FONT_PX_RANGE = tuple(range(8, 25))  # settings.DISTANCE_TICK_FONT_PX_MIN..MAX inclusive


@pytest.mark.parametrize("scale", [1e-4, 0.001, 0.04, 0.5, 1.0, 7.5, 400.0])
@pytest.mark.parametrize("font_px", _FONT_PX_RANGE)
def test_the_pad_contains_the_device_reach_under_flats_worst_axis(scale: float, font_px: int) -> None:
    """The discriminating form. Dropping the ratio (asserting only
    >= device_reach_px(font_px)) passes even with PAD_SAFETY cut to 1.0,
    which is exactly the regression this guards. Parametrized over every
    legal font_px: the sqrt(2)/PAD_SAFETY containment argument is
    font-independent in its derivation but not proven so unless it's
    actually checked across the range."""
    pad = edge_ticks.scene_pad(scale, font_px)
    assert pad * scale * _FLAT_ISO_MIN_SINGULAR_RATIO >= edge_ticks.device_reach_px(font_px)


def test_pad_safety_clears_the_flat_isometric_floor() -> None:
    assert edge_ticks.PAD_SAFETY >= 1.0 / _FLAT_ISO_MIN_SINGULAR_RATIO


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_a_non_positive_scale_yields_no_pad_rather_than_dividing(scale: float) -> None:
    assert edge_ticks.scene_pad(scale, edge_ticks.LABEL_FONT_PX) == 0.0


def test_the_pad_grows_as_the_view_zooms_out() -> None:
    font_px = edge_ticks.LABEL_FONT_PX
    assert edge_ticks.scene_pad(0.01, font_px) > edge_ticks.scene_pad(1.0, font_px) > edge_ticks.scene_pad(
        100.0, font_px
    )


@pytest.mark.parametrize("font_px", _FONT_PX_RANGE)
def test_device_reach_dominates_every_mark_the_item_draws(font_px: int) -> None:
    """Each drawn element's furthest device-space distance from its anchor,
    re-derived here rather than copied from the function's own expression."""
    box_w, box_h = edge_ticks.label_box_px(font_px)
    label_center = edge_ticks.label_center_px(font_px)
    label_far_corner = label_center + math.hypot(box_w, box_h) / 2.0
    reach = edge_ticks.device_reach_px(font_px)
    assert reach >= edge_ticks.MINOR_TICK_PX
    assert reach >= edge_ticks.MAJOR_TICK_PX
    assert reach >= label_far_corner
    assert edge_ticks.MAJOR_TICK_PX > edge_ticks.MINOR_TICK_PX
    # The gap is a real gap: the label box's NEAR edge clears the major tick.
    # approx, not ==: label_center and box_h are each their own independent
    # float computation, and their difference can land a ULP off the direct
    # sum even though both derive from the same multiply-before-divide value.
    assert label_center - box_h / 2.0 == pytest.approx(edge_ticks.MAJOR_TICK_PX + edge_ticks.LABEL_GAP_PX)


def test_label_box_and_derived_geometry_are_exact_at_every_font_px_no_drift() -> None:
    """Multiply-before-divide (font_px * 40 / 12), not a ratio constant
    ((40 / 12) * font_px): the former is exact by construction at every
    integer font_px in [8, 24] since 12 divides 480 and 192 evenly, while
    the latter disagrees by a ULP at 10, 14 and 20. Checked by `==`, not
    pytest.approx, since the whole point is no drift -- this is the test
    that would catch a naive ratio-constant implementation red."""
    for font_px in _FONT_PX_RANGE:
        box_w, box_h = edge_ticks.label_box_px(font_px)
        assert box_w == font_px * 40.0 / 12.0
        assert box_h == font_px * 16.0 / 12.0

    # At the default (12), every function reproduces today's literals
    # exactly -- a config with no distance_tick_font_px key must be
    # byte-identical to current behaviour.
    assert edge_ticks.label_box_px(12) == (40.0, 16.0)
    assert edge_ticks.label_center_px(12) == 23.0
    assert edge_ticks.device_reach_px(12) == 23.0 + math.hypot(40.0, 16.0) / 2.0


# --- obligation 3: the LOD ladder ------------------------------------------


def test_labels_drop_before_minors_which_drop_before_the_edge() -> None:
    """The ordering claim, checked as thresholds rather than at three
    cherry-picked spacings: a ladder that fired out of order would still
    pass a spot check at a spacing above all three."""
    def last_true(attr: str) -> float:
        spacing = 200.0
        while spacing > 1e-4 and getattr(edge_ticks.tick_lod(spacing), attr):
            spacing -= 0.01
        return spacing

    labels_off = last_true("draw_labels")
    minors_off = last_true("draw_minors")
    edge_off = last_true("draw_edge")
    assert labels_off > minors_off > edge_off


@pytest.mark.parametrize(
    "spacing, expected",
    [
        (40.0, (True, True, True)),
        (6.0, (True, True, False)),  # majors still 30px apart, labels want 32
        (4.0, (True, False, False)),
        (0.5, (False, False, False)),
    ],
)
def test_the_ladder_at_representative_spacings(spacing: float, expected: tuple[bool, bool, bool]) -> None:
    lod = edge_ticks.tick_lod(spacing)
    assert (lod.draw_edge, lod.draw_minors, lod.draw_labels) == expected


def test_nothing_survives_a_dropped_edge() -> None:
    """draw_edge is the outer gate, so neither of the others may be True
    while it is False. A flat three-way comparison would let a coarse view
    still draw minors."""
    spacing = 20.0
    while spacing > 1e-4:
        lod = edge_ticks.tick_lod(spacing)
        if not lod.draw_edge:
            assert not lod.draw_minors and not lod.draw_labels
        spacing -= 0.01


# --- dispatch --------------------------------------------------------------


def test_edge_runs_needs_a_projection_or_a_tile_size() -> None:
    with pytest.raises(ValueError):
        edge_ticks.edge_runs(20, 16, 4)


def test_a_projection_wins_over_tile_px() -> None:
    """Stepped and Sloped both pass a proj; the flat branch is the fallback,
    matching how MapView.set_source already dispatches."""
    proj = _proj(20, 16)
    runs = edge_ticks.edge_runs(20, 16, 4, proj=proj, tile_px=8)
    assert runs[0].anchors[0] == edge_ticks.iso_corner(0, 0, proj)


def test_all_four_edges_are_present_in_a_stable_order() -> None:
    for kwargs in ({"proj": _proj(20, 16)}, {"tile_px": 8}):
        assert tuple(run.edge for run in edge_ticks.edge_runs(20, 16, 4, **kwargs)) == ("y0", "y1", "x0", "x1")


def test_a_one_tile_map_still_produces_a_ruler() -> None:
    """The degenerate size. Every edge is one tick at index 0, and it is a
    major, so nothing downstream has to special-case an empty run."""
    for kwargs in ({"proj": _proj(1, 1)}, {"tile_px": 8}):
        for run in edge_ticks.edge_runs(1, 1, 4, **kwargs):
            assert run.tiles == (0,)
            assert run.majors == (True,)
