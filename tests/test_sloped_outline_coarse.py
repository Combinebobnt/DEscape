"""GH #306: Sloped's brush-hover outline (iso_geometry.sloped_tile_outline_coarse)
is the tile's 4 projected corner tips, and stays within a measured bound of
the painted pixels' exact outline (sloped_tile_outline)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from descape import iso_geometry, settings, unit_pick

# Each ring is sampled this finely; distance is 1-Lipschitz, so a sampled max under-reads by at most half of it.
_SAMPLE_STEP = 1 / 16
_TILE_PX = (32, 64)


def _elev_step(tile_px: int, pct: int) -> int:
    return iso_geometry.canvas_size_and_origin(1, 1, tile_px, 0, 1, elev_step_pct=pct).elev_step


def _segments(ring) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(ring, dtype=float)
    return a, np.roll(a, -1, axis=0)


def _sample(ring) -> np.ndarray:
    out = []
    for p, q in zip(*_segments(ring), strict=True):
        n = max(1, int(np.ceil(np.hypot(*(q - p)) / _SAMPLE_STEP)))
        out.append(p + np.linspace(0.0, 1.0, n + 1)[:, None] * (q - p))
    return np.vstack(out)


def _max_distance(points: np.ndarray, ring) -> float:
    a, b = _segments(ring)
    ab = b - a
    length2 = (ab**2).sum(1)
    length2[length2 == 0] = 1.0
    p = points[:, None, :]
    t = np.clip(((p - a) * ab).sum(2) / length2, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return float(np.sqrt(((p - nearest) ** 2).sum(2)).min(1).max())


def _boundary_gap(ring_a, ring_b) -> float:
    """Symmetric (Hausdorff) distance between two closed outlines' boundaries."""
    return max(_max_distance(_sample(ring_a), ring_b), _max_distance(_sample(ring_b), ring_a))


def test_a_flat_tile_is_steppeds_plain_diamond() -> None:
    for tile_px in (16, 32, 64, 96):
        half_w, half_h = iso_geometry.half_dims(tile_px)
        for d in (0, 7):
            got = iso_geometry.sloped_tile_outline_coarse(tile_px, d, d, d, d)
            assert list(got) == unit_pick.diamond_points(0, 0, half_w, half_h), f"tile_px={tile_px} d={d}"


def test_a_sloped_tile_is_exactly_its_four_projected_corner_tips() -> None:
    """N, E, S, W tips, each lifted by its own -d_min-normalized corner rise."""
    tile_px = 64
    half_w, half_h = iso_geometry.half_dims(tile_px)
    nw, ne, sw, se = 5, 21, 13, 37
    got = iso_geometry.sloped_tile_outline_coarse(tile_px, nw, ne, sw, se)
    m = min(nw, ne, sw, se)
    assert got == (
        (half_w, -(ne - m)),
        (2 * half_w, half_h - (se - m)),
        (half_w, 2 * half_h - (sw - m)),
        (0, half_h - (nw - m)),
    )


@pytest.mark.parametrize(
    ("pcts", "bound"),
    [
        (tuple(p for p in settings.ELEV_STEP_PCT_STOPS if p <= 125), 1.0),
        (tuple(p for p in settings.ELEV_STEP_PCT_STOPS if p > 125), 1.8),
    ],
    ids=["pct<=125", "pct>125"],
)
def test_corner_quad_stays_within_the_measured_bound_of_the_painted_outline(pcts, bound) -> None:
    """Every one-level corner combo (the steepest legal tile under the max
    corner rule). The 1.0 is the flat diamond's own side-tip offset (column
    0 holds no pixels); at pct 200 a south-rising tile is nearly edge-on
    and _sloped_column_runs' 1px run clamp puts its sliver up to ~1.79px
    off the quad. Accepted for a display-only hover (GH #306)."""
    worst = []
    for tile_px, pct in itertools.product(_TILE_PX, pcts):
        step = _elev_step(tile_px, pct)
        for levels in itertools.product((0, 1), repeat=4):
            corners = tuple(v * step for v in levels)
            gap = _boundary_gap(
                iso_geometry.sloped_tile_outline(tile_px, *corners),
                iso_geometry.sloped_tile_outline_coarse(tile_px, *corners),
            )
            if gap > bound + 1e-9:
                worst.append((tile_px, pct, levels, round(gap, 3)))
    assert not worst, f"over {bound} source px: {worst}"
