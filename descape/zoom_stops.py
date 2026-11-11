"""The wheel zoom's notch lattice (TASK-600): every zoom level's own 1:1
scale is a point, so a notch can land where a level is drawn with no second
resample. Pure, no Qt. Scales here are device scales, the quantity
`mip_for_scale()` reads: a level `L` is 1:1 at `1 / mip_scale(L)`.

Each gap between neighbouring level stops, ratio `r`, is split into
`max(1, round(log r / log NOTCH_STEP))` equal geometric steps; past the
ladder's ends the lattice continues by octaves split the same way, out to the
floor and ceiling. A view off the lattice (after a fit or a resize) moves to
the first point at least MIN_OFF_LATTICE_STEP away in the wheel's direction.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Iterable, Sequence

NOTCH_STEP = 1.25
MIN_OFF_LATTICE_STEP = NOTCH_STEP**0.5
# Relative distance under which a scale is on a point: well above a few float multiplies' drift.
ON_POINT_TOL = 1e-9


def _split(ratio: float, step: float) -> int:
    return max(1, round(math.log(ratio) / math.log(step)))


def level_stops(ratios: Iterable[float], slack: float = ON_POINT_TOL) -> list[float]:
    """The device scales where each level is 1:1 (`ratios`: each level's
    `1 / mip_scale(L)`), sorted, values within `slack` of each other merged."""
    stops: list[float] = []
    for r in sorted(r for r in ratios if r > 0):
        if not stops or r > stops[-1] * (1.0 + slack):
            stops.append(r)
    return stops


def lattice(stops: Sequence[float], lo: float, hi: float, step: float = NOTCH_STEP) -> list[float]:
    """Every notch point in [lo, hi]: the stops themselves exactly, each gap
    split evenly, and octave extensions past either end. Empty with no stops."""
    if not stops:
        return []
    octave = _split(2.0, step)
    points: list[float] = []
    # Below the coarsest stop, by octaves.
    k = 1
    while stops[0] * 2.0 ** (-k / octave) >= lo * (1.0 - ON_POINT_TOL):
        points.append(stops[0] * 2.0 ** (-k / octave))
        k += 1
    points.reverse()
    for a, b in itertools.pairwise(stops):
        n = _split(b / a, step)
        points.append(a)
        points.extend(a * (b / a) ** (i / n) for i in range(1, n))
    points.append(stops[-1])
    k = 1
    while stops[-1] * 2.0 ** (k / octave) <= hi * (1.0 + ON_POINT_TOL):
        points.append(stops[-1] * 2.0 ** (k / octave))
        k += 1
    return [p for p in points if lo * (1.0 - ON_POINT_TOL) <= p <= hi * (1.0 + ON_POINT_TOL)]


def on_point(scale: float, points: Sequence[float]) -> int | None:
    """The index of the point `scale` sits on (within ON_POINT_TOL), or None."""
    i = bisect.bisect_left(points, scale)
    for j in (i - 1, i):
        if 0 <= j < len(points) and abs(scale / points[j] - 1.0) <= ON_POINT_TOL:
            return j
    return None


def notch_target(current: float, notches: int, points: Sequence[float]) -> float | None:
    """The scale `notches` points away from `current` (positive: in). On a
    point, the first move is to its neighbour; off one, to the first point
    at least MIN_OFF_LATTICE_STEP away. None when the walk runs past the last
    point that way (the caller lands on the bound) or `notches` is 0."""
    if notches == 0 or not points:
        return None
    here = on_point(current, points)
    if notches > 0:
        if here is not None:
            first = here + 1
        else:
            first = bisect.bisect_left(points, current * MIN_OFF_LATTICE_STEP * (1.0 - ON_POINT_TOL))
        index = first + notches - 1
        return points[index] if index < len(points) else None
    if here is not None:
        first = here - 1
    else:
        first = bisect.bisect_right(points, current / MIN_OFF_LATTICE_STEP * (1.0 + ON_POINT_TOL)) - 1
    index = first + notches + 1
    return points[index] if index >= 0 else None
