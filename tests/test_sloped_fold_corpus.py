"""Re-measures iso_geometry.tile_folds() against the game (TASK-008, GH #62).

A unit's stored `z` is the game's own ground height at its point (exact on
137,571 of 137,618 flat-tile corpus placements, the rest stale). On a
non-planar Sloped tile the two folds give different heights at most points,
so every placement whose `z` matches exactly one fold says which one the
game used there. The rule is a fit, not a derivation: a counterexample here
means the rule changes, not this test. Run with -s for the coverage line.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from descape import batch_api, iso_geometry

pytestmark = pytest.mark.corpus


def _height(c: tuple[int, int, int, int], p: Fraction, q: Fraction, nwse: bool) -> Fraction:
    nw, ne, sw, se = c
    if nwse:
        if p >= q:
            return nw + p * (ne - nw) + q * (se - ne)
        return nw + q * (sw - nw) + p * (se - sw)
    if p + q <= 1:
        return nw + p * (ne - nw) + q * (sw - nw)
    return se - (1 - p) * (se - sw) - (1 - q) * (se - ne)


def test_stored_z_matches_the_tile_fold(corpus_files):
    deciding = neither = 0
    bad = []
    for path in corpus_files:
        scenario = batch_api.load(str(path))
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        elevations = np.zeros((h, w), dtype=np.int64)
        for t in mm.terrain:
            elevations[t.y, t.x] = t.elevation
        # Max-rule corners in levels: corner_rise_px reads only elev_step.
        corners = iso_geometry.corner_rise_px(elevations, SimpleNamespace(elev_step=1), "max")
        folds = iso_geometry.tile_folds(elevations)
        for units in scenario.unit_manager.units:
            for u in units:
                x, y = int(u.x), int(u.y)
                if not (0 <= x < w and 0 <= y < h):
                    continue
                c = (int(corners[y, x]), int(corners[y, x + 1]), int(corners[y + 1, x]), int(corners[y + 1, x + 1]))
                if c[0] + c[3] == c[1] + c[2]:
                    continue
                p, q = Fraction(u.x) - x, Fraction(u.y) - y
                hit = {nwse: abs(float(_height(c, p, q, nwse)) - u.z) < 0.01 for nwse in (False, True)}
                if hit[False] == hit[True]:
                    neither += not hit[False]
                    continue
                deciding += 1
                if not hit[bool(folds[y, x])]:
                    bad.append((path.name, x, y, u.x, u.y, u.z))
    print(f"\nfold-deciding placements: {deciding}, matching neither fold (stale z): {neither}")
    assert deciding > 0, "no corpus placement decides the fold"
    assert not bad, f"{len(bad)} placements contradict tile_folds(), first: {bad[:5]}"
