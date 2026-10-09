"""Undo of a large Elevate / Set elevation (2026-09-30 undo-large-elevation
plan). Its Step 0 probe dropped the separate in-op splice cap (an uncapped
splice cost ~0.9x the wholesale rebuild, not <= 0.5x) and chose two changes:

- render._dirty_screen_bbox() reads a dirty set of BBOX_ARRAY_PATH_MIN_TILES
  or more as index arrays (_dirty_seed_mask). It must give exactly the loop's
  rect, elevation_changed set and written elevations, on Stepped's radii and
  Sloped's, with and without sprites, units_changed and flatten.
- A patch() whose elevation splice goes wholesale says so on the Perf Trace
  op line: `elev_splice_refused=<members>/<cap>`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from test_elevation_unit_splice import BASE_ELEVATION, _place
from test_invalidate_units_splice import MILL_CONST, WALL_CONST, traced  # noqa: F401 -- traced is a fixture

from descape import debug_log, perf_trace, render, render_cache
from descape.render import _canvas_pixel_dims, dirty_screen_bbox_iso, elevations_and_proj, tile_pixels_for_map
from descape.render_cache import IsoChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

from test_unit_sprites import CONST as SPRITE_CONST

LOOP, ARRAY = 10**9, 0


def _bumpy_scenario(seed: int):
    """The blank template with random elevations and units of three spans,
    some on the map edge, so footprints and anchors meet the dirty sets."""
    rng = random.Random(seed)
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for t in mm.terrain:
        t.elevation = rng.randrange(0, 8)
    w, h = mm.map_width, mm.map_height
    for k in range(120):
        const = (SPRITE_CONST, MILL_CONST, WALL_CONST)[k % 3]
        x, y = rng.randrange(0, w), rng.randrange(0, h)
        _place(scenario, 1 + k % 2, const, x + 0.5, y + 0.5)
    for x, y in ((0, 0), (w - 1, h - 1), (0, h - 1)):
        _place(scenario, 1, MILL_CONST, x + 0.5, y + 0.5)
    return scenario


def _dirty_sets(mm, rng: random.Random) -> list[list[int]]:
    """A brush-like square, a scattered set with duplicates, and a strip along
    two map edges, each at least 600 tiles."""
    w, h = mm.map_width, mm.map_height
    x0, y0 = rng.randrange(0, w - 30), rng.randrange(0, h - 30)
    square = [y * w + x for y in range(y0, y0 + 30) for x in range(x0, x0 + 30)]
    scattered = [rng.randrange(0, w * h) for _ in range(700)]
    scattered += scattered[:50]
    edges = list(range(w)) + [y * w for y in range(h)] + [(h - 1) * w + x for x in range(w)]
    edges += [y * w + w - 1 for y in range(h)]
    return [square, scattered, edges]


def _edit(mm, dirty, rng: random.Random, top: int = 15) -> None:
    """Moves about two thirds of `dirty`'s elevations; the rest stay (terrain-only tiles)."""
    for i in set(dirty):
        if rng.random() < 0.67:
            mm.terrain[i].elevation = rng.randrange(0, top + 1)


def _run(monkeypatch, threshold, scenario, dirty, pre, proj, **kw):
    monkeypatch.setattr(render, "BBOX_ARRAY_PATH_MIN_TILES", threshold)
    elevations = pre.copy()
    changed: set = set()
    rect = render._dirty_screen_bbox(
        scenario, dirty, elevations, proj, _canvas_pixel_dims(proj), elevation_changed=changed, **kw
    )
    return rect, changed, elevations


@pytest.mark.parametrize("radius", [0, 1], ids=["stepped", "sloped"])
@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"with_sprites": True},
        {"with_sprites": True, "units_changed": True, "extra_anchor_tiles": {(3, 3), (50, 60)}},
        {"with_sprites": True, "flatten_elevations": True, "units_changed": True},
        {"with_units": False},
    ],
    ids=["plain", "sprites", "units_changed", "flatten", "no_units"],
)
@pytest.mark.parametrize("seed", [1, 2])
def test_the_array_path_matches_the_loop_exactly(monkeypatch, radius, flags, seed):
    scenario = _bumpy_scenario(seed)
    mm = scenario.map_manager
    rng = random.Random(seed * 101)
    pre, proj = elevations_and_proj(scenario)
    radii = {"sprite_band_radius": radius, "unit_band_radius": radius}
    for dirty in _dirty_sets(mm, rng):
        _edit(mm, dirty, rng)
        loop = _run(monkeypatch, LOOP, scenario, dirty, pre, proj, **radii, **flags)
        array = _run(monkeypatch, ARRAY, scenario, dirty, pre, proj, **radii, **flags)
        assert loop[0] is not None
        assert array[0] == loop[0]
        assert array[1] == loop[1]
        assert all(type(t[0]) is int and type(t[1]) is int for t in array[1])
        assert np.array_equal(array[2], loop[2])
        pre = loop[2]


def test_an_out_of_range_tile_refuses_on_both_paths_after_the_same_writes(monkeypatch):
    scenario = _bumpy_scenario(3)
    mm = scenario.map_manager
    pre, proj = elevations_and_proj(scenario)
    dirty = list(range(1000))
    for i in dirty[::3]:
        mm.terrain[i].elevation += 1
    mm.terrain[dirty[500]].elevation = proj.max_elev + 1
    loop = _run(monkeypatch, LOOP, scenario, dirty, pre, proj)
    array = _run(monkeypatch, ARRAY, scenario, dirty, pre, proj)
    assert loop[0] is None and array[0] is None
    assert array[1] == loop[1] and len(loop[1]) > 300
    assert np.array_equal(array[2], loop[2])


def test_the_array_path_starts_at_its_threshold(monkeypatch):
    scenario = _bumpy_scenario(4)
    pre, proj = elevations_and_proj(scenario)
    calls = []
    real = render._dirty_seed_mask

    def spy(*args, **kwargs):
        calls.append(len(args[1]))
        return real(*args, **kwargs)

    monkeypatch.setattr(render, "_dirty_seed_mask", spy)
    n = render.BBOX_ARRAY_PATH_MIN_TILES
    for size in (n - 1, n):
        dirty_screen_bbox_iso(scenario, list(range(size)), pre.copy(), proj)
    assert calls == [n]


# --- the refusal trace ------------------------------------------------------


def _unit_cache(monkeypatch, cap: int):
    """A Stepped cache over four lone units on raised ground, cap patched."""
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", cap)
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    mm = scenario.map_manager
    for t in mm.terrain:
        t.elevation = BASE_ELEVATION
    tiles = [(30, 30), (34, 30), (38, 30), (42, 30)]
    for x, y in tiles:
        _place(scenario, 1, SPRITE_CONST, x + 0.5, y + 0.5)
    elevations, proj = elevations_and_proj(scenario)
    cache = IsoChunkCache(scenario, elevations, proj, tile_pixels_for_map(mm.map_width, mm.map_height))
    cache.render_rect(0, 0, 4096, 4096, mip=0)
    return scenario, cache, tiles


def _undo_line(scenario, cache, tiles) -> str:
    mm = scenario.map_manager
    with perf_trace.op("Undo"):
        for x, y in tiles:
            mm.get_tile(x, y).elevation += 1
        dirty = [y * mm.map_width + x for x, y in tiles]
        changed: set = set()
        bbox = dirty_screen_bbox_iso(scenario, dirty, cache.elevations, cache.proj, elevation_changed=changed)
        cache.patch(bbox, elevation_changed=changed, rebuild_levels=(0,))
    perf_trace.flush_idle()
    return next(line for line in debug_log.get_log_text().splitlines() if "perf op undo" in line)


def test_a_refused_elevation_splice_says_so_on_the_op_line(monkeypatch, traced):  # noqa: F811
    scenario, cache, tiles = _unit_cache(monkeypatch, cap=2)
    gen = cache._source_gen
    line = _undo_line(scenario, cache, tiles)
    assert cache._source_gen == gen + 1, "the patch should have gone wholesale"
    assert "elev_splice_refused=4/2" in line, line


def test_a_spliced_elevation_patch_reports_no_refusal(monkeypatch, traced):  # noqa: F811
    scenario, cache, tiles = _unit_cache(monkeypatch, cap=4)
    gen = cache._source_gen
    line = _undo_line(scenario, cache, tiles)
    assert cache._source_gen == gen, "the patch should have spliced"
    assert "elev_splice_refused" not in line, line
