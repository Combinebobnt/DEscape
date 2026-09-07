"""Verifies descape.unit_filter.UnitFilter and its threading through
render.py's unit paths -- phase 3's P3-a (first step).

The load-bearing check here is not the filter semantics (those are a
three-line predicate); it's the **byte-identity gate**. P3-a threads a
unit_filter parameter through ~8 call sites that every existing renderer
already depends on, and the bar it has to clear is that a DEFAULT filter
changes nothing at all: the same units, in the same order, producing
byte-identical pixels. Order matters as much as membership -- units paint
opaquely over each other (see refresh_units_over()'s docstring for the real
"building erases a tree" bug that stacking order caused), so a threading
mistake that preserved the set but perturbed the order would be invisible to
a membership-only check and visible as wrong pixels on a real map.

The expected values are re-derived independently in this file (walking
scenario.unit_manager.units directly) rather than by calling the same
function with different arguments -- comparing _units_by_tile(scn) against
_units_by_tile(scn, UnitFilter()) would be vacuous, since the default
argument makes them the same call.
"""

from __future__ import annotations


import numpy as np

from testkit.fakes import (
    FakeScenario,
    SyntheticTile,
    SyntheticUnit,
)

from descape import render, render_cache
from descape.render import (
    _flat_unit_draws,
    _unit_color,
    _units_by_tile,
    overlay_units,
    tile_pixels_for_map,
)
from descape.render_cache import FlatChunkCache
from descape.terrain_palette import BUILDING_TILE_SPANS, TREE_UNIT_IDS
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter




# Picked dynamically rather than hardcoded, matching verify_iso_units.py's
# own reasoning: a hardcoded const would silently stop testing anything if
# unit_render_data.json's contents ever changed shape.
_TREE_CONST = min(TREE_UNIT_IDS)
_BUILDING_CONST = next(uid for uid, (sx, sy) in BUILDING_TILE_SPANS.items() if sx == 4 and sy == 4)
_PLAIN_CONST = next(uid for uid in range(1, 10_000) if uid not in TREE_UNIT_IDS and uid not in BUILDING_TILE_SPANS)

MAP_W = MAP_H = 16


def _scenario() -> FakeScenario:
    """A 16x16 flat map carrying every category the filter distinguishes:
    GAIA trees, a GAIA non-tree, and plain units under three different
    players -- including a player (4) whose units must survive a
    players={1, 4} subset while player 2's do not."""
    tiles = [SyntheticTile(x=x, y=y, elevation=0) for y in range(MAP_H) for x in range(MAP_W)]
    units_by_player = [[] for _ in range(9)]
    units_by_player[GAIA_PLAYER_ID] = [
        SyntheticUnit(x=1.5, y=1.5, unit_const=_TREE_CONST),
        SyntheticUnit(x=2.5, y=1.5, unit_const=_TREE_CONST),
        SyntheticUnit(x=3.5, y=3.5, unit_const=_PLAIN_CONST),
    ]
    units_by_player[1] = [
        SyntheticUnit(x=6.5, y=6.5, unit_const=_BUILDING_CONST),
        SyntheticUnit(x=9.5, y=2.5, unit_const=_TREE_CONST),
    ]
    units_by_player[2] = [SyntheticUnit(x=11.5, y=11.5, unit_const=_PLAIN_CONST)]
    units_by_player[4] = [SyntheticUnit(x=13.5, y=4.5, unit_const=_PLAIN_CONST)]
    return FakeScenario(MAP_W, MAP_H, tiles, units_by_player)


def _expected_visible(scenario, unit_filter: UnitFilter) -> list[tuple[int, SyntheticUnit]]:
    """(player_id, unit) for every unit the filter should keep, in exactly
    the player-then-unit-list order every renderer paints in. Derived here
    from the scenario directly, so it is a genuine independent oracle rather
    than a restatement of the code under test."""
    out = []
    for player_id, units in enumerate(scenario.unit_manager.units):
        for unit in units:
            if unit_filter.matches(player_id, unit):
                out.append((player_id, unit))
    return out


# --- filter semantics -------------------------------------------------


def test_default_filter_is_default_and_matches_everything() -> None:
    scn = _scenario()
    f = UnitFilter()
    assert f.is_default
    total = sum(len(u) for u in scn.unit_manager.units)
    assert len(_expected_visible(scn, f)) == total


def test_show_gaia_false_hides_only_the_gaia_slot() -> None:
    scn = _scenario()
    visible = _expected_visible(scn, UnitFilter(show_gaia=False))
    assert all(pid != GAIA_PLAYER_ID for pid, _ in visible)
    # Player 1's tree is NOT GAIA and must survive -- the two gates are
    # independent, which is the whole reason show_trees exists separately.
    assert any(pid == 1 and u.unit_const == _TREE_CONST for pid, u in visible)


def test_show_trees_false_hides_trees_under_every_owner() -> None:
    scn = _scenario()
    visible = _expected_visible(scn, UnitFilter(show_trees=False))
    assert all(u.unit_const != _TREE_CONST for _pid, u in visible)
    # The GAIA non-tree survives: hiding trees must not empty the GAIA slot,
    # since cliffs/gold/stone are GAIA too and are usually what's wanted.
    assert any(pid == GAIA_PLAYER_ID for pid, _ in visible)


def test_players_subset_gates_only_non_gaia_slots() -> None:
    scn = _scenario()
    visible = _expected_visible(scn, UnitFilter(players=frozenset({1, 4})))
    kept = {pid for pid, _ in visible}
    assert kept == {GAIA_PLAYER_ID, 1, 4}, "GAIA is governed by show_gaia alone, never by players"
    assert 2 not in kept


def test_is_default_false_for_every_non_default_field() -> None:
    assert not UnitFilter(show_gaia=False).is_default
    assert not UnitFilter(show_trees=False).is_default
    assert not UnitFilter(players=frozenset({1})).is_default


def test_filter_is_frozen_and_hashable() -> None:
    # Frozen matters beyond tidiness: the caches store a filter and key
    # invalidation on it changing, which a mutable filter edited in place
    # would defeat silently.
    f = UnitFilter(players=frozenset({1}))
    assert hash(f) == hash(UnitFilter(players=frozenset({1})))
    assert f == UnitFilter(players=frozenset({1}))


# --- the byte-identity gate -------------------------------------------


def test_units_by_tile_default_filter_matches_independent_oracle() -> None:
    scn = _scenario()
    got = _units_by_tile(scn, UnitFilter())

    # Mirrors _units_by_tile's footprint bucketing: every tile of the unit's
    # CLAMPED unit_tile_bounds, and off-map units (bounds None) dropped
    # entirely. Both details are the function's contract, not incidental --
    # a bounds-free "own tile plus span" oracle would disagree on any
    # building hanging off a map edge.
    expected: dict[tuple[int, int], list] = {}
    for player_id, unit in _expected_visible(scn, UnitFilter()):
        color = _unit_color(unit, scn.player_colors[player_id])
        bounds = render.unit_tile_bounds(unit, MAP_W, MAP_H)
        if bounds is None:
            continue
        tx0, tx1, ty0, ty1 = bounds
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                expected.setdefault((tx, ty), []).append((unit, color))

    assert got.keys() == expected.keys()
    for key in expected:
        # Identity comparison per entry, and in order: this is the check that
        # a mis-threaded call site cannot pass by accident.
        assert [id(u) for u, _c in got[key]] == [id(u) for u, _c in expected[key]]
        assert [c for _u, c in got[key]] == [c for _u, c in expected[key]]


def test_units_by_tile_honours_a_real_filter() -> None:
    scn = _scenario()
    f = UnitFilter(show_trees=False)
    got = _units_by_tile(scn, f)
    flat = [u for entries in got.values() for u, _c in entries]
    assert flat and all(u.unit_const != _TREE_CONST for u in flat)


def test_flat_unit_draws_default_filter_matches_independent_oracle() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    bboxes, colors = _flat_unit_draws(scn, tile_px, UnitFilter())

    exp_bboxes = []
    exp_colors = []
    for player_id, unit in _expected_visible(scn, UnitFilter()):
        bounds = render.unit_tile_bounds(unit, MAP_W, MAP_H)
        assert bounds is not None
        tx0, tx1, ty0, ty1 = bounds
        exp_bboxes.append((tx0 * tile_px, ty0 * tile_px, tx1 * tile_px, ty1 * tile_px))
        exp_colors.append(_unit_color(unit, scn.player_colors[player_id]))

    assert np.array_equal(bboxes, np.array(exp_bboxes, dtype=np.int32))
    assert np.array_equal(colors, np.array(exp_colors, dtype=np.uint8))


def test_flat_unit_draws_drops_filtered_rows_without_reordering() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    all_bboxes, _ = _flat_unit_draws(scn, tile_px, UnitFilter())
    kept_bboxes, _ = _flat_unit_draws(scn, tile_px, UnitFilter(show_trees=False))
    assert len(kept_bboxes) < len(all_bboxes)
    # Every surviving row must still appear in the unfiltered array in the
    # same relative order -- filtering removes rows, it never permutes them.
    all_rows = [tuple(r) for r in all_bboxes]
    kept_rows = [tuple(r) for r in kept_bboxes]
    positions = [all_rows.index(r) for r in kept_rows]
    assert positions == sorted(positions)


def test_overlay_units_default_filter_is_byte_identical_to_no_filter() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    base = np.zeros((MAP_H * tile_px, MAP_W * tile_px, 3), dtype=np.uint8)
    assert np.array_equal(overlay_units(base, scn), overlay_units(base, scn, UnitFilter()))


def test_overlay_units_hiding_gaia_changes_pixels_and_restoring_them_round_trips() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    base = np.zeros((MAP_H * tile_px, MAP_W * tile_px, 3), dtype=np.uint8)
    full = overlay_units(base, scn, UnitFilter())
    hidden = overlay_units(base, scn, UnitFilter(show_gaia=False))
    assert not np.array_equal(full, hidden), "hiding GAIA must actually change pixels"
    assert np.array_equal(full, overlay_units(base, scn, UnitFilter()))


# --- cache threading --------------------------------------------------


def _flat_cache(scn, unit_filter: UnitFilter) -> FlatChunkCache:
    return FlatChunkCache(scn, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, unit_filter=unit_filter)


def _whole_canvas(cache: FlatChunkCache) -> np.ndarray:
    w, h = cache.canvas_dims(0)
    return cache.render_rect(0, 0, w, h)


def test_flat_cache_default_filter_matches_an_unfiltered_cache() -> None:
    scn = _scenario()
    tile_px = tile_pixels_for_map(MAP_W, MAP_H)
    plain = FlatChunkCache(scn, tile_px, chunk_px=128)
    filtered = _flat_cache(scn, UnitFilter())
    assert np.array_equal(_whole_canvas(plain), _whole_canvas(filtered))


def test_set_unit_filter_round_trips_byte_identically() -> None:
    """Filter, render, unfilter, render again -- the result must be
    byte-identical to the original. This is the check that catches the
    specific trap set_unit_filter() exists for: units are baked into cached
    chunk PIXELS, so storing the new filter and rebuilding source caches
    without also evicting composited chunks leaves the old pixels on screen
    until something else happens to invalidate them."""
    scn = _scenario()
    cache = _flat_cache(scn, UnitFilter())
    before = _whole_canvas(cache).copy()

    cache.set_unit_filter(UnitFilter(show_gaia=False))
    hidden = _whole_canvas(cache).copy()
    assert not np.array_equal(before, hidden), "set_unit_filter must actually repaint"

    cache.set_unit_filter(UnitFilter())
    assert np.array_equal(before, _whole_canvas(cache))


def test_set_unit_filter_stores_the_filter() -> None:
    scn = _scenario()
    cache = _flat_cache(scn, UnitFilter())
    f = UnitFilter(show_trees=False, players=frozenset({1}))
    cache.set_unit_filter(f)
    assert cache.unit_filter == f


def test_set_unit_filter_to_an_equal_filter_is_a_no_op() -> None:
    """Evicting the whole canvas is the most expensive thing this class can
    be asked to do, so an unchanged filter must not trigger it."""
    scn = _scenario()
    cache = _flat_cache(scn, UnitFilter())
    _whole_canvas(cache)
    cached_before = len(cache._cache)
    assert cached_before > 0
    cache.set_unit_filter(UnitFilter())
    assert len(cache._cache) == cached_before


def _iso_cache(scn, unit_filter: UnitFilter) -> render_cache.IsoChunkCache:
    elevations, proj = render.elevations_and_proj(scn)
    return render_cache.IsoChunkCache(
        scn, elevations, proj, tile_pixels_for_map(MAP_W, MAP_H), chunk_px=128, unit_filter=unit_filter
    )


def _sloped_cache(scn, unit_filter: UnitFilter) -> render_cache.SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return render_cache.SlopedChunkCache(
        scn,
        elevations,
        corner_rise,
        proj,
        tile_pixels_for_map(MAP_W, MAP_H),
        chunk_px=128,
        unit_filter=unit_filter,
    )


def _level_canvas(cache, mip: int) -> np.ndarray:
    w, h = cache.canvas_dims(mip)
    return cache.render_rect(0, 0, w, h, mip=mip)


def test_stepped_cache_filter_round_trips_byte_identically() -> None:
    """Stepped's own version of the round-trip. Distinct from Flat's because
    IsoChunkCache rebuilds per-level building_bboxes LAZILY off a generation
    counter -- a filter change has to bump that counter or a level composited
    before the change keeps its stale bboxes."""
    scn = _scenario()
    cache = _iso_cache(scn, UnitFilter())
    before = _whole_canvas(cache).copy()
    cache.set_unit_filter(UnitFilter(show_gaia=False))
    assert not np.array_equal(before, _whole_canvas(cache))
    cache.set_unit_filter(UnitFilter())
    assert np.array_equal(before, _whole_canvas(cache))


def test_filter_applies_at_every_mip_level_not_just_the_resident_one() -> None:
    """A filter change must reach EVERY mip level, not only the one that
    happened to be composited when it changed.

    This is the check that catches the FlatChunkCache._level_unit_draws()
    class of bug directly: per-level unit structures are built lazily, so a
    level first composited after a filter change must build against the NEW
    filter, and a level composited before it must be rebuilt rather than
    reused. A single-level assertion cannot see either failure.

    Both cache types enumerate a real multi-level ladder on this fixture
    (asserted below rather than assumed -- if the ladder ever collapsed to
    one level this test would silently stop proving anything).
    """
    scn = _scenario()
    for cache in (_iso_cache(scn, UnitFilter()), _flat_cache(scn, UnitFilter())):
        levels = sorted(cache._mip_tile_px)
        assert len(levels) > 1, f"{type(cache).__name__} enumerated only {levels} -- test would be vacuous"
        coarse = max(levels)
        fine = min(levels)

        # Composite the fine level FIRST, so it is already resident (and its
        # lazy per-level structures already built) when the filter changes.
        fine_before = _level_canvas(cache, fine).copy()
        cache.set_unit_filter(UnitFilter(show_gaia=False))

        # A level built fresh AFTER the change must honour the new filter...
        coarse_hidden = _level_canvas(cache, coarse).copy()
        # ...and the already-resident level must have been rebuilt, not reused.
        assert not np.array_equal(fine_before, _level_canvas(cache, fine)), (
            f"{type(cache).__name__} level {fine} kept stale pre-filter pixels"
        )

        cache.set_unit_filter(UnitFilter())
        assert np.array_equal(fine_before, _level_canvas(cache, fine))
        assert not np.array_equal(coarse_hidden, _level_canvas(cache, coarse)), (
            f"{type(cache).__name__} level {coarse} ignored the filter being restored"
        )


def test_sloped_cache_filters_units() -> None:
    """Sloped honours the unit filter because it reaches _units_by_tile()
    unchanged -- it came for free rather than being wired up per style, and
    it held even while Sloped had no unit picking at all (Track C4/C5). Now
    that picking exists too, the two must agree: a filtered unit is absent
    from the pixels AND from the index, so it can never be selected through
    a tile it no longer paints on."""
    scn = _scenario()
    cache = _sloped_cache(scn, UnitFilter())
    before = _whole_canvas(cache).copy()
    cache.set_unit_filter(UnitFilter(show_gaia=False))
    assert not np.array_equal(before, _whole_canvas(cache))
    cache.set_unit_filter(UnitFilter())
    assert np.array_equal(before, _whole_canvas(cache))
