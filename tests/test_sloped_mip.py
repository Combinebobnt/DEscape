"""Sloped's mip ladder: SlopedChunkCache enumerates Stepped's
mip_projections_for() set unfiltered (the coarser levels since TASK-031.60,
the finer ones since TASK-146), with each level's corner_rise and headroom
recomputed at its own proj.

Every pixel oracle compares a level's stitched chunks against an independent
render: a full render_terrain_sloped_with_proj() with tile_pixels_for_map()
forced to that level's tile_px, computed after the cache side so the patch
cannot reach the cache (tests/test_mip_geometry.py's non-tautology rule).
The fixture is a low-amplitude bumpy field painted as a BLACK / DESERT_SAND
checkerboard, the demanding case for Sloped seams and tile ownership.

The edit oracles (patch vs fresh, the deferred flush) compare a level other
than 0 against a fresh cache's same level after each step, pixels and unit
layers, with the whole level rendered right before each checked edit so no
evicted chunk can redraw fresh and hide a stale one; a control that breaks
only step k must fail naming step k. A +1 level is 4x the level-0 canvas the
LRU is sized to, so every +1 edit oracle runs on the 20x20 FIELD sites, never
the 120x120 template, with max_bytes raised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from test_elevation_unit_splice import _deferred_cases, _raise_edit, _sloped_bbox, _source_state
from test_invalidate_units_splice import (
    MILL_CONST,
    _add_splice,
    _delete_splice,
    _place,
    shared_art_install,  # noqa: F401 -- a fixture
    traced,  # noqa: F401 -- a fixture
)
from test_sloped_sprites import FARM_CONST, Unit, sprite_install  # noqa: F401 -- a fixture

from descape import iso_geometry, render, render_cache, settings
from descape.elevation_tools import set_tiles_elevation
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit.fakes import FakeMapManager, FakeScenario, SyntheticTile

from test_unit_sprites import CONST

MAP_W = MAP_H = 20
BLACK, DESERT_SAND = 47, 14
MARK_CONST = CONST + 1  # no art in sprite_install, so it paints its mark


def _bumpy(x: int, y: int) -> int:
    """test_sloped_pick's blocked one-level field: runs of equal and of differing corners."""
    return (x // 2 + y // 3) % 2


def _scenario(units: bool = True, elevation_of=_bumpy) -> FakeScenario:
    tiles = [
        SyntheticTile(x=x, y=y, elevation=elevation_of(x, y), terrain_id=BLACK if (x + y) % 2 else DESERT_SAND)
        for y in range(MAP_H)
        for x in range(MAP_W)
    ]
    players = [[] for _ in range(9)]
    if units:
        players[1] = [
            Unit(x=6.5, y=7.5, unit_const=CONST),
            Unit(x=12.3, y=5.8, unit_const=CONST),
            Unit(x=9.5, y=12.5, unit_const=MARK_CONST),
            Unit(x=15.5, y=15.5, unit_const=FARM_CONST),
        ]
    return FakeScenario(MAP_W, MAP_H, tiles, players)


def _cache(scn, **kwargs) -> SlopedChunkCache:
    elevations, corner_rise, proj = render.sloped_elevations_and_proj(scn)
    return SlopedChunkCache(scn, elevations, corner_rise, proj, proj.tile_px, **kwargs)


def _whole(cache, mip: int) -> np.ndarray:
    return cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)


def _forced_render(scn, tile_px: int, sprites: bool):
    """(img, corner_rise, proj) of a full render at tile_px, the patch scoped to this call."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(render, "tile_pixels_for_map", lambda _w, _h: tile_px)
        img, _elevations, corner_rise, proj = render.render_terrain_sloped_with_proj(scn, with_sprites=sprites)
    return img, corner_rise, proj


def _assert_same(got: np.ndarray, want: np.ndarray, what: str) -> None:
    if not np.array_equal(got, want):
        ys, xs = np.nonzero((got != want).any(axis=2))
        pytest.fail(f"{what}: {len(xs)} px differ, x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}")


# --- 1. each level against a full render at its own tile_px ---------------


@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
def test_each_level_matches_a_full_render_at_its_own_tile_px(sprite_install, sprites) -> None:  # noqa: F811
    scn = _scenario()
    cache = _cache(scn, sprites=sprites)
    assert cache.mip_levels() == [-2, -1, 0, 1]
    stitched = {mip: _whole(cache, mip) for mip in cache.mip_levels()}
    for mip, got in stitched.items():
        lvl = cache._levels[mip]
        truth, corner_rise, proj = _forced_render(scn, cache.mip_tile_px(mip), sprites)
        assert proj == lvl.proj
        assert np.array_equal(corner_rise, lvl.corner_rise)
        h, w = got.shape[:2]
        _assert_same(got, truth[:h, :w], f"level {mip}")
        if sprites:
            assert lvl.sprites.by_anchor and lvl.sprites.farm_by_tile, f"level {mip}: no sprite or farm -- vacuous"


def test_the_levels_really_differ(sprite_install) -> None:  # noqa: F811
    """Negative control for the oracle above: a level is not level 0 in disguise."""
    cache = _cache(_scenario(), sprites=True)
    shapes = {mip: _whole(cache, mip).shape for mip in cache.mip_levels()}
    assert len(set(shapes.values())) == len(shapes)


# --- 2. recomputed per level, never scaled --------------------------------


@pytest.mark.parametrize("rule", ["max", "average"])
def test_each_level_recomputes_its_corner_rise_and_headroom_at_its_own_proj(monkeypatch, rule) -> None:
    """Values under both rules, and the spy: level steps are exact powers of
    two, so level 0 floor-divided down agrees with the recompute under `max`
    and `average` alike (floor(floor(a)/n) == floor(a/n)), and only the spy
    tells recomputing from scaling. It runs once per rebinding of level 0's
    corner_rise."""
    monkeypatch.setattr(render, "SLOPE_CORNER_RULE", rule)
    scn = _scenario(units=False, elevation_of=lambda x, y: (x * 7 + y * 3) % 3 // 2)
    cache = _cache(scn)
    calls: list[int] = []
    real = iso_geometry.corner_rise_px

    def spy(elevations, proj, rule="average"):
        calls.append(proj.tile_px)
        return real(elevations, proj, rule=rule)

    monkeypatch.setattr(iso_geometry, "corner_rise_px", spy)
    for mip in cache.mip_levels():
        lvl = cache._levels[mip]
        rise = cache._level_corner_rise(lvl)
        assert np.array_equal(rise, real(cache.elevations, lvl.proj, rule=rule)), f"level {mip}"
        assert cache._level_extra_top(lvl) == render._unit_rise_headroom_px(rise, cache.elevations, lvl.proj)
    assert cache._level_corner_rise(cache._levels[0]) is cache.corner_rise
    assert sorted(calls) == [16, 32, 128], "level 0 shares self.corner_rise; every other level computes its own"
    assert cache._levels[-1].headroom > 0, "a flat field has headroom 0 at every level -- vacuous"

    for mip in cache.mip_levels():
        cache._level_corner_rise(cache._levels[mip])
    assert len(calls) == 3, "re-derived without level 0's corner_rise changing"

    cache.elevations[3, 3] += 1
    cache.patch((0, 0, *cache.canvas_dims(0)), elevation_changed={(3, 3)})
    for mip in cache.mip_levels():
        lvl = cache._levels[mip]
        assert np.array_equal(cache._level_corner_rise(lvl), real(cache.elevations, lvl.proj, rule=rule))


# --- 3. canvas dims --------------------------------------------------------


@pytest.mark.parametrize("sprites", [False, True], ids=["marks", "sprites"])
def test_every_level_canvas_is_an_exact_mip_of_level_0(sprites) -> None:
    cache = _cache(_scenario(units=False), sprites=sprites)
    ref_w, ref_h = cache.canvas_dims(0)
    t0 = cache.mip_tile_px(0)
    for mip in cache.mip_levels():
        proj = cache._levels[mip].proj
        w, h = cache.canvas_dims(mip)
        assert (w, h) == (render._canvas_pixel_dims(proj) if sprites else (proj.canvas_w, proj.canvas_h))
        t = cache.mip_tile_px(mip)
        assert w * t0 == ref_w * t and h * t0 == ref_h * t, f"level {mip}: {(w, h)} vs {(ref_w, ref_h)}"


@pytest.mark.parametrize("mip", [-2, 1])
def test_a_sprite_toggle_while_another_level_is_shown_paints_its_grown_canvas(sprite_install, mip) -> None:  # noqa: F811
    """Sprites on grows the canvas (Stepped's skirt-padded height). The toggle
    rebuilds the resident level eagerly, leaves level 0 unbuilt, and the grown
    strip at `mip` must be painted, not left black."""
    scn = _scenario()
    cache = _cache(scn, sprites=False)
    small = cache.canvas_dims(mip)
    _whole(cache, mip)
    cache.set_sprites_enabled(True)
    assert cache.canvas_dims(mip)[1] > small[1]
    assert cache._levels[mip].gen == cache._source_gen, "the resident level was not warmed under the cursor"
    assert cache._levels[0].gen != cache._source_gen, "level 0 was built though nothing showed it"
    _assert_same(_whole(cache, mip), _whole(_cache(scn, sprites=True), mip), "after the toggle")


# --- 7. small tile_px: the ladder per pct, and the column-run clamp ---------


def _flat_scenario(size: int) -> FakeScenario:
    tiles = [SyntheticTile(x=x, y=y, elevation=0, terrain_id=DESERT_SAND) for y in range(size) for x in range(size)]
    return FakeScenario(size, size, tiles, [[] for _ in range(9)])


@pytest.mark.parametrize("size, pct", [(MAP_W, 25), (MAP_W, 50), (MAP_W, 100), (MAP_W, 200), (252, 100)])
def test_the_ladder_is_steppeds_at_every_pct(monkeypatch, size, pct) -> None:
    """Plain equality with Stepped's ladder (TASK-146). A 252-wide map has a
    32 px base, so it gets two finer levels, +1 and +2."""
    monkeypatch.setattr(settings, "_elev_step_pct", pct)
    scn = _scenario(units=False) if size == MAP_W else _flat_scenario(size)
    elevations, proj = render.elevations_and_proj(scn)
    stepped = IsoChunkCache(scn, elevations, proj, proj.tile_px).mip_levels()
    cache = _cache(scn)
    assert cache.mip_levels() == stepped
    assert min(stepped) < 0 < max(stepped), f"pct {pct}: no coarser or no finer level -- vacuous"
    if size != MAP_W:
        assert proj.tile_px == 32 and {1, 2} <= set(stepped), (proj.tile_px, stepped)


def _footprint(tx: int, ty: int, proj, corner_rise) -> tuple[int, int, int, int]:
    """Tile (tx, ty)'s painted pixel bbox, through _render_tile_sloped's own placement."""
    corners = (corner_rise[ty, tx], corner_rise[ty, tx + 1], corner_rise[ty + 1, tx], corner_rise[ty + 1, tx + 1])
    base_x, base_y, dst_y, dst_x, *_rest = render._sloped_tile_quad(tx, ty, proj.tile_px, proj, *map(int, corners))
    rows, cols = base_y + dst_y, base_x + dst_x
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _inside(inner, outer) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


@pytest.mark.parametrize("pct", [25, 50, 100, 200])
def test_a_coarser_tile_never_paints_outside_its_finer_twins_scaled_footprint(monkeypatch, pct) -> None:
    """_sloped_column_runs' max(_, 1) clamp lengthens a run at small tile_px.
    Of a tile's footprints at level 0 and level L, the coarser must sit inside
    the finer converted to its pixels: by _bbox_to_level() for L < 0 (the
    conversion every patch bbox takes), by _level_bbox_to_reference() for
    L > 0 (the one a level extent takes back to scene space). A finer tile
    paints up to half a reference pixel past its scaled level-0 footprint, so
    at L > 0 it must also sit inside its converted diamond box
    (tile_screen_bounds_over over its 3x3 heights), which a patch bbox is
    built from. At pct 200 the clamp must actually fire somewhere, or that
    case proves nothing."""
    monkeypatch.setattr(settings, "_elev_step_pct", pct)
    scn = _scenario(units=False, elevation_of=lambda x, y: min(2, (x // 2 + y // 3) % 3))
    cache = _cache(scn)
    ref = cache._levels[0]
    assert max(cache.mip_levels()) > 0, "no finer level -- vacuous"
    clamped = 0
    for mip in cache.mip_levels():
        lvl = cache._levels[mip]
        rise = cache._level_corner_rise(lvl)
        for ty in range(MAP_H):
            for tx in range(MAP_W):
                got = _footprint(tx, ty, lvl.proj, rise)
                ref_fp = _footprint(tx, ty, ref.proj, cache.corner_rise)
                if mip <= 0:
                    inner, outer = got, cache._bbox_to_level(mip, ref_fp)
                else:
                    inner, outer = ref_fp, cache._level_bbox_to_reference(mip, got)
                    near = cache.elevations[max(0, ty - 1) : ty + 2, max(0, tx - 1) : tx + 2]
                    box = iso_geometry.tile_screen_bounds_over(tx, ty, int(near.min()), int(near.max()), ref.proj)
                    box = cache._bbox_to_level(mip, tuple(int(v) for v in box))
                    assert _inside(got, box), f"pct {pct} level {mip} tile {(tx, ty)}: {got} outside its box {box}"
                assert _inside(inner, outer), f"pct {pct} level {mip} tile {(tx, ty)}: {inner} outside {outer}"
                corners = (rise[ty, tx], rise[ty, tx + 1], rise[ty + 1, tx], rise[ty + 1, tx + 1])
                raw_len = iso_geometry._sloped_column_runs(lvl.tile_px, *map(int, corners))[3]
                clamped += int((raw_len[raw_len != 0] < 1).any())
    if pct == 200:
        assert clamped, "the clamp never fired at pct 200 -- vacuous"


# --- 8. rebuild counts ------------------------------------------------------


def test_a_gen_bump_rebuilds_each_resident_level_once_and_never_an_unshown_one(sprite_install, monkeypatch) -> None:  # noqa: F811
    """One build per resident level per gen bump, through the lazy _level();
    and no scene-space read (corner_rise, the pick plane, the seed tiles)
    builds level 0's unit layers."""
    cache = _cache(_scenario(), sprites=True)
    builds: list[int] = []
    real = cache._commit_level
    monkeypatch.setattr(cache, "_commit_level", lambda mip, *a: (builds.append(mip), real(mip, *a))[1])

    shown = (-2, -1, 1)
    for mip in shown:
        _whole(cache, mip)
    assert sorted(builds) == list(shown)
    builds.clear()
    cache.invalidate_units()
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)), terrain_changed=False)
    for mip in shown:
        _whole(cache, mip)
    assert sorted(builds) == list(shown), "each resident level rebuilds exactly once after one gen bump"

    builds.clear()
    assert cache.corner_rise is not None and cache.proj is cache._levels[0].proj
    cache.pick_tile(*(d // 2 for d in cache.canvas_dims(0)))
    cache.elevation_seed_tiles({(5, 5)}, {(5, 5): 1})
    cache.sprite_extent_before([], -2)
    assert builds == [] and cache._levels[0].gen == -1, "a scene-space read built level 0"


# --- 9. pick at level 0 while only -2 is resident ---------------------------


@pytest.mark.parametrize("mip", [-2, 1])
def test_pick_resolves_from_level_0_while_only_another_level_was_composited(mip) -> None:
    """Pick planes stay level 0 (scene space) whatever level is on screen.
    Compared plane by plane, every pixel, against an independent rasterization
    from iso_geometry alone (test_sloped_pick's oracle)."""
    from test_sloped_pick import _pick_oracle

    scn = _scenario(units=False)
    cache = _cache(scn, chunk_px=256)
    _whole(cache, mip)
    oracle, _counts = _pick_oracle(scn, cache.corner_rise, cache.proj, cache.tile_px)
    w, h = cache.canvas_dims(0)
    for cy in range(-(-h // cache.chunk_px)):
        for cx in range(-(-w // cache.chunk_px)):
            x0, y0 = cx * cache.chunk_px, cy * cache.chunk_px
            plane = cache._pick_plane(cx, cy)
            want = oracle[y0 : y0 + plane.shape[0], x0 : x0 + plane.shape[1]]
            assert np.array_equal(plane[: want.shape[0], : want.shape[1]], want), f"chunk {(cx, cy)}"
    for tx, ty in [(0, 0), (7, 11), (MAP_W - 1, MAP_H - 1)]:
        ys, xs = np.nonzero(oracle == ty * MAP_W + tx)
        assert cache.pick_tile(int(xs[len(xs) // 2]), int(ys[len(ys) // 2])) == (tx, ty)
    assert {key[0] for key in cache._cache} == {mip}, "a level-0 chunk was composited"
    assert cache._levels[0].gen == -1


# --- 4. patch vs fresh at every level other than 0 ---------------------------


@dataclass(frozen=True)
class _Sites:
    """Where the edit oracles edit: the sprite unit's tile, a tile no unit
    reaches (its raise moves only the headroom), mill A of the three-mill
    chain (B and C follow at x + 1 and x + 2; A's own tile is the one raised),
    the far mill, the unit added later, a bump far from all of them, and a
    building with no sprite. Only that one's bbox moves with the headroom
    alone: a sprited building's column slices (TASK-221) put a sprite bbox,
    far taller than the widening, on every footprint key."""

    unit: tuple[int, int]
    quiet: tuple[int, int]
    chain: tuple[float, float]
    far: tuple[float, float]
    added: tuple[float, float]
    bump: tuple[int, int]
    plain: tuple[float, float]

    @property
    def mill(self) -> tuple[int, int]:
        return int(self.chain[0]), int(self.chain[1])


TEMPLATE = _Sites(
    unit=(40, 40), quiet=(20, 20), chain=(60.0, 60.0), far=(30.0, 30.0), added=(45.5, 44.5), bump=(110, 110),
    plain=(80.0, 30.0),
)
# The same layout on _scenario()'s 20x20 field, where a whole +1 level is 2560 px wide, not 15360.
FIELD = _Sites(
    unit=(8, 8), quiet=(16, 4), chain=(14.0, 15.0), far=(4.0, 16.0), added=(12.5, 11.5), bump=(1, 18),
    plain=(4.0, 4.0),
)
PLAIN_CONST = 70  # a 2x2 House; shared_art_install draws no sprite for it


class _FieldTile(SyntheticTile):
    @property
    def xy(self) -> tuple[int, int]:
        return self.x, self.y


class _FieldMapManager(FakeMapManager):
    """FakeMapManager plus the members set_tiles_elevation() reads (with _FieldTile's xy)."""

    @property
    def map_size(self) -> int:
        return self.map_width

    def get_tile_safe(self, x: int, y: int):
        return self._by_xy.get((x, y))


def _base(sites: _Sites, flat: bool = True):
    """At elevation 5 (flat, so the first raise moves the headroom from 0;
    else with sites.bump one higher): the blank template, or a 20x20
    DESERT_SAND field for FIELD."""
    if sites is TEMPLATE:
        scn = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    else:
        tiles = [_FieldTile(x=x, y=y, elevation=5, terrain_id=DESERT_SAND) for y in range(MAP_H) for x in range(MAP_W)]
        scn = FakeScenario(MAP_W, MAP_H, tiles, [[] for _ in range(9)])
        scn.map_manager = _FieldMapManager(MAP_W, MAP_H, tiles)
        scn.unit_gen = 0
    for tile in scn.map_manager.terrain:
        tile.elevation = 5
    if not flat:
        scn.map_manager.get_tile(*sites.bump).elevation = 6
    return scn


def _mill_chain(scn, sites: _Sites) -> None:
    """test_elevation_unit_splice's chain: mills A-B-C overlapping in a row (owners 5, 3, 1), plus one far away."""
    cx, cy = sites.chain
    for player, dx in ((5, 0), (3, 1), (1, 2)):
        _place(scn, player, MILL_CONST, cx + dx, cy)
    _place(scn, 3, MILL_CONST, *sites.far)


def _edit_scenario(sites: _Sites = TEMPLATE):
    """Flat at 5, with the three-mill chain, one far mill, one sprite unit and the sprite-less building."""
    scn = _base(sites)
    _mill_chain(scn, sites)
    _place(scn, 1, CONST, sites.unit[0] + 0.5, sites.unit[1] + 0.5)
    _place(scn, 2, PLAIN_CONST, *sites.plain)
    return scn


def _draw(cache, scn, sites: _Sites = TEMPLATE) -> None:
    mm = scn.map_manager
    # A run through the unit's tile and out past its sprite on both sides: the
    # sprite's opaque square hides the ground under it (GH #107), four tiles each way.
    tiles = [(sites.unit[0] + dx, sites.unit[1]) for dx in range(-6, 7)]
    dirty = [i for i, t in enumerate(mm.terrain) if (t.x, t.y) in tiles]
    for i in dirty:
        mm.terrain[i].terrain_id = BLACK
    changed: set = set()
    bbox = render.dirty_screen_bbox_sloped(
        scn, dirty, cache.elevations, cache.proj, with_units=True, with_sprites=True, elevation_changed=changed
    )
    assert not changed
    cache.patch(bbox, elevation_changed=changed)


def _elevate(cache, scn, tiles, delta: int = 1) -> None:
    mm = scn.map_manager
    edit = _raise_edit(scn, tiles) if delta > 0 else (
        lambda: set_tiles_elevation(mm, [(x, y, mm.get_tile(x, y).elevation - 1) for x, y in tiles])
    )
    bbox, changed = _sloped_bbox(cache, scn, edit)
    cache.patch(bbox, elevation_changed=changed)


def _unit_edit(cache, splice) -> None:
    """As _after_unit_mutation's wholesale repaint: the splice, then the whole canvas evicted."""
    cache.invalidate_units([splice])
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)), terrain_changed=False)


def _over_cap(cache, scn, sites: _Sites) -> None:
    with pytest.MonkeyPatch.context() as m:
        m.setattr(render_cache, "_SLOPED_ELEV_SPLICE_MAX_UNITS", 2)
        gen = cache._source_gen
        _elevate(cache, scn, [sites.mill])
    assert cache._source_gen > gen, "the mill chain spliced under a cap of 2 -- vacuous"


def _steps(scn, sites: _Sites):
    """(name, action, splices) in order; splices says whether the step reaches _splice_level()."""
    added = {}

    def add(cache):
        unit = _place(scn, 2, CONST, *sites.added)
        added["unit"] = unit
        _unit_edit(cache, _add_splice(scn, 2, unit))

    def remove(cache):
        index = next(i for i, u in enumerate(scn.unit_manager.units[2]) if u is added["unit"])
        _unit_edit(cache, _delete_splice(scn, 2, index, added["unit"]))

    def headroom(cache, delta, tiles, want_steps):
        _elevate(cache, scn, tiles, delta)
        assert cache._target_headroom == want_steps * cache.proj.elev_step, "the headroom did not move -- vacuous"

    return [
        ("draw", lambda cache: _draw(cache, scn, sites), False),
        ("headroom_up", lambda cache: headroom(cache, 1, [sites.quiet], 1), True),
        ("elev_splice", lambda cache: _elevate(cache, scn, [sites.unit]), True),
        ("over_cap", lambda cache: _over_cap(cache, scn, sites), False),
        ("unit_add", add, True),
        ("unit_remove", remove, True),
        # Flat again, so the headroom falls back to 0 while the unit and the mills re-anchor.
        ("headroom_down", lambda cache: headroom(cache, -1, [sites.quiet, sites.unit, sites.mill], 0), True),
    ]


def _level_differs(cache, scn, mip: int) -> str | None:
    fresh = _cache(scn, sprites=True)
    got, want = _whole(cache, mip), _whole(fresh, mip)
    if not np.array_equal(got, want):
        return f"{int((got != want).any(axis=2).sum())} px"
    if _source_state(cache, mip) != _source_state(fresh, mip):
        return "unit layers"
    return None


def _run_steps(scn, levels, broken=None, sites: _Sites = TEMPLATE) -> list[str]:
    """Applies _steps() with every level in `levels` rendered whole before each
    step, and returns '<step> @<mip>: <what>' for each level that then differs
    from a fresh cache. broken=(step, mip) skips that level's splice (or, for a
    step with none, its patch composite) on that step alone."""
    cache = _cache(scn, sprites=True)
    cache.max_bytes *= 8  # every level stays resident: an eviction would hide a stale chunk
    current = {}
    real_splice, real_patch = cache._splice_level, cache.patch

    def splice(lvl, *a):
        if broken and current["step"] == broken[0] and lvl is cache._levels[broken[1]]:
            return None
        return real_splice(lvl, *a)

    def patch(bbox, elevation_changed=None, levels=None, rebuild_levels=None):
        if broken and current["step"] == broken[0] and not current["splices"]:
            levels = [m for m in cache.resident_levels() if m != broken[1]]
        return real_patch(bbox, elevation_changed, levels, rebuild_levels=rebuild_levels)

    cache._splice_level, cache.patch = splice, patch
    failures = []
    for name, action, splices in _steps(scn, sites):
        for mip in levels:
            _whole(cache, mip)
        resident = len(cache._cache)
        current.update(step=name, splices=splices)
        action(cache)
        if name in ("draw", "headroom_up", "elev_splice", "over_cap", "headroom_down"):
            assert len(cache._cache) == resident, f"{name}: a patch evicted chunks, which would hide a stale one"
        failures += [f"{name} @{mip}: {why}" for mip in levels if (why := _level_differs(cache, scn, mip))]
    return failures


EDIT_CASES = [(TEMPLATE, (-1,)), (TEMPLATE, (-2,)), (FIELD, (1,)), (FIELD, (-2, -1, 0, 1))]
STEP_NAMES = ["draw", "headroom_up", "elev_splice", "over_cap", "unit_add", "unit_remove", "headroom_down"]


@pytest.mark.parametrize("sites, levels", EDIT_CASES, ids=["-1", "-2", "+1", "all"])
def test_every_edit_kind_patched_at_another_level_matches_a_fresh_cache(shared_art_install, sites, levels) -> None:  # noqa: F811
    assert _run_steps(_edit_scenario(sites), levels, sites=sites) == []


@pytest.mark.parametrize("step", STEP_NAMES)
def test_the_edit_oracle_names_the_one_step_broken_at_level_minus_2(shared_art_install, step) -> None:  # noqa: F811
    """Control for the test above: only step `step` is broken, at -2 alone
    while -1 is resident too. The first difference must be that step at -2,
    and -1 must stay clean on it (later steps may inherit the damage)."""
    failures = _run_steps(_edit_scenario(), (-1, -2), broken=(step, -2))
    assert failures and failures[0].startswith(f"{step} @-2:"), failures
    assert not any(f.startswith(f"{step} @-1:") for f in failures), failures


@pytest.mark.parametrize("step", STEP_NAMES)
def test_the_edit_oracle_names_the_one_step_broken_at_level_plus_1(shared_art_install, step) -> None:  # noqa: F811
    """The same control at +1 on FIELD, with level 0 resident too: the first
    difference must be that step at +1, and 0 must stay clean on it."""
    failures = _run_steps(_edit_scenario(FIELD), (0, 1), broken=(step, 1), sites=FIELD)
    assert failures and failures[0].startswith(f"{step} @1:"), failures
    assert not any(f.startswith(f"{step} @0:") for f in failures), failures


@pytest.mark.parametrize("sites, levels", [(TEMPLATE, (-1, -2)), (FIELD, (1, -1))], ids=["coarser", "finer"])
def test_a_headroom_change_names_each_spliced_level_on_the_op_line(shared_art_install, traced, sites, levels) -> None:  # noqa: F811
    """Each current level checks its own headroom, so each one that moves
    says so with its own mip and its own pixel values."""
    from descape import debug_log, perf_trace

    scn = _edit_scenario(sites)
    cache = _cache(scn, sprites=True)
    for mip in levels:
        _whole(cache, mip)
    with perf_trace.op("Set elevation"):
        _elevate(cache, scn, [sites.quiet])
    perf_trace.flush_idle()

    line = next(line for line in debug_log.get_log_text().splitlines() if "perf op set-elevation" in line)
    for mip in levels:
        assert f"elev_headroom=0->{cache._levels[mip].proj.elev_step}@{mip}" in line, line
    assert "@0" not in line, "level 0 was never built, so it has no headroom to check"


# --- 5. a level outside rebuild_levels defers, then flushes when shown -------


def _deferred(case: str, monkeypatch, sites: _Sites):
    """test_elevation_unit_splice's _deferred_cases() on `sites`: (scenario, raise steps)."""
    if sites is TEMPLATE:
        return _deferred_cases(case, monkeypatch)
    if case == "headroom_empty":
        scn = _base(sites)
        _place(scn, 1, MILL_CONST, *sites.chain)
        return scn, [[sites.quiet]]
    scn = _base(sites, flat=False)
    _mill_chain(scn, sites)
    if case == "two_splices":
        _place(scn, 1, CONST, sites.unit[0] + 0.5, sites.unit[1] + 0.5)
        return scn, [[sites.mill], [sites.unit]]
    assert case == "between_caps"
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 2)
    monkeypatch.setattr(render_cache, "_SLOPED_ELEV_SPLICE_MAX_UNITS", 3)
    return scn, [[sites.mill]]


@pytest.mark.parametrize("sites, shown, deferred", [(TEMPLATE, -2, -1), (FIELD, -1, 1)], ids=["coarser", "finer"])
@pytest.mark.parametrize("case", ["two_splices", "headroom_empty", "between_caps"])
def test_a_level_left_out_of_rebuild_levels_flushes_to_a_fresh_cache_when_shown(
    case, sites, shown, deferred, shared_art_install, monkeypatch  # noqa: F811
) -> None:
    """Patched with rebuild_levels=(shown,) while `deferred` is resident too:
    `shown` splices in the patch, `deferred` stays current with the tiles
    pending and flushes once when it is next composited. Both then equal a
    fresh cache. The cases are test_elevation_unit_splice's deferred-flush
    ones: two splices into one union, a headroom move with no unit in reach,
    and a union over Stepped's cap but within Sloped's. `finer` defers +1, on
    FIELD."""
    scn, steps = _deferred(case, monkeypatch, sites)
    cache = _cache(scn, sprites=True)
    cache.max_bytes *= 8  # both levels stay resident: an eviction would hide a stale chunk
    for mip in (deferred, shown):
        _whole(cache, mip)
    flushes: list[int] = []
    real_flush = cache._flush_pending
    monkeypatch.setattr(cache, "_flush_pending", lambda mip: (flushes.append(mip), real_flush(mip))[1])

    for tiles in steps:
        bbox, changed = _sloped_bbox(cache, scn, _raise_edit(scn, tiles))
        cache.patch(bbox, elevation_changed=changed, rebuild_levels=(shown,))
        lvl = cache._levels[deferred]
        assert lvl.gen == cache._source_gen and lvl.pending_elev, f"{deferred} did not defer"
        assert not cache._levels[shown].pending_elev, "the shown level deferred its own splice"
    assert not flushes
    assert _level_differs(cache, scn, shown) is None
    assert flushes == [], f"showing {shown} flushed another level"

    assert _level_differs(cache, scn, deferred) is None
    assert flushes == [deferred]
    assert cache._levels[deferred].gen == cache._source_gen, "the flush went wholesale"
    assert cache._levels[0].gen == -1, "level 0 was built though nothing showed it"


def test_an_empty_splice_patch_never_recomputes_a_deferred_levels_rise(shared_art_install, monkeypatch) -> None:  # noqa: F811
    """The early-return check must not pay corner_rise_px for -1, which only
    gets pending tiles; -1 then pays it once, on the composite that flushes it."""
    scn, steps = _deferred_cases("headroom_empty", monkeypatch)
    cache = _cache(scn, sprites=True)
    for mip in (-1, -2):
        _whole(cache, mip)
    calls: list[int] = []
    real = iso_geometry.corner_rise_px

    def spy(elevations, proj, rule="average"):
        calls.append(proj.tile_px)
        return real(elevations, proj, rule=rule)

    monkeypatch.setattr(iso_geometry, "corner_rise_px", spy)
    (tiles,) = steps
    bbox, changed = _sloped_bbox(cache, scn, _raise_edit(scn, tiles))
    calls.clear()
    cache.patch(bbox, elevation_changed=changed, rebuild_levels=(-2,))
    assert cache._levels[-1].pending_elev, "-1 did not defer, so the check is vacuous"
    assert cache._levels[-1].tile_px not in calls, "the deferred level's rise was recomputed in the patch"
    assert cache._levels[-2].tile_px in calls
    calls.clear()
    _whole(cache, -1)
    assert calls == [cache._levels[-1].tile_px]


@pytest.mark.parametrize(
    "sites, levels, span", [(TEMPLATE, (-1, -2), range(35, 66)), (FIELD, (1, -1), range(3, 18))], ids=["coarser", "finer"]
)
def test_a_wholesale_refresh_re_derives_every_level(shared_art_install, sites, levels, span) -> None:  # noqa: F811
    """refresh_wholesale() (TASK-031.61's in-place full re-render) rebinds
    level 0's corner_rise, so every level re-derives its own and rebuilds on
    its next composite; both shown levels then equal a fresh cache."""
    scn = _edit_scenario(sites)
    cache = _cache(scn, sprites=True)
    cache.max_bytes *= 8  # both levels stay resident: an eviction would hide a stale chunk
    for mip in levels:
        _whole(cache, mip)
    mm = scn.map_manager
    tiles = [(x, y) for x in span for y in span if (x + y) % 3 == 0]
    set_tiles_elevation(mm, [(x, y, 6) for x, y in tiles])
    dirty = [i for i, t in enumerate(mm.terrain) if (t.x, t.y) in set(tiles)]
    old = {mip: cache._level_corner_rise(cache._levels[mip]) for mip in levels}

    assert cache.refresh_wholesale(dirty) is True
    for mip in levels:
        assert cache._level_corner_rise(cache._levels[mip]) is not old[mip]
        assert _level_differs(cache, scn, mip) is None, f"level {mip}"
    assert cache._levels[0].gen == -1
