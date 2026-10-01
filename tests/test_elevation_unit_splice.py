"""An elevation patch re-anchors only the units whose placement reads a
changed height, instead of rebuilding every unit's sprite and building bbox
(2026-09-24 plan, "anchor-local unit-source splice"). See
render_cache._elevation_splices and _reanchor_units for the guard and the
batch splice this file exercises.

Stepped reads a unit's OWN tile only; Sloped reads its own tile's four
corners, each of which blends the 3x3 around it, so Sloped dilates the
changed set by one tile and Stepped does not. The neighbour tests below pin
both halves.

Same fixture posture as tests/test_invalidate_units_splice.py (whose helpers
this reuses): BLANK_TEMPLATE_PATH with duck-typed units appended BEFORE the
cache is built. That ordering matters here: render.unit_own_tile_index() and
render.wall_variant_rotation_overrides() are memoized on scenario.unit_gen,
which a bare list append never bumps.

Sloped fixtures carry one far-away bump (BUMP_TILE) so the first raise does
not move _unit_rise_headroom_px from 0 to one elev_step: that transition
changes every building's bbox and falls back by design, and has its own test.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_bystander_grid_patch import grid_state
from test_invalidate_units_splice import (
    MILL_CONST,
    Unit,
    _call_counts,
    _decorated_mill,
    shared_art_install,  # noqa: F401 -- pytest fixture, imported for its name
    slotted_composite_install,  # noqa: F401 -- pytest fixture, imported for its name
)
from test_sprite_edit_bbox import REACH_NAMES, sprite_install  # noqa: F401 -- fixture

from descape import asset_source, render, render_cache, unit_sprites
from descape.elevation_tools import set_tiles_elevation
from descape.render import (
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    render_terrain_iso_with_proj,
    render_terrain_sloped_with_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter

import conftest
from test_unit_sprites import CONST as SPRITE_CONST
from test_unit_sprites import FILE_NAME, build_sld

STYLES = ["stepped", "sloped"]
BASE_ELEVATION = 5
BUMP_TILE = (110, 110)
UNIT_TILE = (40, 40)
NEIGHBOUR_TILE = (41, 40)
MARK_CONST = 999999  # resolves to no graphic: a plain coloured mark, span (1, 1)
WALL_CONST = 72  # a real _ROTATION_VARIANT_CONSTS member


def _scenario(flat: bool = False):
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for t in scenario.map_manager.terrain:
        t.elevation = BASE_ELEVATION
    if not flat:
        scenario.map_manager.get_tile(*BUMP_TILE).elevation = BASE_ELEVATION + 1
    return scenario


def _place(scenario, player_id: int, const: int, x: float, y: float, rotation: float = 0.0) -> Unit:
    unit = Unit(x, y, const, rotation)
    scenario.unit_manager.units[player_id].append(unit)
    return unit


def _make_cache(style: str, scenario, sprites: bool = False, unit_filter: UnitFilter = UnitFilter()):
    mm = scenario.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    if style == "stepped":
        elevations, proj = elevations_and_proj(scenario)
        cache = IsoChunkCache(scenario, elevations, proj, tile_px, sprites=sprites, unit_filter=unit_filter)
    else:
        elevations, corner_rise, proj = sloped_elevations_and_proj(scenario)
        cache = SlopedChunkCache(
            scenario, elevations, corner_rise, proj, tile_px, sprites=sprites, unit_filter=unit_filter
        )
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)
    return cache


def _raise(style: str, cache, scenario, tiles) -> set:
    """The Elevate tool's +1 on `tiles`, through the same three calls
    ViewerWindow's stroke makes: set_tiles_elevation, the dirty bbox (which
    also writes cache.elevations in place and fills elevation_changed), then
    patch(). Repaints the whole canvas afterwards so the pixel oracle checks
    source state, not the dirty bbox."""
    mm = scenario.map_manager
    before = [t.elevation for t in mm.terrain]
    set_tiles_elevation(mm, [(x, y, mm.get_tile(x, y).elevation + 1) for x, y in tiles])
    dirty = [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]
    changed: set = set()
    bbox_fn = dirty_screen_bbox_iso if style == "stepped" else dirty_screen_bbox_sloped
    bbox = bbox_fn(
        scenario, dirty, cache.elevations, cache.proj, with_units=True,
        with_sprites=cache.sprites_enabled, elevation_changed=changed,
    )
    assert changed, "the raise changed nothing"
    cache.patch(bbox, elevation_changed=changed)
    cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
    return changed


def _oracle(style: str, scenario, sprites: bool) -> np.ndarray:
    if style == "stepped":
        return render_terrain_iso_with_proj(scenario, with_units=True, with_sprites=sprites)[0]
    return render_terrain_sloped_with_proj(scenario, with_units=True, with_sprites=sprites)[0]


def _assert_pixels_match(style: str, cache, scenario, sprites: bool, counts=None) -> None:
    """counts, if given, is checked between the cache's repaint and the
    oracle's own render, which calls the wholesale walks itself."""
    canvas_w, canvas_h = cache.canvas_dims(0)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    if counts is not None:
        _no_wholesale(counts)
    assert np.array_equal(stitched, _oracle(style, scenario, sprites)[:canvas_h, :canvas_w])


def _source_state(cache, mip: int = 0):
    """building_bboxes, its bystander grid and the SpriteLayer as plain
    comparable values. A SpriteDraw holds an ndarray, so draws compare by
    identity: both caches get theirs from unit_sprites' own memo."""
    if isinstance(cache, IsoChunkCache):
        lvl = cache._level(mip)
        bboxes, sprites, grid = lvl.building_bboxes, lvl.sprites, lvl.bystander_grid
    else:
        bboxes, sprites, grid = cache.building_bboxes, cache.sprites, cache.bystander_grid
    layer = None if sprites is None else (
        {k: [(id(d), px, py) for d, px, py in v] for k, v in sprites.by_anchor.items()},
        dict(sprites.bboxes), sprites.skip_ids, dict(sprites.farm_by_tile),
    )
    return dict(bboxes), layer, grid_state(grid)


def _assert_matches_fresh_cache(style: str, cache, scenario, sprites: bool, unit_filter=UnitFilter()) -> None:
    fresh = _make_cache(style, scenario, sprites=sprites, unit_filter=unit_filter)
    assert _source_state(cache) == _source_state(fresh)


def _resolve_calls(monkeypatch) -> list:
    """Every unit render._resolve_unit_sprite() is asked about, in order."""
    calls = []
    real = render._resolve_unit_sprite

    def counted(*args, **kwargs):
        calls.append(args[9])
        return real(*args, **kwargs)

    monkeypatch.setattr(render, "_resolve_unit_sprite", counted)
    return calls


def _no_wholesale(counts) -> None:
    assert counts == {"building_bboxes": 0, "sprites": 0}, "the elevation patch took the wholesale path"


# --- the splice fires -------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("sprites", [False, True])
def test_a_lone_unit_on_a_raised_tile_splices(style, sprites, sprite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    unit = _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    _place(scenario, 2, MILL_CONST, 60.0, 60.0)
    cache = _make_cache(style, scenario, sprites=sprites)
    before = _source_state(cache)
    counts = _call_counts(monkeypatch)
    resolved = _resolve_calls(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE, (60, 60), (59, 59)])

    _no_wholesale(counts)
    if sprites:
        assert unit in resolved, "the raised unit was never re-resolved"
    assert _source_state(cache) != before, "the raise moved nothing, so this proves nothing"
    _assert_pixels_match(style, cache, scenario, sprites, counts)
    _assert_matches_fresh_cache(style, cache, scenario, sprites)


@pytest.mark.parametrize("style", STYLES)
def test_a_neighbour_of_the_raised_tile_moves_on_sloped_only(style, sprite_install, monkeypatch):  # noqa: F811
    """Sloped's radius 1 vs Stepped's radius 0. On Sloped a unit one tile off
    the raised tile rides the raised corners; on Stepped it reads only its own
    tile, so it must not even be re-resolved."""
    scenario = _scenario()
    neighbour = _place(scenario, 1, SPRITE_CONST, NEIGHBOUR_TILE[0] + 0.5, NEIGHBOUR_TILE[1] + 0.5)
    cache = _make_cache(style, scenario, sprites=True)
    before = _source_state(cache)
    counts = _call_counts(monkeypatch)
    resolved = _resolve_calls(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

    _no_wholesale(counts)
    if style == "sloped":
        assert neighbour in resolved
        assert _source_state(cache)[1] != before[1], "the neighbour's sprite did not move"
    else:
        assert neighbour not in resolved, "Stepped re-resolved a unit whose own tile did not change"
        assert _source_state(cache) == before
    _assert_pixels_match(style, cache, scenario, True, counts)
    _assert_matches_fresh_cache(style, cache, scenario, True)


@pytest.fixture
def wall_install(tmp_path, monkeypatch):
    """WALL_CONST as a 5-variant graphic whose frames differ in colour, so a
    wall drawn at the wrong variant is a pixel mismatch. Reach pinned to the
    canvas like sprite_install's, for the same reason."""
    canvas = 4 * unit_sprites.NATIVE_TILE_W
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    # No player-colour mask: a full-coverage one tints every frame alike.
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(5, canvas=canvas, playercolor=False))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {WALL_CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 5,
                              "mirroring_mode": 0, "frame_count": 1, "rotation_is_variant": True}},
    )
    for name in REACH_NAMES:
        monkeypatch.setattr(unit_sprites, name, canvas // 2)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


@pytest.mark.parametrize("style", STYLES)
def test_a_wall_beside_the_change_keeps_its_neighbour_derived_shape(style, wall_install, monkeypatch):
    """Walls are splice-safe for an elevation edit (their override is a pure
    function of unit data), but only if the splice passes the REAL override
    dict: the stored rotation here is a radian-encoded variant the override
    replaces, so resolving with {} would draw the wrong frame."""
    scenario = _scenario()
    radian = 2 * (2 * np.pi / 5)  # variant 2, radian-encoded, so the file is radian
    for x in (39, 40, 41):
        _place(scenario, 1, WALL_CONST, x + 0.5, UNIT_TILE[1] + 0.5, radian)
    overrides = render.wall_variant_rotation_overrides(scenario)
    assert overrides, "no wall is overridden, so {} and the real dict coincide"
    assert any(unit_sprites.variant_index(WALL_CONST, v) != 2 for v in overrides.values()), (
        "every override equals the stored variant, so {} and the real dict coincide"
    )
    cache = _make_cache(style, scenario, sprites=True)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

    _no_wholesale(counts)
    _assert_pixels_match(style, cache, scenario, True, counts)
    _assert_matches_fresh_cache(style, cache, scenario, True)


@pytest.mark.parametrize("style", STYLES)
def test_a_slotted_composite_re_anchors_every_slot(style, slotted_composite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    _place(scenario, 1, MILL_CONST, 40.0, 40.0)
    cache = _make_cache(style, scenario, sprites=True)
    assert len(_source_state(cache)[1][0]) == 2, "the fixture is not producing two slots"
    before = _source_state(cache)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

    _no_wholesale(counts)
    after = _source_state(cache)
    assert after[1][0].keys() == before[1][0].keys()
    assert all(after[1][0][k] != before[1][0][k] for k in before[1][0]), "a slot stayed at the old height"
    _assert_pixels_match(style, cache, scenario, True, counts)
    _assert_matches_fresh_cache(style, cache, scenario, True)


@pytest.mark.parametrize("style", STYLES)
def test_an_off_centre_mark_keeps_its_bystander_bbox(style, monkeypatch):
    """_building_bboxes_iso keeps an off-centre 1x1 mark (it straddles a tile
    boundary); the pre-plan splice treated only span > 1 as a building and
    would have dropped it. Sprites off, so the building part is all there is."""
    scenario = _scenario()
    mark = _place(scenario, 1, MARK_CONST, UNIT_TILE[0] + 0.85, UNIT_TILE[1] + 0.2)
    assert render.unit_paint_offset(mark) != (0.0, 0.0)
    cache = _make_cache(style, scenario)
    assert UNIT_TILE in _source_state(cache)[0], "the fixture mark carries no bbox to keep"
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

    _no_wholesale(counts)
    assert UNIT_TILE in _source_state(cache)[0], "the splice dropped the off-centre mark's bbox"
    _assert_pixels_match(style, cache, scenario, False, counts)
    _assert_matches_fresh_cache(style, cache, scenario, False)


@pytest.mark.parametrize("style", STYLES)
def test_a_filtered_out_unit_stays_absent(style, sprite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    hidden = _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    _place(scenario, 2, SPRITE_CONST, 60.5, 60.5)
    unit_filter = UnitFilter(players=frozenset({2}))
    cache = _make_cache(style, scenario, sprites=True, unit_filter=unit_filter)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE, (60, 60)])

    _no_wholesale(counts)
    assert id(hidden) not in _source_state(cache)[1][2]
    canvas_w, canvas_h = cache.canvas_dims(0)
    fresh = _make_cache(style, scenario, sprites=True, unit_filter=unit_filter)
    got = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    assert np.array_equal(got, fresh.render_rect(0, 0, canvas_w, canvas_h, mip=0))
    assert _source_state(cache) == _source_state(fresh)


# --- shared tiles splice their whole component (2026-09-28 plan) ------------
# shared_art_install draws MILL_CONST and SPRITE_CONST with one team-tinted
# graphic, so co-occupants overlap on screen and paint order shows as pixels.


def _splice_calls(monkeypatch) -> list:
    """Every _elevation_splices() result, in call order."""
    calls = []
    real = render_cache._elevation_splices

    def recorded(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(render_cache, "_elevation_splices", recorded)
    return calls


def _component(calls) -> list[int]:
    """The one splice call's unit ids, in the order it spliced them. Ids, since
    the fixture Unit is a dataclass: two stacked units compare equal."""
    assert len(calls) == 1 and calls[0] is not None, f"expected one component splice, got {calls}"
    return [id(s.unit) for s in calls[0]]


def _ids(*units) -> list[int]:
    return [id(u) for u in units]


def _assert_spliced(style, cache, scenario, sprites, counts, unit_filter=UnitFilter()) -> None:
    """No wholesale walk, then source state and pixels equal a fresh cache's
    (and the oracle's, unfiltered)."""
    canvas_w, canvas_h = cache.canvas_dims(0)
    stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
    _no_wholesale(counts)
    fresh = _make_cache(style, scenario, sprites=sprites, unit_filter=unit_filter)
    assert _source_state(cache) == _source_state(fresh)
    assert np.array_equal(stitched, fresh.render_rect(0, 0, canvas_w, canvas_h, mip=0))
    if unit_filter == UnitFilter():
        assert np.array_equal(stitched, _oracle(style, scenario, sprites)[:canvas_h, :canvas_w])


@pytest.mark.parametrize("owners", [(1, 2), (2, 1)], ids=["low-first", "high-first"])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_shared_own_tile_splices(style, sprites, owners, shared_art_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    placed = [_place(scenario, p, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5) for p in owners]
    cache = _make_cache(style, scenario, sprites=sprites)
    before = _source_state(cache)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

    walk = [u for _, u in sorted(zip(owners, placed, strict=True), key=lambda pu: pu[0])]
    assert _component(calls) == _ids(*walk), "the component is not in walk order"
    # Sprites off, a centred 1x1 unit holds no source key at all.
    assert _source_state(cache) != before or not sprites, "the raise moved nothing, so this proves nothing"
    _assert_spliced(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("raised", ["mill", "decoration"])
@pytest.mark.parametrize(("mill_owner", "decor_owner"), [(1, 3), (3, 1)])
@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_decorated_mill_splices_as_one_component(
    style, sprites, mill_owner, decor_owner, raised, shared_art_install, monkeypatch  # noqa: F811
):
    """Raising either one's own tile re-derives both, in walk order: in one
    owner order the raised unit sorts after its co-occupant."""
    scenario = _scenario()
    mill, decor = _decorated_mill(scenario, mill_owner, decor_owner)
    cache = _make_cache(style, scenario, sprites=sprites)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    target = mill if raised == "mill" else decor
    _raise(style, cache, scenario, [(int(target.x), int(target.y))])

    want = [mill, decor] if mill_owner < decor_owner else [decor, mill]
    assert _component(calls) == _ids(*want)
    _assert_spliced(style, cache, scenario, sprites, counts)


def _mill_chain(scenario):
    """Mills A-B-C overlapping in a row (owners 5, 3, 1), plus one far away."""
    a = _place(scenario, 5, MILL_CONST, 60.0, 60.0)  # x 59..60
    b = _place(scenario, 3, MILL_CONST, 61.0, 60.0)  # x 60..61
    c = _place(scenario, 1, MILL_CONST, 62.0, 60.0)  # x 61..62
    _place(scenario, 3, MILL_CONST, 30.0, 30.0)
    return a, b, c


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_chain_component_re_derives_units_off_the_raised_tile(style, sprites, shared_art_install, monkeypatch):  # noqa: F811
    """Only A's own tile rises; C shares no tile with A but does with B."""
    scenario = _scenario()
    a, b, c = _mill_chain(scenario)
    cache = _make_cache(style, scenario, sprites=sprites)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [(60, 60)])

    assert _component(calls) == _ids(c, b, a), "the component is not [C, B, A] in walk order"
    _assert_spliced(style, cache, scenario, sprites, counts)


@pytest.mark.parametrize("sprites", [False, True])
@pytest.mark.parametrize("style", STYLES)
def test_a_filter_hidden_co_occupant_stays_out_of_the_component(style, sprites, shared_art_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    mill, decor = _decorated_mill(scenario, 1, 3)
    hidden = _place(scenario, 2, SPRITE_CONST, decor.x, decor.y)
    unit_filter = UnitFilter(players=frozenset({1, 3}))
    cache = _make_cache(style, scenario, sprites=sprites, unit_filter=unit_filter)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [(int(decor.x), int(decor.y))])

    component = _component(calls)
    assert id(hidden) not in component
    assert component == _ids(mill, decor)
    _assert_spliced(style, cache, scenario, sprites, counts, unit_filter)


@pytest.mark.parametrize("sprites", [False, True])
def test_a_sloped_neighbour_pair_is_seeded_by_the_dilation(sprites, shared_art_install, monkeypatch):  # noqa: F811
    """The shared tile is one off the raised tile, so only Sloped's radius-1
    dilation reaches it."""
    scenario = _scenario()
    pair = [_place(scenario, p, SPRITE_CONST, NEIGHBOUR_TILE[0] + 0.5, NEIGHBOUR_TILE[1] + 0.5) for p in (2, 1)]
    cache = _make_cache("sloped", scenario, sprites=sprites)
    before = _source_state(cache)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    _raise("sloped", cache, scenario, [UNIT_TILE])

    assert _component(calls) == _ids(*pair[::-1])
    assert _source_state(cache) != before or not sprites, "the pair did not move with the raised corners"
    _assert_spliced("sloped", cache, scenario, sprites, counts)


@pytest.mark.parametrize("style", STYLES)
def test_a_wall_in_a_shared_component_keeps_its_neighbour_derived_shape(style, wall_install, monkeypatch):
    """A Mill overlapping the run's end wall pulls it into the component (on
    Stepped as a co-occupant, not a seed), and it must re-derive with the real
    overrides, not {}."""
    scenario = _scenario()
    radian = 2 * (2 * np.pi / 5)
    walls = [_place(scenario, 1, WALL_CONST, x + 0.5, UNIT_TILE[1] + 0.5, radian) for x in (39, 40, 41)]
    mill = _place(scenario, 2, MILL_CONST, 42.0, 41.0)  # x 41..42, y 40..41: shares the (41, 40) wall's tile
    overrides = render.wall_variant_rotation_overrides(scenario)
    assert any(unit_sprites.variant_index(WALL_CONST, v) != 2 for v in overrides.values()), (
        "every override equals the stored variant, so {} and the real dict coincide"
    )
    cache = _make_cache(style, scenario, sprites=True)
    calls = _splice_calls(monkeypatch)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [(42, 41)])

    component = _component(calls)
    assert id(walls[2]) in component and id(mill) in component
    _assert_spliced(style, cache, scenario, True, counts)


# --- fallbacks --------------------------------------------------------------


def _assert_fell_back(cache, counts) -> None:
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)  # Stepped's wholesale path is lazy
    assert counts["building_bboxes"] >= 1, "this edit should have taken the wholesale fallback"


@pytest.mark.parametrize("style", STYLES)
def test_a_component_over_the_cap_falls_back(style, shared_art_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    _mill_chain(scenario)
    cache = _make_cache(style, scenario, sprites=True)
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 2)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [(60, 60)])

    _assert_fell_back(cache, counts)
    _assert_pixels_match(style, cache, scenario, True)


@pytest.mark.parametrize("style", STYLES)
def test_a_stroke_over_the_unit_threshold_falls_back(style, sprite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    _place(scenario, 1, SPRITE_CONST, 30.5, 30.5)
    _place(scenario, 1, SPRITE_CONST, 50.5, 50.5)
    cache = _make_cache(style, scenario, sprites=True)
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 1)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [(30, 30), (50, 50)])

    _assert_fell_back(cache, counts)
    _assert_pixels_match(style, cache, scenario, True)


def test_a_sloped_headroom_change_falls_back(sprite_install, monkeypatch):  # noqa: F811
    """The first raise on a truly flat map moves the Sloped bbox headroom from
    0 to one elev_step, which widens EVERY building's bbox, not just the
    re-anchored ones."""
    scenario = _scenario(flat=True)
    _place(scenario, 1, MILL_CONST, 60.0, 60.0)
    _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    cache = _make_cache("sloped", scenario, sprites=True)
    counts = _call_counts(monkeypatch)

    _raise("sloped", cache, scenario, [UNIT_TILE])

    assert counts["building_bboxes"] >= 1, "a headroom change should have taken the wholesale fallback"
    _assert_pixels_match("sloped", cache, scenario, True)
    _assert_matches_fresh_cache("sloped", cache, scenario, True)


# --- Stepped's per-level laziness and warms ---------------------------------


def test_a_stale_stepped_level_is_left_stale_and_rebuilds_fresh(sprite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    cache = _make_cache("stepped", scenario, sprites=True)
    assert not cache.is_level_resident(1), "level 1 was built by the fixture"

    _raise("stepped", cache, scenario, [UNIT_TILE])

    assert cache._levels[1].sprites is None and not cache.is_level_resident(1), "the splice touched a stale level"
    fresh = _make_cache("stepped", scenario, sprites=True)
    assert _source_state(cache, mip=1) == _source_state(fresh, mip=1)


def test_a_warm_started_before_an_elevation_splice_is_refused(sprite_install):  # noqa: F811
    """The splice does not bump _source_gen, so the warm's install predicate
    needs the splice epoch to see that its half-walked layer is stale."""
    scenario = _scenario()
    _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    cache = _make_cache("stepped", scenario, sprites=True)
    job = cache.level_warm_job(1)
    assert job is not None
    gen_before = cache._source_gen

    _raise("stepped", cache, scenario, [UNIT_TILE])

    assert cache._source_gen == gen_before, "the splice bumped the gen, so this is not testing the epoch"
    assert job.install(render._drain(job.gen)) is False
    assert not cache.is_level_resident(1)


# --- end to end through the viewer ------------------------------------------


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("style", ["Stepped", "Sloped"])
def test_a_brush_9_elevate_stroke_matches_a_fresh_render(style, sprite_install, monkeypatch):  # noqa: F811
    """A real multi-step Elevate drag through ViewerWindow, with NO repaint of
    the whole canvas afterwards: this checks the dirty bbox and the splice
    together, the way the app composes them. A stacked sprite pair (a shared
    by_anchor key, so paint order is in pixels) and a Mill with a unit on its
    anchor tile (MILL_CONST has no art here, so that one checks the bbox union
    and the component walk) make it the end-to-end shared-tile check."""
    window = conftest.terrain_edit_window()
    try:
        scenario = window.scenario
        mm = scenario.map_manager
        mm.get_tile(*BUMP_TILE).elevation = 1
        units = window._ensure_unit_edits()
        for x, y in ((32.5, 40.5), (36.5, 37.5), (40.5, 43.5), (47.5, 40.5)):
            units.add(1, SPRITE_CONST, x, y)
        units.add(2, MILL_CONST, 44.0, 38.0)
        units.add(3, SPRITE_CONST, 34.5, 41.5)
        units.add(2, SPRITE_CONST, 34.5, 41.5)
        units.add(1, SPRITE_CONST, 37.5, 41.5)  # the Mill below's sprite anchor tile, (37, 41)
        units.add(3, MILL_CONST, 38.0, 41.0)  # x 37..38, y 40..41
        # A style round trip rebuilds the cache from the edited scenario.
        other = "Sloped" if style == "Stepped" else "Stepped"
        window.terrain_style_combo.setCurrentText(other)
        window.terrain_style_combo.setCurrentText(style)
        cache = window._cache
        assert cache.sprites_enabled
        canvas_w, canvas_h = cache.canvas_dims(0)
        cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
        counts = _call_counts(monkeypatch)

        window._on_tool_selected("elevation")
        window.brush_size_spin.setValue(9)
        window.on_edit_stroke_start()
        for x in range(30, 50):
            window.on_edit_stroke_tile(x, 40, 0)
        window.on_edit_stroke_end()

        assert window._cache is cache, "the stroke rebuilt the cache"
        stitched = cache.render_rect(0, 0, canvas_w, canvas_h, mip=0)
        _no_wholesale(counts)
        full = _oracle(style.lower(), scenario, True)
        assert np.array_equal(stitched, full[:canvas_h, :canvas_w])
    finally:
        conftest.close_window(window)


# --- a gen-bumping step rebuilds the visible level only -------------------


def _keys_over(cache, mip: int, bbox) -> set:
    lx0, ly0, lx1, ly1 = cache._bbox_to_level(mip, bbox)
    cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, lx0, ly0, lx1, ly1)
    return {(mip, cx, cy) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)}


def test_rebuild_levels_leaves_another_stale_level_stale_and_evicts_only_its_bbox(
    sprite_install, monkeypatch,  # noqa: F811
):
    """A stacked pair over a cap of 1 makes the elevation splice fall back (gen
    bump). With rebuild_levels=(0,) only level 0 rebuilds; level -1's chunks
    over the bbox are evicted and the rest kept, and both levels still match a
    fresh cache."""
    scenario = _scenario()
    _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    _place(scenario, 2, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    cache = _make_cache("stepped", scenario, sprites=True)
    cache.render_rect(0, 0, *cache.canvas_dims(-1), mip=-1)
    assert cache.is_level_resident(-1)
    resident_before = {k for k in cache._cache if k[0] == -1}
    gen = cache._source_gen
    monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 1)
    counts = _call_counts(monkeypatch)

    mm = scenario.map_manager
    before = [t.elevation for t in mm.terrain]
    set_tiles_elevation(mm, [(*UNIT_TILE, BASE_ELEVATION + 1)])
    dirty = [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]
    changed: set = set()
    bbox = dirty_screen_bbox_iso(
        scenario, dirty, cache.elevations, cache.proj, with_units=True, with_sprites=True, elevation_changed=changed,
    )
    cache.patch(bbox, elevation_changed=changed, rebuild_levels=(0,))

    assert cache._source_gen > gen, "the splice did not fall back -- vacuous"
    assert cache.is_level_resident(0), "the visible level was not rebuilt"
    assert not cache.is_level_resident(-1), "the other level was rebuilt inside the patch"
    assert counts["building_bboxes"] == 1, "a level other than 0 rebuilt its bboxes"
    over = _keys_over(cache, -1, bbox)
    kept = {k for k in cache._cache if k[0] == -1}
    assert not kept & over, "a stale level's chunk over the bbox survived"
    assert kept == resident_before - over, "chunks off the bbox were evicted too"
    assert kept, "every level -1 chunk lay over the bbox -- vacuous"

    fresh = _make_cache("stepped", scenario, sprites=True)
    for key in sorted(k for k in cache._cache if k[0] == 0):
        assert np.array_equal(cache._cache[key], fresh.get_chunk(*key)), f"patched {key} differs from a fresh cache"
    for key in sorted(resident_before):
        assert np.array_equal(cache.get_chunk(*key), fresh.get_chunk(*key)), f"level -1 {key} differs from a fresh cache"


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_gen_bumping_elevate_step_in_a_shown_window_rebuilds_the_visible_level_only(
    sprite_install, monkeypatch,  # noqa: F811
):
    """The viewer's Stepped branch passes the viewport's mip as rebuild_levels.
    Needs a SHOWN window: with no viewport target it keeps rebuilding every level.
    A stacked pair over a cap of 1 forces the gen bump."""
    window = conftest.shown_terrain_window()
    try:
        scenario = window.scenario
        scenario.map_manager.get_tile(*BUMP_TILE).elevation = 1
        units = window._ensure_unit_edits()
        units.add(1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
        units.add(2, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
        window.terrain_style_combo.setCurrentText("Sloped")
        window.terrain_style_combo.setCurrentText("Stepped")
        cache = window._cache
        assert cache.sprites_enabled
        visible = window.map_view.viewport_chunk_target()[0]
        other = visible + 1
        for mip in (visible, other):
            cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
        assert cache.is_level_resident(other)
        gen = cache._source_gen
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 1)

        window._on_tool_selected("elevation")
        window.brush_size_spin.setValue(1)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*UNIT_TILE, 0)
        window.on_edit_stroke_end()

        assert window._cache is cache, "the stroke rebuilt the cache"
        assert cache._source_gen > gen, "the splice did not fall back -- vacuous"
        assert cache.is_level_resident(visible)
        assert not cache.is_level_resident(other), "the non-visible level rebuilt inside the stroke"
        mm = scenario.map_manager
        fresh = IsoChunkCache(
            scenario, window._iso_elevations.copy(), window._iso_proj,
            tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True,
        )
        for mip in (visible, other):
            dims = cache.canvas_dims(mip)
            assert np.array_equal(cache.render_rect(0, 0, *dims, mip=mip), fresh.render_rect(0, 0, *dims, mip=mip)), (
                f"level {mip} differs from a fresh cache"
            )
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_the_level_warm_after_a_gen_bumping_step_rebuilds_a_stale_non_neighbour_level(
    sprite_install, monkeypatch,  # noqa: F811
):
    """After a gen-bump step leaves a resident level stale, the stroke-end level
    warm rebuilds it even when it is not a neighbour of the opening level.
    A stacked pair over a cap of 1 forces the gen bump."""
    from PyQt5.QtGui import QTransform

    from descape import level_warm, settings

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.shown_terrain_window()
    try:
        scenario = window.scenario
        scenario.map_manager.get_tile(*BUMP_TILE).elevation = 1
        units = window._ensure_unit_edits()
        units.add(1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
        units.add(2, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
        window.terrain_style_combo.setCurrentText("Sloped")
        window.terrain_style_combo.setCurrentText("Stepped")
        cache = window._cache
        view = window.map_view
        view.setTransform(QTransform())
        visible = view.viewport_chunk_target()[0]
        neighbours = level_warm.neighbour_mips(cache, view._fit_baseline_scale() * view.devicePixelRatioF())
        others = [m for m in cache.mip_levels() if m != visible and m not in neighbours]
        assert others, "every level is visible or a neighbour -- vacuous"
        other = others[0]
        for mip in (visible, other):
            cache.render_rect(0, 0, *cache.canvas_dims(mip), mip=mip)
        gen = cache._source_gen
        monkeypatch.setattr(render_cache, "_ELEV_SPLICE_MAX_UNITS", 1)

        window._on_tool_selected("elevation")
        window.brush_size_spin.setValue(1)
        window.on_edit_stroke_start()
        window.on_edit_stroke_tile(*UNIT_TILE, 0)
        window.on_edit_stroke_end()

        assert cache._source_gen > gen, "the splice did not fall back -- vacuous"
        assert not cache.is_level_resident(other), "the stroke itself rebuilt the other level -- vacuous"
        window._level_warmer.run_to_completion()
        assert window._cache is cache
        assert cache.is_level_resident(other), "the level warm left the stale level stale"
        mm = scenario.map_manager
        fresh = IsoChunkCache(
            scenario, window._iso_elevations.copy(), window._iso_proj,
            tile_pixels_for_map(mm.map_width, mm.map_height), sprites=True,
        )
        dims = cache.canvas_dims(other)
        assert np.array_equal(cache.render_rect(0, 0, *dims, mip=other), fresh.render_rect(0, 0, *dims, mip=other))
    finally:
        conftest.close_window(window)
