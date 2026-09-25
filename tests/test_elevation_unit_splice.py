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
from test_invalidate_units_splice import (
    MILL_CONST,
    Unit,
    _call_counts,
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
    """building_bboxes and the SpriteLayer as plain comparable values. A
    SpriteDraw holds an ndarray, so draws compare by identity: both caches
    get theirs from unit_sprites' own memo."""
    if isinstance(cache, IsoChunkCache):
        lvl = cache._level(mip)
        bboxes, sprites = lvl.building_bboxes, lvl.sprites
    else:
        bboxes, sprites = cache.building_bboxes, cache.sprites
    layer = None if sprites is None else (
        {k: [(id(d), px, py) for d, px, py in v] for k, v in sprites.by_anchor.items()},
        dict(sprites.bboxes), sprites.skip_ids, dict(sprites.farm_by_tile),
    )
    return dict(bboxes), layer


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


# --- fallbacks --------------------------------------------------------------


def _assert_fell_back(cache, counts) -> None:
    cache.render_rect(0, 0, *cache.canvas_dims(0), mip=0)  # Stepped's wholesale path is lazy
    assert counts["building_bboxes"] >= 1, "this edit should have taken the wholesale fallback"


@pytest.mark.parametrize("style", STYLES)
def test_a_shared_own_tile_falls_back(style, sprite_install, monkeypatch):  # noqa: F811
    scenario = _scenario()
    _place(scenario, 1, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    _place(scenario, 2, SPRITE_CONST, UNIT_TILE[0] + 0.5, UNIT_TILE[1] + 0.5)
    cache = _make_cache(style, scenario, sprites=True)
    counts = _call_counts(monkeypatch)

    _raise(style, cache, scenario, [UNIT_TILE])

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
    together, the way the app composes them."""
    window = conftest.terrain_edit_window()
    try:
        scenario = window.scenario
        mm = scenario.map_manager
        mm.get_tile(*BUMP_TILE).elevation = 1
        units = window._ensure_unit_edits()
        for x, y in ((32.5, 40.5), (36.5, 37.5), (40.5, 43.5), (47.5, 40.5)):
            units.add(1, SPRITE_CONST, x, y)
        units.add(2, MILL_CONST, 44.0, 38.0)
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
