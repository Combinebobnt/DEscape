"""A memo-built SpriteLayer must equal a fresh sprite_draws_by_anchor() after
any edit, and must actually reuse the contributions the edit didn't touch
(2026-09-25 sprite-contribution memo plan, Step 5).

Every case builds a memo on the starting state, applies one change, walks
again with that memo, and compares against a memo-free walk of the changed
state. Draws are compared by pixels and hotspot rather than identity, since
a fresh resolve may hand back a different (equal) SpriteDraw object. The
reuse half counts _resolve_unit_sprite() calls: a memo that re-resolved
everything would still match, so equality alone would pass vacuously.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sprite_edit_bbox import CONST, Unit, sprite_install  # noqa: F401 -- fixture

from descape import iso_geometry, render
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter
from descape.view_layers import LayerState

# A const with a foundation terrain and no sprite in the fixture's graphic map: a farm.
FARM = min(c for c in render.FOUNDATION_TERRAIN if c not in render.DRAPED_SPRITE_CONSTS)
NO_ART = 999_999
_DEFAULT_LAYERS = LayerState()

# Every test reads the fixture's synthetic install.
pytestmark = pytest.mark.usefixtures("sprite_install")


def _scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in scenario.map_manager.terrain:
        tile.elevation = 2
    units = scenario.unit_manager.units
    units[1] += [Unit(40.5, 40.5, CONST), Unit(44.5, 41.5, CONST, rotation=1.0), Unit(50.0, 50.0, FARM)]
    units[2] += [Unit(60.25, 60.75, CONST), Unit(62.5, 62.5, NO_ART)]
    units[0] += [Unit(70.5, 70.5, CONST)]
    return scenario


def _elevations(scenario) -> np.ndarray:
    mm = scenario.map_manager
    return np.array([[int(mm.get_tile(x, y).elevation) for x in range(mm.map_width)] for y in range(mm.map_height)])


class _Style:
    """Stepped or Sloped walk arguments, re-derived from the scenario on demand."""

    def __init__(self, sloped: bool, scenario):
        self.sloped = sloped
        self.scenario = scenario
        _, self.proj = render.elevations_and_proj(scenario)
        if sloped:
            _, _, self.proj = render.sloped_elevations_and_proj(scenario)

    def args(self):
        elevations = _elevations(self.scenario)
        corner_rise = (
            iso_geometry.corner_rise_px(elevations, self.proj, rule=render.SLOPE_CORNER_RULE) if self.sloped else None
        )
        return elevations, corner_rise


def _normalized(layer: render.SpriteLayer):
    by_anchor = {
        key: [(d.rgba.shape, d.rgba.tobytes(), d.hotspot_x, d.hotspot_y, x, y) for d, x, y in draws]
        for key, draws in layer.by_anchor.items()
    }
    return by_anchor, layer.bboxes, layer.skip_ids, layer.farm_by_tile


class _ResolveCounter:
    def __init__(self, monkeypatch):
        self.calls = 0
        real = render._resolve_unit_sprite

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(render, "_resolve_unit_sprite", counted)


def _walk(style: _Style, memo, unit_filter=UnitFilter(), with_farms=True, tree_scale=1.0, hero_glow=False):
    elevations, corner_rise = style.args()
    return render._drain(render.sprite_draws_by_anchor_sliced(
        style.scenario, style.proj, elevations, unit_filter, corner_rise, with_farms, tree_scale, hero_glow,
        memo=memo,
    ))


def _fresh(style: _Style, unit_filter=UnitFilter(), with_farms=True, tree_scale=1.0, hero_glow=False):
    elevations, corner_rise = style.args()
    return render.sprite_draws_by_anchor(
        style.scenario, style.proj, elevations, unit_filter, corner_rise, with_farms, tree_scale, hero_glow
    )


def _units(scenario):
    return scenario.unit_manager.units


def _move(s):
    _units(s)[1][0].x += 1.0


def _rotate(s):
    _units(s)[1][1].rotation = 2.0


def _const_change(s):
    _units(s)[2][1].unit_const = CONST


def _const_to_farm(s):
    _units(s)[1][0].unit_const = FARM


def _reassign(s):
    _units(s)[3].append(_units(s)[1].pop(0))


def _elevation_under_unit(s):
    s.map_manager.get_tile(40, 40).elevation = 3


def _elevation_beside_unit(s):
    # A neighbour's elevation moves Sloped's corners under (40, 40), never Stepped's own tile.
    s.map_manager.get_tile(41, 41).elevation = 3


def _player_color(s):
    colors = list(s.player_colors)
    colors[1] = (1, 2, 3)
    s.player_colors = tuple(colors)


def _team_index(s):
    indices = list(s.team_indices)
    indices[2] = (indices[2] + 1) % 8
    s.team_indices = tuple(indices)


def _delete(s):
    del _units(s)[2][0]


def _add(s):
    _units(s)[1].append(Unit(80.5, 80.5, CONST))


# (edit, resolves expected on the second walk: Stepped, Sloped)
EDITS = {
    "none": (lambda s: None, 0, 0),
    "move": (_move, 1, 1),
    "rotate": (_rotate, 1, 1),
    "const_change": (_const_change, 1, 1),
    "const_to_farm": (_const_to_farm, 1, 1),
    "reassign": (_reassign, 1, 1),
    "elevation_under_unit": (_elevation_under_unit, 1, 1),
    "elevation_beside_unit": (_elevation_beside_unit, 0, 1),
    "player_color": (_player_color, 3, 3),
    "team_index": (_team_index, 2, 2),
    "delete": (_delete, 0, 0),
    "add": (_add, 1, 1),
}


@pytest.mark.parametrize("sloped", [False, True], ids=["stepped", "sloped"])
@pytest.mark.parametrize("edit", list(EDITS))
def test_memo_layer_equals_a_fresh_one_and_resolves_only_the_edited_units(monkeypatch, sloped, edit):
    scenario = _scenario()
    style = _Style(sloped, scenario)
    _layer, memo = _walk(style, render.SpriteMemo())
    fn, stepped_misses, sloped_misses = EDITS[edit]
    fn(scenario)
    counter = _ResolveCounter(monkeypatch)
    layer, new_memo = _walk(style, memo)
    assert counter.calls == (sloped_misses if sloped else stepped_misses)
    assert _normalized(layer) == _normalized(_fresh(style))
    assert set(new_memo.entries) == {id(u) for units in _units(scenario) for u in units}


def test_the_fixture_draws_real_sprites_and_a_farm():
    """Non-vacuity: the cases above compare layers that actually hold sprites and a farm."""
    layer = _fresh(_Style(False, _scenario()))
    assert sum(len(d) for d in layer.by_anchor.values()) == 4
    assert layer.farm_by_tile


def test_a_wall_override_change_reresolves_the_unit(monkeypatch):
    scenario = _scenario()
    style = _Style(False, scenario)
    overrides: dict = {}
    monkeypatch.setattr(render, "wall_variant_rotation_overrides", lambda _s: overrides)
    _layer, memo = _walk(style, render.SpriteMemo())
    overrides[(1, 0)] = 3.0
    counter = _ResolveCounter(monkeypatch)
    layer, _ = _walk(style, memo)
    assert counter.calls == 1
    assert _normalized(layer) == _normalized(_fresh(style))


def test_a_filter_change_matches_fresh_without_reresolving(monkeypatch):
    """The filter is evaluated per walk, outside the memo, so hiding and
    re-showing a player costs no resolves at all."""
    scenario = _scenario()
    style = _Style(False, scenario)
    _layer, memo = _walk(style, render.SpriteMemo())
    hide = UnitFilter(players=frozenset({0, 2}))
    counter = _ResolveCounter(monkeypatch)
    hidden, hidden_memo = _walk(style, memo, unit_filter=hide)
    shown, _ = _walk(style, hidden_memo)
    assert counter.calls == 0
    assert hidden_memo.entries.keys() == memo.entries.keys()
    assert _normalized(hidden) == _normalized(_fresh(style, unit_filter=hide))
    assert _normalized(shown) == _normalized(_fresh(style))


@pytest.mark.parametrize(
    "toggle",
    [{"with_farms": False}, {"tree_scale": 0.5}, {"hero_glow": True}],
    ids=["farm_overlay", "small_trees", "hero_glow"],
)
def test_a_layer_toggle_matches_fresh(monkeypatch, toggle):
    """Each build-time layer field is a memo invariant: flipping it must
    re-resolve rather than serve the old art. CONST is patched into the tree
    and hero sets so the two toggles change pixels at all."""
    monkeypatch.setattr(render, "TREE_UNIT_IDS", frozenset({CONST}))
    monkeypatch.setattr(render, "HERO_GLOW_CONSTS", frozenset({CONST}))
    scenario = _scenario()
    style = _Style(False, scenario)
    _layer, memo = _walk(style, render.SpriteMemo())
    toggled, _ = _walk(style, memo, **toggle)
    fresh = _fresh(style, **toggle)
    assert _normalized(fresh) != _normalized(_fresh(style)), "the toggle must change the layer"
    assert _normalized(toggled) == _normalized(fresh)


def test_a_reused_id_with_a_different_unit_is_a_miss(monkeypatch):
    """An entry whose unit isn't this one (id reuse after a delete) must not
    be served. Injected, since the memo's own reference keeps the id alive."""
    scenario = _scenario()
    style = _Style(False, scenario)
    _layer, memo = _walk(style, render.SpriteMemo())
    target = _units(scenario)[1][0]
    _unit, key, _contribution = memo.entries[id(target)]
    bogus = render._SpriteContribution(skip_id=id(target), by_anchor={}, bboxes={}, farm_tiles={})
    memo.entries[id(target)] = (Unit(target.x, target.y, CONST), key, bogus)
    counter = _ResolveCounter(monkeypatch)
    layer, _ = _walk(style, memo)
    assert counter.calls == 1
    assert _normalized(layer) == _normalized(_fresh(style))


def test_no_memo_returns_the_layer_alone():
    style = _Style(False, _scenario())
    assert isinstance(_walk(style, None), render.SpriteLayer)


def _cache(sloped: bool, scenario, layers=_DEFAULT_LAYERS):
    mm = scenario.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    if sloped:
        elevations, corner_rise, proj = render.sloped_elevations_and_proj(scenario)
        return SlopedChunkCache(scenario, elevations, corner_rise, proj, tile_px, sprites=True, layers=layers)
    elevations, proj = render.elevations_and_proj(scenario)
    return IsoChunkCache(scenario, elevations, proj, tile_px, sprites=True, layers=layers)


def _cache_layer(cache, sloped: bool) -> render.SpriteLayer:
    return cache.sprites if sloped else cache._level(0).sprites


def _cache_fresh(cache, sloped: bool) -> render.SpriteLayer:
    proj = cache.proj if sloped else cache._levels[0].proj
    return render.sprite_draws_by_anchor(
        cache.scenario, proj, cache.elevations, cache.unit_filter,
        corner_rise=cache.corner_rise if sloped else None, with_farms=cache.layers.farm_overlay,
        tree_scale=cache.layers.tree_scale, hero_glow=cache.layers.hero_glow,
    )


@pytest.mark.parametrize("sloped", [False, True], ids=["stepped", "sloped"])
def test_the_caches_wholesale_rebuild_reuses_the_memo(monkeypatch, sloped):
    scenario = _scenario()
    cache = _cache(sloped, scenario)
    _cache_layer(cache, sloped)
    _move(scenario)
    counter = _ResolveCounter(monkeypatch)
    cache.invalidate_units()
    layer = _cache_layer(cache, sloped)
    assert counter.calls == 1
    assert _normalized(layer) == _normalized(_cache_fresh(cache, sloped))


@pytest.mark.parametrize("sloped", [False, True], ids=["stepped", "sloped"])
def test_the_caches_layer_toggle_rebuilds_against_the_new_layers(monkeypatch, sloped):
    monkeypatch.setattr(render, "TREE_UNIT_IDS", frozenset({CONST}))
    scenario = _scenario()
    cache = _cache(sloped, scenario)
    before = _normalized(_cache_layer(cache, sloped))
    cache.set_layers(LayerState(small_trees=True))
    after = _normalized(_cache_layer(cache, sloped))
    assert after != before
    assert after == _normalized(_cache_fresh(cache, sloped))


def test_the_warm_installs_its_memo(monkeypatch):
    scenario = _scenario()
    cache = _cache(False, scenario)
    mip = next(m for m in cache.mip_levels() if m != 0)
    job = cache.level_warm_job(mip)
    assert job.install(render._drain(job.gen)) is True
    assert cache._levels[mip].memo is not None and cache._levels[mip].memo.entries
    _move(scenario)
    cache.invalidate_units()
    counter = _ResolveCounter(monkeypatch)
    cache._level(mip)
    assert counter.calls == 1


def test_sprites_off_drops_the_memo():
    scenario = _scenario()
    cache = _cache(False, scenario)
    cache._level(0)
    cache.set_sprites_enabled(False)
    assert cache._levels[0].memo is None
    sloped = _cache(True, scenario)
    sloped.set_sprites_enabled(False)
    assert sloped._sprite_memo is None


def test_patch_area_counts_what_patch_recomposites(monkeypatch):
    """viewer._if_spliceable() prices the eager patch with patch_area(); it
    must be exactly the pixel count patch() hands to _composite_rect()."""
    scenario = _scenario()
    cache = _cache(False, scenario)
    w, h = cache.canvas_dims(0)
    cache.render_rect(0, 0, w // 2, h // 2, mip=0)
    other = next(m for m in cache.mip_levels() if m != 0)
    ow, oh = cache.canvas_dims(other)
    cache.render_rect(0, 0, ow, oh, mip=other)
    bbox = (w // 5, h // 5, w * 3 // 5, h * 2 // 5)
    composited = []
    real = cache._composite_rect

    def counted(mip, x0, y0, x1, y1):
        composited.append((x1 - x0) * (y1 - y0))
        return real(mip, x0, y0, x1, y1)

    monkeypatch.setattr(cache, "_composite_rect", counted)
    expected = cache.patch_area(bbox)
    cache.patch(bbox, elevation_changed=set())
    assert expected == sum(composited) > 0
