"""Verifies descape.terrain_palette.resolve_player_colors() and its wiring
into scenario_io.LoadedScenario.

DEscape used to render every unit by PLAYER_COLORS[player_id], assuming
player N always gets color N. Real scenarios pick each player's color
independently in the in-game editor and store it -- 17 of the 20 corpus
files render with wrong colors under that assumption. This module pins:
the pure resolver in isolation, the GAIA-first/player-first off-by-one it
has to get right for real sprites, the load-path wiring against the
shipped blank template, the render-path regression the wrong cache key
would have reintroduced, and a corpus-wide guard for the 17/20 finding
itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from test_invalidate_units_splice import _make_cache, _oracle

from descape import asset_source, render, unit_sprites
from descape.options_model import OptionsEditModel
from descape.player_fields import parse_player_field_id, player_field_id
from descape.render import tile_pixels_for_map
from descape.render_cache import FlatChunkCache
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units, refresh_player_colors
from descape.terrain_palette import PLAYER_COLOR_BY_ID, PLAYER_COLORS, resolve_player_colors
from descape.unit_model import UnitEditModel
from descape.unit_sprites import TEAM_COLORS

from test_unit_sprites import CONST, FILE_NAME, build_sld

IDENTITY_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

MAP_W = MAP_H = 8


# --- the pure resolver --------------------------------------------------


def test_identity_ids_round_trip_players_1_through_6():
    dots, _team_indices = resolve_player_colors(IDENTITY_IDS)
    assert dots[0] == PLAYER_COLORS[0]  # GAIA untouched
    for player_id in range(1, 7):
        assert dots[player_id] == PLAYER_COLORS[player_id]


def test_identity_ids_swap_gray_and_orange_for_players_7_and_8():
    """The one pair PLAYER_COLORS itself gets wrong: it was built assuming
    player N always gets color N, and labeled its last two RGBs "P7 orange,
    P8 white/gray" -- the reverse of the game's own ColorId order (GRAY=6,
    ORANGE=7). A correct reader of identity color ids must therefore NOT
    match PLAYER_COLORS at these two slots."""
    dots, _team_indices = resolve_player_colors(IDENTITY_IDS)
    assert dots[7] == PLAYER_COLORS[8]
    assert dots[8] == PLAYER_COLORS[7]


def test_a_permutation_maps_each_player_to_its_own_stored_color():
    # Measured on 0_June_Event_Scenario (maintainer doc): P1/P2 swapped,
    # P5 stores orange.
    color_ids = [1, 0, 2, 3, 7, 4, 5, 6]
    identity_dots, _ = resolve_player_colors(IDENTITY_IDS)
    dots, _team_indices = resolve_player_colors(color_ids)
    for player_id, color_id in enumerate(color_ids, start=1):
        assert dots[player_id] == identity_dots[color_id + 1]


def test_duplicate_stored_colors_are_preserved_not_deduped():
    # Measured on C2_ElCid_coop_1_v0_16 (maintainer doc): three players
    # share yellow (ColorId 3).
    color_ids = [1, 7, 3, 0, 0, 3, 3, 7]
    dots, _team_indices = resolve_player_colors(color_ids)
    assert dots[3] == dots[6] == dots[7]  # all three stored ColorId 3


def test_an_out_of_range_id_falls_back_to_that_players_identity_default():
    color_ids = [0, 0, 99, 0, 0, 0, 0, 0]
    dots, team_indices = resolve_player_colors(color_ids)
    assert dots[3] == PLAYER_COLORS[3]
    assert team_indices[3] == 3


def test_gaia_is_never_overridden():
    for color_ids in (IDENTITY_IDS, [7, 7, 7, 7, 7, 7, 7, 7], [99] * 8):
        dots, team_indices = resolve_player_colors(color_ids)
        assert dots[0] == PLAYER_COLORS[0]
        assert team_indices[0] == 0


def test_team_index_is_gaia_first_offset_not_the_bare_color_id():
    """The failure this guards: TEAM_COLORS is GAIA-first, so tinting with
    the bare ColorId (rather than ColorId + 1) would render every BLUE
    (ColorId 0) player with GAIA's own untinted white."""
    _dots, team_indices = resolve_player_colors(IDENTITY_IDS)
    assert team_indices[1] == 1
    assert TEAM_COLORS[team_indices[1]] == (0, 0, 255)  # real BLUE
    assert TEAM_COLORS[team_indices[1]] != TEAM_COLORS[0]  # not GAIA's white


# --- load-path integration ----------------------------------------------


def test_blank_template_resolves_via_the_same_identity_path():
    """descape/templates/blank_120x120.aoe2scenario is tracked and stores
    the default [0..7] color ids -- the one binary fixture this repo can
    pin a default-tier load-path assertion against (examples/ is
    gitignored)."""
    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    expected_dots, expected_team_indices = resolve_player_colors(IDENTITY_IDS)
    assert scenario.player_colors == expected_dots
    assert scenario.team_indices == expected_team_indices


# --- render-path regression guard ----------------------------------------


@dataclass
class _Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, units_by_player, team_index_for_player_1):
        tiles = [_Tile(x, y, 0) for y in range(MAP_H) for x in range(MAP_W)]
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = (0, team_index_for_player_1, *range(2, 9))


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(1))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 1,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_two_scenarios_player_1_never_share_a_cached_tint(sprite_install):
    """The regression this whole plan closes: render.py used to pass a
    scenario-agnostic player_id into unit_sprites, so any two scenarios'
    player 1 always resolved to the SAME tint no matter what color each
    file actually stored. With team_indices threaded through
    (sprite_draws_by_anchor's real call site), the same player_id=1 slot
    across two scenarios with different stored colors must render two
    genuinely different sprite tints -- and, since unit_sprites' scaled
    cache is a process-global LRU, must not silently serve one scenario's
    cached tint to the other."""
    blue_scn = _Scenario([[], [_Unit(4.0, 4.0, CONST)]], team_index_for_player_1=1)
    red_scn = _Scenario([[], [_Unit(4.0, 4.0, CONST)]], team_index_for_player_1=2)

    _, blue_elev, blue_proj = render.render_terrain_iso_with_proj(blue_scn, with_sprites=True)
    blue_sprites = render.sprite_draws_by_anchor(blue_scn, blue_proj, blue_elev)
    _, red_elev, red_proj = render.render_terrain_iso_with_proj(red_scn, with_sprites=True)
    red_sprites = render.sprite_draws_by_anchor(red_scn, red_proj, red_elev)

    (blue_draw, _ax, _ay), = next(iter(blue_sprites.by_anchor.values()))
    (red_draw, _ax, _ay), = next(iter(red_sprites.by_anchor.values()))
    assert not np.array_equal(blue_draw.rgba, red_draw.rgba)


# --- corpus regression guard ---------------------------------------------


@pytest.mark.corpus
def test_corpus_still_has_wrong_default_colors_on_most_files(corpus_files):
    """Guards the guard: the whole point of this feature is that most real
    files DON'T store the identity assignment. If a future corpus swap (or
    a regression back to the identity path) ever made every file resolve
    to [0..7], this would go quiet while the bug it exists to catch came
    back."""
    non_identity = 0
    for path in corpus_files:
        scenario = load_map_and_units(str(path))
        if scenario.player_colors != resolve_player_colors(IDENTITY_IDS)[0]:
            non_identity += 1
    # Measured 17/20 (85%) on the full corpus; the quick 3-file subset
    # (conftest.QUICK_CORPUS_NAMES) happens to be 3/3. 80% is comfortably
    # below both -- loud regardless of which subset ran -- while still
    # catching a regression back to the identity path (which would read 0%).
    assert non_identity / len(corpus_files) >= 0.8, (
        f"only {non_identity}/{len(corpus_files)} corpus files have a non-default "
        "color assignment -- expected at least ~85% per the maintainer doc"
    )


# --- a colour edit reaching the render (scenario_io.refresh_player_colors) ---
#
# Three gaps had to close for a Players-tab colour edit to appear on the map:
# the derived tuples were computed once at load and never again, nothing
# invalidated the render caches for a Players-mode edit, and both panels'
# swatch combos are built behind a `same_document` guard that is True on
# every edit. These pin the first, plus the recompute-before-invalidate
# ordering the second depends on.


def _blank_with_options():
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    return loaded, OptionsEditModel(loaded)


def _pending_colors(model) -> dict[int, int]:
    """viewer._pending_player_values()["color"], rebuilt here so these stay
    Qt-free -- same parse, one field."""
    out = {}
    for key, value in model.pending_values().items():
        if key.startswith("player:"):
            field_id, player_id = parse_player_field_id(key)
            if field_id == "color":
                out[player_id] = value
    return out


def test_a_colour_edit_re_derives_both_tuples_and_undoing_it_reverts_them():
    loaded, model = _blank_with_options()
    before_dots, before_teams = loaded.player_colors, loaded.team_indices
    assert before_dots[1] == PLAYER_COLOR_BY_ID[0]

    model.set_value(player_field_id("color", 1), 5)
    assert refresh_player_colors(loaded, _pending_colors(model)) is True
    assert loaded.player_colors[1] == PLAYER_COLOR_BY_ID[5]
    # team_indices is GAIA-first, so the sprite tint index is the id + 1 --
    # asserting only player_colors would pass on a half-fix that left every
    # unit's sprite tinted with the pre-edit colour.
    assert loaded.team_indices[1] == 6
    assert loaded.player_colors[2:] == before_dots[2:]

    # Undo, as the model sees it: the edit drops back out of pending_values()
    # once the field is set back to what the file stores.
    model.set_value(player_field_id("color", 1), 0)
    assert refresh_player_colors(loaded, _pending_colors(model)) is True
    assert loaded.player_colors == before_dots
    assert loaded.team_indices == before_teams


def test_a_non_colour_player_edit_reports_no_change():
    """What gates the invalidate: set_player_field() runs this tail for gold,
    wood, lock_personality and the rest, and _move_history()'s "options"
    branch fires for Map Options and Diplomacy edits too. Reporting a change
    there would evict the whole canvas on a gold-amount edit."""
    loaded, model = _blank_with_options()
    model.set_value(player_field_id("gold", 1), 1234)
    assert refresh_player_colors(loaded, _pending_colors(model)) is False


def test_an_out_of_range_pending_id_falls_back_rather_than_raising():
    loaded, _model = _blank_with_options()
    refresh_player_colors(loaded, {1: 99})
    assert loaded.player_colors[1] == PLAYER_COLORS[1]
    assert loaded.team_indices[1] == 1


def _flat_cache_pixels(loaded, sprites: bool):
    mm = loaded.map_manager
    tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
    cache = FlatChunkCache(loaded, tile_px, sprites=sprites)
    w, h = cache.canvas_dims()
    return cache, cache.render_rect(0, 0, w, h).copy()


def _recolour_and_recomposite(cache, loaded, model, sprites: bool):
    """The exact order viewer._after_player_color_change() uses: recompute
    first, THEN invalidate. FlatChunkCache.invalidate_units() is the eager-
    rebuild override, so swapping these two lines re-derives the draws from
    the stale tuples and bakes the pre-edit colour straight back in -- this
    helper is what the ordering assertion below rides on."""
    refresh_player_colors(loaded, _pending_colors(model))
    cache.invalidate_units()
    w, h = cache.canvas_dims()
    cache.invalidate_region((0, 0, w, h))
    return cache.render_rect(0, 0, w, h).copy()


def test_a_colour_edit_repaints_the_flat_unit_dots():
    """The load-bearing one. A green model-level test is not evidence of a
    fix here: the unit-mutation version of this same bug shipped with a
    green tier because nothing asserted on pixels
    (viewer._after_unit_mutation()'s own docstring). No ViewerWindow and no
    gui marker, deliberately -- a test that can skip is the worst possible
    shape for the one assertion pinning this ordering."""
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    old_rgb, new_rgb = PLAYER_COLOR_BY_ID[0], PLAYER_COLOR_BY_ID[5]

    cache, before = _flat_cache_pixels(loaded, sprites=False)
    assert np.any(np.all(before == old_rgb, axis=-1))

    model.set_value(player_field_id("color", 1), 5)
    after = _recolour_and_recomposite(cache, loaded, model, sprites=False)

    assert np.any(np.all(after == new_rgb, axis=-1))
    assert not np.any(np.all(after == old_rgb, axis=-1))


def test_a_colour_edit_repaints_the_flat_sprite_tint(sprite_install):
    """The sprite half of the same pixel assertion: the icon layer reads
    team_indices, the dots read player_colors, and they are separate
    readers -- a fix that re-derived only player_colors passes the dot test
    above while leaving every sprite tinted with the pre-edit colour."""
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)

    cache, before = _flat_cache_pixels(loaded, sprites=True)

    model.set_value(player_field_id("color", 1), 5)
    after = _recolour_and_recomposite(cache, loaded, model, sprites=True)

    assert not np.array_equal(before, after)


# --- the same recolour on the Stepped and Sloped caches -----------------
#
# The Flat tests above pin the ordering; these pin that the same production
# tail leaves an Iso/Sloped cache byte-identical to a fresh render.


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_colour_edit_repaints_the_iso_unit_dots_like_a_fresh_render(style):
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    cache = _make_cache(style, loaded)
    w, h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, w, h, mip=0).copy()

    model.set_value(player_field_id("color", 1), 5)
    after = _recolour_and_recomposite(cache, loaded, model, sprites=False)

    assert not np.array_equal(after, before), "the recolour did not reach the canvas"
    assert np.array_equal(after, _oracle(style, loaded)[:h, :w])


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_a_colour_edit_retints_the_iso_sprite_like_a_fresh_render(style, sprite_install):
    loaded, model = _blank_with_options()
    UnitEditModel(loaded).add(1, CONST, 4.0, 4.0)
    cache = _make_cache(style, loaded, sprites=True)
    w, h = cache.canvas_dims(0)
    before = cache.render_rect(0, 0, w, h, mip=0).copy()

    model.set_value(player_field_id("color", 1), 5)
    after = _recolour_and_recomposite(cache, loaded, model, sprites=True)

    assert not np.array_equal(after, before), "the recolour did not retint the sprite"
    assert np.array_equal(after, _oracle(style, loaded, sprites=True)[:h, :w])
