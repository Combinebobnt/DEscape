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

from descape import asset_source, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import PLAYER_COLORS, resolve_player_colors
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
        self.team_indices = (0, team_index_for_player_1) + tuple(range(2, 9))


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
