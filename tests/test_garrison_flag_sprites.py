"""A garrisoned host draws its garrison flag (GH #203, TASK-264).

unit_sprites.garrison_flag_pieces_for() turns an entry's `garrison_flags`
into SpritePieces, and render._resolve_unit_sprite() appends them to a host's
own pieces while render.garrison_host_ids() names it. Synthetic `.sld` art
throughout, so each check can tell the flag from the body by colour; the real
flag files are tests/test_garrison_flag_real_assets.py's.

The flag's synthetic graphic has three angles, one frame each, in three
colours. Real flags have one angle, which no rotation can move off frame 0;
three here make a host rotation that leaked into the flag's dispatch pick
another frame, so "always frame 0" is a check rather than a tautology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from descape import asset_source, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter

from test_unit_sprites import build_sld

HOST = 4242
OCCUPANT = 4243
FLAG_FILE = "t_garrison_flag_x1"
HOST_FILE = "t_host_x1"
OCCUPANT_FILE = "t_occupant_x1"
FLAG_DX, FLAG_DY = -12, -40
HOST_REF, OCCUPANT_REF = 500, 600
POINT = (40.5, 40.5)
NATIVE_HALF_W = unit_sprites.NATIVE_TILE_W // 2


def _flag(file_name: str = FLAG_FILE, dx: int = FLAG_DX, dy: int = FLAG_DY) -> dict:
    return {"unit_id": HOST, "file_name": file_name, "angle_count": 3, "frame_count": 1, "dx": dx, "dy": dy}


@pytest.fixture
def art(tmp_path, monkeypatch):
    """A host with four facings and one flag, an occupant, and their files.
    Returns the host's entry, so a test can change its flags."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{HOST_FILE}.sld").write_bytes(build_sld(4, canvas=32))
    (graphics / f"{FLAG_FILE}.sld").write_bytes(build_sld(3, canvas=16, colour_base=10))
    (graphics / f"{OCCUPANT_FILE}.sld").write_bytes(build_sld(1, canvas=16, colour_base=20))
    host = {
        "graphic_id": 1, "file_name": HOST_FILE, "angle_count": 4, "mirroring_mode": 0, "frame_count": 1,
        "garrison_flags": [_flag()],
    }
    occupant = {"graphic_id": 2, "file_name": OCCUPANT_FILE, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {HOST: host, OCCUPANT: occupant})
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield host
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def _frame_colour(file_name: str, frame: int, team: int = 0) -> tuple[int, ...]:
    """The centre colour of `file_name`'s frame `frame`, tinted for `team`."""
    main, playercolor, _hx, _hy = unit_sprites._native_frame(file_name, frame)
    tinted = unit_sprites._tinted(main, playercolor, unit_sprites.TEAM_COLORS[team])
    return _colour(tinted)


def _colour(rgba) -> tuple[int, ...]:
    return tuple(int(c) for c in rgba[rgba.shape[0] // 2, rgba.shape[1] // 2, :3])


# --- garrison_flag_pieces_for -------------------------------------------------


@pytest.mark.parametrize("half_w", [12, 24, 48, 96])
def test_a_flag_sits_at_its_offset_scaled_like_a_composite_piece(art, half_w):
    (piece,) = unit_sprites.garrison_flag_pieces_for(HOST, 1, half_w)
    scale = unit_sprites.sprite_scale(half_w)
    assert (piece.dx, piece.dy) == (round(FLAG_DX * scale), round(FLAG_DY * scale))
    assert piece.slot is None
    assert piece.draw.rgba.shape[:2] == (round(16 * scale),) * 2


def test_a_flag_is_frame_zero(art):
    (piece,) = unit_sprites.garrison_flag_pieces_for(HOST, 0, NATIVE_HALF_W)
    assert _colour(piece.draw.rgba) == _frame_colour(FLAG_FILE, 0)
    assert _frame_colour(FLAG_FILE, 0) != _frame_colour(FLAG_FILE, 1), "the fixture's frames must differ"


def test_a_flag_whose_file_fails_is_skipped_and_the_rest_still_draw(art):
    art["garrison_flags"] = [_flag("t_missing_x1"), _flag(dx=30, dy=-60)]
    pieces = unit_sprites.garrison_flag_pieces_for(HOST, 1, NATIVE_HALF_W)
    assert [(p.dx, p.dy) for p in pieces] == [(30, -60)]


def test_a_flag_takes_its_owners_team_colour(art):
    gaia, red, blue = (unit_sprites.garrison_flag_pieces_for(HOST, team, NATIVE_HALF_W)[0] for team in (0, 1, 2))
    assert not np.array_equal(gaia.draw.rgba, red.draw.rgba)
    assert not np.array_equal(red.draw.rgba, blue.draw.rgba)


def test_a_const_without_flags_or_an_entry_gets_none(art):
    assert unit_sprites.garrison_flag_pieces_for(OCCUPANT, 1, NATIVE_HALF_W) == []
    assert unit_sprites.garrison_flag_pieces_for(999_999, 1, NATIVE_HALF_W) == []


def test_the_body_and_icon_paths_never_draw_a_flag(art):
    """sprite_pieces_for() keeps its callers (icons, panel, placement ghost) flag-free."""
    (body,) = unit_sprites.sprite_pieces_for(HOST, 0.0, 1, NATIVE_HALF_W)
    assert (body.dx, body.dy) == (0, 0) and body.draw.rgba.shape[:2] == (32, 32)


# --- render._resolve_unit_sprite ----------------------------------------------


@dataclass
class _Unit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    garrisoned_in_id: int = -1
    rotation: float = 0.0


def _scenario(occupant_link: int = HOST_REF):
    """The blank template with player 1's host and player 2's occupant on one point."""
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    units = scenario.unit_manager.units
    units[1].append(_Unit(*POINT, HOST, HOST_REF))
    units[2].append(_Unit(*POINT, OCCUPANT, OCCUPANT_REF, garrisoned_in_id=occupant_link))
    return scenario


def _resolve(scenario, player_id: int = 1, unit_filter: UnitFilter = UnitFilter()):
    elevations, proj = render.elevations_and_proj(scenario)
    unit = scenario.unit_manager.units[player_id][-1]
    index = len(scenario.unit_manager.units[player_id]) - 1
    return render._resolve_unit_sprite(scenario, proj, elevations, unit_filter, None, {}, True, player_id, index, unit)


def _relink(scenario, link: int) -> None:
    """Re-points the occupant and bumps unit_gen, as UnitEditModel.set_garrisoned_in() does."""
    scenario.unit_manager.units[2][-1].garrisoned_in_id = link
    scenario.unit_gen += 1


def _draws(contribution) -> tuple[tuple[int, int], list]:
    """The one anchor key a 1x1 host's contribution has, and its draws."""
    ((key, draws),) = contribution.by_anchor.items()
    return key, draws


def test_a_host_with_an_occupant_paints_its_flag_after_its_body_at_its_anchor(art):
    scenario = _scenario()
    assert render.garrison_host_ids(scenario) == frozenset({HOST_REF})
    key, draws = _draws(_resolve(scenario))
    assert key == (int(POINT[0]), int(POINT[1]))
    (_body, bx, by, _), (flag, fx, fy, _) = draws
    scale = unit_sprites.sprite_scale(render.elevations_and_proj(scenario)[1].half_w)
    assert (fx - bx, fy - by) == (round(FLAG_DX * scale), round(FLAG_DY * scale))
    assert flag.rgba.shape[:2] == (round(16 * scale),) * 2


def test_the_flag_grows_the_hosts_bbox(art):
    scenario = _scenario()
    flagged = _resolve(scenario)
    _relink(scenario, -1)
    plain = _resolve(scenario)
    (key,) = plain.bboxes
    _, (flag, fx, fy, _) = flagged.by_anchor[key]
    h, w = flag.rgba.shape[:2]
    flag_box = (fx - flag.hotspot_x, fy - flag.hotspot_y, fx - flag.hotspot_x + w, fy - flag.hotspot_y + h)
    grown, before = flagged.bboxes[key], plain.bboxes[key]
    assert grown != before
    assert grown == (
        min(before[0], flag_box[0]), min(before[1], flag_box[1]),
        max(before[2], flag_box[2]), max(before[3], flag_box[3]),
    )


@pytest.mark.parametrize("link", [-1, OCCUPANT_REF], ids=["unlinked", "self_reference"])
def test_no_flag_without_a_real_occupant(art, link):
    """-1 and a unit naming itself both read as "not garrisoned", UnitFilter.matches()' rule."""
    scenario = _scenario(occupant_link=link)
    assert render.garrison_host_ids(scenario) == frozenset()
    _key, draws = _draws(_resolve(scenario))
    assert len(draws) == 1


def test_a_filtered_out_host_draws_nothing_flag_included(art):
    assert _resolve(_scenario(), unit_filter=UnitFilter(players=frozenset({0, 2}))) is None


def test_a_filter_hidden_occupant_still_raises_the_flag(art):
    """Show Garrisoned off hides the occupant, never the host's state (the user's 2026-10-12 call)."""
    scenario = _scenario()
    hidden = UnitFilter(show_garrisoned=False)
    assert _resolve(scenario, player_id=2, unit_filter=hidden) is None
    _key, draws = _draws(_resolve(scenario, unit_filter=hidden))
    assert len(draws) == 2


@pytest.mark.parametrize("rotation", [1.6, 3.2, 4.8])
def test_the_hosts_facing_turns_its_body_but_never_its_flag(art, rotation):
    """The body takes the facing's frame, so the facing reached this resolve;
    the flag keeps frame 0, where a leaked facing would pick frame 1 or 2."""
    scenario = _scenario()
    scenario.unit_manager.units[1][-1].rotation = rotation
    team = scenario.team_indices[1]
    _key, ((body, *_), (flag, *_)) = _draws(_resolve(scenario))
    facing = unit_sprites.angle_index(rotation, 4)
    assert facing != 0
    assert _colour(body.rgba) == _frame_colour(HOST_FILE, facing, team)
    assert _colour(flag.rgba) == _frame_colour(FLAG_FILE, 0, team)
    assert unit_sprites.angle_index(rotation, 3) != 0, "the fixture's flag would show a leaked facing"


def test_the_hosts_civ_art_picks_its_own_flags(art, monkeypatch):
    """The flags come from the entry the body draws from (resolve_entry())."""
    civ_entry = dict(art, garrison_flags=[_flag(dx=50, dy=-90)])
    monkeypatch.setattr(unit_sprites, "resolve_entry", lambda const, civ, age: civ_entry if civ == 7 else art)
    gaia = unit_sprites.garrison_flag_pieces_for(HOST, 1, NATIVE_HALF_W, art=(None, 2))
    civ = unit_sprites.garrison_flag_pieces_for(HOST, 1, NATIVE_HALF_W, art=(7, 2))
    assert [(p.dx, p.dy) for p in gaia] == [(FLAG_DX, FLAG_DY)]
    assert [(p.dx, p.dy) for p in civ] == [(50, -90)]
