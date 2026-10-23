"""GH #241: the game's _x2 (UHD) art, drawn at Graphics Quality Enhanced where
x1 would be enlarged, on synthetic x1/x2 pairs in a tmp install.

x1 files here are 8 px frames with the hotspot at 4, x2 ones 16 px at 8, so a
correct x2 draw has x1's size and hotspot. colour_base gives every file its
own per-frame colours, so a draw names the file and frame it came from.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from test_mod_graphics import PID, _mod

from descape import asset_source, debug_log, mod_graphics, render, unit_sprites
from descape.sld_decoder import SLDFile

import conftest
from test_unit_sprites import CONST, FILE_NAME, build_sld, install  # noqa: F401 -- a fixture

X2_NAME = FILE_NAME[:-3] + "_x2"
X1_BASE, X2_BASE, MOD_X1_BASE, MOD_X2_BASE = 0, 10, 20, 25
ENLARGED, NATIVE, SHRUNK = 64, 48, 32  # half_w: sprite_scale 1.33, 1.0, 0.67


def _pair(install, x2_frames: int = 4, x2: bool = True) -> None:  # noqa: F811
    install.write(build_sld(4, colour_base=X1_BASE))
    if x2:
        install.write(build_sld(x2_frames, canvas=16, colour_base=X2_BASE), X2_NAME)
    install.register(angle_count=4, frame_count=1)


def _red(base: int, frame: int = 0) -> int:
    """The red byte of frame `frame` in a file built with colour_base `base`."""
    return int(SLDFile(build_sld(frame + 1, colour_base=base)).decode_frame(frame).main[0, 0, 0])


def _draw(half_w: int, rotation: float = 0.0):
    return unit_sprites.sprite_for(CONST, rotation, 0, half_w)


def _frame_index(rotation: float) -> int:
    return unit_sprites._frame_for(CONST, unit_sprites.graphic_map()[CONST], rotation)


# --- the flag ----------------------------------------------------------------


def test_set_hi_res_reports_a_change_and_bumps_the_generation_only_then():
    start = unit_sprites.hi_res_generation()
    assert not unit_sprites.set_hi_res(False)
    assert unit_sprites.set_hi_res(True) and unit_sprites.hi_res()
    assert not unit_sprites.set_hi_res(True)
    assert unit_sprites.hi_res_generation() == start + 1


def test_x2_stem():
    assert unit_sprites._x2_stem("u_vil_x1") == "u_vil_x2"
    assert unit_sprites._x2_stem("u_vil_x2") is None
    assert unit_sprites._x2_stem("p_butterfly") is None


# --- what is drawn -----------------------------------------------------------


@pytest.mark.parametrize("half_w", [ENLARGED, NATIVE, SHRUNK])
def test_off_is_byte_identical_to_an_install_without_x2(install, half_w):  # noqa: F811
    _pair(install, x2=False)
    want = _draw(half_w)
    install.write(build_sld(4, canvas=16, colour_base=X2_BASE), X2_NAME)
    unit_sprites.clear_caches()
    got = _draw(half_w)
    assert np.array_equal(got.rgba, want.rgba) and (got.hotspot_x, got.hotspot_y) == (want.hotspot_x, want.hotspot_y)


def test_on_draws_x2_where_x1_would_be_enlarged_at_x1s_size_and_hotspot(install):  # noqa: F811
    _pair(install)
    x1 = _draw(ENLARGED)
    unit_sprites.set_hi_res(True)
    x2 = _draw(ENLARGED)
    assert x1.rgba[0, 0, 0] == _red(X1_BASE, 0) and x2.rgba[0, 0, 0] == _red(X2_BASE, 0)
    assert x2.rgba.shape == x1.rgba.shape
    assert abs(x2.hotspot_x - x1.hotspot_x) <= 1 and abs(x2.hotspot_y - x1.hotspot_y) <= 1


@pytest.mark.parametrize("half_w", [NATIVE, SHRUNK])
def test_on_draws_x1_where_x1_is_not_enlarged(install, half_w):  # noqa: F811
    _pair(install)
    unit_sprites.set_hi_res(True)
    assert _draw(half_w).rgba[0, 0, 0] == _red(X1_BASE, 0)


def test_x2_draws_the_frame_x1s_dispatch_picks(install):  # noqa: F811
    _pair(install)
    unit_sprites.set_hi_res(True)
    for rotation in (0.0, 1.6, 3.2):
        index = _frame_index(rotation)
        assert _draw(ENLARGED, rotation).rgba[0, 0, 0] == _red(X2_BASE, index), rotation


def test_the_flag_keys_the_cache_apart(install):  # noqa: F811
    _pair(install)
    colours = []
    for flag in (False, True, False):
        unit_sprites.set_hi_res(flag)
        colours.append(int(_draw(ENLARGED).rgba[0, 0, 0]))
    assert colours == [_red(X1_BASE), _red(X2_BASE), _red(X1_BASE)]


# --- fallbacks ---------------------------------------------------------------


def test_no_x2_file_draws_x1(install):  # noqa: F811
    _pair(install, x2=False)
    unit_sprites.set_hi_res(True)
    assert _draw(ENLARGED).rgba[0, 0, 0] == _red(X1_BASE)


def test_a_frame_count_mismatch_draws_x1_and_logs_once(install, monkeypatch):  # noqa: F811
    _pair(install, x2_frames=5)
    logged = []
    monkeypatch.setattr(debug_log, "log", logged.append)
    unit_sprites.set_hi_res(True)
    assert _draw(ENLARGED).rgba[0, 0, 0] == _red(X1_BASE)
    assert _draw(ENLARGED, 1.6).rgba[0, 0, 0] == _red(X1_BASE, _frame_index(1.6))
    assert [m for m in logged if X2_NAME in m and "frames" in m] == [logged[0]]


def test_an_unreadable_x2_file_draws_x1(install):  # noqa: F811
    _pair(install, x2=False)
    install.write(b"NOPE" + bytes(60), X2_NAME)
    unit_sprites.set_hi_res(True)
    assert _draw(ENLARGED).rgba[0, 0, 0] == _red(X1_BASE)


def test_a_particle_stem_never_gets_x2(install, monkeypatch):  # noqa: F811
    _pair(install)
    # lru_cached like the real one: clear_caches() calls cache_clear() on it.
    monkeypatch.setattr(unit_sprites, "_particle_stems", lru_cache(maxsize=1)(lambda: frozenset({FILE_NAME})))
    assert unit_sprites._x2_native_frame(FILE_NAME, 0) is None


def test_clear_caches_empties_every_x2_cache(install, monkeypatch):  # noqa: F811
    _pair(install, x2_frames=5)
    monkeypatch.setattr(debug_log, "log", lambda _m: None)
    unit_sprites.set_hi_res(True)
    _draw(ENLARGED)
    assert unit_sprites._x2_index_cache and unit_sprites._native_x2_cache and unit_sprites._x2_mismatch_logged
    unit_sprites.clear_caches()
    assert not unit_sprites._x2_index_cache and not unit_sprites._native_x2_cache
    assert not unit_sprites._x2_bytes_cache and not unit_sprites._x2_mismatch_logged


def test_leaving_enhanced_frees_the_x2_frames(install):  # noqa: F811
    _pair(install)
    unit_sprites.set_hi_res(True)
    _draw(ENLARGED)
    assert unit_sprites._x2_index_cache and unit_sprites._native_x2_cache
    unit_sprites.set_hi_res(False)
    assert not unit_sprites._x2_index_cache and not unit_sprites._native_x2_cache
    assert not unit_sprites._x2_bytes_cache


# --- same source as x1 -------------------------------------------------------


def test_x2_comes_from_the_folder_x1_does(install, tmp_path, monkeypatch):  # noqa: F811
    """Under Return of Rome, _sld_path() can resolve x1 to the RoR root (GH #67):
    x2 is read beside it, never from the AoE2 folder."""
    _pair(install)  # AoE2 folder: x1 and x2 at X1_BASE / X2_BASE
    ror = tmp_path / "ror_graphics"
    ror.mkdir()
    (ror / f"{FILE_NAME}.sld").write_bytes(build_sld(4, colour_base=MOD_X1_BASE))
    (ror / f"{X2_NAME}.sld").write_bytes(build_sld(4, canvas=16, colour_base=MOD_X2_BASE))
    real = unit_sprites._sld_path
    monkeypatch.setattr(
        unit_sprites, "_sld_path", lambda directory, name: ror / f"{name}.sld" if name == FILE_NAME else real(directory, name)
    )
    unit_sprites.set_hi_res(True)
    assert _draw(ENLARGED).rgba[0, 0, 0] == _red(MOD_X2_BASE)



def _profile(tmp_path: Path, monkeypatch, mods: dict[str, dict[str, bytes]]) -> None:
    """A profile whose mods are listed in `mods` order, highest priority first."""
    profile = tmp_path / "profile"
    for folder, files in mods.items():
        _mod(profile, folder, {f"drs/graphics/{name}.sld": data for name, data in files.items()})
    status = {"Mods": [
        {"Enabled": True, "Path": f"subscribed//{folder}", "Priority": i + 1} for i, folder in enumerate(mods)
    ]}
    (profile / PID / "mods" / "mod-status.json").write_text(json.dumps(status))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    mod_graphics.set_enabled(True)
    unit_sprites.clear_caches()


MOD_X1 = build_sld(4, colour_base=MOD_X1_BASE)
MOD_X2 = build_sld(4, canvas=16, colour_base=MOD_X2_BASE)


@pytest.mark.parametrize(
    ("mods", "want"),
    [
        ({"A": {FILE_NAME: MOD_X1}}, MOD_X1_BASE),  # a mod's x1 never gets the install's x2
        ({"A": {FILE_NAME: MOD_X1, X2_NAME: MOD_X2}}, MOD_X2_BASE),  # one mod's x1 and x2
        ({"A": {X2_NAME: MOD_X2}}, X1_BASE),  # the install's x1 never gets a mod's x2
        ({"A": {FILE_NAME: MOD_X1}, "B": {X2_NAME: MOD_X2}}, MOD_X1_BASE),  # two mods never mix
        # a mod x2 whose frame count differs from its x1 could map a different facing
        ({"A": {FILE_NAME: MOD_X1, X2_NAME: build_sld(1, canvas=16, colour_base=MOD_X2_BASE)}}, MOD_X1_BASE),
    ],
    ids=["mod-x1", "mod-x1-and-x2", "mod-x2-only", "two-mods", "mod-x2-frame-count"],
)
def test_x2_comes_from_where_x1_does(install, tmp_path, monkeypatch, mods, want):  # noqa: F811
    _pair(install)
    _profile(tmp_path, monkeypatch, mods)
    unit_sprites.set_hi_res(True)
    assert _draw(ENLARGED).rgba[0, 0, 0] == _red(want)


def test_mod_reach_counts_a_mods_x2_art_at_enhanced(monkeypatch):
    from types import SimpleNamespace

    wide = SimpleNamespace(frame_count=1, frames=[])
    monkeypatch.setattr(unit_sprites, "referenced_entries", lambda: [("b_wide_x1", 10, 0)])
    monkeypatch.setattr(unit_sprites, "_mod_index", lambda name: (wide, Path("w"), "sld") if name == "b_wide_x2" else None)
    # As _walked_reach() moves the anchor: a piece offset to the right shortens the left reach.
    monkeypatch.setattr(unit_sprites, "_walked_reach", lambda walked, fmt, dx, dy: (2001 - dx, 0, 0, 0))
    monkeypatch.setattr(unit_sprites, "sld_frame_count", lambda name: 1)
    mod_graphics.set_enabled(True)
    assert unit_sprites.sprite_reach()[0] == unit_sprites.MAX_SPRITE_REACH_LEFT
    unit_sprites.set_hi_res(True)
    assert unit_sprites.sprite_reach()[0] == 991  # (2001 - 2 * 10) x2 px, halved and rounded up


# --- composites, walls and Flat ----------------------------------------------


def _composite(install, x2_for_second: bool) -> None:  # noqa: F811
    second = "t_second_x1"
    install.write(build_sld(4, colour_base=X1_BASE))
    install.write(build_sld(4, canvas=16, colour_base=X2_BASE), X2_NAME)
    install.write(build_sld(4, colour_base=5), second)
    if x2_for_second:
        install.write(build_sld(4, canvas=16, colour_base=15), "t_second_x2")
    piece = {"unit_id": CONST, "angle_count": 4, "frame_count": 1}
    install.register_composite(CONST, [
        {**piece, "file_name": FILE_NAME, "dx": 0, "dy": 0},
        {**piece, "file_name": second, "dx": 12, "dy": -6},
    ])


def test_a_composite_piece_without_x2_stays_x1_at_the_same_offset(install):  # noqa: F811
    _composite(install, x2_for_second=False)
    before = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, ENLARGED)
    unit_sprites.set_hi_res(True)
    after = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, ENLARGED)
    assert [p.draw.rgba[0, 0, 0] for p in after] == [_red(X2_BASE), _red(5)]
    assert [(p.dx, p.dy) for p in after] == [(p.dx, p.dy) for p in before]


def test_translucent_walls_key_on_the_flag(install):  # noqa: F811
    _pair(install)
    entry = unit_sprites.graphic_map()[CONST]
    off = unit_sprites._wall_draw(CONST, entry, 0.0, 0, ENLARGED, None, 0.5)
    unit_sprites.set_hi_res(True)
    on = unit_sprites._wall_draw(CONST, entry, 0.0, 0, ENLARGED, None, 0.5)
    assert (off.rgba[0, 0, 0], on.rgba[0, 0, 0]) == (_red(X1_BASE), _red(X2_BASE))


def test_a_flat_icon_uses_x2_when_it_enlarges_x1_and_stays_in_its_footprint(install):  # noqa: F811
    _pair(install)
    assert unit_sprites.icon_for(CONST, 0.0, 0, 64, 40).rgba[0, 0, 0] == _red(X1_BASE)
    unit_sprites.set_hi_res(True)
    icon = unit_sprites.icon_for(CONST, 0.0, 0, 64, 40)
    assert icon.rgba[0, 0, 0] == _red(X2_BASE)
    assert icon.rgba.shape[0] <= 40 and icon.rgba.shape[1] <= 64
    small = unit_sprites.icon_for(CONST, 0.0, 0, 6, 6)  # x1 shrinks to fit: no x2
    assert small.rgba[0, 0, 0] == _red(X1_BASE)


@pytest.mark.parametrize("x2_for_second", [False, True], ids=["one-piece-lacks-x2", "every-piece-has-x2"])
def test_a_flat_composite_icon_is_x1_unless_every_piece_has_x2(install, x2_for_second):  # noqa: F811
    _composite(install, x2_for_second)
    x1 = unit_sprites.icon_for(CONST, 0.0, 0, 128, 128)
    unit_sprites.set_hi_res(True)
    on = unit_sprites.icon_for(CONST, 0.0, 0, 128, 128)
    assert np.array_equal(on.rgba, x1.rgba) is not x2_for_second


@pytest.mark.parametrize(("composite", "size"), [(False, 12), (True, 30)], ids=["single", "composite"])
def test_a_flat_icon_shrinking_x2_is_box_filtered_and_tinted_once(install, monkeypatch, composite, size):  # noqa: F811
    """The footprint fits the x1 art at 1.5x (8 px, or the composite's 20x14),
    so x2 at twice the size shrinks to 0.75."""
    if composite:
        _composite(install, x2_for_second=True)
    else:
        _pair(install)
    calls = []
    real = unit_sprites._box_resize_rgba
    monkeypatch.setattr(unit_sprites, "_box_resize_rgba", lambda *a: calls.append(a[1:]) or real(*a))
    unit_sprites.set_hi_res(True)
    icon = unit_sprites.icon_for(CONST, 0.0, 1, size, size)
    assert calls
    main, playercolor, _hx, _hy = unit_sprites._x2_native_frame(FILE_NAME, 0)
    once = unit_sprites._tinted(main[:1, :1], playercolor[:1, :1], unit_sprites.TEAM_COLORS[1])
    corner = icon.rgba[-1, 0] if composite else icon.rgba[0, 0]  # the first piece sits lower left
    assert corner.tolist() == once[0, 0].tolist()


def test_a_body_offset_lands_at_the_same_place_from_x2(install, monkeypatch):  # noqa: F811
    """A single-graphic entry's own dx/dy (a fish body, GH #178) is x1 px,
    doubled for x2, so the hotspot does not move between x1 and x2."""
    _pair(install)
    plain = _draw(ENLARGED)
    entry = dict(unit_sprites.graphic_map()[CONST], dx=3, dy=-5)
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {CONST: entry})
    unit_sprites.clear_caches()
    x1 = _draw(ENLARGED)
    assert (x1.hotspot_x, x1.hotspot_y) != (plain.hotspot_x, plain.hotspot_y), "the offset moved nothing -- vacuous"
    unit_sprites.set_hi_res(True)
    x2 = _draw(ENLARGED)
    assert x2.rgba[0, 0, 0] == _red(X2_BASE)
    assert abs(x2.hotspot_x - x1.hotspot_x) <= 1 and abs(x2.hotspot_y - x1.hotspot_y) <= 1


# --- the box filter ----------------------------------------------------------


def test_box_resize_keeps_a_flat_colour_and_adds_none_from_transparent_px():
    flat = np.full((9, 6, 4), 200, np.uint8)
    assert (unit_sprites._box_resize_rgba(flat, 4, 6) == 200).all()
    row = np.zeros((1, 3, 4), np.uint8)
    row[0, 0] = (255, 0, 0, 255)
    row[0, 1] = (0, 0, 255, 0)  # transparent blue must not tint its neighbours
    row[0, 2] = (0, 255, 0, 255)
    out = unit_sprites._box_resize_rgba(row, 2, 1)
    assert out[0, 0].tolist() == [255, 0, 0, 170] and out[0, 1].tolist() == [0, 255, 0, 170]


def test_box_resize_falls_back_to_nearest_on_a_growing_axis():
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, (4, 4, 4), dtype=np.uint8)
    assert np.array_equal(unit_sprites._box_resize_rgba(rgba, 6, 3), unit_sprites._resize_rgba(rgba, 6, 3))


def test_a_shrunk_x2_draw_is_box_filtered(install, monkeypatch):  # noqa: F811
    _pair(install)
    unit_sprites.set_hi_res(True)
    calls = []
    real = unit_sprites._box_resize_rgba
    monkeypatch.setattr(unit_sprites, "_box_resize_rgba", lambda *a: calls.append(a[1:]) or real(*a))
    _draw(ENLARGED)
    assert calls, "x2 at 2/3 went through the nearest gather"


# --- memos and the viewer ----------------------------------------------------


def test_the_sprite_memo_invariants_change_with_the_flag():
    from types import SimpleNamespace

    mm = SimpleNamespace(map_width=4, map_height=4)
    before = render._sprite_memo_invariants(None, mm, None, False, 1.0, False)
    unit_sprites.set_hi_res(True)
    assert render._sprite_memo_invariants(None, mm, None, False, 1.0, False) != before


_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]


def _dlc(monkeypatch, installed: bool) -> None:
    # lru_cached like the real one: clear_caches() calls cache_clear() on it.
    monkeypatch.setattr(unit_sprites, "x2_art_installed", lru_cache(maxsize=1)(lambda: installed))


@_GUI[0]
@_GUI[1]
@pytest.mark.parametrize("dlc", [True, False], ids=["dlc", "no-dlc"])
def test_the_quality_slider_sets_the_flag_and_names_the_dlc(qtbot, monkeypatch, dlc, install):  # noqa: F811
    """The flag follows Enhanced either way (each file falls back to x1);
    the DLC probe only picks the status-log wording."""
    from descape import settings

    _dlc(monkeypatch, dlc)
    dialog, window = conftest.dialog_and_window()
    logged = []
    monkeypatch.setattr(window, "_log_status", logged.append)
    try:
        assert not unit_sprites.hi_res()
        for value, want in ((settings.GRAPHICS_QUALITY_MAX, True), (settings.GRAPHICS_QUALITY_DEFAULT, False)):
            dialog.graphics_quality_slider.setValue(value)
            dialog._apply_graphics_quality()
            assert unit_sprites.hi_res() is want
        assert ("DLC not installed" in logged[0]) is not dlc
        assert "unit art" not in logged[1]  # Default says nothing about it
    finally:
        dialog.close()
        conftest.close_window(window)


@_GUI[0]
@_GUI[1]
@pytest.mark.parametrize("dlc", [True, False], ids=["dlc", "no-dlc"])
def test_a_window_opened_at_enhanced_turns_the_flag_on(qtbot, monkeypatch, dlc):
    from descape import settings
    from descape.viewer import ViewerWindow

    _dlc(monkeypatch, dlc)
    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_MAX)
    window = ViewerWindow()
    try:
        assert unit_sprites.hi_res()
    finally:
        conftest.close_window(window)


@_GUI[0]
@_GUI[1]
@pytest.mark.parametrize("ror", [True, False], ids=["ror-x2", "no-ror-x2"])
@pytest.mark.parametrize("dlc", [True, False], ids=["dlc", "no-dlc"])
def test_the_settings_say_whether_the_dlc_was_detected(qtbot, monkeypatch, tmp_path, dlc, ror):
    _dlc(monkeypatch, dlc)
    monkeypatch.setattr(unit_sprites, "ror_x2_art_installed", lru_cache(maxsize=1)(lambda: ror))
    monkeypatch.setattr(asset_source, "get_install_path", lambda: tmp_path)
    dialog, window = conftest.dialog_and_window()
    try:
        text = dialog.x2_art_label.text()
        assert ("DLC: detected" in text) is dlc and ("not detected" in text) is not dlc
        assert ("Return of Rome's own high-resolution art" in text) is ror
        assert bool(dialog.x2_art_label.styleSheet()) is dlc  # green only when detected
    finally:
        dialog.close()
        conftest.close_window(window)


def test_the_dlc_probe_is_the_villagers_x2_file(install, monkeypatch):  # noqa: F811
    _pair(install, x2=False)
    monkeypatch.setattr(unit_sprites, "X2_PROBE_FILE", f"{X2_NAME}.sld")
    assert not unit_sprites.x2_art_installed()
    install.write(build_sld(4, canvas=16), X2_NAME)
    assert not unit_sprites.x2_art_installed(), "cached until the art caches clear"
    unit_sprites.clear_caches()
    assert unit_sprites.x2_art_installed()


def test_the_ror_probe_is_any_x2_file_in_the_ror_folder(install):  # noqa: F811
    from descape import game_dataset

    ror_graphics = game_dataset.ror_root(asset_source.get_install_path()) / unit_sprites.GRAPHICS_SUBPATH
    ror_graphics.mkdir(parents=True)
    (ror_graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    assert not unit_sprites.ror_x2_art_installed()
    (ror_graphics / f"{X2_NAME}.sld").write_bytes(build_sld(4, canvas=16))
    unit_sprites.clear_caches()
    assert unit_sprites.ror_x2_art_installed()
    assert not unit_sprites.x2_art_installed(), "RoR's x2 is not the DLC"


@_GUI[0]
@_GUI[1]
@pytest.mark.parametrize("ror", [True, False], ids=["ror", "aoe2"])
def test_the_status_log_names_ror_art_under_ror(monkeypatch, ror, install):  # noqa: F811
    """RoR ships its own x2 (GH #241 test issue, 2026-10-20): under RoR the
    log says so rather than that the DLC is missing."""
    from descape import game_dataset, settings, viewer

    _dlc(monkeypatch, False)
    monkeypatch.setattr(unit_sprites, "ror_x2_art_installed", lru_cache(maxsize=1)(lambda: True))
    monkeypatch.setattr(game_dataset, "is_ror", lambda: ror)
    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_MAX)
    suffix = viewer._sync_hi_res()
    assert unit_sprites.hi_res()
    assert ("Return of Rome high-resolution unit art on" in suffix) is ror
    assert ("DLC not installed" in suffix) is not ror


@_GUI[0]
@_GUI[1]
def test_the_status_log_names_no_unit_art_without_an_install(monkeypatch):
    """With no install there is no unit art at all, so Enhanced does not blame a missing DLC."""
    from descape import settings, viewer

    monkeypatch.setattr(asset_source, "get_install_path", lambda: None)
    settings.set_graphics_quality(settings.GRAPHICS_QUALITY_MAX)
    assert viewer._sync_hi_res() == ""
    assert unit_sprites.hi_res()


@_GUI[0]
@_GUI[1]
@pytest.mark.parametrize("native", [True, False], ids=["native", "numpy"])
@pytest.mark.parametrize(("quality", "smooth"), [(1, False), (2, False), (3, True), (4, True)])
def test_default_and_above_shrink_smoothly(monkeypatch, quality, smooth, native):
    """GH #254: box-filtered shrinking from Default up, the nearest gather at
    Potato and below, set beside the x2 flag. Never on numpy (TASK-440's
    load budget)."""
    from descape import composite_backend, settings, viewer

    monkeypatch.setattr(settings, "_graphics_quality", quality)
    monkeypatch.setattr(composite_backend, "native", object() if native else None)
    viewer._sync_hi_res()
    assert unit_sprites.smooth_shrink() is (smooth and native)
    assert unit_sprites.hi_res() is (quality == settings.GRAPHICS_QUALITY_MAX)


# --- the real install --------------------------------------------------------


def test_a_real_villager_draws_x2_at_the_top_level():
    """Default tier, skipping without AOE2DE_INSTALL_PATH and the Enhanced Graphics
    pack (see test_sprite_edit_bbox's corpus test for why the env var): a real x2
    file is decoded whenever one is visible."""
    root = asset_source.get_install_path()
    if root is None or not os.path.isfile(Path(root) / unit_sprites.GRAPHICS_SUBPATH / "u_vil_male_villager_idleA_x2.sld"):
        pytest.skip("no AoE2:DE install with x2 art visible -- set AOE2DE_INSTALL_PATH")
    unit_sprites.clear_caches()
    x1 = unit_sprites.sprite_for(83, 0.0, 1, ENLARGED)
    unit_sprites.set_hi_res(True)
    x2 = unit_sprites.sprite_for(83, 0.0, 1, ENLARGED)
    assert unit_sprites._native_x2_cache, "x2 was never decoded"
    assert abs(x2.rgba.shape[0] - x1.rgba.shape[0]) <= 2 and abs(x2.rgba.shape[1] - x1.rgba.shape[1]) <= 2
    assert abs(x2.hotspot_x - x1.hotspot_x) <= 1 and abs(x2.hotspot_y - x1.hotspot_y) <= 1
    assert not np.array_equal(x2.rgba, x1.rgba)


@pytest.mark.corpus
@pytest.mark.parametrize("dataset", ["aoe2", "ror"])
def test_no_real_x2_sprite_reaches_past_the_committed_constants(dataset):
    """x2 reach / 2 against MAX_SPRITE_REACH_*, through tools/scan_x2_graphics.py's
    own scan (as test_sprite_edit_bbox's x1 corpus test): an x2 file a game
    patch grows past them would leave stale fragments at the top level only.
    Return of Rome's x2 art is scanned as the tool's --ror does."""
    from descape import game_dataset

    root = asset_source.get_install_path()
    if root is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH")
    game_dataset.set_active(dataset)
    scan = conftest.load_verify_module("scan_x2_graphics")
    result = scan.scan(Path(root) / unit_sprites.GRAPHICS_SUBPATH, check_ink=False)
    if not result["usable"]:
        pytest.skip("no usable x2 art on this install (Enhanced Graphics pack not installed)")
    for side, const, worst in zip(result["reach"], scan.REACH_CONSTS, result["reach_worst"], strict=True):
        assert side <= getattr(unit_sprites, const), f"{const}: x2 reaches {side} ({worst})"
