"""descape/mod_graphics.py and its unit_sprites wiring, on a synthetic profile
tree under tmp_path. conftest's autouse fixture resets the module and drops
AOE2DE_PROFILE_PATH; each test points the variable at its own tree, except the
corpus byte-identity test, which needs this machine's real profile."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_smx_decoder import _PROFILE, _REQUIRE, DRAW, _file, _frame
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- fixture

from descape import asset_source, mod_graphics, render, slp_decoder, unit_sprites
from descape.scenario_io import load_map_and_units

import conftest

PID = "7656"


def _mod(profile: Path, folder: str, files: dict[str, bytes], kind: str = "subscribed", caps=False) -> Path:
    common = profile / PID / "mods" / kind / folder / ("Resources" if caps else "resources") / "_common"
    for rel, data in files.items():
        path = common / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return common


def _tree(tmp_path: Path, monkeypatch) -> Path:
    profile = tmp_path / "profile"
    _mod(profile, "1_High", {"drs/graphics/n_tree_x1.smx": b"high", "terrain/textures/2x/g_gr1.dds": b"t"})
    _mod(profile, "2_Low", {"drs/graphics/n_tree_x1.smx": b"low", "drs/graphics/b_hut_x1.sld": b"s",
                            "drs/graphics/b_hut_x1.smx": b"m"})
    _mod(profile, "9_Unlisted", {"drs/graphics/N_Rock_X1.SMX": b"u"}, caps=True)
    _mod(profile, "3_Off", {"drs/graphics/n_rock_x1.smx": b"off"})
    _mod(profile, "Loose", {"drs/graphics/u_x1.smx": b"l"}, kind="local")
    status = {"Mods": [
        {"Enabled": True, "Path": "subscribed//2_Low", "Priority": 2},
        {"Enabled": True, "Path": "subscribed//1_High", "Priority": 1},
        {"Enabled": False, "Path": "subscribed//3_Off", "Priority": 3},
        {"Enabled": True, "Path": "local//Loose", "Priority": 4},
    ]}
    (profile / PID / "mods" / "mod-status.json").write_text(json.dumps(status))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    return profile


def test_off_by_default_resolves_nothing(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    assert not mod_graphics.enabled()
    assert mod_graphics.resolve_sprite("n_tree_x1") is None
    assert mod_graphics.resolve_terrain_texture("g_gr1.dds") is None


def test_priority_order_formats_case_and_exclusions(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    start = mod_graphics.generation()
    assert mod_graphics.set_enabled(True) and mod_graphics.generation() > start
    assert mod_graphics.resolve_sprite("n_tree_x1")[0].read_bytes() == b"high"
    assert mod_graphics.resolve_sprite("b_hut_x1")[1] == "sld"  # .sld beats .smx inside one mod
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"u"  # disabled 3_Off skipped
    assert mod_graphics.resolve_terrain_texture("G_GR1.DDS").read_bytes() == b"t"
    mods = mod_graphics.mods_with_art()
    # GH #264: the game's disabled 3_Off is listed at its stored Priority, off.
    assert [m.status_path for m in mods] == [
        "subscribed//1_High", "subscribed//2_Low", "subscribed//3_Off", "local//Loose", "subscribed//9_Unlisted",
    ]
    assert [(m.in_game, m.drawn) for m in mods] == [(True, True)] * 2 + [(False, False)] + [(True, True)] * 2
    before = mod_graphics.generation()
    assert mod_graphics.set_excluded({"subscribed//1_High"}) and mod_graphics.generation() > before
    assert mod_graphics.resolve_sprite("n_tree_x1")[0].read_bytes() == b"low"
    assert mod_graphics.resolve_terrain_texture("g_gr1.dds") is None
    assert mod_graphics.mods_with_art()[0].excluded
    assert not mod_graphics.mods_with_art()[0].drawn
    assert not mod_graphics.set_excluded({"subscribed//1_High"})


def _drawn() -> dict[str, bool]:
    return {m.status_path: m.drawn for m in mod_graphics.mods_with_art()}


def test_a_disabled_mod_draws_only_when_included(tmp_path, monkeypatch):
    """GH #264: a mod the game disables is opt-in, at its stored Priority
    (3, above the unlisted mod); the opt-out set never touches it, and the
    opt-in set never touches a mod the game enables."""
    _tree(tmp_path, monkeypatch)
    mod_graphics.set_enabled(True)
    before = mod_graphics.generation()
    assert mod_graphics.set_included({"subscribed//3_Off"}) and mod_graphics.generation() > before
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"off"
    assert _drawn()["subscribed//3_Off"]
    assert not mod_graphics.set_included({"subscribed//3_Off"})

    mod_graphics.set_excluded({"subscribed//3_Off"})
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"off", "an opt-out of a disabled mod"
    assert _drawn()["subscribed//3_Off"]

    mod_graphics.set_excluded(())
    mod_graphics.set_included({"subscribed//1_High"})
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"u"
    assert mod_graphics.resolve_sprite("n_tree_x1")[0].read_bytes() == b"high"
    assert _drawn() == {
        "subscribed//1_High": True, "subscribed//2_Low": True, "subscribed//3_Off": False,
        "local//Loose": True, "subscribed//9_Unlisted": True,
    }


def _set_enabled_in_game(profile: Path, **enabled: bool) -> None:
    status_file = profile / PID / "mods" / "mod-status.json"
    status = json.loads(status_file.read_text())
    for mod in status["Mods"]:
        folder = mod["Path"].split("//")[1]
        if folder in enabled:
            mod["Enabled"] = enabled[folder]
    status_file.write_text(json.dumps(status))


def test_a_mod_flipped_in_the_game_keeps_the_choice_for_its_new_state(tmp_path, monkeypatch):
    """GH #264 Decision 3: the opt-out applies while the game enables a mod,
    the opt-in while it disables it, so a rescan after a flip in the game's
    mod list picks the choice that belongs to the new state."""
    profile = _tree(tmp_path, monkeypatch)
    mod_graphics.set_enabled(True)
    mod_graphics.set_excluded({"subscribed//3_Off"})  # an opt-out kept from when the game enabled it
    mod_graphics.set_included({"subscribed//3_Off"})
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"off"
    assert mod_graphics.resolve_sprite("b_hut_x1") is not None

    _set_enabled_in_game(profile, **{"3_Off": True, "2_Low": False})
    mod_graphics.clear()
    drawn = _drawn()
    assert not drawn["subscribed//3_Off"], "enabled now: its opt-out applies"
    assert not drawn["subscribed//2_Low"], "disabled now: off until ticked"
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"u"
    assert mod_graphics.resolve_sprite("b_hut_x1") is None
    assert {m.status_path: m.in_game for m in mod_graphics.mods_with_art()}["subscribed//2_Low"] is False

    _set_enabled_in_game(profile, **{"3_Off": False, "2_Low": True})
    mod_graphics.clear()
    assert _drawn()["subscribed//3_Off"] and _drawn()["subscribed//2_Low"]
    assert mod_graphics.resolve_sprite("n_rock_x1")[0].read_bytes() == b"off"


def test_palx_alpha_and_pal_fourth_column(tmp_path):
    palx = tmp_path / "a.palx"
    palx.write_text("JASC-PAL\n0100\n2\n$ALPHA 2\n1 2 3 40\n# note\n5 6 7 255\n")
    pal = tmp_path / "b.pal"
    pal.write_text("JASC-PAL\n0100\n2\n1 2 3 40\n5 6 7 9\n")
    assert slp_decoder.load_palette_rgba(palx) == [(1, 2, 3, 40), (5, 6, 7, 255)]
    assert slp_decoder.load_palette_rgba(pal) == [(1, 2, 3, 255), (5, 6, 7, 255)]
    assert slp_decoder.load_palette(palx) == [(1, 2, 3), (5, 6, 7)]


# -- unit_sprites wiring ------------------------------------------------------


def test_frame_mapping():
    assert unit_sprites._mod_frame_index("a", 7, 16, 16) == 7
    assert unit_sprites._mod_frame_index("a", 7, 16, 1) == 0
    assert unit_sprites._mod_frame_index("a", 7, 961, 480) is None  # falls back, never a modulo


def _install_with(tmp_path: Path, monkeypatch, stem: str, frames: int) -> Path:
    from descape import asset_source

    install = tmp_path / "install"
    graphics = install / "resources" / "_common" / "drs" / "graphics"
    graphics.mkdir(parents=True)
    palettes = install / "resources" / "_common" / "palettes"
    palettes.mkdir(parents=True)
    rows = "\n".join(f"{i % 256} 0 0" for i in range(1024))
    (palettes / "b_scen.pal").write_text(f"JASC-PAL\n0100\n1024\n{rows}\n")
    (palettes / "palettes.conf").write_text("// comment\n28,b_scen.pal\n")
    one = _frame([(0, [(DRAW, [(9, 0)] * 4)])], width=4)
    (graphics / f"{stem}.smx").write_bytes(_file(*[one] * frames))
    monkeypatch.setattr(unit_sprites, "load_sld", lambda path, **_kw: _fake_install(path, frames))
    # Recorded before the override, or the undo restores `install` for the rest of the worker.
    monkeypatch.setattr(asset_source, "_override_path", None)
    asset_source.set_install_path_override(install)
    return install


def _fake_install(path, frames):
    from descape.smx_decoder import load_smx

    real = Path(str(path)[:-4] + ".smx")
    if "mods" in real.parts:
        return None
    return _Install(load_smx(real), frames)


class _Install:
    """An install .sld stand-in: frame count, and a decode that paints red."""

    def __init__(self, smx, frames):
        self.frame_count = frames
        self.smx = smx

    def release_data(self):
        return b""

    def decode_frame(self, index, data, **_kw):
        from descape.sld_decoder import DecodedFrame

        main = np.zeros((1, 4, 4), np.uint8)
        main[..., 0] = 255
        main[..., 3] = 255
        return DecodedFrame(4, 1, 3, 4, main=main)


def test_one_frame_mod_replacement_keeps_install_counts(tmp_path, monkeypatch):
    stem = "n_test_x1"
    _install_with(tmp_path, monkeypatch, stem, frames=8)
    profile = tmp_path / "profile"
    _mod(profile, "1_Mod", {f"drs/graphics/{stem}.smx": _file(_frame([(0, [(DRAW, [(77, 0)] * 2)])], width=2))})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    unit_sprites.clear_caches()
    assert unit_sprites._native_frame(stem, 5)[0][0, 0, 0] == 255  # install art while off
    mod_graphics.set_enabled(True)
    unit_sprites.clear_caches()
    main = unit_sprites._native_frame(stem, 5)[0]
    assert main.shape[:2] == (1, 2) and main[0, 0, 0] == 77  # frame 0 of the 1-frame file
    assert unit_sprites.sld_frame_count(stem) == 8
    assert unit_sprites._native_frame(stem, 8) is None  # range-checked against the install


def test_mismatched_mod_frame_count_draws_the_install(tmp_path, monkeypatch):
    stem = "n_test_x1"
    _install_with(tmp_path, monkeypatch, stem, frames=8)
    profile = tmp_path / "profile"
    three = _file(*[_frame([(0, [(DRAW, [(77, 0)])])], width=1)] * 3)
    _mod(profile, "1_Mod", {f"drs/graphics/{stem}.smx": three})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    mod_graphics.set_enabled(True)
    unit_sprites.clear_caches()
    assert unit_sprites._native_frame(stem, 2)[0][0, 0, 0] == 255


def test_a_mod_blanked_composite_parent_still_draws_the_other_pieces(tmp_path, monkeypatch):
    # "Open TC with Range" ships an empty town centre back piece, the composite's parent.
    back, main = "b_tc_back_x1", "b_tc_main_x1"
    install = _install_with(tmp_path, monkeypatch, back, frames=1)
    one = _frame([(0, [(DRAW, [(9, 0)] * 4)])], width=4)
    (install / "resources" / "_common" / "drs" / "graphics" / f"{main}.smx").write_bytes(_file(one))
    piece = {"angle_count": 1, "frame_count": 1, "dx": 0, "dy": 0}
    entry = {"file_name": back, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1, "pieces": [
        {**piece, "unit_id": 618, "file_name": main},
        {**piece, "unit_id": 109, "file_name": back, "parent": True},
    ]}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {109: entry})
    profile = tmp_path / "profile"
    _mod(profile, "1_Open", {f"drs/graphics/{back}.smx": _file(_frame([None], width=4))})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    unit_sprites.clear_caches()
    assert len(unit_sprites.sprite_pieces_for(109, 0.0, 0, 32)) == 2
    mod_graphics.set_enabled(True)
    unit_sprites.clear_caches()
    pieces = unit_sprites.sprite_pieces_for(109, 0.0, 0, 32)
    assert len(pieces) == 1  # the main piece, not [] (the coloured footprint mark)
    assert len(unit_sprites._native_pieces_for(109, entry, 0.0)) == 1  # Flat's icon path


GRASS, GRASS_STEM = 1358, "n_plant_grass_green_x1"
_BLANK = _file(_frame([None], width=4))
_INKED = _file(_frame([(0, [(DRAW, [(77, 0)] * 2)])], width=2))  # (77, 0, 0) where the install paints 255


def _remover(tmp_path: Path, monkeypatch, frame: bytes) -> None:
    """Grass, Green as one graphic over install_with()'s red frame, plus an
    enabled mod shipping `frame` for its x1 and x2 stems, as Useless Plants
    Remover does (Mod Art still off)."""
    _install_with(tmp_path, monkeypatch, GRASS_STEM, frames=1)
    entry = {"file_name": GRASS_STEM, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {GRASS: entry})
    x2 = GRASS_STEM[:-3] + "_x2"
    _mod(tmp_path / "profile", "1205_Remover", {f"drs/graphics/{GRASS_STEM}.smx": frame, f"drs/graphics/{x2}.smx": frame})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))


def _draws(half_w: int = 32) -> list:
    """Every path a single graphic draws through: Stepped/Sloped, the composite walk, Flat's icon."""
    unit_sprites.clear_caches()
    pieces = unit_sprites.sprite_pieces_for(GRASS, 0.0, 0, half_w)
    return [
        unit_sprites.sprite_for(GRASS, 0.0, 0, half_w),
        pieces[0].draw if len(pieces) == 1 else None,
        unit_sprites.icon_for(GRASS, 0.0, 0, 16, 16),
    ]


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    return np.array_equal(a.rgba, b.rgba) and (a.hotspot_x, a.hotspot_y) == (b.hotspot_x, b.hotspot_y)


@pytest.mark.parametrize("hi_res", [False, True])
def test_a_mod_blanked_single_graphic_draws_the_install_art(tmp_path, monkeypatch, hi_res):
    """A blank mod frame on a single-graphic unit draws the install's art,
    pixel-equal to Mod Art off, not the coloured mark (TASK-437). half_w 64
    enlarges x1, so at Enhanced the mod's blank x2 is asked first."""
    _remover(tmp_path, monkeypatch, _BLANK)
    monkeypatch.setattr(unit_sprites, "_hi_res", hi_res)
    half_w = 64 if hi_res else 32
    off = _draws(half_w)
    assert all(draw is not None for draw in off), "the install must draw with Mod Art off, or the identity is vacuous"
    mod_graphics.set_enabled(True)
    on = _draws(half_w)
    assert GRASS_STEM in unit_sprites._mod_blank_stems, "the mod's blank frame must be what was read"
    for path, (got, want) in enumerate(zip(on, off, strict=True)):
        assert _same(got, want), f"draw path {path} lost the install's art"


def test_a_mod_frame_with_ink_still_wins_over_the_install(tmp_path, monkeypatch):
    _remover(tmp_path, monkeypatch, _INKED)
    mod_graphics.set_enabled(True)
    for path, draw in enumerate(_draws()):
        ink = draw.rgba[draw.rgba[..., 3] > 0]
        assert ink.size and (ink[:, 0] == 77).all(), f"draw path {path} did not draw the mod's art"
    assert not unit_sprites._mod_blank_stems


def test_one_blank_stem_is_a_gap_as_a_piece_and_install_art_as_a_single_graphic(tmp_path, monkeypatch):
    """The same stem, frame, team and scale through a composite and a single
    graphic: _scaled_cache must not serve one's answer to the other, in
    either order."""
    back, main = "b_tc_back_x1", "b_tc_main_x1"
    install = _install_with(tmp_path, monkeypatch, back, frames=1)
    graphics = install / "resources" / "_common" / "drs" / "graphics"
    (graphics / f"{main}.smx").write_bytes(_file(_frame([(0, [(DRAW, [(9, 0)] * 4)])], width=4)))
    piece = {"angle_count": 1, "frame_count": 1, "dx": 0, "dy": 0}
    single = {"file_name": back, "angle_count": 1, "mirroring_mode": 0, "frame_count": 1}
    composite = {**single, "pieces": [
        {**piece, "unit_id": 618, "file_name": main},
        {**piece, "unit_id": 109, "file_name": back, "parent": True},
    ]}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {109: composite, GRASS: single})
    _mod(tmp_path / "profile", "1_Open", {f"drs/graphics/{back}.smx": _BLANK})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))
    mod_graphics.set_enabled(True)
    for order in ((109, GRASS), (GRASS, 109)):
        unit_sprites.clear_caches()
        got = {const: unit_sprites.sprite_pieces_for(const, 0.0, 0, 32) for const in order}
        assert len(got[109]) == 1, f"{order}: the composite's blank back piece must stay a gap"
        assert len(got[GRASS]) == 1, f"{order}: the single graphic must draw the install's art"


@pytest.mark.corpus
def test_no_eye_candy_const_that_draws_without_mods_is_lost_to_a_blank_mod_frame(monkeypatch):
    """Real install and profile: with Mod Art on, every eye-candy const that
    draws with it off still draws. Needs a profile whose mods blank some (the
    Useless Plants Remover blanked 19 before TASK-437)."""
    from descape import unit_kind

    def unmet(reason: str):
        if _REQUIRE:
            pytest.fail(f"DESCAPE_REQUIRE_MODS=1 but {reason}")
        pytest.skip(reason)

    if asset_source.get_install_path() is None or not _PROFILE:
        unmet("needs AOE2DE_INSTALL_PATH and AOE2DE_PROFILE_PATH")
    consts = sorted(unit_kind.eye_candy_consts())

    def drawn() -> set[int]:
        unit_sprites.clear_caches()
        return {c for c in consts if unit_sprites.sprite_for(c, 0.0, 0, 32) and unit_sprites.icon_for(c, 0.0, 0, 16, 16)}

    off = drawn()
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", _PROFILE)
    mod_graphics.clear()
    mod_graphics.set_enabled(True)
    on = drawn()
    if not unit_sprites._mod_blank_stems:
        unmet("no active mod blanks an eye-candy stem")
    assert off, "nothing draws with Mod Art off: the check is vacuous"
    assert not off - on, f"lost to a blank mod frame: {sorted(off - on)}"


def test_sprite_reach_px_keeps_its_side_order(monkeypatch):
    from types import SimpleNamespace

    from descape import render

    for name, value in (("LEFT", 96), ("RIGHT", 192), ("UP", 288), ("DOWN", 384)):
        monkeypatch.setattr(unit_sprites, f"MAX_SPRITE_REACH_{name}", value)
    proj = SimpleNamespace(half_w=48)  # sprite_scale 1.0
    assert render._sprite_reach_px(proj) == (97, 289, 193, 385)  # (left, up, right, down) at scale 1


def test_mod_reach_widens_sprite_reach(tmp_path, monkeypatch):
    from descape.smx_decoder import SMXFile

    wide = SMXFile(_file(_frame([(0, [(DRAW, [(1, 0)])])], width=2000, hotspot=(1000, 0))))
    monkeypatch.setattr(unit_sprites, "referenced_entries", lambda: [("b_wide_x1", 0, 0)])
    monkeypatch.setattr(unit_sprites, "_mod_index", lambda name: (wide, Path("w"), "smx"))
    monkeypatch.setattr(unit_sprites, "sld_frame_count", lambda name: 1)
    mod_graphics.set_enabled(True)
    left, right, up, down = unit_sprites.sprite_reach()
    assert (left, right) == (1000, 1000)
    assert up == unit_sprites.MAX_SPRITE_REACH_UP and down == unit_sprites.MAX_SPRITE_REACH_DOWN
    mod_graphics.set_enabled(False)
    assert unit_sprites.sprite_reach()[0] == unit_sprites.MAX_SPRITE_REACH_LEFT


def test_mod_terrain_texture_overrides_and_reverts(tmp_path, monkeypatch):
    """A mod's same-named texture wins while on; turning it off through
    clear_art_caches() serves the install's again, cached array included."""
    from PIL import Image

    from descape import asset_source

    tid, name = next(iter(asset_source._terrain_texture_map().items()))
    install = tmp_path / "install"
    folder = install / asset_source.TERRAIN_TEXTURE_SUBPATH
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 0, 0)).save(folder / name, format="DDS")
    profile = tmp_path / "profile"
    common = _mod(profile, "1_Tex", {})
    (common / "terrain" / "textures" / "2x").mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 200)).save(common / "terrain" / "textures" / "2x" / name, format="DDS")
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    asset_source.set_install_path_override(install)
    try:
        assert asset_source.get_terrain_texture_array(tid)[0, 0].tolist() == [200, 0, 0]
        mod_graphics.set_enabled(True)
        generation = asset_source.install_generation()
        asset_source.clear_art_caches()
        assert asset_source.install_generation() > generation
        assert asset_source.get_terrain_texture_path(tid).parent == common / "terrain" / "textures" / "2x"
        assert asset_source.get_terrain_texture_array(tid)[0, 0].tolist() == [0, 0, 200]
        mod_graphics.set_enabled(False)
        asset_source.clear_art_caches()
        assert asset_source.get_terrain_texture_array(tid)[0, 0].tolist() == [200, 0, 0]
    finally:
        asset_source.set_install_path_override(None)


def test_an_unreadable_mod_texture_draws_the_install_one(tmp_path, monkeypatch):
    from PIL import Image

    from descape import asset_source

    tid, name = next(iter(asset_source._terrain_texture_map().items()))
    install = tmp_path / "install"
    folder = install / asset_source.TERRAIN_TEXTURE_SUBPATH
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 0, 0)).save(folder / name, format="DDS")
    _mod(tmp_path / "profile", "1_Bad", {f"terrain/textures/2x/{name}": b"DDS \x00garbage"})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))
    asset_source.set_install_path_override(install)
    try:
        mod_graphics.set_enabled(True)
        asset_source.clear_art_caches()
        assert "1_Bad" in str(asset_source.get_terrain_texture_path(tid))
        assert asset_source.get_terrain_texture_array(tid)[0, 0].tolist() == [200, 0, 0]
        assert asset_source.get_terrain_thumbnail(tid)[0, 0].tolist() == [200, 0, 0]
    finally:
        asset_source.set_install_path_override(None)


def _wide_mod_frame(width: int) -> bytes:
    """One opaque row `width` px wide, hotspot at its centre: reach width/2 each side."""
    runs = [(DRAW, [(9, 0)] * 50)] * (width // 50)
    return _file(_frame([(0, runs)], width=width, hotspot=(width // 2, 0)))


@pytest.mark.usefixtures("sprite_install")
def test_wide_mod_art_widens_the_edit_bbox_to_cover_itself(tmp_path, monkeypatch):
    """The synthetic stand-in for a moved range-ring building (no real mod here
    draws past the constants): mod art wider than MAX_SPRITE_REACH_* must
    widen the edit's dirty bbox so the whole sprite repaints."""
    import test_sprite_edit_bbox as bbox

    palettes = tmp_path / "resources" / "_common" / "palettes"
    palettes.mkdir(parents=True)
    rows = "\n".join("9 9 9" for _ in range(1024))
    (palettes / "b_scen.pal").write_text(f"JASC-PAL\n0100\n1024\n{rows}\n")
    (palettes / "palettes.conf").write_text("28,b_scen.pal\n")
    width = 3 * bbox.SPRITE_CANVAS
    _mod(tmp_path / "profile", "1_Wide", {f"drs/graphics/{bbox.FILE_NAME}.smx": _wide_mod_frame(width)})
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))
    mod_graphics.set_enabled(True)
    unit_sprites.clear_caches()
    assert unit_sprites.sprite_reach()[:2] == (width // 2, width // 2)

    scenario, dirty, elevations, proj = bbox._edited_scenario()
    widened = render.dirty_screen_bbox_iso(scenario, dirty, elevations.copy(), proj, with_sprites=True)
    with monkeypatch.context() as m:
        m.setattr(unit_sprites, "_active_mod_reach", lambda: (0, 0, 0, 0))
        plain = render.dirty_screen_bbox_iso(scenario, dirty, elevations.copy(), proj, with_sprites=True)
    for elevation in bbox.OBSERVED_ELEVATIONS:
        rect = bbox._sprite_rect(scenario, proj, elevation)
        assert rect[2] - rect[0] > bbox.SPRITE_CANVAS, "the mod's wide art must be what draws"
        assert bbox._clamped(rect, proj) == rect
        assert not bbox._covers(plain, rect), "the constants alone already cover it: vacuous"
        assert bbox._covers(widened, rect)


# Two small tree-bearing quick-corpus maps (120x120, 144x144): four full renders in three styles each.
_IDENTITY_FILES = ("C2_ElCid_coop_1_v0_16.aoe2scenario", "2_Joan_coop_1_v0_13.aoe2scenario")


def _renders(scenario) -> dict[str, np.ndarray]:
    """Full sprite renders in every style, after the apply path's own cache drop."""
    asset_source.clear_art_caches()
    return {
        "flat": render.overlay_units(render.render_terrain(scenario), scenario, with_sprites=True),
        "stepped": render.render_terrain_iso_with_proj(scenario, with_sprites=True)[0],
        "sloped": render.render_terrain_sloped_with_proj(scenario, with_sprites=True)[0],
    }


@pytest.mark.corpus
@pytest.mark.parametrize("name", _IDENTITY_FILES)
def test_mod_art_off_or_all_opted_out_renders_like_no_profile(name, monkeypatch):
    """Plan Verification T3, in one process: off with this machine's mods is
    byte-identical to no profile at all, and on with every mod opted out to
    off. On with the mods active must differ in every style, or the rest is vacuous."""

    def unmet(reason: str):
        if _REQUIRE:
            pytest.fail(f"DESCAPE_REQUIRE_MODS=1 but {reason}")
        pytest.skip(reason)

    path = conftest.ROOT / "examples" / name
    if asset_source.get_install_path() is None or not path.is_file():
        unmet("needs AOE2DE_INSTALL_PATH and examples/" + name)
    if not _PROFILE:
        unmet("needs AOE2DE_PROFILE_PATH naming a profile with enabled art mods")
    scenario = load_map_and_units(path)
    mod_graphics.set_enabled(True)
    assert not mod_graphics.mods_with_art(), "conftest should hide every mod without a profile"
    no_profile = _renders(scenario)

    monkeypatch.setenv("AOE2DE_PROFILE_PATH", _PROFILE)
    mod_graphics.clear()
    # Disabled mods are listed too (GH #264), and draw only when included.
    mods = [mod for mod in mod_graphics.mods_with_art() if mod.in_game]
    if not mods:
        unmet("the named profile has no enabled art mods")
    on = _renders(scenario)
    mod_graphics.set_enabled(False)
    off = _renders(scenario)
    mod_graphics.set_enabled(True)
    mod_graphics.set_excluded(mod.status_path for mod in mods)
    opted_out = _renders(scenario)

    changed = [style for style in on if not np.array_equal(on[style], off[style])]
    assert changed == list(on), f"mod art drew only in {changed}: the other styles' identities would be vacuous"
    for style in no_profile:
        assert np.array_equal(off[style], no_profile[style]), f"{style}: off differs from no profile"
        assert np.array_equal(opted_out[style], off[style]), f"{style}: all opted out differs from off"


@pytest.fixture(autouse=True)
def _clean_sprites():
    # Setup only: a test's monkeypatched sld_frame_count is still in place at this fixture's teardown.
    unit_sprites.clear_caches()
