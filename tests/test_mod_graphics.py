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
    assert [m.status_path for m in mod_graphics.mods_with_art()] == [
        "subscribed//1_High", "subscribed//2_Low", "local//Loose", "subscribed//9_Unlisted",
    ]
    before = mod_graphics.generation()
    assert mod_graphics.set_excluded({"subscribed//1_High"}) and mod_graphics.generation() > before
    assert mod_graphics.resolve_sprite("n_tree_x1")[0].read_bytes() == b"low"
    assert mod_graphics.resolve_terrain_texture("g_gr1.dds") is None
    assert mod_graphics.mods_with_art()[0].excluded
    assert not mod_graphics.set_excluded({"subscribed//1_High"})


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
    asset_source.set_install_path_override(install)
    monkeypatch.setattr(asset_source, "_override_path", install)
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
    mods = mod_graphics.mods_with_art()
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
