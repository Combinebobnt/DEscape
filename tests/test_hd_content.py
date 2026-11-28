"""TASK-431 stage 2: HD Edition's install and derived root, built from a synthetic install.

descape/hd_content.py's validator against HD, AoE2:DE and 1999 folders (H6),
the root it gathers (H7: the art folders' order, the textures' suffix, the
palette, the strings, and the hardlink, symlink, copy fallback), its key, and
what the viewer does with it: Auto and the build before activating (H5),
the hand-pick line for a header-3 HD file (H4) and the Settings row.
"""

from __future__ import annotations

import errno
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_legacy_content import PALETTE, PALETTE_TEXT, _window, make_install

from descape import asset_source, game_dataset, games, hd_content, legacy_content, scenario_format

import conftest

ROOT = Path(__file__).resolve().parent.parent
DRS = "resources/_common/drs"
ART = {"graphics": {"8.slp": b"base", "305.slp": b"base"}, "gamedata_x1": {"8.slp": b"x1"},
       "gamedata_x2": {"8.slp": b"x2", "6000.slp": b"x2"}}
TEXTURES = {"g_grs_00_color.png": b"grass", "g_rck_00_COLOR.png": b"rock"}
STRINGS = "strings/key-value/key-value-strings-utf8.txt"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_hd(root: Path, palette_text: bytes = PALETTE_TEXT) -> Path:
    """An HD Edition install: three art folders sharing id 8, two textures (one upper-case suffix), two languages."""
    for folder, files in ART.items():
        for name, data in files.items():
            _write(root / DRS / folder / name, data)
    _write(root / DRS / "graphics" / "readme.txt", b"not art")
    _write(root / hd_content.PALETTE, palette_text)
    _write(root / games.DE_DAT_SUBPATH, b"read at regen time only")
    _write(root / hd_content.BLENDOMATIC, b"blend")
    for name, data in TEXTURES.items():
        _write(root / hd_content.TEXTURES_DIR / name, data)
    _write(root / "resources/en" / STRINGS, b'4 "Archer"\n')
    _write(root / "resources/nl" / STRINGS, b'4 "Boogschutter"\n')
    _write(root / "resources/_packages/strings-en.json", b"{}")
    return root


def make_de_shaped(root: Path) -> Path:
    """What an AoE2:DE install shares with HD: the .dat, graphics SLPs and the palette, no gamedata_x1 or PNGs."""
    _write(root / games.DE_DAT_SUBPATH, b"dat")
    _write(root / DRS / "graphics" / "8.slp", b"de")
    _write(root / hd_content.PALETTE, PALETTE_TEXT)
    _write(root / hd_content.TEXTURES_DIR / "2x" / "g_grs.dds", b"dds")
    return root


@pytest.fixture
def hd_install(tmp_path, monkeypatch) -> Path:
    """A valid install, configured through HD_INSTALL_PATH (conftest unsets it and redirects CACHE_ROOT)."""
    path = make_hd(tmp_path / "Age2HD")
    monkeypatch.setenv(hd_content.INSTALL_ENV, str(path))
    asset_source.clear_install_caches(None)
    yield path
    asset_source.clear_install_caches(None)


def _header3():
    hd3 = b"1.21" + struct.pack("<Ii", 4, 3) + b"\x00" * 8
    return SimpleNamespace(source_format=scenario_format.detect_format(hd3), game_variant=None)


def test_recipe_gives_each_install_kind_its_builder():
    assert legacy_content.recipe(games.AOC_INSTALL) is legacy_content
    assert legacy_content.recipe(games.HD_INSTALL) is hd_content
    assert {legacy_content.recipe(g.install) for g in games.GAMES if g.derived_root} == {legacy_content, hd_content}
    with pytest.raises(ValueError, match="no derived root"):
        legacy_content.recipe(games.AOE2DE_INSTALL)


def test_the_validator_takes_hd_and_refuses_de_and_1999_folders(tmp_path):
    hd = make_hd(tmp_path / "hd")
    assert hd_content.validate_install(hd) == (True, "HD Edition install found.")
    ok, message = hd_content.validate_install(make_de_shaped(tmp_path / "de"))
    assert not ok and "gamedata_x1/" in message and "_00_color.png" in message and "graphics/*.slp" not in message
    ok, message = hd_content.validate_install(make_install(tmp_path / "aoc"))
    assert not ok and games.DE_DAT_SUBPATH in message
    assert not legacy_content.validate_install(hd)[0], "the 1999 validator refuses HD too"
    assert hd_content.validate_install(tmp_path / "nowhere") == (False, f"Not a folder: {tmp_path / 'nowhere'}")


def test_the_graphics_order_is_the_generators(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import _drs

    assert hd_content.GRAPHICS_ORDER == _drs.HD_GRAPHICS_ORDER


def test_the_root_gathers_the_install_into_des_layout(tmp_path):
    install = make_hd(tmp_path / "hd")
    root = hd_content.ensure_root(install)
    assert root.parent == hd_content.CACHE_ROOT and hd_content.built_root(install) == root
    graphics = root / legacy_content.GRAPHICS_SUBPATH
    assert sorted(p.name for p in graphics.iterdir()) == ["slp_305.slp", "slp_6000.slp", "slp_8.slp"]
    assert (graphics / "slp_8.slp").read_bytes() == b"x2", "gamedata_x2 wins the shared id"
    assert (graphics / "slp_305.slp").read_bytes() == b"base"
    textures = root / asset_source.TERRAIN_TEXTURE_SUBPATH
    assert sorted(p.name for p in textures.iterdir()) == ["g_grs_00_color.png", "g_rck_00_color.png"]
    assert (textures / "g_rck_00_color.png").read_bytes() == b"rock"
    assert (root / hd_content.BLENDOMATIC).read_bytes() == b"blend"
    assert (root / "resources/nl" / STRINGS).read_bytes() == b'4 "Boogschutter"\n'
    palettes = root / legacy_content.PALETTE_SUBPATH
    assert (palettes / legacy_content.PALETTE_NAME).read_bytes() == PALETTE_TEXT
    ramp = (palettes / legacy_content.WHITE_PLAYER_PALETTE).read_text(encoding="latin-1").splitlines()[3:]
    assert ramp[0].split() == [str(v) for v in PALETTE[legacy_content.GREY_RAMP_BASE]]


@pytest.mark.parametrize("refused", [(), ("link",), ("link", "symlink")])
def test_a_refused_link_falls_back_to_the_next_way(tmp_path, monkeypatch, refused):
    if "link" in refused and not conftest.symlinks_supported() and "symlink" not in refused:
        pytest.skip("no symlink privilege here")

    def refuse(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    for name in refused:
        monkeypatch.setattr(os, name, refuse)
    install = make_hd(tmp_path / "hd")
    root = hd_content.ensure_root(install)
    meta = (root / legacy_content.META_NAME).read_text()
    expected = hd_content._Linker.MODES[len(refused)]
    assert f'"placed_by": "{expected}"' in meta
    placed = root / legacy_content.GRAPHICS_SUBPATH / "slp_8.slp"
    assert placed.read_bytes() == b"x2" and placed.is_symlink() == (expected == "symlink")


def test_the_key_follows_the_art_folders_and_a_bad_palette_fails_before_staging(tmp_path):
    install = make_hd(tmp_path / "hd")
    first = hd_content.ensure_root(install)
    _write(install / DRS / "gamedata_x2" / "7000.slp", b"new")
    assert hd_content.built_root(install) is None, "an added SLP is another key"
    second = hd_content.ensure_root(install)
    assert second != first and (second / legacy_content.GRAPHICS_SUBPATH / "slp_7000.slp").is_file()
    _write(install / DRS / "gamedata_x2" / "real_world_france.scx.bak", b"saved beside the templates")
    assert hd_content.built_root(install) == second, "a non-art file beside the art keeps the key"
    short = b"JASC-PAL\r\n0100\r\n16\r\n" + b"1 2 3\r\n" * 16
    broken = make_hd(tmp_path / "broken", palette_text=short)
    with pytest.raises(legacy_content.RootBuildError, match="16 colours"):
        hd_content.ensure_root(broken)
    assert hd_content.build_failed(broken) and not any(hd_content.CACHE_ROOT.glob("*.partial-*"))


def test_the_key_follows_steams_build_id(tmp_path):
    install = make_hd(tmp_path / "steamapps" / "common" / "Age2HD")
    manifest = tmp_path / "steamapps" / hd_content.STEAM_MANIFEST
    no_manifest = hd_content.root_key(install)
    _write(manifest, b'"AppState"\n{\n\t"appid"\t\t"221380"\n\t"buildid"\t\t"3062235"\n}\n')
    first = hd_content.root_key(install)
    _write(manifest, b'"AppState"\n{\n\t"appid"\t\t"221380"\n\t"buildid"\t\t"3062236"\n}\n')
    assert len({no_manifest, first, hd_content.root_key(install)}) == 3


def test_two_spellings_of_one_texture_make_one_target(tmp_path):
    install = make_hd(tmp_path / "hd")
    _write(install / hd_content.TEXTURES_DIR / "g_grs_00_COLOR.png", b"grass again")
    if len(list((install / hd_content.TEXTURES_DIR).iterdir())) < 3:
        pytest.skip("case-insensitive filesystem: one file already")
    plain = json.loads((hd_content.ensure_root(make_hd(tmp_path / "plain")) / legacy_content.META_NAME).read_text())
    root = hd_content.ensure_root(install)
    textures = root / asset_source.TERRAIN_TEXTURE_SUBPATH
    assert sorted(p.name for p in textures.iterdir()) == ["g_grs_00_color.png", "g_rck_00_color.png"]
    meta = json.loads((root / legacy_content.META_NAME).read_text())
    assert meta["placed_by"] == plain["placed_by"], "the collision moved the linker on"


def test_browse_starts_beside_the_de_installs_steam_library(tmp_path):
    de = tmp_path / "steamapps" / "common" / "AoE2DE"
    de.mkdir(parents=True)
    asset_source.set_install_path_override(de)
    try:
        assert hd_content.browse_start() is None, "no Age2HD folder there"
        (tmp_path / "steamapps" / "common" / hd_content.STEAM_FOLDER).mkdir()
        assert hd_content.browse_start() == tmp_path / "steamapps" / "common" / hd_content.STEAM_FOLDER
    finally:
        asset_source.set_install_path_override(None)
    assert legacy_content.browse_start() is None


def test_auto_shows_a_header_2_hd_file_with_hd_data_and_builds_its_root_first(qapp, hd_install):
    from descape import viewer_game_root

    window = _window(conftest.legacy_loaded(game=scenario_format.HD))
    assert game_dataset.configured(games.HD) and window._shown_game_dataset() == "hd"
    assert viewer_game_root.ready_shown_game(window) == "hd"
    root = hd_content.built_root(hd_install)
    assert root is not None and game_dataset.install_root(games.HD) == root
    game_dataset.set_active(game_dataset.HD)
    assert asset_source.get_install_path() == root
    assert viewer_game_root.status_lines(window) == []
    assert game_dataset.lock_reason("hd", "hd", legacy=True) is None, "a header-2 save is legacy_write's call"


def test_a_header_3_hd_file_under_auto_names_the_hand_pick_while_hd_is_configured(tmp_path, monkeypatch):
    from descape import viewer_game_root

    window = _window(_header3())
    assert window._shown_game_dataset() == "aoe2"
    assert viewer_game_root.status_lines(window) == [], "no HD install: no line"
    monkeypatch.setenv(hd_content.INSTALL_ENV, str(make_hd(tmp_path / "hd")))
    asset_source.clear_install_caches(None)
    assert window._shown_game_dataset() == "aoe2", "Auto still shows it with AoE2:DE data"
    assert viewer_game_root.status_lines(window) == [
        (
            "An HD Edition file: Settings > Advanced > Game data > Age of Empires II: HD Edition shows it "
            "with that game's own data, view-only."
        )
    ]


def test_the_settings_rows_are_one_per_install_kind_and_the_hd_row_saves(qapp, tmp_path):
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    from descape import settings_game_install

    assert settings_game_install.install_games() == [games.AOC, games.HD]
    calls = []
    window = SimpleNamespace(
        _log_status=lambda text, level=None: calls.append(text),
        apply_game_data=lambda: calls.append("apply"),
        _refresh_game_assets=lambda: calls.append("refresh"),
    )
    dialog = QWidget()
    dialog._window = window
    row = settings_game_install.InstallRow(dialog, QVBoxLayout(dialog), games.HD)
    assert row.edit.placeholderText() == hd_content.PLACEHOLDER and row.browse_start() == str(Path.home())
    row.edit.setText(str(make_install(tmp_path / "aoc")))
    row.load()
    assert hd_content.install_path() is None and "apply" not in calls and "HD Edition install" in row.status.text()
    path = make_hd(tmp_path / "hd")
    row.edit.setText(str(path))
    row.load()
    assert hd_content.install_path() == path and legacy_content.install_path() is None and calls.count("apply") == 1
    assert row.browse_start() == str(path)
    row.clear()
    assert hd_content.install_path() is None and row.edit.text() == ""
