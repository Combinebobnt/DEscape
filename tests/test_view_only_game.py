"""A third game registered as data (TASK-570 stage C1, docs/GAME_DATA.md):
view-only, with its own root over the install. The lock and write_scenario()
refuse its files, its root chain falls back file by file for sprites and
portraits, and the window shows its title tag, its missing-root line and its
row in Settings, all from the descriptor alone."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from descape import asset_source, batch_api, game_dataset, games, scenario_io, unit_sprites
from descape.scenario_new import blank_template_path_for
from descape.scenario_write import WriteBlockedError, write_scenario
from testkit import fake_game

import conftest

_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]
FAKE_MATERIALS = "fakematerials.json"


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    """The view-only fake game registered over an empty install; registered()'s
    finally makes AoE2 active again before it unregisters the game."""
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
    asset_source.clear_install_caches()
    game = fake_game.view_only(tmp_path, title_tag=" [Fake Game]", icon_materials=FAKE_MATERIALS)
    with fake_game.registered(game):
        yield SimpleNamespace(game=game, install=install, root=game.root(install))


def test_the_lock_says_view_only_whatever_data_shows_the_file(fake):
    for shown in ("fake", "aoe2"):
        reason = game_dataset.lock_reason("fake", shown)
        assert "view-only" in reason and "Fake Game" in reason, shown
    mismatch = game_dataset.lock_reason("aoe2", "fake")
    assert "Fake Game game data for a Age of Empires II file" in mismatch and "Settings > Advanced" in mismatch
    assert "view-only" not in mismatch, "a writable file's own game is not view-only"
    for game_id in games.ids():
        if game_id != "fake":
            assert game_dataset.lock_reason(game_id, game_id) is None, game_id


def test_write_scenario_refuses_a_view_only_games_file(fake, tmp_path):
    path = fake_game.scenario_copy(tmp_path)
    out = tmp_path / "out.aoe2scenario"
    out.write_bytes(b"earlier")  # an existing file: a refused write must not back it up either
    loaded = scenario_io.load_map_and_units(path)
    assert games.for_scenario(loaded) is fake.game
    with pytest.raises(WriteBlockedError, match=r"Fake Game scenario.*view-only"):
        write_scenario(loaded, out)

    # A batch script reads it with its own game's data and is refused the same way.
    loaded = batch_api.load(path)
    assert game_dataset.game() is fake.game
    with pytest.raises(WriteBlockedError, match="view-only"):
        batch_api.save(loaded, out)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("out")) == ["out.aoe2scenario"]
    assert out.read_bytes() == b"earlier"
    assert write_scenario(scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH), out, backup=False).wrote


def test_its_sprites_resolve_along_its_root_chain(fake):
    base = fake.install / unit_sprites.GRAPHICS_SUBPATH
    own = fake.root / unit_sprites.GRAPHICS_SUBPATH
    base.mkdir(parents=True)
    own.mkdir(parents=True)
    (own / "u_own_x1.sld").write_bytes(b"")
    (base / "u_base_x1.sld").write_bytes(b"")
    assert unit_sprites._sld_path(base, "u_own_x1") == base / "u_own_x1.sld", "AoE2 never reads another game's root"
    assert unit_sprites._install_lacks(base, "u_own_x1")
    game_dataset.set_active(fake_game.FAKE_ID)
    assert unit_sprites._sld_path(base, "u_own_x1") == own / "u_own_x1.sld"
    assert unit_sprites._sld_path(base, "u_base_x1") == base / "u_base_x1.sld"
    assert unit_sprites._sld_path(base, "u_none_x1") == base / "u_none_x1.sld", "the bottom root's path when none has it"
    assert not unit_sprites._install_lacks(base, "u_own_x1"), "its own root can hold a stem the install lacks"
    assert not unit_sprites.own_x2_art_installed()
    (own / "u_own_x2.sld").write_bytes(b"")
    unit_sprites.clear_caches()
    assert unit_sprites.own_x2_art_installed() and unit_sprites.own_x2_art_installed(fake.game)


def test_a_root_without_a_fallback_resolves_sprites_under_itself_alone(tmp_path, monkeypatch):
    install = tmp_path / "install"
    base = install / unit_sprites.GRAPHICS_SUBPATH
    base.mkdir(parents=True)
    (base / "u_base_x1.sld").write_bytes(b"")
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
    asset_source.clear_install_caches()
    with fake_game.registered(fake_game.view_only(tmp_path, fallback=None)) as game:
        game_dataset.set_active(game.id)
        own = game.root(install) / unit_sprites.GRAPHICS_SUBPATH
        assert unit_sprites._sld_path(base, "u_base_x1") == own / "u_base_x1.sld", "never the install's art"


def _atlas(*textures: tuple[str, str]) -> str:
    entries = [{"RefName": ref, "FileName": name} for ref, name in textures]
    return json.dumps({"AtlasTextures": [{"AtlasDef": {"Name": "a", "Textures": entries}}]})


def test_its_portraits_resolve_through_its_icon_table_then_its_root_chain(fake):
    own_ui, base_ui = fake.root / asset_source.UI_DIR, fake.install / asset_source.UI_DIR
    for path in (
        own_ui / "textures/ingame/units/007_50730.dds",
        own_ui / "textures/ingame/units/003_50730.dds",
        base_ui / "textures/ingame/units/004_50730.dds",
        base_ui / "textures/ingame/units/005_50730.dds",
        base_ui / "textures/ingame/buildings/012_house.dds",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (own_ui / asset_source.ICON_TABLE_NAME).write_text(json.dumps({"Units": {"001": "Own7", "002": "Base4"}}), encoding="utf-8")
    (own_ui / FAKE_MATERIALS).write_text(_atlas(("Own7", "textures/ingame/units/007_50730.dds")), encoding="utf-8")
    (base_ui / asset_source.MATERIALS_NAME).write_text(_atlas(("Base4", "textures/ingame/units/004_50730.dds")), encoding="utf-8")
    assert asset_source.unit_icon_path(1) is None, "AoE2 ignores the fake game's table"
    assert asset_source.unit_icon_path(3) is None, "and its root"
    game_dataset.set_active(fake_game.FAKE_ID)
    assert asset_source.unit_icon_path(1) == own_ui / "textures/ingame/units/007_50730.dds", "a RefName in its own materials"
    assert asset_source.unit_icon_path(2) == base_ui / "textures/ingame/units/004_50730.dds", "a RefName in the fallback's"
    assert asset_source.unit_icon_path(3) == own_ui / "textures/ingame/units/003_50730.dds", "no table row: its own set first"
    assert asset_source.unit_icon_path(5) == base_ui / "textures/ingame/units/005_50730.dds", "then the install's"
    assert asset_source.unit_icon_path(12, building=True) == base_ui / "textures/ingame/buildings/012_house.dds"


def test_new_map_takes_a_games_blank_template_and_the_default_for_any_other(tmp_path):
    assert blank_template_path_for("nowhere") == scenario_io.BLANK_TEMPLATE_PATH
    with fake_game.registered(fake_game.fake(tmp_path, blank_template=games.ROR.blank_template)):
        assert blank_template_path_for(fake_game.FAKE_ID) == scenario_io.BLANK_ROR_TEMPLATE_PATH
    with fake_game.registered(fake_game.fake(tmp_path)):
        assert blank_template_path_for(fake_game.FAKE_ID) == scenario_io.BLANK_TEMPLATE_PATH


@_GUI[0]
@_GUI[1]
def test_the_window_opens_its_file_locked_and_names_it(fake, tmp_path, monkeypatch):
    window = conftest.blank_window(load=False)
    lines: list[str] = []
    real = window._log_status
    monkeypatch.setattr(window, "_log_status", lambda message, **kwargs: (lines.append(message), real(message, **kwargs)))
    try:
        window.load_scenario(fake_game.scenario_copy(tmp_path))
        assert game_dataset.game() is fake.game, "Auto shows the file's own game"
        assert "view-only" in window.game_data_lock_reason()
        assert not window.game_data_banner.isHidden()
        assert not window.save_action.isEnabled()
        assert window.windowTitle().startswith("fake_blank.aoe2scenario [Fake Game]")
        assert any(line.startswith("Fake Game art is not installed") for line in lines)

        lines.clear()
        fake.root.mkdir(parents=True)
        window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
        assert game_dataset.active() == game_dataset.AOE2 and window.game_data_lock_reason() is None
        assert "[Fake Game]" not in window.windowTitle()
        window.load_scenario(fake_game.scenario_copy(tmp_path))
        assert not any("art is not installed" in line for line in lines), "its root is there now"
    finally:
        conftest.close_window(window)


@_GUI[0]
@_GUI[1]
def test_settings_lists_it_and_data_mods_name_it(fake):
    dialog, window = conftest.dialog_and_window()
    try:
        combo = dialog.game_data_combo
        assert [combo.itemData(i) for i in range(combo.count())][-1] == fake_game.FAKE_ID
        assert combo.itemText(combo.count() - 1) == "Fake Game"
        assert "The other three show" in combo.toolTip() and "another game" in combo.toolTip()
        # The fake game takes data mods (Game.data_mods), so File > Data Mod names it beside AoE2.
        game_dataset.set_active(game_dataset.ROR)
        reason = window.data_mod_menu.unusable_reason(SimpleNamespace(status_path="x", title="x"))
        assert reason == "Data mods change Age of Empires II or Fake Game data; this document shows Return of Rome's"
    finally:
        dialog.close()
        conftest.close_window(window)
