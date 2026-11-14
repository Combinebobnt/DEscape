"""GH #58 in the window: File > Data Mod puts a data mod's tables in effect,
remembers the pick per saved scenario, parses an uncached mod in a subprocess
and swaps only once the window is idle.

The mods are fakes: discovery, the cache key and the cache lookup are
patched on data_mods, and an uncached pick runs a tiny Python child in place
of the real derive (the real parse is tests/test_dat_cache.py's and the
in-session smoke's). The tables are a copy of the AoE2 ones with Spearman's
hit points changed, so "in effect" is one stats lookup."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import (
    asset_source,
    autosave,
    dat_derive,
    data_mods,
    game_dataset,
    object_catalog,
    scenario_io,
    settings,
    unit_stats_table,
)
from testkit.ror_fixture import ror_copy

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

DESCAPE = Path(game_dataset.__file__).resolve().parent
SPEARMAN = 93
ARCHER = 4
MOD_HP = 4242
# FakeMods patches it; the GH #363 re-pick test needs the real side_key.
_REAL_TABLES_FOR = data_mods.tables_for


def _mod(tmp_path: Path, name: str) -> data_mods.DataMod:
    root = tmp_path / "mods" / name
    return data_mods.DataMod(f"subscribed//{name}", name, None, root, root / "empires2_x2_p1.dat")


def _tables(tmp_path: Path, mod: data_mods.DataMod) -> game_dataset.DataModTables:
    folder = tmp_path / "tables" / mod.title
    folder.mkdir(parents=True)
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(DESCAPE / name, folder / name)
    stats = json.loads((folder / "unit_stats.json").read_text())
    stats["stats"][str(SPEARMAN)]["hp"] = MOD_HP
    (folder / "unit_stats.json").write_text(json.dumps(stats))
    return game_dataset.DataModTables(f"key-{mod.title}", folder, mod.root, mod.title)


class FakeMods:
    """Two discoverable mods; `cache` maps a key to its tables (a hit), and an
    uncached pick runs `child` (Python source) as the derive subprocess."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.alpha, self.beta = _mod(tmp_path, "Alpha"), _mod(tmp_path, "Beta")
        self.tables = {mod: _tables(tmp_path, mod) for mod in (self.alpha, self.beta)}
        self.cache: dict[str, game_dataset.DataModTables] = {}
        self.listed = [self.alpha, self.beta]
        self.child = "pass"
        self.derives: list[str] = []
        monkeypatch.setattr(data_mods, "discover", lambda roots=None: list(self.listed))
        monkeypatch.setattr(data_mods, "entry_key", lambda mod, install: f"key-{mod.title}")
        monkeypatch.setattr(data_mods, "tables_for", lambda mod, key: self.cache.get(key))
        monkeypatch.setattr(data_mods, "derive_command", self._command)
        monkeypatch.setattr(data_mods, "genieutils_available", lambda: True)
        install = tmp_path / "install"
        install.mkdir()
        monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
        asset_source.clear_install_caches()  # get_install_path() is cached, maybe by a window built first

    def cached(self, *mods: data_mods.DataMod) -> None:
        for mod in mods:
            self.cache[f"key-{mod.title}"] = self.tables[mod]

    def _command(self, mod, install, key):
        self.derives.append(mod.title)
        return [sys.executable, "-c", self.child], Path.cwd()


@pytest.fixture()
def mods(tmp_path, monkeypatch) -> FakeMods:
    return FakeMods(tmp_path, monkeypatch)


@pytest.fixture()
def window():
    window = conftest.blank_window(load=False)
    yield window
    conftest.close_window(window)


def _scenario(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    shutil.copy(scenario_io.BLANK_TEMPLATE_PATH, path)
    return path


def _hp() -> int:
    return unit_stats_table.unit_stats(SPEARMAN)["hp"]


def _logged(window, monkeypatch) -> list[str]:
    lines: list[str] = []
    real = window._log_status
    monkeypatch.setattr(window, "_log_status", lambda message, **kwargs: (lines.append(message), real(message, **kwargs)))
    return lines


def _menu(window) -> dict[str, object]:
    window.data_mod_menu.rebuild()
    return {action.text(): action for action in window.data_mod_menu.menu.actions() if not action.isSeparator()}


def _pump_until(condition, timeout: float = 15.0) -> None:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting on the data mod parse"
        time.sleep(0.01)
        QApplication.processEvents()


def test_the_menu_lists_the_install_then_each_mod(window, mods):
    menu = _menu(window)
    assert list(menu) == ["None (the game's own data)", "Alpha", "Beta"]
    assert menu["None (the game's own data)"].isChecked()
    assert [action.toolTip() for action in menu.values() if not action.isEnabled()] == []


def test_with_no_mods_the_menu_says_so(window, mods):
    mods.listed = []
    assert not _menu(window)["No data mods found"].isEnabled()


def test_a_cached_pick_applies_at_once_and_titles_the_window(window, mods, tmp_path):
    mods.cached(mods.alpha)
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    install_hp = _hp()
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert _hp() == MOD_HP != install_hp
    assert window.windowTitle().startswith("a.aoe2scenario [Alpha]")
    assert _menu(window)["Alpha"].isChecked()
    assert mods.derives == [], "a hit never parses"
    _menu(window)["None (the game's own data)"].trigger()
    assert game_dataset.data_mod() is None and _hp() == install_hp
    assert "[Alpha]" not in window.windowTitle()


def test_each_saved_scenario_reopens_with_its_own_mod(window, mods, tmp_path):
    mods.cached(mods.alpha)
    a, b = _scenario(tmp_path, "a.aoe2scenario"), _scenario(tmp_path, "b.aoe2scenario")
    window.load_scenario(a)
    _menu(window)["Alpha"].trigger()
    assert settings.get_data_mod_for(a) == data_mods.identity(mods.alpha)
    window.load_scenario(b)
    assert game_dataset.data_mod() is None, "one file's mod never leaks onto the next"
    seen_at_render = []
    real_render = window._render_current
    window._render_current = lambda **kwargs: (seen_at_render.append(game_dataset.data_mod()), real_render(**kwargs))[1]
    window.load_scenario(a)
    assert seen_at_render[0] == mods.tables[mods.alpha], "applied before the first render"
    assert settings.get_data_mod_for(b) is None


def test_a_vanished_mod_falls_back_to_the_install_with_one_line(window, mods, tmp_path, monkeypatch):
    a = _scenario(tmp_path, "a.aoe2scenario")
    settings.set_data_mod_for(a, data_mods.identity(mods.alpha))
    mods.listed = [mods.beta]
    lines = _logged(window, monkeypatch)
    window.load_scenario(a)
    assert game_dataset.data_mod() is None
    alpha_lines = [line for line in lines if "Alpha" in line]
    assert len(alpha_lines) == 1 and "not found" in alpha_lines[0]
    assert settings.get_data_mod_for(a) is not None, "kept until the next pick or Save As"


def test_save_as_remembers_the_mod_under_the_new_path(window, mods, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    mods.cached(mods.alpha)
    a, c = _scenario(tmp_path, "a.aoe2scenario"), tmp_path / "c.aoe2scenario"
    settings.set_data_mod_for(c, data_mods.identity(mods.beta))  # a stale key at the target
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(c), "")))
    window.load_scenario(a)
    _menu(window)["Alpha"].trigger()
    window.save_as()
    assert settings.get_data_mod_for(c) == data_mods.identity(mods.alpha)
    _menu(window)["None (the game's own data)"].trigger()
    assert settings.get_data_mod_for(c) is None, "the document now lives at c"


def test_file_new_keeps_the_session_mod_and_never_remembers(window, mods):
    mods.cached(mods.alpha, mods.beta)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha], "a pick works with no document open"
    window.new_map()
    assert window._untitled and game_dataset.data_mod() == mods.tables[mods.alpha]
    _menu(window)["Beta"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.beta]
    assert settings._load_config().get("data_mod_by_scenario") is None


def test_autosave_recovery_reapplies_the_slots_mod(window, mods, tmp_path):
    mods.cached(mods.alpha)
    slot = _scenario(tmp_path, f"a.aoe2scenario.20261025-120000{autosave.AUTOSAVE_SUFFIX}")
    autosave.record("k", slot, str(tmp_path / "a.aoe2scenario"), untitled=False, data_mod=data_mods.identity(mods.alpha))
    entry = autosave.entries()[0]
    window._open_autosave_entry(entry)
    assert window._untitled and game_dataset.data_mod() == mods.tables[mods.alpha]
    assert settings._load_config().get("data_mod_by_scenario") is None


def test_the_autosave_tick_records_the_chosen_mod(window, mods, tmp_path):
    mods.cached(mods.alpha)
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    window._autosave_write()
    assert autosave.entries()[0].data_mod == data_mods.identity(mods.alpha)


def test_a_mods_derive_warnings_are_logged_once_and_counted_in_its_tooltip(window, mods, tmp_path, monkeypatch):
    """GH #262: the summary, up to WARNING_LINES warning lines and the rest's count,
    once per cache key; once it is logged, the menu item's tooltip carries the count."""
    from descape import data_mod_menu

    cap = data_mod_menu.WARNING_LINES
    warnings = tuple(f"gen_unit_graphic_map.py: unit_const {i} broke a rule (drawn alone)" for i in range(cap + 5))
    mods.tables[mods.alpha] = dataclasses.replace(mods.tables[mods.alpha], warnings=warnings)
    mods.cached(mods.alpha)
    lines = _logged(window, monkeypatch)
    a, b = _scenario(tmp_path, "a.aoe2scenario"), _scenario(tmp_path, "b.aoe2scenario")
    window.load_scenario(a)
    assert _menu(window)["Alpha"].toolTip() == "subscribed//Alpha", "no count before the log has the summary"
    _menu(window)["Alpha"].trigger()

    def summaries():
        return [line for line in lines if line.startswith(f"Data mod Alpha: {cap + 5} warnings")]

    assert len(summaries()) == 1
    body = summaries()[0].splitlines()
    assert len(body) == 1 + cap + 1
    assert body[1:cap + 1] == [f"  {warning}" for warning in warnings[:cap]]
    folder = mods.tables[mods.alpha].tables_dir
    assert "5 more" in body[-1] and str(folder / "meta.json") in body[-1]
    assert f"{cap + 5} warnings" in _menu(window)["Alpha"].toolTip()
    assert "warning" not in _menu(window)["Beta"].toolTip(), "a mod not picked has no count"
    _menu(window)["None (the game's own data)"].trigger()
    window.load_scenario(b)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert len(summaries()) == 1, "once per cache key"


def test_a_cached_mods_tooltip_counts_without_a_second_summary(window, mods, tmp_path, monkeypatch):
    """Beta shares Alpha's derive (same .dat, same key): its summary is not repeated, its count still shows."""
    warnings = ("gen_unit_stats.py: one rule (dropped)",)
    alpha = dataclasses.replace(mods.tables[mods.alpha], warnings=warnings)
    mods.tables[mods.alpha] = alpha
    mods.tables[mods.beta] = dataclasses.replace(mods.tables[mods.beta], key=alpha.key, warnings=warnings)
    mods.cached(mods.alpha)
    mods.cached(mods.beta)
    lines = _logged(window, monkeypatch)
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    _menu(window)["Beta"].trigger()
    assert len([line for line in lines if line.startswith("Data mod ")]) == 1
    assert "(1 warning;" in _menu(window)["Beta"].toolTip()


def test_a_mod_without_warnings_logs_no_summary(window, mods, monkeypatch):
    mods.cached(mods.alpha)
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert not [line for line in lines if "warnings" in line]
    assert _menu(window)["Alpha"].toolTip() == "subscribed//Alpha"


def _missing_art(monkeypatch, stems: list[str], textures: list[str]) -> None:
    from descape import unit_sprites

    monkeypatch.setattr(unit_sprites, "undrawable_stems", lambda: list(stems))
    monkeypatch.setattr(asset_source, "undrawable_terrain_textures", lambda: list(textures))


def test_a_data_mods_missing_art_is_logged_on_each_switch(window, mods, monkeypatch):
    """GH #272: the art nothing provides, a count, up to WARNING_LINES names and the rest's count."""
    from descape import data_mod_menu, mod_graphics

    cap = data_mod_menu.WARNING_LINES
    stems = [f"trainsiege{i:02d}-c1_x1" for i in range(cap + 3)]
    _missing_art(monkeypatch, stems, ["swgb_metal.dds"])
    mods.cached(mods.alpha)
    mod_graphics.set_enabled(True)
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()

    def logged():
        return [line for line in lines if line.startswith(f"Data mod Alpha: {cap + 4} art files that no active mod")]

    assert len(logged()) == 1
    body = logged()[0].splitlines()
    assert body[1:] == [*(f"  {stem}" for stem in stems[:cap]), "  ... and 4 more"]
    _menu(window)["None (the game's own data)"].trigger()
    _menu(window)["Alpha"].trigger()
    assert len(logged()) == 2, "once per switch to the mod"


def test_with_mod_art_off_the_missing_art_is_one_line(window, mods, monkeypatch):
    from descape import mod_graphics

    _missing_art(monkeypatch, ["swgb_new_x1"], [])
    mods.cached(mods.alpha)
    mod_graphics.set_enabled(False)
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    assert [line for line in lines if "art file" in line] == [
        "Data mod Alpha: 1 art file not in the game's install; View > Layers > Mod Art draws them from its art mod"
    ]


def test_a_data_mod_whose_art_is_all_there_logs_no_art_line(window, mods, monkeypatch):
    _missing_art(monkeypatch, [], [])
    mods.cached(mods.alpha)
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert not [line for line in lines if "art file" in line]


def test_an_uncached_pick_parses_in_a_subprocess_then_swaps(window, mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    marker = tmp_path / "derived"
    mods.child = f"open({str(marker)!r}, 'w').close()"
    real_lookup = data_mods.tables_for
    monkeypatch.setattr(data_mods, "tables_for", lambda mod, key: real_lookup(mod, key) if marker.exists() else None)
    mods.cached(mods.alpha)
    _menu(window)["Alpha"].trigger()
    assert mods.derives == ["Alpha"]
    label = window.data_mod_menu.parsing_label
    assert not label.isHidden() and "Alpha" in label.text()
    assert game_dataset.data_mod() is None, "the install's data until the parse is done"
    _pump_until(lambda: game_dataset.data_mod() is not None)
    assert game_dataset.data_mod() == mods.tables[mods.alpha] and _hp() == MOD_HP
    assert label.isHidden() and "[Alpha]" in window.windowTitle()


def test_an_unreadable_mod_keeps_the_data_and_is_disabled(window, mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    message = "genieutils-py cannot read this .dat (VER 7.2, AssertionError)"
    mods.child = f"import sys; print('noise', file=sys.stderr); print({message!r}, file=sys.stderr); sys.exit(1)"
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    _pump_until(window.data_mod_menu.parsing_label.isHidden)
    assert game_dataset.data_mod() is None
    assert f"Data mod Alpha cannot be used: {message}" in lines
    alpha = _menu(window)["Alpha"]
    assert not alpha.isEnabled() and alpha.toolTip() == message
    assert _menu(window)["None (the game's own data)"].isChecked()
    assert settings.get_data_mod_for(window.scenario.path) is None, "the failed pick is not remembered"


def test_a_finished_parse_waits_for_an_idle_window(window, mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    controller = window.data_mod_menu
    controller._wanted = mods.alpha
    renders = []
    real_refresh = window.refresh_map
    monkeypatch.setattr(window, "refresh_map", lambda: (renders.append(game_dataset.data_mod()), real_refresh()))
    window._busy = True
    try:
        controller._swap(mods.tables[mods.alpha], mods.alpha)
        assert game_dataset.data_mod() is None and renders == [], "no swap inside a busy section"
        assert controller._apply_timer.isActive()
    finally:
        window._busy = False
    _pump_until(lambda: game_dataset.data_mod() is not None, timeout=5.0)
    assert renders == [mods.tables[mods.alpha]], "the swap re-renders once idle"


def test_a_mid_stroke_parse_waits_for_the_stroke(window, mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    controller = window.data_mod_menu
    controller._wanted = mods.alpha
    stroking = [True]
    monkeypatch.setattr(window, "_stroke_in_progress", lambda: stroking[0])
    controller._swap(mods.tables[mods.alpha], mods.alpha)
    assert game_dataset.data_mod() is None
    stroking[0] = False
    _pump_until(lambda: game_dataset.data_mod() is not None, timeout=5.0)


def test_return_of_rome_disables_the_mods_and_remembers_nothing(window, mods, tmp_path):
    mods.cached(mods.alpha)
    _menu(window)["Alpha"].trigger()
    window.load_scenario(ror_copy(tmp_path))
    assert game_dataset.active() == game_dataset.ROR and game_dataset.data_mod() is None
    alpha = _menu(window)["Alpha"]
    assert not alpha.isEnabled() and "Return of Rome" in alpha.toolTip()
    assert window.data_mod_menu.identity() is None
    assert "[Alpha]" not in window.windowTitle()


def test_closing_the_window_stops_a_parse(window, mods):
    mods.child = "import time; time.sleep(30)"
    _menu(window)["Alpha"].trigger()
    process = window.data_mod_menu._job.process
    conftest.close_window(window)
    from PyQt5.QtCore import QProcess

    assert process.state() == QProcess.NotRunning


def test_opening_another_document_mid_parse_drops_the_parse(window, mods, tmp_path, monkeypatch):
    """The pick was A's: B (no memory) opens with the install's data, and the parse never lands on it."""
    a, b = _scenario(tmp_path, "a.aoe2scenario"), _scenario(tmp_path, "b.aoe2scenario")
    window.load_scenario(a)
    marker = tmp_path / "derived"
    mods.child = f"import time; time.sleep(0.5); open({str(marker)!r}, 'w').close()"
    real_lookup = data_mods.tables_for
    monkeypatch.setattr(data_mods, "tables_for", lambda mod, key: real_lookup(mod, key) if marker.exists() else None)
    mods.cached(mods.alpha)
    _menu(window)["Alpha"].trigger()
    window.load_scenario(b)
    assert window.data_mod_menu._job is None and window.data_mod_menu.parsing_label.isHidden()
    deadline = time.monotonic() + 1.5  # three times the child's run
    _pump_until(lambda: time.monotonic() > deadline)
    assert game_dataset.data_mod() is None and not marker.exists(), "the killed parse never finished or swapped"
    assert settings.get_data_mod_for(a) == data_mods.identity(mods.alpha), "A keeps its pick"


def test_a_parser_that_cannot_start_fails_readably(window, mods, monkeypatch):
    monkeypatch.setattr(data_mods, "derive_command", lambda mod, install, key: (["/nonexistent/descape-derive"], Path.cwd()))
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    _pump_until(window.data_mod_menu.parsing_label.isHidden)
    assert game_dataset.data_mod() is None
    assert any(line.startswith("Data mod Alpha cannot be used: the parser could not start") for line in lines)
    assert not _menu(window)["Alpha"].isEnabled()


# --- GH #263: a mod's civilizations.json and eras.json ------------------


def _pixel() -> tuple[int, ...]:
    return tuple(asset_source.get_unit_icon(*object_catalog.portrait_for(ARCHER))[0, 0])


def test_a_repick_after_an_icon_table_edit_shows_the_new_portrait(window, mods, tmp_path, monkeypatch):
    """GH #363: the mod's own widgetui/icons.json is in side_key, so picking the mod again re-reads it."""
    from PIL import Image

    entry = SimpleNamespace(path=mods.tables[mods.alpha].tables_dir, mod_only_ids=frozenset(), warnings=())
    monkeypatch.setattr(data_mods.dat_cache, "lookup", lambda key: entry)
    monkeypatch.setattr(data_mods, "tables_for", _REAL_TABLES_FOR)
    ui = mods.alpha.root / "widgetui"
    textures = []
    for name, color in (("Red", (255, 0, 0, 255)), ("Green", (0, 255, 0, 255))):
        path = ui / "textures" / "ingame" / "units" / f"{name}.dds"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (4, 4), color).save(path, format="PNG")
        textures.append({"RefName": name, "FileName": f"textures/ingame/units/{name}.dds"})
    (ui / "materials.json").write_text(json.dumps({"AtlasTextures": [{"AtlasDef": {"Textures": textures}}]}))
    icon = object_catalog.icon_for(ARCHER)
    (ui / "icons.json").write_text(json.dumps({"Units": {f"{icon:03d}": "Red"}}))
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod().root == mods.alpha.root
    assert _pixel() == (255, 0, 0, 255)
    (ui / "icons.json").write_text(json.dumps({"Units": {f"{icon:03d}": "Green"}}))
    assert _pixel() == (255, 0, 0, 255), "cached until the mod is picked again"
    _menu(window)["Alpha"].trigger()
    assert _pixel() == (0, 255, 0, 255)


def _side_files(mod: data_mods.DataMod) -> None:
    dat = mod.root / "resources" / "_common" / "dat"
    dat.mkdir(parents=True)
    civs = [
        {"data_name": "GAIA", "internal_name": "Gaia", "name_string_id": 1, "era": "base"},
        {"data_name": "MOD-A-CIV", "internal_name": "Alpha Civ", "name_string_id": 2, "era": "future"},
        {"data_name": "MOD-B-CIV", "internal_name": "Beta Civ", "name_string_id": 3, "era": "future"},
        {"data_name": "CHINESE-CIV", "internal_name": "Old Civ", "name_string_id": 4, "era": "base"},
    ]
    (dat / "civilizations.json").write_text(json.dumps({"civilization_list": civs}))
    eras = [
        {"Name": "base", "Ages": [{"NameId": id_} for id_ in range(11, 16)]},
        {"Name": "future", "Ages": [{"NameId": id_} for id_ in range(21, 26)]},
    ]
    (dat / "eras.json").write_text(json.dumps(eras))
    strings = mod.root / "resources" / "en" / "strings" / "key-value"
    strings.mkdir(parents=True)
    lines = [f'{id_} "Base {id_}"' for id_ in range(11, 16)] + [f'{id_} "Future {id_}"' for id_ in range(21, 26)]
    (strings / "key-value-modded-strings-utf8.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def side_mods(mods, monkeypatch) -> FakeMods:
    monkeypatch.setattr(asset_source, "get_language", lambda: "en")
    _side_files(mods.alpha)
    mods.cached(mods.alpha, mods.beta)
    return mods


def _pick_civ(window, token: str, field_id: str = "civilization") -> None:
    window.mode_combo.setCurrentText("Players")
    combo = window.players_panel.widget_for(field_id)
    index = combo.findData(token)
    assert index >= 0, token
    combo.setCurrentIndex(index)


def _save_as(window, path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    window.save_as()


def _stored_civ(path: Path, player: int = 1):
    from descape import civ_availability

    return civ_availability.player_civ_value(scenario_io.load_map_and_units(path), player)


def test_players_tab_lists_the_mods_civs_and_relabels_ages_on_a_civ_edit(window, side_mods, tmp_path):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    window.mode_combo.setCurrentText("Players")
    panel = window.players_panel
    civs = panel.widget_for("civilization")
    assert civs.currentText() == "Random", "the blank file's Random stays Random"
    assert [civs.itemText(i) for i in range(civs.count())][-2:] == ["Full Random", "Random"]
    assert {civs.itemData(i) for i in range(civs.count())} >= {"MOD-A-CIV", "MOD-B-CIV", "CHINESE-CIV"}
    assert "GAIA" not in {civs.itemData(i) for i in range(civs.count())}
    ages = panel.widget_for("starting_age")
    ages.setCurrentIndex(ages.findData(4))

    _pick_civ(window, "CHINESE-CIV")
    assert panel.widget_for("starting_age") is ages, "relabelled in place, not rebuilt"
    assert [ages.itemText(i) for i in range(ages.count())] == [f"Base {id_}" for id_ in range(11, 16)]
    assert ages.currentData() == 4
    _pick_civ(window, "MOD-A-CIV")
    assert [ages.itemText(i) for i in range(ages.count())] == [f"Future {id_}" for id_ in range(21, 26)]
    assert ages.currentData() == 4
    assert "Dark Age" in ages.toolTip()
    assert ages.findData(2) == 0, "values stay 2..6"

    panel.select_player(0)
    gaia_ages = panel.widget_for("starting_age")
    assert gaia_ages.itemText(0) == "Future 21", "the era most civs share, not GAIA's own base"

    _menu(window)["None (the game's own data)"].trigger()
    panel.select_player(1)
    ages = panel.widget_for("starting_age")
    assert ages.itemText(0) == "Dark Age" and not ages.toolTip()
    civs = panel.widget_for("civilization")
    assert civs.findText("Britons") >= 0


def test_a_mod_civs_player_shows_that_civs_own_portrait(window, side_mods, tmp_path):
    """GH #372: a mod's civ_icons are keyed by its own civ ids (civilizations.json's order)."""
    from descape import civ_art

    folder = side_mods.tables[side_mods.alpha].tables_dir
    catalog = json.loads((folder / "object_catalog.json").read_text())
    catalog["objects"][str(ARCHER)]["civ_icons"] = {"2": 4242}
    (folder / "object_catalog.json").write_text(json.dumps(catalog))
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    _pick_civ(window, "MOD-B-CIV")
    assert civ_art.player_portrait_civ(window.scenario, 1) == 2
    assert object_catalog.icon_for(ARCHER, 2) == 4242
    window.units_panel.select_owner(1)
    assert window.units_panel.placing_card._civ == 2, "the Players edit reached the units panel"
    _pick_civ(window, "MOD-A-CIV")
    assert window.units_panel.placing_card._civ == 1
    assert object_catalog.icon_for(ARCHER, 1) == object_catalog.icon_for(ARCHER), "no portrait of its own"


def test_a_civ_the_mod_lacks_shows_its_first_civ_and_random_and_comes_back_without_it(
    window, side_mods, tmp_path, monkeypatch
):
    window.load_scenario(_scenario(tmp_path, "seed.aoe2scenario"))
    _pick_civ(window, "HUN-CIV")
    _pick_civ(window, "HUN-CIV", "architecture")
    stored = tmp_path / "huns.aoe2scenario"
    _save_as(window, stored, monkeypatch)
    settings.set_data_mod_for(stored, data_mods.identity(side_mods.alpha))
    window.load_scenario(stored)
    window.mode_combo.setCurrentText("Players")
    panel = window.players_panel
    civs, style = panel.widget_for("civilization"), panel.widget_for("architecture")
    assert (civs.currentText(), civs.currentData()) == ("Alpha Civ", "MOD-A-CIV")
    assert (style.currentText(), style.currentData()) == ("Random", "RANDOM-CIV")
    assert not any(civs.itemText(i).startswith("unknown") for i in range(civs.count())), "no extra row"
    assert "first civilization, Alpha Civ" in civs.toolTip() and "Random is shown" in style.toolTip()
    assert panel.current_values()["civilization"] == "HUN-CIV"
    assert not window.edit_history.is_dirty, "shown, not written"

    civs.activated.emit(civs.currentIndex())
    assert window.option_edits.current_value("player:civilization:1") == "MOD-A-CIV", "re-picking the shown civ writes it"
    window.undo()
    assert not window.edit_history.is_dirty
    _menu(window)["None (the game's own data)"].trigger()
    civs, style = panel.widget_for("civilization"), panel.widget_for("architecture")
    assert (civs.currentText(), civs.currentData()) == ("Huns", "HUN-CIV")
    assert style.currentData() == "HUN-CIV"


def test_a_mod_only_civ_still_saves_undoes_and_redoes_after_a_switch_to_none(window, side_mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "a.aoe2scenario"))
    _menu(window)["Alpha"].trigger()
    _pick_civ(window, "MOD-A-CIV")
    _menu(window)["None (the game's own data)"].trigger()

    window.undo()
    window.redo()
    out = tmp_path / "out.aoe2scenario"
    _save_as(window, out, monkeypatch)
    assert _stored_civ(out) == "MOD-A-CIV"

    # The saved file stores it, and this session had the mod, so its Players tab stays writable.
    window.load_scenario(out)
    window.mode_combo.setCurrentText("Players")
    assert window.players_panel.widget_for("food").isEnabled()
    assert window.players_panel.widget_for("civilization").currentData() == "MOD-A-CIV"


def test_stored_building_art_follows_the_mod_on_open_and_on_a_switch(window, side_mods, tmp_path, monkeypatch):
    window.load_scenario(_scenario(tmp_path, "seed.aoe2scenario"))
    _pick_civ(window, "CHINESE-CIV")
    stored = tmp_path / "stored.aoe2scenario"
    _save_as(window, stored, monkeypatch)
    assert window.scenario.player_art[1][0] == 6, "AoE2's index with no mod"

    other = tmp_path / "other.aoe2scenario"
    shutil.copy(stored, other)
    settings.set_data_mod_for(stored, data_mods.identity(side_mods.alpha))
    settings.set_data_mod_for(other, None)
    window.load_scenario(stored)
    assert game_dataset.data_mod() == side_mods.tables[side_mods.alpha]
    assert window.scenario.player_art[1][0] == 3, "the mod's list position, on open"
    window.load_scenario(other)
    assert game_dataset.data_mod() is None
    assert window.scenario.player_art[1][0] == 6, "not the previous file's mod"

    drawn = []
    real = window.refresh_map
    monkeypatch.setattr(window, "refresh_map", lambda: (drawn.append(window.scenario.player_art[1][0]), real()))
    _menu(window)["Alpha"].trigger()
    assert drawn == [3], "updated before the switch's redraw"
    _menu(window)["None (the game's own data)"].trigger()
    assert drawn == [3, 6]


def test_opening_a_file_ignores_the_previous_documents_pending_civ(window, side_mods, tmp_path):
    a, b = _scenario(tmp_path, "a.aoe2scenario"), _scenario(tmp_path, "b.aoe2scenario")
    settings.set_data_mod_for(b, data_mods.identity(side_mods.alpha))
    window.load_scenario(a)
    _pick_civ(window, "CHINESE-CIV")
    assert window.scenario.player_art[1][0] == 6
    window.load_scenario(b)
    assert game_dataset.data_mod() == side_mods.tables[side_mods.alpha]
    assert window.scenario.player_art[1][0] is None, "b stores a random civ, not a's pending one"
