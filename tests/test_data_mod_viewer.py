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

import pytest

from descape import (
    asset_source,
    autosave,
    dat_derive,
    data_mods,
    game_dataset,
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
MOD_HP = 4242


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
    monkeypatch.setattr(window, "_log_status", lambda message: (lines.append(message), real(message)))
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
    import descape.viewer as viewer_module

    mods.cached(mods.alpha)
    a, c = _scenario(tmp_path, "a.aoe2scenario"), tmp_path / "c.aoe2scenario"
    settings.set_data_mod_for(c, data_mods.identity(mods.beta))  # a stale key at the target
    monkeypatch.setattr(viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(c), "")))
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
    """GH #262: the summary, up to 20 warning lines and the rest's count, once
    per cache key; once it is logged, the menu item's tooltip carries the count."""
    from descape import data_mod_menu

    warnings = tuple(f"gen_unit_graphic_map.py: unit_const {i} broke a rule (drawn alone)" for i in range(25))
    mods.tables[mods.alpha] = dataclasses.replace(mods.tables[mods.alpha], warnings=warnings)
    mods.cached(mods.alpha)
    lines = _logged(window, monkeypatch)
    a, b = _scenario(tmp_path, "a.aoe2scenario"), _scenario(tmp_path, "b.aoe2scenario")
    window.load_scenario(a)
    assert _menu(window)["Alpha"].toolTip() == "subscribed//Alpha", "no count before the log has the summary"
    _menu(window)["Alpha"].trigger()

    def summaries():
        return [line for line in lines if line.startswith("Data mod Alpha: 25 warnings: ")]

    assert len(summaries()) == 1
    body = summaries()[0].splitlines()
    assert len(body) == 1 + data_mod_menu.WARNING_LINES + 1
    assert body[1:21] == [f"  {warning}" for warning in warnings[:20]]
    folder = mods.tables[mods.alpha].tables_dir
    assert body[-1] == f"  ... and 5 more (full list in {folder / 'meta.json'})"
    assert _menu(window)["Alpha"].toolTip() == "subscribed//Alpha (25 warnings; see the status log)"
    assert _menu(window)["Beta"].toolTip() == "subscribed//Beta", "a mod not picked has no count"
    _menu(window)["None (the game's own data)"].trigger()
    window.load_scenario(b)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert len(summaries()) == 1, "once per cache key"


def test_a_mod_without_warnings_logs_no_summary(window, mods, monkeypatch):
    mods.cached(mods.alpha)
    lines = _logged(window, monkeypatch)
    _menu(window)["Alpha"].trigger()
    assert game_dataset.data_mod() == mods.tables[mods.alpha]
    assert not [line for line in lines if "warnings" in line]
    assert _menu(window)["Alpha"].toolTip() == "subscribed//Alpha"


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
    assert game_dataset.is_ror() and game_dataset.data_mod() is None
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
