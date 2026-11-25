"""File > Browse Game Scenarios… (TASK-123): the dialog over a fixture list,
and the viewer's menu entry and open path."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape import settings
from descape.content_roots import KIND_MOD_LOCAL, KIND_MOD_SUBSCRIBED, KIND_PROFILE, ContentRoot
from descape.scenario_io import BLANK_TEMPLATE_PATH
from descape.scenario_library import ScenarioEntry

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_PROFILE = ContentRoot(KIND_PROFILE, "Profile 7656", Path("/p/7656"), profile_id="7656")
_SUB = ContentRoot(KIND_MOD_SUBSCRIBED, "m", Path("/p/7656/mods/subscribed/1_F"), mod_id="1_F", title="Fixture Mod")
_OFF = ContentRoot(KIND_MOD_LOCAL, "m", Path("/p/7656/mods/local/Off"), mod_id="Off", title="Off", enabled=False)


def _entries() -> list[ScenarioEntry]:
    return [
        ScenarioEntry(Path("/p/a/Alpha.aoe2scenario"), _PROFILE, 10 * 1024, 1_000.0),
        ScenarioEntry(Path("/p/a/Beta.aoe2scenario"), _PROFILE, 9 * 1024, 3_000.0),
        ScenarioEntry(Path("/p/s/Workshop.aoe2scenario"), _SUB, 50 * 1024, 2_000.0),
        ScenarioEntry(Path("/p/o/Hidden.aoe2scenario"), _OFF, 1024, 500.0),
    ]


def _dialog(entries=None, has_profile=True, on_open=None):
    conftest.ensure_qapp()
    from descape.scenario_browser import ScenarioBrowserDialog

    rows = _entries() if entries is None else entries
    return ScenarioBrowserDialog(None, on_open or (lambda _p: None), source=lambda: (rows, has_profile))


def _names(dialog) -> list[str]:
    return [item.text(0) for item in dialog.visible_items()]


def test_rows_columns_and_newest_first():
    dialog = _dialog()
    try:
        assert [dialog.tree.headerItem().text(i) for i in range(4)] == ["Name", "Source", "Modified", "Size"]
        assert _names(dialog) == ["Beta", "Workshop", "Alpha", "Hidden"]
        sources = {item.text(0): item.text(1) for item in dialog.visible_items()}
        assert sources["Workshop"] == "Fixture Mod (subscribed mod)"
        assert sources["Hidden"] == "Off (local mod, disabled in game)"
        assert dialog.count_label.text() == "4 scenarios"
        assert dialog.visible_items()[0].toolTip(0) == str(Path("/p/a/Beta.aoe2scenario"))
    finally:
        dialog.close()


def test_size_sorts_numerically():
    from PyQt5.QtCore import Qt

    dialog = _dialog()
    try:
        dialog.tree.sortByColumn(3, Qt.AscendingOrder)
        assert _names(dialog) == ["Hidden", "Beta", "Alpha", "Workshop"]  # 9 KB below 10 KB
    finally:
        dialog.close()


def test_filter_on_source_hides_rows_and_updates_count():
    dialog = _dialog()
    try:
        dialog.filter_edit.setText("FIXTURE")
        assert _names(dialog) == ["Workshop"]
        assert dialog.count_label.text() == "1 of 4 scenarios"
        dialog.filter_edit.setText("nothing matches")
        assert _names(dialog) == []
        assert not dialog.open_button.isEnabled()
    finally:
        dialog.close()


def test_activating_a_row_reports_its_path():
    opened: list[Path] = []
    dialog = _dialog(on_open=opened.append)
    try:
        item = dialog.visible_items()[1]
        dialog.tree.itemActivated.emit(item, 0)
        assert opened == [Path("/p/s/Workshop.aoe2scenario")]
    finally:
        dialog.close()


def test_return_on_a_row_opens_it_once():
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    opened: list[Path] = []
    dialog = _dialog(on_open=opened.append)
    try:
        dialog.show()
        dialog.tree.setFocus()
        QTest.keyClick(dialog.tree, Qt.Key_Return)
        assert opened == [Path("/p/a/Beta.aoe2scenario")]  # not again through the default Open button
    finally:
        dialog.close()


def test_empty_states():
    from descape.scenario_browser import NO_FILES_TEXT, NO_PROFILE_TEXT

    for has_profile, text in ((False, NO_PROFILE_TEXT), (True, NO_FILES_TEXT)):
        dialog = _dialog(entries=[], has_profile=has_profile)
        try:
            assert dialog.count_label.text() == text
            assert not dialog.open_button.isEnabled()
        finally:
            dialog.close()


@pytest.fixture
def window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    win = ViewerWindow()
    yield win
    win.edit_history.mark_saved()
    win.close()


def test_menu_entry_follows_open_recent_and_is_rebindable(window):
    file_menu = window.recent_menu.menuAction().associatedWidgets()[0]
    actions = file_menu.actions()
    index = actions.index(window.recent_menu.menuAction())
    assert actions[index + 1] is window.browse_scenarios_action
    assert window.browse_scenarios_action.isEnabled()
    assert ("file_browse_scenarios", "Browse Game Scenarios…", "") in settings.REBINDABLE_ACTIONS


def _profile_copy(tmp_path: Path, monkeypatch) -> Path:
    folder = tmp_path / "profile" / "7656" / "resources" / "_common" / "scenario"
    folder.mkdir(parents=True)
    target = folder / "Mine.aoe2scenario"
    shutil.copy(BLANK_TEMPLATE_PATH, target)
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "profile"))
    return target


def test_browser_lists_and_opens_a_profile_scenario(tmp_path, monkeypatch, window):
    target = _profile_copy(tmp_path, monkeypatch)
    window.show_scenario_browser()
    dialog = window._scenario_browser_dialog
    assert dialog.isVisible()
    assert _names(dialog) == ["Mine"]
    window._open_browsed_scenario(target)
    assert window.scenario is not None and window.scenario.path == target
    assert not dialog.isVisible()


def test_cancelling_the_unsaved_prompt_keeps_the_document(tmp_path, monkeypatch, window):
    from PyQt5.QtWidgets import QMessageBox

    target = _profile_copy(tmp_path, monkeypatch)
    window.new_map()
    window.edit_history.saved_at_cursor = None  # no save point: unsaved changes
    assert window.edit_history.is_dirty
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Cancel))
    before = window.scenario
    window.show_scenario_browser()
    window._open_browsed_scenario(target)
    assert window.scenario is before
    assert window._scenario_browser_dialog.isVisible()
