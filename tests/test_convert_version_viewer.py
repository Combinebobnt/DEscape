"""File > Convert to Version 1.59… (GH #307): the viewer's wiring around
descape/scenario_convert.py. The conversion itself is
tests/test_scenario_convert.py's; this is the action, the prompt, and the
document replacement.

Every ViewerWindow ends with edit_history.mark_saved() before close(): a
converted document is dirty on purpose, and closeEvent would otherwise block
on a modal forever.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape.scenario_io import load_map_and_units, parse_triggers

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _questions(monkeypatch, answer):
    from PyQt5.QtWidgets import QMessageBox

    asked = []

    def question(_parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def _info_version(window) -> str:
    section = window._info_sections["File"]
    rows = {section.child(i).text(0): section.child(i).text(1) for i in range(section.childCount())}
    return rows["Scenario version"]


def _capture_flag_enabled(window) -> bool:
    window.mode_combo.setCurrentText("Units")  # builds the unit index
    window.units_panel.show_unit(window.map_view._unit_index.entries[0])
    return window.units_panel.unit_field_editors["capture_flag"].isEnabled()


@pytest.fixture
def window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    w = ViewerWindow()
    yield w
    w.edit_history.mark_saved()
    w.close()


@pytest.fixture
def copy_158(tmp_path) -> Path:
    path = tmp_path / "triggers.aoe2scenario"
    shutil.copyfile(TRIGGER_FIXTURE, path)
    return path


def test_the_action_follows_the_open_document(window, copy_158) -> None:
    action = window.convert_version_action
    assert not action.isEnabled() and action.toolTip(), "no map: off, with a tooltip saying what it needs"
    assert window._keybind_actions["file_convert"] is action
    window.load_scenario(copy_158)
    assert action.isEnabled()
    window.new_action.trigger()
    assert window.scenario.scenario_version == "1.59"
    assert not action.isEnabled() and "already" in action.toolTip()


def test_cancel_changes_nothing(window, copy_158, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window.load_scenario(copy_158)
    before = window.scenario
    committed = []
    monkeypatch.setattr(window, "_commit_focused_field", lambda: committed.append(1))
    asked = _questions(monkeypatch, QMessageBox.Cancel)
    window.convert_version_action.trigger()
    assert committed, "a typed value is committed before the models are converted and dropped"
    assert len(asked) == 1 and "Save overwrites triggers.aoe2scenario" in asked[0][1]
    assert window.scenario is before and not window.edit_history.is_dirty


def test_accept_converts_in_place_and_save_writes_159(window, copy_158, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window.load_scenario(copy_158)
    assert not _capture_flag_enabled(window)
    _questions(monkeypatch, QMessageBox.Yes)
    window.convert_version_action.trigger()

    assert window.scenario.scenario_version == "1.59" and _info_version(window) == "1.59"
    assert window.scenario.path == copy_158, "the document keeps its path"
    assert window.windowTitle().startswith("*")
    assert window.edit_history.records == [] and window.edit_history.is_dirty
    assert not window.convert_version_action.isEnabled()
    assert _capture_flag_enabled(window)

    window.save()
    back = load_map_and_units(copy_158)
    assert (back.scenario_version, back.trigger_version) == ("1.59", 5.0)
    assert len(parse_triggers(back).triggers) == 4
    assert not window.edit_history.is_dirty


def test_a_dirty_document_keeps_its_edits(window, copy_158, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window.load_scenario(copy_158)
    window.mode_combo.setCurrentText("Triggers")
    window.trigger_panel.select_triggers([0])
    window.copy_triggers()
    window.trigger_structural_edit("paste", [])
    names = [t.name for t in window.trigger_edits.manager().triggers]
    assert len(names) == 5 and window.edit_history.is_dirty
    _questions(monkeypatch, QMessageBox.Yes)
    window.convert_version_action.trigger()
    assert window.scenario.scenario_version == "1.59"
    assert [t.name for t in parse_triggers(window.scenario).triggers] == names


def test_a_failed_conversion_leaves_the_158_document_open(window, copy_158, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    from descape.scenario_convert import ConversionRefusedError
    from testkit.viewer_family import set_viewer_global

    window.load_scenario(copy_158)
    before = window.scenario
    _questions(monkeypatch, QMessageBox.Yes)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda _p, title, text, *a, **k: shown.append(title)))

    def refuse(*args, **kwargs):
        raise ConversionRefusedError("forced")

    set_viewer_global(monkeypatch, "convert_scenario", refuse)
    window.convert_version_action.trigger()
    assert shown == ["Convert to 1.59 failed"]
    assert window.scenario is before and before.scenario_version == "1.58"
    assert window.convert_version_action.isEnabled()
    assert "forced" in window.status_log.toPlainText()
