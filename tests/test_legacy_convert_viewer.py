"""File > Convert to .aoe2scenario (GH #315): the viewer's wiring around
descape/legacy_convert.py and batch_api.convert_legacy(). The conversion
itself is tests/test_legacy_convert.py's and the oracle's; this is the
action's gate, the untitled document it opens, and the report.

A converted document is dirty on purpose: every window ends with
edit_history.mark_saved() before close(), or closeEvent would block on a modal.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from descape import batch_api, legacy_convert

import conftest

TEST_MAP = Path(__file__).resolve().parent / "fixtures" / "test_map_240x240.aoe2scenario"

_GUI = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]


@pytest.fixture
def window():
    w = conftest.blank_window(load=False)
    yield w
    w.edit_history.mark_saved()
    w.close()


@pytest.fixture
def boxes(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "exec_", lambda box: shown.append(box) or QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda _p, title, text, *a: shown.append((title, text))))
    return shown


def _open_legacy(window, tmp_path, fixture: Path | None = None):
    loaded = conftest.legacy_loaded(fixture)
    loaded.path = tmp_path / "Old Map.scx"
    window.load_scenario(loaded.path, loaded=loaded)
    assert window.scenario is loaded
    return loaded


@_GUI[0]
@_GUI[1]
def test_the_action_is_on_only_for_a_legacy_document(window, tmp_path) -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    action = window.convert_legacy_action
    assert window._keybind_actions["file_convert_legacy"] is action
    assert action in window.menuBar().actions()[0].menu().actions()
    assert not action.isEnabled() and action.toolTip(), "no map: off, saying what it needs"
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert not action.isEnabled(), "a DE document has nothing to convert"
    _open_legacy(window, tmp_path)
    assert action.isEnabled()


@_GUI[0]
@_GUI[1]
def test_converting_opens_an_unsaved_editable_copy_and_saves_beside_the_original(window, tmp_path, boxes) -> None:
    legacy = _open_legacy(window, tmp_path)
    assert not window.save_as_action.isEnabled(), "the legacy document itself stays read-only"
    window.convert_legacy_action.trigger()
    doc = window.scenario
    assert doc is not legacy and not doc.source_format.legacy
    assert window._untitled and window.edit_history.is_dirty
    assert window.windowTitle().startswith("*")
    assert window.save_as_action.isEnabled() and window.game_data_lock_reason() is None
    assert not window.convert_legacy_action.isEnabled()
    assert window._save_as_start_path() == str(tmp_path / "Old Map.aoe2scenario")
    assert "Converted Old Map.scx" in window.status_log.toPlainText()
    assert boxes == [], "nothing was left out, so no message box"
    window.edit_history.mark_saved()
    window.new_action.trigger()
    assert Path(window._save_as_start_path()).parent != tmp_path, "the start path is the converted document's only"


@pytest.mark.test_map
@_GUI[0]
@_GUI[1]
def test_what_was_left_out_is_logged_and_shown_once(window, tmp_path, boxes) -> None:
    legacy = _open_legacy(window, tmp_path, TEST_MAP)
    count = sum(map(len, legacy.unit_manager.units))
    window.convert_legacy_action.trigger()
    assert not window.scenario.source_format.legacy
    assert sum(map(len, window.scenario.unit_manager.units)) == count
    [box] = boxes
    assert "Left out" in box.detailedText() and "triggers" in box.detailedText()
    assert "Convert: Left out" in window.status_log.toPlainText()


@_GUI[0]
@_GUI[1]
def test_a_failed_conversion_keeps_the_legacy_document(window, tmp_path, boxes, monkeypatch) -> None:
    legacy = _open_legacy(window, tmp_path)

    def fail(_doc):
        raise ValueError("boom")

    monkeypatch.setattr(legacy_convert, "convert", fail)
    window.convert_legacy_action.trigger()
    assert window.scenario is legacy and not window._untitled
    assert boxes == [("Convert to .aoe2scenario failed", "ValueError: boom")]


@pytest.mark.test_map
def test_the_batch_api_converts_a_file_reproducibly_by_seed(monkeypatch) -> None:
    monkeypatch.setattr(batch_api, "load", lambda path: conftest.legacy_loaded(TEST_MAP))

    def images(loaded):
        return [(u.reference_id, u.rotation, u.initial_animation_frame) for units in loaded.unit_manager.units for u in units]

    first, report = batch_api.convert_legacy("old.scx", seed=7)
    again, _ = batch_api.convert_legacy("old.scx", seed=7)
    assert not first.source_format.legacy and report.rewritten["trees"]
    assert images(first) == images(again)
    plain, _ = legacy_convert.load_converted(conftest.legacy_loaded(TEST_MAP), rng=random.Random(7))
    assert images(plain) == images(first), "seed=7 is the same as passing Random(7)"


@_GUI[0]
@_GUI[1]
def test_a_subscribed_originals_save_as_starts_in_the_scenario_folder(window, tmp_path, boxes, monkeypatch) -> None:
    from descape import scenario_library

    folder = tmp_path / "mods" / "subscribed" / "123"
    folder.mkdir(parents=True)
    monkeypatch.setattr(scenario_library, "default_folder", lambda: tmp_path)
    loaded = conftest.legacy_loaded()
    loaded.path = folder / "Old Map.scx"
    window.load_scenario(loaded.path, loaded=loaded)
    window.convert_legacy_action.trigger()
    assert window._save_as_start_path() == str(tmp_path / "Old Map.aoe2scenario")
