"""File > Resize Map… (TASK-107, GH #45): the viewer's wiring around
descape/scenario_resize.py. The resize itself is tests/test_scenario_resize.py's;
this is the action, the prompts, and the document replacement.

Every ViewerWindow ends with edit_history.mark_saved() before close(): a
resized document is dirty on purpose, and closeEvent would otherwise block
on a modal forever.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape.scenario_io import TEMPLATE_DIR, load_map_and_units

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _fake_dialog(monkeypatch, size: int | None, anchor_name: str = "TOP_LEFT"):
    """Replaces ResizeDialog.exec_ with one that sets the real dialog's own
    widgets and accepts (size None: reject). Returns the list of dialogs shown."""
    import descape.viewer as viewer_module
    from descape.resize_dialog import ResizeDialog
    from descape.scenario_resize import Anchor

    shown = []

    def exec_(self):
        shown.append(self)
        if size is None:
            return viewer_module.QDialog.Rejected
        self.width_spin.setValue(size)
        self.anchor_buttons[Anchor[anchor_name]].setChecked(True)
        return viewer_module.QDialog.Accepted

    monkeypatch.setattr(ResizeDialog, "exec_", exec_)
    return shown


def _questions(monkeypatch, answer):
    import descape.viewer as viewer_module

    asked = []

    def question(_parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return answer

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(question))
    return asked


@pytest.fixture
def window(tmp_path):
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    path = tmp_path / "units.aoe2scenario"
    shutil.copyfile(UNITS_FIXTURE, path)
    w = ViewerWindow()
    w.load_scenario(path)
    yield w
    w.edit_history.mark_saved()
    w.close()


def test_the_action_is_greyed_with_no_map() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    w = ViewerWindow()
    try:
        assert not w.resize_action.isEnabled()
        w.resize_map()  # a no-op, not a crash
        assert w.scenario is None
    finally:
        w.edit_history.mark_saved()
        w.close()


def test_a_destructive_resize_confirms_then_replaces_the_document(window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    assert window.resize_action.isEnabled()
    shown = _fake_dialog(monkeypatch, 100, "BOTTOM_RIGHT")
    asked = _questions(monkeypatch, QMessageBox.Yes)
    window.resize_action.trigger()

    assert len(shown) == 1
    assert [title for title, _ in asked] == ["Resize map"]
    assert "6 objects will be deleted" in asked[0][1]
    mm = window.scenario.map_manager
    assert (mm.map_width, mm.map_height) == (100, 100)
    assert sum(len(u) for u in window.scenario.unit_manager.units) == 2
    assert window.unit_edits is not None and window.unit_edits.loaded is window.scenario
    assert window.edit_history.is_dirty
    assert not window.undo_action.isEnabled()


def test_declining_the_destructive_confirm_leaves_the_document_untouched(window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    before = window.scenario
    _fake_dialog(monkeypatch, 100, "BOTTOM_RIGHT")
    asked = _questions(monkeypatch, QMessageBox.Cancel)
    window.resize_map()
    assert len(asked) == 1
    assert window.scenario is before
    assert window.scenario.map_manager.map_width == 120
    assert not window.edit_history.is_dirty


def test_cancelling_the_dialog_leaves_the_document_untouched(window, monkeypatch) -> None:
    before = window.scenario
    _fake_dialog(monkeypatch, None)
    asked = _questions(monkeypatch, None)
    window.resize_map()
    assert asked == []
    assert window.scenario is before


def test_a_grow_from_top_left_does_not_confirm_and_still_reads_dirty(window, monkeypatch) -> None:
    _fake_dialog(monkeypatch, 144, "TOP_LEFT")
    asked = _questions(monkeypatch, None)
    window.resize_map()
    assert asked == []
    assert window.scenario.map_manager.map_width == 144
    assert window.unit_edits is None  # nothing moved
    assert window.edit_history.is_dirty


def test_a_dirty_document_is_asked_about_exactly_once(window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window.edit_history.saved_at_cursor = None
    assert window.edit_history.is_dirty
    _fake_dialog(monkeypatch, 144, "TOP_LEFT")
    asked = _questions(monkeypatch, QMessageBox.Yes)
    window.resize_map()
    assert [title for title, _ in asked] == ["Unsaved changes"]
    assert "keeps them" in asked[0][1]
    assert window.scenario.map_manager.map_width == 144


def test_declining_the_unsaved_prompt_shows_no_dialog(window, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    window.edit_history.saved_at_cursor = None
    shown = _fake_dialog(monkeypatch, 144)
    _questions(monkeypatch, QMessageBox.Cancel)
    window.resize_map()
    assert shown == []
    assert window.scenario.map_manager.map_width == 120


def test_save_as_after_a_resize_writes_the_resized_map(window, monkeypatch, tmp_path) -> None:
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module

    _fake_dialog(monkeypatch, 109, "RIGHT")
    _questions(monkeypatch, QMessageBox.Yes)
    window.resize_map()
    dest = tmp_path / "resized.aoe2scenario"
    monkeypatch.setattr(viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(dest), "")))
    window.save_as()
    reloaded = load_map_and_units(dest)
    assert reloaded.map_manager.map_width == 109
    positions = {u.reference_id: (u.x, u.y) for us in reloaded.unit_manager.units for u in us}
    assert positions == {201: (0.5, 4.5), 300: (9.5, 14.5), 301: (10.5, 14.5)}
    assert not window.edit_history.is_dirty


def test_resizing_a_new_map_never_touches_the_template_dir(monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in TEMPLATE_DIR.iterdir()}
    _fake_dialog(monkeypatch, 168, "CENTER")
    _questions(monkeypatch, None)
    w = ViewerWindow()
    try:
        w.new_map()
        w.resize_map()
        assert w.scenario.map_manager.map_width == 168
        assert w._untitled is True
    finally:
        w.edit_history.mark_saved()
        w.close()
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in TEMPLATE_DIR.iterdir()} == before


def test_the_dialog_locks_square_and_refuses_the_current_size(window) -> None:
    from PyQt5.QtWidgets import QDialogButtonBox

    from descape.resize_dialog import ResizeDialog
    from descape.scenario_resize import Anchor

    dialog = ResizeDialog(window, 120, 120, window._resize_plan_for)
    try:
        ok = dialog.buttons.button(QDialogButtonBox.Ok)
        assert not dialog.square_box.isEnabled() and dialog.square_box.isChecked()
        assert not ok.isEnabled() and "already 120×120" in dialog.summary_label.text()
        dialog.width_spin.setValue(100)
        assert dialog.height_spin.value() == 100
        dialog.anchor_buttons[Anchor.BOTTOM_RIGHT].setChecked(True)
        assert ok.isEnabled()
        assert "6 objects will be deleted" in dialog.summary_label.text()
        assert "cannot be undone" in dialog.summary_label.text()
    finally:
        dialog.deleteLater()
