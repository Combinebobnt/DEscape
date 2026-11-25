"""Tools > Scripts in the window (descape/script_menu.py, ViewerWindow.run_code()).

The claims, in the order the risk runs:

1. **Consent before code.** A file runs only after "Run script?" was answered
   Yes for its exact bytes; an edit asks again, No runs nothing.
2. **One undo entry or none.** A run that paints tiles and moves a unit is one
   history record, backed out by one Undo; a failing run leaves the history,
   the dirty flag and the map as they were, and says why in a box.
3. **Nothing reaches the window mid-run.** A key pressed while the progress
   dialog pumps events does nothing, and Cancel rolls back.
4. **Refusals and the folder.** No scenario, a locked document, an open edit
   and `DESCAPE_NO_ADDONS` refuse before any prompt; the rows follow the folder.

Every ViewerWindow() here closes through conftest.close_window().
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from descape import debug_log, scripting, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
_ARCHER_P1 = 201
_PAINT_ID = 2

PAINT = f"""
for x in range(5):
    for y in range(5):
        editor.set_terrain(x, y, {_PAINT_ID})
archer = next(u for u in editor.units(1) if u.reference_id == {_ARCHER_P1})
editor.move_unit(archer, archer.x + 3, archer.y)
print("painted the patch")
"""

BOOM = f"""
editor.set_terrain(0, 0, {_PAINT_ID})
raise ValueError("boom")
"""


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    root = tmp_path / "scripts"
    root.mkdir()
    monkeypatch.setenv(scripting.ENV_SCRIPT_PATH, str(root))
    monkeypatch.delenv(settings.ENV_NO_ADDONS, raising=False)
    return root


class _Prompts(list):
    def __init__(self):
        super().__init__()
        self.answer = {"yes": True}


@pytest.fixture
def prompts(monkeypatch):
    """Records each "Run script?" box and answers Yes unless told otherwise."""
    from descape import script_menu

    asked = _Prompts()

    def confirm(_window, path):
        asked.append(path)
        return asked.answer["yes"]

    monkeypatch.setattr(script_menu, "confirm_run", confirm)
    return asked


@pytest.fixture
def errors(monkeypatch):
    from descape import script_menu

    shown: list[tuple[str, str, str]] = []
    monkeypatch.setattr(script_menu, "show_error", lambda _w, title, text, details: shown.append((title, text, details)))
    return shown


def _write(root: Path, name: str, source: str) -> Path:
    path = root / f"{name}.py"
    path.write_text(textwrap.dedent(source))
    return path


def _window(load: bool = True, shown: bool = False):
    # Shown only where a shortcut must fire: Qt delivers none to a hidden window.
    window = conftest.shown_window() if shown else conftest.blank_window(load=False)
    if load:
        window.load_scenario(FIXTURE_PATH)
        assert window.scenario is not None
    return window


def _rows(window):
    """The submenu's script rows (the placeholder included) after it opens: everything above its separator."""
    window.scripts_menu.aboutToShow.emit()
    rows = []
    for action in window.scripts_menu.actions():
        if action.isSeparator() or action is window.open_scripts_folder_action:
            break
        rows.append(action)
    return rows


def _refreshes(window, monkeypatch) -> list[tuple[frozenset, set]]:
    """Records each _refresh_after_history_move()'s kinds and indices, then lets it run."""
    seen: list[tuple[frozenset, set]] = []
    original = window._refresh_after_history_move

    def spy(kinds, dirty, *args, **kwargs):
        seen.append((frozenset(kinds), set(dirty)))
        return original(kinds, dirty, *args, **kwargs)

    monkeypatch.setattr(window, "_refresh_after_history_move", spy)
    return seen


def _press_cancel(dialog) -> None:
    from PyQt5.QtWidgets import QPushButton

    [button] = dialog.findChildren(QPushButton)
    button.click()


def _row(window, title: str):
    [row] = [action for action in _rows(window) if action.text() == title]
    return row


def _map_state(window):
    tiles = [(t.terrain_id, t.elevation, t.layer) for t in window.scenario.map_manager.terrain]
    archer = next(u for u in window.scenario.unit_manager.units[1] if u.reference_id == _ARCHER_P1)
    return tiles, (archer.x, archer.y)


def test_rows_follow_the_folder(scripts):
    window = _window(load=False)
    try:
        [placeholder] = _rows(window)
        assert not placeholder.isEnabled() and placeholder.toolTip()
        for name in ("b_second", "a_first", "_private", ".hidden"):
            _write(scripts, name, "pass\n")
        (scripts / "notes.txt").write_text("not a script")
        assert [row.text() for row in _rows(window)] == ["a_first", "b_second"]
        assert all(row.isEnabled() and row.toolTip().endswith(".py") for row in _rows(window))
        actions = window.scripts_menu.actions()
        assert actions[-1] is window.open_scripts_folder_action and actions[-2].isSeparator()
        (scripts / "a_first.py").unlink()
        assert [row.text() for row in _rows(window)] == ["b_second"]
        assert len(window.scripts_menu.actions()) == 3
    finally:
        conftest.close_window(window)


def test_no_addons_greys_out_every_row_and_refuses(scripts, prompts, monkeypatch):
    _write(scripts, "paint", PAINT)
    monkeypatch.setenv(settings.ENV_NO_ADDONS, "1")
    window = _window()
    try:
        row = _row(window, "paint")
        assert not row.isEnabled()
        assert settings.ENV_NO_ADDONS in row.toolTip()
        before = _map_state(window)
        from descape import script_menu

        assert script_menu.run_entry(window, scripting.discover_scripts(scripts)[0]) is None
        assert prompts == [] and window.edit_history.records == [] and _map_state(window) == before
    finally:
        conftest.close_window(window)


def test_open_scripts_folder_creates_it_then_opens_it(tmp_path, monkeypatch):
    from descape import script_menu

    root = tmp_path / "not" / "yet"
    monkeypatch.setenv(scripting.ENV_SCRIPT_PATH, str(root))
    opened: list[Path] = []
    monkeypatch.setattr(script_menu, "open_folder", lambda path: opened.append(path) or True)
    window = _window(load=False)
    try:
        window.open_scripts_folder_action.trigger()
        assert root.is_dir() and opened == [root]
        assert str(root) in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


def test_a_run_is_one_undo_entry_and_one_undo_backs_it_out(scripts, prompts, errors, monkeypatch):
    _write(scripts, "paint", PAINT)
    window = _window()
    try:
        before = _map_state(window)
        assert any(tile[0] != _PAINT_ID for tile in before[0][:5])
        refreshes = _refreshes(window, monkeypatch)
        _row(window, "paint").trigger()
        width = window.scenario.map_manager.map_width
        assert refreshes == [(frozenset({"tile", "unit"}), {y * width + x for x in range(5) for y in range(5)})]
        tiles, archer = _map_state(window)
        mm = window.scenario.map_manager
        assert all(tiles[y * mm.map_width + x][0] == _PAINT_ID for x in range(5) for y in range(5))
        assert archer == (before[1][0] + 3, before[1][1])
        [record] = window.edit_history.records
        assert record.label == 'Script "paint"' and record.kinds() == {"tile", "unit"}
        assert window.edit_history.is_dirty
        assert 'Script "paint" changed terrain and units' in window.status_log.toPlainText()
        assert "painted the patch" in debug_log.get_log_text()
        assert errors == []

        window.undo()
        assert _map_state(window) == before
        assert 'Undo: Script "paint"' in window.status_log.toPlainText()
        assert not window.edit_history.is_dirty
        window.redo()
        assert _map_state(window) == (tiles, archer)
    finally:
        conftest.close_window(window)


def test_consent_is_asked_once_per_digest_and_no_runs_nothing(scripts, prompts, errors):
    path = _write(scripts, "paint", PAINT)
    window = _window()
    try:
        _row(window, "paint").trigger()
        _row(window, "paint").trigger()
        assert prompts == [path]
        assert len(window.edit_history.records) == 2

        path.write_text(textwrap.dedent(PAINT) + "\n# edited\n")
        prompts.answer["yes"] = False
        before = _map_state(window)
        _row(window, "paint").trigger()
        assert prompts == [path, path]
        assert _map_state(window) == before and len(window.edit_history.records) == 2
        # A declined digest is not stored: the next run asks again.
        prompts.answer["yes"] = True
        _row(window, "paint").trigger()
        assert len(prompts) == 3 and len(window.edit_history.records) == 3
        assert (str(path.resolve()), scripting.content_digest(path)) in settings.get_confirmed_scripts()
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("why", ["no_scenario", "locked", "stroke"])
def test_refusals_come_before_the_prompt(scripts, prompts, errors, why):
    _write(scripts, "paint", PAINT)
    window = _window(load=why != "no_scenario")
    try:
        if why == "locked":
            window.game_data_lock_reason = lambda: "shown with another game's data"
        if why == "stroke":
            window.edit_history.begin_stroke(window.scenario.map_manager.terrain)
        _row(window, "paint").trigger()
        assert prompts == [] and errors == []
        assert window.edit_history.records == []
        assert 'Script "paint" refused' in window.status_log.toPlainText()
        if why == "stroke":
            window.edit_history.abort_stroke()
    finally:
        conftest.close_window(window)


def test_a_failing_run_changes_nothing_and_shows_why(scripts, prompts, errors, monkeypatch):
    _write(scripts, "boom", BOOM)
    window = _window()
    try:
        before = _map_state(window)
        refreshes = _refreshes(window, monkeypatch)
        _row(window, "boom").trigger()
        # The rolled-back tile is repainted: the canvas may have drawn it mid-run.
        assert refreshes == [(frozenset({"tile"}), {0})]
        assert _map_state(window) == before
        assert window.edit_history.records == [] and not window.edit_history.is_dirty
        assert not window.edit_history.in_stroke
        [(title, text, details)] = errors
        assert title == 'Script "boom" failed'
        assert "nothing changed" in text and "ValueError: boom" in text
        assert "Traceback" in details and "boom.py" in details
        assert "ValueError: boom" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


def _undo_key_click(window) -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    sequence = window.undo_action.shortcut()
    assert not sequence.isEmpty()
    combo = sequence[0]
    window.activateWindow()
    QApplication.setActiveWindow(window)
    QApplication.processEvents()
    QTest.keyClick(
        window, combo & ~Qt.KeyboardModifierMask, Qt.KeyboardModifiers(combo & Qt.KeyboardModifierMask)
    )
    QApplication.processEvents()


def test_a_key_pressed_during_a_run_does_nothing_and_cancel_rolls_back(scripts, prompts, errors, monkeypatch):
    from descape import script_menu

    _write(scripts, "paint", PAINT)
    _write(
        scripts,
        "slow",
        f"""
        editor.set_terrain(10, 10, {_PAINT_ID})
        editor.progress(1, 3, "first")
        editor.progress(2, 3, "second")
        editor.set_terrain(11, 10, {_PAINT_ID})
        """,
    )
    window = _window(shown=True)
    try:
        _row(window, "paint").trigger()
        painted = _map_state(window)
        calls: list[str] = []
        original = script_menu.Progress.__call__

        def pressing(self, done, total, note):
            original(self, done, total, note)
            calls.append(note)
            _undo_key_click(window)
            if mode["cancel"]:
                _press_cancel(self.dialog)

        mode = {"cancel": False}
        monkeypatch.setattr(script_menu.Progress, "__call__", pressing)
        _row(window, "slow").trigger()
        assert calls == ["first", "second"]
        assert [r.label for r in window.edit_history.records] == ['Script "paint"', 'Script "slow"']
        assert not window.edit_history.can_redo

        # Cancel at the first call: the second raises Cancelled, and the run rolls back whole.
        window.undo()
        assert _map_state(window) == painted
        calls.clear()
        mode["cancel"] = True
        monkeypatch.setattr(script_menu.Progress, "__call__", pressing)
        _row(window, "slow").trigger()
        assert calls == ["first"]
        assert _map_state(window) == painted and errors == []
        assert 'Script "slow" was cancelled, nothing changed' in window.status_log.toPlainText()

        # A rolled-back run pushes nothing, so the undone "slow" is still there to redo.
        assert window.edit_history.can_undo and window.edit_history.can_redo
        # The control: the same key click outside a run does undo.
        _undo_key_click(window)
        assert not window.edit_history.can_undo
    finally:
        conftest.close_window(window)


def test_cancel_and_a_key_in_one_pump_still_reach_nothing(scripts, prompts, errors, monkeypatch):
    """Cancel keeps the dialog shown, so its modal block holds for a key queued
    behind the click in the same event pump (hiding it would let Undo run mid-run)."""
    from PyQt5.QtCore import QTimer

    from descape import script_menu

    _write(scripts, "paint", PAINT)
    _write(
        scripts,
        "slow",
        f"""
        editor.set_terrain(10, 10, {_PAINT_ID})
        editor.progress(1, 3, "first")
        editor.progress(2, 3, "second")
        """,
    )
    window = _window(shown=True)
    try:
        _row(window, "paint").trigger()
        painted = _map_state(window)
        original = script_menu.Progress.__call__
        undo_while_busy: list[bool] = []
        window.undo_action.triggered.connect(lambda: undo_while_busy.append(window._busy))

        def queued(self, done, total, note):
            if self.dialog is None:
                QTimer.singleShot(0, lambda: _press_cancel(self.dialog))
                QTimer.singleShot(0, lambda: _undo_key_click(window))
            original(self, done, total, note)

        monkeypatch.setattr(script_menu.Progress, "__call__", queued)
        _row(window, "slow").trigger()
        assert True not in undo_while_busy
        assert _map_state(window) == painted
        assert [r.label for r in window.edit_history.records] == ['Script "paint"']
        assert window.edit_history.can_undo and not window.edit_history.can_redo
        assert 'Script "slow" was cancelled, nothing changed' in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)
