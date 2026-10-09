"""GH #126 Step 5: the Players panel's Personality combo and its window
wiring. Offscreen ViewerWindow, same technique as test_players_panel.py.
The AI choices come from a fake content-root tree (the suite hides the
real install), patched in where the window scans."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import ai_scripts
from descape.content_roots import KIND_INSTALL, KIND_MOD_SUBSCRIBED, ContentRoot

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
LONG_NAME = "An Extremely Long Subscribed Mod Artificial Intelligence Script Name v12.ai"


def _touch(path: Path, data: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def fake_roots(tmp_path, monkeypatch):
    install = ContentRoot(KIND_INSTALL, "Install", tmp_path / "install")
    mod = ContentRoot(KIND_MOD_SUBSCRIBED, "Mod (subscribed): 12_Pack", tmp_path / "mod", "0", "12_Pack")
    for root in (install, mod):
        ai = root.common / "ai"
        _touch(ai / "Same.ai")
        _touch(ai / "Same.per", root.label.encode())
    ai = install.common / "ai"
    _touch(ai / "E3-p2.ai")
    _touch(ai / "E3-p2.per", b'(load "Lib\\x")')
    _touch(ai / "Lib" / "x.per", b"(defrule)")
    _touch(ai / "Broken.ai")
    _touch(ai / "Broken.per", b'(load "missing")')
    _touch(mod.common / "ai" / LONG_NAME)
    _touch(mod.common / "ai" / LONG_NAME.replace(".ai", ".per"), b"x")
    roots = [install, mod]
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module.content_roots, "content_roots", lambda *a, **k: list(roots))
    return roots


def _players_window():
    window = conftest.shown_window(1500, 900)
    window.load_scenario(BLANK_FIXTURE)
    window.mode_combo.setCurrentText("Players")
    return window


def _combo(window):
    return window.players_panel.widget_for("personality")


def _labels(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _pick(window, key: str) -> None:
    from PyQt5.QtWidgets import QApplication

    combo = _combo(window)
    index = combo.findData(key)
    assert index >= 0, (key, _labels(combo))
    combo.setCurrentIndex(index)
    QApplication.processEvents()  # the window repopulates on the next event-loop turn


def _key(window, name: str) -> str:
    return next(c.key for c in window._ai_scan() if c.stored_name == name)


def test_a_blank_file_shows_standard_unset_then_every_choice(fake_roots) -> None:
    from PyQt5.QtWidgets import QComboBox

    window = _players_window()
    try:
        combo = _combo(window)
        assert isinstance(combo, QComboBox) and combo.isEnabled()
        labels = _labels(combo)
        assert labels[:3] == ["Standard (unset)", "Standard", "None"]
        assert combo.currentIndex() == 0
        # Same-named copies carry their source; unique names do not.
        assert "Same.ai (Install)" in labels and "Same.ai (Mod (subscribed): 12_Pack)" in labels
        assert "E3-p2.ai" in labels
        assert window.players_panel.widget_for("player_type").currentData() == 1
        assert not window.edit_history.can_undo
    finally:
        conftest.close_window(window)


def test_picking_none_then_undo_restores(fake_roots) -> None:
    window = _players_window()
    try:
        panel = window.players_panel
        panel.player_combo.setCurrentIndex(2)
        _pick(window, ai_scripts.KEY_NONE)
        assert panel.shown_personality() == ai_scripts.KEY_NONE
        assert panel.widget_for("player_type").currentData() == 2
        assert window.option_edits.current_personality(3).stored_name == "NoneAi"
        window.undo()
        assert not window.option_edits.has_edits
        assert panel.shown_personality() == "stored"
        assert panel.widget_for("player_type").currentData() == 1
        window.redo()
        assert panel.shown_personality() == ai_scripts.KEY_NONE
    finally:
        conftest.close_window(window)


def test_picking_a_custom_ai_resolves_it_and_labels_the_undo_step(fake_roots) -> None:
    window = _players_window()
    try:
        _pick(window, _key(window, "E3-p2.ai"))
        choice = window.option_edits.current_personality(1)
        assert choice.text == b'(load "Lib\\x")'
        assert [k for k, _ in choice.library] == [b"E3-p2.per\x00", b"Lib\\x.per\x00"]
        assert window.players_panel.widget_for("player_type").currentData() == 0
        assert window.edit_history.peek_undo().label == "Set P1 Personality: E3-p2.ai"
    finally:
        conftest.close_window(window)


def test_the_exact_same_named_copy_stays_selected(fake_roots) -> None:
    window = _players_window()
    try:
        mod_key = next(c.key for c in window._ai_scan() if c.stored_name == "Same.ai" and c.root.kind != KIND_INSTALL)
        _pick(window, mod_key)
        assert window.players_panel.shown_personality() == mod_key
        assert window.option_edits.current_personality(1).text == b"Mod (subscribed): 12_Pack"
    finally:
        conftest.close_window(window)


def test_an_unresolvable_ai_is_refused_and_its_row_disabled(fake_roots) -> None:
    from PyQt5.QtCore import Qt

    window = _players_window()
    try:
        key = _key(window, "Broken.ai")
        _pick(window, key)
        assert window.option_edits is None or not window.option_edits.has_edits
        combo = _combo(window)
        assert combo.currentData() == "stored"
        index = combo.findData(key)
        assert not combo.model().item(index).isEnabled()
        assert "missing include missing.per" in combo.itemData(index, Qt.ToolTipRole)
    finally:
        conftest.close_window(window)


def test_custom_rows_are_disabled_when_the_library_does_not_verify(fake_roots, monkeypatch) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "personality_custom_supported", lambda _loaded: False)
    window = _players_window()
    try:
        combo = _combo(window)
        for i in range(combo.count()):
            key = combo.itemData(i)
            custom = key not in ("stored", ai_scripts.KEY_STANDARD, ai_scripts.KEY_NONE)
            assert combo.model().item(i).isEnabled() is not custom, combo.itemText(i)
    finally:
        conftest.close_window(window)


def test_a_personality_gate_failure_leaves_every_other_row_editable(fake_roots, monkeypatch) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module, "personality_write_supported", lambda _loaded: False)
    window = _players_window()
    try:
        panel = window.players_panel
        assert not _combo(window).isEnabled()
        assert "personality block failed" in _combo(window).toolTip()
        assert panel.widget_for("food").isEnabled()
        assert "personality" not in window._editable_player_fields()
    finally:
        conftest.close_window(window)


def test_an_unmatched_stored_name_gets_a_stored_row(fake_roots, tmp_path) -> None:
    from descape.options_model import OptionsEditModel
    from descape.scenario_io import load_map_and_units
    from descape.scenario_write import write_scenario

    loaded = load_map_and_units(BLANK_FIXTURE)
    model = OptionsEditModel(loaded)
    model.set_personality(1, ai_scripts.AiChoice("t", "RandomGame", 1, "t", text=b"stub", resolved=True))
    out = tmp_path / "random_game.aoe2scenario"
    write_scenario(loaded, out, options=model, backup=False)

    window = conftest.shown_window(1500, 900)
    try:
        window.load_scenario(out)
        window.mode_combo.setCurrentText("Players")
        combo = _combo(window)
        assert combo.currentText() == "(stored) RandomGame"
        _pick(window, ai_scripts.KEY_STANDARD)
        _pick(window, "stored")  # restores the file's own bytes
        assert not window.option_edits.has_edits
    finally:
        conftest.close_window(window)


def test_an_install_path_change_rescans(fake_roots) -> None:
    window = _players_window()
    try:
        first = window._ai_scan()
        window.invalidate_ai_choices()
        assert window._ai_choices is not None and window._ai_choices is not first
    finally:
        conftest.close_window(window)


@pytest.mark.font_sensitive
def test_a_long_mod_name_fits_min_useful_width(fake_roots) -> None:
    from descape.players_panel import PlayersPanel

    window = _players_window()
    try:
        combo = _combo(window)
        assert any(LONG_NAME in label for label in _labels(combo))
        assert combo.minimumSizeHint().width() <= PlayersPanel.MIN_USEFUL_WIDTH
    finally:
        conftest.close_window(window)
