"""GH #55: every menu-bar action and the Elevation View combo's entries carry
a tooltip. The walk below is the guard that keeps new menu actions covered.

Every ViewerWindow() constructed here must call edit_history.mark_saved()
before close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _qt_default_tip(text: str) -> str:
    """What QAction.toolTip() returns when none was set: the text with "..." and
    mnemonic ampersands removed (Qt's qt_strippedText)."""
    return text.replace("...", "").replace("&", "").strip()


def _menu_items(menu, path: str):
    """(path, action) for every non-separator action under `menu`, submenus included."""
    for action in menu.actions():
        if action.isSeparator():
            continue
        where = f"{path} > {action.text()}"
        yield where, action
        if action.menu() is not None:
            yield from _menu_items(action.menu(), where)


def _missing_tooltips(window) -> list[str]:
    missing = []
    for top in window.menuBar().actions():
        menu = top.menu()
        if menu is None:
            continue
        if not menu.toolTipsVisible():
            missing.append(f"{top.text()}: setToolTipsVisible(True) missing")
        for where, action in _menu_items(menu, top.text()):
            tip = action.toolTip().strip()
            if not tip or tip == _qt_default_tip(action.text()):
                missing.append(f"{where}: no tooltip")
            sub = action.menu()
            if sub is not None and not sub.toolTipsVisible():
                missing.append(f"{where}: setToolTipsVisible(True) missing")
    return missing


def test_every_menu_action_has_a_tooltip_in_every_mode() -> None:
    """Some actions change text per mode (Copy/Paste), so each mode is walked."""
    window = conftest.blank_window()
    try:
        missing = set(_missing_tooltips(window))
        for i in range(window.mode_combo.count()):
            window.mode_combo.setCurrentIndex(i)
            missing.update(_missing_tooltips(window))
        assert not missing, "\n".join(sorted(missing))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_every_menu_action_has_a_tooltip_with_no_map_and_no_recent_files() -> None:
    from descape import settings

    settings.clear_recent_files()
    window = conftest.blank_window(load=False)
    try:
        assert window.recent_menu.actions()[0].text() == "(No recent files)"
        missing = _missing_tooltips(window)
        assert not missing, "\n".join(missing)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_guard_catches_a_new_action_without_a_tooltip() -> None:
    from PyQt5.QtWidgets import QAction

    window = conftest.blank_window()
    try:
        window.menuBar().actions()[0].menu().addAction(QAction("&Brand New…", window))
        assert any("Brand New" in line for line in _missing_tooltips(window))
    finally:
        window.edit_history.mark_saved()
        window.close()


def _mnemonic(text: str) -> str | None:
    """The letter after a single "&" ("&&" is a literal ampersand), lowercased."""
    stripped = text.replace("&&", "")
    i = stripped.find("&")
    return stripped[i + 1].lower() if 0 <= i < len(stripped) - 1 else None


def _menu_mnemonic_clashes(menu, path: str) -> list[str]:
    """Each letter two or more of one menu's direct entries share, submenus walked too."""
    clashes = []
    owners: dict[str, list[str]] = {}
    for action in menu.actions():
        if action.isSeparator():
            continue
        if (key := _mnemonic(action.text())) is not None:
            owners.setdefault(key, []).append(action.text())
        if action.menu() is not None:
            clashes += _menu_mnemonic_clashes(action.menu(), f"{path} > {action.text()}")
    clashes += [f"{path}: {key.upper()} in {texts}" for key, texts in owners.items() if len(texts) > 1]
    return clashes


def _mnemonic_clashes(window) -> list[str]:
    bar = window.menuBar()
    top_owners: dict[str, list[str]] = {}
    clashes = []
    for top in bar.actions():
        if (key := _mnemonic(top.text())) is not None:
            top_owners.setdefault(key, []).append(top.text())
        if top.menu() is not None:
            clashes += _menu_mnemonic_clashes(top.menu(), top.text())
    clashes += [f"menu bar: {key.upper()} in {texts}" for key, texts in top_owners.items() if len(texts) > 1]
    return clashes


def test_no_two_entries_of_one_menu_share_a_mnemonic_in_any_mode() -> None:
    """TASK-006: Alt+letter must pick one entry. Copy/Paste/Cut retitle per mode and focus, so each mode is walked."""
    window = conftest.blank_window()
    try:
        clashes = set(_mnemonic_clashes(window))
        for i in range(window.mode_combo.count()):
            window.mode_combo.setCurrentIndex(i)
            clashes.update(_mnemonic_clashes(window))
        assert not clashes, "\n".join(sorted(clashes))
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_mnemonic_guard_catches_a_clash() -> None:
    from PyQt5.QtWidgets import QAction

    window = conftest.blank_window()
    try:
        window.menuBar().actions()[0].menu().addAction(QAction("&Brand New…", window))
        window.menuBar().actions()[0].menu().addAction(QAction("&Bravo", window))
        assert any("Brand New" in line and "Bravo" in line for line in _mnemonic_clashes(window))
        assert _mnemonic("Fish && &Chips") == "c" and _mnemonic("No Mnemonic") is None
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_each_elevation_view_entry_explains_its_own_mode() -> None:
    from PyQt5.QtCore import Qt

    window = conftest.blank_window()
    try:
        combo = window.terrain_style_combo
        tips = {combo.itemText(i): combo.itemData(i, Qt.ToolTipRole) for i in range(combo.count())}
        assert set(tips) == {"Flat", "Stepped", "Sloped"}
        for label, tip in tips.items():
            assert tip, f"{label} has no tooltip"
        assert len(set(tips.values())) == 3
        assert "no elevation" in tips["Flat"].lower()
        assert "block" in tips["Stepped"].lower()
        assert "slope" in tips["Sloped"].lower()
    finally:
        window.edit_history.mark_saved()
        window.close()
