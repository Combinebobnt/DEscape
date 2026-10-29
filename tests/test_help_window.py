"""GH #160's help window, offscreen: the full Help menu, F1 per mode, link
handling, the live Keyboard and Mouse page and the topic search. Links go
through conftest's opened_urls fixture: no browser opens."""

from __future__ import annotations

import shutil

import pytest

from descape import __version__, help_guide, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_SEP = "---"


def _close(window) -> None:
    if window._help_window is not None:
        window._help_window.close()
    conftest.close_window(window)


def _row_text(help_window) -> str:
    return help_window.topic_list.currentItem().text()


def test_full_help_menu_order() -> None:
    from test_help_links import help_menu_texts

    window = conftest.blank_window(load=False)
    try:
        assert help_menu_texts(window) == [
            "DEscape Guide",
            "Keyboard and Mouse",
            _SEP,
            "AoE2 Scenario Editor Guide (ugc.aoe2.rocks)",
            _SEP,
            f"Release Notes for {__version__}",
            "Report a Bug…",
            "DEscape on GitHub",
            _SEP,
            "Debug Log",
            "Perf Trace",
            _SEP,
            "About",
        ]
        assert settings.get_default_keybind("help_guide") == "F1"
        assert settings.get_default_keybind("help_keys") == ""
    finally:
        _close(window)


def test_f1_opens_one_window_on_the_current_modes_topic() -> None:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeySequence
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape.help_window import HelpWindow

    window = conftest.blank_window()
    try:
        assert window.help_guide_action.shortcut() == QKeySequence("F1")
        window.mode_combo.setCurrentText("Units")
        window.show()
        window.activateWindow()
        QApplication.setActiveWindow(window)
        QApplication.processEvents()
        QTest.keyClick(window, Qt.Key_F1)
        QApplication.processEvents()
        help_window = window._help_window
        assert isinstance(help_window, HelpWindow)
        assert help_window.isVisible() and not help_window.isModal()
        assert help_window.current_topic_id() == "units"
        assert _row_text(help_window) == "Units"

        window.mode_combo.setCurrentText("Triggers")
        window.help_guide_action.trigger()
        assert window._help_window is help_window
        assert help_window.current_topic_id() == "triggers"
        assert _row_text(help_window) == "Triggers"

        window.mode_combo.setCurrentText("View")
        window.help_guide_action.trigger()
        assert help_window.current_topic_id() == "getting_started"
        window.help_keys_action.trigger()
        assert help_window.current_topic_id() == help_guide.KEYS_TOPIC_ID
    finally:
        _close(window)


def test_links_switch_topics_open_the_web_or_do_nothing(opened_urls, tmp_path) -> None:
    from PyQt5.QtCore import QUrl

    window = conftest.blank_window(load=False)
    try:
        window._show_help("getting_started")
        help_window = window._help_window
        browser = help_window.browser
        start = browser.source()

        browser.anchorClicked.emit(QUrl("units.md"))
        assert help_window.current_topic_id() == "units"
        assert _row_text(help_window) == "Units"
        assert help_window.back_button.isEnabled()
        help_window.back_button.click()
        assert browser.source() == start
        assert _row_text(help_window) == "Getting Started"
        assert help_window.forward_button.isEnabled()

        browser.anchorClicked.emit(QUrl("https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/"))
        assert opened_urls.urls == ["https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/"]
        assert browser.source() == start

        outside = tmp_path / "units.md"
        outside.write_text("# Elsewhere\n", encoding="utf-8")
        for url in (QUrl.fromLocalFile(str(outside)), QUrl("../help_guide.py"), QUrl("nope.md")):
            browser.anchorClicked.emit(url)
            assert browser.source() == start, url.toString()
            assert window.status_log.toPlainText().splitlines()[-1].startswith("The guide only opens its own pages")
        assert opened_urls.urls == ["https://ugc.aoe2.rocks/scenarios/triggers/effects/effects/"]
    finally:
        _close(window)


def _key_under(lines: list[str], heading: str, label: str) -> str:
    """The key cell after `label`'s cell, inside `heading`'s table (a table's plain text has one cell per line)."""
    start = lines.index(heading)
    return lines[lines.index(label, start) + 1]


def _native(sequence: str) -> str:
    """The page shows NativeText, so macOS reads Ctrl+Alt+P as ⌥⌘P."""
    from PyQt5.QtGui import QKeySequence

    return QKeySequence(sequence).toString(QKeySequence.NativeText)


def test_keys_page_shows_the_live_bindings_and_follows_a_rebind() -> None:
    window = conftest.blank_window(load=False)
    try:
        settings.set_keybind("edit_paste", "Ctrl+Alt+P")
        window.help_keys_action.trigger()
        browser = window._help_window.browser
        lines = browser.toPlainText().splitlines()
        assert lines[0] == "Keyboard and Mouse"
        assert _key_under(lines, "Edit", "Paste") == _native("Ctrl+Alt+P")
        assert _key_under(lines, "Help", "DEscape Guide") == _native("F1")
        # Markdown-special keys survive the escaping and Qt's renderer.
        assert _key_under(lines, "Tool Value", "Increase Tool Value") == "]"
        assert _key_under(lines, "Units", "Rotate Unit Anticlockwise (Quarter Turn)") == "<"

        settings.set_keybind("edit_paste", "Ctrl+Alt+O")
        window.help_keys_action.trigger()
        text = browser.toPlainText()
        assert _key_under(text.splitlines(), "Edit", "Paste") == _native("Ctrl+Alt+O")
        assert _native("Ctrl+Alt+P") not in text
    finally:
        _close(window)


def test_search_shows_only_matching_topics_and_finds_in_the_page(monkeypatch, tmp_path) -> None:
    from PyQt5.QtGui import QTextCursor

    from descape.help_window import HelpWindow

    for topic in help_guide.TOPICS:
        shutil.copyfile(help_guide.topic_path(topic), tmp_path / topic.filename)
    with (tmp_path / "diplomacy.md").open("a", encoding="utf-8") as f:
        f.write("\nThe ZzProbeWord appears here only.\n")
    monkeypatch.setattr(help_guide, "GUIDE_DIR", tmp_path)

    conftest.ensure_qapp()
    reports: list[str] = []
    help_window = HelpWindow(None, reports.append)
    try:
        help_window.select_topic("diplomacy")
        help_window.search_edit.setText("zzprobeword")
        visible = [
            help_window.topic_list.item(row).text()
            for row in range(help_window.topic_list.count())
            if not help_window.topic_list.item(row).isHidden()
        ]
        assert visible == ["Diplomacy"]

        help_window.browser.moveCursor(QTextCursor.End)
        help_window.search_edit.returnPressed.emit()
        assert help_window.browser.textCursor().selectedText() == "ZzProbeWord"

        help_window.search_edit.clear()
        assert not any(help_window.topic_list.item(r).isHidden() for r in range(help_window.topic_list.count()))
    finally:
        help_window.close()
