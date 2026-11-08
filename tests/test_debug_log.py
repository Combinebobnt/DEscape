"""Structured debug levels and the optional, plain-text-preserving dialog colours."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from PyQt5.QtGui import QPalette, QTextFormat
from PyQt5.QtWidgets import QApplication, QPushButton

from descape import debug_log, settings, themes, viewer_dialogs

import conftest


@pytest.fixture(autouse=True)
def _empty_log():
    debug_log.clear()
    yield
    debug_log.clear()


def test_levels_keep_info_text_unchanged_and_tag_problems():
    debug_log.log("Application started")
    debug_log.warning("fallback\ncontinued")
    debug_log.error("save failed")
    rows = debug_log.entries()
    assert [row.level for row in rows] == list(debug_log.LEVELS)
    assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2}\] Application started", rows[0].text())
    assert rows[1].text() == f"[{rows[1].timestamp}] warning: fallback\ncontinued"
    assert rows[2].text() == f"[{rows[2].timestamp}] error: save failed"
    assert debug_log.get_log_text() == "\n".join(row.text() for row in rows)


def test_unknown_level_does_not_append():
    with pytest.raises(ValueError, match="Unknown debug log level"):
        debug_log.log("typo", level="warn")
    assert debug_log.entries() == []


def test_ring_buffer_snapshot_and_clear():
    for index in range(1005):
        debug_log.log(str(index))
    snapshot = debug_log.entries()
    assert len(snapshot) == 1000
    assert (snapshot[0].message, snapshot[-1].message) == ("5", "1004")
    snapshot.clear()
    assert len(debug_log.entries()) == 1000
    debug_log.clear()
    assert debug_log.entries() == []
    assert debug_log.get_log_text() == "(empty)"


def test_text_uses_one_snapshot_when_a_worker_logs_during_formatting(monkeypatch):
    debug_log.log("original")
    original = debug_log.entries()[0]
    text = debug_log.Entry.text

    def append_while_formatting(entry):
        debug_log.warning("worker appended")
        return text(entry)

    monkeypatch.setattr(debug_log.Entry, "text", append_while_formatting)
    assert debug_log.get_log_text() == text(original)
    assert len(debug_log.entries()) == 2


def _untagged_failures(source: str) -> list[int]:
    failures = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args or not isinstance(node.func, ast.Attribute):
            continue
        is_log = (
            node.func.attr == "log" and isinstance(node.func.value, ast.Name) and node.func.value.id == "debug_log"
        )
        if not is_log and node.func.attr != "_log_status":
            continue
        if any(keyword.arg == "level" for keyword in node.keywords) or len(node.args) > 1:
            continue  # Explicit INFO is the escape hatch for a deliberate exception.
        message = node.args[0]
        parts = message.values if isinstance(message, ast.JoinedStr) else [message]
        text = " ".join(part.value for part in parts if isinstance(part, ast.Constant) and isinstance(part.value, str))
        if any(word in f" {text.lower()}" for word in (" failed", "raised", "unreadable")):
            failures.append(node.lineno)
    return failures


def test_failure_guard_rejects_untagged_literals_and_fstrings():
    assert _untagged_failures('debug_log.log("Export failed")') == [1]
    assert _untagged_failures('window._log_status(f"{operation} failed: {error}")') == [1]
    assert _untagged_failures('debug_log.log("Unreadable config")') == [1]
    assert _untagged_failures('debug_log.log("raised", level=debug_log.ERROR)') == []
    assert _untagged_failures('debug_log.log("failed op metric", level=debug_log.INFO)') == []
    assert _untagged_failures('debug_log.log(f"perf {operation} {failed_count}")') == []


def test_failure_log_calls_have_explicit_levels():
    root = Path(__file__).resolve().parent.parent / "descape"
    failures = [
        f"{path.name}:{line}"
        for path in sorted(root.glob("*.py"))
        for line in _untagged_failures(path.read_text(encoding="utf-8"))
    ]
    assert not failures, "Failure log calls need an explicit level: " + ", ".join(failures)


def _block_formats(dialog):
    block = dialog.text.document().firstBlock()
    formats = []
    while block.isValid():
        formats.append(block.begin().fragment().charFormat())
        block = block.next()
    return formats


@pytest.mark.gui
@pytest.mark.parametrize("preset", ["light", "dark", "high_contrast_light", "high_contrast_dark"])
def test_dialog_colors_are_legible_and_toggle_persists_without_changing_text(preset):
    assert preset in themes.PRESETS
    with conftest.restoring_app_theme() as app:
        settings.set_theme(preset)
        viewer_dialogs.apply_theme(app, preset)
        debug_log.warning("warning\ncontinuation")
        debug_log.error("<error & not HTML>")
        debug_log.log("normal info")
        dialog = viewer_dialogs.DebugLogDialog(None)
        try:
            assert dialog.text.isReadOnly()
            assert dialog.text.toPlainText() == debug_log.get_log_text()
            warning, continuation, error, info = _block_formats(dialog)
            base = dialog.text.palette().color(QPalette.Base).name()
            assert warning.foreground() == continuation.foreground()
            assert warning.foreground() != error.foreground()
            assert not info.hasProperty(QTextFormat.ForegroundBrush)
            for fmt in (warning, error):
                floor = 7 if preset in themes.HIGH_CONTRAST_PRESETS else 4.5
                assert themes.contrast_ratio(fmt.foreground().color().name(), base) >= floor
            dialog.text.selectAll()
            dialog.text.copy()
            assert QApplication.clipboard().text() == debug_log.get_log_text()
            dialog.color_by_level.setChecked(False)
            assert not settings.get_debug_log_colors()
            assert all(not fmt.hasProperty(QTextFormat.ForegroundBrush) for fmt in _block_formats(dialog))
            assert dialog.text.toPlainText() == debug_log.get_log_text()
            settings._debug_log_colors = None
            reopened = viewer_dialogs.DebugLogDialog(None)
            try:
                assert not reopened.color_by_level.isChecked()
                reopened.color_by_level.setChecked(True)
                assert _block_formats(reopened)[0].hasProperty(QTextFormat.ForegroundBrush)
            finally:
                reopened.close()
        finally:
            dialog.close()


@pytest.mark.gui
def test_refresh_clear_and_empty_keep_snapshot_behavior():
    conftest.ensure_qapp()
    dialog = viewer_dialogs.DebugLogDialog(None)
    try:
        assert dialog.text.toPlainText() == "(empty)"
        debug_log.error("later")
        assert dialog.text.toPlainText() == "(empty)"
        buttons = {button.text(): button for button in dialog.findChildren(QPushButton)}
        buttons["Refresh"].click()
        assert dialog.text.toPlainText() == debug_log.get_log_text()
        assert dialog.text.textCursor().atEnd()
        buttons["Clear"].click()
        assert dialog.text.toPlainText() == "(empty)"
        assert not _block_formats(dialog)[0].hasProperty(QTextFormat.ForegroundBrush)
        assert debug_log.entries() == []
    finally:
        dialog.close()


@pytest.mark.gui
def test_status_pane_forwards_level_without_changing_pane_text():
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    window = ViewerWindow()
    try:
        window._log_status("test failure", level=debug_log.ERROR)
        assert window.status_log.toPlainText().endswith("test failure")
        assert debug_log.entries()[-1].level == debug_log.ERROR
    finally:
        window.close()
