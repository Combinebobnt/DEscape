"""GH #221: dropping a .aoe2scenario file on the window opens it. Drag
events are built by hand and sent through QApplication, so the same
propagation a real drag gets (child to parent) runs with no drag source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _copy_template(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copyfile(BLANK_TEMPLATE_PATH, target)
    return target


def _mime(paths=(), urls=(), text: str | None = None):
    from PyQt5.QtCore import QMimeData, QUrl

    mime = QMimeData()
    all_urls = [QUrl.fromLocalFile(str(p)) for p in paths] + [QUrl(u) for u in urls]
    if all_urls:
        mime.setUrls(all_urls)
    if text is not None:
        mime.setText(text)
    return mime


def _enter(widget, mime) -> bool:
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtGui import QDragEnterEvent
    from PyQt5.QtWidgets import QApplication

    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def _send_enter(widget, mime) -> bool:
    """A drag that enters and leaves: leaving clears Qt's drop target, so no
    later test's drop can land on this window."""
    from PyQt5.QtGui import QDragLeaveEvent
    from PyQt5.QtWidgets import QApplication

    accepted = _enter(widget, mime)
    QApplication.sendEvent(widget, QDragLeaveEvent())
    return accepted


def _send_drop(widget, mime) -> bool:
    """Qt hands a Drop only to the widget that accepted the drag's enter."""
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtGui import QDropEvent
    from PyQt5.QtWidgets import QApplication

    assert _enter(widget, mime)
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    # The open itself is deferred to the next event loop pass.
    QApplication.processEvents()
    return event.isAccepted()


def test_dropped_scenario_paths_keeps_local_scenarios_in_order(tmp_path) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtCore import QUrl

    from descape.viewer import dropped_scenario_paths

    b = tmp_path / "b.aoe2scenario"
    a = tmp_path / "A.AOE2SCENARIO"
    urls = [
        QUrl.fromLocalFile(str(b)),
        QUrl.fromLocalFile(str(tmp_path / "old.scx")),
        QUrl.fromLocalFile(str(tmp_path / "notes.txt")),
        QUrl("https://example.com/map.aoe2scenario"),
        QUrl.fromLocalFile(str(a)),
    ]
    assert dropped_scenario_paths(urls) == [b, a]
    assert dropped_scenario_paths([]) == []


def test_drag_enter_accepts_only_a_scenario_file(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    try:
        assert window.acceptDrops()
        assert _send_enter(window, _mime(paths=[tmp_path / "x.aoe2scenario"]))
        assert not _send_enter(window, _mime(paths=[tmp_path / "x.txt"]))
        assert not _send_enter(window, _mime(text="x.aoe2scenario"))
        assert not _send_enter(window, _mime(urls=["https://example.com/x.aoe2scenario"]))
    finally:
        conftest.close_window(window)


def test_drag_enter_is_refused_while_busy(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    try:
        window._busy = True
        assert not _send_enter(window, _mime(paths=[tmp_path / "x.aoe2scenario"]))
    finally:
        window._busy = False
        conftest.close_window(window)


def test_map_view_passes_drops_up_to_the_window(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    try:
        assert not window.map_view.acceptDrops()
        assert not window.map_view.viewport().acceptDrops()
        mime = _mime(paths=[tmp_path / "x.aoe2scenario"])
        assert _send_enter(window.map_view.viewport(), mime)
        # The read-only log refuses file drags itself, so they reach the window too.
        assert _send_enter(window.status_log.viewport(), mime)
    finally:
        conftest.close_window(window)


def test_drop_opens_the_scenario_after_the_drop_returns(tmp_path, monkeypatch) -> None:
    window = conftest.blank_window(load=False)
    target = tmp_path / "x.aoe2scenario"
    loaded: list[Path] = []
    monkeypatch.setattr(window, "load_scenario", loaded.append)
    try:
        assert _send_drop(window, _mime(paths=[tmp_path / "x.txt", target]))
        assert loaded == [target]
    finally:
        conftest.close_window(window)


def test_the_open_waits_until_the_drop_has_returned(tmp_path, monkeypatch) -> None:
    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtGui import QDropEvent
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window(load=False)
    target = tmp_path / "x.aoe2scenario"
    loaded: list[Path] = []
    monkeypatch.setattr(window, "load_scenario", loaded.append)
    try:
        mime = _mime(paths=[target])
        assert _enter(window, mime)
        QApplication.sendEvent(window, QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        assert loaded == [], "a discard prompt inside dropEvent would stall the drag source"
        QApplication.processEvents()
        assert loaded == [target]
    finally:
        conftest.close_window(window)


def test_a_shift_drag_is_answered_as_a_copy_never_a_move(tmp_path, monkeypatch) -> None:
    """Explorer deletes the source file after a Move result; opening it consumes nothing."""
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QDragEnterEvent, QDropEvent
    from PyQt5.QtWidgets import QApplication

    window = conftest.blank_window(load=False)
    monkeypatch.setattr(window, "load_scenario", lambda path: None)
    try:
        mime = _mime(paths=[tmp_path / "x.aoe2scenario"])
        actions = Qt.CopyAction | Qt.MoveAction
        enter = QDragEnterEvent(QPoint(5, 5), actions, mime, Qt.LeftButton, Qt.ShiftModifier)
        assert enter.proposedAction() == Qt.MoveAction
        window.dragEnterEvent(enter)
        assert enter.isAccepted() and enter.dropAction() == Qt.CopyAction
        drop = QDropEvent(QPointF(5, 5), actions, mime, Qt.LeftButton, Qt.ShiftModifier)
        window.dropEvent(drop)
        QApplication.processEvents()
        assert drop.isAccepted() and drop.dropAction() == Qt.CopyAction
        move_only = QDragEnterEvent(QPoint(5, 5), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dragEnterEvent(move_only)
        assert not move_only.isAccepted()
    finally:
        conftest.close_window(window)


def test_a_drop_on_the_map_opens_the_file(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    target = _copy_template(tmp_path, "on_map.aoe2scenario")
    try:
        _send_drop(window.map_view.viewport(), _mime(paths=[target]))
        assert window.scenario is not None
        assert window.scenario.path == target
    finally:
        conftest.close_window(window)


def test_drop_of_a_non_scenario_is_refused(tmp_path, monkeypatch) -> None:
    window = conftest.blank_window(load=False)
    loaded: list[Path] = []
    monkeypatch.setattr(window, "load_scenario", loaded.append)
    try:
        from PyQt5.QtCore import QPointF, Qt
        from PyQt5.QtGui import QDropEvent
        from PyQt5.QtWidgets import QApplication

        mime = _mime(paths=[tmp_path / "x.txt"])
        event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        window.dropEvent(event)
        QApplication.processEvents()
        assert not event.isAccepted()
        assert loaded == []
    finally:
        conftest.close_window(window)


def test_drop_on_a_dirty_document_honours_cancel(tmp_path, monkeypatch) -> None:
    window = conftest.blank_window(load=False)
    loaded: list[Path] = []
    monkeypatch.setattr(window, "load_scenario", loaded.append)
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda: False)
    try:
        _send_drop(window, _mime(paths=[tmp_path / "x.aoe2scenario"]))
        assert loaded == []
    finally:
        conftest.close_window(window)


def test_drop_really_loads_the_file(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    target = _copy_template(tmp_path, "dropped.aoe2scenario")
    try:
        _send_drop(window, _mime(paths=[target]))
        assert window.scenario is not None
        assert window.scenario.path == target
    finally:
        conftest.close_window(window)


def test_dropping_several_opens_the_first_and_reports_the_rest(tmp_path) -> None:
    window = conftest.blank_window(load=False)
    first = _copy_template(tmp_path, "first.aoe2scenario")
    second = _copy_template(tmp_path, "second.aoe2scenario")
    try:
        _send_drop(window, _mime(paths=[first, second]))
        assert window.scenario.path == first
        message = window.statusBar().currentMessage()
        assert "first.aoe2scenario" in message
        assert "skipped 1 other" in message
        assert "skipped 1 other" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)
