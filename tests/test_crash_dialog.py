"""CrashReportDialog, plus the two sites in descape/viewer.py that create
it: the in-app crash-time path (main()'s installed sys.excepthook) and the
next-launch sweep. @pytest.mark.gui but still default-tier -- pytest.ini's
addopts only deselects "corpus"/"slow", not "gui".

Never calls CrashReportDialog.exec_() for real: tests/conftest.py records
that a modal dialog blocks forever offscreen, which is exactly the trap a
QMessageBox falls into and the reason this dialog is a plain QDialog
instead. The two hook-level tests monkeypatch the dialog class itself so
the hook's own `.exec_()` call never actually blocks.
"""

from __future__ import annotations

import faulthandler
import sys
import threading

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QPushButton

import conftest
from descape import crash_report
from descape.viewer_dialogs import CrashReportDialog


@pytest.mark.gui
def test_live_dialog_shows_dump_text_and_action_buttons(tmp_path):
    conftest.ensure_qapp()
    dialog = CrashReportDialog(
        None,
        summary="RuntimeError: boom",
        dump_path=tmp_path / "crash-x.txt",
        dump_text="the dump body",
    )
    assert "the dump body" in dialog.text.toPlainText()
    labels = {btn.text() for btn in dialog.findChildren(QPushButton)}
    assert {"Copy to clipboard", "Open containing folder", "Save As...", "Continue", "Quit"} <= labels


@pytest.mark.gui
def test_from_last_session_dialog_has_only_close(tmp_path):
    conftest.ensure_qapp()
    dialog = CrashReportDialog(
        None,
        summary="RuntimeError: boom",
        dump_path=tmp_path / "crash-x.txt",
        dump_text="the dump body",
        from_last_session=True,
    )
    labels = {btn.text() for btn in dialog.findChildren(QPushButton)}
    assert "Close" in labels
    assert not {"Continue", "Quit", "Save As..."} & labels


class _FakeDialog:
    """Stand-in for CrashReportDialog that records what it was built with
    instead of ever calling the real exec_() -- see module docstring."""

    opened: list[tuple] = []

    def __init__(self, parent, *, summary, dump_path, dump_text, from_last_session=False):
        _FakeDialog.opened.append((parent, summary, dump_path, dump_text, from_last_session))

    def exec_(self) -> None:
        pass


@pytest.fixture
def _fake_dialog(monkeypatch):
    from descape import viewer as viewer_module

    _FakeDialog.opened = []
    monkeypatch.setattr(viewer_module, "CrashReportDialog", _FakeDialog)
    return _FakeDialog


@pytest.fixture
def _crash_hooks_installed(monkeypatch):
    """Installs the real hooks for one test, then restores everything --
    sys.excepthook/threading.excepthook are process-global, and
    faulthandler.enable() must not leak into unrelated tests."""
    from descape import viewer as viewer_module

    old_excepthook = sys.excepthook
    old_thread_excepthook = threading.excepthook
    viewer_module.install_crash_hooks()
    try:
        yield viewer_module
    finally:
        sys.excepthook = old_excepthook
        threading.excepthook = old_thread_excepthook
        faulthandler.disable()
        viewer_module._crash_host_ref = None
        viewer_module._crash_rate_limiter = crash_report.RateLimiter()


@pytest.mark.gui
def test_unhandled_exception_writes_dump_and_opens_live_dialog(_fake_dialog, _crash_hooks_installed):
    viewer_module = _crash_hooks_installed
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.show()
    QApplication.processEvents()
    try:

        def _boom():
            raise RuntimeError("deliberate test crash")

        QTimer.singleShot(0, _boom)
        for _ in range(20):
            QApplication.processEvents()

        dump_dir = viewer_module._crash_dump_dir()
        dumps = list(dump_dir.glob("crash-*.txt"))
        assert len(dumps) == 1
        assert "RuntimeError: deliberate test crash" in dumps[0].read_text()

        assert len(_fake_dialog.opened) == 1
        _, summary, dump_path, _dump_text, from_last_session = _fake_dialog.opened[0]
        assert summary == "RuntimeError: deliberate test crash"
        assert dump_path == dumps[0]
        assert from_last_session is False
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
def test_pending_reports_are_swept_on_next_launch_and_marked_reported(_fake_dialog, monkeypatch):
    from descape import viewer as viewer_module
    from descape.viewer import ViewerWindow

    dump_dir = viewer_module._crash_dump_dir()
    text = crash_report.build_report(
        RuntimeError, RuntimeError("old crash"), None, version="0.3", log_text="(empty)"
    )
    path = crash_report.write_report(text, dump_dir)

    window = ViewerWindow()
    window.show()
    QApplication.processEvents()
    try:
        viewer_module._sweep_pending_crash_reports()

        assert len(_fake_dialog.opened) == 1
        _, summary, dump_path, _dump_text, from_last_session = _fake_dialog.opened[0]
        assert summary == "RuntimeError: old crash"
        assert dump_path == path
        assert from_last_session is True

        assert not path.exists()
        assert path.with_name(path.name + ".reported").exists()
    finally:
        window.edit_history.mark_saved()
        window.close()
