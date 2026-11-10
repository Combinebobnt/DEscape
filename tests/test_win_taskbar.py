"""Windows taskbar identity is platform-gated and cannot prevent startup."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from descape import win_taskbar


@pytest.mark.parametrize("platform", ["linux", "darwin", "freebsd14"])
def test_non_windows_never_calls_shell(platform):
    shell = Mock()
    assert win_taskbar.set_app_user_model_id(platform, shell) is None
    shell.SetCurrentProcessExplicitAppUserModelID.assert_not_called()


def test_windows_sets_stable_identity_once():
    shell = Mock()
    shell.SetCurrentProcessExplicitAppUserModelID.return_value = 0
    line = win_taskbar.set_app_user_model_id("win32", shell)
    shell.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with("DEscape.ScenarioEditor")
    assert line == "Windows taskbar AppUserModelID set: DEscape.ScenarioEditor"


def test_windows_defaults_to_real_platform_and_shell(monkeypatch):
    shell = Mock()
    shell.SetCurrentProcessExplicitAppUserModelID.return_value = 0
    monkeypatch.setattr(win_taskbar.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(shell32=shell), raising=False)
    assert "set:" in win_taskbar.set_app_user_model_id()
    shell.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with(win_taskbar.APP_USER_MODEL_ID)


@pytest.mark.parametrize("result", [1, -2147467259])
def test_shell_failure_is_logged_not_raised(result):
    shell = Mock()
    shell.SetCurrentProcessExplicitAppUserModelID.return_value = result
    assert win_taskbar.set_app_user_model_id("win32", shell) == (
        f"Windows taskbar AppUserModelID failed: HRESULT 0x{result & 0xFFFFFFFF:08X}"
    )


def test_shell_exception_is_logged_not_raised():
    shell = Mock()
    shell.SetCurrentProcessExplicitAppUserModelID.side_effect = OSError("shell unavailable")
    assert win_taskbar.set_app_user_model_id("win32", shell) == "Windows taskbar AppUserModelID failed: shell unavailable"


def test_shell_load_exception_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(), raising=False)
    assert "failed:" in win_taskbar.set_app_user_model_id("win32")


@pytest.mark.gui
@pytest.mark.parametrize("line", [None, "Windows taskbar AppUserModelID set: test", "Windows taskbar AppUserModelID failed: test"])
def test_viewer_sets_identity_and_logs_result_before_qapplication(monkeypatch, line):
    pytest.importorskip("PyQt5")
    from descape import viewer

    events = []
    monkeypatch.setattr(viewer, "install_crash_hooks", lambda: None)
    monkeypatch.setattr(viewer.asset_source, "migrate_legacy_config", lambda: None)
    monkeypatch.setattr(viewer.settings_file, "apply_pending_import", lambda: None)
    monkeypatch.setattr(viewer.settings, "get_xwayland_on_wayland", lambda: False)
    monkeypatch.setattr(viewer.qt_platform, "inherited_line", lambda: None)
    monkeypatch.setattr(viewer.qt_platform, "apply_default", lambda **kwargs: None)
    monkeypatch.setattr(viewer.qt_platform, "apply_default_theme", lambda: None)
    monkeypatch.setattr(viewer.qt_messages, "install", lambda: None)
    log = Mock()
    monkeypatch.setattr(viewer.debug_log, "log", log)

    def set_identity():
        events.append("identity")
        return line

    class StopStartup(Exception):
        pass

    def create_app(argv):
        events.append("QApplication")
        if line is not None:
            log.assert_called_with(line)
        raise StopStartup

    monkeypatch.setattr(win_taskbar, "set_app_user_model_id", set_identity)
    monkeypatch.setattr(viewer, "QApplication", create_app)
    with pytest.raises(StopStartup):
        viewer.main()
    assert events == ["identity", "QApplication"]
