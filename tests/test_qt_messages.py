"""descape/qt_messages.py: the handler that drops Qt 5's Wayland
requestActivate() warning (GH #236) and prints every other message.
"""

from __future__ import annotations

import pytest

from descape import qt_messages

import conftest

WAYLAND_WARNING = "Wayland does not support QWindow::requestActivate()"


def test_the_wayland_warning_is_suppressed():
    assert qt_messages.is_suppressed(WAYLAND_WARNING)


@pytest.mark.parametrize(
    "message",
    [
        "",
        "Wayland does not support QWindow::requestActivate",
        WAYLAND_WARNING + " ",
        "qt.qpa.wayland: " + WAYLAND_WARNING,
        "Wayland does not support QWindow::setOpacity()",
    ],
)
def test_near_misses_pass_through(message):
    assert not qt_messages.is_suppressed(message)


@pytest.fixture
def installed():
    from PyQt5.QtCore import qInstallMessageHandler

    qt_messages.install()
    yield
    qInstallMessageHandler(None)
    qt_messages._handler = None


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_installed_handler_drops_only_the_wayland_warning(installed, capfd):
    from PyQt5.QtCore import qWarning

    qWarning(WAYLAND_WARNING.encode())
    qWarning(b"gh236 other warning")
    err = capfd.readouterr().err
    assert "requestActivate" not in err
    assert "gh236 other warning" in err


@pytest.fixture
def pattern(monkeypatch):
    from PyQt5.QtCore import qSetMessagePattern

    # QT_MESSAGE_PATTERN overrides qSetMessagePattern().
    monkeypatch.delenv("QT_MESSAGE_PATTERN", raising=False)
    yield qSetMessagePattern
    qSetMessagePattern("%{if-category}%{category}: %{endif}%{message}")


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_pass_through_keeps_the_message_pattern(installed, pattern, capfd):
    from PyQt5.QtCore import qWarning

    pattern("gh236[%{type}] %{message}")
    qWarning(b"gh236 other warning")
    assert capfd.readouterr().err == "gh236[warning] gh236 other warning\n"


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_message_the_pattern_filters_out_prints_nothing(installed, pattern, capfd):
    from PyQt5.QtCore import qWarning

    pattern("%{if-critical}%{message}%{endif}")
    qWarning(b"gh236 filtered warning")
    assert capfd.readouterr().err == ""


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("stream_kind", ["raising", "none"])
def test_a_broken_stderr_never_raises_out_of_the_handler(installed, monkeypatch, stream_kind):
    import sys

    from PyQt5.QtCore import qWarning

    class Raising:
        def write(self, _text):
            raise OSError(32, "Broken pipe")

        def flush(self):
            raise OSError(32, "Broken pipe")

    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: seen.append(exc))
    monkeypatch.setattr(sys, "stderr", Raising() if stream_kind == "raising" else None)
    qWarning(b"gh236 other warning")
    assert seen == []


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_install_twice_is_a_no_op(installed):
    handler = qt_messages._handler
    qt_messages.install()
    assert qt_messages._handler is handler


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_uninstall_hands_messages_back_to_qt(installed, capfd):
    """GH #236 review: Qt's default handler prints everything, the Wayland warning included."""
    from PyQt5.QtCore import qWarning

    qt_messages.uninstall()
    assert qt_messages._handler is None
    qWarning(WAYLAND_WARNING.encode())
    assert "requestActivate" in capfd.readouterr().err
    qt_messages.uninstall()
    qt_messages.install()
    qWarning(WAYLAND_WARNING.encode())
    assert "requestActivate" not in capfd.readouterr().err, "a later install filters again"


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_main_uninstalls_the_handler_once_the_event_loop_returns(monkeypatch):
    """GH #236 review: PyQt's handler wrapper must not outlive the event loop
    into interpreter shutdown."""
    import sys
    from types import SimpleNamespace

    from descape import (
        asset_source,
        catalog_thumbs,
        debug_log,
        qt_platform,
        settings,
        settings_file,
        sprite_prefetch,
        viewer,
        win_taskbar,
    )
    from testkit.viewer_family import set_viewer_global

    events = []

    class FakeApp:
        def __init__(self, argv):
            events.append("QApplication")

        @staticmethod
        def platformName():
            return "offscreen"

        def setDesktopFileName(self, _name):
            pass

        def setWindowIcon(self, _icon):
            pass

        def exec_(self):
            events.append("exec")
            return 0

    window = SimpleNamespace(show=lambda: None, load_scenario=lambda _path: None)
    set_viewer_global(monkeypatch, "install_crash_hooks", lambda: None)
    set_viewer_global(monkeypatch, "QApplication", FakeApp)
    set_viewer_global(monkeypatch, "apply_theme", lambda *args: None)
    set_viewer_global(monkeypatch, "apply_ui_font", lambda *args: None)
    set_viewer_global(monkeypatch, "_app_window", lambda: window)
    set_viewer_global(monkeypatch, "QTimer", SimpleNamespace(singleShot=lambda *args: None))
    monkeypatch.setattr(asset_source, "migrate_legacy_config", lambda: None)
    monkeypatch.setattr(settings_file, "apply_pending_import", lambda: None)
    monkeypatch.setattr(settings, "get_xwayland_on_wayland", lambda: False)
    monkeypatch.setattr(qt_platform, "inherited_line", lambda: None)
    monkeypatch.setattr(qt_platform, "apply_default", lambda **kwargs: None)
    monkeypatch.setattr(qt_platform, "apply_default_theme", lambda *args: None)
    monkeypatch.setattr(win_taskbar, "set_app_user_model_id", lambda: None)
    monkeypatch.setattr(debug_log, "log", lambda *args: None)
    monkeypatch.setattr(qt_messages, "install", lambda: events.append("install"))
    monkeypatch.setattr(qt_messages, "uninstall", lambda: events.append("uninstall"))
    monkeypatch.setattr(asset_source, "shutdown_prefetch_pool", lambda: events.append("pools"))
    monkeypatch.setattr(sprite_prefetch, "shutdown_pool", lambda: None)
    monkeypatch.setattr(catalog_thumbs, "shutdown_thumb_pool", lambda: None)
    monkeypatch.setattr(sys, "argv", ["descape"])
    monkeypatch.delenv("DESCAPE_READY_FILE", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        viewer.main()
    assert exit_info.value.code == 0
    assert events == ["install", "QApplication", "exec", "uninstall", "pools"]
