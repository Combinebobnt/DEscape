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
