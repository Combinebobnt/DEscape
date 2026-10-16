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


@pytest.mark.no_qt_log
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_install_twice_is_a_no_op(installed):
    handler = qt_messages._handler
    qt_messages.install()
    assert qt_messages._handler is handler
