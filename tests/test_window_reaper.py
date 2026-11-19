"""tests/conftest.py's window reaper (testkit/window_reaper.py): a window a
test leaves open is deleted before the next test, one a live wider fixture
holds survives, and deletion never runs closeEvent()."""

from __future__ import annotations

import pytest

from testkit import window_reaper

import conftest

# One worker, file order: each check reads what the test before it left.
pytestmark = [
    pytest.mark.gui,
    pytest.mark.xdist_group("window_reaper"),
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_held: dict = {}


def _window():
    from descape.viewer import ViewerWindow

    conftest.ensure_qapp()
    return ViewerWindow()


def _deleted(window) -> bool:
    from PyQt5 import sip

    return sip.isdeleted(window)


def test_a_test_leaves_a_window_open() -> None:
    _held["leaked"] = _window()


def test_the_left_window_is_deleted_before_the_next_test() -> None:
    assert "leaked" in _held, "needs the test above to run first"
    assert _deleted(_held.pop("leaked"))


@pytest.fixture(scope="class")
def kept():
    window = _window()
    _held["kept"] = window
    yield window


class TestAClassFixturesWindow:
    def test_it_is_built(self, kept) -> None:
        assert not _deleted(kept)

    def test_it_survives_a_sibling_that_never_requests_it(self) -> None:
        assert not _deleted(_held["kept"])

    def test_it_survives_into_a_later_test_that_requests_it(self, kept) -> None:
        assert not _deleted(kept)


def test_it_is_deleted_once_its_class_ends() -> None:
    assert "kept" in _held, "needs the class above to run first"
    assert _deleted(_held.pop("kept"))


def test_reaping_deletes_without_running_close_event(monkeypatch) -> None:
    from descape.viewer import ViewerWindow

    closed = []
    monkeypatch.setattr(ViewerWindow, "closeEvent", lambda self, event: closed.append(self))
    window = _window()
    assert window_reaper.reap(set()) >= 1
    assert _deleted(window)
    assert closed == []
