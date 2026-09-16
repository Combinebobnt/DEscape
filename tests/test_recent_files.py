"""File > Open Recent: settings.py's get/add/clear_recent_files() directly,
plus the viewer wiring that records a real (non-untitled) load_scenario()
call and rebuilds the File menu's submenu from it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import conftest
import descape.settings as settings
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copyfile(FIXTURE_PATH, target)
    return target


def test_get_recent_files_starts_empty() -> None:
    assert settings.get_recent_files() == []


def test_add_recent_file_puts_newest_first(tmp_path) -> None:
    a, b = tmp_path / "a.aoe2scenario", tmp_path / "b.aoe2scenario"
    settings.add_recent_file(a)
    settings.add_recent_file(b)
    assert settings.get_recent_files() == [str(b), str(a)]


def test_add_recent_file_moves_a_repeat_to_front_without_duplicating(tmp_path) -> None:
    a, b = tmp_path / "a.aoe2scenario", tmp_path / "b.aoe2scenario"
    settings.add_recent_file(a)
    settings.add_recent_file(b)
    settings.add_recent_file(a)
    assert settings.get_recent_files() == [str(a), str(b)]


def test_add_recent_file_caps_at_max(tmp_path) -> None:
    for i in range(settings.MAX_RECENT_FILES + 5):
        settings.add_recent_file(tmp_path / f"{i}.aoe2scenario")
    files = settings.get_recent_files()
    assert len(files) == settings.MAX_RECENT_FILES
    # Newest survive, oldest are evicted.
    assert files[0] == str(tmp_path / f"{settings.MAX_RECENT_FILES + 4}.aoe2scenario")


def test_clear_recent_files(tmp_path) -> None:
    settings.add_recent_file(tmp_path / "a.aoe2scenario")
    settings.clear_recent_files()
    assert settings.get_recent_files() == []


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_opening_a_file_adds_it_to_recent_and_rebuilds_the_menu(tmp_path) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    target = _copy_fixture(tmp_path, "map.aoe2scenario")
    window = ViewerWindow()
    try:
        window.load_scenario(target)
        assert settings.get_recent_files() == [str(target)]
        # data() is only ever set on a real recent-file entry, unlike the
        # disabled placeholder or the trailing "Clear Recent Files" action.
        assert [a.text() for a in window.recent_menu.actions() if a.data()] == [target.name]
    finally:
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_does_not_touch_recent_files(tmp_path) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_map()
        assert settings.get_recent_files() == []
    finally:
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_recent_menu_skips_a_file_that_no_longer_exists(tmp_path) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    kept = _copy_fixture(tmp_path, "kept.aoe2scenario")
    gone = _copy_fixture(tmp_path, "gone.aoe2scenario")
    window = ViewerWindow()
    try:
        window.load_scenario(gone)
        window.load_scenario(kept)
        gone.unlink()
        window._rebuild_recent_files_menu()
        # Still persisted -- only the menu filters it, so it reappears if the
        # file (e.g. on a removable drive) comes back.
        assert str(gone) in settings.get_recent_files()
        assert [a.text() for a in window.recent_menu.actions() if a.data()] == [kept.name]
    finally:
        window.close()
