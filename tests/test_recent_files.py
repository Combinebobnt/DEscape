"""File > Open Recent: settings.py's get/add/clear_recent_files() directly,
plus the viewer wiring that records a real (non-untitled) load_scenario()
call and rebuilds the File menu's submenu from it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import descape.settings as settings
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH

import conftest


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


def _answer_save_dialog(monkeypatch, dest: Path) -> None:
    import descape.viewer as viewer_module

    monkeypatch.setattr(viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(dest), "")))


def _recent_menu_names(window) -> list[str]:
    return [a.text() for a in window.recent_menu.actions() if a.data()]


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_gh215_new_map_then_save_as_lists_the_file(tmp_path, monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    dest = tmp_path / "fresh.aoe2scenario"
    _answer_save_dialog(monkeypatch, dest)
    window = ViewerWindow()
    try:
        window.new_map()
        window.save_as()
        assert dest.is_file()
        assert settings.get_recent_files() == [str(dest)]
        assert _recent_menu_names(window) == [dest.name]
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_gh215_save_on_a_new_map_falls_through_to_save_as_and_lists_the_file(tmp_path, monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    dest = tmp_path / "first_save.aoe2scenario"
    _answer_save_dialog(monkeypatch, dest)
    window = ViewerWindow()
    try:
        window.new_map()
        window.save()
        assert settings.get_recent_files() == [str(dest)]
        assert _recent_menu_names(window) == [dest.name]
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_gh215_save_as_under_a_new_name_puts_it_first(tmp_path, monkeypatch) -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    source = _copy_fixture(tmp_path, "source.aoe2scenario")
    dest = tmp_path / "renamed.aoe2scenario"
    _answer_save_dialog(monkeypatch, dest)
    window = ViewerWindow()
    try:
        window.load_scenario(source)
        window.save_as()
        assert settings.get_recent_files() == [str(dest), str(source)]
        assert _recent_menu_names(window) == [dest.name, source.name]
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_gh215_a_blocked_save_as_lists_nothing(tmp_path, monkeypatch) -> None:
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    blocked = tmp_path / "mods" / "subscribed" / "x.aoe2scenario"
    blocked.parent.mkdir(parents=True)
    _answer_save_dialog(monkeypatch, blocked)
    monkeypatch.setattr(viewer_module.QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    window = ViewerWindow()
    try:
        window.new_map()
        window.save_as()
        assert not blocked.exists()
        assert settings.get_recent_files() == []
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_gh215_an_autosave_slot_never_enters_recent_files(tmp_path) -> None:
    from descape import autosave

    window = conftest.blank_window(load=False)
    try:
        window.new_map()
        tiles = window.scenario.map_manager.terrain
        new_terrain = 15 if tiles[0].terrain_id == 2 else 2
        window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", new_terrain))
        window._autosave_tick()
        assert len(list(autosave.autosave_dir().glob(f"*{autosave.AUTOSAVE_SUFFIX}"))) == 1
        assert settings.get_recent_files() == []
    finally:
        conftest.close_window(window)


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
