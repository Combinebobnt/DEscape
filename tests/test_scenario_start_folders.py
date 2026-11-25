"""Where File > Open and Save As start (TASK-123): the game's scenario folder
when there is no better place, and Save on a subscribed-mod scenario asking
for a new name instead of refusing."""

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


def _profile(tmp_path: Path, monkeypatch) -> Path:
    profile = tmp_path / "profile"
    folder = profile / "7656" / "resources" / "_common" / "scenario"
    folder.mkdir(parents=True)
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    return folder


def _capture(monkeypatch, name: str, answer: str = "") -> list[str]:
    from PyQt5.QtWidgets import QFileDialog

    seen: list[str] = []

    def fake(_parent, _caption, start, *_a, **_k):
        seen.append(start)
        return answer, ""

    monkeypatch.setattr(QFileDialog, name, staticmethod(fake))
    return seen


@pytest.fixture
def window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    win = ViewerWindow()
    yield win
    win.edit_history.mark_saved()
    win.close()


def test_open_with_nothing_open_starts_in_the_profile_folder(tmp_path, monkeypatch, window):
    folder = _profile(tmp_path, monkeypatch)
    seen = _capture(monkeypatch, "getOpenFileName")
    window.open_file()
    assert seen == [str(folder)]


def test_open_with_a_file_open_starts_in_its_folder(tmp_path, monkeypatch, window):
    _profile(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "map.aoe2scenario"
    shutil.copy(BLANK_TEMPLATE_PATH, target)
    window.load_scenario(target)
    seen = _capture(monkeypatch, "getOpenFileName")
    window.open_file()
    assert seen == [str(elsewhere)]


def test_open_with_no_profile_keeps_qt_default(tmp_path, monkeypatch, window):
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(tmp_path / "missing"))  # explicit and absent: no profile
    seen = _capture(monkeypatch, "getOpenFileName")
    window.open_file()
    assert seen == [""]


def test_untitled_save_as_starts_in_the_profile_folder(tmp_path, monkeypatch, window):
    from descape.viewer_file import UNTITLED_NAME

    folder = _profile(tmp_path, monkeypatch)
    window.new_map()
    seen = _capture(monkeypatch, "getSaveFileName")
    window.save_as()
    assert seen == [str(folder / UNTITLED_NAME)]


def _subscribed_copy(tmp_path: Path) -> Path:
    mod = tmp_path / "profile" / "7656" / "mods" / "subscribed" / "1_Mod" / "resources" / "_common" / "scenario"
    mod.mkdir(parents=True)
    target = mod / "workshop.aoe2scenario"
    shutil.copy(BLANK_TEMPLATE_PATH, target)
    return target


def test_save_as_of_a_subscribed_file_starts_in_the_profile_folder(tmp_path, monkeypatch, window):
    folder = _profile(tmp_path, monkeypatch)
    target = _subscribed_copy(tmp_path)
    window.load_scenario(target)
    seen = _capture(monkeypatch, "getSaveFileName")
    window.save_as()
    assert seen == [str(folder / target.name)]


def test_save_on_a_subscribed_file_asks_for_a_name_and_cancel_writes_nothing(tmp_path, monkeypatch, window):
    folder = _profile(tmp_path, monkeypatch)
    target = _subscribed_copy(tmp_path)
    before = target.read_bytes()
    window.load_scenario(target)
    seen = _capture(monkeypatch, "getSaveFileName")
    window.save()
    assert seen == [str(folder / target.name)]
    assert target.read_bytes() == before
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]
    assert list(folder.iterdir()) == []
