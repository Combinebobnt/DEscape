"""Tools > Launch in AoE2:DE's viewer wiring (GH #4, viewer_game_launch.py):
the action's gate, the launch copy carrying unsaved edits without touching the
saved mark, the Overwrite question and the running-game branch. spawn is
always faked: no test starts Steam.

Every window ends with edit_history.mark_saved() before close(), or
closeEvent would block on a modal."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from descape import asset_source, content_roots, game_launch, scenario_io

import conftest

pytestmark = [pytest.mark.gui, pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")]

_PROFILE_SUBPATH = Path("compatdata/813780/pfx/drive_c/users/steamuser/Games/Age of Empires 2 DE")


@pytest.fixture(autouse=True)
def _linux(monkeypatch):
    # The launch is Linux and Windows only; macOS CI runs this file too. The darwin test overrides it.
    monkeypatch.setattr(sys, "platform", "linux")


@pytest.fixture
def scenario_dir(tmp_path, monkeypatch) -> Path:
    profile = tmp_path / _PROFILE_SUBPATH
    folder = profile / "76561190000000001" / "resources" / "_common" / "scenario"
    folder.mkdir(parents=True)
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, str(profile))
    monkeypatch.setattr(asset_source, "CONFIG_PATH", tmp_path / "config" / "config.yaml")
    return folder


@pytest.fixture
def spawned(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(game_launch, "spawn", calls.append)
    monkeypatch.setattr(game_launch, "game_running", lambda: False)
    # Fixed, so neither this machine's PATH nor a CI runner's registry decides it.
    monkeypatch.setattr(game_launch, "launch_command", lambda args, **_kw: ["steam", "-applaunch", "813780", *args])
    return calls


@pytest.fixture
def boxes(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    class Shown(list):
        answer: dict

    shown = Shown()
    answer = shown.answer = {"question": QMessageBox.Cancel}

    def exec_(box):
        shown.append(("question", box.windowTitle(), f"{box.text()} {box.informativeText()}"))
        wanted = answer["question"]
        for button in box.buttons():
            if (wanted == "Overwrite" and button.text() == "Overwrite") or box.standardButton(button) == wanted:
                button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", exec_)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda _p, title, text, *a: shown.append(("critical", title, text))))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda _p, title, text, *a: shown.append(("info", title, text))))
    return shown


@pytest.fixture
def window():
    w = conftest.blank_window(load=False)
    yield w
    w.edit_history.mark_saved()
    w.close()


def _paint(window) -> None:
    tiles = window.scenario.map_manager.terrain
    new_terrain = 15 if tiles[0].terrain_id == 2 else 2
    window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", new_terrain))


def _tools_menu(window):
    return next(a.menu() for a in window.menuBar().actions() if a.text() == "&Tools")


def test_the_action_is_gated_and_says_why(window, tmp_path, monkeypatch) -> None:
    action = window.launch_game_action
    assert window._keybind_actions["launch_game"] is action
    assert action in _tools_menu(window).actions()
    assert _tools_menu(window).toolTipsVisible()
    assert not action.isEnabled() and action.toolTip(), "no map: off, saying what it needs"
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert action.isEnabled(), action.toolTip()
    loaded = conftest.legacy_loaded()
    loaded.path = tmp_path / "Old Map.scx"
    window.load_scenario(loaded.path, loaded=loaded)
    assert not action.isEnabled() and "Convert to .aoe2scenario" in action.toolTip()
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    monkeypatch.setattr(sys, "platform", "darwin")
    window._update_tool_enabled()
    assert not action.isEnabled() and action.toolTip() == game_launch.MACOS_REFUSAL


def test_launch_writes_unsaved_edits_and_leaves_the_document_dirty(window, scenario_dir, spawned, boxes) -> None:
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    _paint(window)
    title, autosaved = window.windowTitle(), window._autosaved_content
    window.launch_game_action.trigger()
    target = scenario_dir / game_launch.LAUNCH_NAME
    assert target.is_file() and not boxes
    copy = scenario_io.load_map_and_units(target)
    assert copy.map_manager.terrain[0].terrain_id == window.scenario.map_manager.terrain[0].terrain_id
    assert window.edit_history.is_dirty and window.windowTitle() == title, "a launch is not a Save"
    assert spawned == [["steam", "-applaunch", "813780", *game_launch.DEFAULT_LAUNCH_ARGS]]
    assert window._autosaved_content == autosaved, "the launch copy is no autosave"
    assert game_launch.ledger_matches(target)
    assert "launching AoE2:DE" in window.status_log.toPlainText()


def test_a_foreign_launch_copy_asks_first(window, scenario_dir, spawned, boxes) -> None:
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    target = scenario_dir / game_launch.LAUNCH_NAME
    target.write_bytes(b"saved in-game")
    window.launch_in_game()
    assert boxes[-1][0] == "question" and str(target) in boxes[-1][2]
    assert target.read_bytes() == b"saved in-game" and not spawned, "Cancel writes nothing"
    assert not list(scenario_dir.glob("*.outside-*.bak")), "Cancel keeps nothing either"
    boxes.answer["question"] = "Overwrite"
    window.launch_in_game()
    assert target.read_bytes() != b"saved in-game" and len(spawned) == 1
    (kept,) = scenario_dir.glob("*.outside-*.bak")
    assert kept.read_bytes() == b"saved in-game", "the changed file outlives later launches' rolling .bak"
    asked = len(boxes)
    window.launch_in_game()
    assert len(boxes) == asked, "DEscape's own copy is overwritten without asking"


def test_the_launch_copy_itself_is_refused(window, scenario_dir, spawned, boxes) -> None:
    target = scenario_dir / game_launch.LAUNCH_NAME
    target.write_bytes(scenario_io.BLANK_TEMPLATE_PATH.read_bytes())
    window.load_scenario(target)
    assert not window.launch_game_action.isEnabled()
    assert "Save As" in window.launch_game_action.toolTip()
    window.launch_in_game()
    assert not spawned and not boxes


def test_a_running_game_gets_the_copy_but_no_launch(window, scenario_dir, spawned, boxes, monkeypatch) -> None:
    monkeypatch.setattr(game_launch, "game_running", lambda: True)
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window.launch_in_game()
    assert (scenario_dir / game_launch.LAUNCH_NAME).is_file() and not spawned
    assert "AoE2:DE is running" in window.status_log.toPlainText()


def test_no_profile_explains_itself(window, tmp_path, spawned, boxes, monkeypatch) -> None:
    monkeypatch.setattr(game_launch, "launch_target", lambda: (None, game_launch.NO_PROFILE_REFUSAL))
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window.launch_in_game()
    assert boxes == [("info", "Launch in AoE2:DE", game_launch.NO_PROFILE_REFUSAL)] and not spawned


def test_busy_refuses_at_trigger_time(window, scenario_dir, spawned, boxes) -> None:
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window._busy = True
    try:
        window.launch_in_game()
    finally:
        window._busy = False
    assert not (scenario_dir / game_launch.LAUNCH_NAME).exists() and not spawned
    assert "Finish the current operation first" in window.status_log.toPlainText()


def test_an_untitled_document_launches(window, scenario_dir, spawned, boxes) -> None:
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window._untitled = True
    window.launch_in_game()
    assert (scenario_dir / game_launch.LAUNCH_NAME).is_file() and len(spawned) == 1


def test_a_profile_in_a_subscribed_mod_is_refused_before_asking(window, tmp_path, spawned, boxes, monkeypatch) -> None:
    folder = tmp_path / "mods" / "subscribed" / "1" / "resources" / "_common" / "scenario"
    folder.mkdir(parents=True)
    target = folder / game_launch.LAUNCH_NAME
    target.write_bytes(b"foreign")
    monkeypatch.setattr(game_launch, "launch_target", lambda: (target, None))
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window.launch_in_game()
    assert [kind for kind, *_ in boxes] == ["critical"] and boxes[0][1] == "Launch blocked"
    assert target.read_bytes() == b"foreign" and not spawned
