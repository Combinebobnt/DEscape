"""The viewer's first-parse runner and its cancellable sites (TASK-357,
viewer_file.py _run_first_trigger_parse() and _cancellable_trigger_read()).

The dialog is swapped for a recording subclass through set_viewer_global():
its wasCanceled() answers what a test sets and records what held at that
moment. On these small fixtures
the real dialog never reaches its minimum duration, so it never shows."""

from __future__ import annotations

import gc
from typing import ClassVar

import pytest

from descape import scenario_io
from descape.find_dialog import TRIGGERS_CANCELLED
from descape.trigger_parse_progress import CANCELLED_MESSAGE
from testkit.viewer_family import set_viewer_global

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

FIXTURE = conftest.ROOT / "tests" / "fixtures" / "triggers_120x120.aoe2scenario"
WARMERS = ("_level_warmer", "_margin_warmer", "_load_warmer")


@pytest.fixture
def dialogs(monkeypatch):
    from PyQt5.QtWidgets import QProgressDialog

    class Recording(QProgressDialog):
        made: ClassVar[list] = []
        cancel = False

        def __init__(self, *args):
            super().__init__(*args)
            self.no_cancel_button = False
            self.visible: list[bool] = []
            self.busy: list[bool] = []
            self.warming: list[list[bool]] = []
            Recording.made.append(self)

        def setCancelButton(self, button):
            self.no_cancel_button = button is None
            super().setCancelButton(button)

        def wasCanceled(self):
            self.visible.append(self.isVisible())
            self.busy.append(self.parent()._busy)
            self.warming.append([getattr(self.parent(), n).is_active for n in WARMERS])
            return Recording.cancel

    Recording.made = []
    set_viewer_global(monkeypatch, "QProgressDialog", Recording)
    gc.unfreeze()
    yield Recording
    gc.unfreeze()


@pytest.fixture
def window():
    w = conftest.stepped_window(FIXTURE)
    yield w
    conftest.close_window(w)


def _settle() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(5):
        QApplication.processEvents()


def test_a_normal_file_never_shows_the_dialog(window, dialogs) -> None:
    window.mode_combo.setCurrentText("Triggers")
    assert window.scenario._trigger_manager is not None
    (dialog,) = dialogs.made
    assert dialog.visible and not any(dialog.visible), "the progress window showed on a small file"
    assert not dialog.no_cancel_button, "a mode entry is cancellable"


def test_cancelling_a_triggers_entry_returns_to_the_previous_mode(window, dialogs) -> None:
    window.mode_combo.setCurrentText("Terrain")
    dialogs.cancel = True
    window.mode_combo.setCurrentText("Triggers")
    assert window.scenario.trigger_read_supported is None and window.trigger_panel._loaded is None
    _settle()
    assert window.mode == "terrain", "Cancel did not return to the previous mode"
    assert CANCELLED_MESSAGE in window.status_log.toPlainText()
    dialogs.cancel = False
    window.mode_combo.setCurrentText("Triggers")
    assert window.mode == "triggers" and window.scenario._trigger_manager is not None
    assert window.trigger_panel._loaded is window.scenario, "the re-entry did not populate the panel"


def test_a_cancelled_players_entry_never_fills_the_personality_cache(window, dialogs) -> None:
    dialogs.cancel = True
    window.mode_combo.setCurrentText("Players")
    assert dialogs.made, "vacuous: Players mode did not reach the trigger parse on this fixture"
    assert window._personality_custom_cache is None
    _settle()
    assert window.mode == "view"


def test_a_cancelled_find_triggers_tab_retries_on_its_next_show(window, dialogs) -> None:
    dialogs.cancel = True
    assert window._find_trigger_status() == (TRIGGERS_CANCELLED, False)
    window._show_find_dialog()
    tab = window._find_dialog.triggers_tab
    tab.ensure_initialized()
    assert not tab.initialized and window.scenario.trigger_read_supported is None
    dialogs.cancel = False
    assert tab.ensure_initialized() and tab.initialized
    assert window.scenario._trigger_manager is not None


@pytest.mark.parametrize("entry", ["show_usages", "show_follow_entries"])
def test_finds_jump_to_the_triggers_tab_asks_to_cancel_once(window, dialogs, entry) -> None:
    window._show_find_dialog()
    find = window._find_dialog
    find.tabs.setCurrentWidget(find.objects_tab)
    dialogs.cancel = True
    getattr(find, entry)([])
    assert len(dialogs.made) == 1, "a cancelled jump to the Triggers tab started a second parse"
    assert find.tabs.currentWidget() is find.triggers_tab and not find.triggers_tab.initialized


def test_a_load_while_in_triggers_mode_ends_in_view_on_cancel(window, dialogs) -> None:
    window.mode_combo.setCurrentText("Triggers")
    dialogs.cancel = True
    window.edit_history.mark_saved()
    window.load_scenario(FIXTURE)
    assert window.scenario.trigger_read_supported is None
    _settle()
    assert window.mode == "view"


def test_a_non_cancellable_first_parse_has_no_cancel_button(window, dialogs) -> None:
    assert scenario_io.parse_triggers(window.scenario) is not None
    (dialog,) = dialogs.made
    assert dialog.no_cancel_button


def test_the_parse_is_busy_with_every_warm_stopped(window, dialogs) -> None:
    window._start_level_warm()
    assert any(getattr(window, n).is_active for n in WARMERS), "vacuous: no warm is active before the entry"
    assert not window._busy
    window.mode_combo.setCurrentText("Triggers")
    (dialog,) = dialogs.made
    assert dialog.busy and all(dialog.busy), "not busy during the parse"
    assert not any(any(active) for active in dialog.warming), "a warm was live inside the parse"
    assert not window._busy, "busy was not restored"


def test_a_first_entry_freezes_and_a_second_does_not(window, dialogs, monkeypatch) -> None:
    freezes: list[int] = []
    real = gc.freeze
    monkeypatch.setattr(gc, "freeze", lambda: freezes.append(1) or real())
    window.mode_combo.setCurrentText("Triggers")
    assert freezes == [1] and gc.isenabled() and gc.get_freeze_count() > 0
    window.mode_combo.setCurrentText("View")
    window.mode_combo.setCurrentText("Triggers")
    assert freezes == [1], "the memoized entry froze again"


def test_an_indirect_first_parse_freezes_too(window, dialogs, monkeypatch) -> None:
    freezes: list[int] = []
    real = gc.freeze
    monkeypatch.setattr(gc, "freeze", lambda: freezes.append(1) or real())
    window.mode_combo.setCurrentText("Diplomacy")
    assert window.scenario._trigger_manager is not None, "vacuous: Diplomacy did not parse on this 1.58 fixture"
    assert freezes == [1] and gc.isenabled()
