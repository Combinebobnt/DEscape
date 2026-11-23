"""The viewer's first-parse runner and its cancellable sites (TASK-357,
viewer_file.py _run_first_trigger_parse() and _cancellable_trigger_read()).

The dialog is swapped for a recording subclass through set_viewer_global():
its wasCanceled() answers what a test sets and records what held at that
moment. On these small fixtures
the real dialog never reaches its minimum duration, so it never shows."""

from __future__ import annotations

import gc
import time
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


@pytest.fixture
def shown_parse(monkeypatch, dialogs):
    """The dialog shows at once and every progress call refreshes it, so
    setValue() runs processEvents() as on a large file. The first shown call
    waits out a viewport poll interval and the zoom frames' settle wait, and
    queues a level warm start as a placeholder's pending timer would, so all
    of them fire in the next setValue()'s processEvents(). A cancel answers
    only from the second shown call, after that dispatch."""
    from PyQt5.QtCore import QTimer

    from descape import gc_hold, level_warm, zoom_preframe
    from descape.map_view import MapView

    wait_s = 2 * max(MapView.VIEWPORT_POLL_MS, zoom_preframe.SETTLE_MS) / 1000

    set_viewer_global(monkeypatch, "TRIGGER_PARSE_DIALOG_MS", 0)
    set_viewer_global(monkeypatch, "TRIGGER_PARSE_REFRESH_S", 0)
    inside = {"engage": 0, "schedule": 0, "prebuild": 0}
    state = {"in_parse": False, "closed": False, "shown_calls": 0, "queued_ran": False, "polls": 0, "starts_after": 0}
    real_init, real_close, real_cancel = dialogs.__init__, dialogs.close, dialogs.wasCanceled

    def init(self, *args):
        real_init(self, *args)
        state["in_parse"] = True

    def close(self):
        state["in_parse"], state["closed"] = False, True
        return real_close(self)

    def was_canceled(self):
        if self.isVisible():
            state["shown_calls"] += 1
            if state["shown_calls"] == 1:
                window = self.parent()

                def queued():
                    state["queued_ran"] = state["in_parse"]
                    window._start_level_warm()

                QTimer.singleShot(0, queued)
                window.map_view.retarget_preframes()  # what a resize inside the parse does
                time.sleep(wait_s)
        return real_cancel(self) and state["shown_calls"] >= 2

    def counting(key, real):
        def spy(*args, **kwargs):
            if state["in_parse"]:
                inside[key] += 1
            return real(*args, **kwargs)

        return spy

    monkeypatch.setattr(dialogs, "__init__", init)
    monkeypatch.setattr(dialogs, "close", close)
    monkeypatch.setattr(dialogs, "wasCanceled", was_canceled)
    monkeypatch.setattr(gc_hold, "engage", counting("engage", gc_hold.engage))
    monkeypatch.setattr(
        level_warm._IdleTimerDriver, "_schedule", counting("schedule", level_warm._IdleTimerDriver._schedule)
    )
    return inside, state


def _arm_everything(window, monkeypatch, inside, state) -> None:
    """The warms, a pending unit model prebuild, zoom frames on, a held-key
    pan and a viewport poll due to fire, as when the user enters Triggers
    right after a load."""
    from PyQt5.QtCore import Qt

    view = window.map_view
    view.set_preframes_allowed(True)
    window._start_level_warm()
    assert any(getattr(window, n).is_active for n in WARMERS), "vacuous: no warm is active before the entry"
    real = window._prebuild_unit_model

    def prebuild(doc_id):
        inside["prebuild"] += 1
        return real(doc_id)

    monkeypatch.setattr(window, "_prebuild_unit_model", prebuild)
    window.unit_edits = None
    window._schedule_unit_model_prebuild()
    assert window._unit_model_prebuild.is_active, "vacuous: no prebuild pending"
    real_traced = window._on_viewport_changed_traced

    def traced():
        state["polls"] += state["in_parse"]
        return real_traced()

    monkeypatch.setattr(window, "_on_viewport_changed_traced", traced)
    real_start = window._start_level_warm

    def start(*args, **kwargs):
        state["starts_after"] += state["closed"]
        return real_start(*args, **kwargs)

    monkeypatch.setattr(window, "_start_level_warm", start)
    view._pan_held[(Qt.Key_Right, 0)] = (1, 0)
    view._pan_elapsed.start()
    view._pan_timer.start(view.PAN_TICK_MS)
    view._last_viewport_target = None
    view._note_viewport_changed()


def _assert_events_ran_inside(state) -> None:
    assert state["shown_calls"] >= 2, "vacuous: no processEvents() ran inside the parse after the wait"
    assert state["queued_ran"], "vacuous: the queued level warm start did not fire inside the parse"
    assert state["polls"], "vacuous: no viewport poll fired inside the parse"


def test_no_warm_or_gc_hold_restarts_inside_a_shown_parse(window, dialogs, shown_parse, monkeypatch) -> None:
    inside, state = shown_parse
    _arm_everything(window, monkeypatch, inside, state)
    window.mode_combo.setCurrentText("Triggers")
    (dialog,) = dialogs.made
    _assert_events_ran_inside(state)
    assert inside == {"engage": 0, "schedule": 0, "prebuild": 0}, f"restarted inside the parse: {inside}"
    assert not any(any(active) for active in dialog.warming), "a warm was live inside the parse"
    assert not window._first_parse_running
    _settle_for(window.map_view.VIEWPORT_POLL_MS * 3)
    # An inside fire recorded as seen would leave the repoll's fire nothing to re-anchor.
    assert state["starts_after"], "no viewport fire re-anchored the level warm after the parse"
    assert window.unit_edits is not None, "the held prebuild never ran after the parse"


def test_a_cancelled_shown_parse_restores_gc_and_busy(window, dialogs, shown_parse, monkeypatch) -> None:
    inside, state = shown_parse
    _arm_everything(window, monkeypatch, inside, state)
    freezes: list[int] = []
    real = gc.freeze
    monkeypatch.setattr(gc, "freeze", lambda: freezes.append(1) or real())
    dialogs.cancel = True
    window.mode_combo.setCurrentText("Triggers")
    _assert_events_ran_inside(state)
    assert window.scenario.trigger_read_supported is None, "vacuous: the parse was not cancelled"
    assert freezes == [], "a cancelled parse froze its dropped objects"
    assert gc.isenabled() and not window._busy and not window._first_parse_running
    assert inside == {"engage": 0, "schedule": 0, "prebuild": 0}, f"restarted inside the parse: {inside}"


def _settle_for(ms: int) -> None:
    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.01)


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
