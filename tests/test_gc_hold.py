"""The warm gc hold (descape/gc_hold.py, maintainer plan 2026-09-29): starting
a warm raises threshold2 only, the deferred collection runs once every warm
has gone quiet and no mouse button is held, collects before it restores the
exact saved tuple, an overdue hold is a warm tick's only work, and every
document lifetime boundary leaves the original threshold and no idle timer.

The collector and the hold's clock are fakes (gc_hold._collect, gc_hold._now),
set by autouse fixtures and never undone mid-test: monkeypatch.undo() would
also revert the autouse patches. The idle fire is called directly
(gc_hold._on_idle) rather than waited for.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

from descape import debug_log, gc_hold, level_warm, margin_warm

import conftest

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")

ODD = (500, 7, 9)  # a non-default tuple, so a hard-coded (700, 10, 10) restore fails


@pytest.fixture(autouse=True)
def _clean_hold(monkeypatch):
    gc_hold.reset()
    original = gc.get_threshold()
    monkeypatch.setattr(gc_hold, "_drivers", weakref.WeakSet())
    yield
    gc_hold.reset()
    gc.set_threshold(*original)


@pytest.fixture(autouse=True)
def collects(monkeypatch) -> list:
    """The threshold tuple each fake collect saw, one entry per collect."""
    seen: list = []
    monkeypatch.setattr(gc_hold, "_collect", lambda: seen.append(gc.get_threshold()))
    return seen


@pytest.fixture(autouse=True)
def count2(monkeypatch) -> dict:
    """gc_hold's view of count2; past any threshold (a full collection due) unless a test lowers it."""
    state = {"value": 1 << 20}
    monkeypatch.setattr(gc_hold, "_count", lambda: (0, 0, state["value"]))
    return state


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> dict:
    now = {"t": 1000.0}
    monkeypatch.setattr(gc_hold, "_now", lambda: now["t"])
    return now


@pytest.fixture
def qapp():
    conftest.ensure_qapp()


def _hold_mouse(monkeypatch) -> dict:
    """QApplication.mouseButtons() reads LeftButton while held["value"]."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    held = {"value": True}
    monkeypatch.setattr(
        QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton)
    )
    return held


def _idle_armed() -> bool:
    return gc_hold._timer is not None and gc_hold._timer.isActive()


class _LevelCache:
    """What LevelWarmer.start()/tick() call: one level job of `steps` steps, no pack job."""

    def __init__(self, steps: int = 3) -> None:
        self.steps = steps
        self.stepped = 0
        self.installed: list = []

    def level_warm_job(self, mip):
        return SimpleNamespace(gen=self._walk(), install=self.installed.append, kind="walk")

    def pack_warm_job(self, mip, after_level_warm=False):
        return None

    def _walk(self):
        for _ in range(self.steps):
            self.stepped += 1
            yield
        return "layer"


class _ChunkCache:
    """What an unpooled MarginWarmer calls (no prepare_chunk_job)."""

    def __init__(self) -> None:
        self.built: list = []

    def is_level_resident(self, mip) -> bool:
        return True

    def get_chunk(self, mip, cx, cy) -> None:
        self.built.append((cx, cy))


def _drained_level_warm() -> level_warm.LevelWarmer:
    warmer = level_warm.LevelWarmer()
    warmer.start(_LevelCache(), [0])
    warmer.run_to_completion()
    assert not warmer.is_active
    return warmer


# --- engage, idle collect, restore ---------------------------------------------


def test_starting_a_warm_raises_threshold2_only(qapp) -> None:
    gc.set_threshold(*ODD)
    warmer = level_warm.LevelWarmer()
    warmer.start(_LevelCache(), [0])
    try:
        assert gc_hold.is_engaged()
        assert gc.get_threshold() == (ODD[0], ODD[1], gc_hold.HELD_THRESHOLD2)
    finally:
        warmer.cancel()


def test_the_idle_fire_collects_once_then_restores_the_exact_tuple(qapp, collects) -> None:
    gc.set_threshold(*ODD)
    warmer = _drained_level_warm()
    assert _idle_armed(), "the drain did not arm the idle collect"
    assert collects == [], "collected at the drain, before the quiet period"

    gc_hold._on_idle()

    assert len(collects) == 1
    assert gc.get_threshold() == ODD
    assert not gc_hold.is_engaged()
    assert not _idle_armed()
    del warmer


def test_the_collect_runs_before_the_restore(qapp, collects) -> None:
    gc.set_threshold(*ODD)
    _drained_level_warm()
    gc_hold._on_idle()
    assert collects == [(ODD[0], ODD[1], gc_hold.HELD_THRESHOLD2)], (
        "restored before collecting: count2 is left over threshold for the next handler"
    )


@pytest.mark.parametrize("value", [0, ODD[2]])
def test_the_idle_fire_only_restores_when_no_full_collection_came_due(qapp, collects, count2, value) -> None:
    """count2 at or under the saved threshold2: CPython would not have run a
    full collection, so the hold must not force one."""
    gc.set_threshold(*ODD)
    count2["value"] = value
    debug_log.clear()
    _drained_level_warm()
    gc_hold._on_idle()
    assert collects == [], "forced a full collection that was never due"
    assert gc.get_threshold() == ODD
    assert not gc_hold.is_engaged()
    assert "warm gc: idle release, no full collection due" in debug_log.get_log_text()


def test_the_idle_fire_collects_once_a_full_collection_is_due(qapp, collects, count2) -> None:
    gc.set_threshold(*ODD)
    count2["value"] = ODD[2] + 1
    _drained_level_warm()
    gc_hold._on_idle()
    assert len(collects) == 1


def test_the_idle_collect_is_logged(qapp) -> None:
    debug_log.clear()
    _drained_level_warm()
    gc_hold._on_idle()
    assert "warm gc: idle collect" in debug_log.get_log_text()
    assert " ms, held " in debug_log.get_log_text()


def test_engage_is_idempotent(qapp, collects) -> None:
    gc.set_threshold(*ODD)
    gc_hold.engage()
    gc_hold.engage()
    gc_hold.release()
    assert gc.get_threshold() == ODD, "a second engage saved the raised tuple"
    assert len(collects) == 1


def test_release_and_reset_do_nothing_when_not_engaged(qapp, collects) -> None:
    gc.set_threshold(*ODD)
    gc_hold.release()
    gc_hold.reset()
    gc_hold.idle_soon()
    assert collects == []
    assert gc.get_threshold() == ODD
    assert not _idle_armed()


# --- no collect while warming or while a button is held -------------------------


def test_no_collect_while_a_level_warmer_is_active(qapp, collects) -> None:
    warmer = level_warm.LevelWarmer()
    warmer.start(_LevelCache(), [0])
    try:
        gc_hold.idle_soon()
        gc_hold._on_idle()
        assert collects == []
        assert gc_hold.is_engaged()
        assert _idle_armed(), "the idle fire did not re-arm"
    finally:
        warmer.cancel()


def test_no_collect_while_a_margin_warmer_has_only_in_flight_jobs(qapp, collects) -> None:
    """MarginWarmer stops its own timer while jobs are on the pool; that is not
    the end of warming, so it must not collect."""
    warmer = margin_warm.MarginWarmer()
    warmer.start(_ChunkCache(), 0, [(0, 0)])
    try:
        warmer._queue, warmer._in_flight = [], [(0, 0)]
        assert warmer.tick() is True
        assert collects == [], "the in-flight timer stop collected"
        assert gc_hold.is_engaged()

        gc_hold.idle_soon()
        gc_hold._on_idle()
        assert collects == []
        assert _idle_armed(), "the idle fire did not re-arm"
    finally:
        warmer.cancel()


def test_a_margin_warmers_drain_arms_the_idle_collect(qapp, collects) -> None:
    warmer = margin_warm.MarginWarmer()
    cache = _ChunkCache()
    warmer.start(cache, 0, [(0, 0), (1, 0)])
    warmer.run_to_completion()
    assert cache.built == [(0, 0), (1, 0)]
    assert _idle_armed()
    gc_hold._on_idle()
    assert len(collects) == 1


@pytest.mark.gui
def test_the_viewers_three_warmers_are_all_registered(qapp) -> None:
    window = conftest.blank_window(load=False)
    try:
        registered = set(gc_hold._drivers)
        assert {window._level_warmer, window._margin_warmer, window._load_warmer} <= registered
    finally:
        conftest.close_window(window)


def test_no_collect_while_a_mouse_button_is_held(qapp, collects, monkeypatch) -> None:
    _drained_level_warm()
    held = _hold_mouse(monkeypatch)
    gc_hold._on_idle()
    assert collects == [], "collected mid-stroke"
    assert _idle_armed(), "the idle fire did not re-arm"

    held["value"] = False
    gc_hold._on_idle()
    assert len(collects) == 1


def test_re_activation_before_the_idle_fire_cancels_it(qapp, collects) -> None:
    warmer = _drained_level_warm()
    assert _idle_armed()
    warmer.start(_LevelCache(), [0])
    try:
        assert not _idle_armed(), "a restarted warm left the idle collect armed"
        assert gc_hold.is_engaged()
        assert collects == []
    finally:
        warmer.cancel()


# --- the overdue bound ---------------------------------------------------------


def test_an_overdue_hold_is_the_level_warmers_only_tick_work(qapp, collects, clock) -> None:
    cache = _LevelCache(steps=3)
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [0])
    try:
        clock["t"] += gc_hold.HOLD_MAX_S + 1
        assert warmer.tick() is True
        assert len(collects) == 1
        assert cache.stepped == 0, "the overdue tick also stepped the job"
        assert gc_hold.is_engaged(), "the hold did not resume for the running warm"
        assert not gc_hold.overdue(), "the resumed hold kept the old start time"

        warmer.tick()
        assert cache.stepped > 0
        assert len(collects) == 1
    finally:
        warmer.cancel()


def test_an_overdue_hold_is_the_margin_warmers_only_tick_work(qapp, collects, clock) -> None:
    cache = _ChunkCache()
    warmer = margin_warm.MarginWarmer()
    warmer.start(cache, 0, [(0, 0), (1, 0)])
    try:
        clock["t"] += gc_hold.HOLD_MAX_S + 1
        assert warmer.tick() is True
        assert len(collects) == 1
        assert cache.built == [], "the overdue tick also warmed a chunk"
        assert gc_hold.is_engaged()
        assert not gc_hold.overdue()
    finally:
        warmer.cancel()


def test_a_hold_younger_than_the_bound_is_not_collected_by_a_tick(qapp, collects, clock) -> None:
    cache = _LevelCache(steps=3)
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [0])
    try:
        clock["t"] += gc_hold.HOLD_MAX_S - 0.5
        warmer.tick()
        assert collects == []
        assert cache.stepped > 0
    finally:
        warmer.cancel()


# --- headless and the stroke's own hold -----------------------------------------


def test_without_a_qapplication_a_warm_leaves_the_threshold_alone(collects, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    gc.set_threshold(*ODD)
    cache = _LevelCache()
    warmer = level_warm.LevelWarmer()
    warmer.start(cache, [0])
    assert gc.get_threshold() == ODD
    warmer.run_to_completion()
    assert cache.installed == ["layer"], "vacuous: the warm did not run"
    assert gc.get_threshold() == ODD
    assert not gc_hold.is_engaged()
    assert collects == []


@pytest.mark.gui
def test_a_stroke_release_under_a_warm_hold_keeps_both_states(qapp) -> None:
    """The stroke toggles gc.isenabled(); the hold owns only the threshold.
    No document, so no load-time warm engages the hold before the test does."""
    window = conftest.blank_window(load=False)
    assert not gc_hold.is_engaged()
    was_enabled = gc.isenabled()
    try:
        gc.enable()
        gc.set_threshold(*ODD)
        gc_hold.engage()
        window.map_view._pause_gc_for_stroke()
        assert not gc.isenabled()
        window.map_view.resume_stroke_gc()
        assert gc.isenabled(), "the stroke's release left gc disabled"
        assert gc.get_threshold() == (ODD[0], ODD[1], gc_hold.HELD_THRESHOLD2), "the stroke touched the threshold"

        window.map_view._pause_gc_for_stroke()
        gc_hold.release()
        assert not gc.isenabled(), "the hold's release re-enabled gc mid-stroke"
        assert gc.get_threshold() == ODD
        window.map_view.resume_stroke_gc()
        assert gc.isenabled()
    finally:
        window.map_view.resume_stroke_gc()
        if was_enabled:
            gc.enable()
        conftest.close_window(window)


# --- document lifetime ---------------------------------------------------------


def _warming_window(monkeypatch):
    """A shown window with the blank template loaded and its level warm queued
    (test_level_warm._shown_window's shape: an unshown window warms nothing)."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape.scenario_io import BLANK_TEMPLATE_PATH
    from descape.viewer import ViewerWindow

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = ViewerWindow()
    window.resize(800, 600)
    window.show()
    QApplication.processEvents()
    window.load_scenario(BLANK_TEMPLATE_PATH)
    assert window._level_warmer.is_active, "vacuous: the load queued no warm"
    return window


@pytest.mark.gui
def test_a_load_holds_for_its_warm_after_the_freeze(qapp, monkeypatch) -> None:
    """The load's warm arms before _freeze_loaded_document(), whose reset must
    not leave that warm unheld."""
    gc.set_threshold(*ODD)
    window = _warming_window(monkeypatch)
    try:
        assert gc_hold.is_engaged(), "the load's warm runs without the hold"
        assert gc.get_threshold() == (ODD[0], ODD[1], gc_hold.HELD_THRESHOLD2)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_the_next_load_parses_with_the_hold_reset(qapp, monkeypatch) -> None:
    from descape import viewer
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    gc.set_threshold(*ODD)
    window = _warming_window(monkeypatch)
    try:
        assert gc_hold.is_engaged(), "vacuous: nothing held before the second load"
        seen: list = []
        real = viewer.load_map_and_units

        def recording(path):
            seen.append((gc.get_threshold(), _idle_armed()))
            return real(path)

        monkeypatch.setattr(viewer, "load_map_and_units", recording)
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert seen == [(ODD, False)], "the load kept the previous document's hold"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_close_scenario_resets_the_hold(qapp, monkeypatch) -> None:
    gc.set_threshold(*ODD)
    window = _warming_window(monkeypatch)
    try:
        assert gc_hold.is_engaged(), "vacuous: nothing held"
        window.edit_history.mark_saved()
        window.close_scenario()
        assert window.scenario is None, "close_scenario refused -- vacuous"
        assert gc.get_threshold() == ODD
        assert not _idle_armed()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_closing_the_window_resets_the_hold(qapp, monkeypatch) -> None:
    gc.set_threshold(*ODD)
    window = _warming_window(monkeypatch)
    assert gc_hold.is_engaged(), "vacuous: nothing held"
    conftest.close_window(window)
    assert gc.get_threshold() == ODD, "closeEvent left the threshold raised"
    assert not _idle_armed(), "closeEvent left the idle timer armed"
