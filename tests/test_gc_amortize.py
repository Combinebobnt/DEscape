"""testkit/gc_amortize.py: a load's collect and freeze are stubbed unless a
test is marked real_gc_freeze, a reaped window's document dies at the
periodic collect, and data-mod tables do not outlive their test."""

from __future__ import annotations

import gc
import weakref

import pytest

from testkit import data_mod_fixture, gc_amortize, window_reaper

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _load_with_spies() -> dict:
    """A blank_window() load with gc.collect, gc.freeze and gc_hold.reset counted.
    Its own MonkeyPatch: undo() on the test's would drop the autouse settings isolation."""
    from descape import gc_hold
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    calls = {"collect": 0, "freeze": 0, "reset": 0}

    def counting(name, real):
        def spy(*args, **kwargs):
            calls[name] += 1
            return real(*args, **kwargs)

        return spy

    window = conftest.blank_window(load=False)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(gc, "collect", counting("collect", gc.collect))
            mp.setattr(gc, "freeze", counting("freeze", gc.freeze))
            mp.setattr(gc_hold, "reset", counting("reset", gc_hold.reset))
            window._load_gc_collect_s = None
            window.load_scenario(BLANK_TEMPLATE_PATH)
        calls["timed"] = window._load_gc_collect_s is not None
    finally:
        gc.unfreeze()
        conftest.close_window(window)
    return calls


def test_a_load_runs_the_method_but_not_its_collect_or_freeze() -> None:
    calls = _load_with_spies()
    assert calls["timed"] and calls["reset"] >= 1, f"the method body did not run: {calls}"
    assert calls["collect"] == 0 and calls["freeze"] == 0, f"the load collected or froze: {calls}"


@pytest.mark.real_gc_freeze
def test_a_real_gc_freeze_test_gets_the_real_collect_and_freeze() -> None:
    calls = _load_with_spies()
    assert calls["timed"] and calls["collect"] >= 1 and calls["freeze"] >= 1, calls


def test_a_reaped_window_document_dies_at_the_periodic_collect() -> None:
    from descape import scenario_io

    window = conftest.blank_window()
    document = weakref.ref(window.scenario)
    scenario_io.set_first_parse_runner(None)  # the last window built owns it until teardown
    window._cancel_warms()  # a load warm finishing after the reap would still reach the window
    del window
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication

    assert window_reaper.reap(window_reaper.protected_ids()) == 1
    # PyQt deleteLater()s a deleted sender's slot proxies, which hold its lambdas: the next test's events would.
    QApplication.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()  # what after_test() runs every COLLECT_EVERY tests
    assert document() is None, "the reaped window's document survived a full collection"


def test_data_mod_tables_are_dropped_and_the_base_game_kept(tmp_path) -> None:
    from descape import game_dataset, object_catalog

    object_catalog._dat_table()
    game_dataset.set_data_mod(data_mod_fixture.mod_tables(tmp_path))
    try:
        object_catalog._dat_table()
    finally:
        game_dataset.set_data_mod(None)

    def keys() -> list:
        return [key for per_dataset in game_dataset._caches for key in per_dataset]

    assert any(isinstance(key, tuple) for key in keys()), "vacuous: no data-mod entry was cached"
    assert gc_amortize.drop_data_mod_caches() >= 1
    assert not any(isinstance(key, tuple) for key in keys()), "a data-mod entry survived"
    assert game_dataset.AOE2 in keys(), "the base game's entries were dropped too"
