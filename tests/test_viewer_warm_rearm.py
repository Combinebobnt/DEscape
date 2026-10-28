"""The viewer's warm re-arm wiring around a cache swap and a viewport fire.

`_render_current` cancels the warms, runs `processEvents()`, then replaces
`self._cache`. A viewport poll firing in that turn can start any of the three
warms (margin ring, level, load) on the outgoing cache; a second cancel after
the turn drops them. With that closed, a viewport fire with no previous fire
re-anchors the level warm again (the zoom plan's dropped first-fire clause).
"""

from __future__ import annotations

import pytest
from test_level_warm import UNITS_FIXTURE
from test_native_composite import native_kernel  # noqa: F401 -- a fixture

from descape import composite_backend, level_warm


def _spy_level_warm_starts(window, monkeypatch) -> list:
    """Each LevelWarmer.start()'s mips, in call order; forwards to the real one."""
    starts = []
    real = window._level_warmer.start

    def spy(cache, mips, **kwargs):
        starts.append(list(mips))
        return real(cache, mips, **kwargs)

    monkeypatch.setattr(window._level_warmer, "start", spy)
    return starts


def _warmer_caches(window) -> dict:
    return {
        "level": window._level_warmer._cache,
        "margin": window._margin_warmer._cache,
        "load": window._load_warmer._cache,
    }


@pytest.mark.gui
def test_a_poll_inside_a_style_switch_leaves_no_warm_on_the_outgoing_cache(native_kernel, monkeypatch) -> None:  # noqa: F811
    """A poll firing in _render_current's processEvents() starts the margin,
    level and load warms on the cache about to be replaced. None of them may
    still hold it once the swap is done."""
    from PyQt5.QtWidgets import QApplication

    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    with composite_backend.use_backend("native"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            old = window._cache
            view = window.map_view
            # The load's warm of fit level -2's neighbour (-1) ticks on idle, which settle's two pumps don't promise.
            # A held button backs every tick off, and the drain would spin: fail instead.
            assert not QApplication.mouseButtons(), "a mouse button is held (an earlier test leaked it)"
            window._level_warmer.run_to_completion()
            # Zoomed in to mip -1: resident, with an unwarmed ring and a cold neighbour (0).
            view.scale(4.0, 4.0)
            target = view.viewport_chunk_target()
            assert target is not None and target[0] == -1, f"fixture ladder moved ({target}) -- vacuous"
            assert old.is_level_resident(-1) and not old.is_level_resident(0), "fixture ladder moved -- vacuous"
            started_on_old = []
            real_fire = window._on_viewport_changed

            def spy() -> None:
                real_fire()
                if window._cache is old:
                    started_on_old.append({k for k, c in _warmer_caches(window).items() if c is old})

            view.on_viewport_changed = spy
            real_cancel = window._cancel_warms
            calls = []

            def cancel_with_a_poll_pending() -> None:
                # The first cancel leaves a 0 ms poll with an unseen target for processEvents() to fire.
                real_cancel()
                calls.append(None)
                if len(calls) == 1:
                    view._last_viewport_target = None
                    view._viewport_poll_timer.setInterval(0)
                    view._viewport_poll_timer.start()

            monkeypatch.setattr(window, "_cancel_warms", cancel_with_a_poll_pending)
            window.terrain_style_combo.setCurrentText("Sloped")
            view._viewport_poll_timer.setInterval(view.VIEWPORT_POLL_MS)
            assert window._cache is not old
            assert started_on_old and {"level", "margin"} <= started_on_old[0], (
                f"the poll did not start the warms on the old cache ({started_on_old}) -- vacuous"
            )

            held = {k for k, c in _warmer_caches(window).items() if c is old}
            assert held == set(), f"warms still on the outgoing cache: {sorted(held)}"
            window._level_warmer.run_to_completion()
        finally:
            conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize("toggle", ["layer", "sprites"])
def test_a_poll_inside_a_toggle_leaves_no_warm_and_the_first_fire_unspent(toggle, monkeypatch) -> None:
    """The layer and sprite toggles share the cancel, processEvents(), mutate
    shape. A poll fired in that turn re-arms on the pre-toggle state and records
    its target, so the second cancel must drop both."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    try:
        view = window.map_view
        assert view.viewport_chunk_target() is not None, "no viewport target -- vacuous"
        starts = _spy_level_warm_starts(window, monkeypatch)
        real_cancel = window._cancel_warms
        calls = []

        def cancel_with_a_poll_pending() -> None:
            real_cancel()
            calls.append(None)
            if len(calls) == 1:
                view._last_viewport_target = None
                view._viewport_poll_timer.setInterval(0)
                view._viewport_poll_timer.start()

        monkeypatch.setattr(window, "_cancel_warms", cancel_with_a_poll_pending)
        if toggle == "layer":
            window._on_layer_toggled("farm_overlay", False)
        else:
            window._on_sprites_toggled(window._sprites_enabled)
        view._viewport_poll_timer.setInterval(view.VIEWPORT_POLL_MS)
        assert starts, "the poll did not fire inside the toggle -- vacuous"
        assert window._last_viewport_chunk_target is None, "the toggle's stale fire used up the first-fire re-arm"
        active = [w for w in (window._level_warmer, window._margin_warmer, window._load_warmer) if w.is_active]
        assert not active, "a warm armed before the toggle's mutation is still running"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_first_viewport_fire_after_a_cancel_re_arms_the_level_warm(monkeypatch) -> None:
    """No previous fire means _cancel_warms() just ran (a load, an edit, a
    re-render). The first fire re-anchors the level warm on the mip in view,
    so a zoom right after an edit does not keep the pre-zoom anchor."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    try:
        cache = window._cache
        view = window.map_view
        fake = {"mip": 0}
        monkeypatch.setattr(view, "viewport_chunk_target", lambda: (fake["mip"], *view.viewport_chunk_target_at(fake["mip"])))
        starts = _spy_level_warm_starts(window, monkeypatch)
        window._cancel_warms()
        assert window._last_viewport_chunk_target is None

        window._on_viewport_changed()

        assert len(starts) == 1, f"a first fire after a cancel re-armed {len(starts)} times"
        want = level_warm.neighbour_mips_of(cache, 0)
        assert want and starts[0][:len(want)] == want
        window._on_viewport_changed()
        assert len(starts) == 1, "a same-mip fire after the first re-armed again"
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_poll_pending_at_close_re_arms_no_warm(monkeypatch) -> None:
    """closeEvent cancels the warms, which clears the previous fire, so a poll
    fire still pending then would re-arm them on the closed window (the first-
    fire re-arm), ticking on a shared event loop and engaging the gc hold."""
    import time

    from PyQt5.QtWidgets import QApplication

    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    view = window.map_view
    assert view.viewport_chunk_target() is not None, "no viewport target -- vacuous"
    starts = _spy_level_warm_starts(window, monkeypatch)
    # A pending fire that will see a new target.
    view._last_viewport_target = None
    view._note_viewport_changed()
    conftest.close_window(window)
    deadline = time.perf_counter() + 3 * view.VIEWPORT_POLL_MS / 1000
    while time.perf_counter() < deadline:
        QApplication.processEvents()
    assert starts == [], "a poll fire after close re-armed the level warm"
    # This window's drivers, not gc_hold's process-global state another test's window can engage.
    assert not any(w.is_active for w in (window._level_warmer, window._margin_warmer, window._load_warmer))


@pytest.mark.gui
def test_a_level_warm_re_anchor_leaves_only_the_new_anchors_load_warms_queued(monkeypatch) -> None:
    """Zooming through three levels re-anchors three times. Every level is
    resident with no chunk cached, so each re-anchor queues a load warm per
    neighbour directly while the first one's chunks keep _load_warmer busy.
    Only the last anchor's neighbours may be left in the queue."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    # numpy: no pack derive, so a resident neighbour never waits on on_job_done.
    with composite_backend.use_backend("numpy"):
        window = conftest.stepped_window(UNITS_FIXTURE)
        try:
            cache = window._cache
            view = window.map_view
            for mip in cache.mip_levels():
                cache._level(mip)
            assert all(cache.is_level_resident(m) for m in cache.mip_levels())
            fake = {"mip": -2}
            monkeypatch.setattr(view, "viewport_chunk_target", lambda: (fake["mip"], *view.viewport_chunk_target_at(fake["mip"])))
            window._cancel_warms()
            window._on_viewport_changed()
            starts = _spy_level_warm_starts(window, monkeypatch)
            ever_queued = set()

            for mip in (0, 1, -1):
                fake["mip"] = mip
                window._on_viewport_changed()
                ever_queued |= {m for m, _chunks in window._load_warm_queue}

            assert len(starts) == 3, "not every zoom re-anchored the level warm"
            assert window._load_warmer.is_active, "_load_warmer drained -- vacuous"
            assert ever_queued - set(starts[-1]), "no older anchor's mip was ever queued -- vacuous"
            queued = [m for m, _chunks in window._load_warm_queue]
            assert set(queued) <= set(starts[-1]), f"stale load warms queued: {queued} vs anchor set {starts[-1]}"
            assert len(queued) == len(set(queued)), f"a mip queued twice: {queued}"
        finally:
            conftest.close_window(window)


# --- TASK-326: no warm inside the Eraser and Convert live-mutation window ----


def _armed_window(monkeypatch):
    """UNITS_FIXTURE with Preload zoom levels on and a level warm in flight:
    a wholesale unit step makes every level stale and re-arms it."""
    from descape import settings

    import conftest

    monkeypatch.setattr(settings, "_preload_zoom_levels", True)
    window = conftest.stepped_window(UNITS_FIXTURE)
    window.mode_combo.setCurrentText("Units")
    window.brush_size_spin.setValue(1)
    window._after_unit_mutation()
    return window


def _jobs_in_flight(window) -> list:
    from test_level_warm import _in_flight_jobs

    jobs = _in_flight_jobs(window._level_warmer)
    assert jobs, "vacuous: no level warm job in flight"
    return jobs


def _survivors(window, before) -> list:
    from test_level_warm import _in_flight_jobs

    return [job for job in _in_flight_jobs(window._level_warmer) if any(job is old for old in before)]


def _begin_erase(window):
    """An Eraser stroke over Units and Buildings, opened at press."""
    window._on_tool_selected("eraser")
    for name, action in window.eraser_target_actions.items():
        action.setChecked(name in ("units", "buildings"))
    window.on_edit_stroke_start()
    assert window._erase is not None and window._erase.model is not None


def _erase_next(window) -> None:
    """Erases one more unit the stroke has not touched, by its own tile."""
    from descape import eraser

    stroke = window._erase
    for entry in stroke.index.entries:
        tile = (entry.own_x, entry.own_y)
        if id(entry.unit) in stroke.done or tile in stroke.touched_tiles:
            continue
        if eraser.unit_matches(stroke.targets, entry.player_id, int(entry.unit.unit_const)):
            erased = stroke.erased
            window.on_edit_stroke_tiles([tile], 0)
            window._erase_refresh_timer.stop()
            assert stroke.erased > erased, "the tile erased nothing"
            return
    raise AssertionError("no unit left to erase")


def _convert_entries(window, count: int) -> list:
    """`count` player-1 units on distinct tiles holding only player-1 units, Player 2 the destination."""
    window.units_panel.select_owner(2)
    index = window.map_view._unit_index
    picked, seen = [], set()
    for entry in index.entries:
        tile = (entry.own_x, entry.own_y)
        if entry.player_id != 1 or tile in seen:
            continue
        if all(index.entries[o].player_id == 1 for o in index.by_tile[tile]):
            seen.add(tile)
            picked.append(entry)
        if len(picked) == count:
            return picked
    raise AssertionError("not enough player-1 units")


def _convert(window, entry) -> None:
    window._convert_stroke_tile(entry.own_x, entry.own_y)
    window._unit_stroke_refresh_timer.stop()
    assert window._unit_stroke_pending_splices, "the tile converted nothing"


def _begin_unit_brush(window) -> None:
    """A Unit Brush stroke of trees for GAIA, Free placement on so every tile places."""
    from PyQt5.QtCore import Qt

    assert window.units_panel.select_owner(0)
    assert window.units_panel.pick_object(349)
    window.unit_brush_action.setChecked(True)
    window.unrestricted_check.setChecked(True)
    window.map_view._stroke_button = Qt.LeftButton
    window.on_edit_stroke_start()
    assert window._unit_brush is not None


def _paint_next(window) -> None:
    """Paints one more tile the stroke has not rolled."""
    from PyQt5.QtCore import Qt

    stroke = window._unit_brush
    tile = next((x, 10) for x in range(5, 100) if (x, 10) not in stroke.rolled)
    placed = len(stroke.placed)
    window.on_edit_stroke_tiles([tile], Qt.NoModifier)
    window._unit_stroke_refresh_timer.stop()
    assert len(stroke.placed) > placed and window._unit_stroke_pending_splices, "the tile painted nothing"


def _end_unit_brush(window) -> None:
    window.on_edit_stroke_end()
    window.map_view._stroke_button = None


@pytest.mark.gui
def test_a_unit_brush_add_cancels_the_warms_in_flight_before_it_mutates(monkeypatch) -> None:
    """TASK-326 A, the Unit Brush's add()."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        _begin_unit_brush(window)
        before = _jobs_in_flight(window)
        _paint_next(window)
        assert not _survivors(window, before), "a warm armed before the add still walks the unit lists"
        window._flush_unit_stroke_refresh()
        before = _jobs_in_flight(window)
        _paint_next(window)
        assert not _survivors(window, before), "a warm the flush re-armed survived the next add"
        _end_unit_brush(window)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_an_abandoned_unit_brush_stroke_still_re_arms_the_level_warm(monkeypatch) -> None:
    """A second press with a stroke still open commits it, and its
    _after_unit_mutation() re-arms; B's guard must not swallow that."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        _begin_unit_brush(window)
        _paint_next(window)
        starts = _spy_level_warm_starts(window, monkeypatch)
        window.on_edit_stroke_start()
        assert starts, "the abandoned stroke's commit did not re-arm the level warm"
        assert window.edit_history.records[-1].label == "Paint unit"
        _end_unit_brush(window)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_an_erase_cancels_the_warms_in_flight_before_it_mutates(monkeypatch) -> None:
    """TASK-326 A: no job armed before an erase survives it, the press's and
    the one the mid-stroke flush re-arms alike."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        _begin_erase(window)
        before = _jobs_in_flight(window)
        _erase_next(window)
        assert not _survivors(window, before), "a warm armed before the erase still walks the unit lists"
        window._flush_erase_refresh()
        before = _jobs_in_flight(window)
        _erase_next(window)
        assert not _survivors(window, before), "a warm the flush re-armed survived the next erase"
        window.on_edit_stroke_end()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_a_convert_cancels_the_warms_in_flight_before_it_mutates(monkeypatch) -> None:
    """TASK-326 A, the Convert brush's reassign."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        first, second = _convert_entries(window, 2)
        window._begin_convert_stroke()
        before = _jobs_in_flight(window)
        _convert(window, first)
        assert not _survivors(window, before), "a warm armed before the reassign still walks the unit lists"
        window._flush_unit_stroke_refresh()
        before = _jobs_in_flight(window)
        _convert(window, second)
        assert not _survivors(window, before), "a warm the flush re-armed survived the next reassign"
        window._end_convert_stroke()
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize("tool", ["erase", "convert", "unit_brush"])
def test_nothing_arms_a_level_warm_between_a_live_mutation_and_its_flush(tool, monkeypatch) -> None:
    """TASK-326 B: a direct re-arm and a first viewport fire after the
    mutation's cancel both start nothing until the flush, which starts one."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        starts = _spy_level_warm_starts(window, monkeypatch)
        if tool == "erase":
            _begin_erase(window)
            _erase_next(window)
        elif tool == "convert":
            (entry,) = _convert_entries(window, 1)
            window._begin_convert_stroke()
            _convert(window, entry)
        else:
            _begin_unit_brush(window)
            _paint_next(window)
        window._start_level_warm()
        window._last_viewport_chunk_target = None
        window._on_viewport_changed()
        assert starts == [], f"a level warm armed inside the window: {starts}"
        if tool == "erase":
            window._flush_erase_refresh()
        else:
            window._flush_unit_stroke_refresh()
        assert len(starts) == 1, f"the flush re-armed {len(starts)} times"
        if tool == "erase":
            window.on_edit_stroke_end()
        elif tool == "convert":
            window._end_convert_stroke()
        else:
            _end_unit_brush(window)
    finally:
        conftest.close_window(window)


@pytest.mark.gui
def test_an_abandoned_convert_stroke_still_re_arms_the_level_warm(monkeypatch) -> None:
    """A second press with a stroke still open aborts it, and its
    _after_unit_mutation() repaints the live reassigns and re-arms; B's guard
    must not swallow that re-arm."""
    import conftest

    window = _armed_window(monkeypatch)
    try:
        (entry,) = _convert_entries(window, 1)
        window._begin_convert_stroke()
        _convert(window, entry)
        starts = _spy_level_warm_starts(window, monkeypatch)
        window._begin_convert_stroke()
        assert starts, "the abort path's unit step did not re-arm the level warm"
        window._end_convert_stroke()
    finally:
        conftest.close_window(window)
