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
