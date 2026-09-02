"""The "Show sprites" toggle, driven through a real offscreen ViewerWindow
-- P3-g's last piece.

Same technique and default-tier rationale as tests/test_unit_filter_viewer.py,
which this module is modelled on directly: assert on QAction handles and on
the live chunk cache's own state, never on widget isVisible() (which is
unconditionally False for a never-shown window under
QT_QPA_PLATFORM=offscreen).

The load-bearing one here is test_the_dirty_bbox_call_matches_the_cache_flag.
P3-g5 widened the edit-path dirty bbox so an elevation stroke beside a large
building cannot leave a stale sprite fragment, and that widening is only
correct while the bbox call and IsoChunkCache._level agree about whether
sprites are on. Before the toggle they agreed by both reading one module
global; now they agree only because _apply_dirty passes the cache's own flag.
That is a thread a refactor can cut silently, so it is pinned rather than
commented.

Every ViewerWindow() constructed here calls edit_history.mark_saved() before
close() -- see tests/test_fill_tool.py's module docstring for why.
"""

from __future__ import annotations

import pytest

import conftest
from descape import asset_source, render_cache, unit_sprites, viewer
from descape.scenario_io import BLANK_TEMPLATE_PATH
from test_unit_sprites import CONST, FILE_NAME, build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture
def sprite_install(tmp_path, monkeypatch):
    """A tmp install holding one small synthetic sprite.

    Shaped like tests/test_sprite_edit_bbox.py's fixture of the same name, but
    deliberately NOT its oversized canvas or its pinned reach constants: this
    module never measures a bbox against a sprite's extent, it only needs
    asset_source.is_available() to be True so the toggle is reachable at all.
    A native-tile-sized sprite keeps every toggle-on warm here cheap.

    conftest's autouse _isolated_settings redirects CONFIG_PATH, so without
    this fixture there is no configured install -- which is what the
    no-install case below relies on, and why that test does not request it.
    """
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {CONST: {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": 4,
                         "mirroring_mode": 6, "frame_count": 1}},
    )
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_the_default_is_on_and_reaches_a_fresh_cache(sprite_install) -> None:
    """Ships default-on as of 2026-08-24, superseding P3-g4's opt-in-only
    gate (the 0.5-4.1s cold decode on file open still applies, it's just
    paid up front now). Pinned at both ends: the menu item AND the cache
    it builds."""
    window = conftest.blank_window()
    try:
        assert window.show_sprites_action.isChecked()
        assert window._sprites_enabled is True
        assert window._cache.sprites_enabled is True
        assert isinstance(window._cache, render_cache.IsoChunkCache)
    finally:
        conftest.close_window(window)


def test_it_lives_in_the_view_menu_not_in_filters(sprite_install) -> None:
    """It shipped in the Filters menu and was moved to View on 2026-08-24,
    because it is not a filter: it changes HOW units are drawn, not WHICH
    ones. Pinned at both ends -- present in View, absent from Filters --
    since "still in a menu somewhere" is not what was asked for, and every
    other test here reaches the action through the window attribute, which
    would keep passing wherever it sat.
    """
    window = conftest.blank_window()
    try:
        view_menu = next(
            action.menu()
            for action in window.menuBar().actions()
            if action.menu() is not None and "View" in action.text()
        )
        assert window.show_sprites_action in view_menu.actions()
        assert window.iso_action in view_menu.actions(), "wrong menu found -- expected the View menu"
        assert window.show_sprites_action not in window.filters_button.menu().actions()
    finally:
        conftest.close_window(window)


def test_the_toggle_needs_a_map(sprite_install) -> None:
    window = conftest.blank_window(load=False)
    try:
        assert not window.show_sprites_action.isEnabled()
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.show_sprites_action.isEnabled()
    finally:
        conftest.close_window(window)


def test_the_toggle_is_greyed_out_in_flat(sprite_install) -> None:
    """Flat-only now (Track P3-g6): composite_rect_flat still takes no
    sprite argument, so Flat is the one remaining unlanded piece (P3-g7).
    Sloped had this same pin until g6 gave SlopedChunkCache a real sprite
    path -- see test_the_checked_state_survives_a_terrain_style_round_trip
    for its own (non-greyed) coverage. Greyed with a tooltip naming the
    reason, rather than silently doing nothing."""
    window = conftest.blank_window()
    try:
        assert window.show_sprites_action.isEnabled()
        window.terrain_style_combo.setCurrentText("Flat")
        assert not window.show_sprites_action.isEnabled()
        assert "Stepped and Sloped only" in window.show_sprites_action.toolTip()
        window.terrain_style_combo.setCurrentText("Stepped")
        assert window.show_sprites_action.isEnabled()
    finally:
        conftest.close_window(window)


def test_the_toggle_stays_enabled_in_sloped(sprite_install) -> None:
    """Track P3-g6's own positive of the Flat pin above: Sloped is no longer
    grey, and carries the same tooltip Stepped does."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        assert window.show_sprites_action.isEnabled()
        assert "Stepped and Sloped only" not in window.show_sprites_action.toolTip()
    finally:
        conftest.close_window(window)


def test_the_toggle_is_greyed_out_with_no_install_path(monkeypatch) -> None:
    """Deliberately does NOT request sprite_install. Without a configured
    install every sprite_for() returns None, so the toggle would be a silent
    no-op with nothing on screen to explain it.

    FORCES the no-install condition rather than assuming the ambient one.
    conftest._isolated_settings redirects CONFIG_PATH, but get_install_path()
    checks AOE2DE_INSTALL_PATH FIRST, and no autouse fixture unsets that -- so
    simply asserting is_available() is False would pass on a bare shell and
    fail for anyone who exports the var to run the corpus tier, on a test with
    nothing to do with their change. Patching the lookup covers both.
    """
    monkeypatch.setattr(asset_source, "get_install_path", lambda: None)
    assert not asset_source.is_available()
    window = conftest.blank_window()
    try:
        assert not window.show_sprites_action.isEnabled()
        assert "install path" in window.show_sprites_action.toolTip()
    finally:
        conftest.close_window(window)


def test_configuring_an_install_path_ungreys_the_toggle(tmp_path, monkeypatch) -> None:
    """The greyed tooltip tells the user to go set an install path in
    Settings. That instruction has to actually work: the settings handler
    calls refresh_map(), which re-renders but did NOT re-gate the toolbar
    until this toggle needed it, so the action stayed greyed and the tooltip
    read as a lie.

    Drives set_install_path_override() + refresh_map() -- what the settings
    handler does, minus persisting to config.yaml, which a test has no
    business doing even to a redirected throwaway.
    """
    monkeypatch.setattr(asset_source, "get_install_path", lambda: None)
    window = conftest.blank_window()
    try:
        assert not window.show_sprites_action.isEnabled()

        graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
        graphics.mkdir(parents=True)
        (graphics / f"{FILE_NAME}.sld").write_bytes(build_sld(4))
        monkeypatch.setattr(asset_source, "get_install_path", lambda: tmp_path)
        window.refresh_map()

        assert window.show_sprites_action.isEnabled(), (
            "configuring an install path re-rendered the map but never re-gated the "
            "toolbar, so the toggle's own tooltip points at a step that does nothing"
        )
    finally:
        conftest.close_window(window)


def test_toggling_on_reaches_the_live_cache(sprite_install) -> None:
    window = conftest.blank_window()
    try:
        window.show_sprites_action.setChecked(True)
        assert window._sprites_enabled is True
        assert window._cache.sprites_enabled is True
        window.show_sprites_action.setChecked(False)
        assert window._cache.sprites_enabled is False
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("style", ["Flat", "Sloped"])
def test_the_checked_state_survives_a_terrain_style_round_trip(sprite_install, style: str) -> None:
    """_render_current() rebuilds the cache from scratch on every style
    switch, so the toggle has to be re-applied to the NEW cache -- otherwise
    a trip through Flat or Sloped would silently switch sprites back off on
    the way home. The same trap test_filter_survives_a_terrain_style_switch
    covers for the unit filter.

    Checked in the middle too. Flat's cache still cannot draw sprites
    (P3-g7, unlanded): the flag must be STORED there anyway (that is what
    _ChunkCacheBase.set_sprites_enabled is for) even though nothing acts on
    it, and the menu item must stay checked while greyed rather than being
    silently unchecked. Sloped's cache, as of Track P3-g6, DOES act on it --
    checked here via SlopedChunkCache.sprites actually being built, not
    just the flag being stored.
    """
    window = conftest.blank_window()
    try:
        window.show_sprites_action.setChecked(True)
        window.terrain_style_combo.setCurrentText(style)
        assert window.show_sprites_action.isChecked(), "greying must not uncheck it"
        assert window._cache.sprites_enabled is True, "the flag must survive regardless of style"
        if style == "Sloped":
            assert window._cache.sprites is not None, "Sloped must actually build a sprite layer now (P3-g6)"

        window.terrain_style_combo.setCurrentText("Stepped")
        assert window._cache.sprites_enabled is True
        assert window._cache._level(0).sprites is not None, "back in Stepped it must actually draw again"
    finally:
        conftest.close_window(window)


def test_it_is_not_persisted_across_windows(sprite_install) -> None:
    """Per-session only, same reasoning as the unit filter's own test: a
    view state that survived a restart would present an unexplained map with
    the explanation two clicks deep in a menu.

    This is also what a module-global flag could not have given us -- module
    state outlives close(), so the flag lives on the cache instance instead.
    """
    first = conftest.blank_window()
    try:
        first.show_sprites_action.setChecked(False)
        assert first._cache.sprites_enabled is False
    finally:
        conftest.close_window(first)

    second = conftest.blank_window()
    try:
        assert second.show_sprites_action.isChecked()
        assert second._sprites_enabled is True
        assert second._cache.sprites_enabled is True
    finally:
        conftest.close_window(second)


def test_an_unchanged_value_evicts_nothing(sprite_install) -> None:
    """set_sprites_enabled's no-op guard, for the same reason
    set_unit_filter has one: evicting the whole canvas is the most expensive
    thing the cache can be asked to do, and re-setting the current value is
    a real path (every _render_current() on a Flat/Sloped switch calls it)."""
    window = conftest.blank_window()
    try:
        cache = window._cache
        canvas_w, canvas_h = cache.canvas_dims(0)
        cache.render_rect(0, 0, min(canvas_w, 256), min(canvas_h, 256))
        assert cache._cache, "expected at least one warmed chunk to be evictable"
        resident = dict(cache._cache)

        cache.set_sprites_enabled(window._sprites_enabled)
        assert cache._cache.keys() == resident.keys(), "an unchanged value evicted chunks"
    finally:
        conftest.close_window(window)


def test_sloped_toggle_evicts_chunks(sprite_install) -> None:
    """Sloped's own copy of test_an_unchanged_value_evicts_nothing's setup,
    but for the CHANGING-value case (Track P3-g6): flipping sprites on in
    Sloped must evict every warmed chunk, or the toggle "does nothing until
    you scroll somewhere uncached" (SlopedChunkCache.set_sprites_enabled's
    own docstring names this exact failure mode). Before g6 this had
    nothing to pin -- the base class's set_sprites_enabled only stored the
    flag, with no cache to evict."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        window.show_sprites_action.setChecked(False)
        cache = window._cache
        canvas_w, canvas_h = cache.canvas_dims(0)
        cache.render_rect(0, 0, min(canvas_w, 256), min(canvas_h, 256))
        assert cache._cache, "expected at least one warmed chunk to be evictable"

        window.show_sprites_action.setChecked(True)
        assert not cache._cache, "toggling sprites on in Sloped must evict every warmed chunk"
    finally:
        conftest.close_window(window)


def test_toggling_sloped_sprites_on_widens_the_visible_canvas(sprite_install, monkeypatch) -> None:
    """The wiring gap Track P3-g6's canvas-clip fix would otherwise ship
    inert: SlopedChunkCache.canvas_dims() can now grow when sprites toggle
    on at a low elev_step_pct stop, but MapCanvasItem.boundingRect() and
    MapView's own sceneRect() were both fixed at set_source()-time for
    every style before this, since no other cache's canvas_dims() answer
    ever moved after construction. Without MapView.refresh_canvas_dims()
    (called from _on_sprites_toggled), the newly-composited strip would sit
    in the cache but never reach the screen -- Qt would never even ask to
    paint it."""
    from descape import settings

    monkeypatch.setattr(settings, "_elev_step_pct", 25)
    window = conftest.blank_window()
    try:
        window.show_sprites_action.setChecked(False)
        window.terrain_style_combo.setCurrentText("Sloped")
        before_h = window.map_view._canvas_item.boundingRect().height()
        before_scene_h = window.map_view.sceneRect().height()
        assert before_h == window._cache.proj.canvas_h, "fixture assumption: sprites start off, canvas tight"

        window.show_sprites_action.setChecked(True)
        after_h = window.map_view._canvas_item.boundingRect().height()
        after_scene_h = window.map_view.sceneRect().height()

        assert after_h > before_h, (
            "MapCanvasItem.boundingRect() never grew -- the cache's widened canvas_dims() "
            "is inert on screen"
        )
        assert after_h == window._cache.canvas_dims(0)[1]
        assert after_scene_h > before_scene_h, "MapView's own sceneRect() never grew to match"
    finally:
        conftest.close_window(window)


def test_the_dirty_bbox_call_matches_the_cache_flag(sprite_install, monkeypatch) -> None:
    """THE thread guard, and the reason the per-cache flag is safe.

    dirty_screen_bbox_iso's widening and IsoChunkCache._level's sprite layer
    must agree about whether sprites are on. They used to agree by both
    reading render_cache.SPRITES_ENABLED at call time; now they agree only because
    _apply_dirty passes the cache's own flag through. If that argument is
    ever dropped, hardcoded, or read from the module constant again, the
    under-repaint bug P3-g5 closed comes straight back -- and it comes back
    silently, since nothing else in the suite drives an edit with sprites on.

    Asserts the two AGREE rather than asserting a literal True/False, so the
    test stays honest if the default ever flips.
    """
    window = conftest.blank_window()
    try:
        seen: list[tuple[bool, bool]] = []
        real = viewer.dirty_screen_bbox_iso

        def recording(*args, **kwargs):
            seen.append((kwargs.get("with_sprites"), window._cache.sprites_enabled))
            return real(*args, **kwargs)

        monkeypatch.setattr(viewer, "dirty_screen_bbox_iso", recording)

        mm = window.scenario.map_manager
        for enabled in (True, False):
            window.show_sprites_action.setChecked(enabled)
            before = len(seen)
            window._apply_dirty(
                window.edit_history.apply(
                    "test", mm.terrain, lambda: _bump_one_tile(mm)
                )
            )
            assert len(seen) > before, "the edit never reached dirty_screen_bbox_iso"

        assert seen, "no bbox calls recorded at all"
        for passed, on_cache in seen:
            assert passed == on_cache, (
                f"dirty_screen_bbox_iso got with_sprites={passed!r} while the cache had "
                f"sprites_enabled={on_cache!r} -- the two readers have drifted apart, which is "
                f"exactly the under-repaint bug P3-g5's widening exists to prevent"
            )
        assert {on_cache for _, on_cache in seen} == {True, False}, (
            "both arms must actually run, or this only pins one of the two states"
        )
    finally:
        conftest.close_window(window)


def test_the_sloped_dirty_bbox_call_matches_the_cache_flag(sprite_install, monkeypatch) -> None:
    """Sloped's own copy of test_the_dirty_bbox_call_matches_the_cache_flag
    (Track P3-g6) -- dirty_screen_bbox_sloped's with_sprites and
    SlopedChunkCache.sprites_enabled must agree for the same reason Stepped's
    two readers must: _apply_dirty's Sloped branch passes the cache's own
    flag, and if that argument is ever dropped or hardcoded the same
    under-repaint class of bug comes back, silently."""
    window = conftest.blank_window()
    try:
        window.terrain_style_combo.setCurrentText("Sloped")
        seen: list[tuple[bool, bool]] = []
        real = viewer.dirty_screen_bbox_sloped

        def recording(*args, **kwargs):
            seen.append((kwargs.get("with_sprites"), window._cache.sprites_enabled))
            return real(*args, **kwargs)

        monkeypatch.setattr(viewer, "dirty_screen_bbox_sloped", recording)

        mm = window.scenario.map_manager
        for enabled in (True, False):
            window.show_sprites_action.setChecked(enabled)
            before = len(seen)
            window._apply_dirty(
                window.edit_history.apply("test", mm.terrain, lambda: _bump_one_tile(mm))
            )
            assert len(seen) > before, "the edit never reached dirty_screen_bbox_sloped"

        assert seen, "no bbox calls recorded at all"
        for passed, on_cache in seen:
            assert passed == on_cache, (
                f"dirty_screen_bbox_sloped got with_sprites={passed!r} while the cache had "
                f"sprites_enabled={on_cache!r} -- the two readers have drifted apart"
            )
        assert {on_cache for _, on_cache in seen} == {True, False}, (
            "both arms must actually run, or this only pins one of the two states"
        )
    finally:
        conftest.close_window(window)


def _bump_one_tile(mm):
    """One real elevation edit, returning the terrain indices it changed --
    the same shape tests/test_sprite_edit_bbox.py's own helper uses. A real
    edit rather than a synthetic dirty list, because the point is to travel
    _apply_dirty's actual Stepped branch."""
    from descape.elevation_tools import set_tile_elevation

    x = y = 60
    before = [int(tile.elevation) for tile in mm.terrain]
    set_tile_elevation(mm, x, y, 1 if before[y * mm.map_width + x] == 0 else 0)
    return [i for i, tile in enumerate(mm.terrain) if int(tile.elevation) != before[i]]
